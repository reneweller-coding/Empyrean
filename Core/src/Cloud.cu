/**
 * @file Cloud.cu
 * @brief The partial cloud's kernels and the engine that runs them as one CUDA graph per block.
 *
 * Per block three kernels run, in this order:
 *   - update: one thread per partial. Turns the voice's few numbers into this partial's frequency
 *     (harmonic, place in the cloud, drift, pitch envelope), amplitude (tilt, shimmer, amplitude
 *     envelope, the voice's gain) and place in the field (pan envelope), all for the END of the
 *     block, and writes what the render kernel needs: start phase, phase step and its per-sample
 *     change (a linear glide, so pitch envelopes never step), and the left/right amplitude ramps.
 *   - render: one CUDA block per chunk of Chunk partials, one thread per sample. The chunk's
 *     partials sit in shared memory and every thread of a warp reads the same partial at the same
 *     time (a broadcast), so the inner loop is one __sinf and a handful of FMAs per partial and sample.
 *   - reduce: adds the chunks' partial sums sample by sample, always in the same order, so a block
 *     renders bit for bit the same every time.
 * Phases are 32-bit fixed point: they wrap exactly, so a note may hold for hours without the
 * drift a float phase would gather.
 * @author Rene Weller
 */
#include "additiv/Cloud.h"
#include "additiv/Spectrum.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace additiv
{
namespace
{

constexpr int Chunk = 512;          ///< partials per CUDA block of the render kernel
constexpr int Threads = 256;        ///< threads per CUDA block of the update and render kernels
constexpr int ReduceLanes = 32;     ///< samples per CUDA block of the reduce kernel
constexpr int ReduceGroups = 16;    ///< threads per sample in the reduce kernel, each summing every 16th chunk
constexpr int ModalThreads = 256;   ///< modes per CUDA block of the modal kernel
constexpr int ModalBlocks = ModalModes / ModalThreads;   ///< CUDA blocks of the modal kernel
constexpr int ModalTile = 128;                           ///< samples the modal kernel sums per pass
static_assert( ModalModes % ModalThreads == 0, "the modes fill whole CUDA blocks" );
constexpr int EchoN = 1024;               ///< the spectral echo's FFT size
constexpr int EchoBins = EchoN / 2 + 1;   ///< its bands, 0 .. N/2
constexpr int EchoSlots = 2048;           ///< blocks each band's delay line holds (5.4 s at 128)
constexpr int EchoThreads = EchoN / 2;    ///< threads of its one CUDA block: one butterfly each
constexpr int MeltBands = MeltN / 2;      ///< the melt's bands that live, 0 .. N/2 - 1 (0, the DC, kept silent)
constexpr int MeltThreads = 1024;         ///< threads of its one CUDA block: two bands each
static_assert( MeltBands == 2 * MeltThreads, "two bands a thread" );
constexpr int LifeCells = LifeW * LifeH;   ///< cells of the living field
static_assert( ( LifeW & ( LifeW - 1 ) ) == 0 && ( LifeH & ( LifeH - 1 ) ) == 0, "the torus wraps by masks" );
/** @brief Lenia's ring kernel, normalised, set once at open(). */
__constant__ float cLeniaKernel[( 2 * LifeRadius + 1 ) * ( 2 * LifeRadius + 1 )];
constexpr float    AirShare = 0.2f;   ///< the share of a voice's partials that become its air band when Air is up

/** @brief What every kernel needs to know about the block; uploaded in front of the voices. */
struct BlockHeader
{
    int           m_n;                ///< samples in this block
    int           m_nPrev;            ///< samples of the previous block, by which the phases advance
    float         m_sampleRate;       ///< Hz
    int           m_voices;           ///< voices in the table
    int           m_shapeCount;       ///< shapes in the bank
    int           m_landColumns;      ///< the landscape's frequency columns (0 = none)
    int           m_landRows;         ///< its time rows
    float         m_landRowSeconds;   ///< how long one row lasts
    std::uint32_t m_blockIndex;       ///< blocks launched since open(): where this block's levels go in the ring
    std::uint32_t m_hpad0;            ///< keeps what follows 16-byte aligned
    std::uint32_t m_hpad1;            ///< likewise
    std::uint32_t m_hpad2;            ///< likewise
    ModalHall     m_modal;            ///< the modal hall's settings for this block
    SpectralField m_field;            ///< the field and the dispersion for this block
    Binaural      m_binaural;         ///< the binaural placement for this block
    SpectralEcho  m_echo;             ///< the spectral echo for this block
    LifeField     m_life;             ///< the living landscape for this block
    PartialSync   m_sync;             ///< the partial sync for this block
    Ripple        m_ripple;           ///< the ripple for this block
    Reservoir     m_reservoir;        ///< the reservoir for this block
    Melt          m_melt;             ///< the melt for this block
    SpectralComb  m_comb;             ///< the spectral comb for this block
    Turbulence    m_turb;             ///< the energy cascade for this block
    std::int32_t  m_lifeParity;       ///< which of the two field buffers holds the field before this block's steps
    std::int32_t  m_lpad0;            ///< keeps the header 16-byte aligned
    std::int32_t  m_lpad1;            ///< likewise
    std::int32_t  m_lpad2;            ///< likewise
};
static_assert( sizeof( BlockHeader ) % 16 == 0, "the voices behind the header stay 16-byte aligned" );
static_assert( sizeof( CloudVoice ) % 16 == 0, "CloudVoice is uploaded as an array" );
static_assert( sizeof( EnvLane ) == 64, "EnvLane keeps its layout" );
static_assert( sizeof( EnvShape ) % 16 == 0, "EnvShape is uploaded as an array" );

/**
 * @brief An integer hash with good avalanche (lowbias32, C. Wellons); the dice of the cloud.
 * @param[in] x  any value
 * @return its hash
 */
__host__ __device__ __forceinline__ std::uint32_t hash32( std::uint32_t x )
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/**
 * @brief A uniform number in [0, 1) for one partial of one note.
 * @param[in] seed  the note's seed
 * @param[in] p     the partial within the voice
 * @param[in] salt  which of the partial's dice (offset, drift rates, phases, pan, envelopes, ...)
 * @return 0 .. 1
 */
__device__ __forceinline__ float hash01( std::uint32_t seed, std::uint32_t p, std::uint32_t salt )
{
    return static_cast<float>( hash32( seed * 0x9E3779B9U ^ hash32( p * 0x85EBCA6BU + salt * 0xC2B2AE35U ) ) >> 8 ) *
           ( 1.0f / 16777216.0f );
}

/**
 * @brief A sine of a slow oscillator at `rate` Hz and phase offset `phase` (cycles) at time `t`,
 *        reduced to one cycle first so the fast __sinf stays accurate after hours.
 * @param[in] rate   Hz
 * @param[in] t      seconds
 * @param[in] phase  0 .. 1
 * @return -1 .. 1
 */
__device__ __forceinline__ float slowSine( float rate, float t, float phase )
{
    float x = fmaf( rate, t, phase );
    x -= floorf( x );
    return __sinf( 6.28318531f * x );
}

/**
 * @brief The bend of an envelope segment.
 * @param[in] x  0 .. 1 through the segment
 * @param[in] c  -1 .. 1: 0 straight, positive starts slowly, negative starts fast
 * @return 0 .. 1
 */
__device__ __forceinline__ float bend( float x, float c )
{
    if ( c > 0.001f ) return __powf( x, 1.0f + 5.0f * c );
    if ( c < -0.001f ) return 1.0f - __powf( 1.0f - x, 1.0f - 5.0f * c );
    return x;
}

/**
 * @brief A shape's held course (its start, then its loop forever) at shape time `tau`.
 * @param[in] s      the shape
 * @param[in] tau    seconds of shape time (already scaled by the partial's rate)
 * @param[in] phase  loop cycles added inside the loop
 * @return the level
 */
__device__ float shapeHeld( const EnvShape &s, float tau, float phase )
{
    const int n = s.m_count;
    if ( n <= 0 ) return 0.0f;
    const bool loop = s.m_loopStart >= 0 && s.m_loopEnd > s.m_loopStart && s.m_loopEnd < n;
    if ( loop )
    {
        const float t0 = s.m_p[s.m_loopStart].m_time;
        const float len = s.m_p[s.m_loopEnd].m_time - t0;
        if ( len > 0.0f && tau >= t0 )
        {
            float x = ( tau - t0 ) / len + phase;
            x -= floorf( x );
            tau = t0 + x * len;
        }
    }
    if ( tau <= s.m_p[0].m_time ) return s.m_p[0].m_level;
    const int last = loop ? s.m_loopEnd : n - 1;
    if ( tau >= s.m_p[last].m_time ) return s.m_p[last].m_level;
    int i = 0;
    while ( i + 1 < last && s.m_p[i + 1].m_time <= tau ) ++i;
    const float span = s.m_p[i + 1].m_time - s.m_p[i].m_time;
    const float x = span > 0.0f ? ( tau - s.m_p[i].m_time ) / span : 1.0f;
    return s.m_p[i].m_level + ( s.m_p[i + 1].m_level - s.m_p[i].m_level ) * bend( x, s.m_p[i + 1].m_curve );
}

/**
 * @brief A shape's level: its held course, or after the release its tail from wherever the loop was.
 * @param[in] s       the shape
 * @param[in] tau     shape time now
 * @param[in] tauOff  shape time of the release, negative while held
 * @param[in] phase   loop cycles added inside the loop
 * @return the level
 */
__device__ float shapeValue( const EnvShape &s, float tau, float tauOff, float phase )
{
    const bool tail = s.m_loopStart >= 0 && s.m_loopEnd > s.m_loopStart && s.m_loopEnd + 1 < s.m_count;
    if ( tauOff < 0.0f || !tail || tau < tauOff ) return shapeHeld( s, tau, phase );
    float       prevX = 0.0f;
    float       prevL = shapeHeld( s, tauOff, phase );
    const float x = tau - tauOff;
    const float tEnd = s.m_p[s.m_loopEnd].m_time;
    for ( int j = s.m_loopEnd + 1; j < s.m_count; ++j )
    {
        const float xj = s.m_p[j].m_time - tEnd;
        if ( x < xj )
        {
            const float span = xj - prevX;
            const float u = span > 0.0f ? ( x - prevX ) / span : 1.0f;
            return prevL + ( s.m_p[j].m_level - prevL ) * bend( u, s.m_p[j].m_curve );
        }
        prevX = xj;
        prevL = s.m_p[j].m_level;
    }
    return prevL;
}

/** @brief Everything one partial's envelope lanes are read against. */
struct LaneContext
{
    float           m_src[MapSources];   ///< the partial's map sources, 0 .. 1
    std::uint32_t   m_seed;              ///< the note's seed
    std::uint32_t   m_p;                 ///< the partial within the voice
    float           m_t;                 ///< the note's time at the end of the block, seconds
    float           m_releaseAt;         ///< the note's release time, negative while held
    const EnvShape *m_shapes;            ///< the bank
    int             m_shapeCount;        ///< shapes in it
    const float    *m_land;              ///< the landscape
    int             m_landColumns;       ///< its columns (0 = none)
    int             m_landRows;          ///< its rows
    float           m_landRowSeconds;    ///< seconds per row
    float           m_landColumn;        ///< the partial's column, 0 .. landColumns-1
};

/**
 * @brief The landscape at one column and time, interpolated both ways; it loops over its rows.
 * @param[in] c      the context (landscape and column)
 * @param[in] tau    lane time, seconds
 * @param[in] phase  loop cycles added
 * @return 0 .. 1
 */
__device__ float landValue( const LaneContext &c, float tau, float phase )
{
    const float rows = static_cast<float>( c.m_landRows );
    float       r = tau / c.m_landRowSeconds + phase * rows;
    r -= floorf( r / rows ) * rows;
    int          r0 = min( static_cast<int>( r ), c.m_landRows - 1 );
    const float  fr = r - static_cast<float>( r0 );
    const int    r1 = r0 + 1 == c.m_landRows ? 0 : r0 + 1;
    const int    c0 = min( static_cast<int>( c.m_landColumn ), c.m_landColumns - 1 );
    const float  fc = c.m_landColumn - static_cast<float>( c0 );
    const int    c1 = min( c0 + 1, c.m_landColumns - 1 );
    const float *a = c.m_land + static_cast<size_t>( r0 ) * c.m_landColumns;
    const float *b = c.m_land + static_cast<size_t>( r1 ) * c.m_landColumns;
    const float  top = a[c0] + ( a[c1] - a[c0] ) * fc;
    const float  bottom = b[c0] + ( b[c1] - b[c0] ) * fc;
    return top + ( bottom - top ) * fr;
}

/**
 * @brief One lane's value for one partial: which shape, how fast, where in its loop, how late.
 * @param[in]  L        the lane
 * @param[in]  c        the partial's context
 * @param[in]  salt     the lane's dice (each lane its own)
 * @param[in]  bipolar  pitch and pan read the landscape as -1 .. 1
 * @param[out] depth    receives the partial's depth (amount, tilted by its source)
 * @return the envelope's level
 */
__device__ float laneValue( const EnvLane &L, const LaneContext &c, std::uint32_t salt, bool bipolar, float &depth )
{
    // Random is its own die for every parameter of every lane, so a random shape and a random
    // depth do not come out correlated.
    auto src = [&]( MapSourceE m, std::uint32_t which )
    {
        const int i = static_cast<int>( m );
        if ( m == MapSourceE::RANDOM ) return hash01( c.m_seed, c.m_p, salt + 2 + which );
        return i > 0 && i < MapSources ? c.m_src[i] : 0.0f;
    };
    const float sDepth = src( L.m_depthBy, 0 );
    depth = L.m_amount *
            ( L.m_depthTilt >= 0.0f ? 1.0f - L.m_depthTilt + L.m_depthTilt * sDepth : 1.0f + L.m_depthTilt * sDepth );
    const float rate = L.m_rate * fmaxf( 0.01f, 1.0f + L.m_rateSpread * src( L.m_rateBy, 1 ) ) *
                       exp2f( L.m_chaos * ( 2.0f * hash01( c.m_seed, c.m_p, salt ) - 1.0f ) );
    const float phase = L.m_phase * src( L.m_phaseBy, 2 ) + L.m_chaos * hash01( c.m_seed, c.m_p, salt + 1 );
    const float sd = src( L.m_delayBy, 3 );
    const float delay = L.m_delay >= 0.0f ? L.m_delay * sd : -L.m_delay * ( 1.0f - sd );
    const float tau = rate * fmaxf( 0.0f, c.m_t - delay );
    if ( L.m_landscape && c.m_landColumns > 0 && c.m_landRows > 0 )
    {
        const float v = landValue( c, tau, phase );
        return bipolar ? 2.0f * v - 1.0f : v;
    }
    if ( c.m_shapeCount <= 0 ) return 0.0f;
    const int   spread = max( 1, L.m_shapeSpread );
    const int   pick = min( spread - 1, static_cast<int>( src( L.m_shapeBy, 4 ) * static_cast<float>( spread ) ) );
    const int   idx = min( max( L.m_shape + pick, 0 ), c.m_shapeCount - 1 );
    const float tauOff = c.m_releaseAt >= 0.0f ? rate * fmaxf( 0.0f, c.m_releaseAt - delay ) : -1.0f;
    return shapeValue( c.m_shapes[idx], tau, tauOff, phase );
}

/**
 * @brief The voice filter's response at one partial: the analogue prototype's magnitude (Spectrum.h's
 *        filterResponse, the same the picture draws).
 * @param[in] model   FilterModel
 * @param[in] f       the partial, Hz
 * @param[in] cutoff  Hz
 * @param[in] q       resonance as a Q (Peak: the bell's gain; Formant: the formants' Q)
 * @param[in] vowel   Formant: 0 .. 4, u o a e i
 * @return the gain
 */
__device__ float filterGain( int model, float f, float cutoff, float q, float vowel )
{
    return filterResponse( model, f, cutoff, q, vowel );
}

/**
 * @brief One block of the field for one partial: its offset moved towards what pulls it and drawn back home.
 * @param[in] F      the field
 * @param[in] logHz  log2 of where the partial is now (its own place plus the offset)
 * @param[in] off    the offset, cents
 * @param[in] dt     the block's length, seconds
 * @param[in] keep   log2 offset the partial keeps from what pulls it: its place in its cloud's bell, narrowed to 2
 *                   cents a unit -- partials gathered onto one pitch would add up to one sine of a frozen random level
 *                   (measured: 3 dB down, the shimmer gone), a narrow cloud still beats and keeps its power
 * @return the new offset, cents (at most an octave)
 */
__device__ float fieldStep( const SpectralField &F, float logHz, float off, float dt, float keep )
{
    if ( F.m_mode == FieldModeE::OFF || F.m_strength <= 0.0f || F.m_count <= 0 )
        return off * __expf( -fmaxf( F.m_spring, 1.0f ) * dt );   // switched off: home within a second or so
    const float R = fmaxf( F.m_radius, 1.0f );
    const float maxStep = F.m_strength * dt;
    const float relax = 1.0f - __expf( -( F.m_strength / R ) * dt );   // towards an attractor, never past it
    float       step = 0.0f;
    if ( F.m_mode == FieldModeE::SCALE )
    {
        // The nearest pitch of the scale (ascending): a binary search, then the neighbour.
        int lo = 0, hi = F.m_count - 1;
        while ( hi - lo > 1 )
        {
            const int mid = ( lo + hi ) >> 1;
            if ( F.m_pos[mid] <= logHz )
                lo = mid;
            else
                hi = mid;
        }
        const float a = F.m_pos[lo], b = F.m_pos[hi];
        const float q = fabsf( logHz - a ) <= fabsf( b - logHz ) ? a : b;
        step = 1200.0f * ( q + keep - logHz ) * relax;
    }
    else
    {
        for ( int k = 0; k < min( F.m_count, FieldAttractors ); ++k )
        {
            const float d = 1200.0f * ( F.m_pos[k] + keep - logHz );
            const float u = d / R;
            const float w = __expf( -0.5f * u * u );
            step += F.m_mass[k] >= 0.0f ? F.m_mass[k] * w * d * relax : F.m_mass[k] * w * copysignf( maxStep, d );
        }
    }
    step = fminf( fmaxf( step, -maxStep ), maxStep );
    return fminf( fmaxf( ( off + step ) * __expf( -F.m_spring * dt ), -1200.0f ), 1200.0f );
}

/**
 * @brief A frequency's band on the hall's log axis (HallBins over 20 Hz .. 20 kHz).
 * @param[in] f  Hz
 * @return the band
 */
__device__ __forceinline__ int hallBin( float f )
{
    return min( max( static_cast<int>( log2f( fmaxf( f, 20.0f ) * ( 1.0f / 20.0f ) ) *
                                       ( static_cast<float>( HallBins ) / 9.96578428f ) ),
                     0 ),
                HallBins - 1 );
}

/**
 * @brief One block of the Orbit field for one partial: a charged particle in pitch and side, drawn to
 *        the nearest pitch of the scale (a well, its place in its bell kept) and bent by a magnetic
 *        field -- the velocity turned by the field's angle each block (exactly: the turn never adds
 *        energy), the well's pull and the damping, and a kick of white noise, so that the orbits keep
 *        about half the radius (a damped oscillator's mean square, sigma^2 / (2 g k)). Fast small
 *        circles ride a slow circling of the well.
 * @param[in]     F        the field (orbitWell, orbitField, orbitDamp; radius: cents to a full side)
 * @param[in]     prev     the last block's (offset, side, their velocities) per partial, cents and cents a second
 * @param[in,out] next     this block's
 * @param[in]     me       the partial's index in the tables
 * @param[in]     logHome  log2 of the partial's own place
 * @param[in]     keep     log2 offset kept from the well (its place in its cloud's bell)
 * @param[in]     now      the offset it has now (taken over when the orbit is fresh)
 * @param[in]     seed     the note's dice
 * @param[in]     up       the partial
 * @param[in]     block    the block's index (the kicks' dice)
 * @param[in]     restart  a new note: at home, at rest
 * @param[in]     dt       the block's length, seconds
 * @param[out]    side     receives its shift across the field, -0.6 .. 0.6
 * @return the new offset, cents
 */
__device__ float orbitStep( const SpectralField &F, const float4 *prev, float4 *next, size_t me, float logHome,
                            float keep, float now, std::uint32_t seed, std::uint32_t up, std::uint32_t block,
                            bool restart, float dt, float &side )
{
    float4 s = restart ? make_float4( 0.0f, 0.0f, 0.0f, 0.0f )
                       : ( F.m_orbitFresh ? make_float4( now, 0.0f, 0.0f, 0.0f ) : prev[me] );
    float  well = 0.0f;
    if ( F.m_count > 0 )
    {
        const float at = logHome + s.x * ( 1.0f / 1200.0f );
        int         lo = 0, hi = F.m_count - 1;
        while ( hi - lo > 1 )
        {
            const int mid = ( lo + hi ) >> 1;
            if ( F.m_pos[mid] <= at )
                lo = mid;
            else
                hi = mid;
        }
        const float q = fabsf( at - F.m_pos[lo] ) <= fabsf( F.m_pos[hi] - at ) ? F.m_pos[lo] : F.m_pos[hi];
        well = 1200.0f * ( q + keep - logHome );
    }
    const float R = fmaxf( F.m_radius, 1.0f );
    const float k = F.m_orbitWell * F.m_orbitWell, g = fmaxf( F.m_orbitDamp, 0.0f );
    float       rs, rc;
    __sincosf( F.m_orbitField * dt, &rs, &rc );
    float       vx = s.z * rc + s.w * rs, vy = s.w * rc - s.z * rs;   // the field turns the velocity
    const float sigma = 0.5f * R * sqrtf( 2.0f * fmaxf( g, 0.01f ) * k ) * sqrtf( dt );
    const float kx = ( hash01( seed ^ ( block * 0x9E3779B9U ), up, 20 ) - 0.5f ) * 3.4641016f;   // unit variance
    const float ky = ( hash01( seed ^ ( block * 0x85EBCA6BU ), up, 21 ) - 0.5f ) * 3.4641016f;
    vx += dt * ( -k * ( s.x - well ) - g * vx ) + sigma * kx;
    vy += dt * ( -k * s.y - g * vy ) + sigma * ky;
    const float x = fminf( fmaxf( s.x + vx * dt, -1200.0f ), 1200.0f );
    const float y = fminf( fmaxf( s.y + vy * dt, -2.0f * R ), 2.0f * R );
    next[me] = make_float4( x, y, vx, vy );
    side = 0.6f * fminf( fmaxf( y / R, -1.0f ), 1.0f );
    return x;
}

/**
 * @brief The hall's ear and memory, one CUDA block: the partials' amplitudes gathered over the octaves
 *        this block (whole numbers, cleared here) as a share of the whole, a little blurred; the spectrum's
 *        density followed over a third of a second (the partials' own clocks read it); and, while
 *        anything sounds and Grow is not nothing, the hall's memory a step of Memory towards it.
 *        Summed in a fixed order.
 * @param[in]     hdr      the header
 * @param[in,out] hist     per band the amplitude gathered this block (fixed point, 1e-12)
 * @param[in,out] memory   per band the hall's memory, 1 the average
 * @param[in,out] density  per band the spectrum's density, 1 the average
 */
__global__ void __launch_bounds__( HallBins )
    hallKernel( const BlockHeader *hdr, unsigned long long *hist, float *memory, float *density )
{
    const ModalHall &mh = hdr->m_modal;
    if ( !mh.m_listen ) return;
    __shared__ float e[HallBins];
    __shared__ float red[HallBins];
    const int        b = static_cast<int>( threadIdx.x );
    const float      h = static_cast<float>( hist[b] ) * 1.0e-12f;
    hist[b] = 0ULL;
    red[b] = h;
    __syncthreads();
    for ( int s = HallBins / 2; s > 0; s >>= 1 )
    {
        if ( b < s ) red[b] += red[b + s];
        __syncthreads();
    }
    const float total = red[0];
    e[b] = total > 1.0e-12f ? h / total * static_cast<float>( HallBins ) : 0.0f;
    __syncthreads();
    const float here = 0.25f * e[max( b - 1, 0 )] + 0.5f * e[b] + 0.25f * e[min( b + 1, HallBins - 1 )];
    if ( mh.m_densityCoef > 0.0f ) density[b] += ( here - density[b] ) * mh.m_densityCoef;
    if ( mh.m_memoryCoef > 0.0f && total > 1.0e-9f ) memory[b] += ( here - memory[b] ) * mh.m_memoryCoef;
}

/**
 * @brief One partial's complex gain at each ear: the head's shadow, the pinna and the time round
 *        the head (Brown and Duda 1998; Woodworth), the pair normalised to the pan law's power.
 * @param[in]  B      the binaural settings
 * @param[in]  place  the partial's place, -1 .. 1 (its azimuth over the span, plus the orbit)
 * @param[in]  u      its own die, 0 .. 1 (its elevation)
 * @param[in]  f      its frequency, Hz
 * @param[out] gl     receives the left ear's gain (real, imaginary)
 * @param[out] gr     receives the right ear's gain
 */
__device__ void earGains( const Binaural &B, float place, float u, float f, float *gl, float *gr )
{
    const float az = place * 1.5707963f * ( 1.0f + fminf( fmaxf( B.m_span, 0.0f ), 1.0f ) ) + B.m_angle;
    const float el = fminf( fmaxf( B.m_height, 0.0f ), 1.0f ) * 1.0471976f * ( 2.0f * u - 1.0f );
    float       sa, ca, se, ce;
    sincosf( az, &sa, &ca );
    sincosf( el, &se, &ce );
    // The pinna, the same at both ears here: five reflections whose times follow azimuth and
    // elevation (samples at 44.1 kHz), normalised by its mean power (1 + sum of rho^2 = 2.625).
    const float rho[5] = { 0.5f, -1.0f, 0.5f, -0.25f, 0.25f };
    const float A[5] = { 1.0f, 5.0f, 5.0f, 5.0f, 5.0f };
    const float Bn[5] = { 2.0f, 4.0f, 7.0f, 11.0f, 13.0f };
    const float D[5] = { 1.0f, 0.5f, 0.5f, 0.5f, 0.5f };
    const float half = fabsf( cosf( 0.5f * az ) );
    float       pr = 1.0f, pim = 0.0f;
    for ( int k = 0; k < 5; ++k )
    {
        const float tk = A[k] * half * sinf( D[k] * ( 1.5707963f - el ) ) + Bn[k];
        float       s, c;
        sincosf( -6.2831853f * f * tk * ( 1.0f / 44100.0f ), &s, &c );
        pr += rho[k] * c;
        pim += rho[k] * s;
    }
    pr *= 0.6172134f;
    pim *= 0.6172134f;
    const float x = sa * ce;   // towards the right ear
    float       g[2][2];
    for ( int e = 0; e < 2; ++e )
    {   // 0 left, 1 right
        const float beta = acosf(
            fminf( fmaxf( e == 1 ? x : -x, -1.0f ), 1.0f ) );   // 0 facing this ear .. pi behind the head from it
        const float alpha = 1.05f + 0.95f * cosf( beta * ( 180.0f / 150.0f ) );
        const float w = f * 8.0143e-4f;   // pi f a / c, a = 8.75 cm, c = 343 m/s
        const float den = 1.0f / ( 1.0f + w * w );
        const float hr = ( 1.0f + alpha * w * w ) * den, hi = ( alpha - 1.0f ) * w * den;
        const float tau = beta < 1.5707963f ? -2.551e-4f * cosf( beta ) : 2.551e-4f * ( beta - 1.5707963f );
        float       s, c;
        sincosf( -6.2831853f * f * tau, &s, &c );
        // head shadow x pinna x delay
        const float r1 = hr * pr - hi * pim, i1 = hr * pim + hi * pr;
        g[e][0] = r1 * c - i1 * s;
        g[e][1] = r1 * s + i1 * c;
    }
    const float norm =
        rsqrtf( g[0][0] * g[0][0] + g[0][1] * g[0][1] + g[1][0] * g[1][0] + g[1][1] * g[1][1] + 1.0e-20f );
    gl[0] = g[0][0] * norm;
    gl[1] = g[0][1] * norm;
    gr[0] = g[1][0] * norm;
    gr[1] = g[1][1] * norm;
}

/**
 * @brief What a partial reads from a cell of the living field, 0 .. 1: Lenia's value, or the
 *        Turing field's second chemical (V, which lives under 0.4) scaled up.
 * @param[in] mode  the field's mode
 * @param[in] c     the cell (Lenia: x; Turing: U, V)
 * @return 0 .. 1
 */
__device__ __forceinline__ float lifeValue( LifeModeE mode, float2 c )
{
    if ( mode == LifeModeE::TURING ) return fminf( c.y * 3.0f, 1.0f );
    if ( mode == LifeModeE::RAIN ) return 0.5f + 0.5f * tanhf( 3.0f * c.x );   // the water's height about its rest
    return c.x;
}

// ---- Animated pictures (KaleidoscopeEnhanced's Scene2D/Chain*.frag, its continuous transforms) ----

/**
 * @brief Smooth value noise, 0 .. 1 (corners hashed).
 * @param[in] x  across
 * @param[in] y  along
 * @return the noise
 */
__device__ float pictureNoise( float x, float y )
{
    const float ix = floorf( x ), iy = floorf( y ), fx = x - ix, fy = y - iy;
    const float sx = fx * fx * ( 3.0f - 2.0f * fx ), sy = fy * fy * ( 3.0f - 2.0f * fy );
    auto        h = []( float a, float b )
    {
        return static_cast<float>( hash32( static_cast<std::uint32_t>( static_cast<int>( a ) ) * 0x8DA6B343U ^
                                           static_cast<std::uint32_t>( static_cast<int>( b ) ) * 0xD8163841U ) >>
                                   8 ) *
               ( 1.0f / 16777216.0f );
    };
    const float a = h( ix, iy ), b = h( ix + 1.0f, iy ), c = h( ix, iy + 1.0f ), d = h( ix + 1.0f, iy + 1.0f );
    return a + ( b - a ) * sx + ( c - a ) * sy + ( a - b - c + d ) * sx * sy;
}

/**
 * @brief Three octaves of noise, 0 .. 1 (the shaders' fbm3).
 * @param[in] x  across
 * @param[in] y  along
 * @return the noise
 */
__device__ float pictureFbm( float x, float y )
{
    float v = 0.0f, a = 0.5f;
    for ( int i = 0; i < 3; ++i )
    {
        v += a * pictureNoise( x, y );
        const float nx = 1.6f * x + 1.2f * y + 2.3f, ny = -1.2f * x + 1.6f * y + 2.3f;
        x = nx;
        y = ny;
        a *= 0.5f;
    }
    return v / 0.875f;
}

/**
 * @brief The shaders' mirror repeat: identity on [0, 1], mirrored beyond, so every fold is seamless.
 * @param[in] u  a coordinate
 * @return 0 .. 1
 */
__device__ __forceinline__ float pictureMirror( float u )
{
    const float f = u * 0.5f - floorf( u * 0.5f );
    return 1.0f - fabsf( f * 2.0f - 1.0f );
}

/**
 * @brief The landscape as a picture at (u, v), mirror-repeated: u its time (left to right), v its
 *        frequency (low at the bottom); its brightness on the landscapes' 40 dB scale.
 * @param[in] hdr   the header (the landscape's size)
 * @param[in] land  the landscape
 * @param[in] u     across
 * @param[in] v     up
 * @return 0 .. 1
 */
__device__ float pictureAt( const BlockHeader *hdr, const float *land, float u, float v )
{
    const int cols = hdr->m_landColumns, rows = hdr->m_landRows;
    if ( cols < 2 || rows < 2 ) return 0.5f;
    const float  fr = pictureMirror( u ) * static_cast<float>( rows - 1 ),
                 fc = pictureMirror( v ) * static_cast<float>( cols - 1 );
    const int    r0 = min( static_cast<int>( fr ), rows - 2 ), c0 = min( static_cast<int>( fc ), cols - 2 );
    const float  ar = fr - static_cast<float>( r0 ), ac = fc - static_cast<float>( c0 );
    const float *p = land + static_cast<size_t>( r0 ) * cols + c0;
    const float  top = p[0] + ( p[1] - p[0] ) * ac, bottom = p[cols] + ( p[cols + 1] - p[cols] ) * ac;
    const float  a = top + ( bottom - top ) * ar;
    return fminf( fmaxf( 1.0f + 20.0f * log10f( fmaxf( a, 1.0e-6f ) ) / 40.0f, 0.0f ), 1.0f );
}

/**
 * @brief One cell of an animated picture. The field is the screen (twice as wide as high); the
 *        transforms are the shaders' (tKaleido, tTunnel, tSpiral, tWarp), the feedback and the
 *        rain read the field before this step.
 * @param[in] hdr   the header (its life: time, angle, sides, character)
 * @param[in] land  the landscape
 * @param[in] src   the field before this step
 * @param[in] x     the cell across
 * @param[in] y     the cell along
 * @return the cell (x: the value, or Rain's height; y: Rain's height before)
 */
__device__ float2 pictureCell( const BlockHeader *hdr, const float *land, const float2 *src, int x, int y )
{
    const LifeField &L = hdr->m_life;
    const float      px = ( static_cast<float>( x ) + 0.5f ) / static_cast<float>( LifeW ),
                     py = ( static_cast<float>( y ) + 0.5f ) / static_cast<float>( LifeH );
    const float      dx = ( px - 0.5f ) * 2.0f, dy = py - 0.5f;   // centred, the aspect kept
    const float      r = fmaxf( sqrtf( dx * dx + dy * dy ), 1.0e-4f ), a = atan2f( dy, dx );
    const float      T = L.m_time, ch = L.m_character, S = fmaxf( L.m_sides, 2.0f );
    const float      Pi = 3.14159265f;
    switch ( L.m_mode )
    {
    case LifeModeE::KALEIDOSCOPE:
    {
        // tKaleido round a centre that wanders slowly through the picture, the picture flowing under it.
        const float cx = 0.5f + 0.2f * __sinf( 0.0107f * L.m_seconds ),
                    cy = 0.5f + 0.2f * __cosf( 0.0131f * L.m_seconds );
        const float sec = 2.0f * Pi / S;
        const float w = fabsf( a - sec * floorf( a / sec ) - 0.5f * sec ) + L.m_angle;
        float       u = cx + 0.6f * r * __cosf( w ) + 0.05f * T, v = cy + 0.6f * r * __sinf( w );
        u += ch * 0.15f * ( pictureFbm( 3.0f * u + T, 3.0f * v ) - 0.5f );
        v += ch * 0.15f * ( pictureFbm( 3.0f * u + 5.2f, 3.0f * v - T ) - 0.5f );
        return make_float2( pictureAt( hdr, land, u, v ), 0.0f );
    }
    case LifeModeE::TUNNEL:
    {
        // tTunnel: the angle round (folded into Sides mirror wedges), the depth 1/r along, flying in.
        const float sec = 2.0f * Pi / S;
        const float w = fabsf( a + L.m_angle - sec * floorf( ( a + L.m_angle ) / sec ) - 0.5f * sec ) / sec;
        const float depth = 0.25f / r + 0.3f * T;
        return make_float2( pictureAt( hdr, land, w + ch * 0.3f * depth, depth ), 0.0f );
    }
    case LifeModeE::SPIRAL:
    {
        // tSpiral: log-polar, the angle sheared into the radius -- an endless zoom along Sides arms.
        const float arms = 2.0f * floorf( S * 0.5f );
        const float u = logf( r ) * ( 0.5f + 0.5f * ch ) - 0.3f * T + ( a + L.m_angle ) / Pi;
        const float v = ( a + L.m_angle ) / Pi * arms * 0.5f;
        return make_float2( pictureAt( hdr, land, u, v ), 0.0f );
    }
    case LifeModeE::FLOW:
    {
        // tWarp, twice: the picture carried on a current of noise that itself drifts.
        float       u = px + 0.05f * T * __cosf( L.m_angle ), v = py + 0.05f * T * __sinf( L.m_angle );
        const float s = 0.15f + 0.45f * ch;
        for ( int k = 0; k < 2; ++k )
        {
            const float wu = pictureFbm( 2.5f * u + 0.3f * T, 2.5f * v ) - 0.5f,
                        wv = pictureFbm( 2.5f * u + 5.2f, 2.5f * v - 0.3f * T ) - 0.5f;
            u += s * wu;
            v += s * wv;
        }
        return make_float2( pictureAt( hdr, land, u, v ), 0.0f );
    }
    case LifeModeE::FEEDBACK:
    {
        // Video feedback: this cell takes the field from a point turned and drawn towards the middle
        // (so everything spirals outward), fading a little, and the picture seeps in.
        // A step every block (375 a second): it zooms out by 0.15 .. 0.6 a second and forgets in about three.
        const float step = 0.0004f * ( 1.0f + 3.0f * ch ), rot = L.m_turnStep;
        const float cs = __cosf( rot ), sn = __sinf( rot );
        const float qx = ( dx * cs - dy * sn ) / ( 1.0f + step ), qy = ( dx * sn + dy * cs ) / ( 1.0f + step );
        const float fx = ( qx * 0.5f + 0.5f ) * static_cast<float>( LifeW ) - 0.5f,
                    fy = ( qy + 0.5f ) * static_cast<float>( LifeH ) - 0.5f;
        const int   x0 = static_cast<int>( floorf( fx ) ), y0 = static_cast<int>( floorf( fy ) );
        const float ax = fx - static_cast<float>( x0 ), ay = fy - static_cast<float>( y0 );
        auto at = [&]( int xx, int yy ) { return src[( yy & ( LifeH - 1 ) ) * LifeW + ( xx & ( LifeW - 1 ) )].x; };
        const float back = ( at( x0, y0 ) * ( 1.0f - ax ) + at( x0 + 1, y0 ) * ax ) * ( 1.0f - ay ) +
                           ( at( x0, y0 + 1 ) * ( 1.0f - ax ) + at( x0 + 1, y0 + 1 ) * ax ) * ay;
        const float seep = pictureAt( hdr, land, px + 0.02f * T, py + 0.5f * __sinf( 0.01f * L.m_seconds ) );
        return make_float2( fminf( fmaxf( 0.997f * back + 0.003f * seep, 0.0f ), 1.0f ), 0.0f );
    }
    case LifeModeE::RAIN:
    {
        // A damped 2D wave (leapfrog, the five-point Laplacian), struck at chance cells.
        const float2 c = src[y * LifeW + x];
        const float  l = src[y * LifeW + ( ( x - 1 ) & ( LifeW - 1 ) )].x,
                     rr = src[y * LifeW + ( ( x + 1 ) & ( LifeW - 1 ) )].x;
        const float  u = src[( ( y - 1 ) & ( LifeH - 1 ) ) * LifeW + x].x,
                     d = src[( ( y + 1 ) & ( LifeH - 1 ) ) * LifeW + x].x;
        // A step every block: the rings cross the field in about three seconds; Character keeps them ringing.
        const float damp = 0.0005f + 0.004f * ( 1.0f - ch );
        float next = ( 2.0f * c.x - ( 1.0f - damp ) * c.y + 0.05f * ( l + rr + u + d - 4.0f * c.x ) ) / ( 1.0f + damp );
        const float roll = static_cast<float>( hash32( static_cast<std::uint32_t>( y * LifeW + x ) * 0x9E3779B9U ^
                                                       hdr->m_blockIndex * 0x85EBCA6BU ) >>
                                               8 ) *
                           ( 1.0f / 16777216.0f );
        if ( roll < L.m_drops ) next += 1.5f;
        return make_float2( fminf( fmaxf( next, -4.0f ), 4.0f ), c.x );
    }
    case LifeModeE::STARFIELD:
    {
        // Stars on a log-polar grid streaming outwards; three layers of different density.
        float v = 0.0f;
        for ( int k = 0; k < 3; ++k )
        {
            const float         density = 6.0f * S * ( 1.0f + static_cast<float>( k ) );
            const float         gu = ( a + L.m_angle * ( 1.0f + 0.3f * k ) ) / ( 2.0f * Pi ) * density,
                                gv = ( logf( r ) - T * ( 0.6f + 0.4f * k ) ) * density * 0.25f;
            const float         cu = floorf( gu ), cv = floorf( gv );
            const std::uint32_t hs =
                hash32( static_cast<std::uint32_t>( static_cast<int>( cu ) ) * 0x27D4EB2DU ^
                        static_cast<std::uint32_t>( static_cast<int>( cv ) ) * 0x165667B1U ^ ( k * 0x9E3779B9U ) );
            if ( ( hs & 255U ) < static_cast<std::uint32_t>( 40 + 60 * ch ) )
            {
                const float su = gu - cu - ( 0.2f + 0.6f * static_cast<float>( ( hs >> 8 ) & 255U ) / 255.0f ),
                            sv = gv - cv - ( 0.2f + 0.6f * static_cast<float>( ( hs >> 16 ) & 255U ) / 255.0f );
                v = fmaxf( v, __expf( -( su * su + sv * sv ) * 25.0f ) * fminf( r * 3.0f, 1.0f ) );
            }
        }
        // A faint glow under the stars: a sky of nothing but points is so sparse that no make-up can
        // keep its level (measured 2 dB under), and the partials between them would all but vanish.
        return make_float2( 0.12f + 0.88f * v, 0.0f );
    }
    case LifeModeE::MOIRE:
    {
        // Two line gratings, nearly alike and turning nearly together: where they agree and where they
        // cancel makes broad bands that wander slowly (the moire), far wider than the lines.
        const float f1 = 2.0f * S, f2 = f1 * ( 1.0f + 0.02f + 0.08f * ch );
        const float a1 = L.m_angle, a2 = 0.9f * L.m_angle + 0.08f;
        const float g1 = 0.5f + 0.5f * __cosf( 2.0f * Pi * f1 * ( dx * __cosf( a1 ) + dy * __sinf( a1 ) ) + T );
        const float g2 = 0.5f + 0.5f * __cosf( 2.0f * Pi * f2 * ( dx * __cosf( a2 ) + dy * __sinf( a2 ) ) - 0.7f * T );
        return make_float2( g1 * g2, 0.0f );
    }
    default: return make_float2( 0.0f, 0.0f );
    }
}

/**
 * @brief The field buffer that holds the field after this block's steps.
 * @param[in] hdr  the header
 * @param[in] a    buffer A
 * @param[in] b    buffer B
 * @return the current one
 */
__device__ __forceinline__ const float2 *lifeCurrent( const BlockHeader *hdr, const float2 *a, const float2 *b )
{
    return ( ( hdr->m_lifeParity + hdr->m_life.m_steps ) & 1 ) ? b : a;
}

/**
 * @brief Seeds the living field when asked (hdr->life.reseed): Lenia a few patches of noise, the
 *        Turing field U = 1, V = 0 with a few squares of V -- from a hash of the cell and the salt.
 * @param[in]     hdr  the header
 * @param[in,out] a    buffer A
 * @param[in,out] b    buffer B (the one before the steps is seeded)
 */
__global__ void lifeSeedKernel( const BlockHeader *hdr, float2 *a, float2 *b )
{
    const LifeField &L = hdr->m_life;
    if ( !L.m_reseed || L.m_mode == LifeModeE::OFF ) return;
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    float2             *dst = ( hdr->m_lifeParity & 1 ) ? b : a;
    const int           x = i & ( LifeW - 1 ), y = i >> 8;
    const std::uint32_t salt = static_cast<std::uint32_t>( L.m_salt ) * 0x9E3779B9U;
    float2              c = L.m_mode == LifeModeE::TURING ? make_float2( 1.0f, 0.0f ) : make_float2( 0.0f, 0.0f );
    if ( lifeIsPicture( L.m_mode ) )
    {
        dst[i] = c;
        return;
    }   // the pictures are computed, Rain starts from still water
    // Turing dense (eighty small squares: a sparse field is uneven where the partials read it --
    // measured 3 dB over while it grew from fourteen), Lenia a few large patches of noise.
    const int patches = L.m_mode == LifeModeE::TURING ? 80 : 8;
    for ( int k = 0; k < patches; ++k )
    {
        const std::uint32_t h = hash32( salt + 7919U * static_cast<std::uint32_t>( k ) );
        const int cx = static_cast<int>( h & ( LifeW - 1 ) ), cy = static_cast<int>( ( h >> 8 ) & ( LifeH - 1 ) );
        const int r = L.m_mode == LifeModeE::TURING ? 3 + static_cast<int>( ( h >> 16 ) % 5U )
                                                    : 10 + static_cast<int>( ( h >> 16 ) % 9U );
        int       dx = abs( x - cx ), dy = abs( y - cy );
        dx = min( dx, LifeW - dx );
        dy = min( dy, LifeH - dy );
        if ( dx > r || dy > r ) continue;
        const float noise = static_cast<float>( hash32( salt ^ static_cast<std::uint32_t>( i ) * 2654435761U ) >> 8 ) *
                            ( 1.0f / 16777216.0f );
        if ( L.m_mode == LifeModeE::TURING )
            c = make_float2( 0.5f, 0.25f + 0.05f * noise );
        else
            c.x = fmaxf( c.x, noise );
    }
    dst[i] = c;
}

/**
 * @brief One step of the living field, cell by cell, from one buffer into the other (which one by
 *        the parity and the step's number); a step this block does not have does nothing.
 *
 * Lenia (Chan 2019): the ring kernel's weighted sum u of the neighbourhood, the growth
 * 2 exp(-(u - mu)^2 / 2 sigma^2) - 1, a tenth of it added. Gray-Scott (Pearson 1993):
 * dU = Du lap U - U V^2 + F (1 - U), dV = Dv lap V + U V^2 - (F + k) V, Du 0.2097, Dv 0.105, one
 * unit of time a step, the five-point Laplacian.
 * @param[in]     hdr   the header
 * @param[in]     node  this step's number in the block, 0 or 1
 * @param[in,out] a     buffer A
 * @param[in,out] b     buffer B
 * @param[in]     land  the landscape (the animated pictures read it)
 */
__global__ void lifeStepKernel( const BlockHeader *hdr, int node, float2 *a, float2 *b, const float *land )
{
    const LifeField &L = hdr->m_life;
    if ( L.m_mode == LifeModeE::OFF || L.m_mode == LifeModeE::FLUID || L.m_mode == LifeModeE::ISLANDS ||
         node >= L.m_steps )
        return;   // those two: their own kernels
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    const bool    fromB = ( ( hdr->m_lifeParity + node ) & 1 ) != 0;
    const float2 *src = fromB ? b : a;
    float2       *dst = fromB ? a : b;
    const int     x = i & ( LifeW - 1 ), y = i >> 8;
    if ( lifeIsPicture( L.m_mode ) )
    {
        dst[i] = pictureCell( hdr, land, src, x, y );
    }
    else if ( L.m_mode == LifeModeE::LENIA )
    {
        constexpr int R = LifeRadius, K = 2 * LifeRadius + 1;
        float         u = 0.0f;
        for ( int dy = -R; dy <= R; ++dy )
        {
            const float2 *row = src + static_cast<size_t>( ( y + dy ) & ( LifeH - 1 ) ) * LifeW;
            for ( int dx = -R; dx <= R; ++dx )
            {
                const float k = cLeniaKernel[( dy + R ) * K + ( dx + R )];
                if ( k != 0.0f ) u += k * row[( x + dx ) & ( LifeW - 1 )].x;
            }
        }
        const float d = ( u - L.m_lenMu ) / fmaxf( L.m_lenSigma, 1.0e-4f );
        const float g = 2.0f * __expf( -0.5f * d * d ) - 1.0f;
        dst[i] = make_float2( fminf( fmaxf( src[i].x + 0.1f * g, 0.0f ), 1.0f ), 0.0f );
    }
    else
    {
        const float2 c = src[i];
        const float2 l = src[y * LifeW + ( ( x - 1 ) & ( LifeW - 1 ) )],
                     r = src[y * LifeW + ( ( x + 1 ) & ( LifeW - 1 ) )];
        const float2 u = src[( ( y - 1 ) & ( LifeH - 1 ) ) * LifeW + x],
                     d = src[( ( y + 1 ) & ( LifeH - 1 ) ) * LifeW + x];
        const float  lapU = l.x + r.x + u.x + d.x - 4.0f * c.x, lapV = l.y + r.y + u.y + d.y - 4.0f * c.y;
        const float  uvv = c.x * c.y * c.y;
        dst[i] = make_float2( fminf( fmaxf( c.x + 0.2097f * lapU - uvv + L.m_feed * ( 1.0f - c.x ), 0.0f ), 1.0f ),
                              fminf( fmaxf( c.y + 0.105f * lapV + uvv - ( L.m_feed + L.m_kill ) * c.y, 0.0f ), 1.0f ) );
    }
}

// ---- The Fluid: a lattice Boltzmann fluid (D2Q9, BGK) on the field's torus, with ink ----

__constant__ int   cLbmX[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };   ///< the lattice's directions, across
__constant__ int   cLbmY[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };   ///< and along
__constant__ float cLbmW[9] = { 4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f, 1.0f / 9.0f,
                                1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };   ///< their weights

/**
 * @brief One direction's equilibrium.
 * @param[in] k    the direction
 * @param[in] rho  the density
 * @param[in] ux   the velocity across
 * @param[in] uy   along
 * @return f_eq
 */
__device__ __forceinline__ float lbmEq( int k, float rho, float ux, float uy )
{
    const float eu = static_cast<float>( cLbmX[k] ) * ux + static_cast<float>( cLbmY[k] ) * uy;
    return cLbmW[k] * rho * ( 1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * ( ux * ux + uy * uy ) );
}

/**
 * @brief The shortest way round the torus.
 * @param[in] d  a difference
 * @param[in] n  the size
 * @return it within -n/2 .. n/2
 */
__device__ __forceinline__ float torusDelta( float d, float n )
{
    return d - n * rintf( d / n );
}

/**
 * @brief Seeds the Fluid when asked: at rest but for six eddies of either turn (from the salt).
 * @param[in]     hdr  the header
 * @param[in,out] fa   the lattice, buffer A
 * @param[in,out] fb   buffer B
 */
__global__ void fluidSeedKernel( const BlockHeader *hdr, float *fa, float *fb )
{
    const LifeField &L = hdr->m_life;
    if ( !L.m_reseed || L.m_mode != LifeModeE::FLUID ) return;
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    float              *dst = ( hdr->m_lifeParity & 1 ) ? fb : fa;
    const int           x = i & ( LifeW - 1 ), y = i >> 8;
    const std::uint32_t salt = static_cast<std::uint32_t>( L.m_salt ) * 0x9E3779B9U;
    float               ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 6; ++k )
    {
        const std::uint32_t h = hash32( salt + 104729U * static_cast<std::uint32_t>( k + 1 ) );
        const float cx = static_cast<float>( h & ( LifeW - 1 ) ), cy = static_cast<float>( ( h >> 8 ) & ( LifeH - 1 ) );
        const float turn = ( h >> 20 ) & 1U ? 1.0f : -1.0f;
        const float dx = torusDelta( static_cast<float>( x ) - cx, static_cast<float>( LifeW ) );
        const float dy = torusDelta( static_cast<float>( y ) - cy, static_cast<float>( LifeH ) );
        const float g =
            0.006f * turn * __expf( -( dx * dx + dy * dy ) * ( 1.0f / 128.0f ) );   // an eddy eight cells across
        ux -= g * dy;
        uy += g * dx;
    }
    for ( int k = 0; k < 9; ++k ) dst[static_cast<size_t>( k ) * LifeCells + i] = lbmEq( k, 1.0f, ux, uy );
}

/**
 * @brief One step of the Fluid, cell by cell: the lattice streamed (pulled from the neighbours),
 *        pushed by the jets (the velocity shifted by tau F / rho), relaxed to its equilibrium; the
 *        ink carried back along the flow, the jets' ink added, a little of it fading. Reads one
 *        buffer, writes the other, as the field's steps do.
 * @param[in]     hdr    the header
 * @param[in]     node   this step's number in the block, 0 or 1
 * @param[in,out] fa     the lattice, buffer A
 * @param[in,out] fb     buffer B
 * @param[in,out] lifeA  the field (the ink), buffer A
 * @param[in,out] lifeB  buffer B
 */
__global__ void fluidStepKernel( const BlockHeader *hdr, int node, float *fa, float *fb, float2 *lifeA, float2 *lifeB )
{
    const LifeField &L = hdr->m_life;
    if ( L.m_mode != LifeModeE::FLUID || node >= L.m_steps ) return;
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    const bool    fromB = ( ( hdr->m_lifeParity + node ) & 1 ) != 0;
    const float  *src = fromB ? fb : fa;
    float        *dst = fromB ? fa : fb;
    const float2 *ink = fromB ? lifeB : lifeA;
    float2       *inkOut = fromB ? lifeA : lifeB;
    const int     x = i & ( LifeW - 1 ), y = i >> 8;
    float         f[9], rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 9; ++k )
    {
        const int sx = ( x - cLbmX[k] ) & ( LifeW - 1 ), sy = ( y - cLbmY[k] ) & ( LifeH - 1 );
        f[k] = src[static_cast<size_t>( k ) * LifeCells + sy * LifeW + sx];
        rho += f[k];
        ux += f[k] * static_cast<float>( cLbmX[k] );
        uy += f[k] * static_cast<float>( cLbmY[k] );
    }
    if ( !( rho > 0.05f && rho < 20.0f ) )
    {   // a lattice that left the world: at rest again
        rho = 1.0f;
        ux = uy = 0.0f;
        for ( int k = 0; k < 9; ++k ) f[k] = lbmEq( k, 1.0f, 0.0f, 0.0f );
    }
    else
    {
        ux /= rho;
        uy /= rho;
    }
    // The jets: a little above the bottom, each where its note lies across, blowing up with its lean.
    float fx = 0.0f, fy = 0.0f, inkIn = 0.0f;
    for ( int j = 0; j < LifeJets; ++j )
    {
        const float amt = L.m_jetAmt[j];
        if ( amt <= 0.0f ) continue;
        const float dx = torusDelta( static_cast<float>( x ) - L.m_jetX[j], static_cast<float>( LifeW ) );
        const float dy = torusDelta( static_cast<float>( y ) - 6.0f, static_cast<float>( LifeH ) );
        const float g = amt * __expf( -( dx * dx + dy * dy ) * ( 1.0f / 18.0f ) );   // three cells wide
        fx += g * L.m_jetSway[j];
        fy += g;
        inkIn += g;
    }
    const float tau = fmaxf( L.m_tau, 0.502f );
    float       vx = ux + tau * L.m_push * fx / rho, vy = uy + tau * L.m_push * fy / rho;
    const float speed = sqrtf( vx * vx + vy * vy );
    if ( speed > 0.12f )
    {
        vx *= 0.12f / speed;
        vy *= 0.12f / speed;
    }   // under a tenth of the lattice's sound: it stays a fluid
    for ( int k = 0; k < 9; ++k )
        dst[static_cast<size_t>( k ) * LifeCells + i] = f[k] - ( f[k] - lbmEq( k, rho, vx, vy ) ) / tau;
    // The ink, from where the flow brought it (bilinear on the torus).
    const float px = static_cast<float>( x ) - ux, py = static_cast<float>( y ) - uy;
    const int   x0 = static_cast<int>( floorf( px ) ), y0 = static_cast<int>( floorf( py ) );
    const float ax = px - static_cast<float>( x0 ), ay = py - static_cast<float>( y0 );
    auto        at = [&]( int cx, int cy ) { return ink[( cy & ( LifeH - 1 ) ) * LifeW + ( cx & ( LifeW - 1 ) )].x; };
    const float top = at( x0, y0 ) + ( at( x0 + 1, y0 ) - at( x0, y0 ) ) * ax;
    const float bottom = at( x0, y0 + 1 ) + ( at( x0 + 1, y0 + 1 ) - at( x0, y0 + 1 ) ) * ax;
    const float d = fminf( ( top + ( bottom - top ) * ay ) * L.m_fade + 0.03f * inkIn, 1.0f );
    inkOut[i] = make_float2( d, 10.0f * sqrtf( ux * ux + uy * uy ) );
}

// ---- Islands: Cahn-Hilliard separation with Oono and Puri's long-range term, on a coarse grid ----

/**
 * @brief The Islands, in one CUDA block: its 64 x 32 grid in shared memory, `substeps` explicit
 *        steps of dc = lap(c^3 - c - g lap c) - eps (c - mean) (g 0.6, a fiftieth of a unit each, a
 *        grain of noise so it never quite freezes), then the field drawn from it: each of its cells
 *        read bilinearly, drifted across, as 0.5 + 0.5 tanh(2.5 c). Seeded with a little noise about
 *        the mean when asked.
 * @param[in]     hdr    the header
 * @param[in,out] isles  the grid, kept between blocks
 * @param[in,out] lifeA  the field, buffer A
 * @param[in,out] lifeB  buffer B
 */
__global__ void __launch_bounds__( 1024 )
    islandsKernel( const BlockHeader *hdr, float *isles, float2 *lifeA, float2 *lifeB )
{
    const LifeField &L = hdr->m_life;
    if ( L.m_mode != LifeModeE::ISLANDS || L.m_steps < 1 ) return;
    constexpr int       N = IsleW * IsleH;
    __shared__ float    c[N], mu[N];
    const int           t = static_cast<int>( threadIdx.x );
    const std::uint32_t salt = static_cast<std::uint32_t>( L.m_salt ) * 0x9E3779B9U;
    for ( int k = t; k < N; k += blockDim.x )
        c[k] = L.m_reseed
                   ? L.m_mean + 0.2f * ( static_cast<float>(
                                             hash32( salt ^ ( static_cast<std::uint32_t>( k ) * 2654435761U ) ) >> 8 ) *
                                             ( 1.0f / 16777216.0f ) -
                                         0.5f )
                   : isles[k];
    __syncthreads();
    constexpr float g = 0.6f, dt = 0.02f;
    for ( int s = 0; s < L.m_substeps; ++s )
    {
        for ( int k = t; k < N; k += blockDim.x )
        {
            const int   x = k & ( IsleW - 1 ), y = k / IsleW;
            const float lap = c[y * IsleW + ( ( x + 1 ) & ( IsleW - 1 ) )] +
                              c[y * IsleW + ( ( x - 1 ) & ( IsleW - 1 ) )] +
                              c[( ( y + 1 ) & ( IsleH - 1 ) ) * IsleW + x] +
                              c[( ( y - 1 ) & ( IsleH - 1 ) ) * IsleW + x] - 4.0f * c[k];
            mu[k] = c[k] * c[k] * c[k] - c[k] - g * lap;
        }
        __syncthreads();
        for ( int k = t; k < N; k += blockDim.x )
        {
            const int           x = k & ( IsleW - 1 ), y = k / IsleW;
            const float         lap = mu[y * IsleW + ( ( x + 1 ) & ( IsleW - 1 ) )] +
                                      mu[y * IsleW + ( ( x - 1 ) & ( IsleW - 1 ) )] +
                                      mu[( ( y + 1 ) & ( IsleH - 1 ) ) * IsleW + x] +
                                      mu[( ( y - 1 ) & ( IsleH - 1 ) ) * IsleW + x] - 4.0f * mu[k];
            const std::uint32_t h =
                hash32( ( hdr->m_blockIndex * 16U + static_cast<std::uint32_t>( s ) ) * 2654435761U ^
                        static_cast<std::uint32_t>( k ) ^ salt );
            const float noise = 0.004f * ( static_cast<float>( h >> 8 ) * ( 1.0f / 16777216.0f ) - 0.5f );
            c[k] = fminf( fmaxf( c[k] + dt * ( lap - L.m_eps * ( c[k] - L.m_mean ) ) + noise, -1.5f ), 1.5f );
        }
        __syncthreads();
    }
    for ( int k = t; k < N; k += blockDim.x ) isles[k] = c[k];
    // The field (the buffer this block's one step leads to), drawn from the grid.
    float2         *dst = ( hdr->m_lifeParity & 1 ) ? lifeA : lifeB;
    constexpr float sx = static_cast<float>( IsleW ) / static_cast<float>( LifeW ),
                    sy = static_cast<float>( IsleH ) / static_cast<float>( LifeH );
    for ( int i = t; i < LifeCells; i += blockDim.x )
    {
        const float px = ( static_cast<float>( i & ( LifeW - 1 ) ) + 0.5f ) * sx - 0.5f + L.m_drift,
                    py = ( static_cast<float>( i >> 8 ) + 0.5f ) * sy - 0.5f;
        const int   x0 = static_cast<int>( floorf( px ) ), y0 = static_cast<int>( floorf( py ) );
        const float ax = px - static_cast<float>( x0 ), ay = py - static_cast<float>( y0 );
        auto        at = [&]( int cx, int cy ) { return c[( cy & ( IsleH - 1 ) ) * IsleW + ( cx & ( IsleW - 1 ) )]; };
        const float top = at( x0, y0 ) + ( at( x0 + 1, y0 ) - at( x0, y0 ) ) * ax;
        const float bottom = at( x0, y0 + 1 ) + ( at( x0 + 1, y0 + 1 ) - at( x0, y0 + 1 ) ) * ax;
        const float v = top + ( bottom - top ) * ay;
        dst[i] = make_float2( 0.5f + 0.5f * tanhf( 2.5f * v ), v );
    }
}

/**
 * @brief The living field's mean and mean square of what the partials read, in one CUDA block, in
 *        a fixed order (bit-identical runs).
 * @param[in]  hdr    the header
 * @param[in]  a      buffer A
 * @param[in]  b      buffer B
 * @param[out] stats  receives mean, mean square
 */
__global__ void lifeStatsKernel( const BlockHeader *hdr, const float2 *a, const float2 *b, float *stats )
{
    __shared__ float s1[256], s2[256];
    const LifeField &L = hdr->m_life;
    const int        t = static_cast<int>( threadIdx.x );
    float            m = 0.0f, q = 0.0f;
    if ( L.m_mode != LifeModeE::OFF )
    {
        const float2 *cur = lifeCurrent( hdr, a, b );
        for ( int i = t; i < LifeCells; i += 256 )
        {
            const float v = lifeValue( L.m_mode, cur[i] );
            m += v;
            q += v * v;
        }
    }
    s1[t] = m;
    s2[t] = q;
    __syncthreads();
    for ( int s = 128; s > 0; s >>= 1 )
    {
        if ( t < s )
        {
            s1[t] += s1[t + s];
            s2[t] += s2[t + s];
        }
        __syncthreads();
    }
    if ( t == 0 )
    {
        stats[0] = s1[0] / LifeCells;
        stats[1] = s2[0] / LifeCells;
    }
}

/**
 * @brief The living field at a place, read bilinearly on the torus.
 * @param[in] hdr  the header
 * @param[in] cur  the current buffer
 * @param[in] fx   across, cells (wraps)
 * @param[in] fy   along, cells (wraps)
 * @return what the partial reads, 0 .. 1
 */
__device__ float lifeAt( const BlockHeader *hdr, const float2 *cur, float fx, float fy )
{
    const int       x0 = static_cast<int>( floorf( fx ) ), y0 = static_cast<int>( floorf( fy ) );
    const float     ax = fx - static_cast<float>( x0 ), ay = fy - static_cast<float>( y0 );
    const LifeModeE mode = hdr->m_life.m_mode;
    auto            at = [&]( int x, int y )
    { return lifeValue( mode, cur[( y & ( LifeH - 1 ) ) * LifeW + ( x & ( LifeW - 1 ) )] ); };
    const float top = at( x0, y0 ) + ( at( x0 + 1, y0 ) - at( x0, y0 ) ) * ax;
    const float bottom = at( x0, y0 + 1 ) + ( at( x0 + 1, y0 + 1 ) - at( x0, y0 + 1 ) ) * ax;
    return top + ( bottom - top ) * ay;
}

/**
 * @brief The partial sync's mean fields: for every harmonic of every voice (and its air band, as one
 *        more group) the means of its members' phasors from the last block, summed in member order.
 *        Global: e^(iθ). Chimera: e^(iθ), e^(i(θ - x)), e^(i(θ + x)) with x the member's place round
 *        the ring. Swarm: e^(i(x + θ)), e^(i(x - θ)) with x its place on its circle.
 * @param[in]     hdr     the header
 * @param[in]     voices  the voices
 * @param[in]     pcap    partials per voice in the tables
 * @param[in]     theta   per partial its phase
 * @param[in]     sx      per partial its place on its circle (Swarm)
 * @param[in,out] order   per voice and group the mean phasors, three planes `plane` apart
 * @param[in]     plane   the planes' distance (all the partials)
 */
__global__ void syncOrderKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const float *theta,
                                 const float *sx, float2 *order, size_t plane )
{
    if ( hdr->m_sync.m_depth <= 0.0f ) return;
    const int gid = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    const int v = gid / pcap, g = gid - v * pcap;
    if ( v >= hdr->m_voices ) return;
    const CloudVoice &cv = voices[v];
    if ( !cv.m_active ) return;
    const int H = cv.m_harmonics, K = cv.m_members, HP = cv.m_harmonicPartials;
    int       first, count;
    if ( g < H )
    {
        first = g * K;
        count = K;
    }
    else if ( g == H && cv.m_partials > HP )
    {
        first = HP;
        count = cv.m_partials - HP;
    }
    else if ( g > H && g <= H + cv.m_subClouds )
    {   // a sub cloud (Subharmonics), after the harmonics' clouds
        first = ( g - 1 ) * K;
        count = K;
    }
    else
        return;
    const SyncModeE mode = hdr->m_sync.m_mode;
    float2          z0 = make_float2( 0.0f, 0.0f ), z1 = z0, z2 = z0;
    const float    *th = theta + static_cast<size_t>( v ) * pcap + first;
    const float    *xs = sx + static_cast<size_t>( v ) * pcap + first;
    const float     ring = 6.2831853f / static_cast<float>( max( count, 1 ) );
    for ( int j = 0; j < count; ++j )
    {
        float s, c;
        if ( mode == SyncModeE::SWARM )
        {
            __sincosf( xs[j] + th[j], &s, &c );
            z1.x += c;
            z1.y += s;
            __sincosf( xs[j] - th[j], &s, &c );
            z2.x += c;
            z2.y += s;
            continue;
        }
        __sincosf( th[j], &s, &c );
        z0.x += c;
        z0.y += s;
        if ( mode == SyncModeE::CHIMERA )
        {
            const float x = ring * ( static_cast<float>( j ) + 0.5f );
            __sincosf( th[j] - x, &s, &c );
            z1.x += c;
            z1.y += s;
            __sincosf( th[j] + x, &s, &c );
            z2.x += c;
            z2.y += s;
        }
    }
    const float  inv = 1.0f / static_cast<float>( max( count, 1 ) );
    const size_t at = static_cast<size_t>( v ) * pcap + g;
    order[at] = make_float2( z0.x * inv, z0.y * inv );
    order[at + plane] = make_float2( z1.x * inv, z1.y * inv );
    order[at + 2 * plane] = make_float2( z2.x * inv, z2.y * inv );
}

/**
 * @brief A member's place in its cloud in cents (its slice of the bell, jittered as the update
 *        kernel places it).
 * @param[in] seed   the note's dice
 * @param[in] first  the cloud's first partial in the voice
 * @param[in] j      the member
 * @param[in] count  the cloud's members
 * @param[in] sigma  the cloud's width, cents per unit of the bell
 * @return cents from the cloud's centre
 */
__device__ __forceinline__ float memberCents( std::uint32_t seed, int first, int j, int count, float sigma )
{
    if ( count <= 1 ) return 0.0f;
    const std::uint32_t up = static_cast<std::uint32_t>( first + j );
    return normcdfinvf( ( static_cast<float>( j ) + 0.5f + 0.9f * ( hash01( seed, up, 1 ) - 0.5f ) ) /
                        static_cast<float>( count ) ) *
           sigma;
}

/**
 * @brief One block of the flock for one partial (Reynolds 1987, in frequency): its offset and
 *        velocity steered by the three members on either side in its cloud -- towards their mean
 *        offset (cohesion), towards their mean velocity (alignment), away from any that has come closer
 *        than half their distance at home (separation: the cloud keeps its shape; a fixed distance of
 *        six cents blew a cloud of 7 cents up to 20, its members lie a cent apart) -- plus its cloud's
 *        wind, a spring home and drag.
 *        Read from the last block's state, written into the other buffer: bit-identical.
 * @param[in]     F        the field (spring; strength: the wind)
 * @param[in]     prev     the last block's (offset, velocity) per partial
 * @param[in,out] next     this block's
 * @param[in]     vbase    the voice's first partial in the tables
 * @param[in]     first    the cloud's first partial in the voice
 * @param[in]     count    its members
 * @param[in]     k        this partial's member
 * @param[in]     seed     the note's dice
 * @param[in]     sigma    the cloud's width, cents per unit of the bell
 * @param[in]     wind     its cloud's wind, cents per second squared
 * @param[in]     now      the offset it has now (taken over when the flock is fresh)
 * @param[in]     restart  a new note: offset and velocity 0
 * @param[in]     dt       the block's length, seconds
 * @return the new offset, cents
 */
__device__ float flockStep( const SpectralField &F, const float2 *prev, float2 *next, size_t vbase, int first,
                            int count, int k, std::uint32_t seed, float sigma, float wind, float now, bool restart,
                            float dt )
{
    const size_t me = vbase + static_cast<size_t>( first + k );
    if ( restart )
    {
        next[me] = make_float2( 0.0f, 0.0f );
        return 0.0f;
    }
    if ( F.m_flockFresh )
    {
        next[me] = make_float2( now, 0.0f );
        return now;
    }
    const float2 s = prev[me];
    const float  myHome = memberCents( seed, first, k, count, sigma );
    const float  myX = myHome + s.x;
    float        sumPos = 0.0f, sumVel = 0.0f, push = 0.0f;
    int          nb = 0;
    for ( int d = -3; d <= 3; ++d )
    {
        const int j = k + d;
        if ( d == 0 || j < 0 || j >= count ) continue;
        const float2 o = prev[vbase + static_cast<size_t>( first + j )];
        sumPos += o.x;
        sumVel += o.y;
        ++nb;
        const float home = memberCents( seed, first, j, count, sigma );
        const float sep = 0.5f * fabsf( myHome - home );
        const float dx = myX - ( home + o.x );
        if ( fabsf( dx ) < sep )
            push += ( j < k ? 1.0f : -1.0f ) * ( sep - fabsf( dx ) );   // the order kept: a lower member stays below
    }
    float acc = wind + 40.0f * push - 1.5f * s.y - ( 0.5f + 4.0f * F.m_spring ) * s.x;
    if ( nb > 0 )
        acc += 4.0f * ( sumPos / static_cast<float>( nb ) - s.x ) + 3.0f * ( sumVel / static_cast<float>( nb ) - s.y );
    const float vel = fminf( fmaxf( s.y + acc * dt, -200.0f ), 200.0f );
    const float pos = fminf( fmaxf( s.x + vel * dt, -600.0f ), 600.0f );
    next[me] = make_float2( pos, vel );
    return pos;
}

/**
 * @brief The ring of the voices' levels: this block's level of every voice written where the
 *        block's index points, a voice's whole ring cleared when it starts a note (so a delayed
 *        partial of the new note never hears the old one's level).
 * @param[in]     hdr     the block header
 * @param[in]     voices  the voices
 * @param[in,out] ring    per voice GainRing levels
 */
__global__ void gainRingKernel( const BlockHeader *hdr, const CloudVoice *voices, float *ring )
{
    const int v = static_cast<int>( blockIdx.x );
    if ( v >= hdr->m_voices ) return;
    const CloudVoice &cv = voices[v];
    float            *row = ring + static_cast<size_t>( v ) * GainRing;
    if ( cv.m_restart )
        for ( int i = threadIdx.x; i < GainRing; i += blockDim.x ) row[i] = 0.0f;
    __syncthreads();
    if ( threadIdx.x == 0 ) row[hdr->m_blockIndex % GainRing] = cv.m_active ? cv.m_gain : 0.0f;
}

/**
 * @brief A voice's level `blocks` blocks ago, read from the ring and interpolated.
 * @param[in] row     the voice's ring
 * @param[in] index   this block's index
 * @param[in] blocks  how far back, in blocks (fractional)
 * @return the level; 0 beyond the ring
 */
__device__ float gainAgo( const float *row, std::uint32_t index, float blocks )
{
    if ( blocks <= 0.0f ) return row[index % GainRing];
    if ( blocks >= static_cast<float>( GainRing - 2 ) ) return 0.0f;
    const int   b0 = static_cast<int>( blocks );
    const float fr = blocks - static_cast<float>( b0 );
    const float g0 = row[( index - static_cast<std::uint32_t>( b0 ) ) % GainRing];
    const float g1 = row[( index - static_cast<std::uint32_t>( b0 + 1 ) ) % GainRing];
    return g0 + ( g1 - g0 ) * fr;
}

/**
 * @brief The voice filters transported into each other (Morph.h), one CUDA block a voice that carries
 *        two or more: each filter's power over the octaves (FilterBins bands, floored at -60 dB) read
 *        as a mass, its distribution function by a running sum in a fixed order; the Wasserstein
 *        barycentre's quantile function the weighted mean of theirs (in one dimension the barycentre
 *        is exactly that); its distribution function at the bands' edges found back in it; a band's
 *        power the mean mass times its share, its gain the root. A single filter comes out as itself.
 * @param[in]     hdr     the header
 * @param[in]     voices  the voices
 * @param[in,out] table   per voice FilterBins gains
 */
__global__ void __launch_bounds__( 256 )
    filterTransportKernel( const BlockHeader *hdr, const CloudVoice *voices, float *table )
{
    const int v = static_cast<int>( blockIdx.x );
    if ( v >= hdr->m_voices ) return;
    const CloudVoice &cv = voices[v];
    if ( !cv.m_active || cv.m_filterCount < 2 ) return;
    const int        K = min( cv.m_filterCount, VoiceFilters );
    __shared__ float cdf[VoiceFilters][FilterBins + 1];
    __shared__ float qs[FilterBins + 1];
    __shared__ float mass[VoiceFilters];
    const int        t = static_cast<int>( threadIdx.x );
    for ( int k = 0; k < K; ++k )
        for ( int b = t; b < FilterBins; b += blockDim.x )
        {
            const float f = 20.0f * exp2f( ( static_cast<float>( b ) + 0.5f ) *
                                           ( 9.96578428f / static_cast<float>( FilterBins ) ) );
            const float g = filterGain( static_cast<int>( cv.m_filterModels[k] ), f, cv.m_filterCutoffs[k],
                                        cv.m_filterQs[k], cv.m_filterVowels[k] );
            cdf[k][b + 1] = fmaxf( g * g, 1.0e-6f );
        }
    __syncthreads();
    if ( t < K )
    {   // the running sums, one filter a thread, in band order
        float s = 0.0f;
        cdf[t][0] = 0.0f;
        for ( int b = 1; b <= FilterBins; ++b )
        {
            s += cdf[t][b];
            cdf[t][b] = s;
        }
        mass[t] = s;
    }
    __syncthreads();
    for ( int k = 0; k < K; ++k )
        for ( int b = t; b <= FilterBins; b += blockDim.x ) cdf[k][b] /= mass[k];
    __syncthreads();
    // The barycentre's quantile function at every level j / FilterBins, in bands.
    for ( int j = t; j <= FilterBins; j += blockDim.x )
    {
        const float s = static_cast<float>( j ) / static_cast<float>( FilterBins );
        float       x = 0.0f;
        for ( int k = 0; k < K; ++k )
        {
            int lo = 0, hi = FilterBins;
            while ( hi - lo > 1 )
            {
                const int mid = ( lo + hi ) >> 1;
                if ( cdf[k][mid] <= s )
                    lo = mid;
                else
                    hi = mid;
            }
            const float c0 = cdf[k][lo], c1 = cdf[k][lo + 1];
            const float fr = c1 > c0 ? fminf( fmaxf( ( s - c0 ) / ( c1 - c0 ), 0.0f ), 1.0f ) : 0.0f;
            x += cv.m_filterWeight[k] * ( static_cast<float>( lo ) + fr );
        }
        qs[j] = x;
    }
    __syncthreads();
    float M = 0.0f;
    for ( int k = 0; k < K; ++k ) M += cv.m_filterWeight[k] * mass[k];
    // Its distribution function at an edge (in bands): where the quantile function crosses it.
    auto F = [&]( float edge )
    {
        if ( edge <= qs[0] ) return 0.0f;
        if ( edge >= qs[FilterBins] ) return 1.0f;
        int lo = 0, hi = FilterBins;
        while ( hi - lo > 1 )
        {
            const int mid = ( lo + hi ) >> 1;
            if ( qs[mid] <= edge )
                lo = mid;
            else
                hi = mid;
        }
        const float q0 = qs[lo], q1 = qs[lo + 1];
        const float fr = q1 > q0 ? fminf( fmaxf( ( edge - q0 ) / ( q1 - q0 ), 0.0f ), 1.0f ) : 0.0f;
        return ( static_cast<float>( lo ) + fr ) / static_cast<float>( FilterBins );
    };
    float *row = table + static_cast<size_t>( v ) * FilterBins;
    for ( int b = t; b < FilterBins; b += blockDim.x )
        row[b] = sqrtf( fmaxf( M * ( F( static_cast<float>( b + 1 ) ) - F( static_cast<float>( b ) ) ), 0.0f ) );
}

/**
 * @brief The energy cascade (Turbulence, Cloud.h), one thread a harmonic of a voice: its power p
 *        (against harmonic 1's at rest) loses drive p^2 and gains seven tenths of what the one below
 *        lost and three tenths of what its half lost, and settles back to the spectrum's own over
 *        settle / (1 + h / 8) seconds. Kept as the power against the spectrum's at rest, q = p / p0,
 *        read by the update kernel as sqrt(q) on the harmonic's amplitude. Reads one buffer, writes
 *        the other (by the block's parity); a new note starts at rest.
 * @param[in]     hdr     the header
 * @param[in]     voices  the voices
 * @param[in]     pcap    partials per voice in the tables (room for every harmonic)
 * @param[in,out] qa      per voice and harmonic its q, buffer A
 * @param[in,out] qb      buffer B
 */
__global__ void cascadeKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, float *qa, float *qb )
{
    if ( hdr->m_turb.m_on <= 0.0f ) return;
    const int gid = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    const int v = gid / pcap, k = gid - v * pcap;   // harmonic k + 1
    if ( v >= hdr->m_voices ) return;
    const CloudVoice &cv = voices[v];
    const int         H = cv.m_harmonics;
    if ( !cv.m_active || k >= H ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *src = ( odd ? qb : qa ) + static_cast<size_t>( v ) * pcap;
    float       *dst = ( odd ? qa : qb ) + static_cast<size_t>( v ) * pcap;
    // Kept as q - 1, so a table cleared to nothing is at rest.
    if ( cv.m_restart )
    {
        dst[k] = 0.0f;
        return;
    }
    const float s1 = fmaxf( harmonicShape( 1, cv.m_tilt, cv.m_oddEven, cv.m_brightness ), 1.0e-9f );
    auto        rest = [&]( int h )
    {
        const float s = harmonicShape( h, cv.m_tilt, cv.m_oddEven, cv.m_brightness ) / s1;
        return fmaxf( s * s, 1.0e-12f );
    };
    auto        power = [&]( int h ) { return ( 1.0f + src[h - 1] ) * rest( h ); };
    const int   h = k + 1;
    const float drive = cv.m_turbulence * cv.m_gain * cv.m_gain;
    const float dt = static_cast<float>( hdr->m_n ) / hdr->m_sampleRate;
    const float p0 = rest( h ), p = power( h );
    float       in = 0.0f;
    if ( h > 1 )
    {
        const float pb = power( h - 1 );
        in += 0.7f * drive * pb * pb;
    }
    if ( ( h & 1 ) == 0 )
    {
        const float ph = power( h / 2 );
        in += 0.3f * drive * ph * ph;
    }
    const float settle =
        fminf( 1.0f, dt * ( 1.0f + static_cast<float>( h ) * 0.125f ) / fmaxf( hdr->m_turb.m_settle, 0.1f ) );
    const float next = fmaxf( p + dt * ( in - drive * p * p ) + ( p0 - p ) * settle, 0.0f );
    dst[k] = fminf( next / p0, 1000.0f ) - 1.0f;
}

/**
 * @brief The update kernel: every partial's frequency glide, amplitude ramps and start phase for this block.
 * @param[in]     hdr          the block header (device copy)
 * @param[in]     voices       the voices (device copy)
 * @param[in]     pcap         partials per voice in the tables (a multiple of Chunk)
 * @param[in]     shapes       the envelope bank
 * @param[in]     land         the landscape
 * @param[in,out] run          per partial: start phase, phase step, its change per sample, the field's offset in cents
 *                             (as bits)
 * @param[in,out] amp          per partial: the near left, near right, far left and far right amplitudes the last block
 *                             ended on
 * @param[in,out] gains        per partial three entries: near (left start, left step, right start, right step), far,
 *                             and the near plane's cosine parts (binaural: the ears' phases)
 * @param[in]     ring         per voice GainRing levels (gainRingKernel), for the dispersion
 * @param[in,out] ampQ         per partial the near plane's cosine parts the last block ended on
 * @param[in,out] snap         every SnapStride-th partial's frequency and amplitude, for the picture
 * @param[in]     lifeA        the living field, buffer A
 * @param[in]     lifeB        buffer B
 * @param[in,out] theta        per partial the sync's phase
 * @param[in,out] sx           per partial its place on its circle (Swarm)
 * @param[in]     order        per voice and group the sync's mean phasors (syncOrderKernel), three planes `plane` apart
 * @param[in]     plane        the planes' distance
 * @param[in,out] flockA       the flock's state, buffer A (even blocks read it)
 * @param[in,out] flockB       buffer B
 * @param[in]     qa           the energy cascade, buffer A (cascadeKernel)
 * @param[in]     qb           buffer B
 * @param[in]     filterTable  per voice the transported filter (filterTransportKernel)
 * @param[in,out] hallHist     per band the amplitude gathered this block, for the hall's ear (hallKernel)
 * @param[in]     density      per band the spectrum's density the block before (hallKernel)
 * @param[in,out] tau          per partial its own clock (Time Bend)
 * @param[in,out] orbitA       the Orbit field's state, buffer A (even blocks read it)
 * @param[in,out] orbitB       buffer B
 */
__global__ void updateKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const EnvShape *shapes,
                              const float *land, uint4 *run, float4 *amp, float4 *gains, const float *ring,
                              float2 *ampQ, float2 *snap, const float2 *lifeA, const float2 *lifeB, float *theta,
                              float *sx, const float2 *order, size_t plane, float2 *flockA, float2 *flockB,
                              const float *qa, const float *qb, const float *filterTable, unsigned long long *hallHist,
                              const float *density, float *tau, float4 *orbitA, float4 *orbitB )
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = gid / pcap;
    const int p = gid - v * pcap;
    if ( v >= hdr->m_voices ) return;
    const CloudVoice &cv = voices[v];
    const bool        snapped = p % SnapStride == 0;
    float2 *dot = snapped ? snap + static_cast<size_t>( v ) * ( pcap / SnapStride ) + p / SnapStride : nullptr;
    if ( !cv.m_active || p >= cv.m_partials )
    {
        if ( dot ) *dot = make_float2( 0.0f, 0.0f );
        // Beyond the voice's partials (or a silent voice): the render kernel skips whole chunks, the
        // tail of the last chunk it covers must be silent.
        if ( cv.m_active )
        {
            gains[3 * gid] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            gains[3 * gid + 1] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            gains[3 * gid + 2] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            amp[gid] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            ampQ[gid] = make_float2( 0.0f, 0.0f );
        }
        return;
    }
    const float         fs = hdr->m_sampleRate;
    const int           n = hdr->m_n;
    const std::uint32_t seed = cv.m_seed;
    const std::uint32_t up = static_cast<std::uint32_t>( p );
    const float         t = cv.m_time + static_cast<float>( n ) / fs;   // every target holds at the block's end

    float      target = 0.0f;
    float      f = 0.0f;
    float      fieldOff = cv.m_restart ? 0.0f : __uint_as_float( run[gid].w );   // the field's offset, cents
    float      place = cv.m_stereo * ( 2.0f * hash01( seed, up, 10 ) - 1.0f );   // -1 .. 1 across the field
    const int  H = cv.m_harmonics, K = cv.m_members, S = cv.m_subClouds;
    const int  HP = cv.m_harmonicPartials;
    const bool isAir = p >= HP;
    if ( isAir || p < ( H + S ) * K )
    {
        // Who the partial is: a member of harmonic h's cloud, or one of the air band's partials.
        int   h = 1;
        float z = 0.0f, centre, amp0, sh, drift;
        float slot = 0.5f;   // the partial's place among its cloud's members, evenly 0 .. 1 (the living field's rows)
        int   groupFirst = 0, groupCount = 1, member = 0;   // its cloud in the voice, for the flock
        bool  isSub = false;                                // a member of a sub cloud (Subharmonics)
        int   sub = 0;                                      // which
        int   group = H;                                    // its cloud's row in the sync's table (air: H)
        float grainEnv = 1.0f;                              // its grain's window (Grains), 1 when continuous
        float sigmaCents = 0.0f;                            // the cloud's width, cents per unit of the bell
        // Time Bend: the partial's own clock (its drift, its shimmer, its sync), slower -- or faster --
        // where the spectrum is dense, read at its harmonic from the density of the block before. At
        // nothing it is the voice's clock, kept in step, so turning it up never jumps.
        float tp = t, clockRate = 1.0f;
        if ( hdr->m_field.m_timeBend != 0.0f )
        {
            const int   cl = p / max( K, 1 );   // its cloud: a harmonic, or after them a sub cloud
            const float fh = isAir    ? cv.m_hz * cv.m_airCenter
                             : cl < H ? cv.m_hz * partialRatio( cl + 1, cv.m_inharmonic, cv.m_stretch, cv.m_horizon,
                                                                cv.m_horizonRatio )
                                      : cv.m_hz * subRatio( cl - H );
            const float dense = fminf( fmaxf( log2f( 1.0f + density[hallBin( fh )] ) - 1.0f, 0.0f ), 4.0f );
            const float bend = hdr->m_field.m_timeBend;
            clockRate = bend > 0.0f ? 1.0f / ( 1.0f + bend * dense ) : 1.0f - bend * dense;
            tp = ( cv.m_restart ? 0.0f : tau[gid] ) + clockRate * static_cast<float>( n ) / fs;
        }
        tau[gid] = tp;
        if ( !isAir )
        {
            const int cl = p / K;   // its cloud: harmonics 0 .. H - 1, then the sub clouds
            isSub = cl >= H;
            sub = isSub ? cl - H : 0;
            h = isSub ? 1 : cl + 1;
            group = isSub ? H + 1 + sub : h - 1;
            const int k = p - cl * K;
            slot = ( static_cast<float>( k ) + 0.5f ) / static_cast<float>( K );
            // Grains (Phase 26): a share of the partials sounds in grains -- a Hann window of its own length,
            // a share `duty` of its time on, at a phase of its own -- and every grain is born at a new place
            // in its cloud and in the field; it jumps there while it is silent. Its power over time is the
            // continuous partial's (grainNorm).
            std::uint32_t jitterSeed = seed;
            if ( cv.m_grain > 0.0f && hash01( seed, up, 62 ) < cv.m_grain )
            {
                const float len = cv.m_grainLength * ( 0.6f + 0.8f * hash01( seed, up, 60 ) );
                const float period = len / fminf( fmaxf( cv.m_grainDuty, 0.01f ), 1.0f );
                const float tg = tp + period * hash01( seed, up, 61 );
                const float idx = floorf( tg / period );
                const float u = ( tg - idx * period ) / len;
                const float w = u < 1.0f ? __sinf( 3.1415927f * u ) : 0.0f;
                grainEnv = w * w * cv.m_grainNorm;
                jitterSeed = hash32( seed ^ ( static_cast<std::uint32_t>( idx ) * 0x9E3779B9U ) );
                place = cv.m_stereo * ( 2.0f * hash01( jitterSeed, up, 10 ) - 1.0f );
            }
            // The member's place in its harmonic's cloud: stratified over the bell curve, so K members
            // always cover it evenly and only jitter within their own slice (a grain anywhere in it).
            if ( K > 1 )
                z = jitterSeed != seed
                        ? normcdfinvf( fminf( fmaxf( hash01( jitterSeed, up, 1 ), 1.0e-4f ), 1.0f - 1.0e-4f ) )
                        : normcdfinvf( ( static_cast<float>( k ) + 0.5f + 0.9f * ( hash01( seed, up, 1 ) - 0.5f ) ) /
                                       static_cast<float>( K ) );
            const float hr = isSub ? subRatio( sub ) : static_cast<float>( h );
            const float sigma = 0.5f * cv.m_width * __powf( hr, cv.m_widthSlope );
            groupFirst = cl * K;
            groupCount = K;
            member = k;
            sigmaCents = sigma;
            centre =
                cv.m_hz *
                    ( isSub ? hr : partialRatio( h, cv.m_inharmonic, cv.m_stretch, cv.m_horizon, cv.m_horizonRatio ) ) *
                    exp2f( z * sigma * ( 1.0f / 1200.0f ) ) +
                cv.m_shift;
            drift =
                cv.m_drift *
                ( 0.65f * slowSine( cv.m_driftRate * ( 0.5f + hash01( seed, up, 2 ) ), tp, hash01( seed, up, 3 ) ) +
                  0.35f * slowSine( cv.m_driftRate * ( 1.2f + hash01( seed, up, 4 ) ), tp, hash01( seed, up, 5 ) ) );
            sh = fmaxf( 0.0f,
                        1.0f + cv.m_shimmer * ( 0.7f * slowSine( cv.m_shimmerRate * ( 0.5f + hash01( seed, up, 6 ) ),
                                                                 tp, hash01( seed, up, 7 ) ) +
                                                0.3f * slowSine( cv.m_shimmerRate * ( 1.3f + hash01( seed, up, 8 ) ),
                                                                 tp, hash01( seed, up, 9 ) ) ) );
            amp0 = cv.m_scale * harmonicShape( h, cv.m_tilt, cv.m_oddEven, cv.m_brightness );
            if ( isSub )
            {   // Subharmonics: the engine's amplitude for the cloud, against harmonic 1's
                amp0 *= cv.m_sub[sub];
            }
            else
            {
                if ( hdr->m_turb.m_on > 0.0f && !cv.m_restart )
                {   // the cascade's power against the spectrum's (this block's, cascadeKernel)
                    const float *q = ( ( hdr->m_blockIndex & 1U ) ? qa : qb ) + static_cast<size_t>( v ) * pcap;
                    amp0 *= sqrtf( fmaxf( 1.0f + q[h - 1], 0.0f ) );
                }
                // The harmonics' own lives (Phase 26, Engine::stepHarmonics): the registers that sound, the
                // family whose season it is, the avalanche's glow; the make-up keeps the voice's power.
                if ( cv.m_harmonicLife != 0 )
                {
                    float g = cv.m_lifeMakeup;
                    if ( cv.m_harmonicLife & 1 ) g *= registerGain( h, cv.m_register );
                    if ( ( cv.m_harmonicLife & 2 ) && h > 1 ) g *= cv.m_season[seasonFamily( h )];
                    if ( ( cv.m_harmonicLife & 4 ) && h >= 2 && h < 2 + AvalancheSites ) g *= 1.0f + cv.m_glow[h - 2];
                    amp0 *= g;
                }
                // The power the harmonics 1, 3, 5 and 7 handed down to their subharmonics.
                if ( S > 0 && h <= 7 && ( h & 1 ) ) amp0 *= cv.m_pumpKeep[h >> 1];
            }
        }
        else
        {
            // Air (Noctuary's band-passed noise around a multiple of the fundamental): a wide cloud
            // whose partials flicker in pitch and level at audio-near rates -- dense random sines in
            // a band are heard as its noise.
            const int A = cv.m_partials - HP;
            const int a = p - HP;
            slot = ( static_cast<float>( a ) + 0.5f ) / static_cast<float>( max( A, 1 ) );
            if ( A > 1 )
                z = normcdfinvf( ( static_cast<float>( a ) + 0.5f + 0.9f * ( hash01( seed, up, 1 ) - 0.5f ) ) /
                                 static_cast<float>( A ) );
            groupFirst = HP;
            groupCount = A;
            member = a;
            sigmaCents = 0.5f * cv.m_airWidth;
            centre = cv.m_hz * cv.m_airCenter * exp2f( z * 0.5f * cv.m_airWidth * ( 1.0f / 1200.0f ) ) + cv.m_shift;
            const float j = cv.m_airJitter;
            drift = 40.0f * ( 0.6f * slowSine( j * ( 0.5f + hash01( seed, up, 2 ) ), tp, hash01( seed, up, 3 ) ) +
                              0.4f * slowSine( j * ( 1.3f + hash01( seed, up, 4 ) ), tp, hash01( seed, up, 5 ) ) );
            sh = fmaxf(
                0.0f,
                1.0f + 0.8f * ( 0.7f * slowSine( j * ( 0.7f + hash01( seed, up, 6 ) ), tp, hash01( seed, up, 7 ) ) +
                                0.3f * slowSine( j * ( 1.6f + hash01( seed, up, 8 ) ), tp, hash01( seed, up, 9 ) ) ) );
            amp0 = cv.m_airScale;
        }

        // The envelope lanes, read against this partial's sources.
        LaneContext c;
        centre = fmaxf( centre, 1.0f );   // a partial the shift took below nothing (it is faded out)
        const float pitchPos =
            log2f( fmaxf( centre, LandLowHz ) * ( 1.0f / LandLowHz ) ) * ( 1.0f / 9.96578428f );   // log2(1000)
        c.m_src[0] = 0.0f;
        c.m_src[1] = fminf( fmaxf( pitchPos, 0.0f ), 1.0f );
        c.m_src[2] = isAir   ? 0.5f
                     : isSub ? 0.0f
                             : ( H > 1 ? static_cast<float>( h - 1 ) / static_cast<float>( H - 1 ) : 0.0f );
        c.m_src[3] = fminf( fmaxf( ( z + 3.0f ) * ( 1.0f / 6.0f ), 0.0f ), 1.0f );
        c.m_src[4] = 0.0f;   // Random: laneValue throws its own die per parameter
        c.m_src[5] = ( isAir || isSub || ( h & 1 ) ) ? 0.0f : 1.0f;
        c.m_seed = seed;
        c.m_p = up;
        // The dispersion: this partial's clock late by its share of the dispersion, by its pitch.
        const float D = hdr->m_field.m_dispersion;
        const float lateBy = D >= 0.0f ? D * ( 1.0f - c.m_src[1] ) : -D * c.m_src[1];
        c.m_t = t - lateBy;
        c.m_releaseAt = cv.m_releaseAt;
        c.m_shapes = shapes;
        c.m_shapeCount = hdr->m_shapeCount;
        c.m_land = land;
        c.m_landColumns = hdr->m_landColumns;
        c.m_landRows = hdr->m_landRows;
        c.m_landRowSeconds = fmaxf( hdr->m_landRowSeconds, 1.0e-4f );
        c.m_landColumn = c.m_src[1] * static_cast<float>( max( hdr->m_landColumns - 1, 0 ) );
        float depth = 0.0f;
        float env = 1.0f;
        if ( cv.m_amp.m_amount != 0.0f )
        {
            const float e = laneValue( cv.m_amp, c, 30, false, depth );
            depth = fminf( fmaxf( depth, 0.0f ), 1.0f );
            env = 1.0f - depth + depth * e;
        }
        // Partial sync: the phase turns at its own rate and is pulled to its cloud (Global), to its
        // neighbours round the ring (Chimera), or with its place on its circle (Swarm).
        float swarmCents = 0.0f, swarmSide = 0.0f;
        if ( hdr->m_sync.m_depth > 0.0f )
        {
            const PartialSync &S = hdr->m_sync;
            const float        dtb = static_cast<float>( n ) / fs;
            const float        ringX =
                6.2831853f * ( static_cast<float>( member ) + 0.5f ) / static_cast<float>( max( groupCount, 1 ) );
            float th;
            if ( !cv.m_restart )
            {
                th = theta[gid];
            }
            else if ( S.m_mode == SyncModeE::CHIMERA )
            {
                // Abrams and Strogatz's start: ordered round the middle of the ring, scattered at its ends.
                const float xc = ringX - 3.1415927f;
                th = 6.0f * ( hash01( seed, up, 13 ) - 0.5f ) * __expf( -0.76f * xc * xc );
            }
            else
            {
                th = 6.2831853f * hash01( seed, up, 13 );
            }
            const float  u = 2.0f * hash01( seed, up, 12 ) - 1.0f;
            const size_t at = static_cast<size_t>( v ) * pcap + group;
            float        s, co;
            __sincosf( th, &s, &co );
            if ( S.m_mode == SyncModeE::SWARM )
            {
                // In the frame turning at the rate: what is left of each rate, and the place's own drift.
                float x = cv.m_restart ? 6.2831853f * hash01( seed, up, 14 ) : sx[gid];
                float dth = 6.2831853f * S.m_rate * S.m_spread * u,
                      dx = 3.1415927f * S.m_rate * S.m_spread * ( 2.0f * hash01( seed, up, 15 ) - 1.0f );
                if ( !cv.m_restart )
                {
                    const float2 wp = order[at + plane], wm = order[at + 2 * plane];
                    float        sp, cp, sm, cm;
                    __sincosf( x + th, &sp, &cp );
                    __sincosf( x - th, &sm, &cm );
                    const float ap = wp.y * cp - wp.x * sp;   // S+ sin(Φ+ - (x + θ))
                    const float am = wm.y * cm - wm.x * sm;   // S- sin(Φ- - (x - θ))
                    dx += 0.5f * S.m_swarm * ( ap + am );
                    dth += 0.5f * S.m_coupling * ( ap - am );
                }
                x += dx * dtb * clockRate;
                x -= 6.2831853f * floorf( x * 0.15915494f );
                sx[gid] = x;
                th += dth * dtb * clockRate;
                th -= 6.2831853f * floorf( th * 0.15915494f );
                theta[gid] = th;
                float sxs, sxc;
                __sincosf( x, &sxs, &sxc );
                swarmCents = S.m_reach * sxc;
                swarmSide = 0.5f * fminf( S.m_reach * ( 1.0f / 30.0f ), 1.0f ) * sxs;
                // The heard phase turns on at the rate.
                const float heard = th + 6.2831853f * S.m_rate * tp;
                env *=
                    fmaxf( 0.0f, 1.0f + S.m_depth * __sinf( heard ) ) * rsqrtf( 1.0f + 0.5f * S.m_depth * S.m_depth );
            }
            else
            {
                const float w = 6.2831853f * S.m_rate * ( 1.0f + S.m_spread * u );
                float       pull = 0.0f;
                if ( !cv.m_restart )
                {
                    const float2 z = order[at];
                    if ( S.m_mode == SyncModeE::CHIMERA )
                    {
                        // Σ G(x_i - x_j) e^(iθ_j) / N with G = 1 + A cos: the mean, and the two harmonics of the ring.
                        const float2 zm = order[at + plane], zp = order[at + 2 * plane];
                        float        rs, rc;
                        __sincosf( ringX, &rs, &rc );
                        const float A = 0.4975f;   // A / 2, A = 0.995
                        const float cx = z.x + A * ( ( rc * zm.x - rs * zm.y ) + ( rc * zp.x + rs * zp.y ) );
                        const float cy = z.y + A * ( ( rc * zm.y + rs * zm.x ) + ( rc * zp.y - rs * zp.x ) );
                        float       ls, lc;
                        __sincosf( th + S.m_lag, &ls, &lc );
                        pull = S.m_coupling * ( cy * lc - cx * ls );   // K Im(C e^(-i(θ + α)))
                    }
                    else
                    {
                        pull = S.m_coupling * ( z.y * co - z.x * s );   // K R sin(ψ - θ)
                    }
                }
                th += ( w + pull ) * dtb * clockRate;
                th -= 6.2831853f * floorf( th * 0.15915494f );
                theta[gid] = th;
                env *= fmaxf( 0.0f, 1.0f + S.m_depth * __sinf( th ) ) * rsqrtf( 1.0f + 0.5f * S.m_depth * S.m_depth );
            }
        }
        // The ripple: the wave along the spectrum at this partial's frequency.
        if ( hdr->m_ripple.m_depth > 0.0f )
        {
            const float x = c.m_src[1] * static_cast<float>( RippleCells - 1 );
            const int   x0 = min( static_cast<int>( x ), RippleCells - 2 );
            float       u = hdr->m_ripple.m_u[x0] +
                            ( hdr->m_ripple.m_u[x0 + 1] - hdr->m_ripple.m_u[x0] ) * ( x - static_cast<float>( x0 ) );
            u += hdr->m_ripple.m_periodic * ( 2.0f * __sinf( 0.5f * u ) - u );   // Breathers: periodic in the angle
            env *= fmaxf( 0.0f, 1.0f + hdr->m_ripple.m_depth * tanhf( u ) ) * hdr->m_ripple.m_makeup;
        }
        // The living landscape: the field at this partial's frequency and its place in its cloud.
        if ( hdr->m_life.m_mode != LifeModeE::OFF && hdr->m_life.m_depth > 0.0f )
        {
            const LifeField &L = hdr->m_life;
            // Along: the member's slot, evenly over the rows -- what the field's measure assumes (the
            // bell-shaped member position read the middle rows only: 2.6 dB off while a Turing field grew).
            const float v =
                lifeAt( hdr, lifeCurrent( hdr, lifeA, lifeB ), c.m_src[1] * L.m_zoom * static_cast<float>( LifeW ),
                        slot * static_cast<float>( LifeH ) );
            env *= 1.0f - L.m_depth + L.m_depth * v * L.m_makeup;
        }
        float cents = drift + swarmCents;
        if ( swarmSide != 0.0f ) place = fminf( fmaxf( place + swarmSide, -1.0f ), 1.0f );
        if ( cv.m_pitch.m_amount != 0.0f ) cents += laneValue( cv.m_pitch, c, 40, true, depth ) * depth;
        if ( cv.m_pan.m_amount != 0.0f )
            place = fminf( fmaxf( place + laneValue( cv.m_pan, c, 50, true, depth ) * depth, -1.0f ), 1.0f );

        // The field: the partial's offset moves towards what pulls it, from where it was.
        const float home = centre * exp2f( cents * ( 1.0f / 1200.0f ) );
        if ( hdr->m_field.m_mode == FieldModeE::FLOCK )
        {
            // The flock: every cloud a flock of its own, blown by the wind of its own attractor.
            const bool  odd = ( hdr->m_blockIndex & 1U ) != 0;
            const float wind =
                hdr->m_field.m_pos[( isAir ? 0 : group ) & ( FieldAttractors - 1 )] * 0.2f * hdr->m_field.m_strength;
            fieldOff = flockStep( hdr->m_field, odd ? flockB : flockA, odd ? flockA : flockB,
                                  static_cast<size_t>( v ) * pcap, groupFirst, groupCount, member, seed, sigmaCents,
                                  wind, fieldOff, cv.m_restart != 0, static_cast<float>( n ) / fs );
        }
        else if ( hdr->m_field.m_mode == FieldModeE::ORBIT )
        {
            // The orbit: drawn to the scale's pitches, bent by the field into circles of pitch and side.
            const bool odd = ( hdr->m_blockIndex & 1U ) != 0;
            float      side = 0.0f;
            fieldOff =
                orbitStep( hdr->m_field, odd ? orbitB : orbitA, odd ? orbitA : orbitB, static_cast<size_t>( gid ),
                           log2f( fmaxf( home, 1.0f ) ), z * ( 2.0f / 1200.0f ), fieldOff, seed, up, hdr->m_blockIndex,
                           cv.m_restart != 0, static_cast<float>( n ) / fs, side );
            place = fminf( fmaxf( place + side, -1.0f ), 1.0f );
        }
        else
        {
            fieldOff = fieldStep( hdr->m_field, log2f( fmaxf( home, 1.0f ) ) + fieldOff * ( 1.0f / 1200.0f ), fieldOff,
                                  static_cast<float>( n ) / fs, z * ( 2.0f / 1200.0f ) );
        }
        f = home * exp2f( fieldOff * ( 1.0f / 1200.0f ) );
        // The top fades out between 0.43 and 0.47 of the sample rate: nothing reaches Nyquist; a
        // partial the shift took down fades out between 20 and 5 Hz (and never runs below 1).
        const float fade = fminf( fmaxf( ( 0.47f * fs - f ) / ( 0.04f * fs ), 0.0f ), 1.0f ) *
                           fminf( fmaxf( ( f - 5.0f ) * ( 1.0f / 15.0f ), 0.0f ), 1.0f );
        f = fmaxf( f, 1.0f );
        // Distance (Noctuary's planes): 6 dB per unit, and the air's low pass -- a first-order
        // corner falling from 16 kHz by up to five octaves per unit -- read at this partial's
        // frequency, where in a filter it would have been a whole filter. Taken relative to the
        // corner at the ear, so a partial at distance 0 keeps exactly its level (as in Noctuary).
        const float dist = fminf( fmaxf( cv.m_distance, 0.0f ), 1.0f );
        const float x0 = f * ( 1.0f / 16000.0f );
        const float x = x0 / exp2f( -5.0f * dist * cv.m_airAbsorb );
        const float absorb = sqrtf( ( 1.0f + x0 * x0 ) / ( 1.0f + x * x ) );
        // The voice's colour and filter, each read at this partial's frequency: the filter as it is, or
        // during a morph between different ones their transport (filterTransportKernel's table).
        float colour;
        if ( cv.m_filterCount >= 2 )
        {
            const float *row = filterTable + static_cast<size_t>( v ) * FilterBins;
            const float  pos =
                fminf( fmaxf( log2f( f * ( 1.0f / 20.0f ) ) * ( static_cast<float>( FilterBins ) / 9.96578428f ) - 0.5f,
                              0.0f ),
                       static_cast<float>( FilterBins - 1 ) - 1.0e-3f );
            const int i0 = static_cast<int>( pos );
            colour = row[i0] + ( row[min( i0 + 1, FilterBins - 1 )] - row[i0] ) * ( pos - static_cast<float>( i0 ) );
        }
        else
        {
            colour = filterGain( cv.m_filterModel, f, cv.m_cutoff, cv.m_filterQ, cv.m_vowel );
        }
        if ( hdr->m_comb.m_depth > 0.0f )
        {   // the spectral comb: a raised cosine per notch, on the log axis
            const SpectralComb &K = hdr->m_comb;
            const float         x = log2f( f ) * K.m_density + K.m_phase + 0.25f * K.m_stereo * place;
            colour *= ( 1.0f - K.m_depth * ( 0.5f - 0.5f * __cosf( 6.2831853f * x ) ) ) * K.m_makeup;
        }
        if ( !isAir ) colour *= presenceGain( f, cv.m_presenceDb, dist ) * lowCutGain( f, cv.m_lowCut );
        // The voice's level, as late as this partial's clock (the dispersion), from the ring.
        const float level = lateBy > 0.0f ? gainAgo( ring + static_cast<size_t>( v ) * GainRing, hdr->m_blockIndex,
                                                     lateBy * fs / static_cast<float>( n ) )
                                          : cv.m_gain;
        target = amp0 * sh * fade * env * grainEnv * level * exp2f( -dist ) * absorb * colour;
        // The hall's ear: this partial's amplitude in its band, in whole numbers (the sum is the same in
        // any order). Amplitudes, not powers: a chord's upper harmonics count as heard too, not only the
        // few loudest bands.
        if ( hdr->m_modal.m_listen && target > 0.0f )
            atomicAdd( hallHist + hallBin( f ), static_cast<unsigned long long>( target * 1.0e12f ) );
    }
    const std::uint32_t incTarget = static_cast<std::uint32_t>( fminf( f / fs, 0.49f ) * 4294967296.0f );

    // Phase and glide: the step moves linearly from where the last block left it to this block's
    // target, so the phase is quadratic in the sample and pitch envelopes never step.
    std::uint32_t ph, inc0, dinc = 0;
    float4        a0;
    float2        q0 = make_float2( 0.0f, 0.0f );
    int           onset = 0;
    if ( cv.m_restart )
    {
        // A random phase. (Never aligned: a harmonic's hundreds of members lie within cents of each
        // other, so in phase they add in amplitude -- measured 19 dB louder for half a second, and
        // a plain sine rather than a pulse. The cloud's level assumes they add in power.)
        ph = hash32( seed ^ hash32( up + 0x51ED270BU ) );
        inc0 = incTarget;
        a0 = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        onset = min( max( cv.m_onset, 0 ), n - 1 );   // the ramp starts where the note does
    }
    else
    {
        const uint4         r = run[gid];
        const std::uint32_t np = static_cast<std::uint32_t>( hdr->m_nPrev );
        ph = r.x + np * r.y + r.z * ( np * ( np - 1u ) / 2u );
        inc0 = r.y + r.z * np;
        const long long diff = static_cast<long long>( incTarget ) - static_cast<long long>( inc0 );
        if ( diff > -n && diff < n )
            inc0 = incTarget;   // close enough: land exactly (under a millihertz)
        else
            dinc = static_cast<std::uint32_t>( static_cast<std::int32_t>( diff / n ) );
        a0 = amp[gid];
        q0 = ampQ[gid];
    }

    // Equal-power pan: the partial's place in the field; equal-power planes: near cos, far sin.
    float gl, gr, gn, gf;
    __sincosf( 0.78539816f * ( 1.0f + place ), &gr, &gl );
    __sincosf( 1.57079633f * fminf( fmaxf( cv.m_distance, 0.0f ), 1.0f ), &gf, &gn );
    float4 t4 = make_float4( target * gl * gn, target * gr * gn, target * gl * gf, target * gr * gf );
    float2 tq = make_float2( 0.0f, 0.0f );
    if ( hdr->m_binaural.m_on && target != 0.0f )
    {
        // Binaural: the near plane's two ears as complex gains, a sine part and a cosine part each.
        float el[2], er[2];
        earGains( hdr->m_binaural, place, hash01( seed, up, 11 ), f, el, er );
        t4.x = target * gn * el[0];
        t4.y = target * gn * er[0];
        tq = make_float2( target * gn * el[1], target * gn * er[1] );
    }
    const float inv = 1.0f / static_cast<float>( n - onset );
    gains[3 * gid] = make_float4( a0.x, ( t4.x - a0.x ) * inv, a0.y, ( t4.y - a0.y ) * inv );
    gains[3 * gid + 1] = make_float4( a0.z, ( t4.z - a0.z ) * inv, a0.w, ( t4.w - a0.w ) * inv );
    gains[3 * gid + 2] = make_float4( q0.x, ( tq.x - q0.x ) * inv, q0.y, ( tq.y - q0.y ) * inv );
    amp[gid] = t4;
    ampQ[gid] = tq;
    if ( dot ) *dot = make_float2( f, target );
    run[gid] = make_uint4( ph, inc0, dinc, __float_as_uint( fieldOff ) );
}

/**
 * @brief The render kernel: one chunk of partials summed into its own four-channel block of scratch.
 * @param[in]     hdr      the block header
 * @param[in]     voices   the voices
 * @param[in]     pcap     partials per voice in the tables
 * @param[in]     run      per partial: start phase, phase step, its change per sample
 * @param[in]     gains    per partial three entries: near (left start, left step, right start, right step), far, near
 *                         cosine parts
 * @param[in,out] scratch  per chunk: maxN samples of near left, near right, far left, far right
 * @param[in]     maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__( Threads )
    renderKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const uint4 *run, const float4 *gains,
                  float *scratch, int maxN )
{
    __shared__ float4 sgn[Chunk];                    // the chunk's near amplitude ramps
    __shared__ float4 sgf[Chunk];                    // and its far ones
    __shared__ float4 sgq[Chunk];                    // binaural: the near plane's cosine parts
    __shared__ uint4  sp[Chunk];                     // its start phases, steps and glides
    __shared__ float  red[CloudChannels][Threads];   // the groups' sums when the block is shorter than Threads

    const int chunk = blockIdx.x;
    const int chunksPerVoice = pcap / Chunk;
    const int v = chunk / chunksPerVoice;
    const int first = ( chunk - v * chunksPerVoice ) * Chunk;
    const int n = hdr->m_n;
    float    *out = scratch + static_cast<size_t>( chunk ) * CloudChannels * maxN;

    const CloudVoice &cv = voices[v];
    if ( v >= hdr->m_voices || !cv.m_active || first >= cv.m_partials )
    {
        for ( int s = threadIdx.x; s < n; s += Threads )
            for ( int c = 0; c < CloudChannels; ++c ) out[c * maxN + s] = 0.0f;
        return;
    }
    const int  count = min( Chunk, cv.m_partials - first );
    const int  base = v * pcap + first;
    const bool binaural = hdr->m_binaural.m_on != 0;
    for ( int j = threadIdx.x; j < count; j += Threads )
    {
        sgn[j] = gains[3 * ( base + j )];
        sgf[j] = gains[3 * ( base + j ) + 1];
        if ( binaural ) sgq[j] = gains[3 * ( base + j ) + 2];
        sp[j] = run[base + j];
    }
    // A note that begins in this block begins at its onset: before it every ramp holds at its start.
    const int onset = cv.m_restart ? min( max( cv.m_onset, 0 ), n - 1 ) : 0;
    __syncthreads();

    // One thread per sample; a block shorter than Threads splits the partials among groups.
    const int lanes = min( n, Threads );
    const int groups = Threads / lanes;
    const int lane = threadIdx.x % lanes;
    const int g = threadIdx.x / lanes;
    for ( int s0 = 0; s0 < n; s0 += lanes )
    {
        const int           s = s0 + lane;
        const std::uint32_t su = static_cast<std::uint32_t>( s );
        const std::uint32_t tri = su * ( su - 1u ) / 2u;   // the glide's share: sum of 0 .. s-1
        const float         sa = static_cast<float>( max( s - onset, 0 ) );
        float               nl = 0.0f, nr = 0.0f, fl = 0.0f, fr = 0.0f;
        if ( !binaural )
        {
            for ( int j = g; j < count; j += groups )
            {
                const uint4  pi = sp[j];
                const float4 a = sgn[j];
                const float4 b = sgf[j];
                // The phase as a signed fraction of a turn: -pi .. pi, where __sinf is most accurate.
                const std::uint32_t ph = pi.x + su * pi.y + tri * pi.z;
                const float x = __sinf( static_cast<float>( static_cast<std::int32_t>( ph ) ) * 1.46291808e-9f );
                nl = fmaf( fmaf( a.y, sa, a.x ), x, nl );
                nr = fmaf( fmaf( a.w, sa, a.z ), x, nr );
                fl = fmaf( fmaf( b.y, sa, b.x ), x, fl );
                fr = fmaf( fmaf( b.w, sa, b.z ), x, fr );
            }
        }
        else
        {
            // Binaural: each ear hears the partial at its own phase -- a sine part and a cosine part.
            for ( int j = g; j < count; j += groups )
            {
                const uint4         pi = sp[j];
                const float4        a = sgn[j];
                const float4        b = sgf[j];
                const float4        q = sgq[j];
                const std::uint32_t ph = pi.x + su * pi.y + tri * pi.z;
                float               x, y;
                __sincosf( static_cast<float>( static_cast<std::int32_t>( ph ) ) * 1.46291808e-9f, &x, &y );
                nl = fmaf( fmaf( a.y, sa, a.x ), x, fmaf( fmaf( q.y, sa, q.x ), y, nl ) );
                nr = fmaf( fmaf( a.w, sa, a.z ), x, fmaf( fmaf( q.w, sa, q.z ), y, nr ) );
                fl = fmaf( fmaf( b.y, sa, b.x ), x, fl );
                fr = fmaf( fmaf( b.w, sa, b.z ), x, fr );
            }
        }
        if ( groups > 1 )
        {
            red[0][threadIdx.x] = nl;
            red[1][threadIdx.x] = nr;
            red[2][threadIdx.x] = fl;
            red[3][threadIdx.x] = fr;
            __syncthreads();
            if ( g == 0 )
                for ( int q = 1; q < groups; ++q )
                {
                    nl += red[0][lane + q * lanes];
                    nr += red[1][lane + q * lanes];
                    fl += red[2][lane + q * lanes];
                    fr += red[3][lane + q * lanes];
                }
            __syncthreads();
        }
        if ( g == 0 )
        {
            out[s] = nl;
            out[maxN + s] = nr;
            out[2 * maxN + s] = fl;
            out[3 * maxN + s] = fr;
        }
    }
}

/**
 * @brief The reduce kernel: the chunks' sums added sample by sample into the first CloudChannels
 *        (near L, near R, far L, far R) of interleaved frames of CloudOutChannels.
 * @param[in]     hdr      the block header
 * @param[in]     scratch  the render kernel's per-chunk blocks
 * @param[in]     chunks   how many chunks
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels
 */
__global__ void reduceKernel( const BlockHeader *hdr, const float *scratch, int chunks, int maxN, float *out )
{
    __shared__ float part[ReduceGroups][ReduceLanes + 1];
    const int        n = hdr->m_n;
    const int        t = blockIdx.x * ReduceLanes + threadIdx.x;   // channel-major, as the scratch is laid out
    const int        c = t / maxN;
    const int        s = t - c * maxN;
    const bool       live = c < CloudChannels && s < n;
    float            acc = 0.0f;
    if ( live )
        for ( int k = threadIdx.y; k < chunks; k += ReduceGroups )
            acc += scratch[static_cast<size_t>( k ) * CloudChannels * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if ( threadIdx.y == 0 && live )
    {
        for ( int q = 1; q < ReduceGroups; ++q ) acc += part[q][threadIdx.x];
        out[s * CloudOutChannels + c] = acc;
    }
}

/**
 * @brief A uniform number in [0, 1) for one mode of the modal hall.
 * @param[in] mode  the mode
 * @param[in] salt  which of its numbers
 * @return the number
 */
__device__ __forceinline__ float modalUni( int mode, std::uint32_t salt )
{
    return static_cast<float>( hash32( static_cast<std::uint32_t>( mode ) * 0x9E3779B9U ^ ( salt * 0x85EBCA6BU ) ) >>
                               8 ) *
           ( 1.0f / 16777216.0f );
}

/**
 * @brief The modal hall: one thread per mode, a complex one-pole driven by the block the voices
 *        just made, summed per sample over the CUDA block (warp shuffles, then the warps in order).
 *
 * Each mode works out where it stands and how fast it dies from the header, every block -- a
 * hash gives it its place, its pitch if it is tuned, its drift and its two output phases -- so a
 * change of key, decay or tuning reaches it at the next block, and a ringing mode retunes without
 * a click (its state is kept; only the rotation changes). The input is the near and the far plane
 * together: the hall is a space both stand in.
 * @param[in]     hdr      the block header (its `modal` settings)
 * @param[in]     in       the block's frames, CloudOutChannels each; the four voice channels are read
 * @param[in,out] state    per mode its complex state, carried from block to block
 * @param[in,out] scratch  per CUDA block two channels of maxN sums
 * @param[in]     maxN     the scratch stride
 * @param[in]     land     the landscape (the painted hall reads it)
 * @param[in]     memory   per band the hall's memory, 1 the average (Grow reads it)
 */
__global__ void __launch_bounds__( ModalThreads )
    modalKernel( const BlockHeader *hdr, const float *in, float2 *state, float *scratch, int maxN, const float *land,
                 const float *memory )
{
    __shared__ float sIn[2][ModalTile];
    __shared__ float sAcc[ModalThreads / 32][2][ModalTile];
    const ModalHall &mh = hdr->m_modal;
    const int        n = hdr->m_n;
    const int        mode = static_cast<int>( blockIdx.x ) * ModalThreads + static_cast<int>( threadIdx.x );
    float           *out = scratch + static_cast<size_t>( blockIdx.x ) * 2 * maxN;
    if ( !mh.m_on )
    {   // resting: silent, and cleared so that it starts from silence when it is opened again
        state[mode] = make_float2( 0.0f, 0.0f );
        for ( int s = threadIdx.x; s < n; s += ModalThreads )
        {
            out[s] = 0.0f;
            out[maxN + s] = 0.0f;
        }
        return;
    }
    const float fs = hdr->m_sampleRate;
    const float lo = fmaxf( mh.m_low, 20.0f );
    const float hi = fminf( 16000.0f, 0.45f * fs );
    // Where it stands: on one of the scale's pitches (the share `tune` of the modes), or anywhere
    // between the lowest mode and 16 kHz, evenly in hertz (so the hall is as dense at the top as
    // at the bottom, where a decaying mode is as wide).
    float f = lo + ( hi - lo ) * modalUni( mode, 4 );
    float weight = 1.0f;
    if ( mh.m_tunedCount > 0 && modalUni( mode, 1 ) < mh.m_tune )
    {
        const int   k = static_cast<int>( hash32( static_cast<std::uint32_t>( mode ) ^ 0xA511E9B3U ) %
                                          static_cast<std::uint32_t>( mh.m_tunedCount ) );
        const float p = mh.m_tuned[k];
        f = p * exp2f( mh.m_spread * ( modalUni( mode, 2 ) + modalUni( mode, 3 ) - 1.0f ) * ( 1.0f / 1200.0f ) );
        // Tuned modes stand closer together than spread ones -- all of them within a few cents of a
        // few dozen pitches -- and a sound on those pitches would wake them all: weighted down by
        // that density (the pitches' band against the whole range), the tuned share colours the
        // hall instead of making it louder.
        const float cents = fmaxf( mh.m_spread + mh.m_drift, 1.0f );
        const float bandHz =
            static_cast<float>( mh.m_tunedCount ) * p * cents * ( 2.0f * 0.000577623f );   // ln 2 / 1200
        weight = sqrtf( fminf( 4.0f, bandHz / ( hi - lo ) ) );
    }
    // The drift: a slow sine of its own; every rate a whole number of cycles in 1000 s, the clock's wrap.
    const float rate =
        ( 20.0f + static_cast<float>( hash32( static_cast<std::uint32_t>( mode ) ^ 0x5BD1E995U ) % 100U ) ) * 0.001f;
    float ph = rate * mh.m_time + modalUni( mode, 6 );
    ph -= floorf( ph );
    f *= exp2f( mh.m_drift * ( 1.0f / 1200.0f ) * __sinf( 6.2831853f * ph ) );
    const bool silent = f < 10.0f || f >= 0.49f * fs;
    // How fast it dies: the decay at and below 250 Hz, each octave above 2^damp times faster; with
    // breath, every mode's decay wanders by up to an octave on a slow sine of its own (again a whole
    // number of cycles in 1000 s), so the hall's colour moves while it rings.
    float breathe = 0.0f;
    if ( mh.m_breath > 0.0f )
    {
        const float br =
            ( 10.0f + static_cast<float>( hash32( static_cast<std::uint32_t>( mode ) ^ 0x27D4EB2FU ) % 40U ) ) * 0.001f;
        float bp = br * mh.m_time + modalUni( mode, 10 );
        bp -= floorf( bp );
        breathe = mh.m_breath * __sinf( 6.2831853f * bp );
    }
    // The hall's memory (Grow): where much was heard, the mode rings longer (growth) or shorter
    // (erosion) -- up to 2^1.5 at seven times the average, half an octave the other way where nothing was.
    float grown = 0.0f;
    if ( mh.m_grow != 0.0f )
        grown = mh.m_grow * fminf( fmaxf( log2f( 1.0f + memory[hallBin( f )] ) - 1.0f, -0.5f ), 1.5f );
    const float T = fmaxf(
        0.05f, mh.m_decay * exp2f( breathe + grown - mh.m_damp * fmaxf( 0.0f, log2f( f * ( 1.0f / 250.0f ) ) ) ) );
    const float r = mh.m_freeze ? 0.99999988f : __expf( -6.9077553f / ( T * fs ) );
    float       sw, cw;
    sincosf( 6.2831853f * f / fs, &sw, &cw );
    const float ar = r * cw, ai = r * sw;
    // Driven with sqrt(2 (1 - r^2) / N): white noise in, white noise of the same power out, whatever the decay.
    const float g = ( silent || mh.m_freeze )
                        ? 0.0f
                        : sqrtf( 2.0f * fmaxf( 1.0f - r * r, 0.0f ) / static_cast<float>( ModalModes ) ) * weight;
    float       inR, inL;
    __sincosf( 1.5707963f * modalUni( mode, 7 ), &inR, &inL );   // its share of the left and the right input
    const float phL = 6.2831853f * modalUni( mode, 8 );
    const float phR = phL + mh.m_width * 3.1415927f * ( 2.0f * modalUni( mode, 9 ) - 1.0f );
    float       oLs, oLc, oRs, oRc;
    __sincosf( phL, &oLs, &oLc );
    __sincosf( phR, &oRs, &oRc );
    if ( silent )
    {
        oLs = oLc = oRs = oRc = 0.0f;
    }
    // The painted hall: the mode heard through the landscape at its frequency, as the landscape
    // moves on -- the tail draws the aurora, the rain, the tides. Made up by the landscape's RMS.
    if ( mh.m_paint > 0.0f && hdr->m_landColumns > 0 && hdr->m_landRows > 0 )
    {
        const float pos = fminf(
            fmaxf( log2f( fmaxf( f, LandLowHz ) * ( 1.0f / LandLowHz ) ) * ( 1.0f / 9.96578428f ), 0.0f ), 1.0f );
        const float col = pos * static_cast<float>( hdr->m_landColumns - 1 );
        const float rows = static_cast<float>( hdr->m_landRows );
        float       row = mh.m_paintTime / fmaxf( hdr->m_landRowSeconds, 1.0e-4f );
        row -= floorf( row / rows ) * rows;
        const int r0 = min( static_cast<int>( row ), hdr->m_landRows - 1 ), r1 = r0 + 1 == hdr->m_landRows ? 0 : r0 + 1;
        const int c0 = min( static_cast<int>( col ), hdr->m_landColumns - 1 ),
                  c1 = min( c0 + 1, hdr->m_landColumns - 1 );
        const float  fr = row - static_cast<float>( r0 ), fc = col - static_cast<float>( c0 );
        const float *a = land + static_cast<size_t>( r0 ) * hdr->m_landColumns;
        const float *b = land + static_cast<size_t>( r1 ) * hdr->m_landColumns;
        const float  top = a[c0] + ( a[c1] - a[c0] ) * fc, bottom = b[c0] + ( b[c1] - b[c0] ) * fc;
        const float  k = 1.0f - mh.m_paint + mh.m_paint * ( top + ( bottom - top ) * fr ) * mh.m_paintNorm;
        oLs *= k;
        oLc *= k;
        oRs *= k;
        oRc *= k;
    }
    float2    y = state[mode];
    const int lane = static_cast<int>( threadIdx.x ) & 31, warp = static_cast<int>( threadIdx.x ) >> 5;
    for ( int t0 = 0; t0 < n; t0 += ModalTile )
    {
        const int tn = min( ModalTile, n - t0 );
        for ( int s = threadIdx.x; s < tn; s += ModalThreads )
        {
            const float *fr = in + static_cast<size_t>( t0 + s ) * CloudOutChannels;
            sIn[0][s] = fr[0] + fr[2];
            sIn[1][s] = fr[1] + fr[3];
        }
        __syncthreads();
        for ( int i = 0; i < tn; ++i )
        {
            const float x = g * ( inL * sIn[0][i] + inR * sIn[1][i] );
            const float yr = fmaf( ar, y.x, fmaf( -ai, y.y, x ) );
            const float yi = fmaf( ar, y.y, ai * y.x );
            y.x = yr;
            y.y = yi;
            float cL = yr * oLc - yi * oLs;
            float cR = yr * oRc - yi * oRs;
            for ( int off = 16; off > 0; off >>= 1 )
            {
                cL += __shfl_down_sync( 0xFFFFFFFFU, cL, off );
                cR += __shfl_down_sync( 0xFFFFFFFFU, cR, off );
            }
            if ( lane == 0 )
            {
                sAcc[warp][0][i] = cL;
                sAcc[warp][1][i] = cR;
            }
        }
        __syncthreads();
        for ( int s = threadIdx.x; s < 2 * tn; s += ModalThreads )
        {
            const int c = s / tn, i = s - c * tn;
            float     a = 0.0f;
            for ( int w = 0; w < ModalThreads / 32; ++w ) a += sAcc[w][c][i];
            out[c * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    state[mode] = y;
}

/**
 * @brief The modal hall's reduce: the CUDA blocks' sums added sample by sample, in a fixed order,
 *        into the frames' last two channels.
 * @param[in]     hdr      the block header
 * @param[in]     scratch  the modal kernel's per-block sums
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels; channels 4 and 5 are written
 */
__global__ void modalReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out )
{
    __shared__ float part[ReduceGroups][ReduceLanes + 1];
    const int        n = hdr->m_n;
    const int        t = static_cast<int>( blockIdx.x ) * ReduceLanes +
                         static_cast<int>( threadIdx.x );   // channel-major, as the scratch is laid out
    const int        c = t / maxN;
    const int        s = t - c * maxN;
    const bool       live = c < 2 && s < n;
    float            acc = 0.0f;
    if ( live )
        for ( int k = threadIdx.y; k < ModalBlocks; k += ReduceGroups )
            acc += scratch[static_cast<size_t>( k ) * 2 * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if ( threadIdx.y == 0 && live )
    {
        for ( int q = 1; q < ReduceGroups; ++q ) acc += part[q][threadIdx.x];
        out[s * CloudOutChannels + CloudChannels + c] = acc;
    }
}

/**
 * @brief A 1024-point complex FFT in shared memory, by every thread of a EchoThreads block:
 *        bit-reversed order, then ten stages of one butterfly per thread (unnormalised).
 * @param[in,out] a        the 1024 values, transformed in place
 * @param[in]     inverse  true: the inverse transform (times 1024)
 */
__device__ void fft1024( float2 *a, bool inverse )
{
    for ( int i = threadIdx.x; i < EchoN; i += EchoThreads )
    {
        const int j = static_cast<int>( __brev( static_cast<unsigned>( i ) ) >> 22 );
        if ( j > i )
        {
            const float2 tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    __syncthreads();
    const int t = static_cast<int>( threadIdx.x );
    for ( int len = 2; len <= EchoN; len <<= 1 )
    {
        const int half = len >> 1;
        const int pos = t % half;
        const int i0 = ( t / half ) * len + pos, i1 = i0 + half;
        float     sn, cs;
        sincospif( ( inverse ? 2.0f : -2.0f ) * static_cast<float>( pos ) / static_cast<float>( len ), &sn, &cs );
        const float2 u = a[i0], v = a[i1];
        const float2 w = make_float2( v.x * cs - v.y * sn, v.x * sn + v.y * cs );
        a[i0] = make_float2( u.x + w.x, u.y + w.y );
        a[i1] = make_float2( u.x - w.x, u.y - w.y );
        __syncthreads();
    }
}

/**
 * @brief The spectral echo, one CUDA block: the block into the history, a Hann-windowed frame
 *        through the FFT (left as the real part, right as the imaginary), every band of each ear
 *        through its own delay line with feedback, back through the inverse FFT, overlapped into
 *        the output; the oldest block of the overlap is finished and written to the frames' last
 *        two channels.
 * @param[in]     hdr   the block header (its `echo`)
 * @param[in,out] io    the block's frames, CloudOutChannels each: the voices read, channels 6 and 7 written
 * @param[in,out] hist  the input history, two rings of EchoN
 * @param[in,out] acc   the overlap, two rings of EchoN on the history's indices
 * @param[in,out] ring  per ear and band EchoSlots spectra: the delay lines
 * @param[in,out] st    the history's write index and the frame count
 */
__global__ void __launch_bounds__( EchoThreads )
    spectralEchoKernel( const BlockHeader *hdr, float *io, float *hist, float *acc, float2 *ring, int *st )
{
    __shared__ float2   z[EchoN];
    const SpectralEcho &E = hdr->m_echo;
    const int           n = hdr->m_n;
    const int           t = static_cast<int>( threadIdx.x );
    if ( E.m_clear )
    {
        for ( int i = t; i < 2 * EchoBins * EchoSlots; i += EchoThreads ) ring[i] = make_float2( 0.0f, 0.0f );
        for ( int i = t; i < 2 * EchoN; i += EchoThreads )
        {
            hist[i] = 0.0f;
            acc[i] = 0.0f;
        }
        if ( t == 0 )
        {
            st[0] = 0;
            st[1] = 0;
        }
        __syncthreads();
    }
    if ( !E.m_on || n > EchoN / 4 )
    {   // off, or blocks too long for the window's overlap: silent
        for ( int s = t; s < n; s += EchoThreads )
        {
            io[s * CloudOutChannels + 6] = 0.0f;
            io[s * CloudOutChannels + 7] = 0.0f;
        }
        return;
    }
    const int w0 = st[0], frame = st[1];
    for ( int s = t; s < n; s += EchoThreads )
    {   // the block over the history's oldest samples
        const float *f = io + static_cast<size_t>( s ) * CloudOutChannels;
        const int    i = ( w0 + s ) & ( EchoN - 1 );
        hist[i] = f[0] + f[2];
        hist[EchoN + i] = f[1] + f[3];
    }
    __syncthreads();
    const int start = ( w0 + n ) & ( EchoN - 1 );   // the frame's oldest sample
    for ( int k = t; k < EchoN; k += EchoThreads )
    {
        const int   i = ( start + k ) & ( EchoN - 1 );
        const float win = 0.5f - 0.5f * cospif( 2.0f * static_cast<float>( k ) / static_cast<float>( EchoN ) );
        z[k] = make_float2( win * hist[i], win * hist[EchoN + i] );
    }
    __syncthreads();
    fft1024( z, false );
    // Every band: the two ears' spectra apart, each through its line, the delayed ones kept.
    const float fs = hdr->m_sampleRate;
    float2      yl[2], yr[2];
    const int   bins = t == 0 ? 2 : 1;
    for ( int b = 0; b < bins; ++b )
    {
        const int    k = b == 0 ? t : EchoN / 2;
        const float2 Zk = z[k], Zm = z[( EchoN - k ) & ( EchoN - 1 )];
        const float2 xl = make_float2( 0.5f * ( Zk.x + Zm.x ), 0.5f * ( Zk.y - Zm.y ) );
        const float2 xr = make_float2( 0.5f * ( Zk.y + Zm.y ), -0.5f * ( Zk.x - Zm.x ) );
        const float  fk = static_cast<float>( k ) * fs / static_cast<float>( EchoN );
        const float  place =
            fminf( fmaxf( log2f( fmaxf( fk, 20.0f ) * ( 1.0f / 20.0f ) ) * ( 1.0f / 9.96578428f ), 0.0f ), 1.0f );
        const float scatter =
            1.0f + 0.75f * E.m_scatter *
                       ( 2.0f * static_cast<float>( hash32( static_cast<std::uint32_t>( k ) ^ 0x3C6EF372U ) >> 8 ) *
                             ( 1.0f / 16777216.0f ) -
                         1.0f );
        const float d = E.m_frames * exp2f( 2.0f * E.m_tilt * ( place - 0.5f ) ) * scatter;
        const int   di = min( max( static_cast<int>( lrintf( d ) ), 1 ), EchoSlots - 1 );
        const float fb = E.m_feedback * ( 1.0f - E.m_damp * place );
        const int   slot = frame & ( EchoSlots - 1 ), back = ( frame - di ) & ( EchoSlots - 1 );
        float2     *rl = ring + static_cast<size_t>( k ) * EchoSlots;
        float2     *rr = ring + static_cast<size_t>( EchoBins + k ) * EchoSlots;
        yl[b] = rl[back];
        yr[b] = rr[back];
        rl[slot] = make_float2( xl.x + fb * yl[b].x, xl.y + fb * yl[b].y );
        rr[slot] = make_float2( xr.x + fb * yr[b].x, xr.y + fb * yr[b].y );
    }
    __syncthreads();
    for ( int b = 0; b < bins; ++b )
    {   // the two ears together again: left real, right imaginary
        const int k = b == 0 ? t : EchoN / 2;
        z[k] = make_float2( yl[b].x - yr[b].y, yl[b].y + yr[b].x );
        if ( k > 0 && k < EchoN / 2 ) z[EchoN - k] = make_float2( yl[b].x + yr[b].y, -yl[b].y + yr[b].x );
    }
    __syncthreads();
    fft1024( z, true );
    // Overlap-add under the same window: Hann times Hann over N / n frames sums to 3/8 N / n.
    const float norm =
        1.0f / ( static_cast<float>( EchoN ) * 0.375f * static_cast<float>( EchoN ) / static_cast<float>( n ) );
    for ( int k = t; k < EchoN; k += EchoThreads )
    {
        const int   i = ( start + k ) & ( EchoN - 1 );
        const float win =
            ( 0.5f - 0.5f * cospif( 2.0f * static_cast<float>( k ) / static_cast<float>( EchoN ) ) ) * norm;
        acc[i] += win * z[k].x;
        acc[EchoN + i] += win * z[k].y;
    }
    __syncthreads();
    for ( int s = t; s < n; s += EchoThreads )
    {   // the oldest block is finished: out, and cleared for the next
        const int i = ( start + s ) & ( EchoN - 1 );
        io[s * CloudOutChannels + 6] = acc[i];
        io[s * CloudOutChannels + 7] = acc[EchoN + i];
        acc[i] = 0.0f;
        acc[EchoN + i] = 0.0f;
    }
    if ( t == 0 )
    {
        st[0] = ( w0 + n ) & ( EchoN - 1 );
        st[1] = frame + 1;
    }
}

/**
 * @brief A 4096-point complex FFT in shared memory, by every thread of a MeltThreads block:
 *        bit-reversed order, then twelve stages of two butterflies per thread (unnormalised).
 * @param[in,out] a        the 4096 values, transformed in place
 * @param[in]     inverse  true: the inverse transform (times 4096)
 */
__device__ void fft4096( float2 *a, bool inverse )
{
    for ( int i = threadIdx.x; i < MeltN; i += MeltThreads )
    {
        const int j = static_cast<int>( __brev( static_cast<unsigned>( i ) ) >> 20 );
        if ( j > i )
        {
            const float2 tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    __syncthreads();
    for ( int len = 2; len <= MeltN; len <<= 1 )
    {
        const int half = len >> 1;
        for ( int b = threadIdx.x; b < MeltN / 2; b += MeltThreads )
        {
            const int pos = b & ( half - 1 );
            const int i0 = ( b / half ) * len + pos, i1 = i0 + half;
            float     sn, cs;
            sincospif( ( inverse ? 2.0f : -2.0f ) * static_cast<float>( pos ) / static_cast<float>( len ), &sn, &cs );
            const float2 u = a[i0], v = a[i1];
            const float2 w = make_float2( v.x * cs - v.y * sn, v.x * sn + v.y * cs );
            a[i0] = make_float2( u.x + w.x, u.y + w.y );
            a[i1] = make_float2( u.x - w.x, u.y - w.y );
        }
        __syncthreads();
    }
}

/**
 * @brief A sum of two double-floats (a value as an unevaluated sum hi + lo of two floats, about 48
 *        bits): Knuth's two-sum, then renormalised. Only additions -- nothing to contract or reorder.
 * @param[in] a  one (x the high part, y the low)
 * @param[in] b  the other
 * @return a + b
 */
__device__ float2 dfAdd( float2 a, float2 b )
{
    const float s = a.x + b.x;
    const float bb = s - a.x;
    const float e = ( a.x - ( s - bb ) ) + ( b.x - bb ) + a.y + b.y;
    const float hi = s + e;
    return make_float2( hi, e - ( hi - s ) );
}

/**
 * @brief The difference of two double-floats.
 * @param[in] a  one
 * @param[in] b  the other
 * @return a - b
 */
__device__ float2 dfSub( float2 a, float2 b )
{
    return dfAdd( a, make_float2( -b.x, -b.y ) );
}

/**
 * @brief A cumulative sum below a point on the band axis, band b covering [b, b + 1): the
 *        inclusive sums at whole bands, linear within one.
 * @param[in] c  the inclusive sums as double-floats, MeltBands (c[b] = the sum over bands 0 .. b)
 * @param[in] x  the point, 0 .. MeltBands
 * @return the sum below it
 */
__device__ float2 meltBelow( const float2 *c, float x )
{
    const int    j = min( static_cast<int>( x ), MeltBands );
    const float2 lo = j > 0 ? c[j - 1] : make_float2( 0.0f, 0.0f );
    if ( j >= MeltBands ) return lo;
    const float2 d = dfSub( c[j], lo );
    return dfAdd( lo, make_float2( ( x - static_cast<float>( j ) ) * ( d.x + d.y ), 0.0f ) );
}

/**
 * @brief The melt, one CUDA block: the block into the history; every MeltHop samples a frame --
 *        Hann window, FFT (left real, right imaginary), each band's energy held and its heard
 *        frequency tracked, the held energy blurred across the bands (each band scatters it evenly
 *        over 1 + 2 kappa x bands around its centre x, gathered through a cumulative sum in double-float,
 *        so nothing is gained or lost), sung again
 *        at the tracked frequencies, inverse FFT, overlapped into the output; the oldest block of the
 *        overlap finished and written to channels 8 and 9.
 * @param[in]     hdr     the header (its `melt`)
 * @param[in,out] io      the block's frames: the voices read, channels 8 and 9 written
 * @param[in,out] hist    the input history, two rings of MeltN
 * @param[in,out] acc     the overlap, two rings of MeltN on the history's indices
 * @param[in,out] energy  the held energy per band, left then right
 * @param[in,out] omega   each band's sung frequency, radians a sample
 * @param[in,out] phase   each band's sung phase, left then right
 * @param[in,out] heard   each band's last heard phase, left then right
 * @param[in,out] st      the history's write index, the samples since the last frame
 */
__global__ void __launch_bounds__( MeltThreads )
    meltKernel( const BlockHeader *hdr, float *io, float *hist, float *acc, float *energy, float *omega, float *phase,
                float *heard, int *st )
{
    __shared__ float2 z[MeltN];         // the frame; between the transforms the cumulative sums
    __shared__ float2 warpSum[2][32];   // the scan's warp totals, per ear
    const Melt       &M = hdr->m_melt;
    const int         n = hdr->m_n;
    const int         t = static_cast<int>( threadIdx.x );
    constexpr float   TwoPi = 6.28318531f;
    if ( M.m_clear )
    {
        for ( int i = t; i < 2 * MeltN; i += MeltThreads )
        {
            hist[i] = 0.0f;
            acc[i] = 0.0f;
        }
        for ( int b = t; b < MeltBands; b += MeltThreads )
        {
            energy[b] = energy[MeltBands + b] = 0.0f;
            omega[b] = TwoPi * static_cast<float>( b ) / static_cast<float>( MeltN );
            phase[b] = TwoPi * static_cast<float>( hash32( static_cast<std::uint32_t>( b ) ^ 0x4D454C54U ) >> 8 ) *
                       ( 1.0f / 16777216.0f );
            phase[MeltBands + b] = TwoPi *
                                   static_cast<float>( hash32( static_cast<std::uint32_t>( b ) ^ 0x4D454C55U ) >> 8 ) *
                                   ( 1.0f / 16777216.0f );
            heard[b] = heard[MeltBands + b] = 0.0f;
        }
        if ( t == 0 )
        {
            st[0] = 0;
            st[1] = 0;
        }
        __syncthreads();
    }
    if ( !M.m_on || n > MeltHop )
    {   // off, or blocks longer than the hop: silent
        for ( int s = t; s < n; s += MeltThreads )
        {
            io[s * CloudOutChannels + 8] = 0.0f;
            io[s * CloudOutChannels + 9] = 0.0f;
        }
        return;
    }
    const int w0 = st[0], count = st[1] + n;
    for ( int s = t; s < n; s += MeltThreads )
    {
        const float *f = io + static_cast<size_t>( s ) * CloudOutChannels;
        const int    i = ( w0 + s ) & ( MeltN - 1 );
        hist[i] = f[0] + f[2];
        hist[MeltN + i] = f[1] + f[3];
    }
    __syncthreads();
    const int start = ( w0 + n ) & ( MeltN - 1 );   // the window's oldest sample
    if ( count >= MeltHop )
    {
        for ( int k = t; k < MeltN; k += MeltThreads )
        {
            const int   i = ( start + k ) & ( MeltN - 1 );
            const float win = 0.5f - 0.5f * cospif( 2.0f * static_cast<float>( k ) / static_cast<float>( MeltN ) );
            z[k] = make_float2( win * hist[i], win * hist[MeltN + i] );
        }
        __syncthreads();
        fft4096( z, false );
        // Each of the thread's two bands (2t, 2t + 1): heard, held, tracked.
        const float hop = static_cast<float>( MeltHop );
        float       held[2][2];   // [band][ear]
        for ( int q = 0; q < 2; ++q )
        {
            const int    b = 2 * t + q;
            const float2 Zk = z[b], Zm = z[( MeltN - b ) & ( MeltN - 1 )];
            const float2 xl = make_float2( 0.5f * ( Zk.x + Zm.x ), 0.5f * ( Zk.y - Zm.y ) );
            const float2 xr = make_float2( 0.5f * ( Zk.y + Zm.y ), -0.5f * ( Zk.x - Zm.x ) );
            const float  eL = b > 0 ? xl.x * xl.x + xl.y * xl.y : 0.0f;
            const float  eR = b > 0 ? xr.x * xr.x + xr.y * xr.y : 0.0f;
            // The heard frequency: the phase's advance over the hop, beyond the band centre's, wrapped.
            const float centre = TwoPi * static_cast<float>( b ) / static_cast<float>( MeltN );
            const float pL = atan2f( xl.y, xl.x ), pR = atan2f( xr.y, xr.x );
            float       dL = pL - heard[b] - centre * hop, dR = pR - heard[MeltBands + b] - centre * hop;
            dL -= TwoPi * rintf( dL / TwoPi );
            dR -= TwoPi * rintf( dR / TwoPi );
            heard[b] = pL;
            heard[MeltBands + b] = pR;
            const float wIn = eL + eR > 1.0e-20f ? centre + ( eL * dL + eR * dR ) / ( ( eL + eR ) * hop ) : centre;
            held[q][0] = M.m_keep * energy[b] + ( 1.0f - M.m_keep ) * eL;
            held[q][1] = M.m_keep * energy[MeltBands + b] + ( 1.0f - M.m_keep ) * eR;
            const float fresh =
                fminf( 1.0f, ( 1.0f - M.m_keep ) * ( eL + eR ) / ( held[q][0] + held[q][1] + 1.0e-20f ) );
            float w = omega[b];
            w += fresh * ( wIn - w );
            w += M.m_pull * ( centre - w );
            omega[b] = w;
        }
        __syncthreads();   // every band read: z becomes the cumulative sums
        // The blur scatters band x's energy evenly over [x - h, x + h], h = 0.5 + kappa x (x its
        // centre): a density held / (1 + 2 kappa x). Band b then gathers the densities of the bands
        // whose centres lie in [(x - 0.5) / (1 + kappa), (x + 0.5) / (1 - kappa)] -- the cumulative
        // densities' difference. kappa 0 is the identity; the variance a frame is about (kappa x)^2 / 3.
        const float kappa = 1.7320508f * M.m_widen;
        float       dens[2][2];
        for ( int q = 0; q < 2; ++q )
            for ( int ear = 0; ear < 2; ++ear )
                dens[q][ear] = held[q][ear] / ( 1.0f + 2.0f * kappa * ( static_cast<float>( 2 * t + q ) + 0.5f ) );
        // Inclusive sums over the bands, per ear, in double-float: the thread's pair, a warp's scan by
        // shuffles, the warps' totals by the first warp -- all in a fixed order.
        float2   *c = z;   // [ear * MeltBands + b]
        const int lane = t & 31, warp = t >> 5;
        float2    excl[2], pair[2];
        for ( int ear = 0; ear < 2; ++ear )
        {
            pair[ear] = dfAdd( make_float2( dens[0][ear], 0.0f ), make_float2( dens[1][ear], 0.0f ) );
            float2 s = pair[ear];
            for ( int off = 1; off < 32; off <<= 1 )
            {
                const float2 o =
                    make_float2( __shfl_up_sync( 0xFFFFFFFFU, s.x, off ), __shfl_up_sync( 0xFFFFFFFFU, s.y, off ) );
                if ( lane >= off ) s = dfAdd( o, s );
            }
            if ( lane == 31 ) warpSum[ear][warp] = s;
            excl[ear] = dfSub( s, pair[ear] );
        }
        __syncthreads();
        if ( warp == 0 )
        {
            for ( int ear = 0; ear < 2; ++ear )
            {
                const float2 own = warpSum[ear][lane];
                float2       s = own;
                for ( int off = 1; off < 32; off <<= 1 )
                {
                    const float2 o =
                        make_float2( __shfl_up_sync( 0xFFFFFFFFU, s.x, off ), __shfl_up_sync( 0xFFFFFFFFU, s.y, off ) );
                    if ( lane >= off ) s = dfAdd( o, s );
                }
                warpSum[ear][lane] = dfSub( s, own );   // exclusive
            }
        }
        __syncthreads();
        for ( int ear = 0; ear < 2; ++ear )
        {
            const float2 base = dfAdd( warpSum[ear][warp], excl[ear] );
            c[ear * MeltBands + 2 * t] = dfAdd( base, make_float2( dens[0][ear], 0.0f ) );
            c[ear * MeltBands + 2 * t + 1] = dfAdd( base, pair[ear] );
        }
        __syncthreads();
        float out[2][2];
        for ( int q = 0; q < 2; ++q )
        {
            const int   b = 2 * t + q;
            const float x = static_cast<float>( b ) + 0.5f;
            const float lo = fmaxf( 1.0f, ( x - 0.5f ) / ( 1.0f + kappa ) ),
                        hi = fminf( static_cast<float>( MeltBands ), ( x + 0.5f ) / ( 1.0f - kappa ) );
            for ( int ear = 0; ear < 2; ++ear )
            {
                const float2 *ce = c + ear * MeltBands;
                float         e = 0.0f;
                if ( b > 0 && hi > lo )
                {
                    const float2 d = dfSub( meltBelow( ce, hi ), meltBelow( ce, lo ) );
                    e = d.x + d.y;
                }
                out[q][ear] = fmaxf( e, 0.0f );
            }
        }
        __syncthreads();   // the sums read: z becomes the sung spectrum
        for ( int q = 0; q < 2; ++q )
        {
            const int b = 2 * t + q;
            energy[b] = out[q][0];
            energy[MeltBands + b] = out[q][1];
            float2 y[2];
            for ( int ear = 0; ear < 2; ++ear )
            {
                const float det =
                    M.m_detune *
                    ( 2.0f *
                          static_cast<float>(
                              hash32( static_cast<std::uint32_t>( b ) ^ ( ear ? 0x44455452U : 0x4445544CU ) ) >> 8 ) *
                          ( 1.0f / 16777216.0f ) -
                      1.0f );
                float ph = phase[ear * MeltBands + b] + ( omega[b] + det ) * hop;
                ph -= TwoPi * rintf( ph / TwoPi );
                phase[ear * MeltBands + b] = ph;
                float sn, cs;
                __sincosf( ph, &sn, &cs );
                const float mag = b > 0 ? sqrtf( out[q][ear] ) : 0.0f;
                y[ear] = make_float2( mag * cs, mag * sn );
            }
            z[b] = make_float2( y[0].x - y[1].y, y[0].y + y[1].x );
            if ( b > 0 ) z[MeltN - b] = make_float2( y[0].x + y[1].y, -y[0].y + y[1].x );
        }
        if ( t == 0 ) z[MeltN / 2] = make_float2( 0.0f, 0.0f );
        __syncthreads();
        fft4096( z, true );
        const float norm = 1.0f / ( static_cast<float>( MeltN ) * 0.375f * static_cast<float>( MeltN ) / hop );
        for ( int k = t; k < MeltN; k += MeltThreads )
        {
            const int   i = ( start + k ) & ( MeltN - 1 );
            const float win =
                ( 0.5f - 0.5f * cospif( 2.0f * static_cast<float>( k ) / static_cast<float>( MeltN ) ) ) * norm;
            acc[i] += win * z[k].x;
            acc[MeltN + i] += win * z[k].y;
        }
        __syncthreads();
    }
    for ( int s = t; s < n; s += MeltThreads )
    {   // the oldest block is finished: out, and cleared for the next
        const int i = ( start + s ) & ( MeltN - 1 );
        io[s * CloudOutChannels + 8] = acc[i];
        io[s * CloudOutChannels + 9] = acc[MeltN + i];
        acc[i] = 0.0f;
        acc[MeltN + i] = 0.0f;
    }
    if ( t == 0 )
    {
        st[0] = ( w0 + n ) & ( MeltN - 1 );
        st[1] = count >= MeltHop ? count - MeltHop : count;
    }
}

/**
 * @brief The instrument listening to itself, one CUDA block: the voices' block into a history, a
 *        Hann-windowed frame through the FFT, and three numbers -- the level, the energy's centre
 *        on the log axis, the change (the magnitudes' rise against the last frame over their sum).
 *        Summed in a fixed order.
 * @param[in]     hdr    the header
 * @param[in]     io     the block's frames (the voices' four channels are read)
 * @param[in,out] hist   the history, two rings of EchoN
 * @param[in,out] prev   the last frame's magnitudes, EchoBins
 * @param[in,out] st     the history's write index
 * @param[out]    stats  receives level (energy), centroid, flux at [2], [3], [4]
 */
__global__ void __launch_bounds__( EchoThreads )
    listenKernel( const BlockHeader *hdr, const float *io, float *hist, float *prev, int *st, float *stats )
{
    __shared__ float2 z[EchoN];
    __shared__ float  red[3][EchoThreads];
    const int         n = hdr->m_n;
    const int         t = static_cast<int>( threadIdx.x );
    const int         w0 = st[0];
    for ( int s = t; s < n; s += EchoThreads )
    {
        const float *f = io + static_cast<size_t>( s ) * CloudOutChannels;
        const int    i = ( w0 + s ) & ( EchoN - 1 );
        hist[i] = f[0] + f[2];
        hist[EchoN + i] = f[1] + f[3];
    }
    __syncthreads();
    const int start = ( w0 + n ) & ( EchoN - 1 );
    for ( int k = t; k < EchoN; k += EchoThreads )
    {
        const int   i = ( start + k ) & ( EchoN - 1 );
        const float win = 0.5f - 0.5f * cospif( 2.0f * static_cast<float>( k ) / static_cast<float>( EchoN ) );
        z[k] = make_float2( win * hist[i], win * hist[EchoN + i] );
    }
    __syncthreads();
    fft1024( z, false );
    const float fs = hdr->m_sampleRate;
    float       e = 0.0f, c = 0.0f, rise = 0.0f, sum = 0.0f;
    const int   bins = t == 0 ? 2 : 1;
    for ( int b = 0; b < bins; ++b )
    {
        const int    k = b == 0 ? t : EchoN / 2;
        const float2 Zk = z[k], Zm = z[( EchoN - k ) & ( EchoN - 1 )];
        const float  xr = 0.5f * ( Zk.x + Zm.x ), xi = 0.5f * ( Zk.y - Zm.y );
        const float  yr = 0.5f * ( Zk.y + Zm.y ), yi = -0.5f * ( Zk.x - Zm.x );
        const float  en = xr * xr + xi * xi + yr * yr + yi * yi;
        const float  mag = sqrtf( en );
        const float  fk = static_cast<float>( k ) * fs / static_cast<float>( EchoN );
        const float  place =
            fminf( fmaxf( log2f( fmaxf( fk, 20.0f ) * ( 1.0f / 20.0f ) ) * ( 1.0f / 9.96578428f ), 0.0f ), 1.0f );
        e += en;
        c += en * place;
        rise += fmaxf( 0.0f, mag - prev[k] );
        sum += mag;
        prev[k] = mag;
    }
    red[0][t] = e;
    red[1][t] = c;
    red[2][t] = rise;
    __syncthreads();
    for ( int s = EchoThreads / 2; s > 0; s >>= 1 )
    {
        if ( t < s )
        {
            red[0][t] += red[0][t + s];
            red[1][t] += red[1][t + s];
            red[2][t] += red[2][t + s];
        }
        __syncthreads();
    }
    // The magnitudes' sum for the flux, in a second pass over the energy's shared row (its total kept first).
    __shared__ float total[2];
    if ( t == 0 )
    {
        total[0] = red[0][0];
        total[1] = red[1][0];
    }
    __syncthreads();
    red[1][t] = sum;
    __syncthreads();
    for ( int s = EchoThreads / 2; s > 0; s >>= 1 )
    {
        if ( t < s ) red[1][t] += red[1][t + s];
        __syncthreads();
    }
    if ( t == 0 )
    {
        // Energy to a level: a full-scale sine in one ear gives (N/2)^2 (3/8) per bin pair under Hann.
        const float ref = 0.375f * 0.25f * static_cast<float>( EchoN ) * static_cast<float>( EchoN );
        stats[2] = 10.0f * log10f( total[0] / ref + 1.0e-12f );
        stats[3] = total[0] > 1.0e-12f ? total[1] / total[0] : 0.0f;
        stats[4] = red[1][0] > 1.0e-9f ? red[2][0] / red[1][0] : 0.0f;
        st[0] = ( w0 + n ) & ( EchoN - 1 );
    }
}

/**
 * @brief Fills the reservoir once: the coupling uniform in +-sqrt(3/N) (unit spectral radius, scaled
 *        by `chaos` when it runs), the inputs' and the readouts' weights in +-1, a small start.
 * @param[in,out] w     the coupling, N x N
 * @param[in,out] win   the input weights, N x ReservoirInputs
 * @param[in,out] rout  the readouts, ReservoirOutputs x N
 * @param[in,out] x     the state, both buffers
 */
__global__ void reservoirInitKernel( float *w, float *win, float *rout, float *x )
{
    const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
    const float  a = sqrtf( 3.0f / static_cast<float>( ReservoirN ) );
    auto         uni = []( size_t k, std::uint32_t salt )
    {
        return static_cast<float>( hash32( static_cast<std::uint32_t>( k ) * 0x9E3779B9U ^ salt ) >> 8 ) *
               ( 1.0f / 16777216.0f );
    };
    if ( i < static_cast<size_t>( ReservoirN ) * ReservoirN ) w[i] = a * ( 2.0f * uni( i, 0x52455330U ) - 1.0f );
    if ( i < static_cast<size_t>( ReservoirN ) * ReservoirInputs ) win[i] = 2.0f * uni( i, 0x52455331U ) - 1.0f;
    if ( i < static_cast<size_t>( ReservoirOutputs ) * ReservoirN ) rout[i] = 2.0f * uni( i, 0x52455332U ) - 1.0f;
    if ( i < 2 * static_cast<size_t>( ReservoirN ) ) x[i] = 0.1f * ( 2.0f * uni( i % ReservoirN, 0x52455333U ) - 1.0f );
}

/**
 * @brief One block of the reservoir: a warp per neuron sums its row of the coupling against the last
 *        state (lanes strided, then shuffled in a fixed order), adds its inputs, and leaks towards
 *        tanh of it. Reads one buffer, writes the other (by the block's parity).
 * @param[in]     hdr  the header
 * @param[in]     w    the coupling
 * @param[in]     win  the input weights
 * @param[in,out] x    the state, two buffers of N
 */
__global__ void reservoirStepKernel( const BlockHeader *hdr, const float *w, const float *win, float *x )
{
    const Reservoir &R = hdr->m_reservoir;
    if ( !R.m_on ) return;
    const int warp = static_cast<int>( ( blockIdx.x * blockDim.x + threadIdx.x ) >> 5 ),
              lane = static_cast<int>( threadIdx.x & 31 );
    if ( warp >= ReservoirN ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *xp = x + ( odd ? ReservoirN : 0 );
    float       *xn = x + ( odd ? 0 : ReservoirN );
    const float *row = w + static_cast<size_t>( warp ) * ReservoirN;
    float        s = 0.0f;
    for ( int j = lane; j < ReservoirN; j += 32 ) s = fmaf( row[j], xp[j], s );
    for ( int off = 16; off > 0; off >>= 1 ) s += __shfl_down_sync( 0xFFFFFFFFU, s, off );
    if ( lane == 0 )
    {
        float in = 0.0f;
        for ( int k = 0; k < ReservoirInputs; ++k ) in += win[warp * ReservoirInputs + k] * R.m_u[k];
        const float bias =
            0.2f * ( static_cast<float>( hash32( static_cast<std::uint32_t>( warp ) ^ 0x42494153U ) >> 8 ) *
                         ( 1.0f / 16777216.0f ) -
                     0.5f );
        xn[warp] = ( 1.0f - R.m_leak ) * xp[warp] + R.m_leak * tanhf( R.m_chaos * s + R.m_input * in + bias );
    }
}

/**
 * @brief The reservoir's readouts, one CUDA block: tanh of each fixed projection of the new state, summed in a fixed
 *        order.
 * @param[in]  hdr    the header
 * @param[in]  rout   the readouts
 * @param[in]  x      the state, two buffers
 * @param[out] stats  receives the outputs at [5] .. [8]
 */
__global__ void reservoirReadKernel( const BlockHeader *hdr, const float *rout, const float *x, float *stats )
{
    __shared__ float red[256];
    const Reservoir &R = hdr->m_reservoir;
    if ( !R.m_on ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *xn = x + ( odd ? 0 : ReservoirN );
    const int    t = static_cast<int>( threadIdx.x );
    for ( int k = 0; k < ReservoirOutputs; ++k )
    {
        float s = 0.0f;
        for ( int i = t; i < ReservoirN; i += 256 ) s += rout[k * ReservoirN + i] * xn[i];
        red[t] = s;
        __syncthreads();
        for ( int h = 128; h > 0; h >>= 1 )
        {
            if ( t < h ) red[t] += red[t + h];
            __syncthreads();
        }
        if ( t == 0 ) stats[5 + k] = tanhf( red[0] * 2.0f * rsqrtf( static_cast<float>( ReservoirN ) ) );
        __syncthreads();
    }
}

/**
 * @brief Writes a failed CUDA call into `error`.
 * @param[in]  e      the call's result
 * @param[in]  what   the call, for the message
 * @param[out] error  receives the message (may be null)
 * @return true when the call succeeded
 */
bool ok( cudaError_t e, const char *what, std::string *error )
{
    if ( e == cudaSuccess ) return true;
    if ( error ) *error = std::string( what ) + ": " + cudaGetErrorString( e );
    return false;
}

}   // namespace

/** @brief The CUDA side of CloudEngine. */
struct CloudEngine::Impl
{
    CloudConfig             m_cfg;          ///< the layout
    int                     m_pcap = 0;     ///< partials per voice in the tables, a multiple of Chunk
    int                     m_chunks = 0;   ///< render CUDA blocks: voices * pcap / Chunk
    std::vector<CloudVoice> m_voices;       ///< the host's voices, edited by the caller
    /** @brief What a voice's power sum was computed for: when none of it changed, the sum is reused. */
    struct ShapeKey
    {
        float m_hz = -1.0f;            ///< fundamental
        int   m_harmonics = 0;         ///< harmonics summed
        float m_tilt = 0.0f;           ///< spectral tilt
        float m_oddEven = 0.0f;        ///< odd/even weight
        float m_brightness = 0.0f;     ///< brightness window
        float m_inharmonic = 0.0f;     ///< stretch
        float m_presenceDb = 0.0f;     ///< presence bell
        float m_distance = 0.0f;       ///< the presence's distance
        float m_lowCut = 0.0f;         ///< pad low cut
        float m_stretch = 1.0f;        ///< the harmonics' stretch exponent
        float m_shift = 0.0f;          ///< the partials' shift, Hz
        float m_horizon = 0.0f;        ///< the horizon's amount
        float m_horizonRatio = 1.0f;   ///< the horizon over the fundamental
        /**
         * @brief Equal in every field.
         * @param[in] o  the other key
         * @return true when the sum can be reused
         */
        bool operator==( const ShapeKey &o ) const
        {
            return m_hz == o.m_hz && m_harmonics == o.m_harmonics && m_tilt == o.m_tilt && m_oddEven == o.m_oddEven &&
                   m_brightness == o.m_brightness && m_inharmonic == o.m_inharmonic && m_presenceDb == o.m_presenceDb &&
                   m_distance == o.m_distance && m_lowCut == o.m_lowCut && m_stretch == o.m_stretch &&
                   m_shift == o.m_shift && m_horizon == o.m_horizon && m_horizonRatio == o.m_horizonRatio;
        }
    };
    std::vector<ShapeKey> m_shapeKey;   ///< per voice: what powerSum was made for
    std::vector<double>
        m_powerSum;   ///< per voice: the sum over its harmonics of the squared shape (tilt, odd/even, brightness, presence, low cut)
    int             m_nPrev = 0;               ///< the previous block's length, sent with the next
    int             m_shapeCount = 0;          ///< shapes in the bank on the GPU
    int             m_landColumns = 0;         ///< the landscape on the GPU: columns (0 = none)
    int             m_landRows = 0;            ///< its rows
    float           m_landRowSeconds = 0.1f;   ///< seconds per row
    int             m_device = 0;              ///< the CUDA device
    cudaDeviceProp  m_props{};                 ///< its properties
    cudaStream_t    m_stream = nullptr;        ///< the stream the graph runs on (highest priority)
    cudaEvent_t     m_evStart = nullptr;       ///< recorded before the graph
    cudaEvent_t     m_evStop = nullptr;        ///< recorded after it: wait() waits for this one
    cudaGraph_t     m_graph = nullptr;         ///< the captured block
    cudaGraphExec_t m_exec = nullptr;          ///< its executable form
    unsigned char  *m_hUpload = nullptr;       ///< pinned host: header + voices as the GPU will read them
    unsigned char  *m_dUpload = nullptr;       ///< device: the same
    float          *m_hOut = nullptr;          ///< pinned host: the finished block
    float          *m_dOut = nullptr;          ///< device: the finished block
    float          *m_dScratch = nullptr;      ///< device: per chunk a stereo block
    uint4          *m_dRun = nullptr;          ///< device: per partial start phase, step, glide
    float4 *m_dAmp = nullptr;   ///< device: per partial the four amplitudes (near L/R, far L/R) the last block ended on
    float4 *m_dGains =
        nullptr;   ///< device: per partial three entries, the near, the far and the near cosine ramps of this block
    float2 *m_dAmpQ =
        nullptr;   ///< device: per partial the near plane's cosine parts the last block ended on (binaural)
    float       *m_dEchoHist = nullptr;    ///< device: the spectral echo's input history, two rings of EchoN
    float       *m_dEchoAcc = nullptr;     ///< device: its overlap, two rings of EchoN
    float2      *m_dEchoRing = nullptr;    ///< device: its delay lines, per ear and band EchoSlots spectra
    int         *m_dEchoState = nullptr;   ///< device: its history index and frame count
    float2      *m_dFlockA = nullptr;      ///< device: the flock's state (offset, velocity), buffer A
    float2      *m_dFlockB = nullptr;      ///< device: buffer B
    float       *m_dSyncTheta = nullptr;   ///< device: per partial the sync's phase
    float       *m_dSyncX = nullptr;       ///< device: per partial its place on its circle (Swarm)
    float2      *m_dSyncOrder = nullptr;   ///< device: per voice and group the sync's mean phasors, three planes
    PartialSync  m_sync;                   ///< the host's copy of the partial sync's settings
    Ripple       m_ripple;                 ///< the host's copy of the ripple
    Reservoir    m_reservoir;              ///< the host's copy of the reservoir's settings
    Melt         m_melt;                   ///< the host's copy of the melt's settings
    SpectralComb m_comb;                   ///< the host's copy of the spectral comb
    Turbulence   m_turb;                   ///< the host's copy of the energy cascade's settings
    float       *m_dCascadeA =
        nullptr;   ///< device: per voice and harmonic its power against the spectrum's, buffer A (even blocks read it)
    float              *m_dCascadeB = nullptr;      ///< device: buffer B
    float              *m_dFilterTable = nullptr;   ///< device: per voice the transported filter, FilterBins gains
    unsigned long long *m_dHallHist = nullptr;      ///< device: per band the partials' power this block (fixed point)
    float              *m_dHallMemory = nullptr;    ///< device: the hall's memory, per band
    float              *m_dDensity = nullptr;       ///< device: the spectrum's density, per band
    float              *m_dTau = nullptr;           ///< device: per partial its own clock
    float4             *m_dOrbitA = nullptr;        ///< device: the Orbit field's state, buffer A
    float4             *m_dOrbitB = nullptr;        ///< device: buffer B
    float              *m_hHallMemory = nullptr;    ///< pinned host: the memory as the last snapshot brought it down
    float              *m_hHallUp = nullptr;        ///< pinned host: a memory to send up
    bool                m_hallUp = false;           ///< the next launch sends hHallUp up first
    bool                m_hallDown = false;         ///< hHallMemory holds a memory
    float              *m_dMeltHist = nullptr;      ///< device: the melt's input history, two rings of MeltN
    float              *m_dMeltAcc = nullptr;       ///< device: its overlap, two rings of MeltN
    float              *m_dMeltEnergy = nullptr;    ///< device: its held energy, per ear and band
    float              *m_dMeltOmega = nullptr;     ///< device: its bands' sung frequencies
    float              *m_dMeltPhase = nullptr;     ///< device: their sung phases, per ear
    float              *m_dMeltHeard = nullptr;     ///< device: their last heard phases, per ear
    int                *m_dMeltState = nullptr;     ///< device: its history index and the samples since its last frame
    float              *m_dResW = nullptr;          ///< device: the reservoir's coupling, N x N
    float              *m_dResIn = nullptr;         ///< device: its input weights
    float              *m_dResOut = nullptr;        ///< device: its readouts
    float              *m_dResX = nullptr;          ///< device: its state, two buffers
    float2             *m_dLifeA = nullptr;         ///< device: the living field, buffer A
    float2             *m_dLifeB = nullptr;         ///< device: buffer B
    float              *m_dFluidA = nullptr;        ///< device: the Fluid's lattice, nine directions a cell, buffer A
    float              *m_dFluidB = nullptr;        ///< device: buffer B
    float              *m_dIsles = nullptr;         ///< device: the Islands' grid
    float  *m_dLifeStats = nullptr;   ///< device: its mean and mean square, then what was heard (level, centroid, flux)
    float  *m_dListenHist = nullptr;          ///< device: the listening's history, two rings of EchoN
    float  *m_dListenPrev = nullptr;          ///< device: its last frame's magnitudes
    int    *m_dListenState = nullptr;         ///< device: its history index
    float  *m_hLifeStats = nullptr;           ///< pinned host: the same, brought down every block
    float2 *m_hLife = nullptr;                ///< pinned host: the field brought down with a snapshot
    std::vector<float> m_lifeCells;           ///< the field as the partials read it, for the picture
    bool               m_lifeShown = false;   ///< lifeCells holds a field
    LifeField          m_life;                ///< the host's copy of the living landscape's settings
    int                m_lifeParity = 0;      ///< which buffer holds the field before the next block's steps
    float2       *m_dSnap = nullptr;   ///< device: the picture's snapshot, every SnapStride-th partial (Hz, amplitude)
    float2       *m_hSnap = nullptr;   ///< pinned host: the snapshot brought down
    bool          m_snapAsked = false;         ///< the next launch brings it down
    bool          m_snapInFlight = false;      ///< the launched block brings it down
    bool          m_snapReady = false;         ///< a snapshot came down and was not taken yet
    Binaural      m_binaural;                  ///< the host's copy of the binaural placement
    SpectralEcho  m_echo;                      ///< the host's copy of the spectral echo's settings
    EnvShape     *m_dShapes = nullptr;         ///< device: the envelope bank, EnvShapes
    float        *m_dLand = nullptr;           ///< device: the landscape, room for LandMaxColumns * LandMaxRows
    float2       *m_dModal = nullptr;          ///< device: the modal hall's states, ModalModes
    float        *m_dGainRing = nullptr;       ///< device: per voice GainRing levels, for the dispersion
    SpectralField m_field;                     ///< the host's copy of the field and the dispersion
    std::uint32_t m_blockIndex = 0;            ///< blocks launched since open()
    float        *m_dModalScratch = nullptr;   ///< device: per modal CUDA block two channels of sums
    ModalHall     m_modal;                     ///< the host's copy of the modal hall's settings
    float         m_gpuMs = 0.0f;              ///< GPU time of the last finished block
    bool          m_launched = false;          ///< a block is in flight

    /** @brief Bytes of the upload: the header and the voices. @return bytes */
    size_t uploadBytes() const
    {
        return sizeof( BlockHeader ) + sizeof( CloudVoice ) * static_cast<size_t>( m_cfg.m_voices );
    }

    /** @brief Frees whatever was allocated, in any state. */
    void release()
    {
        if ( m_stream ) cudaStreamSynchronize( m_stream );
        if ( m_exec ) cudaGraphExecDestroy( m_exec );
        if ( m_graph ) cudaGraphDestroy( m_graph );
        if ( m_evStart ) cudaEventDestroy( m_evStart );
        if ( m_evStop ) cudaEventDestroy( m_evStop );
        if ( m_stream ) cudaStreamDestroy( m_stream );
        if ( m_hUpload ) cudaFreeHost( m_hUpload );
        if ( m_hOut ) cudaFreeHost( m_hOut );
        if ( m_hSnap ) cudaFreeHost( m_hSnap );
        if ( m_hLifeStats ) cudaFreeHost( m_hLifeStats );
        if ( m_hLife ) cudaFreeHost( m_hLife );
        for ( void *p : { static_cast<void *>( m_dUpload ),      static_cast<void *>( m_dOut ),
                          static_cast<void *>( m_dScratch ),     static_cast<void *>( m_dRun ),
                          static_cast<void *>( m_dAmp ),         static_cast<void *>( m_dGains ),
                          static_cast<void *>( m_dShapes ),      static_cast<void *>( m_dLand ),
                          static_cast<void *>( m_dModal ),       static_cast<void *>( m_dModalScratch ),
                          static_cast<void *>( m_dGainRing ),    static_cast<void *>( m_dAmpQ ),
                          static_cast<void *>( m_dEchoHist ),    static_cast<void *>( m_dEchoAcc ),
                          static_cast<void *>( m_dEchoRing ),    static_cast<void *>( m_dEchoState ),
                          static_cast<void *>( m_dSnap ),        static_cast<void *>( m_dLifeA ),
                          static_cast<void *>( m_dLifeB ),       static_cast<void *>( m_dLifeStats ),
                          static_cast<void *>( m_dSyncTheta ),   static_cast<void *>( m_dSyncOrder ),
                          static_cast<void *>( m_dFlockA ),      static_cast<void *>( m_dFlockB ),
                          static_cast<void *>( m_dListenHist ),  static_cast<void *>( m_dListenPrev ),
                          static_cast<void *>( m_dListenState ), static_cast<void *>( m_dResW ),
                          static_cast<void *>( m_dResIn ),       static_cast<void *>( m_dResOut ),
                          static_cast<void *>( m_dResX ),        static_cast<void *>( m_dMeltHist ),
                          static_cast<void *>( m_dMeltAcc ),     static_cast<void *>( m_dMeltEnergy ),
                          static_cast<void *>( m_dMeltOmega ),   static_cast<void *>( m_dMeltPhase ),
                          static_cast<void *>( m_dMeltHeard ),   static_cast<void *>( m_dMeltState ),
                          static_cast<void *>( m_dSyncX ),       static_cast<void *>( m_dFluidA ),
                          static_cast<void *>( m_dFluidB ),      static_cast<void *>( m_dIsles ),
                          static_cast<void *>( m_dCascadeA ),    static_cast<void *>( m_dCascadeB ),
                          static_cast<void *>( m_dFilterTable ), static_cast<void *>( m_dHallHist ),
                          static_cast<void *>( m_dHallMemory ),  static_cast<void *>( m_dDensity ),
                          static_cast<void *>( m_dTau ),         static_cast<void *>( m_dOrbitA ),
                          static_cast<void *>( m_dOrbitB ) } )
            if ( p ) cudaFree( p );
        m_exec = nullptr;
        m_graph = nullptr;
        m_evStart = m_evStop = nullptr;
        m_stream = nullptr;
        m_hUpload = m_dUpload = nullptr;
        m_hOut = m_dOut = m_dScratch = m_dLand = nullptr;
        m_dRun = nullptr;
        m_dAmp = nullptr;
        m_dGains = nullptr;
        m_dShapes = nullptr;
        m_dModal = nullptr;
        m_dModalScratch = nullptr;
        m_dGainRing = nullptr;
        m_dAmpQ = nullptr;
        m_dEchoHist = nullptr;
        m_dEchoAcc = nullptr;
        m_dEchoRing = nullptr;
        m_dEchoState = nullptr;
        m_dSnap = nullptr;
        m_hSnap = nullptr;
        m_dLifeA = m_dLifeB = nullptr;
        m_dLifeStats = nullptr;
        m_hLifeStats = nullptr;
        m_hLife = nullptr;
        m_dSyncTheta = nullptr;
        m_dSyncOrder = nullptr;
        m_dFlockA = nullptr;
        m_dFlockB = nullptr;
        m_dListenHist = nullptr;
        m_dListenPrev = nullptr;
        m_dListenState = nullptr;
        m_dResW = nullptr;
        m_dResIn = nullptr;
        m_dResOut = nullptr;
        m_dResX = nullptr;
        m_dMeltHist = m_dMeltAcc = m_dMeltEnergy = m_dMeltOmega = m_dMeltPhase = m_dMeltHeard = nullptr;
        m_dMeltState = nullptr;
        m_dSyncX = nullptr;
        m_dFluidA = m_dFluidB = nullptr;
        m_dIsles = nullptr;
        m_dCascadeA = m_dCascadeB = nullptr;
        m_dFilterTable = nullptr;
        if ( m_hHallMemory ) cudaFreeHost( m_hHallMemory );
        if ( m_hHallUp ) cudaFreeHost( m_hHallUp );
        m_dHallHist = nullptr;
        m_dHallMemory = m_dDensity = m_dTau = nullptr;
        m_dOrbitA = m_dOrbitB = nullptr;
        m_hHallMemory = m_hHallUp = nullptr;
    }

    /** @brief Waits for a block in flight, so the tables may be replaced. */
    void settle()
    {
        if ( m_stream ) cudaStreamSynchronize( m_stream );
        if ( m_launched )
        {
            cudaEventElapsedTime( &m_gpuMs, m_evStart, m_evStop );
            m_launched = false;
        }
    }

    /**
     * @brief Fills a voice's derived fields: how its partials divide between the harmonic clouds and
     *        the air band, the harmonics below the cutoff (with the stiff string's stretch), members
     *        per harmonic, and the scales that make the voice's RMS its level -- normalised after the
     *        whole spectral shape, as Noctuary does, so brightness, odd/even, presence and the low cut
     *        colour the voice without changing its level (the filter, the distance and the motions do).
     * @param[in]     i  the voice
     * @param[in,out] v  its upload copy, completed here
     */
    void derive( int i, CloudVoice &v )
    {
        const float fs = static_cast<float>( m_cfg.m_sampleRate );
        const float cutoff = std::min( 20000.0f, 0.43f * fs );
        const int   partials = std::clamp( v.m_partials, 1, m_cfg.m_partialsPerVoice );
        v.m_partials = partials;
        const int airPartials =
            ( v.m_air > 0.0f && partials >= 2 )
                ? std::clamp( static_cast<int>( std::lround( partials * AirShare ) ), 1, partials - 1 )
                : 0;
        const int   HP = partials - airPartials;
        const float hz = std::max( v.m_hz, 1.0f );
        // Harmonics below the cutoff: f0 h^s sqrt(1 + B h^2) + shift < cutoff -- rising in h, so the
        // last one below is found by halving.
        const float s = std::clamp( v.m_stretch, 0.5f, 2.0f );
        const float hzn = std::clamp( v.m_horizon, 0.0f, 1.0f ), R = std::max( v.m_horizonRatio, 1.0f );
        auto        count = [&]( float amount )
        {
            int  lo = 1, hi = HP;
            auto below = [&]( int h )
            { return hz * partialRatio( h, std::max( 0.0f, v.m_inharmonic ), s, amount, R ) + v.m_shift < cutoff; };
            if ( below( hi ) )
                lo = hi;
            else
                while ( hi - lo > 1 )
                {
                    const int mid = ( lo + hi ) / 2;
                    ( below( mid ) ? lo : hi ) = mid;
                }
            return lo;
        };
        int H = count( 0.0f );
        // Bent towards a horizon below the cutoff no harmonic ever reaches it: as many as unbent (at
        // least 48), the rest of them crowded under the horizon.
        if ( hzn > 0.0f ) H = std::min( count( hzn ), std::max( H, 48 ) );
        H = std::clamp( H, 1, HP );
        // Subharmonics: their clouds after the harmonics', as many members each.
        const int S = v.m_subClouds > 0 && HP > SubClouds ? SubClouds : 0;
        if ( S > 0 ) H = std::min( H, HP - S );
        v.m_subClouds = S;
        v.m_harmonics = H;
        v.m_members = std::max( HP / ( H + S ), 1 );
        v.m_harmonicPartials = HP;
        const ShapeKey key{ hz,
                            H,
                            v.m_tilt,
                            v.m_oddEven,
                            v.m_brightness,
                            v.m_inharmonic,
                            v.m_presenceDb,
                            std::clamp( v.m_distance, 0.0f, 1.0f ),
                            v.m_lowCut,
                            s,
                            v.m_shift,
                            hzn,
                            R };
        if ( !( m_shapeKey[i] == key ) )
        {
            double sum = 0.0;
            for ( int h = 1; h <= H; ++h )
            {
                const float  fh = std::max( hz * partialRatio( h, v.m_inharmonic, s, hzn, R ) + v.m_shift, 1.0f );
                const double a = static_cast<double>( harmonicShape( h, v.m_tilt, v.m_oddEven, v.m_brightness ) ) *
                                 presenceGain( fh, v.m_presenceDb, key.m_distance ) * lowCutGain( fh, v.m_lowCut );
                sum += a * a;
            }
            m_powerSum[i] = std::max( sum, 1.0e-30 );
            m_shapeKey[i] = key;
        }
        // Power of the voice: every member a sine (mean square 1/2), the shimmer's two sines add
        // their own mean square (0.7^2 + 0.3^2) / 2 times its depth squared. The envelopes are
        // the sound's own motion and are not compensated.
        const float shimmerMs = 1.0f + 0.29f * v.m_shimmer * v.m_shimmer;
        v.m_scale = static_cast<float>( v.m_level / std::sqrt( 0.5 * v.m_members * m_powerSum[i] * shimmerMs ) );
        // The air band at `air` times the voice's level; its flicker's mean square is 1 + 0.29 * 0.8^2.
        v.m_airScale =
            airPartials > 0
                ? static_cast<float>( v.m_air * v.m_level / std::sqrt( 0.5 * airPartials * ( 1.0 + 0.29 * 0.64 ) ) )
                : 0.0f;
    }
};

CloudEngine::CloudEngine() = default;
CloudEngine::~CloudEngine()
{
    close();
}

bool CloudEngine::open( const CloudConfig &config, std::string *error )
{
    close();
    const int n = config.m_blockSize;
    if ( n < 32 || n > 4096 || ( n & ( n - 1 ) ) != 0 )
    {
        if ( error ) *error = "blockSize must be a power of two, 32 .. 4096";
        return false;
    }
    if ( config.m_voices < 1 || config.m_partialsPerVoice < 1 || config.m_sampleRate < 8000.0 )
    {
        if ( error ) *error = "voices, partials and sample rate must be positive";
        return false;
    }

    auto im = std::make_unique<Impl>();
    im->m_cfg = config;
    im->m_pcap = ( config.m_partialsPerVoice + Chunk - 1 ) / Chunk * Chunk;
    im->m_chunks = config.m_voices * im->m_pcap / Chunk;
    im->m_voices.assign( config.m_voices, CloudVoice{} );
    im->m_shapeKey.assign( config.m_voices, Impl::ShapeKey{} );
    im->m_powerSum.assign( config.m_voices, 1.0 );

    int count = 0;
    if ( !ok( cudaGetDeviceCount( &count ), "cudaGetDeviceCount", error ) ) return false;
    if ( count < 1 )
    {
        if ( error ) *error = "no CUDA device";
        return false;
    }
    // The flags only take before the context exists; a second engine in the process keeps the first's.
    cudaSetDeviceFlags( config.m_spinWait ? cudaDeviceScheduleSpin : cudaDeviceScheduleBlockingSync );
    cudaGetLastError();
    Impl &m = *im;
    bool  good = ok( cudaSetDevice( m.m_device ), "cudaSetDevice", error ) &&
                 ok( cudaGetDeviceProperties( &m.m_props, m.m_device ), "cudaGetDeviceProperties", error );
    // A card this build has no code for (CMAKE_CUDA_ARCHITECTURES) says so here, by name, not at the
    // first block's launch.
    if ( cudaFuncAttributes attr{}; good && cudaFuncGetAttributes( &attr, updateKernel ) != cudaSuccess )
    {
        cudaGetLastError();
        if ( error )
            *error = std::string( m.m_props.name ) + " (compute " + std::to_string( m.m_props.major ) + "." +
                     std::to_string( m.m_props.minor ) + ") is not among the cards this build is made for";
        return false;
    }
    int lo = 0, hi = 0;
    good = good && ok( cudaDeviceGetStreamPriorityRange( &lo, &hi ), "cudaDeviceGetStreamPriorityRange", error ) &&
           ok( cudaStreamCreateWithPriority( &m.m_stream, cudaStreamNonBlocking, hi ), "cudaStreamCreateWithPriority",
               error ) &&
           ok( cudaEventCreate( &m.m_evStart ), "cudaEventCreate", error ) &&
           ok( cudaEventCreateWithFlags( &m.m_evStop, config.m_spinWait ? cudaEventDefault : cudaEventBlockingSync ),
               "cudaEventCreate", error );
    const size_t partialsTotal = static_cast<size_t>( config.m_voices ) * m.m_pcap;
    const size_t landFloats = static_cast<size_t>( LandMaxColumns ) * LandMaxRows;
    good = good &&
           ok( cudaHostAlloc( &m.m_hUpload, m.uploadBytes(), cudaHostAllocDefault ), "cudaHostAlloc upload", error ) &&
           ok( cudaHostAlloc( &m.m_hOut, sizeof( float ) * CloudOutChannels * n, cudaHostAllocDefault ),
               "cudaHostAlloc out", error ) &&
           ok( cudaMalloc( &m.m_dUpload, m.uploadBytes() ), "cudaMalloc upload", error ) &&
           ok( cudaMalloc( &m.m_dOut, sizeof( float ) * CloudOutChannels * n ), "cudaMalloc out", error ) &&
           ok( cudaMalloc( &m.m_dScratch, sizeof( float ) * CloudChannels * n * static_cast<size_t>( m.m_chunks ) ),
               "cudaMalloc scratch", error ) &&
           ok( cudaMalloc( &m.m_dRun, sizeof( uint4 ) * partialsTotal ), "cudaMalloc run", error ) &&
           ok( cudaMalloc( &m.m_dAmp, sizeof( float4 ) * partialsTotal ), "cudaMalloc amp", error ) &&
           ok( cudaMalloc( &m.m_dGains, sizeof( float4 ) * 3 * partialsTotal ), "cudaMalloc gains", error ) &&
           ok( cudaMalloc( &m.m_dAmpQ, sizeof( float2 ) * partialsTotal ), "cudaMalloc ampQ", error ) &&
           ok( cudaMalloc( &m.m_dEchoHist, sizeof( float ) * 2 * EchoN ), "cudaMalloc echo history", error ) &&
           ok( cudaMalloc( &m.m_dEchoAcc, sizeof( float ) * 2 * EchoN ), "cudaMalloc echo overlap", error ) &&
           ok( cudaMalloc( &m.m_dEchoRing, sizeof( float2 ) * 2 * EchoBins * static_cast<size_t>( EchoSlots ) ),
               "cudaMalloc echo lines", error ) &&
           ok( cudaMalloc( &m.m_dEchoState, sizeof( int ) * 2 ), "cudaMalloc echo state", error ) &&
           ok( cudaMalloc( &m.m_dSnap, sizeof( float2 ) * ( partialsTotal / SnapStride ) ), "cudaMalloc snapshot",
               error ) &&
           ok( cudaHostAlloc( &m.m_hSnap, sizeof( float2 ) * ( partialsTotal / SnapStride ), cudaHostAllocDefault ),
               "cudaHostAlloc snapshot", error ) &&
           ok( cudaMalloc( &m.m_dLifeA, sizeof( float2 ) * LifeCells ), "cudaMalloc life", error ) &&
           ok( cudaMalloc( &m.m_dLifeB, sizeof( float2 ) * LifeCells ), "cudaMalloc life", error ) &&
           ok( cudaMalloc( &m.m_dFluidA, sizeof( float ) * 9 * LifeCells ), "cudaMalloc fluid", error ) &&
           ok( cudaMalloc( &m.m_dFluidB, sizeof( float ) * 9 * LifeCells ), "cudaMalloc fluid", error ) &&
           ok( cudaMalloc( &m.m_dIsles, sizeof( float ) * IsleW * IsleH ), "cudaMalloc islands", error ) &&
           ok( cudaMalloc( &m.m_dCascadeA, sizeof( float ) * partialsTotal ), "cudaMalloc cascade", error ) &&
           ok( cudaMalloc( &m.m_dCascadeB, sizeof( float ) * partialsTotal ), "cudaMalloc cascade", error ) &&
           ok( cudaMalloc( &m.m_dFilterTable, sizeof( float ) * FilterBins * static_cast<size_t>( config.m_voices ) ),
               "cudaMalloc filter table", error ) &&
           ok( cudaMalloc( &m.m_dHallHist, sizeof( unsigned long long ) * HallBins ), "cudaMalloc hall", error ) &&
           ok( cudaMalloc( &m.m_dHallMemory, sizeof( float ) * HallBins ), "cudaMalloc hall", error ) &&
           ok( cudaMalloc( &m.m_dDensity, sizeof( float ) * HallBins ), "cudaMalloc density", error ) &&
           ok( cudaMalloc( &m.m_dTau, sizeof( float ) * partialsTotal ), "cudaMalloc clocks", error ) &&
           ok( cudaMalloc( &m.m_dOrbitA, sizeof( float4 ) * partialsTotal ), "cudaMalloc orbit", error ) &&
           ok( cudaMalloc( &m.m_dOrbitB, sizeof( float4 ) * partialsTotal ), "cudaMalloc orbit", error ) &&
           ok( cudaHostAlloc( &m.m_hHallMemory, sizeof( float ) * HallBins, cudaHostAllocDefault ),
               "cudaHostAlloc hall", error ) &&
           ok( cudaHostAlloc( &m.m_hHallUp, sizeof( float ) * HallBins, cudaHostAllocDefault ), "cudaHostAlloc hall",
               error ) &&
           ok( cudaMalloc( &m.m_dLifeStats, sizeof( float ) * 12 ), "cudaMalloc life stats", error ) &&
           ok( cudaMalloc( &m.m_dResW, sizeof( float ) * ReservoirN * ReservoirN ), "cudaMalloc reservoir", error ) &&
           ok( cudaMalloc( &m.m_dResIn, sizeof( float ) * ReservoirN * ReservoirInputs ), "cudaMalloc reservoir",
               error ) &&
           ok( cudaMalloc( &m.m_dResOut, sizeof( float ) * ReservoirN * ReservoirOutputs ), "cudaMalloc reservoir",
               error ) &&
           ok( cudaMalloc( &m.m_dResX, sizeof( float ) * 2 * ReservoirN ), "cudaMalloc reservoir", error ) &&
           ok( cudaMalloc( &m.m_dMeltHist, sizeof( float ) * 2 * MeltN ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltAcc, sizeof( float ) * 2 * MeltN ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltEnergy, sizeof( float ) * 2 * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltOmega, sizeof( float ) * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltPhase, sizeof( float ) * 2 * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltHeard, sizeof( float ) * 2 * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltState, sizeof( int ) * 2 ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dListenHist, sizeof( float ) * 2 * EchoN ), "cudaMalloc listen", error ) &&
           ok( cudaMalloc( &m.m_dListenPrev, sizeof( float ) * EchoBins ), "cudaMalloc listen", error ) &&
           ok( cudaMalloc( &m.m_dListenState, sizeof( int ) * 2 ), "cudaMalloc listen", error ) &&
           ok( cudaMalloc( &m.m_dSyncTheta, sizeof( float ) * partialsTotal ), "cudaMalloc sync", error ) &&
           ok( cudaMalloc( &m.m_dFlockA, sizeof( float2 ) * partialsTotal ), "cudaMalloc flock", error ) &&
           ok( cudaMalloc( &m.m_dFlockB, sizeof( float2 ) * partialsTotal ), "cudaMalloc flock", error ) &&
           ok( cudaMalloc( &m.m_dSyncOrder, sizeof( float2 ) * 3 * partialsTotal ), "cudaMalloc sync order", error ) &&
           ok( cudaMalloc( &m.m_dSyncX, sizeof( float ) * partialsTotal ), "cudaMalloc sync places", error ) &&
           ok( cudaHostAlloc( &m.m_hLifeStats, sizeof( float ) * 12, cudaHostAllocDefault ), "cudaHostAlloc life stats",
               error ) &&
           ok( cudaHostAlloc( &m.m_hLife, sizeof( float2 ) * LifeCells, cudaHostAllocDefault ), "cudaHostAlloc life",
               error ) &&
           ok( cudaMalloc( &m.m_dShapes, sizeof( EnvShape ) * EnvShapes ), "cudaMalloc shapes", error ) &&
           ok( cudaMalloc( &m.m_dLand, sizeof( float ) * landFloats ), "cudaMalloc landscape", error ) &&
           ok( cudaMalloc( &m.m_dModal, sizeof( float2 ) * ModalModes ), "cudaMalloc modal", error ) &&
           ok( cudaMalloc( &m.m_dModalScratch, sizeof( float ) * 2 * n * static_cast<size_t>( ModalBlocks ) ),
               "cudaMalloc modal scratch", error ) &&
           ok( cudaMalloc( &m.m_dGainRing, sizeof( float ) * GainRing * static_cast<size_t>( config.m_voices ) ),
               "cudaMalloc gain ring", error );
    // Cleared on our own stream and waited for: the stream does not synchronise with the default
    // stream, so a plain cudaMemset could still be running when the first block's update kernel
    // writes the same tables (it did: one render in three differed).
    good =
        good &&
        ok( cudaMemsetAsync( m.m_dRun, 0, sizeof( uint4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dAmp, 0, sizeof( float4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dGains, 0, sizeof( float4 ) * 3 * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dAmpQ, 0, sizeof( float2 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoHist, 0, sizeof( float ) * 2 * EchoN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoAcc, 0, sizeof( float ) * 2 * EchoN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoRing, 0, sizeof( float2 ) * 2 * EchoBins * static_cast<size_t>( EchoSlots ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoState, 0, sizeof( int ) * 2, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSnap, 0, sizeof( float2 ) * ( partialsTotal / SnapStride ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLifeA, 0, sizeof( float2 ) * LifeCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLifeB, 0, sizeof( float2 ) * LifeCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dFluidA, 0, sizeof( float ) * 9 * LifeCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFluidB, 0, sizeof( float ) * 9 * LifeCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dIsles, 0, sizeof( float ) * IsleW * IsleH, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dCascadeA, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dCascadeB, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFilterTable, 0, sizeof( float ) * FilterBins * static_cast<size_t>( config.m_voices ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dHallHist, 0, sizeof( unsigned long long ) * HallBins, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDensity, 0, sizeof( float ) * HallBins, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dTau, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dOrbitA, 0, sizeof( float4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dOrbitB, 0, sizeof( float4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dLifeStats, 0, sizeof( float ) * 12, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dListenHist, 0, sizeof( float ) * 2 * EchoN, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dListenPrev, 0, sizeof( float ) * EchoBins, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dListenState, 0, sizeof( int ) * 2, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSyncTheta, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFlockA, 0, sizeof( float2 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFlockB, 0, sizeof( float2 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dSyncOrder, 0, sizeof( float2 ) * 3 * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dSyncX, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dShapes, 0, sizeof( EnvShape ) * EnvShapes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLand, 0, sizeof( float ) * landFloats, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dModal, 0, sizeof( float2 ) * ModalModes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dGainRing, 0, sizeof( float ) * GainRing * static_cast<size_t>( config.m_voices ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaStreamSynchronize( m.m_stream ), "cudaStreamSynchronize", error );
    if ( good )
    {
        // Lenia's ring kernel: a bell around half the radius, normalised (Noctuary's, on a larger torus).
        constexpr int R = LifeRadius, K = 2 * LifeRadius + 1;
        float         kernel[K * K];
        float         sum = 0.0f;
        for ( int dy = -R; dy <= R; ++dy )
            for ( int dx = -R; dx <= R; ++dx )
            {
                const float r = std::sqrt( static_cast<float>( dx * dx + dy * dy ) ) / static_cast<float>( R );
                const float d = ( r - 0.5f ) / 0.15f;
                const float k = r <= 1.0f ? std::exp( -0.5f * d * d ) : 0.0f;
                kernel[( dy + R ) * K + ( dx + R )] = k;
                sum += k;
            }
        for ( float &k : kernel ) k /= sum;
        good = ok( cudaMemcpyToSymbol( cLeniaKernel, kernel, sizeof( kernel ) ), "cudaMemcpyToSymbol lenia", error );
        m.m_lifeCells.assign( static_cast<size_t>( LifeCells ), 0.0f );
        reservoirInitKernel<<<static_cast<unsigned>( ( static_cast<size_t>( ReservoirN ) * ReservoirN + 255 ) / 256 ),
                              256, 0, m.m_stream>>>( m.m_dResW, m.m_dResIn, m.m_dResOut, m.m_dResX );
        good = ok( cudaStreamSynchronize( m.m_stream ), "reservoir init", error );
        std::memset( m.m_hLifeStats, 0, sizeof( float ) * 12 );
        m.m_melt.m_clear = 1;                                         // the first block fills the melt's tables
        for ( int b = 0; b < HallBins; ++b ) m.m_hHallUp[b] = 1.0f;   // the hall new: every band at the average
        good =
            good && ok( cudaMemcpy( m.m_dHallMemory, m.m_hHallUp, sizeof( float ) * HallBins, cudaMemcpyHostToDevice ),
                        "cudaMemcpy hall", error );
    }
    if ( good )
    {
        std::memset( m.m_hUpload, 0, m.uploadBytes() );
        std::memset( m.m_hOut, 0, sizeof( float ) * CloudOutChannels * n );
        // The block as one graph: voices up, the levels' ring, three kernels for the voices, two for
        // the modal hall and one for the spectral echo (which hear the voices' block on the card),
        // the block (eight channels) down.
        const auto *hdr = reinterpret_cast<const BlockHeader *>( m.m_dUpload );
        const auto *voices = reinterpret_cast<const CloudVoice *>( m.m_dUpload + sizeof( BlockHeader ) );
        good = ok( cudaStreamBeginCapture( m.m_stream, cudaStreamCaptureModeThreadLocal ), "cudaStreamBeginCapture",
                   error );
        if ( good )
        {
            cudaMemcpyAsync( m.m_dUpload, m.m_hUpload, m.uploadBytes(), cudaMemcpyHostToDevice, m.m_stream );
            constexpr unsigned lifeGrid = ( LifeCells + 255 ) / 256;
            lifeSeedKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, m.m_dLifeA, m.m_dLifeB );
            lifeStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 0, m.m_dLifeA, m.m_dLifeB, m.m_dLand );
            lifeStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 1, m.m_dLifeA, m.m_dLifeB, m.m_dLand );
            fluidSeedKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, m.m_dFluidA, m.m_dFluidB );
            fluidStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 0, m.m_dFluidA, m.m_dFluidB, m.m_dLifeA,
                                                               m.m_dLifeB );
            fluidStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 1, m.m_dFluidA, m.m_dFluidB, m.m_dLifeA,
                                                               m.m_dLifeB );
            islandsKernel<<<1, 1024, 0, m.m_stream>>>( hdr, m.m_dIsles, m.m_dLifeA, m.m_dLifeB );
            lifeStatsKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dLifeA, m.m_dLifeB, m.m_dLifeStats );
            gainRingKernel<<<static_cast<unsigned>( config.m_voices ), 256, 0, m.m_stream>>>( hdr, voices,
                                                                                              m.m_dGainRing );
            cascadeKernel<<<static_cast<unsigned>( ( partialsTotal + Threads - 1 ) / Threads ), Threads, 0,
                            m.m_stream>>>( hdr, voices, m.m_pcap, m.m_dCascadeA, m.m_dCascadeB );
            filterTransportKernel<<<static_cast<unsigned>( config.m_voices ), 256, 0, m.m_stream>>>( hdr, voices,
                                                                                                     m.m_dFilterTable );
            syncOrderKernel<<<static_cast<unsigned>( ( partialsTotal + Threads - 1 ) / Threads ), Threads, 0,
                              m.m_stream>>>( hdr, voices, m.m_pcap, m.m_dSyncTheta, m.m_dSyncX, m.m_dSyncOrder,
                                             partialsTotal );
            updateKernel<<<static_cast<unsigned>( ( partialsTotal + Threads - 1 ) / Threads ), Threads, 0,
                           m.m_stream>>>( hdr, voices, m.m_pcap, m.m_dShapes, m.m_dLand, m.m_dRun, m.m_dAmp, m.m_dGains,
                                          m.m_dGainRing, m.m_dAmpQ, m.m_dSnap, m.m_dLifeA, m.m_dLifeB, m.m_dSyncTheta,
                                          m.m_dSyncX, m.m_dSyncOrder, partialsTotal, m.m_dFlockA, m.m_dFlockB,
                                          m.m_dCascadeA, m.m_dCascadeB, m.m_dFilterTable, m.m_dHallHist, m.m_dDensity,
                                          m.m_dTau, m.m_dOrbitA, m.m_dOrbitB );
            hallKernel<<<1, HallBins, 0, m.m_stream>>>( hdr, m.m_dHallHist, m.m_dHallMemory, m.m_dDensity );
            renderKernel<<<static_cast<unsigned>( m.m_chunks ), Threads, 0, m.m_stream>>>(
                hdr, voices, m.m_pcap, m.m_dRun, m.m_dGains, m.m_dScratch, n );
            reduceKernel<<<static_cast<unsigned>( CloudChannels * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ),
                           0, m.m_stream>>>( hdr, m.m_dScratch, m.m_chunks, n, m.m_dOut );
            modalKernel<<<static_cast<unsigned>( ModalBlocks ), ModalThreads, 0, m.m_stream>>>(
                hdr, m.m_dOut, m.m_dModal, m.m_dModalScratch, n, m.m_dLand, m.m_dHallMemory );
            modalReduceKernel<<<static_cast<unsigned>( 2 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dModalScratch, n, m.m_dOut );
            spectralEchoKernel<<<1, EchoThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dEchoHist, m.m_dEchoAcc,
                                                                   m.m_dEchoRing, m.m_dEchoState );
            meltKernel<<<1, MeltThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dMeltHist, m.m_dMeltAcc, m.m_dMeltEnergy,
                                                           m.m_dMeltOmega, m.m_dMeltPhase, m.m_dMeltHeard,
                                                           m.m_dMeltState );
            listenKernel<<<1, EchoThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dListenHist, m.m_dListenPrev,
                                                             m.m_dListenState, m.m_dLifeStats );
            reservoirStepKernel<<<static_cast<unsigned>( ReservoirN * 32 / 256 ), 256, 0, m.m_stream>>>(
                hdr, m.m_dResW, m.m_dResIn, m.m_dResX );
            reservoirReadKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dResOut, m.m_dResX, m.m_dLifeStats );
            cudaMemcpyAsync( m.m_hLifeStats, m.m_dLifeStats, sizeof( float ) * 12, cudaMemcpyDeviceToHost, m.m_stream );
            cudaMemcpyAsync( m.m_hOut, m.m_dOut, sizeof( float ) * CloudOutChannels * n, cudaMemcpyDeviceToHost,
                             m.m_stream );
            good = ok( cudaStreamEndCapture( m.m_stream, &m.m_graph ), "cudaStreamEndCapture", error ) &&
                   ok( cudaGraphInstantiate( &m.m_exec, m.m_graph, 0 ), "cudaGraphInstantiate", error );
        }
    }
    if ( !good )
    {
        m.release();
        return false;
    }
    m_impl = std::move( im );
    return true;
}

void CloudEngine::close()
{
    if ( !m_impl ) return;
    m_impl->release();
    m_impl.reset();
}

const CloudConfig &CloudEngine::config() const
{
    return m_impl->m_cfg;
}

bool CloudEngine::setShapes( const EnvShape *shapes, int count )
{
    Impl &m = *m_impl;
    count = std::clamp( count, 0, EnvShapes );
    m.settle();
    if ( count > 0 && cudaMemcpyAsync( m.m_dShapes, shapes, sizeof( EnvShape ) * count, cudaMemcpyHostToDevice,
                                       m.m_stream ) != cudaSuccess )
        return false;
    if ( cudaStreamSynchronize( m.m_stream ) != cudaSuccess ) return false;
    m.m_shapeCount = count;
    return true;
}

bool CloudEngine::setLandscape( const float *amps, int columns, int rows, float rowSeconds )
{
    Impl &m = *m_impl;
    if ( columns < 2 || columns > LandMaxColumns || rows < 1 || rows > LandMaxRows || !( rowSeconds > 0.0f ) || !amps )
        return false;
    m.settle();
    if ( cudaMemcpyAsync( m.m_dLand, amps, sizeof( float ) * static_cast<size_t>( columns ) * rows,
                          cudaMemcpyHostToDevice, m.m_stream ) != cudaSuccess )
        return false;
    if ( cudaStreamSynchronize( m.m_stream ) != cudaSuccess ) return false;
    m.m_landColumns = columns;
    m.m_landRows = rows;
    m.m_landRowSeconds = rowSeconds;
    return true;
}

CloudVoice &CloudEngine::voice( int i )
{
    return m_impl->m_voices[std::clamp( i, 0, m_impl->m_cfg.m_voices - 1 )];
}

ModalHall &CloudEngine::modal()
{
    return m_impl->m_modal;
}

SpectralField &CloudEngine::field()
{
    return m_impl->m_field;
}

Binaural &CloudEngine::binaural()
{
    return m_impl->m_binaural;
}

SpectralEcho &CloudEngine::spectralEcho()
{
    return m_impl->m_echo;
}

void CloudEngine::requestSnapshot()
{
    m_impl->m_snapAsked = true;
}

LifeField &CloudEngine::life()
{
    return m_impl->m_life;
}

PartialSync &CloudEngine::sync()
{
    return m_impl->m_sync;
}

Ripple &CloudEngine::ripple()
{
    return m_impl->m_ripple;
}

Reservoir &CloudEngine::reservoir()
{
    return m_impl->m_reservoir;
}

Melt &CloudEngine::melt()
{
    return m_impl->m_melt;
}

Turbulence &CloudEngine::turbulence()
{
    return m_impl->m_turb;
}

bool CloudEngine::hallMemory( float *out ) const
{
    if ( !m_impl || !m_impl->m_hallDown ) return false;
    std::memcpy( out, m_impl->m_hHallMemory, sizeof( float ) * HallBins );
    return true;
}

void CloudEngine::setHallMemory( const float *in )
{
    if ( !m_impl ) return;
    std::memcpy( m_impl->m_hHallUp, in, sizeof( float ) * HallBins );
    m_impl->m_hallUp = true;
}

SpectralComb &CloudEngine::comb()
{
    return m_impl->m_comb;
}

void CloudEngine::reservoirOut( float *out ) const
{
    for ( int k = 0; k < ReservoirOutputs; ++k ) out[k] = m_impl->m_hLifeStats[5 + k];
}

void CloudEngine::lifeStats( float &mean, float &meanSq ) const
{
    mean = m_impl->m_hLifeStats[0];
    meanSq = m_impl->m_hLifeStats[1];
}

void CloudEngine::heard( float &levelDb, float &centroid, float &flux ) const
{
    levelDb = m_impl->m_hLifeStats[2];
    centroid = m_impl->m_hLifeStats[3];
    flux = m_impl->m_hLifeStats[4];
}

const float *CloudEngine::lifePicture() const
{
    return m_impl->m_lifeShown ? m_impl->m_lifeCells.data() : nullptr;
}

bool CloudEngine::takeSnapshot( const float *&dots, int &perVoice )
{
    Impl &m = *m_impl;
    if ( !m.m_snapReady ) return false;
    m.m_snapReady = false;
    dots = reinterpret_cast<const float *>( m.m_hSnap );
    perVoice = m.m_pcap / SnapStride;
    return true;
}

void CloudEngine::launch()
{
    Impl     &m = *m_impl;
    const int n = m.m_cfg.m_blockSize;
    auto     *hdr = reinterpret_cast<BlockHeader *>( m.m_hUpload );
    hdr->m_n = n;
    hdr->m_nPrev = m.m_nPrev;
    hdr->m_sampleRate = static_cast<float>( m.m_cfg.m_sampleRate );
    hdr->m_voices = m.m_cfg.m_voices;
    hdr->m_shapeCount = m.m_shapeCount;
    hdr->m_landColumns = m.m_landColumns;
    hdr->m_landRows = m.m_landRows;
    hdr->m_landRowSeconds = m.m_landRowSeconds;
    hdr->m_modal = m.m_modal;
    hdr->m_modal.m_tunedCount = std::clamp( m.m_modal.m_tunedCount, 0, ModalTuned );
    hdr->m_field = m.m_field;
    const bool pitches = m.m_field.m_mode == FieldModeE::SCALE ||
                         m.m_field.m_mode == FieldModeE::ORBIT;   // the scale's pitches, not attractors
    hdr->m_field.m_count = std::clamp( m.m_field.m_count, 0, pitches ? FieldPoints : FieldAttractors );
    hdr->m_field.m_dispersion = std::clamp( m.m_field.m_dispersion, -2.0f, 2.0f );
    hdr->m_blockIndex = m.m_blockIndex++;
    hdr->m_binaural = m.m_binaural;
    hdr->m_echo = m.m_echo;
    m.m_echo.m_clear = 0;   // asked for once
    hdr->m_sync = m.m_sync;
    hdr->m_ripple = m.m_ripple;
    hdr->m_reservoir = m.m_reservoir;
    hdr->m_melt = m.m_melt;
    hdr->m_comb = m.m_comb;
    hdr->m_turb = m.m_turb;
    m.m_melt.m_clear = 0;   // asked for once
    hdr->m_life = m.m_life;
    hdr->m_life.m_steps = m.m_life.m_mode == LifeModeE::OFF ? 0 : std::clamp( m.m_life.m_steps, 0, 2 );
    hdr->m_lifeParity = m.m_lifeParity;
    m.m_lifeParity = ( m.m_lifeParity + hdr->m_life.m_steps ) & 1;   // where the field is after this block
    m.m_life.m_reseed = 0;                                           // asked for once
    auto       *up = reinterpret_cast<CloudVoice *>( m.m_hUpload + sizeof( BlockHeader ) );
    const float dt = static_cast<float>( n / m.m_cfg.m_sampleRate );
    for ( int i = 0; i < m.m_cfg.m_voices; ++i )
    {
        CloudVoice &v = m.m_voices[i];
        if ( v.m_restart ) v.m_time = 0.0f;
        up[i] = v;
        if ( v.m_active )
        {
            m.derive( i, up[i] );
            v.m_harmonics = up[i].m_harmonics;   // back to the host's copy, for logs and pictures
            v.m_members = up[i].m_members;
            v.m_scale = up[i].m_scale;
        }
        v.m_restart = 0;
        if ( v.m_active ) v.m_time += dt;
    }
    if ( m.m_hallUp )
    {   // a memory loaded (or a new hall): up before the block
        cudaMemcpyAsync( m.m_dHallMemory, m.m_hHallUp, sizeof( float ) * HallBins, cudaMemcpyHostToDevice, m.m_stream );
        m.m_hallUp = false;
    }
    cudaEventRecord( m.m_evStart, m.m_stream );
    cudaGraphLaunch( m.m_exec, m.m_stream );
    m.m_snapInFlight = m.m_snapAsked;
    if ( m.m_snapAsked )
    {   // the picture's snapshot behind the block, before the event wait() waits for
        cudaMemcpyAsync( m.m_hSnap, m.m_dSnap,
                         sizeof( float2 ) * ( static_cast<size_t>( m.m_cfg.m_voices ) * m.m_pcap / SnapStride ),
                         cudaMemcpyDeviceToHost, m.m_stream );
        cudaMemcpyAsync( m.m_hHallMemory, m.m_dHallMemory, sizeof( float ) * HallBins, cudaMemcpyDeviceToHost,
                         m.m_stream );             // and the hall's memory
        if ( m.m_life.m_mode != LifeModeE::OFF )   // and the living field, as it stands after this block's steps
            cudaMemcpyAsync( m.m_hLife, m.m_lifeParity ? m.m_dLifeB : m.m_dLifeA, sizeof( float2 ) * LifeCells,
                             cudaMemcpyDeviceToHost, m.m_stream );
        m.m_snapAsked = false;
    }
    cudaEventRecord( m.m_evStop, m.m_stream );
    m.m_nPrev = n;
    m.m_launched = true;
}

bool CloudEngine::ready() const
{
    return !m_impl->m_launched || cudaEventQuery( m_impl->m_evStop ) != cudaErrorNotReady;
}

const float *CloudEngine::wait()
{
    Impl &m = *m_impl;
    if ( m.m_launched )
    {
        if ( m.m_cfg.m_spinWait )
        {
            while ( cudaEventQuery( m.m_evStop ) == cudaErrorNotReady ) _mm_pause();
        }
        else
        {
            cudaEventSynchronize( m.m_evStop );
        }
        cudaEventElapsedTime( &m.m_gpuMs, m.m_evStart, m.m_evStop );
        m.m_launched = false;
        if ( m.m_snapInFlight )
        {
            m.m_snapReady = true;
            m.m_hallDown = true;
            m.m_snapInFlight = false;
            m.m_lifeShown = m.m_life.m_mode != LifeModeE::OFF;
            if ( m.m_lifeShown )
                for ( int i = 0; i < LifeCells; ++i )
                    m.m_lifeCells[static_cast<size_t>( i )] =
                        m.m_life.m_mode == LifeModeE::TURING ? std::min( m.m_hLife[i].y * 3.0f, 1.0f ) : m.m_hLife[i].x;
        }
    }
    return m.m_hOut;
}

float CloudEngine::lastGpuMs() const
{
    return m_impl->m_gpuMs;
}

std::string CloudEngine::deviceName() const
{
    const cudaDeviceProp &p = m_impl->m_props;
    return std::string( p.name ) + " (sm_" + std::to_string( p.major ) + std::to_string( p.minor ) + ")";
}

}   // namespace additiv
