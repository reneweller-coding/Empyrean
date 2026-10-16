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
#include <array>
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
constexpr int ModalThreads = 256;   ///< threads per CUDA block of the modal kernel
constexpr int ModalPer = 4;         ///< modes a thread rings (their sum goes through the warp's reduction once)
constexpr int ModalBlocks = ModalModes / ( ModalThreads * ModalPer );   ///< CUDA blocks of the modal kernel
constexpr int ModalTile = 128;                                          ///< samples the modal kernel sums per pass
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
constexpr int      PhysCells = PhysW * PhysH;                  ///< cells of the physarum's trail (Phase 28)
constexpr float    PhysCellsPerOctave = PhysW / 9.96578428f;   ///< its trail cells an octave (log2 1000)
/** @brief The halo's sites: across and from front to back, -HaloRadius .. HaloRadius (filled at open()). */
__constant__ signed char cHaloXY[HaloSites][2];
/** @brief Each site's neighbours: +x, -x, +y, -y (-1: the rim). */
__constant__ short cHaloNbr[HaloSites][4];

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
    std::int32_t  m_phantomVoice;     ///< the GPU voice of the phantom layer (its chunks reduced apart), -1 none
    std::int32_t  m_dreamVoice;       ///< the GPU voice of the dream, -1 none
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
    ChimeField    m_chimes;           ///< the chime field for this block (Phase 27)
    Dream         m_dream;            ///< the dream for this block (Phase 27)
    Weather       m_weather;          ///< the weather for this block (Phase 28)
    Bloom         m_bloom;            ///< the bloom for this block (Phase 28)
    SwarmSpace    m_swarm;            ///< the swarmalators for this block (Phase 28)
    Physarum      m_phys;             ///< the physarum for this block (Phase 28)
    Halo          m_halo;             ///< the topological halo for this block (Phase 29)
    WaveRoom      m_room;             ///< the wave room for this block (Phase 29)

    std::int32_t m_lifeParity;   ///< which of the two field buffers holds the field before this block's steps
    std::int32_t m_lpad0;        ///< keeps the header 16-byte aligned
    std::int32_t m_lpad1;        ///< likewise
    std::int32_t m_lpad2;        ///< likewise
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
 * @param[in,out] dreamAcc per band what the dream gathers for its next memory (Phase 27)
 */
__global__ void __launch_bounds__( HallBins )
    hallKernel( const BlockHeader *hdr, unsigned long long *hist, float *memory, float *density, float *dreamAcc )
{
    const ModalHall &mh = hdr->m_modal;
    if ( !mh.m_listen ) return;
    __shared__ float e[HallBins];
    __shared__ float red[HallBins];
    const int        b = static_cast<int>( threadIdx.x );
    const float      h = static_cast<float>( hist[b] ) * 1.0e-12f;
    hist[b] = 0ULL;
    if ( hdr->m_dream.m_on ) dreamAcc[b] += h;   // the dream gathers what was heard (Phase 27)
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
    if ( !cv.m_active || cv.m_kind != VoiceKindE::NOTE ) return;
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
 * @brief One block of a physarum agent (Phase 28; Jones 2010): where its partial is on the trail -- its pitch with
 *        the field's offset across, its place with its own offset from near the left to the right --, it senses the
 *        trail ahead, ahead left and ahead right (0.7 rad), turns towards the most (a turn of 45 degrees a cell
 *        moved; at random either way where both sides beat ahead), moves, bounces off the trail's edges and leaves
 *        trail. Its offsets, drawn back by the spring, are what the partial plays.
 * @param[in]     P        the physarum
 * @param[in]     trail    the trail this block (physTrailKernel's)
 * @param[in]     prev     per partial the agent's state from the last block (heading, place offset, -, -)
 * @param[out]    next     per partial its state for the next
 * @param[in,out] deposit  per cell what the agents leave (256 a visit)
 * @param[in]     me       the partial's index in the tables
 * @param[in]     logHome  log2 of the partial's own frequency
 * @param[in]     off      the field's offset, cents
 * @param[in,out] place    the partial's place, -1 .. 1: moved by the agent's offset
 * @param[in]     seed     the voice's seed
 * @param[in]     up       the partial
 * @param[in]     block    the block's index (the dice for a turn)
 * @param[in]     restart  a new note: the agent starts where its partial is
 * @param[in]     dt       the block's length, s
 * @return the new offset, cents
 */
__device__ float physStep( const Physarum &P, const float *trail, const float4 *prev, float4 *next,
                           unsigned int *deposit, size_t me, float logHome, float off, float &place, std::uint32_t seed,
                           std::uint32_t up, std::uint32_t block, bool restart, float dt )
{
    const bool fresh = restart || P.m_fresh;
    float4     s = fresh ? make_float4( 6.2831853f * hash01( seed, up, 91 ), 0.0f, 0.0f, 0.0f ) : prev[me];
    float      h = s.x;
    float      cx = ( logHome + off * ( 1.0f / 1200.0f ) - 4.32192809f ) * PhysCellsPerOctave;   // log2(20)
    float      cy = ( fminf( fmaxf( place + s.y, -1.0f ), 1.0f ) + 1.0f ) * ( 0.5f * PhysH );
    auto       sense = [&]( float a )
    {
        float sn, cs;
        __sincosf( a, &sn, &cs );
        const int x = min( max( static_cast<int>( cx + P.m_sensor * cs ), 0 ), PhysW - 1 );
        const int y = min( max( static_cast<int>( cy + P.m_sensor * sn ), 0 ), PhysH - 1 );
        return trail[y * PhysW + x];
    };
    const float SA = 0.7f, RA = fminf( 0.785f * P.m_speed, 0.785f );
    const float f = sense( h ), l = sense( h + SA ), r = sense( h - SA );
    if ( f >= l && f >= r )
    {
    }
    else if ( f < l && f < r )
        h += hash01( seed ^ ( block * 0x9E3779B9U ), up, 92 ) < 0.5f ? RA : -RA;
    else if ( l > r )
        h += RA;
    else
        h -= RA;
    float sn, cs;
    __sincosf( h, &sn, &cs );
    cx += P.m_speed * cs;
    cy += P.m_speed * sn;
    if ( cy < 0.0f )
    {
        cy = -cy;
        h = -h;
    }
    else if ( cy > static_cast<float>( PhysH ) )
    {
        cy = 2.0f * static_cast<float>( PhysH ) - cy;
        h = -h;
    }
    if ( cx < 0.0f )
    {
        cx = -cx;
        h = 3.1415927f - h;
    }
    else if ( cx > static_cast<float>( PhysW ) )
    {
        cx = 2.0f * static_cast<float>( PhysW ) - cx;
        h = 3.1415927f - h;
    }
    h -= 6.2831853f * floorf( h * 0.15915494f );
    atomicAdd( deposit + min( max( static_cast<int>( cy ), 0 ), PhysH - 1 ) * PhysW +
                   min( max( static_cast<int>( cx ), 0 ), PhysW - 1 ),
               256U );
    const float keep = fmaxf( 1.0f - P.m_spring * dt, 0.0f );
    const float newOff = ( cx / PhysCellsPerOctave + 4.32192809f - logHome ) * 1200.0f * keep;
    const float py = ( cy * ( 2.0f / static_cast<float>( PhysH ) ) - 1.0f - place ) * keep;
    next[me] = make_float4( h, py, 0.0f, 0.0f );
    place = fminf( fmaxf( place + py, -1.0f ), 1.0f );
    return newOff;
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
 * @param[in]     dreamState   the dream's spectrum, DreamBands (dreamRecallKernel), for a voice of the kind DREAM
 * @param[in]     swarmView    the swarm as the partials ride it (swarmViewKernel), SwarmN
 * @param[in]     physTrail    the physarum's trail, both buffers (physTrailKernel wrote this block's)
  * @param[in,out] physDeposit  per trail cell what the physarum's agents leave this block
 * @param[in]     haloState    the halo's lattices (haloKernel), HaloCarriers x HaloComponents

 */
__global__ void updateKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const EnvShape *shapes,
                              const float *land, uint4 *run, float4 *amp, float4 *gains, const float *ring,
                              float2 *ampQ, float2 *snap, const float2 *lifeA, const float2 *lifeB, float *theta,
                              float *sx, const float2 *order, size_t plane, float2 *flockA, float2 *flockB,
                              const float *qa, const float *qb, const float *filterTable, unsigned long long *hallHist,
                              const float *density, float *tau, float4 *orbitA, float4 *orbitB, const float *dreamState,
                              const float4 *swarmView, const float *physTrail, unsigned int *physDeposit,
                              const float2 *haloState )
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
    float      phantomTurn = 0.0f;          // the phantom layer: the turn of phase from the left ear to the right
    float      planeDist = cv.m_distance;   // the partial's distance for its planes (the swarm moves it)
    float2     haloAmp = make_float2( 0.0f, 0.0f );   // the halo (Phase 29): the site's complex amplitude

    if ( cv.m_kind == VoiceKindE::HALO )
    {
        // The topological halo (Phase 29): a partial per site and carrier, the carrier's pitch, the site's complex
        // amplitude from its lattice (its two orbitals together), placed where the site is round the listener.
        const int c = p / HaloSites, s = p - c * HaloSites;
        f = c < HaloCarriers ? cv.m_haloHz[c] : 0.0f;
        if ( f > 1.0f )
        {
            const float2 a0 = haloState[static_cast<size_t>( c ) * HaloComponents + 2 * s];
            const float2 a1 = haloState[static_cast<size_t>( c ) * HaloComponents + 2 * s + 1];
            haloAmp = make_float2( a0.x + a1.x, a0.y + a1.y );
            place = static_cast<float>( cHaloXY[s][0] ) * ( 1.0f / HaloRadius );
            planeDist = 0.35f + 0.3f * static_cast<float>( cHaloXY[s][1] ) * ( 1.0f / HaloRadius );
            const float fade = fminf( fmaxf( ( 0.47f * fs - f ) / ( 0.04f * fs ), 0.0f ), 1.0f );
            const float k = cv.m_scale * cv.m_gain * fade * exp2f( -planeDist );
            haloAmp.x *= k;
            haloAmp.y *= k;
            target = sqrtf( haloAmp.x * haloAmp.x + haloAmp.y * haloAmp.y );
        }
        f = fmaxf( f, 1.0f );
    }
    else if ( cv.m_kind != VoiceKindE::NOTE )
    {
        // The layers that are not notes (Phase 27). Their partials are heard as they are: no filter, no motion.
        float centre, drift, sh;
        if ( cv.m_kind == VoiceKindE::PHANTOM )
        {
            // Noise of partials from 150 Hz to 3 kHz, even per octave, flickering as Air does -- the same in both
            // ears but where a pitch turns the right ear's phase once round over `width` octaves: there the ears
            // disagree, and the brain hears a faint tone that is in neither (Huggins' pitch).
            centre = 150.0f * exp2f( 4.3219281f * hash01( seed, up, 70 ) );   // log2(20)
            const float j = 6.0f;
            drift = 25.0f * ( 0.6f * slowSine( j * ( 0.5f + hash01( seed, up, 2 ) ), t, hash01( seed, up, 3 ) ) +
                              0.4f * slowSine( j * ( 1.3f + hash01( seed, up, 4 ) ), t, hash01( seed, up, 5 ) ) );
            sh = fmaxf(
                0.0f,
                1.0f + 0.6f * ( 0.7f * slowSine( j * ( 0.7f + hash01( seed, up, 6 ) ), t, hash01( seed, up, 7 ) ) +
                                0.3f * slowSine( j * ( 1.6f + hash01( seed, up, 8 ) ), t, hash01( seed, up, 9 ) ) ) );
            place = 0.0f;
        }
        else
        {
            // The dream: its partials spread over the bands, each band as loud as the dream's state there.
            const int b = p % DreamBands;
            centre = 20.0f * exp2f( 9.96578428f * ( static_cast<float>( b ) + hash01( seed, up, 70 ) ) /
                                    static_cast<float>( DreamBands ) );
            drift =
                cv.m_drift * slowSine( cv.m_driftRate * ( 0.5f + hash01( seed, up, 2 ) ), t, hash01( seed, up, 3 ) );
            sh = fmaxf( 0.0f, 1.0f + cv.m_shimmer * slowSine( cv.m_shimmerRate * ( 0.5f + hash01( seed, up, 6 ) ), t,
                                                              hash01( seed, up, 7 ) ) );
            sh *= dreamState[b];
        }
        f = centre * exp2f( drift * ( 1.0f / 1200.0f ) );
        if ( cv.m_kind == VoiceKindE::PHANTOM )
        {
            const float lf = log2f( fmaxf( f, 1.0f ) );
            float       turn = 0.0f;
            for ( int k = 0; k < min( cv.m_phantomCount, PhantomTargets ); ++k )
            {
                const float x = ( lf - log2f( fmaxf( cv.m_phantomHz[k], 1.0f ) ) ) / fmaxf( cv.m_phantomWidth, 0.005f );
                turn += cv.m_phantomWeight[k] * 0.5f * ( 1.0f + tanhf( 2.5f * x ) );
            }
            phantomTurn = 6.2831853f * turn;
        }
        const float fade = fminf( fmaxf( ( 0.47f * fs - f ) / ( 0.04f * fs ), 0.0f ), 1.0f ) *
                           fminf( fmaxf( ( f - 5.0f ) * ( 1.0f / 15.0f ), 0.0f ), 1.0f );
        f = fmaxf( f, 1.0f );
        const float dist = fminf( fmaxf( cv.m_distance, 0.0f ), 1.0f );
        const float x0 = f * ( 1.0f / 16000.0f );
        const float x = x0 / exp2f( -5.0f * dist * cv.m_airAbsorb );
        const float absorb = sqrtf( ( 1.0f + x0 * x0 ) / ( 1.0f + x * x ) );
        target = cv.m_scale * sh * fade * cv.m_gain * exp2f( -dist ) * absorb;
    }
    else if ( isAir || p < ( H + S ) * K )
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
                    if ( ( cv.m_harmonicLife & 8 ) && h <= TalbotHarmonics ) g *= cv.m_talbot[h - 1];
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
        // The swarmalators (Phase 28): the partial rides one -- its place across and from near to far is the
        // swarmalator's, its loudness pulses with its phase (the phase turning on at the rate).
        float swarmDist = 0.0f;
        if ( hdr->m_swarm.m_on && hdr->m_swarm.m_depth > 0.0f )
        {
            const SwarmSpace &W = hdr->m_swarm;
            const float4      sv = swarmView[hash32( seed ^ ( up * 0x9E3779B9U + 0x5A4A0000U ) ) & ( SwarmN - 1 )];
            place = fminf( fmaxf( place + W.m_depth * ( sv.x - place ), -1.0f ), 1.0f );
            swarmDist = 0.5f * W.m_depth * sv.y;
            float heard = sv.z + 6.2831853f * W.m_rate * tp;
            heard -= 6.2831853f * floorf( heard * 0.15915494f );
            env *= fmaxf( 0.0f, 1.0f + W.m_depth * __sinf( heard ) ) * rsqrtf( 1.0f + 0.5f * W.m_depth * W.m_depth );
        }

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
        else if ( hdr->m_field.m_mode == FieldModeE::PHYSARUM )
        {
            // The physarum: a share of the partials crawl over its trail; the others glide home.
            const bool  odd = ( hdr->m_blockIndex & 1U ) != 0;
            const float dtb = static_cast<float>( n ) / fs;
            if ( hdr->m_phys.m_on && hash01( seed, up, 90 ) < hdr->m_phys.m_share )
                fieldOff = physStep( hdr->m_phys, physTrail + ( odd ? 0 : PhysCells ), odd ? orbitB : orbitA,
                                     odd ? orbitA : orbitB, physDeposit, static_cast<size_t>( gid ),
                                     log2f( fmaxf( home, 1.0f ) ), fieldOff, place, seed, up, hdr->m_blockIndex,
                                     cv.m_restart != 0, dtb );
            else
                fieldOff *= __expf( -2.0f * dtb );
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
        const float dist = fminf( fmaxf( cv.m_distance + swarmDist, 0.0f ), 1.0f );
        planeDist = dist;
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
        if ( hdr->m_modal.m_listen && target > 0.0f && cv.m_kind == VoiceKindE::NOTE )
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
        // a plain sine rather than a pulse. The cloud's level assumes they add in power.) The halo's sites of a
        // carrier share theirs: they are one wave, their complex amplitudes its shape.
        ph = cv.m_kind == VoiceKindE::HALO
                 ? hash32( seed ^ ( static_cast<std::uint32_t>( p / HaloSites + 1 ) * 0x9E3779B9U ) )
                 : hash32( seed ^ hash32( up + 0x51ED270BU ) );
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
    __sincosf( 1.57079633f * fminf( fmaxf( planeDist, 0.0f ), 1.0f ), &gf, &gn );
    float4 t4 = make_float4( target * gl * gn, target * gr * gn, target * gl * gf, target * gr * gf );
    float2 tq = make_float2( 0.0f, 0.0f );
    if ( cv.m_kind == VoiceKindE::HALO )
    {
        // The halo: Re(a e^(i phase)) -- the sine part -Im a, the cosine part Re a -- in the near plane, panned.
        t4 = make_float4( -haloAmp.y * gl, -haloAmp.y * gr, 0.0f, 0.0f );
        tq = make_float2( haloAmp.x * gl, haloAmp.x * gr );
    }
    else if ( cv.m_kind == VoiceKindE::PHANTOM )
    {
        // The phantom layer: the left ear hears the partial as it is, the right ear turned by its phase.
        float ts, tc;
        __sincosf( phantomTurn, &ts, &tc );
        t4 = make_float4( target * gl * gn, target * gr * gn * tc, 0.0f, 0.0f );
        tq = make_float2( 0.0f, target * gr * gn * ts );
    }
    else if ( hdr->m_binaural.m_on && target != 0.0f )
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
    if ( dot ) *dot = cv.m_kind == VoiceKindE::PHANTOM ? make_float2( 0.0f, 0.0f ) : make_float2( f, target );
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
    const bool binaural =
        hdr->m_binaural.m_on != 0 || cv.m_kind == VoiceKindE::PHANTOM || cv.m_kind == VoiceKindE::HALO;
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
 * @param[in]     perVoice chunks a voice has (the phantom layer's are left out)
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels
 */
__global__ void reduceKernel( const BlockHeader *hdr, const float *scratch, int chunks, int perVoice, int maxN,
                              float *out )
{
    __shared__ float part[ReduceGroups][ReduceLanes + 1];
    const int        n = hdr->m_n;
    const int        t = blockIdx.x * ReduceLanes + threadIdx.x;   // channel-major, as the scratch is laid out
    const int        c = t / maxN;
    const int        s = t - c * maxN;
    const bool       live = c < CloudChannels && s < n;
    // The phantom layer's chunks are summed apart (phantomReduceKernel).
    const int skip0 = hdr->m_phantomVoice >= 0 ? hdr->m_phantomVoice * perVoice : chunks;
    const int skip1 = hdr->m_phantomVoice >= 0 ? skip0 + perVoice : chunks;
    float     acc = 0.0f;
    if ( live )
        for ( int k = threadIdx.y; k < chunks; k += ReduceGroups )
            if ( k < skip0 || k >= skip1 ) acc += scratch[static_cast<size_t>( k ) * CloudChannels * maxN + t];
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

/** @brief A mode of the modal hall as this block rings it: its pole, its input and its two output phases. */
struct ModalMode
{
    float m_ar;    ///< the pole's real part (decay times the turn a sample)
    float m_ai;    ///< and its imaginary part
    float m_g;     ///< the input's gain (white noise in, the same power out)
    float m_inL;   ///< its share of the left input
    float m_inR;   ///< and of the right
    float m_oLs;   ///< the left ear's phase, sine, times the painted hall's colour
    float m_oLc;   ///< and cosine
    float m_oRs;   ///< the right ear's, sine
    float m_oRc;   ///< and cosine
};

/**
 * @brief Works out where a mode of the modal hall stands and how fast it dies this block (see modalKernel).
 * @param[in] hdr     the block header (its `modal` settings)
 * @param[in] mode    the mode, 0 .. ModalModes - 1
 * @param[in] land    the landscape (the painted hall reads it)
 * @param[in] memory  per band the hall's memory, 1 the average (Grow reads it)
 * @return the mode
 */
__device__ ModalMode modalSetup( const BlockHeader *hdr, int mode, const float *land, const float *memory )
{
    const ModalHall &mh = hdr->m_modal;
    const float      fs = hdr->m_sampleRate;
    const float      lo = fmaxf( mh.m_low, 20.0f );
    const float      hi = fminf( 16000.0f, 0.45f * fs );
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
    ModalMode md;
    md.m_ar = r * cw;
    md.m_ai = r * sw;
    // Driven with sqrt(2 (1 - r^2) / N): white noise in, white noise of the same power out, whatever the decay.
    md.m_g = ( silent || mh.m_freeze )
                 ? 0.0f
                 : sqrtf( 2.0f * fmaxf( 1.0f - r * r, 0.0f ) / static_cast<float>( ModalModes ) ) * weight;
    __sincosf( 1.5707963f * modalUni( mode, 7 ), &md.m_inR, &md.m_inL );   // its share of the left and the right input
    const float phL = 6.2831853f * modalUni( mode, 8 );
    const float phR = phL + mh.m_width * 3.1415927f * ( 2.0f * modalUni( mode, 9 ) - 1.0f );
    __sincosf( phL, &md.m_oLs, &md.m_oLc );
    __sincosf( phR, &md.m_oRs, &md.m_oRc );
    float k = silent ? 0.0f : 1.0f;
    // The painted hall: the mode heard through the landscape at its frequency, as the landscape
    // moves on -- the tail draws the aurora, the rain, the tides. Made up by the landscape's RMS.
    if ( !silent && mh.m_paint > 0.0f && hdr->m_landColumns > 0 && hdr->m_landRows > 0 )
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
        k = 1.0f - mh.m_paint + mh.m_paint * ( top + ( bottom - top ) * fr ) * mh.m_paintNorm;
    }
    md.m_oLs *= k;
    md.m_oLc *= k;
    md.m_oRs *= k;
    md.m_oRc *= k;
    return md;
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
 * @param[in,out] state    per mode its complex state, carried from block to block (a thread rings ModalPer
 *                         modes, ModalThreads apart, and sums them before the warp's reduction)
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
    const int        base = static_cast<int>( blockIdx.x ) * ModalThreads * ModalPer + static_cast<int>( threadIdx.x );
    float           *out = scratch + static_cast<size_t>( blockIdx.x ) * 2 * maxN;
    if ( !mh.m_on )
    {   // resting: silent, and cleared so that it starts from silence when it is opened again
        for ( int k = 0; k < ModalPer; ++k ) state[base + k * ModalThreads] = make_float2( 0.0f, 0.0f );
        for ( int s = threadIdx.x; s < n; s += ModalThreads )
        {
            out[s] = 0.0f;
            out[maxN + s] = 0.0f;
        }
        return;
    }
    ModalMode md[ModalPer];
    float2    y[ModalPer];
    for ( int k = 0; k < ModalPer; ++k )
    {
        md[k] = modalSetup( hdr, base + k * ModalThreads, land, memory );
        y[k] = state[base + k * ModalThreads];
    }
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
            float cL = 0.0f, cR = 0.0f;
#pragma unroll
            for ( int k = 0; k < ModalPer; ++k )
            {
                const float x = md[k].m_g * ( md[k].m_inL * sIn[0][i] + md[k].m_inR * sIn[1][i] );
                const float yr = fmaf( md[k].m_ar, y[k].x, fmaf( -md[k].m_ai, y[k].y, x ) );
                const float yi = fmaf( md[k].m_ar, y[k].y, md[k].m_ai * y[k].x );
                y[k].x = yr;
                y[k].y = yi;
                cL += yr * md[k].m_oLc - yi * md[k].m_oLs;
                cR += yr * md[k].m_oRc - yi * md[k].m_oRs;
            }
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
    for ( int k = 0; k < ModalPer; ++k ) state[base + k * ModalThreads] = y[k];
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

// ---- Phase 27 (16.10.2026): the chime field in its wind, the dream.

constexpr int   WindCells = WindW * WindH;                      ///< cells of the wind's lattice
constexpr float WindTau = 0.53f;                                ///< the wind's relaxation time (its viscosity)
constexpr int   ChimeThreads = 256;                             ///< modes per CUDA block of the chime kernel
constexpr int   ChimeBlocks = ChimeModesTotal / ChimeThreads;   ///< CUDA blocks of the chime kernel
constexpr int   ChimeTile = 128;                                ///< samples the chime kernel sums per pass
constexpr int   ChimesAcross = 64;                              ///< chimes from left to right (32 from near to far)
constexpr int   DreamThreads = 256;                             ///< memories per CUDA block of the dream's likeness
static_assert( ChimeModesTotal % ChimeThreads == 0, "the chimes' modes fill whole CUDA blocks" );
static_assert( DreamMemories % DreamThreads == 0, "the memories fill whole CUDA blocks" );
static_assert( HallBins == 2 * DreamBands, "a dream band is two of the hall's" );

/** @brief D2Q9's directions across. */
__constant__ int cWindX[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
/** @brief And up. */
__constant__ int cWindY[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
/** @brief Each direction's opposite (bounce-back). */
__constant__ int cWindOpp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
/** @brief Their weights. */
__constant__ float cWindW[9] = { 4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f, 1.0f / 9.0f,
                                 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };

/**
 * @brief Whether a cell of the wind is one of the three posts that shed its vortex streets.
 * @param[in] x  across, 0 .. WindW - 1
 * @param[in] y  from near to far, 0 .. WindH - 1
 * @return true inside a post
 */
__device__ __forceinline__ bool windPost( int x, int y )
{
    const int ax = x - 20, ay = y - 12, bx = x - 24, by = y - 34, cx = x - 18, cy = y - 54;
    return ax * ax + ay * ay <= 16 || bx * bx + by * by <= 16 || cx * cx + cy * cy <= 16;
}

/**
 * @brief One step of the wind (Phase 27): a D2Q9 lattice Boltzmann flow (BGK) on the torus, pulled cell by cell from
 *        its neighbours (a post's side bounces back), pushed to the right by the field's force, relaxed towards its
 *        equilibrium. Two steps a block: the first reads buffer A and writes B, the second back into A.
 * @param[in]     hdr   the block header
 * @param[in]     node  0 or 1, the step in the block
 * @param[in,out] a     the lattice, nine directions a cell, buffer A
 * @param[in,out] b     buffer B
 */
__global__ void windStepKernel( const BlockHeader *hdr, int node, float *a, float *b )
{
    const ChimeField &C = hdr->m_chimes;
    if ( !hdr->m_weather.m_wind ) return;   // blowing for the chimes or the weather (Phase 28)
    const int idx = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= WindCells ) return;
    const int    x = idx % WindW, y = idx / WindW;
    const float *src = node == 0 ? a : b;
    float       *dst = node == 0 ? b : a;
    if ( ( hdr->m_weather.m_windFresh && node == 0 ) || windPost( x, y ) )
    {   // at rest (a post's cells stay so)
        for ( int k = 0; k < 9; ++k ) dst[k * WindCells + idx] = cWindW[k];
        return;
    }
    float f[9], rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 9; ++k )
    {
        const int xs = ( x - cWindX[k] ) & ( WindW - 1 ), ys = ( y - cWindY[k] ) & ( WindH - 1 );
        f[k] = windPost( xs, ys ) ? src[cWindOpp[k] * WindCells + idx] : src[k * WindCells + ys * WindW + xs];
        rho += f[k];
        ux += static_cast<float>( cWindX[k] ) * f[k];
        uy += static_cast<float>( cWindY[k] ) * f[k];
    }
    if ( !( rho > 0.2f && rho < 5.0f ) )
    {   // never blown up: a cell gone wrong starts again at rest
        for ( int k = 0; k < 9; ++k ) dst[k * WindCells + idx] = cWindW[k];
        return;
    }
    ux = ux / rho + WindTau * C.m_force / rho;   // the push, as a shift of the equilibrium's velocity
    uy /= rho;
    const float uu = 1.5f * ( ux * ux + uy * uy );
    for ( int k = 0; k < 9; ++k )
    {
        const float cu = 3.0f * ( static_cast<float>( cWindX[k] ) * ux + static_cast<float>( cWindY[k] ) * uy );
        const float eq = cWindW[k] * rho * ( 1.0f + cu + 0.5f * cu * cu - uu );
        dst[k * WindCells + idx] = f[k] - ( f[k] - eq ) * ( 1.0f / WindTau );
    }
}

/**
 * @brief Where a chime hangs, across (0 .. WindW) and from near to far (0 .. WindH): a jittered grid of 64 x 32.
 * @param[in]  c   the chime
 * @param[out] gx  across, cells
 * @param[out] gy  from near to far, cells
 */
__device__ __forceinline__ void chimePlace( int c, float &gx, float &gy )
{
    gx =
        static_cast<float>( ( c % ChimesAcross ) * 2 ) + 2.0f * hash01( 0xC417E5U, static_cast<std::uint32_t>( c ), 1 );
    gy =
        static_cast<float>( ( c / ChimesAcross ) * 2 ) + 2.0f * hash01( 0xC417E5U, static_cast<std::uint32_t>( c ), 2 );
}

/**
 * @brief One block of the chimes' clappers (Phase 27): each a pendulum of its own (0.6 .. 1.4 Hz, lightly damped)
 *        swung by the wind where it hangs -- its sideways part and its gusting part (the wind against the clapper's
 *        own slow average of it) -- striking the chime when it swings past its gap: the strike's strength is the
 *        clapper's speed, its sample where the swing crossed the gap, and the clapper bounces back.
 * @param[in]     hdr      the block header
 * @param[in]     wind     the wind's lattice after this block's steps (buffer A)
 * @param[in,out] state    per chime (angle, angular speed, slow wind, -)
 * @param[out]    impulse  per chime this block's strike (strength, sample), strength 0 when none
 * @param[in,out] meta     the Phase 27 counters: [1] the strikes this block
 */
__global__ void chimeStepKernel( const BlockHeader *hdr, const float *wind, float4 *state, float2 *impulse, int *meta )
{
    const ChimeField &C = hdr->m_chimes;
    const int         c = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( c >= ChimeCount ) return;
    if ( !C.m_on || C.m_reset )
    {
        impulse[c] = make_float2( 0.0f, 0.0f );
        if ( C.m_reset ) state[c] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        return;
    }
    float gx, gy;
    chimePlace( c, gx, gy );
    const int idx = ( static_cast<int>( gy ) & ( WindH - 1 ) ) * WindW + ( static_cast<int>( gx ) & ( WindW - 1 ) );
    float     rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 9; ++k )
    {
        const float f = wind[k * WindCells + idx];
        rho += f;
        ux += static_cast<float>( cWindX[k] ) * f;
        uy += static_cast<float>( cWindY[k] ) * f;
    }
    if ( rho > 1.0e-3f )
    {
        ux /= rho;
        uy /= rho;
    }
    const std::uint32_t uc = static_cast<std::uint32_t>( c );
    const float         dt = static_cast<float>( hdr->m_n ) / hdr->m_sampleRate;
    const float         wp = 6.2831853f * ( 0.6f + 0.8f * hash01( 0xC417E5U, uc, 3 ) );
    const float         gap = 0.15f + 0.15f * hash01( 0xC417E5U, uc, 4 );
    float4              s = state[c];
    s.z += ( ux - s.z ) * fminf( dt * 0.5f, 1.0f );
    const float drive = C.m_kick * ( uy + ( ux - s.z ) );
    s.y += dt * ( -wp * wp * s.x - 2.0f * 0.08f * wp * s.y + drive );
    const float next = s.x + dt * s.y;
    float2      hit = make_float2( 0.0f, 0.0f );
    if ( fabsf( next ) > gap )
    {
        const float before = fabsf( s.x ), after = fabsf( next );
        const float frac = after > before ? fminf( fmaxf( ( gap - before ) / ( after - before ), 0.0f ), 1.0f ) : 0.0f;
        hit = make_float2(
            fminf( fabsf( s.y ) * 2.0f, 1.5f ),
            static_cast<float>( min( static_cast<int>( frac * static_cast<float>( hdr->m_n ) ), hdr->m_n - 1 ) ) );
        s.x = copysignf( gap, next );
        s.y = -0.5f * s.y;
        atomicAdd( meta + 1, 1 );
    }
    else
    {
        s.x = next;
    }
    state[c] = s;
    impulse[c] = hit;
}

/**
 * @brief A chime's mode as a multiple of its pitch, by its kind (ChimeKind): a free bar's for tubes and bamboo,
 *        a glass bowl's, a small bell's (hum, prime, tierce, quint, nominal ..).
 * @param[in] kind  ChimeKindE
 * @param[in] m     the mode, 0 .. 7
 * @return the ratio
 */
__device__ __forceinline__ float chimeRatio( int kind, int m )
{
    switch ( kind )
    {
    case 1:
    {
        const float r[8] = { 1.0f, 2.32f, 4.25f, 6.63f, 9.38f, 12.5f, 16.0f, 19.9f };
        return r[m];
    }
    case 3:
    {
        const float r[8] = { 0.5f, 1.0f, 1.2f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f };
        return r[m];
    }
    default:
    {
        const float r[8] = { 1.0f, 2.756f, 5.404f, 8.933f, 13.34f, 18.64f, 24.81f, 31.87f };
        return r[m];
    }
    }
}

/**
 * @brief The chime field's sound (Phase 27): one thread a mode (ChimeModes a chime: eight, each a doublet a few cents
 *        apart, so a chime beats as a real one does), a complex one-pole as the modal hall's, struck at its chime's
 *        sample by its clapper; heard from where its chime hangs -- across the field, near or far (its share of the
 *        far plane, and the air darkening it). A CUDA block of chimes that all rest writes nothing but silence.
 * @param[in]     hdr      the block header
 * @param[in]     impulse  per chime this block's strike (chimeStepKernel)
 * @param[in,out] state    per mode its complex state
 * @param[out]    scratch  per CUDA block four channels of sums (near L, R, far L, R)
 * @param[in]     maxN     the scratch stride
 */
__global__ void __launch_bounds__( ChimeThreads )
    chimeKernel( const BlockHeader *hdr, const float2 *impulse, float2 *state, float *scratch, int maxN )
{
    __shared__ float  sAcc[ChimeThreads / 32][4][ChimeTile];
    const ChimeField &C = hdr->m_chimes;
    const int         n = hdr->m_n;
    const int         mode = static_cast<int>( blockIdx.x ) * ChimeThreads + static_cast<int>( threadIdx.x );
    const int         c = mode / ChimeModes, k = mode - c * ChimeModes;
    float            *out = scratch + static_cast<size_t>( blockIdx.x ) * 4 * maxN;
    float2            y = ( C.m_on && !C.m_reset ) ? state[mode] : make_float2( 0.0f, 0.0f );
    const float2      hit = C.m_on ? impulse[c] : make_float2( 0.0f, 0.0f );
    const bool        alive = C.m_on && C.m_pitchCount > 0 && ( y.x * y.x + y.y * y.y > 1.0e-14f || hit.x > 0.0f );
    if ( !__syncthreads_or( alive ) )
    {   // the whole block rests: silence, and its modes cleared
        state[mode] = make_float2( 0.0f, 0.0f );
        for ( int s = threadIdx.x; s < n; s += ChimeThreads )
            for ( int ch = 0; ch < 4; ++ch ) out[ch * maxN + s] = 0.0f;
        return;
    }
    const float         fs = hdr->m_sampleRate;
    const std::uint32_t uc = static_cast<std::uint32_t>( c );
    const int           m = k >> 1;
    const float pitch = C.m_pitchCount > 0
                            ? C.m_pitch[hash32( uc ^ 0x5E1E5E1EU ) % static_cast<std::uint32_t>( C.m_pitchCount )]
                            : 0.0f;
    const float cents = 6.0f * ( 2.0f * hash01( 0xC417E5U, uc, 5 ) - 1.0f ) + ( ( k & 1 ) ? 1.5f : -1.5f );
    const float f = pitch * chimeRatio( C.m_kind, m ) * exp2f( cents * ( 1.0f / 1200.0f ) );
    float       T = fmaxf( C.m_decay, 0.05f ) * __powf( fmaxf( chimeRatio( C.m_kind, m ), 0.25f ), -0.6f );
    if ( C.m_kind == 2 ) T *= 0.12f;   // bamboo: wood, a knock that rings briefly
    const float r = __expf( -6.9077553f / ( T * fs ) );
    float       sw, cw;
    sincosf( 6.2831853f * f / fs, &sw, &cw );
    const float ar = r * cw, ai = r * sw;
    float       gx, gy;
    chimePlace( c, gx, gy );
    const float place = 2.0f * gx / static_cast<float>( WindW ) - 1.0f;
    const float dist = gy / static_cast<float>( WindH );
    float       gl, gr, gn, gf;
    __sincosf( 0.78539816f * ( 1.0f + place ), &gr, &gl );
    __sincosf( 1.57079633f * dist, &gf, &gn );
    // The strike's share of the mode (the higher, the less), the air's darkening with distance, the level.
    const float w = ( f > 20.0f && f < 0.45f * fs ) ? 0.5f * __powf( 1.0f + static_cast<float>( m ), -0.8f ) *
                                                          __expf( -dist * f * ( 1.0f / 9000.0f ) ) * C.m_level
                                                    : 0.0f;
    const float gNL = gl * gn * w, gNR = gr * gn * w, gFL = gl * gf * w, gFR = gr * gf * w;
    const int   at = hit.x > 0.0f ? static_cast<int>( hit.y ) : -1;
    const int   lane = static_cast<int>( threadIdx.x ) & 31, warp = static_cast<int>( threadIdx.x ) >> 5;
    for ( int t0 = 0; t0 < n; t0 += ChimeTile )
    {
        const int tn = min( ChimeTile, n - t0 );
        for ( int i = 0; i < tn; ++i )
        {
            const float yr = fmaf( ar, y.x, -ai * y.y );
            const float yi = fmaf( ar, y.y, ai * y.x );
            y.x = yr + ( t0 + i == at ? hit.x : 0.0f );   // struck: a real kick, so the sine starts from nothing
            y.y = yi;
            float v[4] = { yi * gNL, yi * gNR, yi * gFL, yi * gFR };
            for ( int ch = 0; ch < 4; ++ch )
                for ( int off = 16; off > 0; off >>= 1 ) v[ch] += __shfl_down_sync( 0xFFFFFFFFU, v[ch], off );
            if ( lane == 0 )
                for ( int ch = 0; ch < 4; ++ch ) sAcc[warp][ch][i] = v[ch];
        }
        __syncthreads();
        for ( int s = threadIdx.x; s < 4 * tn; s += ChimeThreads )
        {
            const int ch = s / tn, i = s - ch * tn;
            float     a = 0.0f;
            for ( int wv = 0; wv < ChimeThreads / 32; ++wv ) a += sAcc[wv][ch][i];
            out[ch * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    state[mode] = ( fabsf( y.x ) + fabsf( y.y ) > 1.0e-9f ) ? y : make_float2( 0.0f, 0.0f );
}

/**
 * @brief The chime field's reduce: the CUDA blocks' sums added sample by sample, in a fixed order, into the frames'
 *        channels 10 .. 13 (near left and right, far left and right).
 * @param[in]     hdr      the block header
 * @param[in]     scratch  the chime kernel's per-block sums
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels
 */
__global__ void chimeReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out )
{
    __shared__ float part[ReduceGroups][ReduceLanes + 1];
    const int        n = hdr->m_n;
    const int        t = static_cast<int>( blockIdx.x ) * ReduceLanes + static_cast<int>( threadIdx.x );
    const int        c = t / maxN;
    const int        s = t - c * maxN;
    const bool       live = c < 4 && s < n;
    float            acc = 0.0f;
    if ( live && hdr->m_chimes.m_on )
        for ( int k = threadIdx.y; k < ChimeBlocks; k += ReduceGroups )
            acc += scratch[static_cast<size_t>( k ) * 4 * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if ( threadIdx.y == 0 && live )
    {
        for ( int q = 1; q < ReduceGroups; ++q ) acc += part[q][threadIdx.x];
        out[s * CloudOutChannels + 10 + c] = acc;
    }
}

/**
 * @brief The phantom layer's reduce: its voice's chunks summed sample by sample, in a fixed order, into the frames'
 *        channels 14 and 15 -- kept apart from the voices, so that it reaches the ears untouched (the echoes, the
 *        rooms and the width would blur the turn of phase between them that is all there is to hear).
 * @param[in]     hdr      the block header
 * @param[in]     scratch  per chunk four channels of sums (the render kernel's)
 * @param[in]     perVoice chunks a voice has
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels
 */
__global__ void phantomReduceKernel( const BlockHeader *hdr, const float *scratch, int perVoice, int maxN, float *out )
{
    const int s = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    const int n = hdr->m_n;
    if ( s >= n ) return;
    const int v = hdr->m_phantomVoice;
    float     l = 0.0f, r = 0.0f;
    if ( v >= 0 )
        for ( int k = v * perVoice; k < ( v + 1 ) * perVoice; ++k )
        {
            l += scratch[static_cast<size_t>( k ) * CloudChannels * maxN + s];
            r += scratch[static_cast<size_t>( k ) * CloudChannels * maxN + maxN + s];
        }
    out[s * CloudOutChannels + 14] = l;
    out[s * CloudOutChannels + 15] = r;
}

/**
 * @brief A dream band's place in the noise of a query: the same in every CUDA block of a block (hashed from the
 *        block's index and the band), uniform, -1 .. 1.
 * @param[in] block  the block's index
 * @param[in] b      the band
 * @return the noise
 */
__device__ __forceinline__ float dreamNoise( std::uint32_t block, int b )
{
    return 2.0f * hash01( block * 0x9E3779B9U + 0xD7EA3U, static_cast<std::uint32_t>( b ), 71 ) - 1.0f;
}

/**
 * @brief The dream's likeness (Phase 27): every memory against the query -- the dream's state with a wander in it --
 *        and, in a block that keeps what it gathered, against the gathered spectrum too (whether it is new). The
 *        memories are kept band by band (DreamMemories each), so neighbouring threads read neighbouring memories.
 * @param[in]  hdr    the block header
 * @param[in]  acc    the hall's bands gathered since the last memory was kept (amplitudes)
 * @param[in]  mem    the memories, DreamBands x DreamMemories
 * @param[in]  state  the dream's state, DreamBands
 * @param[out] sims   per memory its likeness to the query, then (DreamMemories on) to the gathered spectrum
 * @param[in]  meta   the Phase 27 counters: [0] memories kept
 */
__global__ void __launch_bounds__( DreamThreads )
    dreamSimKernel( const BlockHeader *hdr, const float *acc, const float *mem, const float *state, float *sims,
                    const int *meta )
{
    const Dream &D = hdr->m_dream;
    if ( !D.m_on ) return;
    __shared__ float q[DreamBands];
    __shared__ float g[DreamBands];
    __shared__ float red[2][DreamThreads];
    const int        t = static_cast<int>( threadIdx.x );
    float            qq = 0.0f, gg = 0.0f;
    for ( int b = t; b < DreamBands; b += DreamThreads )
    {
        q[b] = state[b] + D.m_noise * dreamNoise( hdr->m_blockIndex, b ) * 0.0441942f;   // 1 / sqrt(512)
        g[b] = acc[2 * b] + acc[2 * b + 1];
        qq += q[b] * q[b];
        gg += g[b] * g[b];
    }
    red[0][t] = qq;
    red[1][t] = gg;
    __syncthreads();
    for ( int s = DreamThreads / 2; s > 0; s >>= 1 )
    {
        if ( t < s )
        {
            red[0][t] += red[0][t + s];
            red[1][t] += red[1][t + s];
        }
        __syncthreads();
    }
    const float iq = red[0][0] > 1.0e-20f ? rsqrtf( red[0][0] ) : 0.0f;
    const float ig = red[1][0] > 1.0e-20f ? rsqrtf( red[1][0] ) : 0.0f;
    const int   mu = static_cast<int>( blockIdx.x ) * DreamThreads + t;
    if ( mu >= meta[0] ) return;
    float sq = 0.0f, sg = 0.0f;
    for ( int b = 0; b < DreamBands; ++b )
    {
        const float x = mem[static_cast<size_t>( b ) * DreamMemories + mu];
        sq = fmaf( x, q[b], sq );
        sg = fmaf( x, g[b], sg );
    }
    sims[mu] = sq * iq;
    if ( D.m_store ) sims[DreamMemories + mu] = sg * ig;
}

/**
 * @brief The dream's memory and its recall (Phase 27), one CUDA block, a thread a band: in a block that keeps what it
 *        gathered, the spectrum is kept as a memory if it is new (less like every memory than `novelty`; past
 *        DreamMemories the oldest goes) and the gathering begins again; then the DreamRecall memories nearest the
 *        query are blended by the softmax of their likeness (sharpness `focus`), and the state moves a little of the
 *        way there (`ease`), its length kept 1. All in a fixed order: the same input dreams the same dream.
 * @param[in]     hdr    the block header
 * @param[in,out] acc    the hall's bands gathered since the last memory was kept (cleared when one is)
 * @param[in,out] mem    the memories, DreamBands x DreamMemories
 * @param[in,out] state  the dream's state, DreamBands
 * @param[in]     sims   the likenesses (dreamSimKernel)
 * @param[in,out] meta   the Phase 27 counters: [0] memories kept, [2] where the next goes
 */
__global__ void __launch_bounds__( DreamBands )
    dreamRecallKernel( const BlockHeader *hdr, float *acc, float *mem, float *state, const float *sims, int *meta )
{
    const Dream &D = hdr->m_dream;
    if ( !D.m_on ) return;
    __shared__ float red[DreamBands];
    __shared__ int   who[DreamBands];
    __shared__ float topSim[DreamRecall];
    __shared__ int   topWho[DreamRecall];
    const int        b = static_cast<int>( threadIdx.x );
    const int        count = meta[0];
    if ( D.m_clear )
    {
        state[b] = 0.0f;
        acc[2 * b] = acc[2 * b + 1] = 0.0f;
        __syncthreads();
        if ( b == 0 ) meta[0] = meta[2] = 0;
        return;
    }
    // A thread's band of the block-wide sums, and the block-wide maximum (value, lowest index first).
    auto sum = [&]( float x )
    {
        red[b] = x;
        __syncthreads();
        for ( int s = DreamBands / 2; s > 0; s >>= 1 )
        {
            if ( b < s ) red[b] += red[b + s];
            __syncthreads();
        }
        const float total = red[0];
        __syncthreads();
        return total;
    };
    auto argmax = [&]( float x, int i, float &best )
    {
        red[b] = x;
        who[b] = i;
        __syncthreads();
        for ( int s = DreamBands / 2; s > 0; s >>= 1 )
        {
            if ( b < s && ( red[b + s] > red[b] || ( red[b + s] == red[b] && who[b + s] < who[b] ) ) )
            {
                red[b] = red[b + s];
                who[b] = who[b + s];
            }
            __syncthreads();
        }
        best = red[0];
        const int at = who[0];
        __syncthreads();
        return at;
    };
    if ( D.m_store )
    {
        const float g = acc[2 * b] + acc[2 * b + 1];
        const float energy = sum( g * g );
        float       nearest = -2.0f;
        for ( int mu = b; mu < count; mu += DreamBands ) nearest = fmaxf( nearest, sims[DreamMemories + mu] );
        float     most = -2.0f;
        const int dummy = argmax( nearest, 0, most );
        (void)dummy;
        if ( energy > 1.0e-10f && ( count == 0 || most < D.m_novelty ) )
        {
            const int slot = meta[2];
            mem[static_cast<size_t>( b ) * DreamMemories + slot] = g * rsqrtf( energy );
            __syncthreads();
            if ( b == 0 )
            {
                meta[2] = ( slot + 1 ) % DreamMemories;
                meta[0] = min( count + 1, DreamMemories );
            }
        }
        acc[2 * b] = acc[2 * b + 1] = 0.0f;
    }
    if ( count == 0 ) return;   // nothing remembered yet: nothing to dream (the new memory is recalled from the next)
    // The nearest memories to the query, one by one.
    const int k = min( DreamRecall, count );
    for ( int j = 0; j < k; ++j )
    {
        float bestHere = -3.0f;
        int   whoHere = 0x7FFFFFFF;
        for ( int mu = b; mu < count; mu += DreamBands )
        {
            bool taken = false;
            for ( int i = 0; i < j; ++i ) taken = taken || topWho[i] == mu;
            if ( !taken && sims[mu] > bestHere )
            {
                bestHere = sims[mu];
                whoHere = mu;
            }
        }
        float     best = -3.0f;
        const int at = argmax( bestHere, whoHere, best );
        if ( b == 0 )
        {
            topSim[j] = best;
            topWho[j] = at;
        }
        __syncthreads();
    }
    // Their blend by the softmax of their likeness, and the state a step towards it.
    float wsum = 0.0f, r = 0.0f;
    for ( int j = 0; j < k; ++j )
    {
        const float w = __expf( D.m_focus * ( topSim[j] - topSim[0] ) );
        wsum += w;
        r = fmaf( w, mem[static_cast<size_t>( b ) * DreamMemories + topWho[j]], r );
    }
    r /= fmaxf( wsum, 1.0e-20f );
    const float rr = sum( r * r );
    r *= rr > 1.0e-20f ? rsqrtf( rr ) : 0.0f;
    const float x0 = state[b];
    const float xx0 = sum( x0 * x0 );
    float       x = xx0 > 1.0e-12f ? x0 + D.m_ease * ( r - x0 ) : r;
    const float xx = sum( x * x );
    state[b] = xx > 1.0e-20f ? x * rsqrtf( xx ) : 0.0f;
}

/**
 * @brief The Phase 27 counters brought to the block's statistics: [12] the memories the dream keeps, [13] the chimes
 *        that struck; the strikes counted again from the next block.
 * @param[in,out] meta   the counters
 * @param[out]    stats  the statistics the host reads every block
 */
__global__ void phase27StatsKernel( int *meta, float *stats )
{
    if ( threadIdx.x != 0 ) return;
    stats[12] = static_cast<float>( meta[0] );
    stats[13] = static_cast<float>( meta[1] );
    meta[1] = 0;
}

// ---- Phase 28 (16.10.2026): the weather in the wind, the bloom, the swarmalators, the physarum.

constexpr int   RainThreads = 256;                         ///< drops per CUDA block of the rain kernel
constexpr int   RainBlocks = RainSlots / RainThreads;      ///< CUDA blocks of the rain kernel
constexpr int   LeafThreads = 256;                         ///< clusters per CUDA block of the leaf kernel
constexpr int   LeafBlocks = LeafClusters / LeafThreads;   ///< CUDA blocks of the leaf kernel
constexpr int   WeatherTile = 128;                         ///< samples the rain kernel sums per pass
constexpr int   MixTile = 32;   ///< samples the leaves and the sea mix through shared memory per pass
constexpr int   WeatherBlocks = RainBlocks + LeafBlocks + 1;   ///< the weather's scratch: rain, leaves, the sea
constexpr int   BloomThreads = 256;                            ///< modes per CUDA block of the bloom kernel
constexpr int   BloomBlocks = BloomModes / BloomThreads;       ///< CUDA blocks of the bloom kernel
constexpr int   BloomTile = 128;                               ///< samples the bloom kernel sums per pass
constexpr float BloomLow = 30.0f;                              ///< the bloom's lowest mode that sounds, Hz
constexpr float BloomHigh = 18000.0f;                          ///< its highest, Hz
constexpr int   SwarmChunks = 16;                              ///< the swarm's partners split over CUDA blocks
constexpr int   SwarmThreads = 256;                            ///< swarmalators per CUDA block
constexpr int   StatsFloats = 64;                              ///< the stats brought down every block
static_assert( RainSlots % RainThreads == 0 && LeafClusters % LeafThreads == 0, "whole CUDA blocks" );
static_assert( BloomModes % BloomThreads == 0 && SwarmN % SwarmThreads == 0, "whole CUDA blocks" );
static_assert( SwarmN == 1024, "the swarm's view is reduced by one CUDA block of 1024" );

/**
 * @brief The wind where a cell of its lattice is: its velocity, lattice units a step.
 * @param[in] wind  the lattice (buffer A, after the block's steps)
 * @param[in] idx   the cell
 * @return across and from near to far
 */
__device__ __forceinline__ float2 windAt( const float *wind, int idx )
{
    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 9; ++k )
    {
        const float f = wind[k * WindCells + idx];
        rho += f;
        ux += static_cast<float>( cWindX[k] ) * f;
        uy += static_cast<float>( cWindY[k] ) * f;
    }
    return rho > 1.0e-3f ? make_float2( ux / rho, uy / rho ) : make_float2( 0.0f, 0.0f );
}

/**
 * @brief The showers (Phase 28): the rain's density over the field, carried by the wind -- semi-Lagrangian, from
 *        where the wind brought it, bilinear on the torus -- and fed at the windward side (the left) by the host's
 *        slow randoms, drifting back to even slowly. Two buffers, by the block's parity.
 * @param[in]     hdr      the block header
 * @param[in]     wind     the wind's lattice after this block's steps
 * @param[in,out] showers  two buffers of WindCells
 */
__global__ void rainFieldKernel( const BlockHeader *hdr, const float *wind, float *showers )
{
    const Weather &W = hdr->m_weather;
    if ( !W.m_rainOn ) return;
    const int idx = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= WindCells ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *src = showers + ( odd ? WindCells : 0 );
    float       *dst = showers + ( odd ? 0 : WindCells );
    const int    x = idx % WindW, y = idx / WindW;
    float        c = 1.0f;
    if ( !W.m_reset && !W.m_rainFresh )
    {
        const float2 u = windAt( wind, idx );
        const float  sx = static_cast<float>( x ) - 2.0f * W.m_advect * u.x;
        const float  sy = static_cast<float>( y ) - 2.0f * W.m_advect * u.y;
        const float  fx = floorf( sx ), fy = floorf( sy );
        const int    x0 = static_cast<int>( fx ) & ( WindW - 1 ), y0 = static_cast<int>( fy ) & ( WindH - 1 );
        const int    x1 = ( x0 + 1 ) & ( WindW - 1 ), y1 = ( y0 + 1 ) & ( WindH - 1 );
        const float  ax = sx - fx, ay = sy - fy;
        const float  top = src[y0 * WindW + x0] + ( src[y0 * WindW + x1] - src[y0 * WindW + x0] ) * ax;
        const float  bottom = src[y1 * WindW + x0] + ( src[y1 * WindW + x1] - src[y1 * WindW + x0] ) * ax;
        c = top + ( bottom - top ) * ay;
        c += ( 1.0f - c ) *
             0.00002f;   // back to even over two minutes (at 5 s a shower died before a quarter of the field)
    }
    if ( x < 2 )
    {   // the windward side: fed by the slow randoms, from near to far
        const float fy = static_cast<float>( y ) * ( 3.0f / static_cast<float>( WindH - 1 ) );
        const int   i0 = min( static_cast<int>( fy ), 2 );
        const float g = W.m_shower[i0] + ( W.m_shower[i0 + 1] - W.m_shower[i0] ) * ( fy - static_cast<float>( i0 ) );
        c = 1.0f + 2.0f * W.m_showers * g;   // at the top: no rain to three times the rain
    }
    dst[idx] = fminf( fmaxf( c, 0.0f ), 3.0f );
}

/** @brief A drop's dice: the block it fell in, made a seed. @param[in] block the block @return the seed */
__device__ __forceinline__ std::uint32_t dropSeed( int block )
{
    return static_cast<std::uint32_t>( block ) * 0x9E3779B9U + 0x4A1E5EEDU;
}

/** @brief A drop as it sounds: worked out from its dice every block (nothing of it is stored but its state). */
struct DropSpec
{
    float m_amp;          ///< its ring's amplitude
    float m_omega;        ///< its ring's angular step a sample at its start
    float m_dOmega;       ///< the step's rise a sample (a bubble rising)
    float m_r;            ///< its ring's decay a sample
    float m_click;        ///< its impact's amplitude
    float m_clickDecay;   ///< the impact's decay a sample
    float m_g[4];         ///< its gains: near left, near right, far left, far right
    float m_gClick[4];    ///< the impact's, darker with distance
};

/**
 * @brief A drop's sound from its dice: where it fell, what it fell on, how big it was (Phase 28).
 * @param[in] W     the weather
 * @param[in] slot  its slot
 * @param[in] born  the block it fell in
 * @param[in] at    where it fell: across and from near to far, wind cells (the spawner's)
 * @param[in] fs    the sample rate
 * @return the drop
 */
__device__ DropSpec dropSpec( const Weather &W, int slot, int born, float2 at, float fs )
{
    const std::uint32_t ds = dropSeed( born ), us = static_cast<std::uint32_t>( slot );
    DropSpec            d;
    const float         gx = at.x, gy = at.y;
    const float         u = hash01( ds, us, 5 ), v = hash01( ds, us, 6 ), w = hash01( ds, us, 7 );
    const bool          tuned = W.m_pitchCount > 0 && hash01( ds, us, 8 ) < W.m_tune;
    const float         pitch =
        W.m_pitchCount > 0
            ? W.m_pitch[hash32( ds ^ ( us * 0x85EBCA6BU ) ) % static_cast<std::uint32_t>( W.m_pitchCount )]
            : 1000.0f;
    float f = 1000.0f, tau = 0.01f, chirp = 0.0f;
    switch ( W.m_surface )
    {
    case 1:   // leaves: a tick, a leaf's own ring
        f = 1200.0f * __expf( 1.32f * u );
        tau = 0.008f + 0.017f * v;
        d.m_amp = 0.25f;
        d.m_click = 0.3f * ( 0.5f + w );
        d.m_clickDecay = __expf( -1.0f / ( 0.00015f * fs ) );
        break;
    case 2:   // stone: a click and a glint
        f = 6000.0f * __expf( 0.7f * u );
        tau = 0.0015f;
        d.m_amp = 0.15f;
        d.m_click = 0.6f * ( 0.5f + w );
        d.m_clickDecay = __expf( -1.0f / ( 0.0001f * fs ) );
        break;
    case 3:   // a tin roof: the sheet rings where the drop fell
        f = tuned ? pitch : 300.0f * __expf( 2.08f * u );
        tau = 0.08f + 0.27f * v;
        d.m_amp = 0.2f;
        d.m_click = 0.25f * ( 0.5f + w );
        d.m_clickDecay = __expf( -1.0f / ( 0.0002f * fs ) );
        break;
    default:
    {   // water: a bubble trapped -- Minnaert's ring, 3.26 kHz over its radius in mm, damped as van den Doel
        // measured, rising as it rises -- or only the splash
        const float rmm = 0.25f * __expf( 2.485f * powf( u, 1.4f ) );
        f = tuned ? pitch : 3260.0f / rmm;
        const float delta = 0.043f * f + 0.0014f * f * sqrtf( f );
        tau = 1.0f / delta;
        chirp = ( 0.1f + 0.4f * v ) * delta;
        d.m_amp = hash01( ds, us, 9 ) < 0.6f ? 0.5f * sqrtf( rmm ) * ( 0.5f + w ) : 0.0f;
        d.m_click = 0.15f * ( 0.5f + w );
        d.m_clickDecay = __expf( -1.0f / ( 0.00025f * fs ) );
        break;
    }
    }
    f = fminf( f, 0.42f * fs );
    d.m_omega = 6.2831853f * f / fs;
    d.m_dOmega = d.m_omega * chirp / fs;
    d.m_r = __expf( -1.0f / ( tau * fs ) );
    const float place = 2.0f * gx / static_cast<float>( WindW ) - 1.0f, dist = gy / static_cast<float>( WindH );
    float       gl, gr, gn, gf;
    __sincosf( 0.78539816f * ( 1.0f + place ), &gr, &gl );
    __sincosf( 1.57079633f * dist, &gf, &gn );
    const float a = W.m_rainLevel * __expf( -dist * f * ( 1.0f / 9000.0f ) );
    const float c = W.m_rainLevel * __expf( -dist * 0.8f );
    d.m_g[0] = gl * gn * a;
    d.m_g[1] = gr * gn * a;
    d.m_g[2] = gl * gf * a;
    d.m_g[3] = gr * gf * a;
    d.m_gClick[0] = gl * gn * c;
    d.m_gClick[1] = gr * gn * c;
    d.m_gClick[2] = gl * gf * c;
    d.m_gClick[3] = gr * gf * c;
    return d;
}

/**
 * @brief The rain's spawner (Phase 28): the drops that fall this block -- as many as the rate says (Poisson; a normal
 *        draw from 30 on), each where the showers are (drawn against their density, up to eight tries), each into
 *        the next free slot in ring order. So the drops that sound stay together in a few CUDA blocks of the rain
 *        kernel and the others rest (scattered, all 64 worked: 35 us a block at 600 drops a second). One thread.
 * @param[in]     hdr      the block header
 * @param[in]     showers  the showers' density, both buffers (this block's is read: rainFieldKernel's)
 * @param[in,out] birth    per slot the block its drop fell in (-1 free) and its sample there
 * @param[out]    at       per slot where its drop fell
 * @param[in,out] meta     the counters: [6] drops born, [9] where the ring stands
 */
__global__ void rainSpawnKernel( const BlockHeader *hdr, const float *showers, int2 *birth, float2 *at, int *meta )
{
    const Weather &W = hdr->m_weather;
    if ( threadIdx.x != 0 || !W.m_rainOn || W.m_reset || !( W.m_rainRate > 0.0f ) ) return;
    const int           n = hdr->m_n;
    const int           blk = static_cast<int>( hdr->m_blockIndex & 0x3FFFFFFFU );
    const std::uint32_t ds = dropSeed( blk ) ^ 0x5FA3C1E7U;
    const float         lambda = W.m_rainRate * static_cast<float>( n ) / hdr->m_sampleRate;
    int                 count = 0;
    if ( lambda < 30.0f )
    {   // Knuth's: multiply uniforms until the product falls under e^-lambda
        const float limit = __expf( -lambda );
        float       p = 1.0f;
        for ( int k = 0; k < 200; ++k )
        {
            p *= hash01( ds, static_cast<std::uint32_t>( k ), 31 );
            if ( p <= limit ) break;
            ++count;
        }
    }
    else
    {
        const float u1 = fmaxf( hash01( ds, 0U, 32 ), 1.0e-7f ), u2 = hash01( ds, 0U, 33 );
        const float z = sqrtf( -2.0f * __logf( u1 ) ) * __cosf( 6.2831853f * u2 );
        count = max( 0, static_cast<int>( lambda + sqrtf( lambda ) * z + 0.5f ) );
    }
    count = min( count, 256 );
    const float *dens = showers + ( ( hdr->m_blockIndex & 1U ) ? 0 : WindCells );
    int          ring = ( ( meta[9] % RainSlots ) + RainSlots ) % RainSlots;
    int          born = 0;
    for ( int k = 0; k < count; ++k )
    {
        // where: drawn against the showers' density (at most three times the mean)
        float gx = -1.0f, gy = 0.0f;
        for ( int tries = 0; tries < 8; ++tries )
        {
            const std::uint32_t key = static_cast<std::uint32_t>( k * 8 + tries );
            const float         x = static_cast<float>( WindW ) * hash01( ds, key, 34 );
            const float         y = static_cast<float>( WindH ) * hash01( ds, key, 35 );
            const float         c =
                dens[min( static_cast<int>( y ), WindH - 1 ) * WindW + min( static_cast<int>( x ), WindW - 1 )];
            if ( hash01( ds, key, 36 ) * 3.0f < c )
            {
                gx = x;
                gy = y;
                break;
            }
        }
        if ( gx < 0.0f ) continue;   // a dry patch
        // into the next free slot
        int slot = -1;
        for ( int s = 0; s < RainSlots; ++s )
        {
            const int c = ( ring + s ) % RainSlots;
            if ( birth[c].x < 0 )
            {
                slot = c;
                break;
            }
        }
        if ( slot < 0 ) break;   // every slot rings
        birth[slot] = make_int2(
            blk, min( static_cast<int>( static_cast<float>( n ) * hash01( ds, static_cast<std::uint32_t>( k ), 37 ) ),
                      n - 1 ) );
        at[slot] = make_float2( gx, gy );
        ring = ( slot + 1 ) % RainSlots;
        ++born;
    }
    meta[9] = ring;
    meta[6] += born;
}

/**
 * @brief The rain (Phase 28): one thread a slot, RainSlots of them. A slot the spawner gave a drop (rainSpawnKernel)
 *        rings it out (a complex one-pole turned every sample, its step rising for a bubble) with its impact's click,
 *        and is free again once both are under -60 dB. A CUDA block of free slots writes nothing but silence.
 * @param[in]     hdr      the block header
 * @param[in]     at       per slot where its drop fell (the spawner's)
 * @param[in,out] drop     per slot its ring (re, im, step, samples since it fell)
 * @param[in,out] birth    per slot the block it fell in (-1 free) and its sample there
 * @param[out]    scratch  per CUDA block four channels of sums (near L, R, far L, R)
 * @param[in]     maxN     the scratch stride
 * @param[in,out] meta     the counters: [4] drops sounding this block, [5] the last block's, [6] drops born
 */
__global__ void __launch_bounds__( RainThreads ) rainKernel( const BlockHeader *hdr, const float2 *at, float4 *drop,
                                                             int2 *birth, float *scratch, int maxN, int *meta )
{
    __shared__ float sAcc[RainThreads / 32][4][WeatherTile];
    const Weather   &W = hdr->m_weather;
    const int        n = hdr->m_n;
    const int        slot = static_cast<int>( blockIdx.x ) * RainThreads + static_cast<int>( threadIdx.x );
    const int        blk = static_cast<int>( hdr->m_blockIndex & 0x3FFFFFFFU );
    float           *out = scratch + static_cast<size_t>( blockIdx.x ) * 4 * maxN;
    const float      fs = hdr->m_sampleRate;
    int2             b = ( W.m_reset || !W.m_rainOn ) ? make_int2( -1, 0 ) : birth[slot];
    float4           d = drop[slot];
    const bool       alive = b.x >= 0;
    if ( !__syncthreads_or( alive ) )
    {
        birth[slot] = b;
        for ( int s = threadIdx.x; s < n; s += RainThreads )
            for ( int ch = 0; ch < 4; ++ch ) out[ch * maxN + s] = 0.0f;
        return;
    }
    DropSpec spec{};
    int      fallAt = -1;   // the sample it falls at in this block
    if ( alive )
    {
        spec = dropSpec( W, slot, b.x, at[slot], fs );
        if ( b.x == blk ) fallAt = b.y;
    }
    const int lane = static_cast<int>( threadIdx.x ) & 31, warp = static_cast<int>( threadIdx.x ) >> 5;
    for ( int t0 = 0; t0 < n; t0 += WeatherTile )
    {
        const int tn = min( WeatherTile, n - t0 );
        for ( int i = 0; i < tn; ++i )
        {
            float vb = 0.0f, vc = 0.0f;
            if ( alive )
            {
                if ( t0 + i == fallAt ) d = make_float4( spec.m_amp, 0.0f, spec.m_omega, 0.0f );
                if ( fallAt < 0 || t0 + i >= fallAt )
                {
                    vb = d.y;
                    const float tc = d.w;
                    if ( tc < 400.0f )
                        vc = spec.m_click * __powf( spec.m_clickDecay, tc ) *
                             ( 2.0f *
                                   hash01( dropSeed( b.x ),
                                           static_cast<std::uint32_t>( slot ) * 512U + static_cast<std::uint32_t>( tc ),
                                           11 ) -
                               1.0f );
                    float sn, cs;
                    __sincosf( d.z, &sn, &cs );
                    const float re = spec.m_r * ( d.x * cs - d.y * sn );
                    const float im = spec.m_r * ( d.x * sn + d.y * cs );
                    d = make_float4( re, im, d.z + spec.m_dOmega, tc + 1.0f );
                }
            }
            float v[4];
            for ( int ch = 0; ch < 4; ++ch )
            {
                v[ch] = vb * spec.m_g[ch] + vc * spec.m_gClick[ch];
                for ( int off = 16; off > 0; off >>= 1 ) v[ch] += __shfl_down_sync( 0xFFFFFFFFU, v[ch], off );
            }
            if ( lane == 0 )
                for ( int ch = 0; ch < 4; ++ch ) sAcc[warp][ch][i] = v[ch];
        }
        __syncthreads();
        for ( int s = threadIdx.x; s < 4 * tn; s += RainThreads )
        {
            const int ch = s / tn, i = s - ch * tn;
            float     a = 0.0f;
            for ( int wv = 0; wv < RainThreads / 32; ++wv ) a += sAcc[wv][ch][i];
            out[ch * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    if ( alive )
    {   // rung out (-60 dB and the click done): free again
        if ( d.w > 400.0f && d.x * d.x + d.y * d.y < 1.0e-6f * spec.m_amp * spec.m_amp + 1.0e-14f )
            b = make_int2( -1, 0 );
        else
            atomicAdd( meta + 4, 1 );
    }
    birth[slot] = b;
    drop[slot] = d;
}

/** @brief A foliage's leaves: their ring's band, its decay, their clicks a second at full wind, their loudness. */
struct FoliageSpec
{
    float m_lo;     ///< the lowest ring, Hz
    float m_hi;     ///< the highest, Hz
    float m_tau;    ///< a click's ring, s
    float m_rate;   ///< clicks a second a cluster at full wind
    float m_amp;    ///< the foliage's loudness against the others
};

/** @brief The foliages (FoliageE): broad leaves, pine needles, grass, dry leaves. */
__constant__ FoliageSpec cFoliage[4] = { { 1500.0f, 4000.0f, 0.0005f, 3000.0f, 1.0f },
                                         { 4000.0f, 9000.0f, 0.00015f, 8000.0f, 0.8f },
                                         { 800.0f, 2500.0f, 0.0005f, 2000.0f, 0.8f },
                                         { 2000.0f, 7000.0f, 0.0008f, 400.0f, 1.3f } };

/**
 * @brief The leaves (Phase 28): LeafClusters clusters over the field (64 x 64, jittered), each rustling as the wind
 *        where it stands blows -- its speed and its gusting part (the wind against the cluster's slow average of
 *        it), squared: a rustle's power. Every leaf is a click, a cluster a complex one-pole at its own pitch in its
 *        foliage's band. A CUDA block of clusters in calm air that rings no more writes nothing but silence.
 * @param[in]     hdr      the block header
 * @param[in]     wind     the wind's lattice after this block's steps
 * @param[in,out] state    per cluster (re, im, slow wind across, slow wind from near to far)
 * @param[out]    scratch  the weather's scratch (the leaves' CUDA blocks after the rain's)
 * @param[in]     maxN     the scratch stride
 * @param[in,out] meta     the counters: [7] the clusters' rustle, thousandths, summed
 */
__global__ void __launch_bounds__( LeafThreads )
    leafKernel( const BlockHeader *hdr, const float *wind, float4 *state, float *scratch, int maxN, int *meta )
{
    __shared__ float    sV[LeafThreads][MixTile + 1];
    __shared__ float    sG[LeafThreads][4];
    const Weather      &W = hdr->m_weather;
    const int           n = hdr->m_n;
    const int           c = static_cast<int>( blockIdx.x ) * LeafThreads + static_cast<int>( threadIdx.x );
    float              *out = scratch + static_cast<size_t>( RainBlocks + blockIdx.x ) * 4 * maxN;
    const float         fs = hdr->m_sampleRate;
    const std::uint32_t uc = static_cast<std::uint32_t>( c );
    float4              s = ( W.m_leavesOn && !W.m_reset ) ? state[c] : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    const float         gx = static_cast<float>( ( c % 64 ) * 2 ) + 2.0f * hash01( 0x1EAF5EEDU, uc, 1 );
    const float         gy = static_cast<float>( c / 64 ) + hash01( 0x1EAF5EEDU, uc, 2 );
    float               p = 0.0f, I = 0.0f;
    if ( W.m_leavesOn && W.m_wind )
    {
        const int idx = ( static_cast<int>( gy ) & ( WindH - 1 ) ) * WindW + ( static_cast<int>( gx ) & ( WindW - 1 ) );
        const float2 u = windAt( wind, idx );
        const float  dt = static_cast<float>( n ) / fs;
        s.z += ( u.x - s.z ) * fminf( dt * 0.5f, 1.0f );
        s.w += ( u.y - s.w ) * fminf( dt * 0.5f, 1.0f );
        const float gust = sqrtf( ( u.x - s.z ) * ( u.x - s.z ) + ( u.y - s.w ) * ( u.y - s.w ) );
        const float speed = sqrtf( u.x * u.x + u.y * u.y );
        const float k = ( speed + 2.0f * gust ) / fmaxf( W.m_windRef, 1.0e-5f );
        I = fminf( k * k, 4.0f );
        p = fminf( cFoliage[min( max( W.m_foliage, 0 ), 3 )].m_rate * I / fs, 0.5f );
    }
    const bool alive = W.m_leavesOn && ( p > 1.0e-7f || s.x * s.x + s.y * s.y > 1.0e-12f );
    if ( !__syncthreads_or( alive ) )
    {
        state[c] = make_float4( 0.0f, 0.0f, s.z, s.w );
        for ( int t = threadIdx.x; t < n; t += LeafThreads )
            for ( int ch = 0; ch < 4; ++ch ) out[ch * maxN + t] = 0.0f;
        return;
    }
    const FoliageSpec &F = cFoliage[min( max( W.m_foliage, 0 ), 3 )];
    const float        f = F.m_lo * exp2f( log2f( F.m_hi / F.m_lo ) * hash01( 0x1EAF5EEDU, uc, 3 ) );
    const float        r = __expf( -1.0f / ( F.m_tau * fs ) );
    float              sw, cw;
    __sincosf( 6.2831853f * f / fs, &sw, &cw );
    const float ar = r * cw, ai = r * sw;
    // A click's power at full wind, made one: the clicks a sample, a third (uniform), through the pole's gain.
    const float p1 = F.m_rate / fs;
    const float norm = rsqrtf( p1 * ( 1.0f / 3.0f ) / fmaxf( 1.0f - r * r, 1.0e-6f ) ) * ( 1.0f / 64.0f );
    const float place = 2.0f * gx / static_cast<float>( WindW ) - 1.0f, dist = gy / static_cast<float>( WindH );
    float       gl, gr, gn, gf;
    __sincosf( 0.78539816f * ( 1.0f + place ), &gr, &gl );
    __sincosf( 1.57079633f * dist, &gf, &gn );
    const float         w = W.m_leavesLevel * F.m_amp * norm * __expf( -dist * f * ( 1.0f / 9000.0f ) );
    const float         g4[4] = { gl * gn * w, gr * gn * w, gl * gf * w, gr * gf * w };
    const std::uint32_t seed = hash32( hdr->m_blockIndex * 0x9E3779B9U ^ ( uc * 0x85EBCA6BU ) );
    for ( int ch = 0; ch < 4; ++ch ) sG[threadIdx.x][ch] = g4[ch];
    for ( int t0 = 0; t0 < n; t0 += MixTile )
    {
        const int tn = min( MixTile, n - t0 );
        for ( int i = 0; i < tn; ++i )
        {
            const std::uint32_t si = static_cast<std::uint32_t>( t0 + i );
            const float x = ( p > 0.0f && hash01( seed, si, 1 ) < p ) ? 2.0f * hash01( seed, si, 2 ) - 1.0f : 0.0f;
            const float re = fmaf( ar, s.x, fmaf( -ai, s.y, x ) );
            const float im = fmaf( ar, s.y, ai * s.x );
            s.x = re;
            s.y = im;
            sV[threadIdx.x][i] = im;
        }
        __syncthreads();
        for ( int o = threadIdx.x; o < 4 * tn; o += LeafThreads )
        {   // a sum down the threads, in their order
            const int ch = o / tn, i = o - ch * tn;
            float     a = 0.0f;
            for ( int k = 0; k < LeafThreads; ++k ) a = fmaf( sG[k][ch], sV[k][i], a );
            out[ch * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    if ( s.x * s.x + s.y * s.y < 1.0e-14f ) s.x = s.y = 0.0f;
    state[c] = s;
    atomicAdd( meta + 7, static_cast<int>( I * 250.0f ) );
}

/**
 * @brief The sea (Phase 28): one thread a stretch of the shore (SurfSegments, 6 m each). The swell is SurfWaves
 *        trains about its period (a bell around the peak, deep water: k = w^2 / g), coming in up to 34 degrees off
 *        straight, so crests run along the shore and come in sets. Where the water rises past its break it foams
 *        (the foam fades over three seconds); the foam hisses (noise through a low pass that opens with it), the
 *        backwash crackles through pebbles, and on rocks each breaking wave booms (a low resonance struck).
 * @param[in]     hdr      the block header
 * @param[in,out] state    per stretch two entries: (foam, low pass 1, low pass 2, the water before), (boom re, im,
 *                         crackle re, im)
 * @param[out]    scratch  the weather's scratch (the sea's CUDA block after the leaves')
 * @param[in]     maxN     the scratch stride
 * @param[in,out] meta     the counters: [8] the foam, thousandths, summed
 */
__global__ void __launch_bounds__( SurfSegments )
    surfKernel( const BlockHeader *hdr, float4 *state, float *scratch, int maxN, int *meta )
{
    __shared__ float sV[SurfSegments][MixTile + 1];
    __shared__ float sG[SurfSegments][4];
    const Weather   &W = hdr->m_weather;
    const int        n = hdr->m_n;
    const int        seg = static_cast<int>( threadIdx.x );
    float           *out = scratch + static_cast<size_t>( RainBlocks + LeafBlocks ) * 4 * maxN;
    if ( !W.m_surfOn )
    {
        if ( W.m_reset )
        {
            state[2 * seg] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            state[2 * seg + 1] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        }
        for ( int t = threadIdx.x; t < n; t += SurfSegments )
            for ( int ch = 0; ch < 4; ++ch ) out[ch * maxN + t] = 0.0f;
        return;
    }
    const float fs = hdr->m_sampleRate, dt = static_cast<float>( n ) / fs;
    float4      s = W.m_reset ? make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) : state[2 * seg];
    float4      q = W.m_reset ? make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) : state[2 * seg + 1];
    // The water here at the block's end: the trains, normalised to a standard deviation of one, times the height.
    const float x = 6.0f * static_cast<float>( seg );
    const float fp = 1.0f / fmaxf( W.m_surfPeriod, 1.0f );
    float       eta = 0.0f, var = 0.0f;
    for ( int k = 0; k < SurfWaves; ++k )
    {
        const std::uint32_t uk = static_cast<std::uint32_t>( k );
        const float         f = fp * ( 0.75f + 0.6f * hash01( 0x5EA5EA5EU, uk, 1 ) );
        const float         z = ( f / fp - 1.0f ) * ( 1.0f / 0.12f );
        const float         a = __expf( -0.5f * z * z ) * ( 0.6f + 0.8f * hash01( 0x5EA5EA5EU, uk, 2 ) );
        const float         om = 6.2831853f * f;
        const float         kk = om * om * ( 1.0f / 9.81f );
        const float         th = 0.6f * ( 2.0f * hash01( 0x5EA5EA5EU, uk, 3 ) - 1.0f );
        const float         ph = 6.2831853f * hash01( 0x5EA5EA5EU, uk, 4 );
        float               arg = kk * sinf( th ) * x - om * W.m_time + ph;
        arg -= 6.2831853f * floorf( arg * 0.15915494f );
        eta += a * __cosf( arg );
        var += 0.5f * a * a;
    }
    eta = eta * rsqrtf( fmaxf( var, 1.0e-9f ) ) * W.m_surfHeight;
    const float before = s.w;
    const float brk = fmaxf( 0.0f, eta - 1.4f );   // past its break: the crest foams
    s.x += dt * ( 6.0f * brk * W.m_surfHeight - s.x * ( 1.0f / 3.0f ) );
    s.x = fmaxf( s.x, 0.0f );
    const float fall = fmaxf( 0.0f, ( before - eta ) / dt ) * W.m_surfPeriod * ( 1.0f / 6.2831853f );   // backwash
    const bool  crash = brk > 0.0f && before <= 1.4f;
    s.w = eta;
    // Per shore: the foam's hiss, the backwash's crackle, the boom.
    const int   shore = min( max( W.m_shore, 0 ), 2 );
    const float hiss = shore == 0 ? 1.0f : shore == 1 ? 0.7f : 0.8f;
    const float crackle = shore == 1 ? 1.0f : shore == 2 ? 0.2f : 0.0f;
    const float boom = shore == 2 ? 1.2f : 0.3f;
    if ( crash )
    {
        q.x += boom * fminf( W.m_surfHeight, 2.0f );
    }
    const float foam = fminf( s.x, 4.0f );
    const float fc = ( shore == 2 ? 400.0f : 700.0f ) + 3500.0f * fminf( sqrtf( foam ), 1.5f );
    const float lp = 1.0f - __expf( -6.2831853f * fc / fs );
    const float pc = fminf( crackle * fall * foam * 0.03f, 0.3f );   // crackles a sample
    float       bs, bc;
    __sincosf( 6.2831853f * 55.0f / fs, &bs, &bc );
    const float br = __expf( -1.0f / ( 0.4f * fs ) );
    float       cs, cc;
    __sincosf( 6.2831853f * 3200.0f / fs, &cs, &cc );
    const float cr = __expf( -1.0f / ( 0.0006f * fs ) );
    const float place = 2.0f * static_cast<float>( seg ) / static_cast<float>( SurfSegments - 1 ) - 1.0f;
    const float dist = fminf(
        fmaxf( W.m_surfDistance + 0.08f * ( hash01( 0x5EA5EA5EU, 77U, static_cast<std::uint32_t>( seg ) ) - 0.5f ),
               0.0f ),
        1.0f );
    float gl, gr, gn, gf;
    __sincosf( 0.78539816f * ( 1.0f + place ), &gr, &gl );
    __sincosf( 1.57079633f * dist, &gf, &gn );
    const float         w = W.m_surfLevel * 0.0884f;   // 1 / sqrt(128): the stretches add in power
    const float         g4[4] = { gl * gn * w, gr * gn * w, gl * gf * w, gr * gf * w };
    const float         amp = hiss * sqrtf( foam ) * 1.7f;
    const std::uint32_t seed =
        hash32( hdr->m_blockIndex * 0x9E3779B9U ^ ( static_cast<std::uint32_t>( seg ) * 0xC2B2AE35U ) );
    for ( int ch = 0; ch < 4; ++ch ) sG[threadIdx.x][ch] = g4[ch];
    for ( int t0 = 0; t0 < n; t0 += MixTile )
    {
        const int tn = min( MixTile, n - t0 );
        for ( int i = 0; i < tn; ++i )
        {
            const std::uint32_t si = static_cast<std::uint32_t>( t0 + i );
            const float         noise = 2.0f * hash01( seed, si, 1 ) - 1.0f;
            s.y += lp * ( noise - s.y );
            s.z += lp * ( s.y - s.z );
            const float kick = ( pc > 0.0f && hash01( seed, si, 2 ) < pc ) ? 2.0f * hash01( seed, si, 3 ) - 1.0f : 0.0f;
            const float qr = fmaf( br * bc, q.x, -br * bs * q.y ), qi = fmaf( br * bc, q.y, br * bs * q.x );
            q.x = qr;
            q.y = qi;
            const float kr = fmaf( cr * cc, q.z, fmaf( -cr * cs, q.w, kick ) ),
                        ki = fmaf( cr * cc, q.w, cr * cs * q.z );
            q.z = kr;
            q.w = ki;
            sV[threadIdx.x][i] = amp * 1.8f * s.z + 0.5f * qi + 0.6f * ki;
        }
        __syncthreads();
        for ( int o = threadIdx.x; o < 4 * tn; o += SurfSegments )
        {   // a sum down the stretches, in their order
            const int ch = o / tn, i = o - ch * tn;
            float     a = 0.0f;
            for ( int k = 0; k < SurfSegments; ++k ) a = fmaf( sG[k][ch], sV[k][i], a );
            out[ch * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    state[2 * seg] = s;
    state[2 * seg + 1] = q;
    atomicAdd( meta + 8, static_cast<int>( foam * 1000.0f ) );
}

/**
 * @brief The weather's reduce: the rain's, the leaves' and the sea's CUDA blocks added sample by sample, in a fixed
 *        order, into the frames' channels 16 .. 19 (near left and right, far left and right).
 * @param[in]     hdr      the block header
 * @param[in]     scratch  the weather's per-block sums
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels
 */
__global__ void weatherReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out )
{
    __shared__ float part[ReduceGroups][ReduceLanes + 1];
    const Weather   &W = hdr->m_weather;
    const int        n = hdr->m_n;
    const int        t = static_cast<int>( blockIdx.x ) * ReduceLanes + static_cast<int>( threadIdx.x );
    const int        c = t / maxN;
    const int        s = t - c * maxN;
    const bool       live = c < 4 && s < n;
    float            acc = 0.0f;
    if ( live && ( W.m_rainOn || W.m_leavesOn || W.m_surfOn ) )
        for ( int k = threadIdx.y; k < WeatherBlocks; k += ReduceGroups )
            acc += scratch[static_cast<size_t>( k ) * 4 * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if ( threadIdx.y == 0 && live )
    {
        for ( int q = 1; q < ReduceGroups; ++q ) acc += part[q][threadIdx.x];
        out[s * CloudOutChannels + 16 + c] = acc;
    }
}

/**
 * @brief A bloom mode's frequency: evenly in hertz as a plate's modes stand (Weyl), each jittered within its place.
 * @param[in] B  the bloom
 * @param[in] k  the mode
 * @return Hz
 */
__device__ __forceinline__ float bloomFreq( const Bloom &B, int k )
{
    return fmaxf( B.m_spacing, 0.05f ) *
           ( static_cast<float>( k ) + 0.5f +
             0.8f * ( hash01( 0xB100B100U, static_cast<std::uint32_t>( k ), 1 ) - 0.5f ) );
}

/** @brief A frequency's band of the bloom's cascade. @param[in] f Hz @return 0 .. BloomBands - 1 */
__device__ __forceinline__ int bloomBand( float f )
{
    return min( max( static_cast<int>( log2f( fmaxf( f, BloomLow ) * ( 1.0f / BloomLow ) ) *
                                       ( static_cast<float>( BloomBands ) / 9.22881869f ) ),   // log2(18000 / 30)
                     0 ),
                BloomBands - 1 );
}

/**
 * @brief The bloom (Phase 28): one thread a mode of the plate, a complex one-pole as the modal hall's, driven by
 *        everything in the field (the voices, the chimes, the weather; near and far together), heard through a phase
 *        of its own in each ear. The cascade from the last block's bands: a band loses each block a share of its
 *        energy -- the drive times its modes' mean energy against a reference (an input at -20 dBFS), a second's
 *        worth eighty bands at full drive, at most half -- as extra damping, and the band above gains it as noise of
 *        that power, spread over its modes. The pitch rises with the plate's energy (Berger), the low modes most.
 * @param[in]     hdr      the block header
 * @param[in]     in       the block's frames (channels 0 .. 3, 10 .. 13, 16 .. 19 are read)
 * @param[in,out] state    per mode its complex state
 * @param[in]     bands    the last block's band energies and mode counts, then the active modes and the total
 * @param[out]    scratch  per CUDA block two channels of sums
 * @param[in]     maxN     the scratch stride
 */
__global__ void __launch_bounds__( BloomThreads )
    bloomKernel( const BlockHeader *hdr, const float *in, float2 *state, const float *bands, float *scratch, int maxN )
{
    __shared__ float sIn[2][BloomTile];
    __shared__ float sAcc[BloomThreads / 32][2][BloomTile];
    const Bloom     &B = hdr->m_bloom;
    const int        n = hdr->m_n;
    const int        k = static_cast<int>( blockIdx.x ) * BloomThreads + static_cast<int>( threadIdx.x );
    float           *out = scratch + static_cast<size_t>( blockIdx.x ) * 2 * maxN;
    if ( !B.m_on )
    {
        state[k] = make_float2( 0.0f, 0.0f );
        for ( int s = threadIdx.x; s < n; s += BloomThreads )
        {
            out[s] = 0.0f;
            out[maxN + s] = 0.0f;
        }
        return;
    }
    const float fs = hdr->m_sampleRate;
    const float top = fminf( BloomHigh, 0.45f * fs );
    const float f0 = bloomFreq( B, k );
    const bool  silent = f0 < BloomLow || f0 > top;
    const int   b = bloomBand( f0 );
    const float sp = fmaxf( B.m_spacing, 0.05f );
    const float active = fmaxf( fminf( top / sp, static_cast<float>( BloomModes ) ) - BloomLow / sp, 1.0f );
    const float eRef = 0.02f / active;   // a mode's energy at an input of -20 dBFS
    const float dtb = static_cast<float>( n ) / fs;
    auto        shareOf = [&]( int band )
    {
        const float e = bands[band] / fmaxf( bands[BloomBands + band], 1.0f );
        return fminf( 0.5f, B.m_drive * 80.0f * ( e / eRef ) * dtb );
    };
    const float out1 = shareOf( b );
    const float in1 = b > 0 ? shareOf( b - 1 ) * bands[b - 1] : 0.0f;
    const float sigma = sqrtf( 0.5f * in1 / ( fmaxf( bands[BloomBands + b], 1.0f ) * static_cast<float>( n ) ) );
    const float loss = __powf( 1.0f - out1, 0.5f / static_cast<float>( n ) );
    const float loud = bands[2 * BloomBands + 1] / ( active * eRef );
    const float bend =
        fminf( fmaxf( 1.0f + B.m_glide * 0.1f * loud * ( 100.0f / fmaxf( f0, BloomLow ) ), 0.25f ), 4.0f );
    const float f = f0 * sqrtf( bend );
    const float T = fmaxf( B.m_decay * sqrtf( 200.0f / fmaxf( f, BloomLow ) ), 0.05f );
    const float r0 = __expf( -6.9077553f / ( T * fs ) ), r = r0 * loss;
    float       sw, cw;
    sincosf( 6.2831853f * fminf( f, 0.49f * fs ) / fs, &sw, &cw );
    const float ar = r * cw, ai = r * sw;
    // The input as if no energy left (the cascade's loss must not open the plate to more of it).
    const float         g = silent ? 0.0f : sqrtf( 2.0f * fmaxf( 1.0f - r0 * r0, 0.0f ) / active );
    const std::uint32_t uk = static_cast<std::uint32_t>( k );
    float               inR, inL;
    __sincosf( 1.5707963f * hash01( 0xB100B100U, uk, 2 ), &inR, &inL );
    const float phL = 6.2831853f * hash01( 0xB100B100U, uk, 3 );
    const float phR = phL + B.m_width * 3.1415927f * ( 2.0f * hash01( 0xB100B100U, uk, 4 ) - 1.0f );
    float       oLs, oLc, oRs, oRc;
    __sincosf( phL, &oLs, &oLc );
    __sincosf( phR, &oRs, &oRc );
    const float lvl = silent ? 0.0f : B.m_level;
    oLs *= lvl;
    oLc *= lvl;
    oRs *= lvl;
    oRc *= lvl;
    float2              y = silent ? make_float2( 0.0f, 0.0f ) : state[k];
    const std::uint32_t seed = hash32( hdr->m_blockIndex * 0x9E3779B9U ^ ( uk * 0x27D4EB2FU ) );
    const float         sn = silent ? 0.0f : sigma * 1.7320508f;   // uniform -1 .. 1 has a third's variance
    const int           lane = static_cast<int>( threadIdx.x ) & 31, warp = static_cast<int>( threadIdx.x ) >> 5;
    for ( int t0 = 0; t0 < n; t0 += BloomTile )
    {
        const int tn = min( BloomTile, n - t0 );
        for ( int s = threadIdx.x; s < tn; s += BloomThreads )
        {
            const float *fr = in + static_cast<size_t>( t0 + s ) * CloudOutChannels;
            sIn[0][s] = fr[0] + fr[2] + fr[10] + fr[12] + fr[16] + fr[18];
            sIn[1][s] = fr[1] + fr[3] + fr[11] + fr[13] + fr[17] + fr[19];
        }
        __syncthreads();
        for ( int i = 0; i < tn; ++i )
        {
            const float x = g * ( inL * sIn[0][i] + inR * sIn[1][i] );
            float       nr = 0.0f, ni = 0.0f;
            if ( sn > 0.0f )
            {
                const std::uint32_t si = static_cast<std::uint32_t>( t0 + i );
                nr = sn * ( 2.0f * hash01( seed, si, 1 ) - 1.0f );
                ni = sn * ( 2.0f * hash01( seed, si, 2 ) - 1.0f );
            }
            const float yr = fmaf( ar, y.x, fmaf( -ai, y.y, x + nr ) );
            const float yi = fmaf( ar, y.y, fmaf( ai, y.x, ni ) );
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
        for ( int s = threadIdx.x; s < 2 * tn; s += BloomThreads )
        {
            const int c = s / tn, i = s - c * tn;
            float     a = 0.0f;
            for ( int w = 0; w < BloomThreads / 32; ++w ) a += sAcc[w][c][i];
            out[c * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    state[k] = ( fabsf( y.x ) + fabsf( y.y ) > 1.0e-12f ) ? y : make_float2( 0.0f, 0.0f );
}

/**
 * @brief The bloom's bands (Phase 28): one CUDA block a band sums its modes' energies (and counts them) in a fixed
 *        order, for the next block's cascade; one more counts the plate's active modes and sums its energy, and
 *        writes the share above 2 kHz into the stats.
 * @param[in]  hdr    the block header
 * @param[in]  state  per mode its complex state after this block
 * @param[out] bands  per band the energy, then per band the modes, then the active modes and the total energy
 * @param[out] stats  [21] the energy's share above 2 kHz
 */
__global__ void __launch_bounds__( 256 )
    bloomBandKernel( const BlockHeader *hdr, const float2 *state, float *bands, float *stats )
{
    __shared__ float sE[256], sN[256];
    const Bloom     &B = hdr->m_bloom;
    if ( !B.m_on ) return;
    const int   b = static_cast<int>( blockIdx.x );
    const float sp = fmaxf( B.m_spacing, 0.05f );
    const float top = fminf( BloomHigh, 0.45f * hdr->m_sampleRate );
    float       e = 0.0f, cnt = 0.0f, hi = 0.0f;
    if ( b < BloomBands )
    {
        const float lo = BloomLow * exp2f( 9.22881869f * static_cast<float>( b ) / static_cast<float>( BloomBands ) );
        const float up =
            BloomLow * exp2f( 9.22881869f * static_cast<float>( b + 1 ) / static_cast<float>( BloomBands ) );
        const int k0 = max( static_cast<int>( lo / sp ) - 2, 0 );
        const int k1 = min( static_cast<int>( up / sp ) + 2, BloomModes - 1 );
        for ( int k = k0 + static_cast<int>( threadIdx.x ); k <= k1; k += 256 )
        {
            const float f = bloomFreq( B, k );
            if ( f < BloomLow || f > top || bloomBand( f ) != b ) continue;
            const float2 y = state[k];
            e += y.x * y.x + y.y * y.y;
            cnt += 1.0f;
        }
    }
    else
    {   // the whole plate: its active modes, its energy, the share above 2 kHz
        for ( int k = static_cast<int>( threadIdx.x ); k < BloomModes; k += 256 )
        {
            const float f = bloomFreq( B, k );
            if ( f < BloomLow || f > top ) continue;
            const float2 y = state[k];
            const float  ek = y.x * y.x + y.y * y.y;
            e += ek;
            cnt += 1.0f;
            if ( f > 2000.0f ) hi += ek;
        }
    }
    sE[threadIdx.x] = e;
    sN[threadIdx.x] = cnt;
    __syncthreads();
    for ( int s = 128; s > 0; s >>= 1 )
    {
        if ( static_cast<int>( threadIdx.x ) < s )
        {
            sE[threadIdx.x] += sE[threadIdx.x + s];
            sN[threadIdx.x] += sN[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float total = sE[0], count = sN[0];
    __syncthreads();
    if ( b == BloomBands )
    {
        sE[threadIdx.x] = hi;
        __syncthreads();
        for ( int s = 128; s > 0; s >>= 1 )
        {
            if ( static_cast<int>( threadIdx.x ) < s ) sE[threadIdx.x] += sE[threadIdx.x + s];
            __syncthreads();
        }
        if ( threadIdx.x == 0 )
        {
            bands[2 * BloomBands] = count;
            bands[2 * BloomBands + 1] = total;
            stats[21] = total > 0.0f ? sE[0] / total : 0.0f;
        }
    }
    else if ( threadIdx.x == 0 )
    {
        bands[b] = total;
        bands[BloomBands + b] = count;
    }
}

/**
 * @brief The bloom's reduce: the CUDA blocks' sums added sample by sample, in a fixed order, into the frames'
 *        channels 20 and 21.
 * @param[in]     hdr      the block header
 * @param[in]     scratch  the bloom kernel's per-block sums
 * @param[in]     maxN     the scratch stride
 * @param[in,out] out      n frames of CloudOutChannels
 */
__global__ void bloomReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out )
{
    __shared__ float part[ReduceGroups][ReduceLanes + 1];
    const int        n = hdr->m_n;
    const int        t = static_cast<int>( blockIdx.x ) * ReduceLanes + static_cast<int>( threadIdx.x );
    const int        c = t / maxN;
    const int        s = t - c * maxN;
    const bool       live = c < 2 && s < n;
    float            acc = 0.0f;
    if ( live && hdr->m_bloom.m_on )
        for ( int k = threadIdx.y; k < BloomBlocks; k += ReduceGroups )
            acc += scratch[static_cast<size_t>( k ) * 2 * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if ( threadIdx.y == 0 && live )
    {
        for ( int q = 1; q < ReduceGroups; ++q ) acc += part[q][threadIdx.x];
        out[s * CloudOutChannels + 20 + c] = acc;
    }
}

/**
 * @brief The swarmalators' forces (Phase 28; O'Keeffe, Ha and Strogatz 2017, with A = B = 1), a chunk of the partners
 *        at a time: every one pulled to every other by 1 + J cos(their phases' difference), pushed off by one over their
 *        distance squared (so they keep apart), its phase pulled by sin(the difference) over the distance. The
 *        partners split into SwarmChunks chunks, one CUDA block a chunk and a tile of swarmalators -- 64 blocks (all
 *        pairs in four blocks were bound by latency on four multiprocessors: 60 us a block). The step sums the chunks.
 * @param[in]  hdr      the block header
 * @param[in]  a        the swarm (x, y, phase, -), buffer A
 * @param[in]  b        buffer B
 * @param[out] partial  per chunk and swarmalator the sums (pull across, pull from near to far, phase pull, -)
 */
__global__ void __launch_bounds__( SwarmThreads )
    swarmForceKernel( const BlockHeader *hdr, const float4 *a, const float4 *b, float4 *partial )
{
    __shared__ float4 tile[SwarmN / SwarmChunks];
    const SwarmSpace &S = hdr->m_swarm;
    if ( !S.m_on || S.m_reset ) return;
    constexpr int Per = SwarmN / SwarmChunks;
    const int     i = static_cast<int>( blockIdx.x ) * SwarmThreads + static_cast<int>( threadIdx.x );
    const int     chunk = static_cast<int>( blockIdx.y );
    const float4 *src = ( hdr->m_blockIndex & 1U ) ? b : a;
    if ( static_cast<int>( threadIdx.x ) < Per )
    {   // the chunk with its phases as cosine and sine (cos and sin of a difference then by the identities)
        const float4 o = src[chunk * Per + static_cast<int>( threadIdx.x )];
        float        os, oc;
        __sincosf( o.z, &os, &oc );
        tile[threadIdx.x] = make_float4( o.x, o.y, oc, os );
    }
    __syncthreads();
    const float4 me = src[i];
    float        ms, mc;
    __sincosf( me.z, &ms, &mc );
    float vx = 0.0f, vy = 0.0f, dth = 0.0f;
    for ( int j = 0; j < Per; ++j )
    {
        const float4 o = tile[j];
        const float  dx = o.x - me.x, dy = o.y - me.y;
        const float  inv = rsqrtf( dx * dx + dy * dy + 1.0e-4f );
        const float  c = o.z * mc + o.w * ms;   // cos(theta_j - theta_i)
        const float  s = o.w * mc - o.z * ms;   // sin(theta_j - theta_i)
        const float  pull = ( 1.0f + S.m_j * c ) * inv - inv * inv;
        vx = fmaf( dx, pull, vx );
        vy = fmaf( dy, pull, vy );
        dth = fmaf( s, inv, dth );
    }
    partial[static_cast<size_t>( chunk ) * SwarmN + static_cast<size_t>( i )] = make_float4( vx, vy, dth, 0.0f );
}

/**
 * @brief One step of the swarmalators (Phase 28): the chunks' forces summed in their order, every one moved and its
 *        phase turned at its own rate and pulled by K; the step is the block's time by Speed. Two buffers, by the
 *        block's parity.
 * @param[in]     hdr      the block header
 * @param[in,out] a        the swarm (x, y, phase, -), buffer A
 * @param[in,out] b        buffer B
 * @param[in]     partial  the forces by chunk (swarmForceKernel)
 */
__global__ void __launch_bounds__( SwarmThreads )
    swarmStepKernel( const BlockHeader *hdr, float4 *a, float4 *b, const float4 *partial )
{
    const SwarmSpace &S = hdr->m_swarm;
    if ( !S.m_on ) return;
    const int           i = static_cast<int>( blockIdx.x ) * SwarmThreads + static_cast<int>( threadIdx.x );
    const bool          odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4       *src = odd ? b : a;
    float4             *dst = odd ? a : b;
    const std::uint32_t ui = static_cast<std::uint32_t>( i );
    if ( S.m_reset )
    {   // scattered over the unit disc, at random phases
        const float r = sqrtf( hash01( 0x5A4A5A4AU, ui, 1 ) ), ang = 6.2831853f * hash01( 0x5A4A5A4AU, ui, 2 );
        dst[i] = make_float4( r * __cosf( ang ), r * __sinf( ang ), 6.2831853f * hash01( 0x5A4A5A4AU, ui, 3 ), 0.0f );
        return;
    }
    const float4 me = src[i];
    float        vx = 0.0f, vy = 0.0f, dth = 0.0f;
    for ( int c = 0; c < SwarmChunks; ++c )
    {
        const float4 p = partial[static_cast<size_t>( c ) * SwarmN + static_cast<size_t>( i )];
        vx += p.x;
        vy += p.y;
        dth += p.z;
    }
    const float h = S.m_speed * static_cast<float>( hdr->m_n ) / hdr->m_sampleRate;
    const float omega = S.m_spread * ( 2.0f * hash01( 0x5A4A5A4AU, ui, 4 ) - 1.0f );
    const float inv = 1.0f / static_cast<float>( SwarmN );
    float       th = me.z + h * ( omega + S.m_k * dth * inv );
    th -= 6.2831853f * floorf( th * 0.15915494f );
    float x = me.x + h * vx * inv, y = me.y + h * vy * inv;
    if ( !( fabsf( x ) < 100.0f && fabsf( y ) < 100.0f ) )
    {   // never lost: one gone wrong starts again near the middle
        x = 0.1f * ( 2.0f * hash01( 0x5A4A5A4AU, ui, 5 ) - 1.0f );
        y = 0.1f * ( 2.0f * hash01( 0x5A4A5A4AU, ui, 6 ) - 1.0f );
    }
    dst[i] = make_float4( x, y, th, 0.0f );
}

/**
 * @brief The swarm as the partials ride it (Phase 28): one CUDA block of SwarmN threads finds its middle and its
 *        size (the root mean square of the distances), and writes every swarmalator's place in the field, -1 .. 1
 *        across and from near to far (the swarm fills 2/3 of the field's reach), and its phase. And its orders:
 *        R, how much the phases agree, and S, how much the phase follows the angle round the middle (a phase wave).
 * @param[in]  hdr    the block header
 * @param[in]  a      the swarm, buffer A
 * @param[in]  b      buffer B
 * @param[out] view   per swarmalator (across, near to far, phase, -)
 * @param[out] stats  [22] R, [23] S
 */
__global__ void __launch_bounds__( SwarmN )
    swarmViewKernel( const BlockHeader *hdr, const float4 *a, const float4 *b, float4 *view, float *stats )
{
    __shared__ float  sx[SwarmN], sy[SwarmN], sz[SwarmN];
    const SwarmSpace &S = hdr->m_swarm;
    if ( !S.m_on ) return;
    const int    i = static_cast<int>( threadIdx.x );
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4 me = ( odd ? a : b )[i];   // what the step just wrote
    auto         sum3 = [&]( float x, float y, float z, float &ox, float &oy, float &oz )
    {
        sx[i] = x;
        sy[i] = y;
        sz[i] = z;
        __syncthreads();
        for ( int s = SwarmN / 2; s > 0; s >>= 1 )
        {
            if ( i < s )
            {
                sx[i] += sx[i + s];
                sy[i] += sy[i + s];
                sz[i] += sz[i + s];
            }
            __syncthreads();
        }
        ox = sx[0];
        oy = sy[0];
        oz = sz[0];
        __syncthreads();
    };
    float cx, cy, unused;
    sum3( me.x, me.y, 0.0f, cx, cy, unused );
    cx *= 1.0f / SwarmN;
    cy *= 1.0f / SwarmN;
    const float dx = me.x - cx, dy = me.y - cy;
    float       r2, rs, rc;
    float       ts, tc;
    __sincosf( me.z, &ts, &tc );
    sum3( dx * dx + dy * dy, ts, tc, r2, rs, rc );
    const float radius = sqrtf( fmaxf( r2 * ( 1.0f / SwarmN ), 1.0e-8f ) );
    const float k = 1.0f / ( 1.5f * radius );
    view[i] = make_float4( fminf( fmaxf( dx * k, -1.0f ), 1.0f ), fminf( fmaxf( dy * k, -1.0f ), 1.0f ), me.z, 0.0f );
    // S: the phase against the angle round the middle, either way round.
    const float phi = atan2f( dy, dx );
    float       ps, pc, ms, mc;
    __sincosf( phi + me.z, &ps, &pc );
    __sincosf( phi - me.z, &ms, &mc );
    float a1, a2, a3, b1, b2, b3;
    sum3( ps, pc, 0.0f, a1, a2, a3 );
    sum3( ms, mc, 0.0f, b1, b2, b3 );
    if ( i == 0 )
    {
        stats[22] = sqrtf( rs * rs + rc * rc ) * ( 1.0f / SwarmN );
        stats[23] = fmaxf( sqrtf( a1 * a1 + a2 * a2 ), sqrtf( b1 * b1 + b2 * b2 ) ) * ( 1.0f / SwarmN );
    }
}

/**
 * @brief The physarum's trail (Phase 28): each block it spreads (a share to the mean of its 3 x 3 neighbourhood),
 *        fades, takes in what the agents left in the last block (whole numbers, so in any order the same) and the
 *        food at the notes' harmonics -- a line across the field at each, two cells wide, a fifth of a visit a block
 *        (stronger, it held the agents on the lines: measured 0.1 % of the power between the harmonics; so, with a
 *        sensor of 200 cents, a quarter). Two buffers, by the block's parity; the update kernel reads the one written
 *        here.
 * @param[in]     hdr      the block header
 * @param[in,out] trail    two buffers of PhysCells
 * @param[in,out] deposit  per cell what the agents left, 256 a visit (cleared here)
 */
__global__ void physTrailKernel( const BlockHeader *hdr, float *trail, unsigned int *deposit )
{
    const Physarum &P = hdr->m_phys;
    if ( !P.m_on ) return;
    const int idx = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= PhysCells ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *src = trail + ( odd ? PhysCells : 0 );
    float       *dst = trail + ( odd ? 0 : PhysCells );
    if ( P.m_fresh )
    {
        dst[idx] = 0.0f;
        deposit[idx] = 0U;
        return;
    }
    const int x = idx % PhysW, y = idx / PhysW;
    float     sum = 0.0f;
    for ( int dy = -1; dy <= 1; ++dy )
        for ( int dx = -1; dx <= 1; ++dx )
        {
            const int xx = min( max( x + dx, 0 ), PhysW - 1 ), yy = min( max( y + dy, 0 ), PhysH - 1 );
            sum += src[yy * PhysW + xx];
        }
    float v = ( 1.0f - P.m_spread ) * src[idx] + P.m_spread * sum * ( 1.0f / 9.0f );
    v = v * P.m_keep + static_cast<float>( deposit[idx] ) * ( 1.0f / 256.0f );
    const float fx = static_cast<float>( x ) + 0.5f;
    for ( int k = 0; k < min( P.m_count, PhysFood ); ++k )
    {
        const float d = fabsf( fx - P.m_food[k] );
        if ( d < 2.0f ) v += 0.2f * P.m_foodWeight[k] * ( 1.0f - 0.5f * d );
    }
    dst[idx] = v;
    deposit[idx] = 0U;
}

/**
 * @brief The physarum's trail in two numbers (Phase 28): its mean and its spread (standard deviation over the
 *        mean), one CUDA block of 1024 over the trail just written.
 * @param[in]  hdr    the block header
 * @param[in]  trail  the two buffers
 * @param[out] stats  [24] the mean, [25] the spread
 */
__global__ void __launch_bounds__( 1024 ) physStatsKernel( const BlockHeader *hdr, const float *trail, float *stats )
{
    __shared__ float s1[1024], s2[1024];
    const Physarum  &P = hdr->m_phys;
    if ( !P.m_on ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *t = trail + ( odd ? 0 : PhysCells );
    float        a = 0.0f, b = 0.0f;
    for ( int k = static_cast<int>( threadIdx.x ); k < PhysCells; k += 1024 )
    {
        a += t[k];
        b += t[k] * t[k];
    }
    s1[threadIdx.x] = a;
    s2[threadIdx.x] = b;
    __syncthreads();
    for ( int s = 512; s > 0; s >>= 1 )
    {
        if ( static_cast<int>( threadIdx.x ) < s )
        {
            s1[threadIdx.x] += s1[threadIdx.x + s];
            s2[threadIdx.x] += s2[threadIdx.x + s];
        }
        __syncthreads();
    }
    if ( threadIdx.x == 0 )
    {
        const float mean = s1[0] / PhysCells, var = fmaxf( s2[0] / PhysCells - mean * mean, 0.0f );
        stats[24] = mean;
        stats[25] = mean > 1.0e-9f ? sqrtf( var ) / mean : 0.0f;
    }
}

// ---- Phase 29 (16.10.2026): the topological halo, the wave room.

constexpr int HaloThreads = 1024;   ///< a halo carrier's CUDA block: a thread a component
constexpr int RoomFirTaps = 64;     ///< the wave room's low pass, down and up
constexpr int RoomHist = 128;       ///< samples of history a source keeps for the low pass
static_assert( HaloComponents <= HaloThreads, "a carrier's lattice in one CUDA block" );
static_assert( RoomFirTaps % RoomDecim == 0, "the interpolation's phases are whole" );

/** @brief The wave room's low pass (a windowed sinc at 2.2 kHz, unity gain at DC). */
__constant__ float cRoomFir[RoomFirTaps];

/**
 * @brief The halo's lattice in one carrier, one block (Phase 29): dpsi/dt = -i omega H psi - damp psi + drive, H Qi,
 *        Wu and Zhang's: u sigma_z at every site (and the inside's disorder), hopping T_x = (sigma_z - i sigma_x) / 2
 *        and T_y = (sigma_z - i sigma_y) / 2 to the next site across and along (their adjoints back). Fourth-order
 *        Runge-Kutta over the block, in shared memory; one CUDA block a carrier. Its energy, where round the disc it
 *        stands and its share on the rim go into the stats.
 * @param[in]     hdr    the block header
 * @param[in,out] state  per carrier HaloComponents complex amplitudes
 * @param[out]    stats  [32 + c] energy, [40 + c] angle, [48 + c] rim share
 */
__global__ void __launch_bounds__( HaloThreads ) haloKernel( const BlockHeader *hdr, float2 *state, float *stats )
{
    __shared__ float2 sPsi[HaloComponents], sAcc[HaloComponents], sTmp[HaloComponents];
    const Halo       &H = hdr->m_halo;
    const int         c = static_cast<int>( blockIdx.x ), t = static_cast<int>( threadIdx.x );
    float2           *psi = state + static_cast<size_t>( c ) * HaloComponents;
    if ( !H.m_on || ( H.m_clear >> c ) & 1 )
    {
        if ( t < HaloComponents ) psi[t] = make_float2( 0.0f, 0.0f );
        if ( t == 0 && H.m_on ) stats[32 + c] = stats[40 + c] = stats[48 + c] = 0.0f;
        return;
    }
    const int   s = t >> 1, o = t & 1;
    const bool  live = t < HaloComponents;
    const float dt = static_cast<float>( hdr->m_n ) / hdr->m_sampleRate;
    const float om = H.m_omega;
    // The site's own potential: the mass on its orbital, the disorder (a fixed pattern inside).
    float onsite = 0.0f, drive = 0.0f;
    if ( live )
    {
        const int   x = cHaloXY[s][0], y = cHaloXY[s][1];
        const bool  inside = x * x + y * y < ( HaloRadius - 2 ) * ( HaloRadius - 2 );
        const float w = 2.0f * hash01( 0x4A10A10U, static_cast<std::uint32_t>( s ), 1 ) - 1.0f;
        onsite = ( o == 0 ? H.m_mass : -H.m_mass ) + ( inside ? H.m_disorder * w : 0.0f );
        if ( s == H.m_site[c] ) drive = H.m_drive[c];
        sPsi[t] = psi[t];
    }
    __syncthreads();
    // dpsi/dt for the component t, from `in`.
    auto deriv = [&]( const float2 *in )
    {
        // H psi: on site, then T_x psi(s - x) + T_x^+ psi(s + x) + T_y psi(s - y) + T_y^+ psi(s + y), row o.
        float     hr = onsite * in[t].x, hi = onsite * in[t].y;
        const int ax = cHaloNbr[s][1], bx = cHaloNbr[s][0], ay = cHaloNbr[s][3], by = cHaloNbr[s][2];
        auto      add = [&]( int site, float ar, float ai, float br, float bi )
        {   // + (a, b) . psi(site): a times its first orbital, b times its second
            if ( site < 0 ) return;
            const float2 p0 = in[2 * site], p1 = in[2 * site + 1];
            hr += ar * p0.x - ai * p0.y + br * p1.x - bi * p1.y;
            hi += ar * p0.y + ai * p0.x + br * p1.y + bi * p1.x;
        };
        if ( o == 0 )
        {
            add( ax, 0.5f, 0.0f, 0.0f, -0.5f );   // T_x row 0: (1/2, -i/2)
            add( bx, 0.5f, 0.0f, 0.0f, 0.5f );    // T_x^+ row 0: (1/2, i/2)
            add( ay, 0.5f, 0.0f, -0.5f, 0.0f );   // T_y row 0: (1/2, -1/2)
            add( by, 0.5f, 0.0f, 0.5f, 0.0f );    // T_y^+ row 0: (1/2, 1/2)
        }
        else
        {
            add( ax, 0.0f, -0.5f, -0.5f, 0.0f );   // T_x row 1: (-i/2, -1/2)
            add( bx, 0.0f, 0.5f, -0.5f, 0.0f );    // T_x^+ row 1: (i/2, -1/2)
            add( ay, 0.5f, 0.0f, -0.5f, 0.0f );    // T_y row 1: (1/2, -1/2)
            add( by, -0.5f, 0.0f, -0.5f, 0.0f );   // T_y^+ row 1: (-1/2, -1/2)
        }
        // -i omega H psi - damp psi + drive (on the first orbital in phase, the second a quarter on)
        return make_float2( om * hi - H.m_damp * in[t].x + ( o == 0 ? drive : 0.0f ),
                            -om * hr - H.m_damp * in[t].y + ( o == 1 ? drive : 0.0f ) );
    };
    const float w[4] = { 1.0f / 6.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 6.0f };
    const float a[3] = { 0.5f, 0.5f, 1.0f };
    // Substeps when the block is long for the speed (the spectrum reaches about 7 omega; fourth-order Runge-Kutta
    // holds to 2.8): one at 128 samples and any speed the knob gives.
    const int   subs = max( 1, static_cast<int>( ceilf( om * 7.0f * dt ) ) );
    const float h = dt / static_cast<float>( subs );
    for ( int sub = 0; sub < subs; ++sub )
    {
        if ( live ) sAcc[t] = make_float2( 0.0f, 0.0f );
        const float2 *from = sPsi;
        for ( int stage = 0; stage < 4; ++stage )
        {
            float2 k = make_float2( 0.0f, 0.0f );
            if ( live ) k = deriv( from );
            __syncthreads();
            if ( live )
            {
                sAcc[t].x += w[stage] * k.x;
                sAcc[t].y += w[stage] * k.y;
                if ( stage < 3 )
                    sTmp[t] = make_float2( sPsi[t].x + a[stage] * h * k.x, sPsi[t].y + a[stage] * h * k.y );
            }
            __syncthreads();
            from = sTmp;
        }
        if ( live ) sPsi[t] = make_float2( sPsi[t].x + h * sAcc[t].x, sPsi[t].y + h * sAcc[t].y );
        __syncthreads();
    }
    float2 next = make_float2( 0.0f, 0.0f );
    if ( live )
    {
        next = sPsi[t];
        if ( !( fabsf( next.x ) < 1.0e6f && fabsf( next.y ) < 1.0e6f ) ) next = make_float2( 0.0f, 0.0f );
        psi[t] = next;
    }
    // Its energy, its middle round the disc, its rim.
    float e = live ? next.x * next.x + next.y * next.y : 0.0f, ex = 0.0f, ey = 0.0f, rim = 0.0f;
    if ( live )
    {
        const int x = cHaloXY[s][0], y = cHaloXY[s][1];
        ex = e * static_cast<float>( x );
        ey = e * static_cast<float>( y );
        rim = x * x + y * y > ( HaloRadius - 2 ) * ( HaloRadius - 2 ) ? e : 0.0f;
    }
    float r4[4] = { e, ex, ey, rim };
    for ( int k = 0; k < 4; ++k )
        for ( int off = 16; off > 0; off >>= 1 ) r4[k] += __shfl_down_sync( 0xFFFFFFFFU, r4[k], off );
    __shared__ float sSum[4][32];
    if ( ( t & 31 ) == 0 )
        for ( int k = 0; k < 4; ++k ) sSum[k][t >> 5] = r4[k];
    __syncthreads();
    if ( t == 0 )
    {
        float tot[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for ( int wv = 0; wv < HaloThreads / 32; ++wv )
            for ( int k = 0; k < 4; ++k ) tot[k] += sSum[k][wv];
        stats[32 + c] = tot[0];
        stats[40 + c] = atan2f( tot[2], tot[1] );
        stats[48 + c] = tot[0] > 0.0f ? tot[3] / tot[0] : 0.0f;
    }
}

/**
 * @brief The wave room's sources (Phase 29): one CUDA block a source -- its voice's block, near and far, left and
 *        right together, summed over the voice's chunks in a fixed order -- down to the room's rate through the low
 *        pass (its history kept per source).
 * @param[in]     hdr       the block header
 * @param[in]     scratch   the render kernel's per-chunk sums
 * @param[in]     perVoice  chunks a voice has
 * @param[in]     maxN      the scratch stride
 * @param[in,out] hist      per source RoomHist samples of history
 * @param[out]    in        per source the block's samples at the room's rate
 */
__global__ void __launch_bounds__( 128 )
    roomSourceKernel( const BlockHeader *hdr, const float *scratch, int perVoice, int maxN, float *hist, float *in )
{
    __shared__ float sX[RoomHist + 4096 / RoomDecim * RoomDecim];
    const WaveRoom  &W = hdr->m_room;
    const int        src = static_cast<int>( blockIdx.x );
    if ( !W.m_on || src >= W.m_count ) return;
    const int n = hdr->m_n;
    const int v = W.m_voiceOf[src];
    float    *h = hist + static_cast<size_t>( src ) * RoomHist;
    for ( int i = threadIdx.x; i < RoomHist; i += blockDim.x ) sX[i] = h[i];
    for ( int i = threadIdx.x; i < n; i += blockDim.x )
    {
        float acc = 0.0f;
        if ( v >= 0 && v < hdr->m_voices )
            for ( int k = v * perVoice; k < ( v + 1 ) * perVoice; ++k )
            {
                const float *ch = scratch + static_cast<size_t>( k ) * CloudChannels * maxN;
                acc += ch[i] + ch[maxN + i] + ch[2 * maxN + i] + ch[3 * maxN + i];
            }
        sX[RoomHist + i] = 0.5f * acc * W.m_gain[src];
    }
    __syncthreads();
    const int steps = n / RoomDecim;
    for ( int j = threadIdx.x; j < steps; j += blockDim.x )
    {   // the low pass at the room's sample j: the input up to RoomHist + (j + 1) RoomDecim - 1
        const int end = RoomHist + ( j + 1 ) * RoomDecim - 1;
        float     y = 0.0f;
        for ( int k = 0; k < RoomFirTaps; ++k ) y += cRoomFir[k] * sX[end - k];
        in[static_cast<size_t>( src ) * ( 4096 / RoomDecim ) + j] = y;
    }
    __syncthreads();
    for ( int i = threadIdx.x; i < RoomHist; i += blockDim.x ) h[i] = sX[n + i];
}

/**
 * @brief The three pressure fields of a room step: before, now, next, by the step's place in the room's time.
 * @param[in] hdr   the block header
 * @param[in] step  the step in the block
 * @return 0 .. 2: the field `now` is in (before is the one before it, next the one after)
 */
__device__ __forceinline__ int roomNow( const BlockHeader *hdr, int step )
{
    const int steps = ( hdr->m_n / RoomDecim ) % 3;
    return static_cast<int>(
        ( ( hdr->m_blockIndex % 3U ) * static_cast<std::uint32_t>( steps ) + static_cast<std::uint32_t>( step ) + 1U ) %
        3U );
}

/**
 * @brief One step of the wave room (Phase 29): the standard leapfrog scheme, p+ = (2 - K l^2) p + l^2 (the sum of the
 *        K neighbours inside) - p-, at a wall with the walls' admittance (Bilbao's finite-volume form: a cell with
 *        6 - K faces on a wall takes (1 + l b (6 - K) / 2) p+ on the left and (1 - l b (6 - K) / 2) p- on the right).
 *        The sources add at their cells; the ears read theirs.
 * @param[in]     hdr     the block header
 * @param[in]     step    the step in the block, 0 .. n / RoomDecim - 1
 * @param[in,out] field   three fields of RoomCells
 * @param[in]     in      per source the block's samples at the room's rate
 * @param[out]    ears    two rows (left, right) of the block's samples at the room's rate
 */
__global__ void roomStepKernel( const BlockHeader *hdr, int step, float *field, const float *in, float *ears )
{
    const WaveRoom &W = hdr->m_room;
    if ( !W.m_on || step >= hdr->m_n / RoomDecim ) return;
    const int idx = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= RoomCells ) return;
    const int x = idx % RoomNX, y = ( idx / RoomNX ) % RoomNY, z = idx / ( RoomNX * RoomNY );
    if ( x >= W.m_nx || y >= W.m_ny || z >= W.m_nz ) return;
    const int    now = roomNow( hdr, step );
    const float *p = field + static_cast<size_t>( now ) * RoomCells;
    const float *pm = field + static_cast<size_t>( ( now + 2 ) % 3 ) * RoomCells;
    float       *pn = field + static_cast<size_t>( ( now + 1 ) % 3 ) * RoomCells;
    if ( W.m_clear )
    {
        pn[idx] = 0.0f;
        return;
    }
    float sum = 0.0f;
    int   K = 0;
    if ( x > 0 ) sum += p[idx - 1], ++K;
    if ( x + 1 < W.m_nx ) sum += p[idx + 1], ++K;
    if ( y > 0 ) sum += p[idx - RoomNX], ++K;
    if ( y + 1 < W.m_ny ) sum += p[idx + RoomNX], ++K;
    if ( z > 0 ) sum += p[idx - RoomNX * RoomNY], ++K;
    if ( z + 1 < W.m_nz ) sum += p[idx + RoomNX * RoomNY], ++K;
    const float l2 = W.m_lambda2, lb = sqrtf( l2 ) * W.m_beta * 0.5f * static_cast<float>( 6 - K );
    float v = ( ( 2.0f - static_cast<float>( K ) * l2 ) * p[idx] + l2 * sum - ( 1.0f - lb ) * pm[idx] ) / ( 1.0f + lb );
    if ( z == W.m_srcZ )
        for ( int s = 0; s < min( W.m_count, RoomSources ); ++s )
            if ( W.m_source[s] == idx ) v += in[static_cast<size_t>( s ) * ( 4096 / RoomDecim ) + step];
    if ( !( fabsf( v ) < 1.0e4f ) ) v = 0.0f;   // never blown up
    pn[idx] = v;
    if ( idx == W.m_ear[0] ) ears[step] = v;
    if ( idx == W.m_ear[1] ) ears[4096 / RoomDecim + step] = v;
}

/**
 * @brief The wave room's ears (Phase 29): up to the sample rate through the low pass (each output sample a phase of
 *        it over the room's last samples, the history kept), into the frames' channels 22 and 23.
 * @param[in]     hdr   the block header
 * @param[in]     ears  the block's samples at the room's rate, two rows
 * @param[in,out] hist  per ear the room's last RoomFirTaps / RoomDecim samples
 * @param[in,out] out   n frames of CloudOutChannels
 */
__global__ void __launch_bounds__( 256 )
    roomOutKernel( const BlockHeader *hdr, const float *ears, float *hist, float *out )
{
    __shared__ float sE[2][RoomFirTaps / RoomDecim + 4096 / RoomDecim];
    const WaveRoom  &W = hdr->m_room;
    const int        n = hdr->m_n, steps = n / RoomDecim, past = RoomFirTaps / RoomDecim;
    if ( !W.m_on )
    {
        for ( int i = threadIdx.x; i < n; i += blockDim.x )
            out[i * CloudOutChannels + 22] = out[i * CloudOutChannels + 23] = 0.0f;
        return;
    }
    for ( int e = 0; e < 2; ++e )
    {
        for ( int i = threadIdx.x; i < past; i += blockDim.x ) sE[e][i] = hist[e * past + i];
        for ( int i = threadIdx.x; i < steps; i += blockDim.x ) sE[e][past + i] = ears[e * ( 4096 / RoomDecim ) + i];
    }
    __syncthreads();
    for ( int i = threadIdx.x; i < 2 * n; i += blockDim.x )
    {
        const int e = i / n, s = i - e * n;
        const int m = s / RoomDecim, ph = s - m * RoomDecim;
        float     y = 0.0f;
        for ( int k = 0; k < past; ++k ) y += cRoomFir[k * RoomDecim + ph] * sE[e][past + m - k];
        out[s * CloudOutChannels + 22 + e] = y * static_cast<float>( RoomDecim ) * W.m_level;
    }
    __syncthreads();
    for ( int e = 0; e < 2; ++e )
        for ( int i = threadIdx.x; i < past; i += blockDim.x ) hist[e * past + i] = sE[e][steps + i];
}

/**
 * @brief The Phase 28 counters into the stats, and cleared for the next block: the drops sounding (kept for the
 *        next block's odds too), born, the leaves' mean rustle, the sea's mean foam.
 * @param[in]     hdr    the block header
 * @param[in]     wind   the wind's lattice after this block's steps (its mean speed is measured)
 * @param[in,out] meta   the counters
 * @param[out]    stats  [16] drops sounding, [17] born, [18] the wind's mean speed, [19] the rustle, [20] the foam
 */
__global__ void __launch_bounds__( 256 )
    phase28StatsKernel( const BlockHeader *hdr, const float *wind, int *meta, float *stats )
{
    __shared__ float sv[256];
    float            v = 0.0f;
    if ( hdr->m_weather.m_wind )
        for ( int idx = static_cast<int>( threadIdx.x ) * 8; idx < WindCells; idx += 256 * 8 )   // every eighth cell
        {
            const float2 u = windAt( wind, idx );
            v += sqrtf( u.x * u.x + u.y * u.y );
        }
    sv[threadIdx.x] = v;
    __syncthreads();
    for ( int s = 128; s > 0; s >>= 1 )
    {
        if ( static_cast<int>( threadIdx.x ) < s ) sv[threadIdx.x] += sv[threadIdx.x + s];
        __syncthreads();
    }
    if ( threadIdx.x != 0 ) return;
    stats[16] = static_cast<float>( meta[4] );
    stats[17] = static_cast<float>( meta[6] );
    stats[18] = sv[0] / ( WindCells / 8 );
    stats[19] = static_cast<float>( meta[7] ) * ( 1.0f / 250.0f ) / LeafClusters;
    stats[20] = static_cast<float>( meta[8] ) * ( 1.0f / 1000.0f ) / SurfSegments;
    meta[5] = meta[4];
    meta[4] = 0;
    meta[6] = 0;
    meta[7] = 0;
    meta[8] = 0;
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
    bool          m_snapAsked = false;           ///< the next launch brings it down
    bool          m_snapInFlight = false;        ///< the launched block brings it down
    bool          m_snapReady = false;           ///< a snapshot came down and was not taken yet
    Binaural      m_binaural;                    ///< the host's copy of the binaural placement
    SpectralEcho  m_echo;                        ///< the host's copy of the spectral echo's settings
    EnvShape     *m_dShapes = nullptr;           ///< device: the envelope bank, EnvShapes
    float        *m_dLand = nullptr;             ///< device: the landscape, room for LandMaxColumns * LandMaxRows
    float2       *m_dModal = nullptr;            ///< device: the modal hall's states, ModalModes
    float        *m_dGainRing = nullptr;         ///< device: per voice GainRing levels, for the dispersion
    SpectralField m_field;                       ///< the host's copy of the field and the dispersion
    std::uint32_t m_blockIndex = 0;              ///< blocks launched since open()
    float        *m_dModalScratch = nullptr;     ///< device: per modal CUDA block two channels of sums
    float        *m_dWindA = nullptr;            ///< device: the wind's lattice, nine directions a cell, buffer A
    float        *m_dWindB = nullptr;            ///< device: buffer B
    float4       *m_dChimeState = nullptr;       ///< device: per chime its clapper (angle, speed, slow wind)
    float2       *m_dChimeHit = nullptr;         ///< device: per chime this block's strike
    float2       *m_dChimeModes = nullptr;       ///< device: per chime mode its complex state
    float        *m_dChimeScratch = nullptr;     ///< device: per chime CUDA block four channels of sums
    float        *m_dDreamAcc = nullptr;         ///< device: the hall's bands gathered for the dream's next memory
    float        *m_dDreamMem = nullptr;         ///< device: the dream's memories, DreamBands x DreamMemories
    float        *m_dDreamState = nullptr;       ///< device: the dream's state, DreamBands
    float        *m_dDreamSims = nullptr;        ///< device: the memories' likenesses, two sets
    int          *m_dMeta = nullptr;             ///< device: the counters (Phase 27: memories, strikes, next slot; 28)
    float        *m_dShowers = nullptr;          ///< device: the showers' density, two buffers of WindCells
    float4       *m_dDrop = nullptr;             ///< device: per rain slot its ring
    int2         *m_dBirth = nullptr;            ///< device: per rain slot when its drop fell
    float2       *m_dDropAt = nullptr;           ///< device: per rain slot where its drop fell
    float4       *m_dLeaf = nullptr;             ///< device: per leaf cluster its ring and its slow wind
    float4       *m_dSurf = nullptr;             ///< device: per stretch of shore two entries
    float        *m_dWeatherScratch = nullptr;   ///< device: per weather CUDA block four channels of sums
    float2       *m_dBloom = nullptr;            ///< device: per bloom mode its complex state
    float        *m_dBloomBands = nullptr;       ///< device: the bloom's bands, the active modes, the energy
    float        *m_dBloomScratch = nullptr;     ///< device: per bloom CUDA block two channels of sums
    float4       *m_dSwarmA = nullptr;           ///< device: the swarmalators, buffer A
    float4       *m_dSwarmB = nullptr;           ///< device: buffer B
    float4       *m_dSwarmView = nullptr;        ///< device: the swarm as the partials ride it
    float4       *m_dSwarmForce = nullptr;       ///< device: the swarm's forces by chunk of partners
    float        *m_dPhysTrail = nullptr;        ///< device: the physarum's trail, two buffers of PhysCells
    unsigned int *m_dPhysDeposit = nullptr;      ///< device: what its agents left, per cell
    Weather       m_weather;                     ///< the host's copy of the weather's settings
    Bloom         m_bloom;                       ///< the host's copy of the bloom's settings
    SwarmSpace    m_swarm;                       ///< the host's copy of the swarmalators' settings
    Physarum      m_phys;                        ///< the host's copy of the physarum's settings and food
    float2       *m_dHalo = nullptr;             ///< device: the halo's lattices, HaloCarriers x HaloComponents
    float        *m_dRoom = nullptr;             ///< device: the wave room's three pressure fields
    float        *m_dRoomHist = nullptr;         ///< device: per source its low pass's history
    float        *m_dRoomIn = nullptr;           ///< device: per source the block at the room's rate
    float        *m_dRoomEars = nullptr;         ///< device: the ears' block at the room's rate
    float        *m_dRoomOutHist = nullptr;      ///< device: the ears' last samples, for the way up
    Halo          m_halo;                        ///< the host's copy of the halo's settings
    WaveRoom      m_room;                        ///< the host's copy of the wave room's settings

    ChimeField m_chimes;              ///< the host's copy of the chime field's settings
    Dream      m_dream;               ///< the host's copy of the dream's settings
    int        m_phantomVoice = -1;   ///< the phantom layer's GPU voice, -1 none
    int        m_dreamVoice = -1;     ///< the dream's GPU voice, -1 none
    ModalHall  m_modal;               ///< the host's copy of the modal hall's settings
    /**
     * @brief A feature's kernels in the graph: switched off in the executable graph while the feature rests
     *        (cudaGraphNodeSetEnabled), so a block pays only for what sounds. A feature stays on three blocks after
     *        it rests, so its kernels clear their state and write their silence as they always did.
     */
    struct Gate
    {
        std::vector<cudaGraphNode_t> m_nodes;            ///< its kernels' nodes in the graph
        bool                         m_enabled = true;   ///< as the executable graph has them
        int                          m_rest = 0;         ///< blocks the feature has rested
    };
    std::array<Gate, 15> m_gates;   ///< life, wind, chimes, weather, swarm, physarum, halo, room, modal hall, bloom,
                                    ///< dream, phantom, reservoir, spectral echo, melt
    float m_gpuMs = 0.0f;           ///< GPU time of the last finished block
    bool  m_launched = false;       ///< a block is in flight

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
        for ( void *p : { static_cast<void *>( m_dUpload ),
                          static_cast<void *>( m_dOut ),
                          static_cast<void *>( m_dScratch ),
                          static_cast<void *>( m_dRun ),
                          static_cast<void *>( m_dAmp ),
                          static_cast<void *>( m_dGains ),
                          static_cast<void *>( m_dShapes ),
                          static_cast<void *>( m_dLand ),
                          static_cast<void *>( m_dModal ),
                          static_cast<void *>( m_dModalScratch ),
                          static_cast<void *>( m_dGainRing ),
                          static_cast<void *>( m_dAmpQ ),
                          static_cast<void *>( m_dEchoHist ),
                          static_cast<void *>( m_dEchoAcc ),
                          static_cast<void *>( m_dEchoRing ),
                          static_cast<void *>( m_dEchoState ),
                          static_cast<void *>( m_dSnap ),
                          static_cast<void *>( m_dLifeA ),
                          static_cast<void *>( m_dLifeB ),
                          static_cast<void *>( m_dLifeStats ),
                          static_cast<void *>( m_dSyncTheta ),
                          static_cast<void *>( m_dSyncOrder ),
                          static_cast<void *>( m_dFlockA ),
                          static_cast<void *>( m_dFlockB ),
                          static_cast<void *>( m_dListenHist ),
                          static_cast<void *>( m_dListenPrev ),
                          static_cast<void *>( m_dListenState ),
                          static_cast<void *>( m_dResW ),
                          static_cast<void *>( m_dResIn ),
                          static_cast<void *>( m_dResOut ),
                          static_cast<void *>( m_dResX ),
                          static_cast<void *>( m_dMeltHist ),
                          static_cast<void *>( m_dMeltAcc ),
                          static_cast<void *>( m_dMeltEnergy ),
                          static_cast<void *>( m_dMeltOmega ),
                          static_cast<void *>( m_dMeltPhase ),
                          static_cast<void *>( m_dMeltHeard ),
                          static_cast<void *>( m_dMeltState ),
                          static_cast<void *>( m_dSyncX ),
                          static_cast<void *>( m_dFluidA ),
                          static_cast<void *>( m_dFluidB ),
                          static_cast<void *>( m_dIsles ),
                          static_cast<void *>( m_dCascadeA ),
                          static_cast<void *>( m_dCascadeB ),
                          static_cast<void *>( m_dFilterTable ),
                          static_cast<void *>( m_dHallHist ),
                          static_cast<void *>( m_dHallMemory ),
                          static_cast<void *>( m_dDensity ),
                          static_cast<void *>( m_dTau ),
                          static_cast<void *>( m_dOrbitA ),
                          static_cast<void *>( m_dOrbitB ),
                          static_cast<void *>( m_dWindA ),
                          static_cast<void *>( m_dWindB ),
                          static_cast<void *>( m_dChimeState ),
                          static_cast<void *>( m_dChimeHit ),
                          static_cast<void *>( m_dChimeModes ),
                          static_cast<void *>( m_dChimeScratch ),
                          static_cast<void *>( m_dDreamAcc ),
                          static_cast<void *>( m_dDreamMem ),
                          static_cast<void *>( m_dDreamState ),
                          static_cast<void *>( m_dDreamSims ),
                          static_cast<void *>( m_dMeta ),
                          static_cast<void *>( m_dShowers ),
                          static_cast<void *>( m_dDrop ),
                          static_cast<void *>( m_dBirth ),
                          static_cast<void *>( m_dLeaf ),
                          static_cast<void *>( m_dDropAt ),
                          static_cast<void *>( m_dSurf ),
                          static_cast<void *>( m_dWeatherScratch ),
                          static_cast<void *>( m_dBloom ),
                          static_cast<void *>( m_dBloomBands ),
                          static_cast<void *>( m_dBloomScratch ),
                          static_cast<void *>( m_dSwarmA ),
                          static_cast<void *>( m_dSwarmB ),
                          static_cast<void *>( m_dSwarmView ),
                          static_cast<void *>( m_dSwarmForce ),
                          static_cast<void *>( m_dPhysTrail ),
                          static_cast<void *>( m_dPhysDeposit ),
                          static_cast<void *>( m_dHalo ),
                          static_cast<void *>( m_dRoom ),
                          static_cast<void *>( m_dRoomHist ),
                          static_cast<void *>( m_dRoomIn ),
                          static_cast<void *>( m_dRoomEars ),
                          static_cast<void *>( m_dRoomOutHist ) } )
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
        m_dWindA = m_dWindB = nullptr;
        m_dChimeState = nullptr;
        m_dChimeHit = nullptr;
        m_dChimeModes = nullptr;
        m_dChimeScratch = nullptr;
        m_dDreamAcc = m_dDreamMem = m_dDreamState = m_dDreamSims = nullptr;
        m_dMeta = nullptr;
        m_dShowers = m_dWeatherScratch = m_dBloomBands = m_dBloomScratch = m_dPhysTrail = nullptr;
        m_dDrop = m_dLeaf = m_dSurf = m_dSwarmA = m_dSwarmB = m_dSwarmView = m_dSwarmForce = nullptr;
        m_dBirth = nullptr;
        m_dDropAt = nullptr;
        m_dBloom = nullptr;
        m_dPhysDeposit = nullptr;
        m_dHalo = nullptr;
        m_dRoom = m_dRoomHist = m_dRoomIn = m_dRoomEars = m_dRoomOutHist = nullptr;
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
        if ( v.m_kind == VoiceKindE::HALO )
        {   // the halo (Phase 29): every partial's amplitude its site's in the lattice, times the level
            v.m_harmonics = v.m_members = v.m_harmonicPartials = v.m_subClouds = 0;
            v.m_scale = v.m_level;
            v.m_airScale = 0.0f;
            return;
        }
        if ( v.m_kind != VoiceKindE::NOTE )
        {   // the phantom layer and the dream (Phase 27): every partial a sine of one amplitude, RMS `level`
            const int per = v.m_kind == VoiceKindE::DREAM ? std::max( partials / DreamBands, 1 ) : partials;
            v.m_harmonics = 0;
            v.m_members = 0;
            v.m_harmonicPartials = 0;
            v.m_subClouds = 0;
            v.m_scale = static_cast<float>( v.m_level * std::sqrt( 2.0 / per ) );
            v.m_airScale = 0.0f;
            return;
        }
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
           ok( cudaMalloc( &m.m_dLifeStats, sizeof( float ) * StatsFloats ), "cudaMalloc life stats", error ) &&
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
           ok( cudaHostAlloc( &m.m_hLifeStats, sizeof( float ) * StatsFloats, cudaHostAllocDefault ),
               "cudaHostAlloc life stats", error ) &&
           ok( cudaHostAlloc( &m.m_hLife, sizeof( float2 ) * LifeCells, cudaHostAllocDefault ), "cudaHostAlloc life",
               error ) &&
           ok( cudaMalloc( &m.m_dShapes, sizeof( EnvShape ) * EnvShapes ), "cudaMalloc shapes", error ) &&
           ok( cudaMalloc( &m.m_dLand, sizeof( float ) * landFloats ), "cudaMalloc landscape", error ) &&
           ok( cudaMalloc( &m.m_dModal, sizeof( float2 ) * ModalModes ), "cudaMalloc modal", error ) &&
           ok( cudaMalloc( &m.m_dModalScratch, sizeof( float ) * 2 * n * static_cast<size_t>( ModalBlocks ) ),
               "cudaMalloc modal scratch", error ) &&
           ok( cudaMalloc( &m.m_dGainRing, sizeof( float ) * GainRing * static_cast<size_t>( config.m_voices ) ),
               "cudaMalloc gain ring", error ) &&
           ok( cudaMalloc( &m.m_dWindA, sizeof( float ) * 9 * WindCells ), "cudaMalloc wind", error ) &&
           ok( cudaMalloc( &m.m_dWindB, sizeof( float ) * 9 * WindCells ), "cudaMalloc wind", error ) &&
           ok( cudaMalloc( &m.m_dChimeState, sizeof( float4 ) * ChimeCount ), "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dChimeHit, sizeof( float2 ) * ChimeCount ), "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dChimeModes, sizeof( float2 ) * ChimeModesTotal ), "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dChimeScratch, sizeof( float ) * 4 * n * static_cast<size_t>( ChimeBlocks ) ),
               "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dDreamAcc, sizeof( float ) * HallBins ), "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dDreamMem, sizeof( float ) * DreamBands * static_cast<size_t>( DreamMemories ) ),
               "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dDreamState, sizeof( float ) * DreamBands ), "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dDreamSims, sizeof( float ) * 2 * DreamMemories ), "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dMeta, sizeof( int ) * 16 ), "cudaMalloc counters", error ) &&
           ok( cudaMalloc( &m.m_dShowers, sizeof( float ) * 2 * WindCells ), "cudaMalloc showers", error ) &&
           ok( cudaMalloc( &m.m_dDrop, sizeof( float4 ) * RainSlots ), "cudaMalloc rain", error ) &&
           ok( cudaMalloc( &m.m_dBirth, sizeof( int2 ) * RainSlots ), "cudaMalloc rain", error ) &&
           ok( cudaMalloc( &m.m_dDropAt, sizeof( float2 ) * RainSlots ), "cudaMalloc rain", error ) &&
           ok( cudaMalloc( &m.m_dLeaf, sizeof( float4 ) * LeafClusters ), "cudaMalloc leaves", error ) &&
           ok( cudaMalloc( &m.m_dSurf, sizeof( float4 ) * 2 * SurfSegments ), "cudaMalloc sea", error ) &&
           ok( cudaMalloc( &m.m_dWeatherScratch, sizeof( float ) * 4 * n * static_cast<size_t>( WeatherBlocks ) ),
               "cudaMalloc weather", error ) &&
           ok( cudaMalloc( &m.m_dBloom, sizeof( float2 ) * BloomModes ), "cudaMalloc bloom", error ) &&
           ok( cudaMalloc( &m.m_dBloomBands, sizeof( float ) * ( 2 * BloomBands + 2 ) ), "cudaMalloc bloom", error ) &&
           ok( cudaMalloc( &m.m_dBloomScratch, sizeof( float ) * 2 * n * static_cast<size_t>( BloomBlocks ) ),
               "cudaMalloc bloom", error ) &&
           ok( cudaMalloc( &m.m_dSwarmA, sizeof( float4 ) * SwarmN ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dSwarmB, sizeof( float4 ) * SwarmN ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dSwarmView, sizeof( float4 ) * SwarmN ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dSwarmForce, sizeof( float4 ) * SwarmN * SwarmChunks ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dPhysTrail, sizeof( float ) * 2 * PhysCells ), "cudaMalloc physarum", error ) &&
           ok( cudaMalloc( &m.m_dPhysDeposit, sizeof( unsigned int ) * PhysCells ), "cudaMalloc physarum", error ) &&
           ok( cudaMalloc( &m.m_dHalo, sizeof( float2 ) * HaloCarriers * HaloComponents ), "cudaMalloc halo", error ) &&
           ok( cudaMalloc( &m.m_dRoom, sizeof( float ) * 3 * static_cast<size_t>( RoomCells ) ), "cudaMalloc room",
               error ) &&
           ok( cudaMalloc( &m.m_dRoomHist, sizeof( float ) * RoomSources * RoomHist ), "cudaMalloc room", error ) &&
           ok( cudaMalloc( &m.m_dRoomIn, sizeof( float ) * RoomSources * ( 4096 / RoomDecim ) ), "cudaMalloc room",
               error ) &&
           ok( cudaMalloc( &m.m_dRoomEars, sizeof( float ) * 2 * ( 4096 / RoomDecim ) ), "cudaMalloc room", error ) &&
           ok( cudaMalloc( &m.m_dRoomOutHist, sizeof( float ) * 2 * ( RoomFirTaps / RoomDecim ) ), "cudaMalloc room",
               error );
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
        ok( cudaMemsetAsync( m.m_dLifeStats, 0, sizeof( float ) * StatsFloats, m.m_stream ), "cudaMemsetAsync",
            error ) &&
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
        ok( cudaMemsetAsync( m.m_dWindA, 0, sizeof( float ) * 9 * WindCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dWindB, 0, sizeof( float ) * 9 * WindCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dChimeState, 0, sizeof( float4 ) * ChimeCount, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dChimeHit, 0, sizeof( float2 ) * ChimeCount, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dChimeModes, 0, sizeof( float2 ) * ChimeModesTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDreamAcc, 0, sizeof( float ) * HallBins, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dDreamMem, 0, sizeof( float ) * DreamBands * static_cast<size_t>( DreamMemories ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dDreamState, 0, sizeof( float ) * DreamBands, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDreamSims, 0, sizeof( float ) * 2 * DreamMemories, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dMeta, 0, sizeof( int ) * 16, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dShowers, 0, sizeof( float ) * 2 * WindCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDrop, 0, sizeof( float4 ) * RainSlots, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dBirth, 0xFF, sizeof( int2 ) * RainSlots, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dDropAt, 0, sizeof( float2 ) * RainSlots, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLeaf, 0, sizeof( float4 ) * LeafClusters, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSurf, 0, sizeof( float4 ) * 2 * SurfSegments, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dWeatherScratch, 0, sizeof( float ) * 4 * n * static_cast<size_t>( WeatherBlocks ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dBloom, 0, sizeof( float2 ) * BloomModes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dBloomBands, 0, sizeof( float ) * ( 2 * BloomBands + 2 ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSwarmA, 0, sizeof( float4 ) * SwarmN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSwarmB, 0, sizeof( float4 ) * SwarmN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSwarmView, 0, sizeof( float4 ) * SwarmN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dPhysTrail, 0, sizeof( float ) * 2 * PhysCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dPhysDeposit, 0, sizeof( unsigned int ) * PhysCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dHalo, 0, sizeof( float2 ) * HaloCarriers * HaloComponents, m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoom, 0, sizeof( float ) * 3 * static_cast<size_t>( RoomCells ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomHist, 0, sizeof( float ) * RoomSources * RoomHist, m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomIn, 0, sizeof( float ) * RoomSources * ( 4096 / RoomDecim ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomEars, 0, sizeof( float ) * 2 * ( 4096 / RoomDecim ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomOutHist, 0, sizeof( float ) * 2 * ( RoomFirTaps / RoomDecim ), m.m_stream ),
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
        {   // the halo's disc (Phase 29): its sites row by row, their neighbours
            signed char xy[HaloSites][2];
            short       nbr[HaloSites][4];
            int         at[2 * HaloRadius + 1][2 * HaloRadius + 1];
            int         sites = 0;
            for ( int y = -HaloRadius; y <= HaloRadius; ++y )
                for ( int x = -HaloRadius; x <= HaloRadius; ++x )
                {
                    at[y + HaloRadius][x + HaloRadius] = -1;
                    if ( x * x + y * y <= HaloRadius * HaloRadius && sites < HaloSites )
                    {
                        xy[sites][0] = static_cast<signed char>( x );
                        xy[sites][1] = static_cast<signed char>( y );
                        at[y + HaloRadius][x + HaloRadius] = sites++;
                    }
                }
            auto siteAt = [&]( int x, int y )
            {
                if ( x < -HaloRadius || x > HaloRadius || y < -HaloRadius || y > HaloRadius ) return -1;
                return at[y + HaloRadius][x + HaloRadius];
            };
            for ( int s = 0; s < HaloSites; ++s )
            {
                const int x = xy[s][0], y = xy[s][1];
                nbr[s][0] = static_cast<short>( siteAt( x + 1, y ) );
                nbr[s][1] = static_cast<short>( siteAt( x - 1, y ) );
                nbr[s][2] = static_cast<short>( siteAt( x, y + 1 ) );
                nbr[s][3] = static_cast<short>( siteAt( x, y - 1 ) );
            }
            good = good && sites == HaloSites &&
                   ok( cudaMemcpyToSymbol( cHaloXY, xy, sizeof( xy ) ), "cudaMemcpyToSymbol halo", error ) &&
                   ok( cudaMemcpyToSymbol( cHaloNbr, nbr, sizeof( nbr ) ), "cudaMemcpyToSymbol halo", error );
            // The wave room's low pass: a windowed sinc (Blackman) at 2.2 kHz, unity at DC.
            float        fir[RoomFirTaps];
            double       firSum = 0.0;
            const double fc = 2200.0 / config.m_sampleRate;
            for ( int k = 0; k < RoomFirTaps; ++k )
            {
                const double x = k - 0.5 * ( RoomFirTaps - 1 );
                const double s =
                    x == 0.0 ? 2.0 * fc : std::sin( 6.283185307179586 * fc * x ) / ( 3.141592653589793 * x );
                const double w = 0.42 - 0.5 * std::cos( 6.283185307179586 * k / ( RoomFirTaps - 1 ) ) +
                                 0.08 * std::cos( 12.566370614359172 * k / ( RoomFirTaps - 1 ) );
                fir[k] = static_cast<float>( s * w );
                firSum += s * w;
            }
            for ( float &h : fir ) h = static_cast<float>( h / firSum );
            good = good && ok( cudaMemcpyToSymbol( cRoomFir, fir, sizeof( fir ) ), "cudaMemcpyToSymbol room", error );
        }
        m.m_lifeCells.assign( static_cast<size_t>( LifeCells ), 0.0f );
        reservoirInitKernel<<<static_cast<unsigned>( ( static_cast<size_t>( ReservoirN ) * ReservoirN + 255 ) / 256 ),
                              256, 0, m.m_stream>>>( m.m_dResW, m.m_dResIn, m.m_dResOut, m.m_dResX );
        good = ok( cudaStreamSynchronize( m.m_stream ), "reservoir init", error );
        std::memset( m.m_hLifeStats, 0, sizeof( float ) * StatsFloats );
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
            // The frames cleared first: a feature's kernels switched off (the gates) write nothing, and its
            // channels must read silence.
            cudaMemsetAsync( m.m_dOut, 0, sizeof( float ) * CloudOutChannels * n, m.m_stream );
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
            windStepKernel<<<( WindCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, 0, m.m_dWindA, m.m_dWindB );
            windStepKernel<<<( WindCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, 1, m.m_dWindA, m.m_dWindB );
            chimeStepKernel<<<( ChimeCount + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dChimeState,
                                                                                 m.m_dChimeHit, m.m_dMeta );
            rainFieldKernel<<<( WindCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dShowers );
            swarmForceKernel<<<dim3( SwarmN / SwarmThreads, SwarmChunks ), SwarmThreads, 0, m.m_stream>>>(
                hdr, m.m_dSwarmA, m.m_dSwarmB, m.m_dSwarmForce );
            swarmStepKernel<<<SwarmN / SwarmThreads, SwarmThreads, 0, m.m_stream>>>( hdr, m.m_dSwarmA, m.m_dSwarmB,
                                                                                     m.m_dSwarmForce );
            swarmViewKernel<<<1, SwarmN, 0, m.m_stream>>>( hdr, m.m_dSwarmA, m.m_dSwarmB, m.m_dSwarmView,
                                                           m.m_dLifeStats );
            physTrailKernel<<<( PhysCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, m.m_dPhysTrail, m.m_dPhysDeposit );
            haloKernel<<<HaloCarriers, HaloThreads, 0, m.m_stream>>>( hdr, m.m_dHalo, m.m_dLifeStats );

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
                                          m.m_dTau, m.m_dOrbitA, m.m_dOrbitB, m.m_dDreamState, m.m_dSwarmView,
                                          m.m_dPhysTrail, m.m_dPhysDeposit, m.m_dHalo );
            hallKernel<<<1, HallBins, 0, m.m_stream>>>( hdr, m.m_dHallHist, m.m_dHallMemory, m.m_dDensity,
                                                        m.m_dDreamAcc );
            dreamSimKernel<<<DreamMemories / DreamThreads, DreamThreads, 0, m.m_stream>>>(
                hdr, m.m_dDreamAcc, m.m_dDreamMem, m.m_dDreamState, m.m_dDreamSims, m.m_dMeta );
            dreamRecallKernel<<<1, DreamBands, 0, m.m_stream>>>( hdr, m.m_dDreamAcc, m.m_dDreamMem, m.m_dDreamState,
                                                                 m.m_dDreamSims, m.m_dMeta );
            renderKernel<<<static_cast<unsigned>( m.m_chunks ), Threads, 0, m.m_stream>>>(
                hdr, voices, m.m_pcap, m.m_dRun, m.m_dGains, m.m_dScratch, n );
            reduceKernel<<<static_cast<unsigned>( CloudChannels * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ),
                           0, m.m_stream>>>( hdr, m.m_dScratch, m.m_chunks, m.m_pcap / Chunk, n, m.m_dOut );
            phantomReduceKernel<<<static_cast<unsigned>( ( n + 127 ) / 128 ), 128, 0, m.m_stream>>>(
                hdr, m.m_dScratch, m.m_pcap / Chunk, n, m.m_dOut );
            // The wave room (Phase 29): its sources from the voices' chunks, its steps, its ears.
            roomSourceKernel<<<RoomSources, 128, 0, m.m_stream>>>( hdr, m.m_dScratch, m.m_pcap / Chunk, n,
                                                                   m.m_dRoomHist, m.m_dRoomIn );
            for ( int step = 0; step < n / RoomDecim; ++step )
                roomStepKernel<<<( RoomCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, step, m.m_dRoom, m.m_dRoomIn,
                                                                                   m.m_dRoomEars );
            roomOutKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dRoomEars, m.m_dRoomOutHist, m.m_dOut );

            chimeKernel<<<static_cast<unsigned>( ChimeBlocks ), ChimeThreads, 0, m.m_stream>>>(
                hdr, m.m_dChimeHit, m.m_dChimeModes, m.m_dChimeScratch, n );
            chimeReduceKernel<<<static_cast<unsigned>( 4 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dChimeScratch, n, m.m_dOut );
            rainSpawnKernel<<<1, 32, 0, m.m_stream>>>( hdr, m.m_dShowers, m.m_dBirth, m.m_dDropAt, m.m_dMeta );
            rainKernel<<<RainBlocks, RainThreads, 0, m.m_stream>>>( hdr, m.m_dDropAt, m.m_dDrop, m.m_dBirth,
                                                                    m.m_dWeatherScratch, n, m.m_dMeta );
            leafKernel<<<LeafBlocks, LeafThreads, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dLeaf, m.m_dWeatherScratch, n,
                                                                    m.m_dMeta );
            surfKernel<<<1, SurfSegments, 0, m.m_stream>>>( hdr, m.m_dSurf, m.m_dWeatherScratch, n, m.m_dMeta );
            weatherReduceKernel<<<static_cast<unsigned>( 4 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                  m.m_stream>>>( hdr, m.m_dWeatherScratch, n, m.m_dOut );
            modalKernel<<<static_cast<unsigned>( ModalBlocks ), ModalThreads, 0, m.m_stream>>>(
                hdr, m.m_dOut, m.m_dModal, m.m_dModalScratch, n, m.m_dLand, m.m_dHallMemory );
            modalReduceKernel<<<static_cast<unsigned>( 2 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dModalScratch, n, m.m_dOut );
            bloomKernel<<<BloomBlocks, BloomThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dBloom, m.m_dBloomBands,
                                                                       m.m_dBloomScratch, n );
            bloomBandKernel<<<BloomBands + 1, 256, 0, m.m_stream>>>( hdr, m.m_dBloom, m.m_dBloomBands, m.m_dLifeStats );
            bloomReduceKernel<<<static_cast<unsigned>( 2 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dBloomScratch, n, m.m_dOut );
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
            phase27StatsKernel<<<1, 32, 0, m.m_stream>>>( m.m_dMeta, m.m_dLifeStats );
            phase28StatsKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dMeta, m.m_dLifeStats );
            physStatsKernel<<<1, 1024, 0, m.m_stream>>>( hdr, m.m_dPhysTrail, m.m_dLifeStats );
            cudaMemcpyAsync( m.m_hLifeStats, m.m_dLifeStats, sizeof( float ) * StatsFloats, cudaMemcpyDeviceToHost,
                             m.m_stream );
            cudaMemcpyAsync( m.m_hOut, m.m_dOut, sizeof( float ) * CloudOutChannels * n, cudaMemcpyDeviceToHost,
                             m.m_stream );
            good = ok( cudaStreamEndCapture( m.m_stream, &m.m_graph ), "cudaStreamEndCapture", error ) &&
                   ok( cudaGraphInstantiate( &m.m_exec, m.m_graph, 0 ), "cudaGraphInstantiate", error );
            // The features' kernels, found in the graph by their functions, for the gates.
            if ( good )
            {
                const std::pair<const void *, int> owners[] = {
                    { reinterpret_cast<const void *>( &lifeSeedKernel ), 0 },
                    { reinterpret_cast<const void *>( &lifeStepKernel ), 0 },
                    { reinterpret_cast<const void *>( &fluidSeedKernel ), 0 },
                    { reinterpret_cast<const void *>( &fluidStepKernel ), 0 },
                    { reinterpret_cast<const void *>( &islandsKernel ), 0 },
                    { reinterpret_cast<const void *>( &lifeStatsKernel ), 0 },
                    { reinterpret_cast<const void *>( &windStepKernel ), 1 },
                    { reinterpret_cast<const void *>( &phase28StatsKernel ), 1 },
                    { reinterpret_cast<const void *>( &chimeStepKernel ), 2 },
                    { reinterpret_cast<const void *>( &chimeKernel ), 2 },
                    { reinterpret_cast<const void *>( &chimeReduceKernel ), 2 },
                    { reinterpret_cast<const void *>( &rainFieldKernel ), 3 },
                    { reinterpret_cast<const void *>( &rainSpawnKernel ), 3 },
                    { reinterpret_cast<const void *>( &rainKernel ), 3 },
                    { reinterpret_cast<const void *>( &leafKernel ), 3 },
                    { reinterpret_cast<const void *>( &surfKernel ), 3 },
                    { reinterpret_cast<const void *>( &weatherReduceKernel ), 3 },
                    { reinterpret_cast<const void *>( &swarmForceKernel ), 4 },
                    { reinterpret_cast<const void *>( &swarmStepKernel ), 4 },
                    { reinterpret_cast<const void *>( &swarmViewKernel ), 4 },
                    { reinterpret_cast<const void *>( &physTrailKernel ), 5 },
                    { reinterpret_cast<const void *>( &physStatsKernel ), 5 },
                    { reinterpret_cast<const void *>( &haloKernel ), 6 },
                    { reinterpret_cast<const void *>( &roomSourceKernel ), 7 },
                    { reinterpret_cast<const void *>( &roomStepKernel ), 7 },
                    { reinterpret_cast<const void *>( &roomOutKernel ), 7 },
                    { reinterpret_cast<const void *>( &modalKernel ), 8 },
                    { reinterpret_cast<const void *>( &modalReduceKernel ), 8 },
                    { reinterpret_cast<const void *>( &bloomKernel ), 9 },
                    { reinterpret_cast<const void *>( &bloomBandKernel ), 9 },
                    { reinterpret_cast<const void *>( &bloomReduceKernel ), 9 },
                    { reinterpret_cast<const void *>( &dreamSimKernel ), 10 },
                    { reinterpret_cast<const void *>( &dreamRecallKernel ), 10 },
                    { reinterpret_cast<const void *>( &phantomReduceKernel ), 11 },
                    { reinterpret_cast<const void *>( &reservoirStepKernel ), 12 },
                    { reinterpret_cast<const void *>( &reservoirReadKernel ), 12 },
                    { reinterpret_cast<const void *>( &spectralEchoKernel ), 13 },
                    { reinterpret_cast<const void *>( &meltKernel ), 14 },
                };
                size_t nodeCount = 0;
                good = ok( cudaGraphGetNodes( m.m_graph, nullptr, &nodeCount ), "cudaGraphGetNodes", error );
                std::vector<cudaGraphNode_t> nodes( nodeCount );
                good =
                    good && ok( cudaGraphGetNodes( m.m_graph, nodes.data(), &nodeCount ), "cudaGraphGetNodes", error );
                for ( size_t k = 0; good && k < nodeCount; ++k )
                {
                    cudaGraphNodeType type{};
                    if ( cudaGraphNodeGetType( nodes[k], &type ) != cudaSuccess || type != cudaGraphNodeTypeKernel )
                        continue;
                    cudaKernelNodeParams p{};
                    if ( cudaGraphKernelNodeGetParams( nodes[k], &p ) != cudaSuccess ) continue;
                    for ( const auto &[fn, gate] : owners )
                        if ( p.func == fn ) m.m_gates[static_cast<size_t>( gate )].m_nodes.push_back( nodes[k] );
                }
            }
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

ChimeField &CloudEngine::chimes()
{
    return m_impl->m_chimes;
}

Dream &CloudEngine::dream()
{
    return m_impl->m_dream;
}

int CloudEngine::dreamMemories() const
{
    return m_impl ? static_cast<int>( m_impl->m_hLifeStats[12] ) : 0;
}

int CloudEngine::chimeStrikes() const
{
    return m_impl ? static_cast<int>( m_impl->m_hLifeStats[13] ) : 0;
}

Weather &CloudEngine::weather()
{
    return m_impl->m_weather;
}

Bloom &CloudEngine::bloom()
{
    return m_impl->m_bloom;
}

SwarmSpace &CloudEngine::swarm()
{
    return m_impl->m_swarm;
}

Physarum &CloudEngine::physarum()
{
    return m_impl->m_phys;
}

void CloudEngine::weatherStats( float *out ) const
{
    for ( int k = 0; k < 8; ++k ) out[k] = m_impl ? m_impl->m_hLifeStats[16 + k] : 0.0f;
}

Halo &CloudEngine::halo()
{
    return m_impl->m_halo;
}

WaveRoom &CloudEngine::room()
{
    return m_impl->m_room;
}

void CloudEngine::haloStats( float *energy, float *angle, float *rim ) const
{
    for ( int c = 0; c < HaloCarriers; ++c )
    {
        energy[c] = m_impl ? m_impl->m_hLifeStats[32 + c] : 0.0f;
        angle[c] = m_impl ? m_impl->m_hLifeStats[40 + c] : 0.0f;
        rim[c] = m_impl ? m_impl->m_hLifeStats[48 + c] : 0.0f;
    }
}

void CloudEngine::physarumStats( float &mean, float &spread ) const
{
    mean = m_impl ? m_impl->m_hLifeStats[24] : 0.0f;
    spread = m_impl ? m_impl->m_hLifeStats[25] : 0.0f;
}

void CloudEngine::setLayers( int phantomVoice, int dreamVoice )
{
    m_impl->m_phantomVoice = phantomVoice;
    m_impl->m_dreamVoice = dreamVoice;
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
    hdr->m_chimes = m.m_chimes;
    hdr->m_chimes.m_pitchCount = std::clamp( m.m_chimes.m_pitchCount, 0, ChimePitches );
    hdr->m_chimes.m_kind = std::clamp( m.m_chimes.m_kind, 0, static_cast<int>( ChimeKindE::COUNT ) - 1 );
    m.m_chimes.m_reset = 0;   // asked for once
    hdr->m_dream = m.m_dream;
    m.m_dream.m_store = 0;   // asked for once
    m.m_dream.m_clear = 0;
    hdr->m_weather = m.m_weather;
    hdr->m_weather.m_pitchCount = std::clamp( m.m_weather.m_pitchCount, 0, RainPitches );
    hdr->m_weather.m_wind = ( m.m_weather.m_wind || m.m_chimes.m_on ) ? 1 : 0;
    m.m_weather.m_reset = 0;   // asked for once
    m.m_weather.m_rainFresh = 0;
    m.m_weather.m_windFresh = 0;
    hdr->m_bloom = m.m_bloom;
    hdr->m_swarm = m.m_swarm;
    m.m_swarm.m_reset = 0;   // asked for once
    hdr->m_phys = m.m_phys;
    hdr->m_phys.m_on = m.m_field.m_mode == FieldModeE::PHYSARUM ? m.m_phys.m_on : 0;
    hdr->m_phys.m_count = std::clamp( m.m_phys.m_count, 0, PhysFood );
    m.m_phys.m_fresh = 0;   // asked for once
    hdr->m_halo = m.m_halo;
    m.m_halo.m_clear = 0;   // asked for once
    hdr->m_room = m.m_room;
    hdr->m_room.m_count = std::clamp( m.m_room.m_count, 0, RoomSources );
    hdr->m_room.m_nx = std::clamp( m.m_room.m_nx, 3, RoomNX );
    hdr->m_room.m_ny = std::clamp( m.m_room.m_ny, 3, RoomNY );
    hdr->m_room.m_nz = std::clamp( m.m_room.m_nz, 3, RoomNZ );
    m.m_room.m_clear = 0;   // asked for once
    hdr->m_phantomVoice = m.m_phantomVoice >= 0 && m.m_phantomVoice < m.m_cfg.m_voices ? m.m_phantomVoice : -1;
    hdr->m_dreamVoice = m.m_dreamVoice >= 0 && m.m_dreamVoice < m.m_cfg.m_voices ? m.m_dreamVoice : -1;
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
    // The gates: a resting feature's kernels switched off in the graph (on again at once when it wakes).
    {
        const int  pv = hdr->m_phantomVoice;
        const bool want[15] = { hdr->m_life.m_mode != LifeModeE::OFF,
                                hdr->m_weather.m_wind != 0,
                                hdr->m_chimes.m_on != 0,
                                hdr->m_weather.m_rainOn || hdr->m_weather.m_leavesOn || hdr->m_weather.m_surfOn,
                                hdr->m_swarm.m_on != 0,
                                hdr->m_phys.m_on != 0,
                                hdr->m_halo.m_on != 0,
                                hdr->m_room.m_on != 0,
                                hdr->m_modal.m_on != 0,
                                hdr->m_bloom.m_on != 0,
                                hdr->m_dream.m_on != 0,
                                pv >= 0 && m.m_voices[static_cast<size_t>( pv )].m_active != 0,
                                hdr->m_reservoir.m_on != 0,
                                hdr->m_echo.m_on != 0,
                                hdr->m_melt.m_on != 0 };
        for ( size_t g = 0; g < m.m_gates.size(); ++g )
        {
            Impl::Gate &gate = m.m_gates[g];
            gate.m_rest = want[g] ? 0 : std::min( gate.m_rest + 1, 1000 );
            const bool on = want[g] || gate.m_rest < 3;
            if ( on == gate.m_enabled ) continue;
            for ( cudaGraphNode_t node : gate.m_nodes ) cudaGraphNodeSetEnabled( m.m_exec, node, on ? 1U : 0U );
            gate.m_enabled = on;
        }
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
