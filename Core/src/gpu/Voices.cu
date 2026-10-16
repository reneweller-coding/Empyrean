/**
 * @file Voices.cu
 * @brief The partials: per block their update, their render and the reduction, and what the update reads -- the
 *        sync's mean fields, the voices' gain ring, the transported filters, the energy cascade.
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
#include "Kernels.cuh"

#include "additiv/Spectrum.h"

namespace additiv::gpu
{

namespace
{

/** @brief The halo's sites, as the update kernel places their partials (Rooms.cu keeps the same; filled at open()). */
__constant__ signed char cHaloXY[HaloSites][2];

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

}   // namespace

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

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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
 * @brief The halo's sites into this file's constant memory (the update kernel places their partials by them).
 * @param[in] xy  per site across and from front to back, HaloSites
 * @return the CUDA call's result
 */
cudaError_t uploadVoiceTables( const signed char ( *xy )[2] )
{
    return cudaMemcpyToSymbol( cHaloXY, xy, sizeof( cHaloXY ) );
}

}   // namespace additiv::gpu
