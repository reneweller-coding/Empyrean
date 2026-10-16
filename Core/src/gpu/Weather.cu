/**
 * @file Weather.cu
 * @brief The weather (Phases 27 and 28): the wind on its lattice Boltzmann grid, the chimes it swings, the showers it
 *        carries and the rain's bubbles, the leaves, the sea; their reduction into the frames and their counters.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

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
 * @brief Whether a cell of the wind is inside one of the headset's hands (Phase 30): a post that moves, its distance on
 *        the torus.
 * @param[in] W  the weather (the hands' cells and radii)
 * @param[in] x  across, 0 .. WindW - 1
 * @param[in] y  from near to far, 0 .. WindH - 1
 * @return true inside a hand
 */
__device__ __forceinline__ bool windHand( const Weather &W, int x, int y )
{
    for ( int h = 0; h < 2; ++h )
    {
        const int r = W.m_handR[h];
        if ( r <= 0 ) continue;
        int dx = abs( x - W.m_handX[h] ), dy = abs( y - W.m_handY[h] );
        dx = min( dx, WindW - dx );
        dy = min( dy, WindH - dy );
        if ( dx * dx + dy * dy <= r * r ) return true;
    }
    return false;
}

}   // namespace

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
    const int      x = idx % WindW, y = idx / WindW;
    const float   *src = node == 0 ? a : b;
    float         *dst = node == 0 ? b : a;
    const Weather &W = hdr->m_weather;
    if ( ( W.m_windFresh && node == 0 ) || windPost( x, y ) || windHand( W, x, y ) )
    {   // at rest (a post's cells stay so, and a hand's)
        for ( int k = 0; k < 9; ++k ) dst[k * WindCells + idx] = cWindW[k];
        return;
    }
    float f[9], rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 9; ++k )
    {
        const int xs = ( x - cWindX[k] ) & ( WindW - 1 ), ys = ( y - cWindY[k] ) & ( WindH - 1 );
        f[k] = windPost( xs, ys ) || windHand( W, xs, ys ) ? src[cWindOpp[k] * WindCells + idx]
                                                           : src[k * WindCells + ys * WindW + xs];
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

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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

}   // namespace additiv::gpu
