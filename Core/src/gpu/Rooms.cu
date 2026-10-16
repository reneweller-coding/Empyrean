/**
 * @file Rooms.cu
 * @brief The rooms (Phase 29): the topological halo (Qi, Wu and Zhang's Chern insulator on a disc, one lattice a
 *        carrier) and the wave room (a 3D finite-difference room with Bilbao's walls, its sources and its ears).
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

/** @brief The halo's sites: across and from front to back, -HaloRadius .. HaloRadius (filled at open()). */
__constant__ signed char cHaloXY[HaloSites][2];

/** @brief Each site's neighbours: +x, -x, +y, -y (-1: the rim). */
__constant__ short cHaloNbr[HaloSites][4];

/** @brief The wave room's low pass (a windowed sinc at 2.2 kHz, unity gain at DC). */
__constant__ float cRoomFir[RoomFirTaps];

}   // namespace

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

namespace
{

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

}   // namespace

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
 * @brief The halo's disc and the wave room's low pass into this file's constant memory.
 * @param[in] xy   per site across and from front to back, HaloSites
 * @param[in] nbr  per site its neighbours (+x, -x, +y, -y; -1 the rim)
 * @param[in] fir  the low pass, RoomFirTaps taps
 * @return the first CUDA call's result that failed, else cudaSuccess
 */
cudaError_t uploadRoomTables( const signed char ( *xy )[2], const short ( *nbr )[4], const float *fir )
{
    cudaError_t e = cudaMemcpyToSymbol( cHaloXY, xy, sizeof( cHaloXY ) );
    if ( e == cudaSuccess ) e = cudaMemcpyToSymbol( cHaloNbr, nbr, sizeof( cHaloNbr ) );
    if ( e == cudaSuccess ) e = cudaMemcpyToSymbol( cRoomFir, fir, sizeof( cRoomFir ) );
    return e;
}

}   // namespace additiv::gpu
