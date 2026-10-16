/**
 * @file Dream.cu
 * @brief The dream (Phase 27): a Hopfield memory of the spectra the hall heard, every memory's likeness to the
 *        wandering query, and the recall the dream's voice sounds.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

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

}   // namespace

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

}   // namespace additiv::gpu
