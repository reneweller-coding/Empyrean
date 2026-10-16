/**
 * @file Reservoir.cu
 * @brief The reservoir: an echo state network of ReservoirN neurons the music drives, its readouts for the conductor.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

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

}   // namespace additiv::gpu
