/**
 * @file Space.cu
 * @brief Space (Phase 33): a star's oscillations heard (asteroseismology), added into the frames' channels 24 .. 27
 *        after the night.
 *
 * A star rings in thousands of modes at once, each a standing sound wave in its interior: the p modes, spaced almost
 * evenly by the large separation (their frequencies nu = dnu (n + l/2 + eps) - small separations), split by the
 * rotation into 2l + 1 azimuthal orders; in red giants the dipole modes mixed with the core's gravity modes into
 * clusters of narrow lines; in white dwarfs the gravity modes, evenly spaced in period. None is driven by a clock: the
 * convection's noise keeps each ringing, so its amplitude wanders as a damped oscillator's under noise does -- a
 * Lorentzian line whose width is its damping. The table of modes is the host's (Engine: the star's type, age, spin
 * and the root it is tuned to); here each mode's complex amplitude walks (an Ornstein-Uhlenbeck process in the plane)
 * and sounds, mixed through shared memory in a fixed order: a block renders bit for bit the same every time.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

/**
 * @brief Star (Phase 33): every mode's complex amplitude for this block, a thread a mode -- decaying by its line width
 *        and kicked by a Gaussian of the variance that keeps its mean power its weight squared (the amplitude at the
 *        block's start kept, for the glide across it).
 * @param[in]     hdr    the block header
 * @param[in]     modes  the table: frequency, line width, weight, place
 * @param[in,out] state  per mode two entries: the amplitude at the block's start and at its end; the phase
 */
__global__ void starStepKernel( const BlockHeader *hdr, const float4 *modes, float4 *state )
{
    const int        i = static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x );
    const StarField &S = hdr->m_star;
    if ( i >= StarModes ) return;
    if ( !S.m_on || S.m_reset || i >= S.m_count )
    {
        state[2 * i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        state[2 * i + 1] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        return;
    }
    const float4 m = modes[i];
    float4       s = state[2 * i];
    const float  dt = static_cast<float>( hdr->m_n ) / hdr->m_sampleRate;
    const float  decay = __expf( -3.1415927f * m.y * dt );
    const float  sigma = m.z * sqrtf( fmaxf( 0.5f * ( 1.0f - decay * decay ), 0.0f ) );
    // a Gaussian pair (Box-Muller) from the block and the mode
    const float u1 = fmaxf( hash01( hdr->m_blockIndex, static_cast<std::uint32_t>( i ), 80 ), 1.0e-7f );
    const float u2 = hash01( hdr->m_blockIndex, static_cast<std::uint32_t>( i ), 81 );
    const float r = sqrtf( -2.0f * __logf( u1 ) );
    float       gs, gc;
    __sincosf( 6.2831853f * u2, &gs, &gc );
    s.x = s.z;   // the end of the last block is this one's start
    s.y = s.w;
    s.z = decay * s.z + sigma * r * gc;
    s.w = decay * s.w + sigma * r * gs;
    state[2 * i] = s;
}

/**
 * @brief Star (Phase 33): the modes sound, a thread a mode -- Re(a(t) e^(i phase)), the amplitude gliding across the
 *        block from its start to its end -- placed across by the table, near and far by the star's share; mixed
 *        through shared memory in a fixed order into its CUDA block's four channels.
 * @param[in]     hdr      the block header
 * @param[in]     modes    the table
 * @param[in,out] state    the amplitudes (read) and the phases (carried on)
 * @param[out]    scratch  per CUDA block four channels of the block
 * @param[in]     maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__( StarThreads )
    starSoundKernel( const BlockHeader *hdr, const float4 *modes, float4 *state, float *scratch, int maxN )
{
    __shared__ float sV[StarThreads][MixTile + 1];
    __shared__ float sG[StarThreads][4];
    const StarField &S = hdr->m_star;
    if ( !S.m_on ) return;
    const int    k = static_cast<int>( threadIdx.x );
    const int    i = static_cast<int>( blockIdx.x ) * StarThreads + k;
    const int    n = hdr->m_n;
    const float  fs = hdr->m_sampleRate;
    const bool   live = i < S.m_count;
    const float4 m = live ? modes[i] : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    const float4 a = state[2 * i];
    float        ph = state[2 * i + 1].x;
    {
        float gl, gr, gn, gf;
        __sincosf( 0.78539816f * ( 1.0f + fminf( fmaxf( m.w, -1.0f ), 1.0f ) ), &gr, &gl );
        __sincosf( 1.57079633f * S.m_far, &gf, &gn );
        const float g = live && m.x > 10.0f && m.x < 0.45f * fs ? S.m_level : 0.0f;
        sG[k][0] = g * gl * gn;
        sG[k][1] = g * gr * gn;
        sG[k][2] = g * gl * gf;
        sG[k][3] = g * gr * gf;
    }
    const float  w = 6.2831853f * m.x / fs;
    float *const out = scratch + static_cast<size_t>( blockIdx.x ) * 4 * maxN;
    for ( int t0 = 0; t0 < n; t0 += MixTile )
    {
        for ( int q = 0; q < MixTile; ++q )
        {
            float v = 0.0f;
            if ( t0 + q < n )
            {
                const float u = static_cast<float>( t0 + q ) / static_cast<float>( n );
                const float ar = a.x + ( a.z - a.x ) * u, ai = a.y + ( a.w - a.y ) * u;
                float       x = ph + w * static_cast<float>( t0 + q ), s, c;
                x -= 6.2831853f * floorf( x * 0.15915494f );   // the fast sine wants its argument small
                __sincosf( x, &s, &c );
                v = ar * c - ai * s;
            }
            sV[k][q] = v;
        }
        __syncthreads();
        for ( int o = k; o < 4 * MixTile; o += StarThreads )
        {
            const int ch = o / MixTile, q = o - ch * MixTile;
            if ( t0 + q >= n ) continue;
            float acc = 0.0f;
            for ( int c = 0; c < StarThreads; ++c ) acc += sG[c][ch] * sV[c][q];
            out[static_cast<size_t>( ch ) * maxN + t0 + q] = acc;
        }
        __syncthreads();
    }
    ph += w * static_cast<float>( n );
    ph -= 6.2831853f * floorf( ph * 0.15915494f );
    state[2 * i + 1] = make_float4( live ? ph : 0.0f, 0.0f, 0.0f, 0.0f );
}

/**
 * @brief Space's mix (Phase 33): the star's CUDA blocks' four channels added, sample by sample in their order, onto the
 *        frames' channels 24 .. 27 (after the night's).
 * @param[in]     hdr      the block header
 * @param[in]     scratch  per CUDA block of the star's sound four channels
 * @param[in]     maxN     the engine's block size (the scratch stride)
 * @param[in,out] out      the frames
 */
__global__ void spaceMixKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out )
{
    const int i = static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x );
    if ( i >= hdr->m_n || !hdr->m_star.m_on ) return;
    for ( int c = 0; c < 4; ++c )
    {
        float v = 0.0f;
        for ( int b = 0; b < StarBlocks; ++b ) v += scratch[( static_cast<size_t>( b ) * 4 + c ) * maxN + i];
        out[i * CloudOutChannels + 24 + c] += v;
    }
}

}   // namespace additiv::gpu
