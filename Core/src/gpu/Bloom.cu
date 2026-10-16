/**
 * @file Bloom.cu
 * @brief The bloom (Phase 28): a plate's modes driven by everything in the field, their energy handed down an energy
 *        cascade between its bands, and the reduction into the frames.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

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

}   // namespace

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

}   // namespace additiv::gpu
