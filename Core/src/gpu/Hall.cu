/**
 * @file Hall.cu
 * @brief The hall: its ear and memory over the octaves, and the modal hall -- ModalModes complex one-poles driven by
 *        the block the voices just made, tuned to the music, summed in a fixed order.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

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

namespace
{

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
    // The redshift hall (Phase 33): every mode glides down through Span octaves, each from a place of its own in the
    // glide, and is reborn at the top -- a Shepard tail: heard in and out by a raised sine over the glide, so a ringing
    // mode sinks away and none appears with a step.
    float window = 1.0f;
    if ( mh.m_redshift > 0.0f )
    {
        float u = mh.m_redPhase + modalUni( mode, 11 );
        u -= floorf( u );
        f *= exp2f( -mh.m_redshift * mh.m_redSpan * u );
        const float s = __sinf( 3.1415927f * u );
        window = 1.0f - mh.m_redshift + mh.m_redshift * s * s;
    }
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
                 : sqrtf( 2.0f * fmaxf( 1.0f - r * r, 0.0f ) / static_cast<float>( ModalModes ) ) * weight * window;
    __sincosf( 1.5707963f * modalUni( mode, 7 ), &md.m_inR, &md.m_inL );   // its share of the left and the right input
    const float phL = 6.2831853f * modalUni( mode, 8 );
    const float phR = phL + mh.m_width * 3.1415927f * ( 2.0f * modalUni( mode, 9 ) - 1.0f );
    __sincosf( phL, &md.m_oLs, &md.m_oLc );
    __sincosf( phR, &md.m_oRs, &md.m_oRc );
    float k = silent ? 0.0f : window;
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

}   // namespace

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

}   // namespace additiv::gpu
