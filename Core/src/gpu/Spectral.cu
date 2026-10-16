/**
 * @file Spectral.cu
 * @brief The effects in the frequency domain, each one CUDA block with its FFT in shared memory: the spectral echo, the
 *        melt, and the instrument listening to itself.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

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

}   // namespace

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

namespace
{

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

}   // namespace

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

}   // namespace additiv::gpu
