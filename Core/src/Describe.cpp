/**
 * @file Describe.cpp
 * @brief What a stretch of stereo sound sounds like (Describe.h); moved here from the measuring tool
 *        (14.10.2026) so that the instrument can describe a recording the same way.
 * @author Rene Weller
 */
#include "additiv/Describe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

namespace additiv
{

namespace
{

constexpr int Fft = 4096;   ///< the analysis frame
constexpr int Hop = 2048;   ///< frames start this far apart
constexpr int Bands = 36;   ///< bands from 40 Hz to 16 kHz, a quarter of an octave each nearly

/**
 * @brief An in-place radix-2 FFT.
 * @param[in,out] x  Fft values
 */
void fft( std::vector<std::complex<double>> &x )
{
    const int n = static_cast<int>( x.size() );
    for ( int i = 1, j = 0; i < n; ++i )
    {
        int bit = n >> 1;
        for ( ; j & bit; bit >>= 1 ) j ^= bit;
        j ^= bit;
        if ( i < j ) std::swap( x[static_cast<size_t>( i )], x[static_cast<size_t>( j )] );
    }
    for ( int len = 2; len <= n; len <<= 1 )
    {
        const double               a = -2.0 * 3.14159265358979323846 / len;
        const std::complex<double> wl( std::cos( a ), std::sin( a ) );
        for ( int i = 0; i < n; i += len )
        {
            std::complex<double> w( 1.0, 0.0 );
            for ( int k = 0; k < len / 2; ++k )
            {
                const std::complex<double> u = x[static_cast<size_t>( i + k )],
                                           v = x[static_cast<size_t>( i + k + len / 2 )] * w;
                x[static_cast<size_t>( i + k )] = u + v;
                x[static_cast<size_t>( i + k + len / 2 )] = u - v;
                w *= wl;
            }
        }
    }
}

}   // namespace

/**
 * @brief The description of a stretch of stereo sound (the file comment).
 */
Description describe( const std::vector<float> &L, const std::vector<float> &R, double sr )
{
    Description  d;
    const size_t n = L.size();
    if ( n < static_cast<size_t>( Fft ) * 4 ) return d;
    std::vector<double> win( Fft );
    for ( int i = 0; i < Fft; ++i )
        win[static_cast<size_t>( i )] = 0.5 - 0.5 * std::cos( 2.0 * 3.14159265358979323846 * i / Fft );
    // The bands' edges in bins.
    std::array<int, Bands + 1> edge{};
    for ( int b = 0; b <= Bands; ++b )
        edge[static_cast<size_t>( b )] = std::clamp(
            static_cast<int>( std::lround( 40.0 * std::pow( 400.0, static_cast<double>( b ) / Bands ) * Fft / sr ) ), 1,
            Fft / 2 );
    std::vector<std::array<double, Bands>> frames;
    std::vector<std::complex<double>>      x( Fft );
    double                                 centroid = 0.0, flat = 0.0;
    int                                    counted = 0;
    for ( size_t at = 0; at + Fft <= n; at += Hop )
    {
        // Left and right in one transform: left the real part, right the imaginary.
        for ( int i = 0; i < Fft; ++i )
            x[static_cast<size_t>( i )] = { L[at + static_cast<size_t>( i )] * win[static_cast<size_t>( i )],
                                            R[at + static_cast<size_t>( i )] * win[static_cast<size_t>( i )] };
        fft( x );
        std::array<double, Bands> band{};
        double                    sum = 0.0, moment = 0.0;
        for ( int b = 0; b < Bands; ++b )
        {
            double p = 0.0;
            for ( int k = edge[static_cast<size_t>( b )];
                  k < std::max( edge[static_cast<size_t>( b ) + 1], edge[static_cast<size_t>( b )] + 1 ); ++k )
            {
                const std::complex<double> xk = x[static_cast<size_t>( k )],
                                           xn = std::conj( x[static_cast<size_t>( Fft - k )] );
                const std::complex<double> l = 0.5 * ( xk + xn ), r = std::complex<double>( 0.0, -0.5 ) * ( xk - xn );
                p += std::norm( l ) + std::norm( r );
            }
            band[static_cast<size_t>( b )] = p;
            sum += p;
            moment += p * ( b + 0.5 ) / Bands * std::log2( 400.0 );
        }
        frames.push_back( band );
        if ( sum > 1e-18 )
        {
            centroid += moment / sum;
            double lg = 0.0;
            for ( double p : band ) lg += std::log( p + sum * 1e-9 );
            flat += std::exp( lg / Bands ) / ( sum / Bands );
            ++counted;
        }
    }
    if ( counted > 0 )
    {
        d.m_brightness = centroid / counted;
        d.m_noise = flat / counted;
    }
    // The motion: the bands' magnitudes against themselves a lag before, as a share of them.
    const double frameSeconds = Hop / sr;
    double       motion = 0.0;
    int          lags = 0;
    for ( double lagSeconds : { 0.25, 1.0, 3.0 } )
    {
        const size_t lag = static_cast<size_t>( std::max( 1L, std::lround( lagSeconds / frameSeconds ) ) );
        if ( lag >= frames.size() ) continue;
        double change = 0.0, total = 0.0;
        for ( size_t f = lag; f < frames.size(); ++f )
            for ( int b = 0; b < Bands; ++b )
            {
                const double m = std::sqrt( frames[f][static_cast<size_t>( b )] ),
                             m0 = std::sqrt( frames[f - lag][static_cast<size_t>( b )] );
                change += std::fabs( m - m0 );
                total += 0.5 * ( m + m0 );
            }
        if ( total > 1e-12 )
        {
            motion += change / total;
            ++lags;
        }
    }
    d.m_motion = lags > 0 ? motion / lags : 0.0;
    double ll = 0.0, rr = 0.0, lr = 0.0;
    for ( size_t i = 0; i < n; ++i )
    {
        ll += static_cast<double>( L[i] ) * L[i];
        rr += static_cast<double>( R[i] ) * R[i];
        lr += static_cast<double>( L[i] ) * R[i];
    }
    d.m_width = ll > 1e-18 && rr > 1e-18 ? 1.0 - lr / std::sqrt( ll * rr ) : 0.0;
    return d;
}

}   // namespace additiv
