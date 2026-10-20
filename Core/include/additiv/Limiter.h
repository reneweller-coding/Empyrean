/**
 * @file Limiter.h
 * @brief The output's true-peak limiter: the family's (Totality's tot/fx/Dynamics.h, 20.10.2026), so that no peak
 *        between the samples passes -1 dBTP -- the soft clip before it rounds the sample peaks, this catches what a
 *        player's reconstruction (and a lossy codec after it) would make of them.
 * @author Rene Weller
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace additiv
{

/** @brief 8x interpolation for true-peak estimates: seven phases of twenty-four taps each. */
class TruePeakInterpolator
{
public:
    static constexpr int Phases = 8;        ///< points per input sample; Phases - 1 are interpolated
    static constexpr int Taps = 24;         ///< taps per phase
    static constexpr int Half = Taps / 2;   ///< samples of lookahead the filter needs
    static constexpr int History = Taps;    ///< samples of input a caller has to keep for between()
    /** @brief Designs the Kaiser-windowed sinc fractional-delay bank (beta 4, full band). */
    TruePeakInterpolator();
    /**
     * @brief Largest magnitude of the Phases - 1 interpolated points between x[0] and x[1].
     * @param[in] x  pointer to the earlier sample; x[-(Half - 1)] .. x[Half] must be readable
     * @return the largest magnitude of the interpolated points
     */
    double between( const float *x ) const
    {
        const float *p = x - Half + 1;
        double       peak = 0.0;
        for ( int k = 0; k < Phases - 1; ++k )
        {
            // Four partial sums, not one. A single accumulator makes the inner loop a chain of Taps
            // dependent additions, and the whole bank is then latency-bound: with Taps at twenty-four
            // that alone cost 64 % of the offline render's time (measured 16.09.2026). Four independent
            // chains of six additions each cut it to a quarter. Taps is a multiple of four by design.
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for ( int m = 0; m < Taps; m += 4 )
            {
                s0 += m_h[k][m] * static_cast<double>( p[m] );
                s1 += m_h[k][m + 1] * static_cast<double>( p[m + 1] );
                s2 += m_h[k][m + 2] * static_cast<double>( p[m + 2] );
                s3 += m_h[k][m + 3] * static_cast<double>( p[m + 3] );
            }
            peak = std::max( peak, std::fabs( ( s0 + s1 ) + ( s2 + s3 ) ) );
        }
        return peak;
    }
    /**
     * @brief Largest magnitude any interpolated point can reach, per unit of the largest input sample.
     *
     * max_k sum_m |h_k[m]|, the induced l-infinity norm of the bank: if every sample the filter can see
     * is at most @c a, no interpolated point can exceed @c gainBound() * a. A caller that only needs to
     * know whether the peak clears a threshold can skip the whole bank when it does not, which is
     * exact, not an approximation -- see TruePeakLimiter::process and LoudnessMeter::process.
     * @return the largest gain any interpolated point can have over the largest input sample
     */
    double gainBound() const { return m_bound; }
    /**
     * @brief Largest magnitude among the Taps samples between() would read from @p x.
     * @param[in] x  the earlier sample of the pair, as between() takes it
     * @return the largest magnitude among the samples between() would read
     */
    static float windowPeak( const float *x )
    {
        const float *p = x - Half + 1;
        float        a = 0.0f, b = 0.0f;
        for ( int m = 0; m < Taps; m += 2 )
        {
            a = std::max( a, std::fabs( p[m] ) );
            b = std::max( b, std::fabs( p[m + 1] ) );
        }
        return std::max( a, b );
    }

private:
    double m_h[Phases - 1][Taps] = {};   ///< the fractional-delay filters, one per interpolated phase
    double m_bound = 1.0;                ///< gainBound()
};

/** @brief Stereo lookahead true-peak limiter. */
class TruePeakLimiter
{
public:
    /**
     * @brief Allocates for a sample rate; @p lookaheadMs sets the window.
     * @param[in] sampleRate   the sample rate in Hz
     * @param[in] lookaheadMs  the lookahead window in milliseconds
     */
    void prepare( double sampleRate, float lookaheadMs = 1.5f );
    /** @brief Clears the delay lines and the gain. */
    void reset();
    /**
     * @brief Ceiling in dBTP and recovery time.
     * @param[in] ceilingDb  the ceiling in dBTP
     * @param[in] releaseMs  the recovery time in milliseconds
     */
    void set( float ceilingDb, float releaseMs );
    /**
     * @brief Processes in place.
     * @param[in,out] L  the left channel, processed in place
     * @param[in,out] R  the right channel, processed in place
     * @param[in]     n  how many samples
     */
    void process( float *L, float *R, int n );
    /**
     * @brief Samples of delay the limiter adds.
     * @return the samples of delay the limiter adds
     */
    int latency() const { return m_window - 1 + TruePeakInterpolator::Half; }
    /**
     * @brief Gain reduction of the last sample, dB (>= 0).
     * @return the gain reduction of the last sample in dB, >= 0
     */
    float reduction() const { return m_reduction; }

private:
    TruePeakInterpolator m_interp;             ///< the true-peak interpolator
    double               m_sr = 48000.0;       ///< the sample rate, Hz
    int                  m_window = 72;        ///< the lookahead window, samples
    float                m_ceiling = 0.891f;   ///< the ceiling, linear
    double               m_release = 0.999;    ///< the gain's per-sample recovery factor
    // Input history for the interpolator (both channels): 2 * Taps slots, every sample written into
    // both halves, so the Taps the filter needs are always contiguous (see process()).
    std::vector<float> m_histL;         ///< the interpolator's input history, left
    std::vector<float> m_histR;         ///< ... right
    int                m_histPos = 0;   ///< where the history is written
    double m_prevBetween = 0.0;   ///< the last sample's interpolated peak (a peak between two samples counts for both)
    /// Required gains, the sliding-minimum deque and the moving average.
    std::vector<double> m_req;
    std::vector<int>    m_dq;                   ///< the sliding minimum's deque: positions of required gains
    int                 m_dqHead = 0;           ///< the deque's head
    int                 m_dqTail = 0;           ///< the deque's tail
    std::vector<double> m_minRing;              ///< the sliding minima of the last window
    double              m_minSum = 0.0;         ///< their sum: the moving average times the window
    long long           m_t = 0;                ///< samples processed
    int                 m_sinceRecompute = 0;   ///< samples since m_minSum was summed afresh (rounding)
    double              m_gain = 1.0;           ///< the gain applied
    // The audio delay.
    std::vector<float> m_delayL;             ///< the audio delay, left
    std::vector<float> m_delayR;             ///< the audio delay, right
    int                m_delayPos = 0;       ///< where the delay is written
    float              m_reduction = 0.0f;   ///< the reduction of the last sample, dB
};

}   // namespace additiv
