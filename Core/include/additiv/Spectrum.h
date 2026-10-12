/**
 * @file Spectrum.h
 * @brief Noctuary's spectral formula (Voice.cpp) as functions both the GPU and the host use: the
 *        kernel shapes every partial with them, the engine normalises a voice and builds the
 *        conductor's spectrum (and the Timbre scale) with exactly the same numbers.
 */
#pragma once

#include <cmath>

#ifdef __CUDACC__
#define ADDITIV_HD __host__ __device__   ///< compiled for the host and the GPU
#else
#define ADDITIV_HD                       ///< compiled for the host and the GPU (here: the host only)
#endif

namespace additiv {

/**
 * @brief The shape of harmonic h: the tilt, the odd/even weight and the brightness window, here over
 *        up to a few hundred harmonics instead of Noctuary's 32.
 * @param h           the harmonic, 1 ..
 * @param tilt        h^-tilt
 * @param oddEven     -1 .. 1: positive thins the even harmonics, negative the odd ones above the first
 * @param brightness  0 .. 1: harmonics above 1 + b^2 * 200 fade out over a quarter of that (at least 6)
 * @return            the relative amplitude
 */
ADDITIV_HD inline float harmonicShape(int h, float tilt, float oddEven, float brightness)
{
    const float hf = static_cast<float>(h);
    float a = powf(hf, -tilt);
    if (oddEven > 0.0f && (h % 2) == 0) a *= 1.0f - oddEven;
    if (oddEven < 0.0f && (h % 2) == 1 && h > 1) a *= 1.0f + oddEven;
    const float hc = 1.0f + brightness * brightness * 200.0f;
    if (hf > hc) {
        const float x = fminf((hf - hc) / fmaxf(6.0f, 0.25f * hc), 1.0f);
        a *= 0.5f * (1.0f + cosf(3.14159265f * x));
    }
    return a;
}

/**
 * @brief The presence bell: a parabola in log frequency on 3.2 kHz, zero at +-0.8 octave, up to
 *        presenceDb at the ear and nothing on the far plane.
 * @param f           Hz
 * @param presenceDb  0 .. 6
 * @param distance    0 .. 1
 * @return            the gain, 1 outside the bell
 */
ADDITIV_HD inline float presenceGain(float f, float presenceDb, float distance)
{
    if (presenceDb <= 0.0f || f <= 0.0f) return 1.0f;
    const float lin = powf(10.0f, presenceDb * (1.0f - distance) * 0.05f) - 1.0f;
    const float x = (log2f(f) - 11.643856f) * (1.0f / 0.8f);   // log2(3200)
    return (x > -1.0f && x < 1.0f) ? 1.0f + lin * (1.0f - x * x) : 1.0f;
}

/**
 * @brief The Pad Low Cut: 12 dB/oct below the cut.
 * @param f       Hz
 * @param lowCut  Hz, 0 = off
 * @return        the gain
 */
ADDITIV_HD inline float lowCutGain(float f, float lowCut)
{
    if (lowCut <= 0.0f || f >= lowCut) return 1.0f;
    const float r = f / lowCut;
    return r * r;
}

/**
 * @brief The stiff string's stretch of harmonic h.
 * @param h  the harmonic
 * @param B  the inharmonicity
 * @return   sqrt(1 + B h^2)
 */
ADDITIV_HD inline float stretchOf(int h, float B)
{
    const float hf = static_cast<float>(h);
    return sqrtf(1.0f + B * hf * hf);
}

/**
 * @brief Where harmonic h of a voice lies, as a multiple of its fundamental: h stretched to
 *        h^s (Stretch: an octave of s times 1200 cents, Sethares' stretched timbres) and then by
 *        the stiff string.
 * @param h  the harmonic
 * @param B  the inharmonicity
 * @param s  the stretch's exponent (1: harmonic)
 * @return   h^s sqrt(1 + B h^2)
 */
ADDITIV_HD inline float partialRatio(int h, float B, float s)
{
    const float hf = static_cast<float>(h);
    return (s == 1.0f ? hf : exp2f(s * log2f(hf))) * stretchOf(h, B);
}

/**
 * @brief The Inharmonic knob as the stiff string's B. Noctuary's 0.02 i^2 was made for 32
 *        harmonics; over a few hundred it would throw the top far past Nyquist, so here the same
 *        knob reaches the same stretch at harmonic 32 and grows like a string above it.
 * @param knob  0 .. 1
 * @return      B
 */
inline float inharmonicB(float knob) { return knob * knob * 0.02f; }

} // namespace additiv
