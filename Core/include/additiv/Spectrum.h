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

/**
 * @brief A two-pole band pass's magnitude, unity at its centre.
 * @param r  frequency over centre
 * @param q  Q
 * @return   the gain
 */
ADDITIV_HD inline float bandPassGain(float r, float q)
{
    const float a = 1.0f - r * r, b = r / q;
    return b / sqrtf(a * a + b * b + 1.0e-12f);
}

/**
 * @brief The formants of u, o, a, e, i (Hz), three each.
 * @param vowel  0 .. 4
 * @param k      0 .. 2
 * @return       Hz
 */
ADDITIV_HD inline float formantHz(int vowel, int k)
{
    switch (vowel * 3 + k) {
    case 0: return 350.0f;  case 1: return 600.0f;  case 2: return 2400.0f;
    case 3: return 450.0f;  case 4: return 800.0f;  case 5: return 2830.0f;
    case 6: return 800.0f;  case 7: return 1150.0f; case 8: return 2900.0f;
    case 9: return 400.0f;  case 10: return 2000.0f; case 11: return 2550.0f;
    case 12: return 270.0f; case 13: return 2140.0f; default: return 2950.0f;
    }
}

/**
 * @brief The voice filter's Q from the Resonance knob (the engine's, and the picture's curve).
 * @param model      FilterModel (Cloud.h): 7 Peak, 8 Formant, the others a Q
 * @param resonance  0 .. 1
 * @return           Q (Peak: the bell's gain; Formant: the formants' Q)
 */
ADDITIV_HD inline float filterQ(int model, float resonance)
{
    if (model == 7) return 1.0f + 3.0f * resonance;   // the bell's gain, up to +12 dB
    if (model == 8) return 4.0f + 16.0f * resonance;
    return 0.707f + resonance * resonance * 11.3f;    // up to Q 12
}

/**
 * @brief The voice filter's response at a frequency: the analogue prototype's magnitude. The GPU reads
 *        it at every partial; the picture draws it.
 * @param model   FilterModel (Cloud.h): 0 Off, 1 LP 6, 2 LP 12, 3 LP 24, 4 HP 12, 5 BP 12, 6 Notch, 7 Peak, 8 Formant
 * @param f       Hz
 * @param cutoff  Hz
 * @param q       filterQ()
 * @param vowel   Formant: 0 .. 4, u o a e i
 * @return        the gain
 */
ADDITIV_HD inline float filterResponse(int model, float f, float cutoff, float q, float vowel)
{
    if (model <= 0) return 1.0f;
    const float r = f / fmaxf(cutoff, 1.0f);
    const float a = 1.0f - r * r, b = r / fmaxf(q, 0.1f);
    const float den2 = a * a + b * b + 1.0e-12f;
    switch (model) {
    case 1: return 1.0f / sqrtf(1.0f + r * r);
    case 2: return 1.0f / sqrtf(den2);
    case 3: return 1.0f / den2;
    case 4: return r * r / sqrtf(den2);
    case 5: return b / sqrtf(den2);
    case 6: return fabsf(a) / sqrtf(den2);
    case 7: return 1.0f + (fmaxf(q, 1.0f) - 1.0f) * bandPassGain(r, 2.0f);
    case 8: {
        const float v = fminf(fmaxf(vowel, 0.0f), 4.0f);
        const int i0 = v < 3.0f ? static_cast<int>(v) : 3;
        const float fr = v - static_cast<float>(i0);
        float g = 0.02f;
        const float w[3] = { 1.0f, 0.5f, 0.25f };
        for (int k = 0; k < 3; ++k) {
            const float lo = formantHz(i0, k), hi = formantHz(i0 + 1, k);
            g += w[k] * bandPassGain(f / (lo * exp2f(fr * log2f(hi / lo))), q);
        }
        // Three band passes at unity pass little between them: measured over the factory chord the
        // formant presets sat 18 dB under the rest of the bank. Made up here, not in every preset.
        return 3.0f * g;
    }
    default: return 1.0f;
    }
}

/**
 * @brief The Formant filter's vowel from the cutoff: u at 40 Hz to i at 18 kHz, on the log axis.
 * @param cutoff  Hz
 * @return        0 .. 4
 */
inline float vowelOf(float cutoff) { return 4.0f * fminf(fmaxf(std::log(cutoff / 40.0f) / std::log(18000.0f / 40.0f), 0.0f), 1.0f); }

} // namespace additiv
