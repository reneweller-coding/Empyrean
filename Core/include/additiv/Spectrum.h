/**
 * @file Spectrum.h
 * @brief Noctuary's spectral formula (Voice.cpp) as functions both the GPU and the host use: the
 *        kernel shapes every partial with them, the engine normalises a voice and builds the
 *        conductor's spectrum (and the Timbre scale) with exactly the same numbers.
 * @author Rene Weller
 *
 * @par Compile Flags:
 *   @arg @c __CUDACC__ - set by nvcc: ADDITIV_HD makes the functions host and device functions
 */
#pragma once

#include <cmath>

#ifdef __CUDACC__
  #define ADDITIV_HD __host__ __device__   ///< compiled for the host and the GPU
#else
  #define ADDITIV_HD   ///< compiled for the host and the GPU (here: the host only)
#endif

namespace additiv
{

/**
 * @brief The shape of harmonic h: the tilt, the odd/even weight and the brightness window, here over
 *        up to a few hundred harmonics instead of Noctuary's 32.
 * @param[in] h           the harmonic, 1 ..
 * @param[in] tilt        h^-tilt
 * @param[in] oddEven     -1 .. 1: positive thins the even harmonics, negative the odd ones above the first
 * @param[in] brightness  0 .. 1: harmonics above 1 + b^2 * 200 fade out over a quarter of that (at least 6)
 * @return the relative amplitude
 */
ADDITIV_HD inline float harmonicShape( int h, float tilt, float oddEven, float brightness )
{
    const float hf = static_cast<float>( h );
    float       a = powf( hf, -tilt );
    if ( oddEven > 0.0f && ( h % 2 ) == 0 ) a *= 1.0f - oddEven;
    if ( oddEven < 0.0f && ( h % 2 ) == 1 && h > 1 ) a *= 1.0f + oddEven;
    const float hc = 1.0f + brightness * brightness * 200.0f;
    if ( hf > hc )
    {
        const float x = fminf( ( hf - hc ) / fmaxf( 6.0f, 0.25f * hc ), 1.0f );
        a *= 0.5f * ( 1.0f + cosf( 3.14159265f * x ) );
    }
    return a;
}

/**
 * @brief The presence bell: a parabola in log frequency on 3.2 kHz, zero at +-0.8 octave, up to
 *        presenceDb at the ear and nothing on the far plane.
 * @param[in] f           Hz
 * @param[in] presenceDb  0 .. 6
 * @param[in] distance    0 .. 1
 * @return the gain, 1 outside the bell
 */
ADDITIV_HD inline float presenceGain( float f, float presenceDb, float distance )
{
    if ( presenceDb <= 0.0f || f <= 0.0f ) return 1.0f;
    const float lin = powf( 10.0f, presenceDb * ( 1.0f - distance ) * 0.05f ) - 1.0f;
    const float x = ( log2f( f ) - 11.643856f ) * ( 1.0f / 0.8f );   // log2(3200)
    return ( x > -1.0f && x < 1.0f ) ? 1.0f + lin * ( 1.0f - x * x ) : 1.0f;
}

/**
 * @brief The Pad Low Cut: 12 dB/oct below the cut.
 * @param[in] f       Hz
 * @param[in] lowCut  Hz, 0 = off
 * @return the gain
 */
ADDITIV_HD inline float lowCutGain( float f, float lowCut )
{
    if ( lowCut <= 0.0f || f >= lowCut ) return 1.0f;
    const float r = f / lowCut;
    return r * r;
}

/**
 * @brief The stiff string's stretch of harmonic h.
 * @param[in] h  the harmonic
 * @param[in] B  the inharmonicity
 * @return sqrt(1 + B h^2)
 */
ADDITIV_HD inline float stretchOf( int h, float B )
{
    const float hf = static_cast<float>( h );
    return sqrtf( 1.0f + B * hf * hf );
}

/**
 * @brief The horizon (Phase 26): a ratio r over the fundamental bent towards a horizon R times above
 *        it, 1 + (R - 1) tanh((r - 1) / (R - 1)) -- the lowest harmonics stay where they were (the slope at
 *        the fundamental is 1), the higher ones crowd ever closer below the horizon and never pass it.
 *        `amount` blends from r (0) to the bent ratio (1); both rise with r, so the blend does too.
 * @param[in] r       the ratio before
 * @param[in] amount  0 .. 1
 * @param[in] R       the horizon over the fundamental, above 1
 * @return the ratio after
 */
ADDITIV_HD inline float horizonBend( float r, float amount, float R )
{
    if ( amount <= 0.0f || R <= 1.0f ) return r;
    const float span = R - 1.0f;
    return r + amount * ( 1.0f + span * tanhf( ( r - 1.0f ) / span ) - r );
}

/**
 * @brief Where harmonic h of a voice lies, as a multiple of its fundamental: h stretched to
 *        h^s (Stretch: an octave of s times 1200 cents, Sethares' stretched timbres), then by
 *        the stiff string, then bent towards the horizon (horizonBend).
 * @param[in] h        the harmonic
 * @param[in] B        the inharmonicity
 * @param[in] s        the stretch's exponent (1: harmonic)
 * @param[in] horizon  0 .. 1: how far the harmonics bend towards the horizon (0: not at all)
 * @param[in] R        the horizon over the fundamental
 * @return h^s sqrt(1 + B h^2), bent
 */
ADDITIV_HD inline float partialRatio( int h, float B, float s, float horizon = 0.0f, float R = 1.0f )
{
    const float hf = static_cast<float>( h );
    return horizonBend( ( s == 1.0f ? hf : exp2f( s * log2f( hf ) ) ) * stretchOf( h, B ), horizon, R );
}

constexpr int SeasonFamilies = 5;    ///< Seasons: the families of harmonics that take turns
constexpr int AvalancheSites = 48;   ///< Avalanche: the harmonics the rice pile lies on (2 .. 49)
constexpr int SubClouds = 8;         ///< Subharmonics: the clouds below and between the harmonics

/**
 * @brief Seasons: harmonic h's family, by the odd number h is a power of two times -- the octaves of
 *        the fundamental (2, 4, 8 ..), of the fifth (3, 6, 12 ..), of the third (5, 10 ..), of the
 *        seventh (7, 14 ..), of the ninth (9, 18 ..); 11 and above go round again.
 * @param[in] h  the harmonic, 2 ..
 * @return 0 .. SeasonFamilies - 1
 */
ADDITIV_HD inline int seasonFamily( int h )
{
    while ( h > 1 && ( h & 1 ) == 0 ) h >>= 1;
    return ( ( h - 1 ) / 2 ) % SeasonFamilies;
}

/**
 * @brief Subharmonics: where sub cloud j lies over the fundamental -- the period doubled (1/2, 3/2,
 *        5/2, 7/2: pumped by harmonics 1, 3, 5, 7) and doubled again (1/4, 3/4, 5/4, 7/4: pumped by the
 *        four above).
 * @param[in] j  0 .. SubClouds - 1
 * @return the ratio
 */
ADDITIV_HD inline float subRatio( int j )
{
    return j < 4 ? 0.5f * static_cast<float>( 2 * j + 1 ) : 0.25f * static_cast<float>( 2 * ( j - 4 ) + 1 );
}

/**
 * @brief Overblow: the amplitude of harmonic h under the registers' shares -- a register k (the note's
 *        k-th harmonic as its fundamental) keeps the harmonics it divides, so the shares of every
 *        register dividing h add, as powers.
 * @param[in] h      the harmonic, 1 ..
 * @param[in] share  the four registers' shares (1, 2, 3, 4), summing to 1
 * @return the amplitude factor, 0 .. 1
 */
ADDITIV_HD inline float registerGain( int h, const float *share )
{
    float p = share[0];
    if ( ( h & 1 ) == 0 ) p += share[1];
    if ( h % 3 == 0 ) p += share[2];
    if ( ( h & 3 ) == 0 ) p += share[3];
    return sqrtf( fminf( p, 1.0f ) );
}

/**
 * @brief The Inharmonic knob as the stiff string's B. Noctuary's 0.02 i^2 was made for 32
 *        harmonics; over a few hundred it would throw the top far past Nyquist, so here the same
 *        knob reaches the same stretch at harmonic 32 and grows like a string above it.
 * @param[in] knob  0 .. 1
 * @return B
 */
inline float inharmonicB( float knob )
{
    return knob * knob * 0.02f;
}

/**
 * @brief A two-pole band pass's magnitude, unity at its centre.
 * @param[in] r  frequency over centre
 * @param[in] q  Q
 * @return the gain
 */
ADDITIV_HD inline float bandPassGain( float r, float q )
{
    const float a = 1.0f - r * r, b = r / q;
    return b / sqrtf( a * a + b * b + 1.0e-12f );
}

/**
 * @brief The formants of u, o, a, e, i (Hz), three each.
 * @param[in] vowel  0 .. 4
 * @param[in] k      0 .. 2
 * @return Hz
 */
ADDITIV_HD inline float formantHz( int vowel, int k )
{
    switch ( vowel * 3 + k )
    {
    case 0: return 350.0f;
    case 1: return 600.0f;
    case 2: return 2400.0f;
    case 3: return 450.0f;
    case 4: return 800.0f;
    case 5: return 2830.0f;
    case 6: return 800.0f;
    case 7: return 1150.0f;
    case 8: return 2900.0f;
    case 9: return 400.0f;
    case 10: return 2000.0f;
    case 11: return 2550.0f;
    case 12: return 270.0f;
    case 13: return 2140.0f;
    default: return 2950.0f;
    }
}

/**
 * @brief The voice filter's Q from the Resonance knob (the engine's, and the picture's curve).
 * @param[in] model      FilterModel (Cloud.h): 7 Peak, 8 Formant, the others a Q
 * @param[in] resonance  0 .. 1
 * @return Q (Peak: the bell's gain; Formant: the formants' Q)
 */
ADDITIV_HD inline float filterQ( int model, float resonance )
{
    if ( model == 7 ) return 1.0f + 3.0f * resonance;   // the bell's gain, up to +12 dB
    if ( model == 8 ) return 4.0f + 16.0f * resonance;
    return 0.707f + resonance * resonance * 11.3f;   // up to Q 12
}

/**
 * @brief The voice filter's response at a frequency: the analogue prototype's magnitude. The GPU reads
 *        it at every partial; the picture draws it.
 * @param[in] model   FilterModel (Cloud.h): 0 Off, 1 LP 6, 2 LP 12, 3 LP 24, 4 HP 12, 5 BP 12, 6 Notch, 7 Peak, 8
 *                    Formant
 * @param[in] f       Hz
 * @param[in] cutoff  Hz
 * @param[in] q       filterQ()
 * @param[in] vowel   Formant: 0 .. 4, u o a e i
 * @return the gain
 */
ADDITIV_HD inline float filterResponse( int model, float f, float cutoff, float q, float vowel )
{
    if ( model <= 0 ) return 1.0f;
    const float r = f / fmaxf( cutoff, 1.0f );
    const float a = 1.0f - r * r, b = r / fmaxf( q, 0.1f );
    const float den2 = a * a + b * b + 1.0e-12f;
    switch ( model )
    {
    case 1: return 1.0f / sqrtf( 1.0f + r * r );
    case 2: return 1.0f / sqrtf( den2 );
    case 3: return 1.0f / den2;
    case 4: return r * r / sqrtf( den2 );
    case 5: return b / sqrtf( den2 );
    case 6: return fabsf( a ) / sqrtf( den2 );
    case 7: return 1.0f + ( fmaxf( q, 1.0f ) - 1.0f ) * bandPassGain( r, 2.0f );
    case 8:
    {
        const float v = fminf( fmaxf( vowel, 0.0f ), 4.0f );
        const int   i0 = v < 3.0f ? static_cast<int>( v ) : 3;
        const float fr = v - static_cast<float>( i0 );
        float       g = 0.02f;
        const float w[3] = { 1.0f, 0.5f, 0.25f };
        for ( int k = 0; k < 3; ++k )
        {
            const float lo = formantHz( i0, k ), hi = formantHz( i0 + 1, k );
            g += w[k] * bandPassGain( f / ( lo * exp2f( fr * log2f( hi / lo ) ) ), q );
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
 * @param[in] cutoff  Hz
 * @return 0 .. 4
 */
inline float vowelOf( float cutoff )
{
    return 4.0f * fminf( fmaxf( std::log( cutoff / 40.0f ) / std::log( 18000.0f / 40.0f ), 0.0f ), 1.0f );
}

}   // namespace additiv
