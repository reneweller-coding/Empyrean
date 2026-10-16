/**
 * @file Modulation.h
 * @brief The modulation system (after Noctuary's Modulation.h): sources -- four LFOs, two
 *        envelopes on the factory shapes, four macros, the mod wheel, pressure, the last velocity,
 *        a slow random, and Noctuary's fields and chaos (Lenia, Lorenz, Roessler, the Kuramoto
 *        bank), the Beat, the conductor's Cascade, Root Age and three per-note sources -- and a
 *        matrix of sixteen rows, source -> target x depth, each with an optional via source.
 *
 * Depth is a fraction of the target's normalised range (toNorm in Params.h), so the same number
 * means the same on a cutoff in hertz and on a mix. The sources' settings are parameters (a host
 * automates them); the matrix rows are data, a short text that travels in presets and the plugin
 * state -- sixteen rows as three parameters each would fill the automation list for little gain.
 * Every source moves without steps: ramps and the square return over 7 % of a cycle, random
 * shapes glide. The envelopes run on a phrase clock: they restart when a note arrives into
 * silence, not on every note of a cluster, or a shape spanning a minute would never get anywhere.
 *
 * The LFOs and the envelopes belong to a sound (ModSetup, OwnModulators): when two sounds blend
 * (Morph.h), each row keeps running on the modulators of the sound it came from -- a slow sine on
 * the brightness stays a slow sine while it fades, though the sound it fades into uses LFO 1 for
 * quick steps on the width. The other sources are the instrument's, one of each.
 * @author Rene Weller
 */
#pragma once

#include "additiv/Dsp.h"
#include "additiv/Params.h"

#include <cstdint>
#include <string>

namespace additiv
{

/** @brief Where a modulation comes from. */
enum class ModSourceE : int
{
    OFF = 0,     ///< nothing (an empty row)
    LFO1,        ///< LFO 1, -1 .. 1
    LFO2,        ///< LFO 2
    LFO3,        ///< LFO 3
    LFO4,        ///< LFO 4
    ENV1,        ///< envelope 1, its shape's levels (0 .. 1, or -1 .. 1 for the bipolar shapes)
    ENV2,        ///< envelope 2
    MACRO1,      ///< macro 1, 0 .. 1
    MACRO2,      ///< macro 2
    MACRO3,      ///< macro 3
    MACRO4,      ///< macro 4
    MOD_WHEEL,   ///< the mod wheel (CC 1), 0 .. 1
    PRESSURE,    ///< channel pressure, 0 .. 1
    VELOCITY,    ///< the last note's velocity, 0 .. 1
    RANDOM,      ///< a slow random curve, -1 .. 1
    // Noctuary's (12.10.2026)
    LENIA1,        ///< the Lenia field read near one corner, 0 .. 1
    LENIA2,        ///< near the second
    LENIA3,        ///< near the third
    LENIA4,        ///< near the fourth
    LORENZ_X,      ///< the Lorenz attractor, -1 .. 1 (its two lobes)
    LORENZ_Y,      ///< its y
    LORENZ_Z,      ///< its z
    ROSSLER_X,     ///< the Roessler attractor, -1 .. 1 (a slow spiral, a sudden climb)
    ROSSLER_Y,     ///< its y
    ROSSLER_Z,     ///< its z
    KURA1,         ///< the first of four coupled Kuramoto oscillators, -1 .. 1
    KURA2,         ///< the second
    KURA3,         ///< the third
    KURA4,         ///< the fourth
    BEAT,          ///< an oscillator at the rate the two lowest voices' interval beats (still when in tune), -1 .. 1
    CASCADE,       ///< the conductor's excitation (a Hawkes process: an event breeds events), 0 .. 1
    ROOT_AGE,      ///< how long the conductor's root has stood, 0 at a change .. 1 after ten minutes
    NOTE,          ///< the loudest voice's note, 0 (C1) .. 1 (C8)
    DISTANCE,      ///< the loudest voice's distance, 0 at the ear .. 1 far
    RANDOM_NOTE,   ///< one draw per note, -1 .. 1, held until the next note
    // Phase 10d: what the instrument hears of itself
    LOUDNESS,    ///< its level, 0 at -70 dBFS .. 1 at -10, followed over a third of a second
    CENTROID,    ///< where its energy stands, 0 at 20 Hz .. 1 at 20 kHz (log), over half a second
    FLUX,        ///< how much its spectrum changes, 0 .. 1, over a fifth of a second
    ROUGHNESS,   ///< how rough its chord is (Sethares' dissonance of the voices' harmonics), 0 .. 1, over a second
    // Phase 10g: the reservoir
    RESERVOIR1,   ///< the reservoir's first readout, -1 .. 1
    RESERVOIR2,   ///< its second
    RESERVOIR3,   ///< its third
    RESERVOIR4,   ///< its fourth
    // Phase 30: the night
    NIGHT,       ///< the real night where the instrument is: 0 while the sun is up .. 1 in astronomical night
    RISSET,      ///< Risset's pulse (the rhythm that slows for ever), 0 .. 1
    LIGHTNING,   ///< a strike's flash (Thunder), 1 at the strike, fading over a second or two
    // Phase 33: space
    MERGER,   ///< a merger's wave (Merger): rising over the inspiral, 1 at the merger, gone with the ringdown
    COUNT     ///< number of sources
};
constexpr int            NumModSources = static_cast<int>( ModSourceE::COUNT );   ///< entries of ModSourceNames
extern const char *const ModSourceNames[NumModSources];                           ///< "Off", "LFO 1", ... for menus
extern const char *const ModSourceTokens[NumModSources];   ///< "off", "lfo1", ... for the text form

constexpr int ModRows = 16;   ///< rows of the matrix

/** @brief One row of the matrix. */
struct ModRow
{
    ModSourceE m_source = ModSourceE::OFF;   ///< where from
    int        m_target = -1;                ///< the ParamId it moves, -1 none
    float      m_depth = 0.0f;               ///< -1 .. 1 of the target's normalised range
    ModSourceE m_via =
        ModSourceE::OFF;   ///< a second source that scales the depth as an amount 0 .. 1 (Off: depth as it is)
    int m_setup =
        0;   ///< whose LFOs and envelopes it reads: an index into its sound's setups (MorphState); 0 in a plain matrix
};

/**
 * @brief Whether a source is one of a sound's own modulators (the four LFOs, the two envelopes).
 * @param[in] s  the source
 * @return true then
 */
inline bool isOwnSource( ModSourceE s )
{
    return s >= ModSourceE::LFO1 && s <= ModSourceE::ENV2;
}

/** @brief The settings of a sound's own modulators: its LFOs' and envelopes' shapes and rates. */
struct ModSetup
{
    int   m_lfoShape[4] = { 0, 1, 5, 6 };                  ///< each LFO's shape (LfoShapeNames)
    float m_lfoRate[4] = { 0.05f, 0.08f, 0.13f, 0.21f };   ///< each LFO's rate, Hz
    int   m_envShape[2] = { 2, 0 };                        ///< each envelope's factory shape
    float m_envRate[2] = { 0.25f, 0.1f };                  ///< each envelope's speed
    /**
     * @brief The same settings.
     * @param[in] o  the other
     * @return true when every one is equal
     */
    bool operator==( const ModSetup &o ) const;
    /**
     * @brief Different settings.
     * @param[in] o  the other
     * @return true when any one differs
     */
    bool operator!=( const ModSetup &o ) const { return !( *this == o ); }
};

/**
 * @brief A sound's own modulators as its parameters set them.
 * @param[in] values  NumParams values
 * @return the setup
 */
ModSetup modSetupFrom( const float *values );

/**
 * @brief A sound's own modulators running: four LFOs (their phases, random curves and steps) and
 *        two envelopes on the phrase clock. A copy runs on exactly as the original would, so a
 *        sound that joins a blend starts where the one already playing stands.
 */
class OwnModulators
{
public:
    /**
     * @brief Back to the start.
     * @param[in] seed  the dice of the random shapes
     */
    void reset( std::uint64_t seed );
    /**
     * @brief Advances by a block.
     * @param[in] dt       seconds
     * @param[in] envTime  seconds since the phrase began (the envelopes' clock, the instrument's)
     * @param[in] s        the settings
     */
    void advance( float dt, double envTime, const ModSetup &s );
    /**
     * @brief An LFO's or envelope's value after the last advance.
     * @param[in] s  the source
     * @return its value, 0 for any other
     */
    float value( ModSourceE s ) const;

private:
    /**
     * @brief One LFO's value at its phase.
     * @param[in] i      the LFO
     * @param[in] shape  LfoShapeNames
     * @return -1 .. 1
     */
    float lfoValue( int i, int shape ) const;

    double  m_phase[4] = { 0.0, 0.0, 0.0, 0.0 };          ///< each LFO's phase, 0 .. 1
    Drifter m_random[4];                                  ///< each LFO's random curve (Random shape)
    float   m_stepFrom[4] = { 0.0f, 0.0f, 0.0f, 0.0f };   ///< each LFO's Steps: the value it glides from
    float   m_stepTo[4] = { 0.0f, 0.0f, 0.0f, 0.0f };     ///< and to
    Rng     m_rng;                                        ///< its own dice, so a copy draws what the original would
    float   m_values[6] = {};                             ///< LFO 1 .. 4, Env 1 and 2
};

/**
 * @brief Whether a source runs 0 .. 1 (envelopes, macros, the hands, Lenia, Cascade, Root Age, Note,
 *        Distance) rather than -1 .. 1. A via source is read as an amount: a 0 .. 1 one as it is, a
 *        -1 .. 1 one as (v + 1) / 2 (Noctuary's rule).
 * @param[in] s  the source
 * @return true when it runs 0 .. 1
 */
bool isUnipolarSource( ModSourceE s );

/**
 * @brief A source's value as a via amount, 0 .. 1.
 * @param[in] s  the source
 * @param[in] v  its value
 * @return the amount
 */
inline float viaAmount( ModSourceE s, float v )
{
    const float a = isUnipolarSource( s ) ? v : 0.5f * ( v + 1.0f );
    return a < 0.0f ? 0.0f : ( a > 1.0f ? 1.0f : a );
}

/**
 * @brief Whether a parameter can be a target: the continuous and whole-number ones.
 * @param[in] id  the parameter
 * @return true when it can
 */
bool isModTarget( ParamIdE id );

/**
 * @brief A row in 64 bits, so the audio side reads it atomically.
 * @param[in] r  the row
 * @return the bits
 */
std::uint64_t packModRow( const ModRow &r );
/**
 * @brief A row back from its bits.
 * @param[in] bits  from packModRow
 * @return the row
 */
ModRow unpackModRow( std::uint64_t bits );

/**
 * @brief The rows as text: "lfo1>cutoff:0.35;macro1>motion_depth:-0.5:lenia2" (empty rows left out;
 *        a via source as a fourth field).
 * @param[in] rows  the rows
 * @param[in] n     how many
 * @return the text
 */
std::string modRowsToText( const ModRow *rows, int n );

/**
 * @brief Rows from their text; rows the text does not fill are emptied. Unknown sources or targets
 *        are skipped.
 * @param[in]  text  the text
 * @param[out] rows  receives the rows
 * @param[in]  n     how many there are room for
 * @return how many rows were read
 */
int modRowsFromText( const std::string &text, ModRow *rows, int n );

/** @brief What the sources need from the parameters and the performer, once a block. */
struct ModInputs
{
    int   m_lfoShape[4] = { 0, 1, 5, 6 };                  ///< each LFO's shape (LfoShapeNames)
    float m_lfoRate[4] = { 0.05f, 0.08f, 0.13f, 0.21f };   ///< each LFO's rate, Hz
    int   m_envShape[2] = { 2, 0 };                        ///< each envelope's factory shape
    float m_envRate[2] = { 0.25f, 0.1f };                  ///< each envelope's speed
    float m_macro[4] = { 0.0f, 0.0f, 0.0f, 0.0f };         ///< the macros
    float m_wheel = 0.0f;                                  ///< the mod wheel
    float m_pressure = 0.0f;                               ///< channel pressure
    float m_velocity = 0.0f;                               ///< the last note's velocity
    bool  m_phraseStart = false;                           ///< a note arrived into silence: the envelopes start over
    float m_coherence = 0.0f;                              ///< the Kuramoto bank's coupling, 0 (free) .. 1 (in step)
    float m_coherenceRate = 1.0f;   ///< its speed, times its natural periods of 23, 31, 41 and 53 s
    float m_leniaRate = 4.0f;       ///< Lenia fields per second
    float m_leniaGrowth = 0.15f;    ///< Lenia's growth centre (mu)
    float m_chaosPeriod = 120.0f;   ///< seconds: about a Lorenz lobe change, a Roessler turn
    float m_beatHz = 0.0f;          ///< the rate the two lowest voices' interval beats at (the engine finds it)
    float m_excitation = 0.0f;      ///< the conductor's Hawkes excitation
    float m_rootAge = 0.0f;         ///< seconds the conductor's root has stood
    float m_note = 0.5f;            ///< the loudest voice's note, 0 .. 1
    float m_distance = 0.5f;        ///< the loudest voice's distance, 0 .. 1
    float m_randomNote = 0.0f;      ///< the draw of the last note, -1 .. 1
    float m_heardDb = -120.0f;      ///< the level heard, dB of full scale
    float m_heardCentroid = 0.0f;   ///< the energy's centre heard, 0 .. 1
    float m_heardFlux = 0.0f;       ///< the change heard, 0 .. 1
    float m_roughness = 0.0f;       ///< the chord's roughness, 0 .. 1 (the engine finds it)
    float m_reservoir[4] = {};      ///< the reservoir's readouts, -1 .. 1 (from the GPU)
    float m_night = 0.0f;           ///< the night where the instrument is, 0 .. 1 (Sun.h)
    float m_risset = 0.0f;          ///< Risset's pulse, 0 .. 1
    float m_lightning = 0.0f;       ///< the last strike's flash, 0 .. 1
    float m_merger = 0.0f;          ///< a merger's wave, 0 .. 1
};

/** @brief The sources, advanced once per block on the renderer's thread. */
class Modulators
{
public:
    /**
     * @brief Back to the start, with a seed for the random shapes.
     * @param[in] seed  the dice
     */
    void reset( std::uint64_t seed );
    /**
     * @brief Advances every source by a block.
     * @param[in] dt  seconds
     * @param[in] in  the settings and the performer's controllers
     */
    void advance( float dt, const ModInputs &in );
    /**
     * @brief A source's value after the last advance.
     * @param[in] s  the source
     * @return its value
     */
    float value( ModSourceE s ) const
    {
        const int i = static_cast<int>( s );
        return i > 0 && i < NumModSources ? m_values[i] : 0.0f;
    }
    /** @brief The LFOs and envelopes as the parameters set them (a sound's own copies start from these). @return them */
    const OwnModulators &own() const { return m_own; }
    /** @brief The phrase clock: seconds since a note arrived into silence. @return seconds */
    double envTime() const { return m_envTime; }

private:
    /**
     * @brief One block of the Lenia field (Noctuary's: rows stepped at the rate, a dead or full field seeded again).
     * @param[in] dt    seconds
     * @param[in] rate  fields per second
     * @param[in] mu    growth centre
     */
    void stepLenia( float dt, float rate, float mu );
    /** @brief A few soft blobs on an empty torus, from the field's own dice. */
    void seedLenia();
    /**
     * @brief One block of the two attractors (fourth-order Runge-Kutta in their own time).
     * @param[in] dt      seconds
     * @param[in] period  the chaos period
     */
    void stepChaos( float dt, float period );

    OwnModulators m_own;    ///< the LFOs and envelopes on the parameters (the panel's, and where a sound's own begin)
    Drifter       m_slow;   ///< the Random source
    double        m_envTime = 0.0;                ///< seconds since the phrase began
    Rng           m_rng;                          ///< the dice
    float         m_values[NumModSources] = {};   ///< every source's value

    static constexpr int LeniaSize = 32;                            ///< the Lenia torus' side, cells
    static constexpr int LeniaRadius = 5;                           ///< its ring kernel's radius, cells
    float                m_lenia[LeniaSize * LeniaSize] = {};       ///< the field
    float                m_leniaNext[LeniaSize * LeniaSize] = {};   ///< the next field, built a few rows a block
    float  m_leniaKernel[( 2 * LeniaRadius + 1 ) * ( 2 * LeniaRadius + 1 )] = {};   ///< the normalised ring kernel
    int    m_leniaRow = 0;                                                          ///< the next row to step
    float  m_leniaRowAcc = 0.0f;               ///< rows owed, carried between blocks
    float  m_leniaTarget[4] = {};              ///< the four readings' new means
    float  m_leniaOut[4] = {};                 ///< the four readings, glided
    Rng    m_leniaRng;                         ///< the field's own dice (seeding)
    double m_lorenz[3] = { 0.1, 0.0, 20.0 };   ///< the Lorenz state
    double m_rossler[3] = { 1.0, 1.0, 1.0 };   ///< the Roessler state
    float  m_kuraPhase[4] = {};                ///< the Kuramoto bank's phases, radians
    float  m_beatHz = 0.0f;                    ///< the beat rate, followed over two seconds
    float  m_beatPhase = 0.0f;                 ///< the Beat oscillator's phase, radians
    float  m_cascade = 0.0f;                   ///< the excitation, smoothed over two seconds
    float  m_heard[4] = {};                    ///< loudness, centroid, flux, roughness, each followed
};

}   // namespace additiv
