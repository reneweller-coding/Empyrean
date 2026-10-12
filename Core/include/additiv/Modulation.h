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
 */
#pragma once

#include "additiv/Dsp.h"
#include "additiv/Params.h"

#include <cstdint>
#include <string>

namespace additiv {

/** @brief Where a modulation comes from. */
enum class ModSource : int {
    Off = 0,    ///< nothing (an empty row)
    Lfo1,       ///< LFO 1, -1 .. 1
    Lfo2,       ///< LFO 2
    Lfo3,       ///< LFO 3
    Lfo4,       ///< LFO 4
    Env1,       ///< envelope 1, its shape's levels (0 .. 1, or -1 .. 1 for the bipolar shapes)
    Env2,       ///< envelope 2
    Macro1,     ///< macro 1, 0 .. 1
    Macro2,     ///< macro 2
    Macro3,     ///< macro 3
    Macro4,     ///< macro 4
    ModWheel,   ///< the mod wheel (CC 1), 0 .. 1
    Pressure,   ///< channel pressure, 0 .. 1
    Velocity,   ///< the last note's velocity, 0 .. 1
    Random,     ///< a slow random curve, -1 .. 1
    // Noctuary's (12.10.2026)
    Lenia1,     ///< the Lenia field read near one corner, 0 .. 1
    Lenia2,     ///< near the second
    Lenia3,     ///< near the third
    Lenia4,     ///< near the fourth
    LorenzX,    ///< the Lorenz attractor, -1 .. 1 (its two lobes)
    LorenzY,    ///< its y
    LorenzZ,    ///< its z
    RosslerX,   ///< the Roessler attractor, -1 .. 1 (a slow spiral, a sudden climb)
    RosslerY,   ///< its y
    RosslerZ,   ///< its z
    Kura1,      ///< the first of four coupled Kuramoto oscillators, -1 .. 1
    Kura2,      ///< the second
    Kura3,      ///< the third
    Kura4,      ///< the fourth
    Beat,       ///< an oscillator at the rate the two lowest voices' interval beats (still when in tune), -1 .. 1
    Cascade,    ///< the conductor's excitation (a Hawkes process: an event breeds events), 0 .. 1
    RootAge,    ///< how long the conductor's root has stood, 0 at a change .. 1 after ten minutes
    Note,       ///< the loudest voice's note, 0 (C1) .. 1 (C8)
    Distance,   ///< the loudest voice's distance, 0 at the ear .. 1 far
    RandomNote, ///< one draw per note, -1 .. 1, held until the next note
    // Phase 10d: what the instrument hears of itself
    Loudness,   ///< its level, 0 at -70 dBFS .. 1 at -10, followed over a third of a second
    Centroid,   ///< where its energy stands, 0 at 20 Hz .. 1 at 20 kHz (log), over half a second
    Flux,       ///< how much its spectrum changes, 0 .. 1, over a fifth of a second
    Roughness,  ///< how rough its chord is (Sethares' dissonance of the voices' harmonics), 0 .. 1, over a second
    Count       ///< number of sources
};
constexpr int kNumModSources = static_cast<int>(ModSource::Count);   ///< entries of kModSourceNames
extern const char* const kModSourceNames[kNumModSources];             ///< "Off", "LFO 1", ... for menus
extern const char* const kModSourceTokens[kNumModSources];            ///< "off", "lfo1", ... for the text form

constexpr int kModRows = 16;   ///< rows of the matrix

/** @brief One row of the matrix. */
struct ModRow {
    ModSource source = ModSource::Off;   ///< where from
    int target = -1;                     ///< the ParamId it moves, -1 none
    float depth = 0.0f;                  ///< -1 .. 1 of the target's normalised range
    ModSource via = ModSource::Off;      ///< a second source that scales the depth as an amount 0 .. 1 (Off: depth as it is)
};

/**
 * @brief Whether a source runs 0 .. 1 (envelopes, macros, the hands, Lenia, Cascade, Root Age, Note,
 *        Distance) rather than -1 .. 1. A via source is read as an amount: a 0 .. 1 one as it is, a
 *        -1 .. 1 one as (v + 1) / 2 (Noctuary's rule).
 * @param s  the source
 * @return   true when it runs 0 .. 1
 */
bool isUnipolarSource(ModSource s);

/** @brief A source's value as a via amount, 0 .. 1. @param s the source @param v its value @return the amount */
inline float viaAmount(ModSource s, float v)
{
    const float a = isUnipolarSource(s) ? v : 0.5f * (v + 1.0f);
    return a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
}

/** @brief Whether a parameter can be a target: the continuous and whole-number ones. @param id the parameter @return true when it can */
bool isModTarget(ParamId id);

/** @brief A row in 64 bits, so the audio side reads it atomically. @param r the row @return the bits */
std::uint64_t packModRow(const ModRow& r);
/** @brief A row back from its bits. @param bits from packModRow @return the row */
ModRow unpackModRow(std::uint64_t bits);

/**
 * @brief The rows as text: "lfo1>cutoff:0.35;macro1>motion_depth:-0.5:lenia2" (empty rows left out;
 *        a via source as a fourth field).
 * @param rows  the rows
 * @param n     how many
 * @return      the text
 */
std::string modRowsToText(const ModRow* rows, int n);

/**
 * @brief Rows from their text; rows the text does not fill are emptied. Unknown sources or targets
 *        are skipped.
 * @param text  the text
 * @param rows  receives the rows
 * @param n     how many there are room for
 * @return      how many rows were read
 */
int modRowsFromText(const std::string& text, ModRow* rows, int n);

/** @brief What the sources need from the parameters and the performer, once a block. */
struct ModInputs {
    int lfoShape[4] = { 0, 1, 5, 6 };        ///< each LFO's shape (kLfoShapeNames)
    float lfoRate[4] = { 0.05f, 0.08f, 0.13f, 0.21f };   ///< each LFO's rate, Hz
    int envShape[2] = { 2, 0 };              ///< each envelope's factory shape
    float envRate[2] = { 0.25f, 0.1f };      ///< each envelope's speed
    float macro[4] = { 0.0f, 0.0f, 0.0f, 0.0f };   ///< the macros
    float wheel = 0.0f;                      ///< the mod wheel
    float pressure = 0.0f;                   ///< channel pressure
    float velocity = 0.0f;                   ///< the last note's velocity
    bool phraseStart = false;                ///< a note arrived into silence: the envelopes start over
    float coherence = 0.0f;                  ///< the Kuramoto bank's coupling, 0 (free) .. 1 (in step)
    float coherenceRate = 1.0f;              ///< its speed, times its natural periods of 23, 31, 41 and 53 s
    float leniaRate = 4.0f;                  ///< Lenia fields per second
    float leniaGrowth = 0.15f;               ///< Lenia's growth centre (mu)
    float chaosPeriod = 120.0f;              ///< seconds: about a Lorenz lobe change, a Roessler turn
    float beatHz = 0.0f;                     ///< the rate the two lowest voices' interval beats at (the engine finds it)
    float excitation = 0.0f;                 ///< the conductor's Hawkes excitation
    float rootAge = 0.0f;                    ///< seconds the conductor's root has stood
    float note = 0.5f;                       ///< the loudest voice's note, 0 .. 1
    float distance = 0.5f;                   ///< the loudest voice's distance, 0 .. 1
    float randomNote = 0.0f;                 ///< the draw of the last note, -1 .. 1
    float heardDb = -120.0f;                 ///< the level heard, dB of full scale
    float heardCentroid = 0.0f;              ///< the energy's centre heard, 0 .. 1
    float heardFlux = 0.0f;                  ///< the change heard, 0 .. 1
    float roughness = 0.0f;                  ///< the chord's roughness, 0 .. 1 (the engine finds it)
};

/** @brief The sources, advanced once per block on the renderer's thread. */
class Modulators {
public:
    /** @brief Back to the start, with a seed for the random shapes. @param seed the dice */
    void reset(std::uint64_t seed);
    /**
     * @brief Advances every source by a block.
     * @param dt  seconds
     * @param in  the settings and the performer's controllers
     */
    void advance(float dt, const ModInputs& in);
    /** @brief A source's value after the last advance. @param s the source @return its value */
    float value(ModSource s) const { const int i = static_cast<int>(s); return i > 0 && i < kNumModSources ? values_[i] : 0.0f; }

private:
    /**
     * @brief One LFO's value at its phase.
     * @param i      the LFO
     * @param shape  kLfoShapeNames
     * @return       -1 .. 1
     */
    float lfoValue(int i, int shape) const;
    /** @brief One block of the Lenia field (Noctuary's: rows stepped at the rate, a dead or full field seeded again). @param dt seconds @param rate fields per second @param mu growth centre */
    void stepLenia(float dt, float rate, float mu);
    /** @brief A few soft blobs on an empty torus, from the field's own dice. */
    void seedLenia();
    /** @brief One block of the two attractors (fourth-order Runge-Kutta in their own time). @param dt seconds @param period the chaos period */
    void stepChaos(float dt, float period);

    double phase_[4] = { 0.0, 0.0, 0.0, 0.0 };      ///< each LFO's phase, 0 .. 1
    Drifter random_[4];                              ///< each LFO's random curve (Random shape)
    float stepFrom_[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; ///< each LFO's Steps: the value it glides from
    float stepTo_[4] = { 0.0f, 0.0f, 0.0f, 0.0f };   ///< and to
    Drifter slow_;                                   ///< the Random source
    double envTime_ = 0.0;                           ///< seconds since the phrase began
    Rng rng_;                                        ///< the dice
    float values_[kNumModSources] = {};              ///< every source's value

    static constexpr int kLeniaSize = 32;            ///< the Lenia torus' side, cells
    static constexpr int kLeniaRadius = 5;           ///< its ring kernel's radius, cells
    float lenia_[kLeniaSize * kLeniaSize] = {};      ///< the field
    float leniaNext_[kLeniaSize * kLeniaSize] = {};  ///< the next field, built a few rows a block
    float leniaKernel_[(2 * kLeniaRadius + 1) * (2 * kLeniaRadius + 1)] = {};   ///< the normalised ring kernel
    int leniaRow_ = 0;                               ///< the next row to step
    float leniaRowAcc_ = 0.0f;                       ///< rows owed, carried between blocks
    float leniaTarget_[4] = {};                      ///< the four readings' new means
    float leniaOut_[4] = {};                         ///< the four readings, glided
    Rng leniaRng_;                                   ///< the field's own dice (seeding)
    double lorenz_[3] = { 0.1, 0.0, 20.0 };          ///< the Lorenz state
    double rossler_[3] = { 1.0, 1.0, 1.0 };          ///< the Roessler state
    float kuraPhase_[4] = {};                        ///< the Kuramoto bank's phases, radians
    float beatHz_ = 0.0f;                            ///< the beat rate, followed over two seconds
    float beatPhase_ = 0.0f;                         ///< the Beat oscillator's phase, radians
    float cascade_ = 0.0f;                           ///< the excitation, smoothed over two seconds
    float heard_[4] = {};                            ///< loudness, centroid, flux, roughness, each followed
};

} // namespace additiv
