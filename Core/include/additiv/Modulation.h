/**
 * @file Modulation.h
 * @brief The modulation system (after Noctuary's Modulation.h): sources -- four LFOs, two
 *        envelopes on the factory shapes, four macros, the mod wheel, pressure, the last velocity
 *        and a slow random -- and a matrix of sixteen rows, source -> target x depth.
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
};

/** @brief Whether a parameter can be a target: the continuous and whole-number ones. @param id the parameter @return true when it can */
bool isModTarget(ParamId id);

/** @brief A row in 64 bits, so the audio side reads it atomically. @param r the row @return the bits */
std::uint64_t packModRow(const ModRow& r);
/** @brief A row back from its bits. @param bits from packModRow @return the row */
ModRow unpackModRow(std::uint64_t bits);

/**
 * @brief The rows as text: "lfo1>cutoff:0.35;macro1>motion_depth:-0.5" (empty rows left out).
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

    double phase_[4] = { 0.0, 0.0, 0.0, 0.0 };      ///< each LFO's phase, 0 .. 1
    Drifter random_[4];                              ///< each LFO's random curve (Random shape)
    float stepFrom_[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; ///< each LFO's Steps: the value it glides from
    float stepTo_[4] = { 0.0f, 0.0f, 0.0f, 0.0f };   ///< and to
    Drifter slow_;                                   ///< the Random source
    double envTime_ = 0.0;                           ///< seconds since the phrase began
    Rng rng_;                                        ///< the dice
    float values_[kNumModSources] = {};              ///< every source's value
};

} // namespace additiv
