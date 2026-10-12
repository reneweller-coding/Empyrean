/**
 * @file Params.h
 * @brief The one parameter table: every parameter with its key, name, section, range, default, skew and unit.
 *
 * The engine, the plugin's parameter layout, the generic editor, presets and the renderer all
 * iterate this table (after Noctuary's Params.h: adding a parameter is one row here and one line
 * where the engine reads it). Also the choice names that have no home elsewhere: the scales,
 * the root names, the conductor's modes and the motions.
 */
#pragma once

#include <array>
#include <cstddef>

namespace additiv {

/** @brief Every parameter, in table order. */
enum class ParamId : int {
    // Master
    MasterGain,     ///< output level, dB
    // Cloud: the voice's spectrum and its own life
    Partials,       ///< partials per voice
    Tilt,           ///< spectral tilt: harmonic h at h^-tilt
    CloudWidth,     ///< the cloud around harmonic 1, cents
    WidthSlope,     ///< how the cloud widens with the harmonic number
    Drift,          ///< each partial's own pitch wander, cents
    DriftRate,      ///< its speed, Hz
    Shimmer,        ///< each partial's own amplitude wander, 0 .. 1
    ShimmerRate,    ///< its speed, Hz
    Stereo,         ///< how far the partials scatter across the field
    // Motion: the envelope lanes, as a scene
    Motion,         ///< which motion the partials' envelopes make (kMotionNames)
    MotionRate,     ///< how fast the motion runs, times its own speed
    MotionDepth,    ///< how deep it reaches, times its own depth
    // Envelope of every voice
    Attack,         ///< seconds
    Decay,          ///< seconds
    Sustain,        ///< 0 .. 1
    Release,        ///< seconds
    // Tuning
    Scale,          ///< kScaleNames
    RootNote,       ///< the scale's root, a pitch class (kRootNames)
    KeyMap,         ///< Snap to 12 keys or consecutive degrees
    RefPitch,       ///< A4 in Hz
    Purity,         ///< 0 = 12-TET, 1 = the scale; the clouds narrow with it so pure intervals can lock
    // Conductor
    BrainOn,        ///< the conductor plays
    AutoMode,       ///< Free or Chords (Autoplay), kBrainModeNames
    BrainRate,      ///< Free: mean seconds between events
    AutoRate,       ///< Chords: seconds between exchanges ("Every")
    BrainDensity,   ///< notes sounding at once
    BrainHoldMin,   ///< Free: shortest hold, seconds
    BrainHoldMax,   ///< Free: longest hold, seconds
    BrainLow,       ///< lowest note the conductor chooses
    BrainHigh,      ///< highest note
    BrainConsonance,///< 0 anything goes .. 1 strictly consonant
    BrainWander,    ///< how often the root moves (Free)
    AutoLead,       ///< Chords: how far the exchanged voice may move, semitones
    AutoTension,    ///< Chords: 0 = the new note must fit the chord, 1 = anything goes
    AutoRootMove,   ///< Chords: how often an exchange moves the root too
    BrainTimbre,    ///< how much consonance is judged from the cloud's own spectrum (Sethares)
    // Space
    Depth,          ///< how deep the conductor's notes sit (near/far planes)
    KeysDepth,      ///< where played notes sit, 0 at the ear .. 1 far
    FarLevel,       ///< the far plane's reverb level
    FarSize,        ///< the far reverb's size
    FarDecay,       ///< its decay, seconds
    FarDamp,        ///< its damping
    BassMono,       ///< below this the mix is mono, Hz
    StereoWidth,    ///< the master's width
    Count           ///< number of parameters
};

constexpr int kNumParams = static_cast<int>(ParamId::Count);   ///< rows of the table

/** @brief How a parameter's value is read. */
enum class ParamKind {
    Float,   ///< continuous, with a skew
    Int,     ///< whole numbers, linear
    Bool,    ///< 0 or 1
    Choice   ///< an index into `choices`
};

/** @brief One row of the parameter table. */
struct ParamDesc {
    ParamId     id;          ///< the parameter; rows are ordered by it
    const char* key;         ///< stable identifier (automation, presets)
    const char* name;        ///< display name
    const char* section;     ///< grouping for editors
    ParamKind   kind;        ///< how the value is read
    float       min;         ///< the lowest value, in the parameter's unit
    float       max;         ///< the highest value
    float       def;         ///< the default
    float       skew;        ///< 1 = linear, < 1 = more resolution at the low end
    const char* unit;        ///< "Hz", "s", "dB", "ct", "st" or ""
    const char* const* choices;   ///< ParamKind::Choice only
    int         numChoices;  ///< entries of `choices`
};

/** @brief The table, ordered by ParamId. @return kNumParams rows */
const std::array<ParamDesc, kNumParams>& paramTable();

/** @brief The row of one parameter. @param id the parameter @return its description */
inline const ParamDesc& paramDesc(ParamId id) { return paramTable()[static_cast<size_t>(id)]; }

/** @brief Looks a parameter up by its key. @param key the stable identifier @return the row, or nullptr */
const ParamDesc* findParam(const char* key);

constexpr int kNumScaleChoices = 13;   ///< entries of kScaleNames: eleven tables, the user slot and the timbre scale
constexpr int kUserScaleIndex = 11;    ///< the Scala slot, the first choice that is not a table (Tuning.cpp checks it)
constexpr int kTimbreScaleIndex = 12;  ///< the scale computed from the spectrum (Sethares)
extern const char* const kScaleNames[kNumScaleChoices];   ///< "12-TET", "JI Major (Ptolemy)", ... (as in Noctuary)
extern const char* const kRootNames[12];                  ///< "C" .. "B"
extern const char* const kKeyMapNames[2];                 ///< "Snap to 12 keys", "Consecutive degrees"

constexpr int kNumMotions = 7;                       ///< entries of kMotionNames
extern const char* const kMotionNames[kNumMotions];  ///< None, Random, Wave, Pendulum, Breathe, Glissando, Aurora

} // namespace additiv
