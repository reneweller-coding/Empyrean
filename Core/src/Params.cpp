/**
 * @file Params.cpp
 * @brief The parameter table and the choice names that have no home elsewhere.
 *
 * The defaults follow Noctuary where a parameter came from there (envelope, tuning, conductor,
 * far reverb, master), with two deliberate differences: the conductor starts in Chords (the user's
 * Autoplay, 12.10.2026) rather than Free, and the motion starts at Breathe so the cloud moves from
 * the first note.
 */
#include "additiv/Params.h"

#include "additiv/Cloud.h"
#include "additiv/ClusterBrain.h"
#include "additiv/Landscape.h"

#include <algorithm>
#include <cstring>

namespace additiv {

const char* const kScaleNames[kNumScaleChoices] = {
    "12-TET",
    "JI Major (Ptolemy)",
    "JI Minor",
    "JI 7-limit",
    "Pythagorean",
    "JI Pentatonic",
    "Harmonic 8-16",
    "Subharmonic 16-8",
    "Slendro (JI)",
    "Bohlen-Pierce (JI)",
    "Otonality 1-11",
    "User (Scala)",
    "Timbre (Sethares)",
};

const char* const kRootNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
const char* const kKeyMapNames[2] = { "Snap to 12 keys", "Consecutive degrees" };
const char* const kBrainModeNames[kNumBrainModes] = { "Free", "Chords" };
const char* const kMotionNames[kNumMotions] = { "None", "Random", "Wave", "Pendulum", "Breathe", "Glissando", "Aurora" };
const char* const kFilterModelNames[kNumFilterModels] = { "Off", "LP 6", "LP 12", "LP 24", "HP 12", "BP 12", "Notch", "Peak", "Formant" };

namespace {

using K = ParamKind;   ///< short for the rows below

/**
 * @brief A continuous parameter's row.
 * @param id    the parameter
 * @param key   the stable identifier
 * @param name  the display name
 * @param sec   the section
 * @param mn    lowest value
 * @param mx    highest value
 * @param def   default
 * @param skew  1 linear, < 1 more resolution low
 * @param unit  unit shown after the value
 * @return      the row
 */
constexpr ParamDesc F(ParamId id, const char* key, const char* name, const char* sec,
                      float mn, float mx, float def, float skew, const char* unit)
{ return { id, key, name, sec, K::Float, mn, mx, def, skew, unit, nullptr, 0 }; }

/**
 * @brief An integer parameter's row.
 * @param id    the parameter
 * @param key   the stable identifier
 * @param name  the display name
 * @param sec   the section
 * @param mn    lowest value
 * @param mx    highest value
 * @param def   default
 * @param unit  unit shown after the value
 * @return      the row
 */
constexpr ParamDesc I(ParamId id, const char* key, const char* name, const char* sec,
                      float mn, float mx, float def, const char* unit = "")
{ return { id, key, name, sec, K::Int, mn, mx, def, 1.0f, unit, nullptr, 0 }; }

/**
 * @brief A switch's row.
 * @param id    the parameter
 * @param key   the stable identifier
 * @param name  the display name
 * @param sec   the section
 * @param def   on or off
 * @return      the row
 */
constexpr ParamDesc B(ParamId id, const char* key, const char* name, const char* sec, bool def)
{ return { id, key, name, sec, K::Bool, 0.0f, 1.0f, def ? 1.0f : 0.0f, 1.0f, "", nullptr, 0 }; }

/**
 * @brief A choice's row.
 * @param id       the parameter
 * @param key      the stable identifier
 * @param name     the display name
 * @param sec      the section
 * @param choices  the names
 * @param n        how many
 * @param def      the default index
 * @return         the row
 */
constexpr ParamDesc C(ParamId id, const char* key, const char* name, const char* sec,
                      const char* const* choices, int n, int def)
{ return { id, key, name, sec, K::Choice, 0.0f, static_cast<float>(n - 1), static_cast<float>(def), 1.0f, "", choices, n }; }

/** @brief The rows, in ParamId order (checked by the self test). */
const std::array<ParamDesc, kNumParams> kTable = { {
    F(ParamId::MasterGain,  "master_gain",  "Master",        "Master",    -40.f, 12.f,  0.f,   1.f,  "dB"),

    I(ParamId::Partials,    "partials",     "Partials",      "Cloud",     200.f, 20000.f, 20000.f),
    F(ParamId::Tilt,        "tilt",         "Spectral Tilt", "Cloud",     0.3f,  3.f,   1.2f,  1.f,  ""),
    F(ParamId::Brightness,  "brightness",   "Brightness",    "Cloud",     0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamId::OddEven,     "odd_even",     "Odd / Even",    "Cloud",     -1.f,  1.f,   0.f,   1.f,  ""),
    F(ParamId::Inharmonic,  "inharmonic",   "Inharmonic",    "Cloud",     0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::CloudWidth,  "cloud_width",  "Width",         "Cloud",     0.f,   100.f, 25.f,  0.5f, "ct"),
    F(ParamId::WidthSlope,  "width_slope",  "Width Slope",   "Cloud",     0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamId::Drift,       "drift",        "Drift",         "Cloud",     0.f,   30.f,  3.f,   0.5f, "ct"),
    F(ParamId::DriftRate,   "drift_rate",   "Drift Rate",    "Cloud",     0.01f, 2.f,   0.1f,  0.3f, "Hz"),
    F(ParamId::Shimmer,     "shimmer",      "Shimmer",       "Cloud",     0.f,   1.f,   0.4f,  1.f,  ""),
    F(ParamId::ShimmerRate, "shimmer_rate", "Shimmer Rate",  "Cloud",     0.01f, 2.f,   0.15f, 0.3f, "Hz"),
    F(ParamId::Stereo,      "stereo",       "Stereo",        "Cloud",     0.f,   1.f,   1.f,   1.f,  ""),

    F(ParamId::Air,         "air",          "Air",           "Air",       0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::AirCenter,   "air_center",   "Centre",        "Air",       1.f,   12.f,  3.f,   0.5f, "x"),
    F(ParamId::AirWidth,    "air_width",    "Width",         "Air",       50.f,  2400.f, 600.f, 0.5f, "ct"),
    F(ParamId::AirJitter,   "air_jitter",   "Flicker",       "Air",       0.5f,  40.f,  8.f,   0.5f, "Hz"),

    C(ParamId::FilterModel, "filter_model", "Model",         "Filter",    kFilterModelNames, kNumFilterModels, 0),
    F(ParamId::Cutoff,      "cutoff",       "Cutoff",        "Filter",    40.f,  18000.f, 2500.f, 0.25f, "Hz"),
    F(ParamId::Resonance,   "resonance",    "Resonance",     "Filter",    0.f,   1.f,   0.15f, 1.f,  ""),
    F(ParamId::FilterEnv,   "filter_env",   "Env Amount",    "Filter",    -4.f,  4.f,   0.f,   1.f,  "oct"),
    F(ParamId::KeyTrack,    "keytrack",     "Key Track",     "Filter",    0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::FilterDrift, "filter_drift", "Drift",         "Filter",    0.f,   1.f,   0.3f,  1.f,  ""),

    C(ParamId::Motion,      "motion",       "Motion",        "Motion",    kMotionNames, kNumMotions, 4),
    F(ParamId::MotionRate,  "motion_rate",  "Rate",          "Motion",    0.1f,  4.f,   1.f,   0.5f, ""),
    F(ParamId::MotionDepth, "motion_depth", "Depth",         "Motion",    0.f,   1.f,   1.f,   1.f,  ""),
    C(ParamId::Landscape,   "landscape",    "Landscape",     "Motion",    kLandscapeNames, kNumFactoryLandscapes, 0),

    F(ParamId::Attack,      "attack",       "Attack",        "Envelope",  0.01f, 60.f,  6.f,   0.3f, "s"),
    F(ParamId::Decay,       "decay",        "Decay",         "Envelope",  0.01f, 60.f,  4.f,   0.3f, "s"),
    F(ParamId::Sustain,     "sustain",      "Sustain",       "Envelope",  0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamId::Release,     "release",      "Release",       "Envelope",  0.05f, 120.f, 12.f,  0.3f, "s"),

    C(ParamId::Scale,       "scale",        "Scale",         "Tuning",    kScaleNames, kNumScaleChoices, 3),
    C(ParamId::RootNote,    "root",         "Root",          "Tuning",    kRootNames, 12, 2),
    C(ParamId::KeyMap,      "keymap",       "Keys",          "Tuning",    kKeyMapNames, 2, 0),
    F(ParamId::RefPitch,    "ref_pitch",    "A4",            "Tuning",    415.f, 466.f, 440.f, 1.f,  "Hz"),
    F(ParamId::Purity,      "purity",       "Purity",        "Tuning",    0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamId::Tide,        "tide",         "Tide",          "Tuning",    0.f,   30.f,  0.f,   0.6f, "ct"),
    F(ParamId::TidePeriod,  "tide_period",  "Tide Period",   "Tuning",    1.f,   60.f,  12.f,  0.5f, "min"),

    B(ParamId::BrainOn,         "brain_on",         "Active",     "Conductor", true),
    C(ParamId::AutoMode,        "auto_mode",        "Autoplay",   "Conductor", kBrainModeNames, kNumBrainModes, 1),
    F(ParamId::BrainRate,       "brain_rate",       "Event Rate", "Conductor", 2.f,  300.f, 25.f,  0.4f, "s"),
    F(ParamId::AutoRate,        "auto_rate",        "Every",      "Conductor", 2.f,  900.f, 45.f,  0.4f, "s"),
    I(ParamId::BrainDensity,    "brain_density",    "Density",    "Conductor", 1.f,  10.f,  5.f),
    F(ParamId::BrainHoldMin,    "brain_hold_min",   "Hold Min",   "Conductor", 5.f,  600.f, 30.f,  0.4f, "s"),
    F(ParamId::BrainHoldMax,    "brain_hold_max",   "Hold Max",   "Conductor", 5.f,  600.f, 120.f, 0.4f, "s"),
    I(ParamId::BrainLow,        "brain_low",        "Lowest",     "Conductor", 24.f, 96.f,  36.f),
    I(ParamId::BrainHigh,       "brain_high",       "Highest",    "Conductor", 24.f, 108.f, 79.f),
    F(ParamId::BrainConsonance, "brain_consonance", "Consonance", "Conductor", 0.f,  1.f,   0.7f,  1.f,  ""),
    F(ParamId::BrainWander,     "brain_wander",     "Wander",     "Conductor", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamId::AutoLead,        "auto_lead",        "Voice Lead", "Conductor", 1.f,  24.f,  7.f,   0.6f, "st"),
    F(ParamId::AutoTension,     "auto_tension",     "Tension",    "Conductor", 0.f,  1.f,   0.f,   1.f,  ""),
    F(ParamId::AutoRootMove,    "auto_root_move",   "Root Move",  "Conductor", 0.f,  1.f,   0.2f,  1.f,  ""),
    F(ParamId::BrainTimbre,     "brain_timbre",     "Timbre",     "Conductor", 0.f,  1.f,   0.f,   1.f,  ""),
    B(ParamId::AutoStep,        "auto_step",        "Step",       "Conductor", false),
    B(ParamId::Brain2On,        "brain2_on",        "Brain 2",    "Conductor", false),
    I(ParamId::Brain2Interval,  "brain2_interval",  "Interval",   "Conductor", -24.f, 24.f, -12.f, "st"),
    I(ParamId::Brain2Density,   "brain2_density",   "Density 2",  "Conductor", 1.f,  6.f,   2.f),
    F(ParamId::Brain2Rate,      "brain2_rate",      "Rate 2",     "Conductor", 2.f,  300.f, 60.f,  0.4f, "s"),
    F(ParamId::Brain2Depth,     "brain2_depth",     "Depth 2",    "Conductor", 0.f,  1.f,   0.9f,  1.f,  ""),

    F(ParamId::Depth,       "depth",        "Depth",         "Space",     0.f,   1.f,   0.6f,  1.f,  ""),
    F(ParamId::KeysDepth,   "keys_depth",   "Keys Depth",    "Space",     0.f,   1.f,   0.2f,  1.f,  ""),
    F(ParamId::FarLevel,    "far_level",    "Far Level",     "Space",     0.f,   1.f,   0.8f,  1.f,  ""),
    F(ParamId::FarSize,     "far_size",     "Far Size",      "Space",     0.5f,  3.f,   2.f,   1.f,  ""),
    F(ParamId::FarDecay,    "far_decay",    "Far Decay",     "Space",     1.f,   90.f,  25.f,  0.3f, "s"),
    F(ParamId::FarDamp,     "far_damp",     "Far Damping",   "Space",     0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::Presence,    "presence",     "Presence",      "Space",     0.f,   6.f,   0.f,   1.f,  "dB"),
    F(ParamId::PadLowCut,   "pad_low_cut",  "Pad Low Cut",   "Space",     0.f,   300.f, 0.f,   0.6f, "Hz"),
    F(ParamId::Breath,      "breath",       "Breath",        "Space",     0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::BreathRate,  "breath_rate",  "Breath Rate",   "Space",     0.005f, 0.2f, 0.03f, 0.5f, "Hz"),
    F(ParamId::Doppler,     "doppler",      "Doppler",       "Space",     0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::BassMono,    "bass_mono",    "Bass Mono",     "Master",    40.f,  300.f, 150.f, 0.5f, "Hz"),
    F(ParamId::StereoWidth, "stereo_width", "Width",         "Master",    0.f,   2.f,   1.2f,  1.f,  ""),
} };

} // namespace

const std::array<ParamDesc, kNumParams>& paramTable() { return kTable; }

const ParamDesc* findParam(const char* key)
{
    if (!key) return nullptr;
    for (const ParamDesc& d : kTable)
        if (std::strcmp(d.key, key) == 0) return &d;
    return nullptr;
}

} // namespace additiv
