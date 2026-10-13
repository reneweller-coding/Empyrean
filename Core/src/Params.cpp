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
#include "additiv/Shapes.h"

#include <algorithm>
#include <cmath>
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
const char* const kLifeModeNames[kNumLifeModes] = { "Off", "Lenia", "Turing", "Kaleidoscope", "Tunnel", "Spiral", "Flow", "Feedback",
                                                     "Rain", "Starfield", "Moire", "Fluid", "Islands" };
const char* const kSyncModeNames[kNumSyncModes] = { "Global", "Chimera", "Swarm" };
const char* const kMpeTargetNames[kNumMpeTargets] = { "Off", "Level", "Filter", "Brightness", "Width", "Nearer", "Shimmer", "Motion" };
const char* const kFieldModeNames[kNumFieldModes] = { "Off", "Scale", "Swarm", "Scatter", "Flock" };
const char* const kFarModeNames[kNumFarModes] = { "Classic", "Scattering", "Colourless", "Rotating" };
const char* const kLfoShapeNames[kNumLfoShapes] = { "Sine", "Triangle", "Ramp Up", "Ramp Down", "Square", "Random", "Steps" };
const char* const kEnvShapeNames[kFactoryShapeCount] = { "Swell", "Pulse", "Bloom", "Flicker", "Wobble", "Rise In", "Fall In", "Saw Up" };

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
    C(ParamId::Landscape,   "landscape",    "Landscape",     "Motion",    kLandscapeChoiceNames, kNumLandscapeChoices, 0),

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

    C(ParamId::Lfo1Shape,   "lfo1_shape",   "LFO 1",         "Modulation", kLfoShapeNames, kNumLfoShapes, 0),
    F(ParamId::Lfo1Rate,    "lfo1_rate",    "LFO 1 Rate",    "Modulation", 0.001f, 20.f, 0.05f, 0.2f, "Hz"),
    C(ParamId::Lfo2Shape,   "lfo2_shape",   "LFO 2",         "Modulation", kLfoShapeNames, kNumLfoShapes, 1),
    F(ParamId::Lfo2Rate,    "lfo2_rate",    "LFO 2 Rate",    "Modulation", 0.001f, 20.f, 0.0809f, 0.2f, "Hz"),
    C(ParamId::Lfo3Shape,   "lfo3_shape",   "LFO 3",         "Modulation", kLfoShapeNames, kNumLfoShapes, 5),
    F(ParamId::Lfo3Rate,    "lfo3_rate",    "LFO 3 Rate",    "Modulation", 0.001f, 20.f, 0.1309f, 0.2f, "Hz"),
    C(ParamId::Lfo4Shape,   "lfo4_shape",   "LFO 4",         "Modulation", kLfoShapeNames, kNumLfoShapes, 6),
    F(ParamId::Lfo4Rate,    "lfo4_rate",    "LFO 4 Rate",    "Modulation", 0.001f, 20.f, 0.2118f, 0.2f, "Hz"),
    C(ParamId::Env1Shape,   "env1_shape",   "Env 1",         "Modulation", kEnvShapeNames, kFactoryShapeCount, 2),
    F(ParamId::Env1Rate,    "env1_rate",    "Env 1 Rate",    "Modulation", 0.02f, 8.f,  0.25f, 0.4f, "x"),
    C(ParamId::Env2Shape,   "env2_shape",   "Env 2",         "Modulation", kEnvShapeNames, kFactoryShapeCount, 0),
    F(ParamId::Env2Rate,    "env2_rate",    "Env 2 Rate",    "Modulation", 0.02f, 8.f,  0.1f,  0.4f, "x"),
    F(ParamId::Macro1,      "macro1",       "Macro 1",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),
    F(ParamId::Macro2,      "macro2",       "Macro 2",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),
    F(ParamId::Macro3,      "macro3",       "Macro 3",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),
    F(ParamId::Macro4,      "macro4",       "Macro 4",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),

    F(ParamId::DelayMix,       "delay_mix",       "Echo",         "Echo",  0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::DelayTimeL,     "delay_time_l",    "Time L",       "Echo",  0.02f, 4.f,   0.75f, 0.5f, "s"),
    F(ParamId::DelayTimeR,     "delay_time_r",    "Time R",       "Echo",  0.02f, 4.f,   1.13f, 0.5f, "s"),
    F(ParamId::DelayFeedback,  "delay_feedback",  "Feedback",     "Echo",  0.f,   0.95f, 0.45f, 1.f,  ""),
    F(ParamId::DelayCross,     "delay_cross",     "Cross",        "Echo",  0.f,   1.f,   0.35f, 1.f,  ""),
    F(ParamId::DelayDamp,      "delay_damp",      "Damping",      "Echo",  0.f,   1.f,   0.45f, 1.f,  ""),
    F(ParamId::DelayAbsorb,    "delay_absorb",    "Absorb",       "Echo",  0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamId::DelayDuck,      "delay_duck",      "Duck",         "Echo",  0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamId::DelayToFar,     "delay_to_far",    "To Far",       "Echo",  0.f,   1.f,   0.4f,  1.f,  ""),
    F(ParamId::Delay2Mix,      "delay2_mix",      "Echo 2",       "Echo",  0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::Delay2TimeL,    "delay2_time_l",   "Time L 2",     "Echo",  0.05f, 4.f,   2.4f,  0.5f, "s"),
    F(ParamId::Delay2TimeR,    "delay2_time_r",   "Time R 2",     "Echo",  0.05f, 4.f,   3.1f,  0.5f, "s"),
    F(ParamId::Delay2Feedback, "delay2_feedback", "Feedback 2",   "Echo",  0.f,   0.95f, 0.55f, 1.f,  ""),
    F(ParamId::Delay2Damp,     "delay2_damp",     "Damping 2",    "Echo",  0.f,   1.f,   0.6f,  1.f,  ""),
    F(ParamId::Delay2ToFar,    "delay2_to_far",   "To Far 2",     "Echo",  0.f,   1.f,   0.6f,  1.f,  ""),

    F(ParamId::EarlyLevel,     "early_level",     "Early",        "Room",  0.f,   1.f,   0.2f,  1.f,  ""),
    F(ParamId::EarlySize,      "early_size",      "Room Size",    "Room",  2.f,   40.f,  14.f,  0.5f, "m"),
    F(ParamId::EarlyAbsorb,    "early_absorb",    "Absorb",       "Room",  0.f,   1.f,   0.4f,  1.f,  ""),
    F(ParamId::NearMix,        "near_mix",        "Near Reverb",  "Room",  0.f,   1.f,   0.1f,  1.f,  ""),
    F(ParamId::NearDecay,      "near_decay",      "Near Decay",   "Room",  0.2f,  8.f,   1.8f,  0.5f, "s"),
    F(ParamId::NearDamp,       "near_damp",       "Near Damping", "Room",  0.f,   1.f,   0.5f,  1.f,  ""),

    F(ParamId::FarDiffuse,     "far_diffuse",     "Diffusion",    "Space", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::FarPreDelay,    "far_predelay",    "Pre-Delay",    "Space", 0.f,   500.f, 20.f,  0.5f, "ms"),
    F(ParamId::FarUnmask,      "far_unmask",      "Unmask",       "Space", 0.f,   1.f,   0.3f,  1.f,  ""),
    C(ParamId::FarMode,        "far_mode",        "Far Mode",     "Space", kFarModeNames, kNumFarModes, 2),
    B(ParamId::FarFreeze,      "far_freeze",      "Far Freeze",   "Space", false),

    F(ParamId::ModalLevel,     "modal_level",     "Modal Hall",   "Modal", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::ModalDecay,     "modal_decay",     "Decay",        "Modal", 0.5f,  60.f,  8.f,   0.4f, "s"),
    F(ParamId::ModalDamp,      "modal_damp",      "Damping",      "Modal", 0.f,   1.f,   0.35f, 1.f,  ""),
    F(ParamId::ModalTune,      "modal_tune",      "Tuned",        "Modal", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::ModalSpread,    "modal_spread",    "Spread",       "Modal", 0.f,   50.f,  6.f,   0.5f, "ct"),
    F(ParamId::ModalDrift,     "modal_drift",     "Drift",        "Modal", 0.f,   30.f,  4.f,   0.5f, "ct"),
    F(ParamId::ModalWidth,     "modal_width",     "Width",        "Modal", 0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamId::ModalLow,       "modal_low",       "Lowest",       "Modal", 30.f,  400.f, 80.f,  0.5f, "Hz"),
    B(ParamId::ModalFreeze,    "modal_freeze",    "Freeze",       "Modal", false),
    F(ParamId::ModalBreath,    "modal_breath",    "Breath",       "Modal", 0.f,   1.f,   0.f,   1.f,  ""),

    C(ParamId::FieldMode,      "field_mode",      "Field",        "Field", kFieldModeNames, kNumFieldModes, 0),
    F(ParamId::FieldStrength,  "field_strength",  "Pull",         "Field", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::FieldRadius,    "field_radius",    "Radius",       "Field", 5.f,   600.f, 60.f,  0.5f, "ct"),
    F(ParamId::FieldSpring,    "field_spring",    "Spring",       "Field", 0.f,   1.f,   0.2f,  1.f,  ""),
    F(ParamId::FieldRate,      "field_rate",      "Wander",       "Field", 0.005f, 1.f,  0.05f, 0.4f, ""),
    I(ParamId::FieldCount,     "field_count",     "Attractors",   "Field", 1.f,   8.f,   4.f),
    F(ParamId::Dispersion,     "dispersion",      "Dispersion",   "Motion", -1.5f, 1.5f, 0.f,   1.f,  "s"),

    B(ParamId::Binaural,       "binaural",        "Binaural",     "Binaural", false),
    F(ParamId::BinauralSpan,   "binaural_span",   "Surround",     "Binaural", 0.f,  1.f,   0.5f,  1.f,  ""),
    F(ParamId::BinauralHeight, "binaural_height", "Height",       "Binaural", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamId::BinauralOrbit,  "binaural_orbit",  "Orbit",        "Binaural", -0.5f, 0.5f, 0.f,   1.f,  "Hz"),

    F(ParamId::MemoryLevel,    "memory_level",    "Memory",       "Memory", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::MemoryTime,     "memory_time",     "Memory Time",  "Memory", 10.f,  300.f, 60.f,  0.5f, "s"),
    F(ParamId::MemoryBloom,    "memory_bloom",    "Bloom",        "Memory", 0.2f,  20.f,  4.f,   0.5f, "s"),
    F(ParamId::MemoryBlur,     "memory_blur",     "Blur",         "Memory", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::MemorySink,     "memory_sink",     "Sink",         "Memory", 0.f,   1.f,   0.6f,  1.f,  ""),

    F(ParamId::TuneGravity,    "tune_gravity",    "Self-Tune",    "Tuning", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::TuneReach,      "tune_reach",      "Reach",        "Tuning", 5.f,   60.f,  30.f,  1.f,  "ct"),
    F(ParamId::TuneAnchor,     "tune_anchor",     "Anchor",       "Tuning", 0.f,   1.f,   0.2f,  1.f,  ""),

    F(ParamId::ModalPaint,     "modal_paint",     "Paint",        "Modal", 0.f,   1.f,   0.f,   1.f,  ""),

    F(ParamId::SpecMix,        "spec_mix",        "Spectral Echo", "Spectral Echo", 0.f,  1.f,   0.f,   1.f,  ""),
    F(ParamId::SpecTime,       "spec_time",       "Time",         "Spectral Echo", 0.05f, 4.f,  0.8f,  0.5f, "s"),
    F(ParamId::SpecTilt,       "spec_tilt",       "Tilt",         "Spectral Echo", -1.f, 1.f,   0.5f,  1.f,  ""),
    F(ParamId::SpecScatter,    "spec_scatter",    "Scatter",      "Spectral Echo", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamId::SpecFeedback,   "spec_feedback",   "Feedback",     "Spectral Echo", 0.f,  0.95f, 0.5f,  1.f,  ""),
    F(ParamId::SpecDamp,       "spec_damp",       "Damping",      "Spectral Echo", 0.f,  1.f,   0.4f,  1.f,  ""),
    F(ParamId::SpecToFar,      "spec_to_far",     "To Far",       "Spectral Echo", 0.f,  1.f,   0.4f,  1.f,  ""),

    F(ParamId::Coherence,      "coherence",       "Coherence",    "Modulation", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamId::CoherenceRate,  "coherence_rate",  "Coherence Rate", "Modulation", 0.2f, 5.f, 1.f,   0.5f, "x"),
    F(ParamId::LeniaRate,      "lenia_rate",      "Lenia Rate",   "Modulation", 0.5f, 20.f,  4.f,   0.5f, "Hz"),
    F(ParamId::LeniaGrowth,    "lenia_growth",    "Lenia Growth", "Modulation", 0.08f, 0.3f, 0.15f, 1.f,  ""),
    F(ParamId::ChaosPeriod,    "chaos_period",    "Chaos Period", "Modulation", 10.f, 1800.f, 120.f, 0.4f, "s"),
    F(ParamId::BrainCascade,   "brain_cascade",   "Cascade",      "Conductor", 0.f,  1.f,   0.f,   1.f,  ""),

    C(ParamId::LifeMode,       "life_mode",       "Life",         "Life", kLifeModeNames, kNumLifeModes, 0),
    F(ParamId::LifeDepth,      "life_depth",      "Life Depth",   "Life", 0.f,   1.f,   0.6f,  1.f,  ""),
    F(ParamId::LifeRate,       "life_rate",       "Life Rate",    "Life", 0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamId::LifeCharacter,  "life_character",  "Character",    "Life", 0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamId::LifeZoom,       "life_zoom",       "Zoom",         "Life", 0.25f, 1.f,   1.f,   1.f,  ""),

    F(ParamId::SyncDepth,      "sync_depth",      "Sync",         "Sync", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::SyncCoupling,   "sync_coupling",   "Coupling",     "Sync", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::SyncRate,       "sync_rate",       "Sync Rate",    "Sync", 0.05f, 6.f,   0.5f,  0.4f, "Hz"),
    F(ParamId::SyncSpread,     "sync_spread",     "Spread",       "Sync", 0.f,   1.f,   0.3f,  1.f,  ""),

    F(ParamId::RippleDepth,    "ripple_depth",    "Ripple",       "Ripple", 0.f,  1.f,   0.f,   1.f,  ""),
    F(ParamId::RippleSpeed,    "ripple_speed",    "Ripple Speed", "Ripple", 0.1f, 4.f,   0.8f,  0.5f, "oct/s"),
    F(ParamId::RippleDecay,    "ripple_decay",    "Ripple Decay", "Ripple", 0.5f, 30.f,  6.f,   0.5f, "s"),

    F(ParamId::ReservoirChaos,  "reservoir_chaos",  "Chaos",      "Modulation", 0.5f, 1.5f, 0.95f, 1.f,  ""),
    F(ParamId::ReservoirMemory, "reservoir_memory", "Memory",     "Modulation", 0.3f, 30.f, 4.f,   0.4f, "s"),
    F(ParamId::ReservoirInput,  "reservoir_input",  "Input",      "Modulation", 0.f,  2.f,  0.7f,  1.f,  ""),

    F(ParamId::MeltMix,         "melt_mix",         "Melt",       "Melt", 0.f,   1.f,  0.f,  1.f,  ""),
    F(ParamId::MeltTime,        "melt_time",        "Melt Time",  "Melt", 0.5f,  60.f, 8.f,  0.4f, "s"),
    F(ParamId::MeltSpread,      "melt_spread",      "Spread",     "Melt", 0.f,   2.f,  0.5f, 0.6f, "oct"),
    F(ParamId::MeltToFar,       "melt_to_far",      "To Far",     "Melt", 0.f,   1.f,  0.4f, 1.f,  ""),

    B(ParamId::MorphActive,     "morph_active",     "Morph",      "Morph", false),
    F(ParamId::MorphPos,        "morph_pos",        "A - B",      "Morph", 0.f,   1.f,  0.f,  1.f,  ""),
    F(ParamId::PresetGlide,     "preset_glide",     "Glide",      "Morph", 0.f,   600.f, 0.f, 0.4f, "s"),

    F(ParamId::Stretch,         "stretch",          "Stretch",    "Cloud", 1080.f, 1320.f, 1200.f, 1.f, "ct/oct"),
    F(ParamId::Shift,           "shift",            "Shift",      "Cloud", -200.f, 200.f, 0.f,   1.f,  "Hz"),
    I(ParamId::LifeSides,       "life_sides",       "Sides",      "Life", 2.f,   12.f,  6.f),
    F(ParamId::LifeTurn,        "life_turn",        "Turn",       "Life", -4.f,  4.f,   0.5f,  1.f,  "/min"),
    F(ParamId::CombDepth,       "comb_depth",       "Comb",       "Comb", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamId::CombDensity,     "comb_density",     "Density",    "Comb", 0.25f, 12.f,  2.f,   0.5f, "/oct"),
    F(ParamId::CombSweep,       "comb_sweep",       "Sweep",      "Comb", -2.f,  2.f,   0.1f,  1.f,  "/s"),
    F(ParamId::CombStereo,      "comb_stereo",      "Stereo",     "Comb", 0.f,   1.f,   0.5f,  1.f,  ""),
    B(ParamId::MapActive,       "map_active",       "Map",        "Map", false),
    F(ParamId::MapX,            "map_x",            "Map X",      "Map", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::MapY,            "map_y",            "Map Y",      "Map", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::MapRadius,       "map_radius",       "Radius",     "Map", 0.1f,  4.f,   0.4f,  0.45f, ""),
    F(ParamId::MapGlide,        "map_glide",        "Map Glide",  "Map", 0.1f,  60.f,  3.f,   0.4f, "s"),
    F(ParamId::RouteSpeed,      "route_speed",      "Route Speed", "Map", 0.25f, 4.f,  1.f,   0.5f, "x"),
    C(ParamId::SyncMode,        "sync_mode",        "Sync Mode",  "Sync", kSyncModeNames, kNumSyncModes, 0),
    F(ParamId::SyncSwarm,       "sync_swarm",       "Swarm",      "Sync", -1.f,  1.f,   0.5f,  1.f,  ""),
    F(ParamId::SyncReach,       "sync_reach",       "Reach",      "Sync", 0.f,   100.f, 20.f,  0.6f, "ct"),
    F(ParamId::SyncLag,         "sync_lag",         "Lag",        "Sync", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamId::Turbulence,      "turbulence",       "Turbulence", "Turbulence", 0.f, 1.f, 0.f,  1.f,  ""),
    F(ParamId::TurbulenceTime,  "turbulence_time",  "Settle",     "Turbulence", 0.5f, 60.f, 6.f, 0.4f, "s"),
    B(ParamId::MpeOn,           "mpe",              "MPE",        "MPE", true),
    I(ParamId::MpeBend,         "mpe_bend",         "Bend Range", "MPE", 1.f,   96.f,  48.f, "st"),
    C(ParamId::MpePress,        "mpe_press",        "Press",      "MPE", kMpeTargetNames, kNumMpeTargets, 1),
    F(ParamId::MpePressDepth,   "mpe_press_depth",  "Press Depth", "MPE", 0.f,  1.f,   0.5f,  1.f,  ""),
    C(ParamId::MpeSlide,        "mpe_slide",        "Slide",      "MPE", kMpeTargetNames, kNumMpeTargets, 3),
    F(ParamId::MpeSlideDepth,   "mpe_slide_depth",  "Slide Depth", "MPE", 0.f,  1.f,   0.5f,  1.f,  ""),
} };

} // namespace

const std::array<ParamDesc, kNumParams>& paramTable() { return kTable; }

float toNorm(const ParamDesc& d, float v)
{
    if (d.max <= d.min) return 0.0f;
    const float x = std::clamp((v - d.min) / (d.max - d.min), 0.0f, 1.0f);
    return d.skew == 1.0f ? x : std::pow(x, d.skew);
}

float fromNorm(const ParamDesc& d, float n)
{
    const float x = std::clamp(n, 0.0f, 1.0f);
    return d.min + (d.max - d.min) * (d.skew == 1.0f ? x : std::pow(x, 1.0f / d.skew));
}

const ParamDesc* findParam(const char* key)
{
    if (!key) return nullptr;
    for (const ParamDesc& d : kTable)
        if (std::strcmp(d.key, key) == 0) return &d;
    return nullptr;
}

} // namespace additiv
