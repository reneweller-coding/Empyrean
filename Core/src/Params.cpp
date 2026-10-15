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

const char* const ScaleNames[NumScaleChoices] = {
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

const char* const RootNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
const char* const KeyMapNames[2] = { "Snap to 12 keys", "Consecutive degrees" };
const char* const BrainModeNames[NumBrainModes] = { "Free", "Chords" };
const char* const MotionNames[NumMotions] = { "None", "Random", "Wave", "Pendulum", "Breathe", "Glissando", "Aurora" };
const char* const FilterModelNames[NumFilterModels] = { "Off", "LP 6", "LP 12", "LP 24", "HP 12", "BP 12", "Notch", "Peak", "Formant" };
const char* const LifeModeNames[NumLifeModes] = { "Off", "Lenia", "Turing", "Kaleidoscope", "Tunnel", "Spiral", "Flow", "Feedback",
                                                     "Rain", "Starfield", "Moire", "Fluid", "Islands" };
const char* const SyncModeNames[NumSyncModes] = { "Global", "Chimera", "Swarm" };
const char* const MpeTargetNames[NumMpeTargets] = { "Off", "Level", "Filter", "Brightness", "Width", "Nearer", "Shimmer", "Motion" };
const char* const FieldModeNames[NumFieldModes] = { "Off", "Scale", "Swarm", "Scatter", "Flock", "Orbit" };
const char* const FarModeNames[NumFarModes] = { "Classic", "Scattering", "Colourless", "Rotating" };
const char* const LfoShapeNames[NumLfoShapes] = { "Sine", "Triangle", "Ramp Up", "Ramp Down", "Square", "Random", "Steps" };
const char* const EnvShapeNames[FACTORY_SHAPE_COUNT] = { "Swell", "Pulse", "Bloom", "Flicker", "Wobble", "Rise In", "Fall In", "Saw Up" };

namespace {

using KT = ParamKindE;   ///< short for the rows below

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
constexpr ParamDesc F(ParamIdE id, const char* key, const char* name, const char* sec,
                      float mn, float mx, float def, float skew, const char* unit)
{ return { id, key, name, sec, KT::FLOAT, mn, mx, def, skew, unit, nullptr, 0 }; }

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
constexpr ParamDesc I(ParamIdE id, const char* key, const char* name, const char* sec,
                      float mn, float mx, float def, const char* unit = "")
{ return { id, key, name, sec, KT::INT, mn, mx, def, 1.0f, unit, nullptr, 0 }; }

/**
 * @brief A switch's row.
 * @param id    the parameter
 * @param key   the stable identifier
 * @param name  the display name
 * @param sec   the section
 * @param def   on or off
 * @return      the row
 */
constexpr ParamDesc B(ParamIdE id, const char* key, const char* name, const char* sec, bool def)
{ return { id, key, name, sec, KT::BOOL, 0.0f, 1.0f, def ? 1.0f : 0.0f, 1.0f, "", nullptr, 0 }; }

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
constexpr ParamDesc C(ParamIdE id, const char* key, const char* name, const char* sec,
                      const char* const* choices, int n, int def)
{ return { id, key, name, sec, KT::CHOICE, 0.0f, static_cast<float>(n - 1), static_cast<float>(def), 1.0f, "", choices, n }; }

/** @brief The rows, in ParamId order (checked by the self test). */
const std::array<ParamDesc, NumParams> Table = { {
    F(ParamIdE::MASTER_GAIN,  "master_gain",  "Master",        "Master",    -40.f, 12.f,  0.f,   1.f,  "dB"),

    I(ParamIdE::PARTIALS,    "partials",     "Partials",      "Cloud",     200.f, 20000.f, 20000.f),
    F(ParamIdE::TILT,        "tilt",         "Spectral Tilt", "Cloud",     0.3f,  3.f,   1.2f,  1.f,  ""),
    F(ParamIdE::BRIGHTNESS,  "brightness",   "Brightness",    "Cloud",     0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamIdE::ODD_EVEN,     "odd_even",     "Odd / Even",    "Cloud",     -1.f,  1.f,   0.f,   1.f,  ""),
    F(ParamIdE::INHARMONIC,  "inharmonic",   "Inharmonic",    "Cloud",     0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::CLOUD_WIDTH,  "cloud_width",  "Width",         "Cloud",     0.f,   100.f, 25.f,  0.5f, "ct"),
    F(ParamIdE::WIDTH_SLOPE,  "width_slope",  "Width Slope",   "Cloud",     0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::DRIFT,       "drift",        "Drift",         "Cloud",     0.f,   30.f,  3.f,   0.5f, "ct"),
    F(ParamIdE::DRIFT_RATE,   "drift_rate",   "Drift Rate",    "Cloud",     0.01f, 2.f,   0.1f,  0.3f, "Hz"),
    F(ParamIdE::SHIMMER,     "shimmer",      "Shimmer",       "Cloud",     0.f,   1.f,   0.4f,  1.f,  ""),
    F(ParamIdE::SHIMMER_RATE, "shimmer_rate", "Shimmer Rate",  "Cloud",     0.01f, 2.f,   0.15f, 0.3f, "Hz"),
    F(ParamIdE::STEREO,      "stereo",       "Stereo",        "Cloud",     0.f,   1.f,   1.f,   1.f,  ""),

    F(ParamIdE::AIR,         "air",          "Air",           "Air",       0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::AIR_CENTER,   "air_center",   "Centre",        "Air",       1.f,   12.f,  3.f,   0.5f, "x"),
    F(ParamIdE::AIR_WIDTH,    "air_width",    "Width",         "Air",       50.f,  2400.f, 600.f, 0.5f, "ct"),
    F(ParamIdE::AIR_JITTER,   "air_jitter",   "Flicker",       "Air",       0.5f,  40.f,  8.f,   0.5f, "Hz"),

    C(ParamIdE::FILTER_MODEL, "filter_model", "Model",         "Filter",    FilterModelNames, NumFilterModels, 0),
    F(ParamIdE::CUTOFF,      "cutoff",       "Cutoff",        "Filter",    40.f,  18000.f, 2500.f, 0.25f, "Hz"),
    F(ParamIdE::RESONANCE,   "resonance",    "Resonance",     "Filter",    0.f,   1.f,   0.15f, 1.f,  ""),
    F(ParamIdE::FILTER_ENV,   "filter_env",   "Env Amount",    "Filter",    -4.f,  4.f,   0.f,   1.f,  "oct"),
    F(ParamIdE::KEY_TRACK,    "keytrack",     "Key Track",     "Filter",    0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::FILTER_DRIFT, "filter_drift", "Drift",         "Filter",    0.f,   1.f,   0.3f,  1.f,  ""),

    C(ParamIdE::MOTION,      "motion",       "Motion",        "Motion",    MotionNames, NumMotions, 4),
    F(ParamIdE::MOTION_RATE,  "motion_rate",  "Rate",          "Motion",    0.1f,  4.f,   1.f,   0.5f, ""),
    F(ParamIdE::MOTION_DEPTH, "motion_depth", "Depth",         "Motion",    0.f,   1.f,   1.f,   1.f,  ""),
    C(ParamIdE::LANDSCAPE,   "landscape",    "Landscape",     "Motion",    LandscapeChoiceNames, NumLandscapeChoices, 0),

    F(ParamIdE::ATTACK,      "attack",       "Attack",        "Envelope",  0.01f, 60.f,  6.f,   0.3f, "s"),
    F(ParamIdE::DECAY,       "decay",        "Decay",         "Envelope",  0.01f, 60.f,  4.f,   0.3f, "s"),
    F(ParamIdE::SUSTAIN,     "sustain",      "Sustain",       "Envelope",  0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamIdE::RELEASE,     "release",      "Release",       "Envelope",  0.05f, 120.f, 12.f,  0.3f, "s"),

    C(ParamIdE::SCALE,       "scale",        "Scale",         "Tuning",    ScaleNames, NumScaleChoices, 3),
    C(ParamIdE::ROOT_NOTE,    "root",         "Root",          "Tuning",    RootNames, 12, 2),
    C(ParamIdE::KEY_MAP,      "keymap",       "Keys",          "Tuning",    KeyMapNames, 2, 0),
    F(ParamIdE::REF_PITCH,    "ref_pitch",    "A4",            "Tuning",    415.f, 466.f, 440.f, 1.f,  "Hz"),
    F(ParamIdE::PURITY,      "purity",       "Purity",        "Tuning",    0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamIdE::TIDE,        "tide",         "Tide",          "Tuning",    0.f,   30.f,  0.f,   0.6f, "ct"),
    F(ParamIdE::TIDE_PERIOD,  "tide_period",  "Tide Period",   "Tuning",    1.f,   60.f,  12.f,  0.5f, "min"),

    B(ParamIdE::BRAIN_ON,         "brain_on",         "Active",     "Conductor", true),
    C(ParamIdE::AUTO_MODE,        "auto_mode",        "Autoplay",   "Conductor", BrainModeNames, NumBrainModes, 1),
    F(ParamIdE::BRAIN_RATE,       "brain_rate",       "Event Rate", "Conductor", 2.f,  300.f, 25.f,  0.4f, "s"),
    F(ParamIdE::AUTO_RATE,        "auto_rate",        "Every",      "Conductor", 2.f,  900.f, 45.f,  0.4f, "s"),
    I(ParamIdE::BRAIN_DENSITY,    "brain_density",    "Density",    "Conductor", 1.f,  10.f,  5.f),
    F(ParamIdE::BRAIN_HOLD_MIN,    "brain_hold_min",   "Hold Min",   "Conductor", 5.f,  600.f, 30.f,  0.4f, "s"),
    F(ParamIdE::BRAIN_HOLD_MAX,    "brain_hold_max",   "Hold Max",   "Conductor", 5.f,  600.f, 120.f, 0.4f, "s"),
    I(ParamIdE::BRAIN_LOW,        "brain_low",        "Lowest",     "Conductor", 24.f, 96.f,  36.f),
    I(ParamIdE::BRAIN_HIGH,       "brain_high",       "Highest",    "Conductor", 24.f, 108.f, 79.f),
    F(ParamIdE::BRAIN_CONSONANCE, "brain_consonance", "Consonance", "Conductor", 0.f,  1.f,   0.7f,  1.f,  ""),
    F(ParamIdE::BRAIN_WANDER,     "brain_wander",     "Wander",     "Conductor", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::AUTO_LEAD,        "auto_lead",        "Voice Lead", "Conductor", 1.f,  24.f,  7.f,   0.6f, "st"),
    F(ParamIdE::AUTO_TENSION,     "auto_tension",     "Tension",    "Conductor", 0.f,  1.f,   0.f,   1.f,  ""),
    F(ParamIdE::AUTO_ROOT_MOVE,    "auto_root_move",   "Root Move",  "Conductor", 0.f,  1.f,   0.2f,  1.f,  ""),
    F(ParamIdE::BRAIN_TIMBRE,     "brain_timbre",     "Timbre",     "Conductor", 0.f,  1.f,   0.f,   1.f,  ""),
    B(ParamIdE::AUTO_STEP,        "auto_step",        "Step",       "Conductor", false),
    B(ParamIdE::BRAIN2_ON,        "brain2_on",        "Brain 2",    "Conductor", false),
    I(ParamIdE::BRAIN2_INTERVAL,  "brain2_interval",  "Interval",   "Conductor", -24.f, 24.f, -12.f, "st"),
    I(ParamIdE::BRAIN2_DENSITY,   "brain2_density",   "Density 2",  "Conductor", 1.f,  6.f,   2.f),
    F(ParamIdE::BRAIN2_RATE,      "brain2_rate",      "Rate 2",     "Conductor", 2.f,  300.f, 60.f,  0.4f, "s"),
    F(ParamIdE::BRAIN2_DEPTH,     "brain2_depth",     "Depth 2",    "Conductor", 0.f,  1.f,   0.9f,  1.f,  ""),

    F(ParamIdE::DEPTH,       "depth",        "Depth",         "Space",     0.f,   1.f,   0.6f,  1.f,  ""),
    F(ParamIdE::KEYS_DEPTH,   "keys_depth",   "Keys Depth",    "Space",     0.f,   1.f,   0.2f,  1.f,  ""),
    F(ParamIdE::FAR_LEVEL,    "far_level",    "Far Level",     "Space",     0.f,   1.f,   0.8f,  1.f,  ""),
    F(ParamIdE::FAR_SIZE,     "far_size",     "Far Size",      "Space",     0.5f,  3.f,   2.f,   1.f,  ""),
    F(ParamIdE::FAR_DECAY,    "far_decay",    "Far Decay",     "Space",     1.f,   90.f,  25.f,  0.3f, "s"),
    F(ParamIdE::FAR_DAMP,     "far_damp",     "Far Damping",   "Space",     0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::PRESENCE,    "presence",     "Presence",      "Space",     0.f,   6.f,   0.f,   1.f,  "dB"),
    F(ParamIdE::PAD_LOW_CUT,   "pad_low_cut",  "Pad Low Cut",   "Space",     0.f,   300.f, 0.f,   0.6f, "Hz"),
    F(ParamIdE::BREATH,      "breath",       "Breath",        "Space",     0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::BREATH_RATE,  "breath_rate",  "Breath Rate",   "Space",     0.005f, 0.2f, 0.03f, 0.5f, "Hz"),
    F(ParamIdE::DOPPLER,     "doppler",      "Doppler",       "Space",     0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::BASS_MONO,    "bass_mono",    "Bass Mono",     "Master",    40.f,  300.f, 150.f, 0.5f, "Hz"),
    F(ParamIdE::STEREO_WIDTH, "stereo_width", "Width",         "Master",    0.f,   2.f,   1.2f,  1.f,  ""),

    C(ParamIdE::LFO1_SHAPE,   "lfo1_shape",   "LFO 1",         "Modulation", LfoShapeNames, NumLfoShapes, 0),
    F(ParamIdE::LFO1_RATE,    "lfo1_rate",    "LFO 1 Rate",    "Modulation", 0.001f, 20.f, 0.05f, 0.2f, "Hz"),
    C(ParamIdE::LFO2_SHAPE,   "lfo2_shape",   "LFO 2",         "Modulation", LfoShapeNames, NumLfoShapes, 1),
    F(ParamIdE::LFO2_RATE,    "lfo2_rate",    "LFO 2 Rate",    "Modulation", 0.001f, 20.f, 0.0809f, 0.2f, "Hz"),
    C(ParamIdE::LFO3_SHAPE,   "lfo3_shape",   "LFO 3",         "Modulation", LfoShapeNames, NumLfoShapes, 5),
    F(ParamIdE::LFO3_RATE,    "lfo3_rate",    "LFO 3 Rate",    "Modulation", 0.001f, 20.f, 0.1309f, 0.2f, "Hz"),
    C(ParamIdE::LFO4_SHAPE,   "lfo4_shape",   "LFO 4",         "Modulation", LfoShapeNames, NumLfoShapes, 6),
    F(ParamIdE::LFO4_RATE,    "lfo4_rate",    "LFO 4 Rate",    "Modulation", 0.001f, 20.f, 0.2118f, 0.2f, "Hz"),
    C(ParamIdE::ENV1_SHAPE,   "env1_shape",   "Env 1",         "Modulation", EnvShapeNames, FACTORY_SHAPE_COUNT, 2),
    F(ParamIdE::ENV1_RATE,    "env1_rate",    "Env 1 Rate",    "Modulation", 0.02f, 8.f,  0.25f, 0.4f, "x"),
    C(ParamIdE::ENV2_SHAPE,   "env2_shape",   "Env 2",         "Modulation", EnvShapeNames, FACTORY_SHAPE_COUNT, 0),
    F(ParamIdE::ENV2_RATE,    "env2_rate",    "Env 2 Rate",    "Modulation", 0.02f, 8.f,  0.1f,  0.4f, "x"),
    F(ParamIdE::MACRO1,      "macro1",       "Macro 1",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),
    F(ParamIdE::MACRO2,      "macro2",       "Macro 2",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),
    F(ParamIdE::MACRO3,      "macro3",       "Macro 3",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),
    F(ParamIdE::MACRO4,      "macro4",       "Macro 4",       "Modulation", 0.f,   1.f,  0.f,   1.f,  ""),

    F(ParamIdE::DELAY_MIX,       "delay_mix",       "Echo",         "Echo",  0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::DELAY_TIME_L,     "delay_time_l",    "Time L",       "Echo",  0.02f, 4.f,   0.75f, 0.5f, "s"),
    F(ParamIdE::DELAY_TIME_R,     "delay_time_r",    "Time R",       "Echo",  0.02f, 4.f,   1.13f, 0.5f, "s"),
    F(ParamIdE::DELAY_FEEDBACK,  "delay_feedback",  "Feedback",     "Echo",  0.f,   0.95f, 0.45f, 1.f,  ""),
    F(ParamIdE::DELAY_CROSS,     "delay_cross",     "Cross",        "Echo",  0.f,   1.f,   0.35f, 1.f,  ""),
    F(ParamIdE::DELAY_DAMP,      "delay_damp",      "Damping",      "Echo",  0.f,   1.f,   0.45f, 1.f,  ""),
    F(ParamIdE::DELAY_ABSORB,    "delay_absorb",    "Absorb",       "Echo",  0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::DELAY_DUCK,      "delay_duck",      "Duck",         "Echo",  0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::DELAY_TO_FAR,     "delay_to_far",    "To Far",       "Echo",  0.f,   1.f,   0.4f,  1.f,  ""),
    F(ParamIdE::DELAY2_MIX,      "delay2_mix",      "Echo 2",       "Echo",  0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::DELAY2_TIME_L,    "delay2_time_l",   "Time L 2",     "Echo",  0.05f, 4.f,   2.4f,  0.5f, "s"),
    F(ParamIdE::DELAY2_TIME_R,    "delay2_time_r",   "Time R 2",     "Echo",  0.05f, 4.f,   3.1f,  0.5f, "s"),
    F(ParamIdE::DELAY2_FEEDBACK, "delay2_feedback", "Feedback 2",   "Echo",  0.f,   0.95f, 0.55f, 1.f,  ""),
    F(ParamIdE::DELAY2_DAMP,     "delay2_damp",     "Damping 2",    "Echo",  0.f,   1.f,   0.6f,  1.f,  ""),
    F(ParamIdE::DELAY2_TO_FAR,    "delay2_to_far",   "To Far 2",     "Echo",  0.f,   1.f,   0.6f,  1.f,  ""),

    F(ParamIdE::EARLY_LEVEL,     "early_level",     "Early",        "Room",  0.f,   1.f,   0.2f,  1.f,  ""),
    F(ParamIdE::EARLY_SIZE,      "early_size",      "Room Size",    "Room",  2.f,   40.f,  14.f,  0.5f, "m"),
    F(ParamIdE::EARLY_ABSORB,    "early_absorb",    "Absorb",       "Room",  0.f,   1.f,   0.4f,  1.f,  ""),
    F(ParamIdE::NEAR_MIX,        "near_mix",        "Near Reverb",  "Room",  0.f,   1.f,   0.1f,  1.f,  ""),
    F(ParamIdE::NEAR_DECAY,      "near_decay",      "Near Decay",   "Room",  0.2f,  8.f,   1.8f,  0.5f, "s"),
    F(ParamIdE::NEAR_DAMP,       "near_damp",       "Near Damping", "Room",  0.f,   1.f,   0.5f,  1.f,  ""),

    F(ParamIdE::FAR_DIFFUSE,     "far_diffuse",     "Diffusion",    "Space", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::FAR_PRE_DELAY,    "far_predelay",    "Pre-Delay",    "Space", 0.f,   500.f, 20.f,  0.5f, "ms"),
    F(ParamIdE::FAR_UNMASK,      "far_unmask",      "Unmask",       "Space", 0.f,   1.f,   0.3f,  1.f,  ""),
    C(ParamIdE::FAR_MODE,        "far_mode",        "Far Mode",     "Space", FarModeNames, NumFarModes, 2),
    B(ParamIdE::FAR_FREEZE,      "far_freeze",      "Far Freeze",   "Space", false),

    F(ParamIdE::MODAL_LEVEL,     "modal_level",     "Modal Hall",   "Modal", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::MODAL_DECAY,     "modal_decay",     "Decay",        "Modal", 0.5f,  60.f,  8.f,   0.4f, "s"),
    F(ParamIdE::MODAL_DAMP,      "modal_damp",      "Damping",      "Modal", 0.f,   1.f,   0.35f, 1.f,  ""),
    F(ParamIdE::MODAL_TUNE,      "modal_tune",      "Tuned",        "Modal", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::MODAL_SPREAD,    "modal_spread",    "Spread",       "Modal", 0.f,   50.f,  6.f,   0.5f, "ct"),
    F(ParamIdE::MODAL_DRIFT,     "modal_drift",     "Drift",        "Modal", 0.f,   30.f,  4.f,   0.5f, "ct"),
    F(ParamIdE::MODAL_WIDTH,     "modal_width",     "Width",        "Modal", 0.f,   1.f,   1.f,   1.f,  ""),
    F(ParamIdE::MODAL_LOW,       "modal_low",       "Lowest",       "Modal", 30.f,  400.f, 80.f,  0.5f, "Hz"),
    B(ParamIdE::MODAL_FREEZE,    "modal_freeze",    "Freeze",       "Modal", false),
    F(ParamIdE::MODAL_BREATH,    "modal_breath",    "Breath",       "Modal", 0.f,   1.f,   0.f,   1.f,  ""),

    C(ParamIdE::FIELD_MODE,      "field_mode",      "Field",        "Field", FieldModeNames, NumFieldModes, 0),
    F(ParamIdE::FIELD_STRENGTH,  "field_strength",  "Pull",         "Field", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::FIELD_RADIUS,    "field_radius",    "Radius",       "Field", 5.f,   600.f, 60.f,  0.5f, "ct"),
    F(ParamIdE::FIELD_SPRING,    "field_spring",    "Spring",       "Field", 0.f,   1.f,   0.2f,  1.f,  ""),
    F(ParamIdE::FIELD_RATE,      "field_rate",      "Wander",       "Field", 0.005f, 1.f,  0.05f, 0.4f, ""),
    I(ParamIdE::FIELD_COUNT,     "field_count",     "Attractors",   "Field", 1.f,   8.f,   4.f),
    F(ParamIdE::DISPERSION,     "dispersion",      "Dispersion",   "Motion", -1.5f, 1.5f, 0.f,   1.f,  "s"),

    B(ParamIdE::BINAURAL,       "binaural",        "Binaural",     "Binaural", false),
    F(ParamIdE::BINAURAL_SPAN,   "binaural_span",   "Surround",     "Binaural", 0.f,  1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::BINAURAL_HEIGHT, "binaural_height", "Height",       "Binaural", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::BINAURAL_ORBIT,  "binaural_orbit",  "Orbit",        "Binaural", -0.5f, 0.5f, 0.f,   1.f,  "Hz"),

    F(ParamIdE::MEMORY_LEVEL,    "memory_level",    "Memory",       "Memory", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::MEMORY_TIME,     "memory_time",     "Memory Time",  "Memory", 10.f,  300.f, 60.f,  0.5f, "s"),
    F(ParamIdE::MEMORY_BLOOM,    "memory_bloom",    "Bloom",        "Memory", 0.2f,  20.f,  4.f,   0.5f, "s"),
    F(ParamIdE::MEMORY_BLUR,     "memory_blur",     "Blur",         "Memory", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::MEMORY_SINK,     "memory_sink",     "Sink",         "Memory", 0.f,   1.f,   0.6f,  1.f,  ""),

    F(ParamIdE::TUNE_GRAVITY,    "tune_gravity",    "Self-Tune",    "Tuning", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::TUNE_REACH,      "tune_reach",      "Reach",        "Tuning", 5.f,   60.f,  30.f,  1.f,  "ct"),
    F(ParamIdE::TUNE_ANCHOR,     "tune_anchor",     "Anchor",       "Tuning", 0.f,   1.f,   0.2f,  1.f,  ""),

    F(ParamIdE::MODAL_PAINT,     "modal_paint",     "Paint",        "Modal", 0.f,   1.f,   0.f,   1.f,  ""),

    F(ParamIdE::SPEC_MIX,        "spec_mix",        "Spectral Echo", "Spectral Echo", 0.f,  1.f,   0.f,   1.f,  ""),
    F(ParamIdE::SPEC_TIME,       "spec_time",       "Time",         "Spectral Echo", 0.05f, 4.f,  0.8f,  0.5f, "s"),
    F(ParamIdE::SPEC_TILT,       "spec_tilt",       "Tilt",         "Spectral Echo", -1.f, 1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::SPEC_SCATTER,    "spec_scatter",    "Scatter",      "Spectral Echo", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::SPEC_FEEDBACK,   "spec_feedback",   "Feedback",     "Spectral Echo", 0.f,  0.95f, 0.5f,  1.f,  ""),
    F(ParamIdE::SPEC_DAMP,       "spec_damp",       "Damping",      "Spectral Echo", 0.f,  1.f,   0.4f,  1.f,  ""),
    F(ParamIdE::SPEC_TO_FAR,      "spec_to_far",     "To Far",       "Spectral Echo", 0.f,  1.f,   0.4f,  1.f,  ""),

    F(ParamIdE::COHERENCE,      "coherence",       "Coherence",    "Modulation", 0.f,  1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::COHERENCE_RATE,  "coherence_rate",  "Coherence Rate", "Modulation", 0.2f, 5.f, 1.f,   0.5f, "x"),
    F(ParamIdE::LENIA_RATE,      "lenia_rate",      "Lenia Rate",   "Modulation", 0.5f, 20.f,  4.f,   0.5f, "Hz"),
    F(ParamIdE::LENIA_GROWTH,    "lenia_growth",    "Lenia Growth", "Modulation", 0.08f, 0.3f, 0.15f, 1.f,  ""),
    F(ParamIdE::CHAOS_PERIOD,    "chaos_period",    "Chaos Period", "Modulation", 10.f, 1800.f, 120.f, 0.4f, "s"),
    F(ParamIdE::BRAIN_CASCADE,   "brain_cascade",   "Cascade",      "Conductor", 0.f,  1.f,   0.f,   1.f,  ""),

    C(ParamIdE::LIFE_MODE,       "life_mode",       "Life",         "Life", LifeModeNames, NumLifeModes, 0),
    F(ParamIdE::LIFE_DEPTH,      "life_depth",      "Life Depth",   "Life", 0.f,   1.f,   0.6f,  1.f,  ""),
    F(ParamIdE::LIFE_RATE,       "life_rate",       "Life Rate",    "Life", 0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::LIFE_CHARACTER,  "life_character",  "Character",    "Life", 0.f,   1.f,   0.3f,  1.f,  ""),
    F(ParamIdE::LIFE_ZOOM,       "life_zoom",       "Zoom",         "Life", 0.25f, 1.f,   1.f,   1.f,  ""),

    F(ParamIdE::SYNC_DEPTH,      "sync_depth",      "Sync",         "Sync", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::SYNC_COUPLING,   "sync_coupling",   "Coupling",     "Sync", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::SYNC_RATE,       "sync_rate",       "Sync Rate",    "Sync", 0.05f, 6.f,   0.5f,  0.4f, "Hz"),
    F(ParamIdE::SYNC_SPREAD,     "sync_spread",     "Spread",       "Sync", 0.f,   1.f,   0.3f,  1.f,  ""),

    F(ParamIdE::RIPPLE_DEPTH,    "ripple_depth",    "Ripple",       "Ripple", 0.f,  1.f,   0.f,   1.f,  ""),
    F(ParamIdE::RIPPLE_SPEED,    "ripple_speed",    "Ripple Speed", "Ripple", 0.1f, 4.f,   0.8f,  0.5f, "oct/s"),
    F(ParamIdE::RIPPLE_DECAY,    "ripple_decay",    "Ripple Decay", "Ripple", 0.5f, 30.f,  6.f,   0.5f, "s"),

    F(ParamIdE::RESERVOIR_CHAOS,  "reservoir_chaos",  "Chaos",      "Modulation", 0.5f, 1.5f, 0.95f, 1.f,  ""),
    F(ParamIdE::RESERVOIR_MEMORY, "reservoir_memory", "Memory",     "Modulation", 0.3f, 30.f, 4.f,   0.4f, "s"),
    F(ParamIdE::RESERVOIR_INPUT,  "reservoir_input",  "Input",      "Modulation", 0.f,  2.f,  0.7f,  1.f,  ""),

    F(ParamIdE::MELT_MIX,         "melt_mix",         "Melt",       "Melt", 0.f,   1.f,  0.f,  1.f,  ""),
    F(ParamIdE::MELT_TIME,        "melt_time",        "Melt Time",  "Melt", 0.5f,  60.f, 8.f,  0.4f, "s"),
    F(ParamIdE::MELT_SPREAD,      "melt_spread",      "Spread",     "Melt", 0.f,   2.f,  0.5f, 0.6f, "oct"),
    F(ParamIdE::MELT_TO_FAR,       "melt_to_far",      "To Far",     "Melt", 0.f,   1.f,  0.4f, 1.f,  ""),

    B(ParamIdE::MORPH_ACTIVE,     "morph_active",     "Morph",      "Morph", false),
    F(ParamIdE::MORPH_POS,        "morph_pos",        "A - B",      "Morph", 0.f,   1.f,  0.f,  1.f,  ""),
    F(ParamIdE::PRESET_GLIDE,     "preset_glide",     "Glide",      "Morph", 0.f,   600.f, 0.f, 0.4f, "s"),

    F(ParamIdE::STRETCH,         "stretch",          "Stretch",    "Cloud", 1080.f, 1320.f, 1200.f, 1.f, "ct/oct"),
    F(ParamIdE::SHIFT,           "shift",            "Shift",      "Cloud", -200.f, 200.f, 0.f,   1.f,  "Hz"),
    I(ParamIdE::LIFE_SIDES,       "life_sides",       "Sides",      "Life", 2.f,   12.f,  6.f),
    F(ParamIdE::LIFE_TURN,        "life_turn",        "Turn",       "Life", -4.f,  4.f,   0.5f,  1.f,  "/min"),
    F(ParamIdE::COMB_DEPTH,       "comb_depth",       "Comb",       "Comb", 0.f,   1.f,   0.f,   1.f,  ""),
    F(ParamIdE::COMB_DENSITY,     "comb_density",     "Density",    "Comb", 0.25f, 12.f,  2.f,   0.5f, "/oct"),
    F(ParamIdE::COMB_SWEEP,       "comb_sweep",       "Sweep",      "Comb", -2.f,  2.f,   0.1f,  1.f,  "/s"),
    F(ParamIdE::COMB_STEREO,      "comb_stereo",      "Stereo",     "Comb", 0.f,   1.f,   0.5f,  1.f,  ""),
    B(ParamIdE::MAP_ACTIVE,       "map_active",       "Map",        "Map", false),
    F(ParamIdE::MAP_X,            "map_x",            "Map X",      "Map", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::MAP_Y,            "map_y",            "Map Y",      "Map", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::MAP_RADIUS,       "map_radius",       "Radius",     "Map", 0.1f,  4.f,   0.4f,  0.45f, ""),
    F(ParamIdE::MAP_GLIDE,        "map_glide",        "Map Glide",  "Map", 0.1f,  60.f,  3.f,   0.4f, "s"),
    F(ParamIdE::ROUTE_SPEED,      "route_speed",      "Route Speed", "Map", 0.25f, 4.f,  1.f,   0.5f, "x"),
    C(ParamIdE::SYNC_MODE,        "sync_mode",        "Sync Mode",  "Sync", SyncModeNames, NumSyncModes, 0),
    F(ParamIdE::SYNC_SWARM,       "sync_swarm",       "Swarm",      "Sync", -1.f,  1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::SYNC_REACH,       "sync_reach",       "Reach",      "Sync", 0.f,   100.f, 20.f,  0.6f, "ct"),
    F(ParamIdE::SYNC_LAG,         "sync_lag",         "Lag",        "Sync", 0.f,   1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::TURBULENCE,      "turbulence",       "Turbulence", "Turbulence", 0.f, 1.f, 0.f,  1.f,  ""),
    F(ParamIdE::TURBULENCE_TIME,  "turbulence_time",  "Settle",     "Turbulence", 0.5f, 60.f, 6.f, 0.4f, "s"),
    B(ParamIdE::MPE_ON,           "mpe",              "MPE",        "MPE", true),
    I(ParamIdE::MPE_BEND,         "mpe_bend",         "Bend Range", "MPE", 1.f,   96.f,  48.f, "st"),
    C(ParamIdE::MPE_PRESS,        "mpe_press",        "Press",      "MPE", MpeTargetNames, NumMpeTargets, 1),
    F(ParamIdE::MPE_PRESS_DEPTH,   "mpe_press_depth",  "Press Depth", "MPE", 0.f,  1.f,   0.5f,  1.f,  ""),
    C(ParamIdE::MPE_SLIDE,        "mpe_slide",        "Slide",      "MPE", MpeTargetNames, NumMpeTargets, 3),
    F(ParamIdE::MPE_SLIDE_DEPTH,   "mpe_slide_depth",  "Slide Depth", "MPE", 0.f,  1.f,   0.5f,  1.f,  ""),
    F(ParamIdE::HALL_GROW,        "hall_grow",        "Grow",       "Hall", -1.f,  1.f,   0.f,   1.f,  ""),
    F(ParamIdE::HALL_MEMORY,      "hall_memory",      "Memory",     "Hall", 60.f,  14400.f, 1200.f, 0.3f, "s"),
    F(ParamIdE::TIME_BEND,        "time_bend",        "Time Bend",  "Time", -1.f,  1.f,   0.f,   1.f,  ""),
    F(ParamIdE::RIPPLE_BEND,      "ripple_bend",      "Bend",       "Ripple", 0.f, 1.f,   0.f,   1.f,  ""),
    F(ParamIdE::BRAIN_LIGHT,      "brain_light",      "Follow Light", "Conductor", 0.f, 1.f, 0.f,  1.f,  ""),
    F(ParamIdE::BRAIN_FOUNDATION, "brain_foundation", "Foundation", "Conductor", 0.f, 1.f,  0.f,   1.f,  ""),
} };

} // namespace

const std::array<ParamDesc, NumParams>& paramTable() { return Table; }

float toNorm(const ParamDesc& d, float v)
{
    if (d.m_max <= d.m_min) return 0.0f;
    const float x = std::clamp((v - d.m_min) / (d.m_max - d.m_min), 0.0f, 1.0f);
    return d.m_skew == 1.0f ? x : std::pow(x, d.m_skew);
}

float fromNorm(const ParamDesc& d, float n)
{
    const float x = std::clamp(n, 0.0f, 1.0f);
    return d.m_min + (d.m_max - d.m_min) * (d.m_skew == 1.0f ? x : std::pow(x, 1.0f / d.m_skew));
}

const ParamDesc* findParam(const char* key)
{
    if (!key) return nullptr;
    for (const ParamDesc& d : Table)
        if (std::strcmp(d.m_key, key) == 0) return &d;
    return nullptr;
}

} // namespace additiv
