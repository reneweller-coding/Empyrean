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
    Brightness,     ///< 0 .. 1: harmonics above 1 + b^2 * 200 fade out (1 keeps them all)
    OddEven,        ///< -1 .. 1: odd against even harmonics
    Inharmonic,     ///< 0 .. 1: the stiff string's stretch
    CloudWidth,     ///< the cloud around harmonic 1, cents
    WidthSlope,     ///< how the cloud widens with the harmonic number
    Drift,          ///< each partial's own pitch wander, cents
    DriftRate,      ///< its speed, Hz
    Shimmer,        ///< each partial's own amplitude wander, 0 .. 1
    ShimmerRate,    ///< its speed, Hz
    Stereo,         ///< how far the partials scatter across the field
    // Air: a band of noise made of partials
    Air,            ///< the band's level against the voice's, 0 .. 1
    AirCenter,      ///< its centre, times the fundamental
    AirWidth,       ///< its width, cents
    AirJitter,      ///< how fast its partials flicker, Hz
    // Filter: read as a frequency response at every partial
    FilterModel,    ///< kFilterModelNames
    Cutoff,         ///< Hz
    Resonance,      ///< 0 .. 1
    FilterEnv,      ///< octaves the voice's envelope moves the cutoff
    KeyTrack,       ///< 0 .. 1: how far the cutoff follows the note
    FilterDrift,    ///< 0 .. 1: the cutoff's own slow wander, up to an octave
    // Motion: the envelope lanes, as a scene
    Motion,         ///< which motion the partials' envelopes make (kMotionNames)
    MotionRate,     ///< how fast the motion runs, times its own speed
    MotionDepth,    ///< how deep it reaches, times its own depth
    Landscape,      ///< which landscape the Aurora motion reads (kLandscapeNames)
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
    Tide,           ///< cents the whole instrument leans on a minute-scale curve
    TidePeriod,     ///< minutes per swing
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
    AutoStep,       ///< Autoplay: a rising edge exchanges a voice now
    Brain2On,       ///< the second conductor, in the background
    Brain2Interval, ///< its root against the first one's, semitones
    Brain2Density,  ///< its notes at once
    Brain2Rate,     ///< seconds between its events
    Brain2Depth,    ///< where its notes stand, 0 near .. 1 far
    // Space
    Depth,          ///< how deep the conductor's notes sit (near/far planes)
    KeysDepth,      ///< where played notes sit, 0 at the ear .. 1 far
    FarLevel,       ///< the far plane's reverb level
    FarSize,        ///< the far reverb's size
    FarDecay,       ///< its decay, seconds
    FarDamp,        ///< its damping
    Presence,       ///< dB: the 2-5 kHz bell on the near plane
    PadLowCut,      ///< Hz: below it the clouds fall 12 dB/oct, leaving the bottom free
    Breath,         ///< 0 .. 1: every voice's distance wanders
    BreathRate,     ///< Hz
    Doppler,        ///< 0 .. 1: a voice coming closer rises, one receding falls
    BassMono,       ///< below this the mix is mono, Hz
    StereoWidth,    ///< the master's width
    // Modulation: the sources' own settings (the matrix rows are data, Modulation.h)
    Lfo1Shape,      ///< kLfoShapeNames
    Lfo1Rate,       ///< Hz
    Lfo2Shape,      ///< kLfoShapeNames
    Lfo2Rate,       ///< Hz
    Lfo3Shape,      ///< kLfoShapeNames
    Lfo3Rate,       ///< Hz
    Lfo4Shape,      ///< kLfoShapeNames
    Lfo4Rate,       ///< Hz
    Env1Shape,      ///< a factory shape (Shapes.h)
    Env1Rate,       ///< how fast it runs, times its own speed
    Env2Shape,      ///< a factory shape
    Env2Rate,       ///< how fast it runs
    Macro1,         ///< 0 .. 1
    Macro2,         ///< 0 .. 1
    Macro3,         ///< 0 .. 1
    Macro4,         ///< 0 .. 1
    // Echo: Noctuary's two asymmetric delays, the second in series after the first (12.10.2026)
    DelayMix,       ///< the echoes on the near plane, 0 .. 1
    DelayTimeL,     ///< s
    DelayTimeR,     ///< s
    DelayFeedback,  ///< 0 .. 0.95
    DelayCross,     ///< 0 .. 1: how much of each side's feedback comes from the other
    DelayDamp,      ///< 0 .. 1: the loop's low pass
    DelayAbsorb,    ///< 0 .. 1: the loop narrows to a band as the feedback rises
    DelayDuck,      ///< 0 .. 1: the loop darkens while the input is loud
    DelayToFar,     ///< 0 .. 1: the echoes receding into the far plane
    Delay2Mix,      ///< the second echo, 0 .. 1
    Delay2TimeL,    ///< s
    Delay2TimeR,    ///< s
    Delay2Feedback, ///< 0 .. 0.95
    Delay2Damp,     ///< 0 .. 1
    Delay2ToFar,    ///< 0 .. 1
    // Room: the near plane's early reflections and small reverb
    EarlyLevel,     ///< 0 .. 1
    EarlySize,      ///< m
    EarlyAbsorb,    ///< 0 .. 1
    NearMix,        ///< the near reverb, 0 .. 1 wet
    NearDecay,      ///< s
    NearDamp,       ///< 0 .. 1
    // The far hall, more of Noctuary's
    FarDiffuse,     ///< 0 .. 1: the swell in front of the far reverb
    FarPreDelay,    ///< ms
    FarUnmask,      ///< 0 .. 1: the far hall steps aside for the near plane, band by band
    FarMode,        ///< kFarModeNames
    FarFreeze,      ///< the far hall held
    // The modal hall: tens of thousands of decaying resonances on the GPU
    ModalLevel,     ///< 0 .. 1
    ModalDecay,     ///< s to -60 dB, at and below 250 Hz
    ModalDamp,      ///< 0 .. 1: how much faster the upper octaves die
    ModalTune,      ///< 0 .. 1: the share of the modes tuned to the scale
    ModalSpread,    ///< ct: how far a tuned mode lies from its note
    ModalDrift,     ///< ct: every mode's slow wander
    ModalWidth,     ///< 0 .. 1: mono .. every mode its own place
    ModalLow,       ///< Hz: the lowest mode
    ModalFreeze,    ///< the modal hall held, its input shut
    ModalBreath,    ///< 0 .. 1: every mode's decay wanders, slowly
    // The field on the partials and the dispersion of their clocks (Cloud.h SpectralField)
    FieldMode,      ///< kFieldModeNames
    FieldStrength,  ///< 0 .. 1: how fast the partials may move (600 x^2 cents a second)
    FieldRadius,    ///< ct: an attractor's reach; the Scale field's stiffness
    FieldSpring,    ///< 0 .. 1: the pull back home (4 x^2 per second)
    FieldRate,      ///< how fast the attractors wander (time of the Lorenz system per second)
    FieldCount,     ///< 1 .. 8 attractors
    Dispersion,     ///< s: the lowest partials' clock late (positive) or the highest (negative)
    // Phase 9 (12.10.2026)
    Binaural,       ///< the near plane placed round the head (headphones)
    BinauralSpan,   ///< 0 .. 1: the places over the front half .. all round
    BinauralHeight, ///< 0 .. 1: the elevations' spread
    BinauralOrbit,  ///< Hz: the whole cloud turning round the head (negative: the other way)
    MemoryLevel,    ///< 0 .. 1: the ghosts of the notes let go
    MemoryTime,     ///< s: how long a ghost takes to fade by 60 dB
    MemoryBloom,    ///< s: how long it takes to rise
    MemoryBlur,     ///< 0 .. 1: how far it widens, drifts and darkens as it fades
    MemorySink,     ///< 0 .. 1: how far it recedes into the far plane
    TuneGravity,    ///< 0 .. 1: self-tuning: voices whose harmonics nearly meet pull together
    TuneReach,      ///< ct: how near they must be
    TuneAnchor,     ///< 0 .. 1: the pull back to the scale
    ModalPaint,     ///< 0 .. 1: the modal hall heard through the landscape
    SpecMix,        ///< 0 .. 1: the spectral echo on the near plane
    SpecTime,       ///< s: its bands' middle delay
    SpecTilt,       ///< -1 .. 1: the highs later (positive) or earlier
    SpecScatter,    ///< 0 .. 1: each band a time of its own
    SpecFeedback,   ///< 0 .. 0.95
    SpecDamp,       ///< 0 .. 1: less feedback towards the top
    SpecToFar,      ///< 0 .. 1: its share into the far plane
    // Phase 10a: Noctuary's modulators
    Coherence,      ///< 0 .. 1: the Kuramoto bank's coupling (free .. in step)
    CoherenceRate,  ///< x: its speed (natural periods 23, 31, 41, 53 s)
    LeniaRate,      ///< Hz: Lenia fields per second
    LeniaGrowth,    ///< Lenia's growth centre (mu)
    ChaosPeriod,    ///< s: about a Lorenz lobe change and a Roessler turn
    BrainCascade,   ///< 0 .. 1: the conductor's events breed events (a Hawkes clock)
    // Phase 10b: the living landscape
    LifeMode,       ///< kLifeModeNames
    LifeDepth,      ///< 0 .. 1: how far the partials' loudness follows the field
    LifeRate,       ///< 0 .. 1: how fast it lives
    LifeCharacter,  ///< 0 .. 1: Lenia's growth; the Turing field's way from spots to mazes
    LifeZoom,       ///< 0.25 .. 1: how much of the field's width the range spans
    // Phase 10c: partial sync
    SyncDepth,      ///< 0 .. 1: how far each partial's loudness follows its oscillator
    SyncCoupling,   ///< 0 .. 1: how strongly the members of a harmonic pull each other into step
    SyncRate,       ///< Hz: the oscillators' middle rate
    SyncSpread,     ///< 0 .. 1: how far their own rates scatter
    // Phase 10f: the ripple
    RippleDepth,    ///< 0 .. 1: how far the partials' loudness follows the wave
    RippleSpeed,    ///< octaves a second the wave runs along the spectrum
    RippleDecay,    ///< s: how long a strike rings on the string
    // Phase 10g: the reservoir
    ReservoirChaos, ///< its coupling's spectral radius: below 1 it forgets, about 1 the edge of chaos
    ReservoirMemory,///< s: how slowly its state moves
    ReservoirInput, ///< how strongly it hears the music
    // Phase 10h: the melt
    MeltMix,        ///< 0 .. 1: the melt on the near plane
    MeltTime,       ///< s: how long the melted sound lingers (to -60 dB)
    MeltSpread,     ///< octaves: how far a held sound has spread after a second
    MeltToFar,      ///< 0 .. 1: its share into the far plane
    // Phase 11: morph and journeys (the player's: presets never set them, Presets.h keptByPresets)
    MorphActive,    ///< the sound is the blend of the morph's slots A and B (Morph.h)
    MorphPos,       ///< 0 (A) .. 1 (B): where between them
    PresetGlide,    ///< s: a preset chosen glides from the sound before it (0: it comes at once)
    // Phase 13: the partials' ratios
    Stretch,        ///< cents an octave of the harmonics spans (1200: harmonic; Sethares' stretched timbres)
    Shift,          ///< Hz every partial is moved by (a frequency shifter: harmonic ratios become inharmonic)
    LifeSides,      ///< animated pictures: mirror wedges, spiral arms, grating lines
    LifeTurn,       ///< animated pictures: turns a minute (negative: the other way)
    CombDepth,      ///< 0 .. 1: the spectral comb's notches (0: none)
    CombDensity,    ///< notches an octave
    CombSweep,      ///< notches a second the comb travels (positive: up, an endless rise)
    CombStereo,     ///< 0 .. 1: left and right combed apart
    // Phase 14: the preset map (the player's: presets never set them)
    MapActive,      ///< the sound is the blend of the presets round the map's cursor (PresetMap.h)
    MapX,           ///< the cursor across, 0 .. 1 (dark to bright)
    MapY,           ///< the cursor up, 0 .. 1 (still to moving)
    MapRadius,      ///< the blend's width on the map
    MapGlide,       ///< s: how fast the sound follows the cursor
    RouteSpeed,     ///< times: how fast a route is walked (its travel and hold times divided by it)
    // Phase 16 (14.10.2026): swarms and chimeras, turbulence, MPE
    SyncMode,       ///< how the partials' oscillators couple: Global (Kuramoto), Chimera (to their neighbours), Swarm (swarmalators)
    SyncSwarm,      ///< -1 .. 1: Swarm: how a partial's place follows its phase (positive: like phases gather, negative: they part)
    SyncReach,      ///< cents: Swarm: how far a partial wanders in pitch (and with it across the field) on its circle
    SyncLag,        ///< 0 .. 1: Chimera: the coupling's phase lag (more: more of the cloud falls out of step)
    Turbulence,     ///< 0 .. 1: the energy cascade: a loud harmonic hands its energy upwards, the louder the more
    TurbulenceTime, ///< s: how long the cascade takes to settle back when the sound grows soft
    MpeOn,          ///< MPE: notes on channels 2 .. 16 carry their own bend, pressure and slide (the lower zone)
    MpeBend,        ///< semitones: a member channel's full bend (48 by the MPE standard; an RPN 0 from the controller sets it)
    MpePress,       ///< what a note's pressure moves (kMpeTargetNames)
    MpePressDepth,  ///< 0 .. 1: how far
    MpeSlide,       ///< what a note's slide (CC 74 on its channel) moves
    MpeSlideDepth,  ///< 0 .. 1: how far
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

/**
 * @brief A value in a parameter's normalised domain: 0 .. 1 across its range, the skew applied as
 *        the hosts apply it ((v - min) / (max - min))^skew -- where modulation depths are measured, so the
 *        same depth means the same on a cutoff in hertz and on a mix.
 * @param d  the row
 * @param v  the value in its unit
 * @return   0 .. 1
 */
float toNorm(const ParamDesc& d, float v);

/** @brief Back from the normalised domain. @param d the row @param n 0 .. 1 @return the value in its unit */
float fromNorm(const ParamDesc& d, float n);

constexpr int kNumScaleChoices = 13;   ///< entries of kScaleNames: eleven tables, the user slot and the timbre scale
constexpr int kUserScaleIndex = 11;    ///< the Scala slot, the first choice that is not a table (Tuning.cpp checks it)
constexpr int kTimbreScaleIndex = 12;  ///< the scale computed from the spectrum (Sethares)
extern const char* const kScaleNames[kNumScaleChoices];   ///< "12-TET", "JI Major (Ptolemy)", ... (as in Noctuary)
constexpr int kNumLifeModes = 13;   ///< entries of kLifeModeNames
extern const char* const kLifeModeNames[kNumLifeModes];     ///< "Off", "Lenia", "Turing", the animated pictures, "Fluid", "Islands" (Cloud.h LifeMode)
constexpr int kNumSyncModes = 3;   ///< entries of kSyncModeNames
extern const char* const kSyncModeNames[kNumSyncModes];     ///< "Global", "Chimera", "Swarm" (Cloud.h SyncMode)
constexpr int kNumMpeTargets = 8;  ///< entries of kMpeTargetNames
extern const char* const kMpeTargetNames[kNumMpeTargets];   ///< "Off", "Level", "Filter", "Brightness", "Width", "Nearer", "Shimmer", "Motion"

/** @brief What a note's pressure or slide moves (MPE), in kMpeTargetNames' order. */
enum class MpeTarget : int { Off = 0, Level, Filter, Brightness, Width, Nearer, Shimmer, Motion };
constexpr int kNumFieldModes = 5;   ///< entries of kFieldModeNames
extern const char* const kFieldModeNames[kNumFieldModes];   ///< "Off", "Scale", "Swarm", "Scatter", "Flock" (Cloud.h FieldMode)
constexpr int kNumFarModes = 4;   ///< entries of kFarModeNames
extern const char* const kFarModeNames[kNumFarModes];   ///< the far reverb's networks (Effects.h Reverb::setMode): "Classic", "Scattering", "Colourless", "Rotating"
extern const char* const kRootNames[12];                  ///< "C" .. "B"
extern const char* const kKeyMapNames[2];                 ///< "Snap to 12 keys", "Consecutive degrees"

constexpr int kNumMotions = 7;                       ///< entries of kMotionNames
extern const char* const kMotionNames[kNumMotions];  ///< None, Random, Wave, Pendulum, Breathe, Glissando, Aurora
extern const char* const kFilterModelNames[];        ///< Off, LP 6, LP 12, LP 24, HP 12, BP 12, Notch, Peak, Formant (Cloud.h FilterModel)
constexpr int kNumLfoShapes = 7;                     ///< entries of kLfoShapeNames
extern const char* const kLfoShapeNames[kNumLfoShapes];   ///< Sine, Triangle, Ramp Up, Ramp Down, Square, Random, Steps
extern const char* const kEnvShapeNames[];           ///< the factory shapes' names, for the envelopes' choice (Shapes.h)

} // namespace additiv
