/**
 * @file Cloud.h
 * @brief The GPU partial cloud: every voice a few dozen numbers on the host, tens of thousands of
 *        partials on the GPU.
 *
 * A voice is a harmonic spectrum whose every harmonic is not one sine but a cloud of `members`
 * slightly detuned sines (the PADsynth idea, played live): the cloud's width in cents, how it
 * widens towards the top, and every partial's own slow pitch drift, amplitude shimmer and place in
 * the stereo field. The host only fills CloudVoice; the GPU expands it into partials each block
 * (update kernel), sums them sample by sample (render kernel) and folds the chunks into one
 * block of four channels -- near and far, left and right, the two planes of Noctuary's spatial
 * model -- (reduce kernel). The three kernels and the two copies around them are one CUDA graph,
 * so a block costs one launch.
 *
 * Every partial also has its own envelopes, on three lanes: amplitude, pitch and place in the
 * stereo field. A lane does not store 20,000 envelopes; it says how the partials share a bank of
 * loopable multi-segment shapes (or one landscape of time against frequency): which shape each
 * partial takes, how fast it runs, where in its loop it starts, how late it begins and how deep it
 * reaches -- each as a function of the partial's pitch, harmonic, place in its cloud, parity or
 * chance (MapSource). Envelopes are pure functions of time, evaluated once per block per partial;
 * the render kernel ramps between them.
 *
 * This header carries no CUDA types: the plugin and the tools include it from plain C++.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace additiv {

constexpr int kEnvPoints = 16;            ///< breakpoints per envelope shape
constexpr int kEnvShapes = 64;            ///< shapes in the bank
constexpr int kLandMaxColumns = 512;      ///< frequency columns a landscape may have
constexpr int kLandMaxRows = 8192;        ///< time rows a landscape may have
constexpr float kLandLowHz = 20.0f;       ///< a landscape's first column
constexpr float kLandHighHz = 20000.0f;   ///< and its last; the columns between are spaced logarithmically

/** @brief What a partial's share of an envelope lane is read from: each source is 0 .. 1 per partial. */
enum class MapSource : std::int32_t {
    None = 0,   ///< always 0: every partial the same
    Pitch,      ///< the partial's centre frequency, logarithmically from kLandLowHz (0) to kLandHighHz (1)
    Harmonic,   ///< its harmonic within the voice: 0 for the fundamental, 1 for the highest
    Member,     ///< its place in its harmonic's cloud: 0 the lowest member, 0.5 the centre, 1 the highest
    Random,     ///< its own dice
    OddEven,    ///< 0 on odd harmonics, 1 on even ones
};
constexpr int kMapSources = 6;   ///< how many MapSource values there are

/** @brief One breakpoint of an envelope shape. */
struct EnvPoint {
    float time = 0.0f;    ///< seconds from the shape's start, rising from point to point
    float level = 0.0f;   ///< 0 .. 1 on the amplitude lane, -1 .. 1 on pitch and pan
    float curve = 0.0f;   ///< -1 .. 1, the bend of the segment arriving here (0 straight, + slow start, - fast start)
    float pad = 0.0f;     ///< keeps the point 16 bytes
};

/**
 * @brief A multi-segment envelope with an optional loop and a release tail.
 *
 * It runs from point 0; between loopStart and loopEnd it loops while the note is held (a loop
 * from the first to the last point makes it an LFO). At the note's release a shape with points
 * after loopEnd leaves its loop wherever it is and glides through those points; a looping shape
 * without such a tail keeps looping (the voice's own envelope fades it). Without a loop it holds
 * its last level.
 */
struct EnvShape {
    EnvPoint p[kEnvPoints];       ///< the points, `count` of them
    std::int32_t count = 0;       ///< points in use, 0 .. kEnvPoints (0 = a flat zero)
    std::int32_t loopStart = -1;  ///< first point of the loop, -1 = no loop
    std::int32_t loopEnd = -1;    ///< last point of the loop
    std::int32_t pad = 0;         ///< keeps the shape a multiple of 16 bytes
};

/**
 * @brief One envelope lane of a voice: how its partials share the shapes (or the landscape).
 *
 * Every per-partial value is a base plus a spread read from a MapSource. A lane with amount 0 is
 * skipped and costs nothing.
 */
struct EnvLane {
    float amount = 0.0f;       ///< amplitude: depth 0 .. 1; pitch: cents; pan: share of the field 0 .. 1
    float rate = 1.0f;         ///< how fast the shapes run (1 = as written)
    float rateSpread = 0.0f;   ///< rate times (1 + rateSpread * source): a linear spread makes a pendulum wave
    float phase = 0.0f;        ///< loop cycles added times the source: a travelling wave (negative: the other way)
    float delay = 0.0f;        ///< seconds of delay times the source (negative: times 1 - source)
    float depthTilt = 0.0f;    ///< -1 .. 1: the amount only where the source is low (-1) or high (+1)
    float chaos = 0.0f;        ///< 0 .. 1: from lockstep to every partial for itself (random loop phase, rate +-1 octave)
    std::int32_t shape = 0;         ///< first shape of the bank this lane uses
    std::int32_t shapeSpread = 1;   ///< how many shapes from there the partials spread over
    std::int32_t landscape = 0;     ///< 1 = read the landscape instead of a shape
    MapSource shapeBy = MapSource::Random;     ///< which source picks the shape
    MapSource rateBy = MapSource::Harmonic;    ///< which source spreads the rate
    MapSource phaseBy = MapSource::Pitch;      ///< which source spreads the loop phase
    MapSource delayBy = MapSource::Pitch;      ///< which source spreads the delay
    MapSource depthBy = MapSource::Pitch;      ///< which source tilts the depth
    std::int32_t pad = 0;           ///< keeps the lane 64 bytes
};

/** @brief One voice as the host describes it; the engine fills the derived fields at launch(). */
struct CloudVoice {
    float hz = 110.0f;            ///< fundamental, Hz
    float level = 0.05f;          ///< RMS the voice aims at with gain 1 (all partials together)
    float gain = 0.0f;            ///< the envelope at the end of the coming block; the partials ramp to it across the block
    float tilt = 1.2f;            ///< spectral tilt: harmonic h at h^-tilt
    float width = 25.0f;          ///< cents: the cloud's width (two standard deviations) around harmonic 1
    float widthSlope = 0.3f;      ///< how the width grows with the harmonic number: width * h^widthSlope
    float drift = 3.0f;           ///< cents each partial wanders on its own
    float driftRate = 0.1f;       ///< Hz, the speed of that wander
    float shimmer = 0.4f;         ///< 0 .. 1, each partial's own amplitude wander
    float shimmerRate = 0.15f;    ///< Hz, the speed of the shimmer
    float stereo = 1.0f;          ///< 0 .. 1, how far the partials scatter across the stereo field
    float time = 0.0f;            ///< seconds since the note began, the clock of every envelope (advanced by the engine)
    float releaseAt = -1.0f;      ///< the note's time of release in seconds, -1 while it is held
    int   partials = 20000;       ///< partials this voice renders, 1 .. the engine's partialsPerVoice
    std::uint32_t seed = 1;       ///< the note's dice: initial phases, member offsets, drift and shimmer rates, pans
    int   active = 0;             ///< 0 = silent: the GPU skips the voice entirely
    int   restart = 0;            ///< 1 = a new note: fresh phases, amplitudes from zero, time 0 (cleared by launch())
    EnvLane amp;                  ///< the amplitude lane
    EnvLane pitch;                ///< the pitch lane, in cents
    EnvLane pan;                  ///< the stereo lane, added to each partial's place
    /**
     * @brief Where the voice stands, 0 at the ear .. 1 deep in the background (Noctuary's planes,
     *        after Robert Rich). Every partial splits between the near and the far output by
     *        cos/sin(d pi/2), loses 6 dB per unit and is darkened by the air (see airAbsorb).
     */
    float distance = 0.0f;
    float airAbsorb = 1.0f;       ///< 0 .. 1: how far distance darkens -- at 1 a partial's corner falls 5 octaves per unit from 16 kHz
    int   onset = 0;              ///< with restart: samples into the block at which the note begins (sample-accurate starts)
    // The spectrum (Noctuary's formula; the voice is normalised after it, so these colour, not level).
    float brightness = 1.0f;      ///< 0 .. 1: harmonics above 1 + b^2 * 200 fade out (raised cosine); 1 keeps them all
    float oddEven = 0.0f;         ///< -1 .. 1: positive thins the even harmonics, negative the odd ones above the first
    float inharmonic = 0.0f;      ///< the stiff string's B: harmonic h at h sqrt(1 + B h^2)
    float presenceDb = 0.0f;      ///< 0 .. 6 dB: a bell over 1.8 .. 5.6 kHz, only near the listener (times 1 - distance)
    float lowCut = 0.0f;          ///< Hz: partials below fall 12 dB/oct (0 = off)
    // The voice filter, read as a frequency response at every partial (FilterModel).
    int   filterModel = 0;        ///< FilterModel; 0 = off
    float cutoff = 2500.0f;       ///< Hz, after key tracking, envelope and drift (the engine's)
    float filterQ = 0.707f;       ///< the resonance as a Q
    float vowel = 0.0f;           ///< Formant: 0 .. 4, u o a e i
    // Air: a share of the partials as a band of noise around a multiple of the fundamental.
    float air = 0.0f;             ///< 0 .. 1: the band's level against the voice's
    float airCenter = 3.0f;       ///< the band's centre, times the fundamental
    float airWidth = 600.0f;      ///< its width in cents (two standard deviations)
    float airJitter = 8.0f;       ///< Hz: how fast its partials flicker in pitch and level -- what makes them noise
    // Filled by the engine at launch(), read by the GPU.
    int   harmonics = 0;          ///< harmonics below the cutoff, at most `partials`
    int   members = 0;            ///< partials per harmonic
    int   harmonicPartials = 0;   ///< partials of the harmonic clouds; the rest up to `partials` are air
    float scale = 0.0f;           ///< amplitude of harmonic 1's every member, so that the voice meets `level`
    float airScale = 0.0f;        ///< amplitude of every air partial
    float filterMix = 1.0f;       ///< 0 .. 1: the filter's gain raised to this (a morph fades one filter out and another in)
    float stretch = 1.0f;         ///< harmonic h at h^stretch (partialRatio)
    float shift = 0.0f;           ///< Hz added to every partial (below 20 Hz they fade out)
    float turbulence = 0.0f;      ///< the energy cascade's drive for this note, per second (0: none; Turbulence)
    float pad2 = 0.0f;            ///< keeps the struct a multiple of 16 bytes
    float pad3 = 0.0f;            ///< likewise
};

/** @brief The voice filter's models; on the GPU each is its analogue prototype's magnitude at the partial's frequency. */
enum class FilterModel : int {
    Off = 0,   ///< no filter
    LP6,       ///< one-pole low pass
    LP12,      ///< two-pole low pass with resonance
    LP24,      ///< two of them in series
    HP12,      ///< two-pole high pass
    BP12,      ///< band pass, unity at the cutoff
    Notch,     ///< band stop
    Peak,      ///< a resonant bell on a flat line
    Formant,   ///< three band passes on the formants of u-o-a-e-i; the cutoff picks the vowel
    Count      ///< number of models
};
constexpr int kNumFilterModels = static_cast<int>(FilterModel::Count);   ///< entries of kFilterModelNames

constexpr int kCloudChannels = 4;      ///< the voices' channels on the GPU: near left, near right, far left, far right
constexpr int kCloudOutChannels = 10;  ///< a finished frame: the four voice channels, the modal hall's left and right, the spectral echo's left and right, the melt's left and right
constexpr int kModalModes = 65536;     ///< the modal hall's resonances
constexpr int kModalTuned = 128;       ///< pitches the hall's tuned modes may sit on

/**
 * @brief The modal hall's settings for the next block: the host's copy, sent up with the voices.
 *
 * A reverb made the way the instrument is made: kModalModes resonances, each a complex one-pole
 * (a damped sine) driven by the voices of the block -- near and far, left and right -- and heard
 * through a phase of its own in each ear. Its tail is a sum of decaying sines, as the voices are a
 * sum of sustained ones (Abel, Coffin and Spratt 2014, a modal architecture for artificial
 * reverberation). Where the modes stand is the instrument's choice: spread evenly over the range
 * they are a hall; a share `tune` of them on the pitches of the scale and the hall rings in key,
 * like the sympathetic strings of a viola d'amore. Every mode's weight is set so that white noise
 * comes out at the level it went in, whatever the decay; tuned modes are weighted down by how much
 * closer they stand together than spread ones, so turning `tune` colours the hall rather than
 * making it louder.
 */
struct ModalHall {
    int   on = 0;           ///< 0: the hall rests -- silent, its modes cleared, almost no GPU time
    int   freeze = 0;       ///< 1: the modes ring on (about 24 minutes to -60 dB) with the input shut
    int   tunedCount = 0;   ///< pitches in `tuned`
    int   pad = 0;          ///< keeps what follows 16-byte aligned
    float decay = 8.0f;     ///< seconds to -60 dB at and below 250 Hz
    float damp = 0.35f;     ///< 0 .. 1: every octave above 250 Hz dies 2^damp times faster
    float tune = 0.5f;      ///< 0 .. 1: the share of the modes on the scale's pitches
    float spread = 6.0f;    ///< cents a tuned mode may lie from its pitch (a triangle)
    float drift = 4.0f;     ///< cents of every mode's slow wander (0.02 .. 0.12 Hz)
    float width = 1.0f;     ///< 0: both ears hear the same phase of every mode (mono) .. 1: each its own
    float low = 80.0f;      ///< Hz: the lowest mode of the spread ones
    float time = 0.0f;      ///< seconds: the drift's clock, wrapped at 1000 s (every drift rate is a whole number of cycles in 1000 s)
    float breath = 0.0f;    ///< 0 .. 1: every mode's decay wanders by up to an octave either way, slowly, each on its own
    float paint = 0.0f;     ///< 0 .. 1: every mode heard through the landscape at its frequency (the painted hall)
    float paintNorm = 1.0f; ///< the landscape's make-up (one over its RMS), so the painted hall keeps its level
    float paintTime = 0.0f; ///< seconds into the landscape's loop, for the painting
    float tuned[kModalTuned] = {};   ///< Hz: the scale's pitches between `low` and 16 kHz
};
static_assert(sizeof(ModalHall) % 16 == 0, "ModalHall rides in the block header");

/**
 * @brief The binaural partial cloud: every partial at its own place around the head.
 *
 * Because every partial is a sine, what a head does to it is one complex gain per ear at its
 * frequency: the head's shadow (Brown and Duda 1998, a pole and a zero whose zero moves with the
 * angle of incidence), the pinna's five reflections by azimuth and elevation (the same model), and
 * the time the sound takes round the head (Woodworth) as a phase. No convolution: the update kernel
 * works out the two gains per partial per block, the render kernel adds a cosine to the sine it
 * already has. Only the near plane is placed so; the far plane goes into the far reverb, which is
 * diffuse anyway. The two ears' power is kept where the stereo pan law had it, so switching changes
 * the place, not the level.
 */
struct Binaural {
    int   on = 0;          ///< 1: the near plane binaural (for headphones)
    float span = 0.5f;     ///< 0: the partials' places spread over the front half (+-90 degrees) .. 1: all round (+-180)
    float height = 0.3f;   ///< 0 .. 1: elevations spread over up to +-60 degrees
    float angle = 0.0f;    ///< radians: the whole cloud turned (the orbit), 0 in front, positive to the right
};
static_assert(sizeof(Binaural) % 16 == 0, "Binaural rides in the block header");

/**
 * @brief The spectral echo: a delay line for every frequency band, on the GPU.
 *
 * The voices (near and far) are taken apart every block by a 1024-point FFT (a Hann window, the
 * block as the hop), every band of each ear goes through a delay line of its own with feedback,
 * and the bands are put back together and overlapped (output: channels 6 and 7, 896
 * samples after the input at 128 a block). The bands' times follow `tilt` (positive: the highs
 * later) and `scatter` (each band a time of its own), the feedback falls towards the top by
 * `damp`. Echoes then come back as cascades rather than as copies.
 */
struct SpectralEcho {
    int   on = 0;            ///< 1: running (0: silent, nothing computed)
    int   clear = 0;         ///< 1 for one block: the lines emptied first (the engine asks when it switches on)
    int   pad0 = 0;          ///< keeps the floats 16-byte aligned
    int   pad1 = 0;          ///< likewise
    float frames = 187.0f;   ///< the bands' middle delay, in blocks
    float tilt = 0.0f;       ///< -1 .. 1: the delay times 2^(2 tilt (place - 0.5)), place 0 at 20 Hz .. 1 at 20 kHz
    float scatter = 0.0f;    ///< 0 .. 1: each band's time scattered by up to +-75 %
    float feedback = 0.0f;   ///< 0 .. 0.95
    float damp = 0.0f;       ///< 0 .. 1: the feedback times (1 - damp place)
    float pad2 = 0.0f;       ///< keeps the table 16-byte aligned
    float pad3 = 0.0f;       ///< likewise
    float pad4 = 0.0f;       ///< likewise
};
static_assert(sizeof(SpectralEcho) % 16 == 0, "SpectralEcho rides in the block header");

constexpr int kMeltN = 4096;     ///< the melt's FFT size (11.7 Hz bands at 48 kHz)
constexpr int kMeltHop = 512;    ///< samples between its frames

/**
 * @brief The melt: a spectral blur on the GPU (output: channels 8 and 9, kMeltN - block samples late).
 *
 * Every kMeltHop samples the voices (near and far) are taken apart by a kMeltN-point FFT. Each band's
 * energy is held -- the new frame mixed in by 1 - `keep` -- and the held spectrum diffuses across the
 * bands, a little every frame (`widen`, in octaves on a log axis), so what has been held longest is
 * spread widest: a chord melts into the space between its notes. A phase vocoder sings it again:
 * each band follows the frequency it hears (its partial's, not the band's centre), is pulled back
 * to its own centre by `pull` as it melts, and is detuned a little, otherwise in each ear.
 */
struct Melt {
    int   on = 0;              ///< 1: running (0: silent, nothing computed)
    int   clear = 0;           ///< 1 for one block: everything emptied first (the engine asks when it switches on)
    int   pad0 = 0;            ///< keeps the floats 16-byte aligned
    int   pad1 = 0;            ///< likewise
    float keep = 0.99f;        ///< per frame: the share of the held energy kept
    float widen = 0.0f;        ///< per frame: the blur's standard deviation in bands, per band index
    float pull = 0.0f;         ///< per frame: each band's frequency pulled towards its centre by this share
    float detune = 0.0f;       ///< radians a sample: each band's offset at most (its own in each ear)
};
static_assert(sizeof(Melt) % 16 == 0, "Melt rides in the block header");

constexpr int kSnapStride = 8;       ///< every eighth partial of every voice is in the picture's snapshot
constexpr int kLifeW = 256;          ///< the living field's width, cells (across the spectrum)
constexpr int kLifeH = 128;          ///< its height, cells (across a cloud's members)
constexpr int kLifeRadius = 10;      ///< Lenia's ring kernel radius, cells
constexpr int kLifeJets = 16;        ///< the Fluid's jets: two for each of the eight loudest notes
constexpr int kIsleW = 64;           ///< Islands: its grid across (each cell four of the field's)
constexpr int kIsleH = 32;           ///< Islands: its grid along

/**
 * @brief The spectral comb: notches evenly spaced per octave, read at every partial's frequency
 *        (a filter only an additive instrument has: no delay line, any spacing, any depth). Its
 *        phase runs on, so the notches travel through the spectrum without end -- a phaser that
 *        rises or falls forever -- and each partial reads it a little further on by its place in
 *        the stereo field, so left and right are combed apart.
 */
struct SpectralComb {
    float depth = 0.0f;     ///< 0 .. 1: the notches' depth (0: no comb)
    float density = 2.0f;   ///< notches an octave
    float phase = 0.0f;     ///< where the notches stand, in notches (the engine runs it on)
    float stereo = 0.0f;    ///< 0 .. 1: half a notch apart from one side of the field to the other
    float makeup = 1.0f;    ///< one over the root of the comb's mean square
    float pad0 = 0.0f;      ///< keeps the struct a multiple of 16 bytes
    float pad1 = 0.0f;      ///< likewise
    float pad2 = 0.0f;      ///< likewise
};
static_assert(sizeof(SpectralComb) % 16 == 0, "SpectralComb rides in the block header");

/**
 * @brief Turbulence: the energy cascade of a struck gong or tam-tam, along every voice's harmonics.
 *
 * In a plate or a shell the modes are not alone: struck hard they hand their energy on to modes
 * above them, the more the louder they are -- the bloom of a tam-tam a second after the strike
 * (Bilbao's nonlinear plates). Here every harmonic of a voice has a power that starts where its
 * spectrum puts it; each block it hands drive p^2 of it on -- the drive by the cube of the note's and the instrument's loudness -- (seven tenths to the next harmonic,
 * three to its octave), so only the loud ones cascade, and it settles back to the spectrum's own
 * shape over `settle` seconds, the higher harmonics faster. The power moves, it is not made: a soft
 * chord stays pure, a dense loud swell breaks slowly into a shimmering glow above itself without
 * growing louder.
 */
struct Turbulence {
    float on = 0.0f;        ///< 1 when any voice cascades (else nothing is computed)
    float settle = 6.0f;    ///< s: how long the harmonics take to settle back
    float pad0 = 0.0f;      ///< keeps the struct a multiple of 16 bytes
    float pad1 = 0.0f;      ///< likewise
};
static_assert(sizeof(Turbulence) % 16 == 0, "Turbulence rides in the block header");

/** @brief What lives in the field: the LifeMode parameter's choices. */
enum class LifeMode : std::int32_t {
    Off = 0,   ///< no field
    Lenia,     ///< Lenia (Chan 2019): Conway's Life made continuous -- blobs that drift, pulse, split, die
    Turing,    ///< Gray-Scott reaction-diffusion (Pearson 1993): spots, mitosis, coral, worms, mazes
    // Animated pictures (Phase 13d): KaleidoscopeEnhanced's continuous transforms, over the landscape
    // (the factory's or the player's own) or drawn from nothing.
    Kaleidoscope,   ///< the landscape folded into turning mirror wedges round a wandering centre
    Tunnel,         ///< a flight down a tunnel lined with the landscape: rings born in the middle, running out
    Spiral,         ///< the landscape wound into a log-polar spiral that zooms for ever (Droste)
    Flow,           ///< the landscape carried on a flowing noise current, like ink in water
    Feedback,       ///< video feedback: the field fed back into itself, turned and zoomed, the landscape seeping in
    Rain,           ///< drops on a pond: a damped 2D wave, rings spreading and crossing
    Starfield,      ///< stars streaming out of the middle (log-polar: they speed up as they go)
    Moire,          ///< two turning line gratings, their slow interference bands
    // Phase 16: two more fields that live
    Fluid,     ///< a fluid (lattice Boltzmann): the notes played blow plumes of ink into it, which roll up into eddies
    Islands,   ///< Cahn-Hilliard separation (Oono-Puri): a fine-grained field gathers into a few large islands that drift
    Count      ///< number of modes
};

/** @brief Whether a living landscape is one of the animated pictures (computed, not grown). @param m the mode @return true then */
#if defined(__CUDACC__)
__host__ __device__
#endif
inline bool lifeIsPicture(LifeMode m) { return m >= LifeMode::Kaleidoscope && m <= LifeMode::Moire; }

/**
 * @brief The living landscape: a field on a torus the GPU steps every block, and every partial reads.
 *
 * Where the landscapes are pictures that loop, this one lives: a Lenia field or a Gray-Scott
 * field of kLifeW x kLifeH cells, stepped up to twice a block in the graph before the partials are
 * updated. Each partial reads it at its frequency (across, log, `zoom` of the field spanning the
 * range) and at its place in its cloud (along), so every harmonic is a slice through the living
 * pattern and the pattern's motion is the partials' loudness. Made up by the field's own power
 * (`makeup`, from the GPU's measure of the last block), so a sparse field changes the colour, not
 * the level; a field that died or filled up is seeded again.
 */
struct LifeField {
    LifeMode mode = LifeMode::Off;   ///< what lives in it
    int   steps = 0;                 ///< steps this block, 0 .. 2
    int   reseed = 0;                ///< 1 for one block: seeded again first (the engine asks)
    int   salt = 0;                  ///< the seeding's dice, counted up at every seeding
    float depth = 0.0f;              ///< 0 .. 1: how far the partials' loudness follows the field
    float zoom = 1.0f;               ///< 0.25 .. 1: how much of the field's width the range spans
    float makeup = 1.0f;             ///< one over the field's RMS (the engine's, from lifeStats)
    float lenMu = 0.15f;             ///< Lenia: the growth bell's centre
    float lenSigma = 0.015f;         ///< Lenia: its width
    float feed = 0.035f;             ///< Turing: the feed rate F
    float kill = 0.065f;             ///< Turing: the kill rate k
    float time = 0.0f;               ///< pictures: how far the flow has run (Life Rate's units)
    float angle = 0.0f;              ///< pictures: how far it has turned, radians (Turn)
    float sides = 6.0f;              ///< pictures: mirror wedges, spiral arms, grating lines (Sides)
    float character = 0.3f;          ///< pictures: how strongly the transform distorts (Character)
    float seconds = 0.0f;            ///< pictures: real seconds, for the centre's slow wander
    float drops = 0.0f;              ///< Rain: the chance a cell is struck this block
    float turnStep = 0.0f;           ///< pictures: the turn this block, radians (Feedback turns by it)
    float fade = 0.999f;             ///< Fluid: the share of the ink kept each step
    float tau = 0.53f;               ///< Fluid: the lattice's relaxation time (its viscosity: (tau - 1/2) / 3)
    float push = 0.004f;             ///< Fluid: the jets' force, lattice units at full strength
    int   substeps = 1;              ///< Islands: steps of the separation this block
    float mean = 0.0f;               ///< Islands: the field's mean (0: a labyrinth; below: islands of the minority)
    float eps = 0.001f;              ///< Islands: Oono and Puri's long-range term (more: smaller domains, which stop growing)
    float drift = 0.0f;              ///< Islands: how far the field has drifted across, cells of its grid
    float pad1 = 0.0f;               ///< keeps the table 16-byte aligned
    float pad2 = 0.0f;               ///< likewise
    float pad3 = 0.0f;               ///< likewise
    float jetX[kLifeJets] = {};      ///< Fluid: where each jet blows, cells across (the notes' frequencies)
    float jetAmt[kLifeJets] = {};    ///< Fluid: how strongly, 0 .. 1 (0: none)
    float jetSway[kLifeJets] = {};   ///< Fluid: its lean, sideways share of its push
};
static_assert(sizeof(LifeField) % 16 == 0, "LifeField rides in the block header");

/** @brief How the partials' oscillators couple (PartialSync::mode). */
enum class SyncMode : std::int32_t {
    Global = 0,   ///< Kuramoto: every member to its cloud's mean
    Chimera,      ///< to its neighbours round the cloud, with a phase lag: an ordered part beside a disordered one
    Swarm,        ///< swarmalators: the phase and a place on a circle in pitch and pan pull each other
};

/**
 * @brief Partial sync: a Kuramoto oscillator for every partial, coupled to its own cloud.
 *
 * Every partial carries a phase that turns at a rate of its own (`rate`, scattered by `spread`)
 * and is pulled towards the mean phase of the members of its harmonic -- Kuramoto's mean field,
 * dθ = ω + K R sin(ψ - θ), measured by the GPU each block in a fixed order. Its loudness follows
 * the phase by `depth`. Uncoupled, the members flicker each for itself and the flicker averages
 * out in the harmonic; coupled past the critical strength they fall into step and the whole cloud
 * pulses as one -- fireflies, pacemaker cells, applause. The mean power is kept whatever happens:
 * the factor is (1 + depth sin θ) / sqrt(1 + depth^2 / 2).
 *
 * Chimera (Kuramoto and Battogtokh 2002, Abrams and Strogatz 2004): the members sit round a ring in
 * their order across the cloud and pull only their neighbours, by 1 + A cos(distance) with A
 * 0.995, through a phase lag a little under a quarter turn. From a start that is ordered in one
 * part and not in the other, the two stay side by side: a coherent pulsing core in a cloud that
 * flickers -- and in a finite cloud the border wanders. Three mean phasors a cloud carry it.
 *
 * Swarm (O'Keeffe, Hong and Strogatz 2017; on a ring, O'Keeffe, Ceron and Petrungaro 2022): every
 * partial also has a place x on a circle -- its pitch moved by reach cos x, its side by up to half
 * the field with sin x -- and place and phase pull each other: dx = v + J/2 (S+ sin(Φ+ - (x + θ))
 * + S- sin(Φ- - (x - θ))), dθ = ω + K/2 (S+ sin(Φ+ - (x + θ)) - S- sin(Φ- - (x - θ))), with
 * S± e^(iΦ±) the mean of e^(i(x ± θ)) over the cloud, in the frame turning at the rate. Its states:
 * the cloud gathered and pulsing as one; scattered and flickering; a phase wave, where each place
 * has its phase and the loudest part of the cloud circles through pitch and field at the rate; and
 * a phase wave broken into several, pulsing against each other.
 */
struct PartialSync {
    float depth = 0.0f;      ///< 0 .. 1: how far the loudness follows the phase (0: nothing computed)
    float coupling = 0.0f;   ///< K, rad/s
    float rate = 0.5f;       ///< Hz: the oscillators' middle rate
    float spread = 0.3f;     ///< 0 .. 1: each rate scattered by up to this share either way
    SyncMode mode = SyncMode::Global;   ///< how they couple
    float swarm = 0.0f;      ///< Swarm: J, rad/s (positive: like phases gather)
    float reach = 0.0f;      ///< Swarm: cents a partial is moved on its circle
    float lag = 1.39f;       ///< Chimera: the phase lag, radians
};
static_assert(sizeof(PartialSync) % 16 == 0, "PartialSync rides in the block header");

constexpr int kRippleCells = 256;   ///< the ripple's string: cells over the range, log (20 Hz .. 20 kHz)

/**
 * @brief The ripple: a damped wave along the spectrum, its displacement read by every partial at its
 *        frequency as a share of its loudness. The engine runs the string (Engine::stepRipple) and sends
 *        it up with the voices.
 */
struct Ripple {
    float depth = 0.0f;                   ///< 0 .. 1: (1 + depth tanh u) times the make-up (0: nothing read)
    float makeup = 1.0f;                  ///< one over the root of (1 + depth^2 mean tanh^2 u)
    float pad0 = 0.0f;                    ///< keeps the table 16-byte aligned
    float pad1 = 0.0f;                    ///< likewise
    float u[kRippleCells] = {};           ///< the string's displacement, cell by cell
};
static_assert(sizeof(Ripple) % 16 == 0, "Ripple rides in the block header");

constexpr int kReservoirN = 2048;      ///< the reservoir's neurons
constexpr int kReservoirInputs = 8;    ///< what it is fed
constexpr int kReservoirOutputs = 4;   ///< what is read from it

/**
 * @brief The reservoir: an echo state network (Jaeger 2001) on the GPU -- kReservoirN neurons, a
 *        fixed random coupling scaled to `chaos` (its spectral radius: below 1 it forgets, about 1
 *        it rings at the edge of chaos, above it runs on by itself), leaking at `leak` a block,
 *        driven by kReservoirInputs numbers about the music, read out by kReservoirOutputs fixed
 *        random projections. Smooth, never periodic, with a memory of what was played.
 */
struct Reservoir {
    int   on = 0;                       ///< 1: stepped this block (a matrix row reads it)
    float chaos = 0.95f;                ///< the coupling's spectral radius
    float leak = 0.001f;                ///< a block's share of the new state
    float input = 0.7f;                 ///< the inputs' gain
    float u[kReservoirInputs] = {};     ///< the inputs, about -1 .. 1
};
static_assert(sizeof(Reservoir) % 16 == 0, "Reservoir rides in the block header");

constexpr int kFieldPoints = 128;    ///< the scale's pitches a Scale field may pull to
constexpr int kFieldAttractors = 8;  ///< attractors a Swarm or Scatter field may have
constexpr int kGainRing = 1024;      ///< blocks of every voice's level the GPU keeps, for the dispersion (2.7 s at 128)

/** @brief What the field does: the FieldMode parameter's choices. */
enum class FieldMode : std::int32_t {
    Off = 0,   ///< every partial at its own place (a field switched off lets them glide home)
    Scale,     ///< every partial pulled towards the nearest pitch of the scale: the cloud collapses into the key
    Swarm,     ///< a few attractors wander through the spectrum on chaotic paths and gather the partials near them
    Scatter,   ///< the same attractors push the partials away: moving gaps, like dust parted by a wind
    Flock,     ///< every cloud's members as a flock of starlings: cohesion, alignment, separation, a wind
    Count      ///< number of modes
};

/**
 * @brief Forces on the partials' frequencies, and the dispersion of their clocks: the host's copy,
 *        sent up with the voices.
 *
 * The field. Every partial keeps an offset in cents from its own place (its harmonic, its member,
 * its drift and pitch lane), carried from block to block on the GPU. Each block it moves towards
 * what pulls it -- the nearest pitch of the scale, or the attractors near it, weighted by a bell
 * of `radius` cents -- by at most `strength` cents a second and never past the attractor, and
 * the spring draws it back towards its own place. So a cloud does not snap to a grid: it is
 * gathered, it gives way, it is let go, at the speeds the knobs say. The mean level does not
 * change: partials that meet keep their random phases, so their powers add as before.
 *
 * The dispersion. Each partial's clock runs `dispersion` seconds late at the bottom of the range
 * (20 Hz) and on time at the top when it is positive, the other way round when negative -- its
 * envelopes, the landscape it reads and the voice's level, which the GPU keeps a history of. An
 * attack or a glint then reaches the ear as a cascade, the highs first and the lows after (a
 * falling chirp, as in a dispersive medium -- a long spring, a stiff string, ice), or rising.
 */
struct SpectralField {
    FieldMode mode = FieldMode::Off;   ///< what pulls the partials
    int   count = 0;                   ///< Scale: pitches in `pos`; Swarm, Scatter: attractors in `pos` and `mass`; Flock: winds in `pos`
    int   flockFresh = 0;              ///< Flock: 1 when the last block was no flock (its state is taken from where the partials are)
    int   pad1 = 0;                    ///< likewise
    float strength = 0.0f;             ///< cents a second a partial may move at most
    float radius = 60.0f;              ///< cents: an attractor's reach (the bell's width); the Scale field's stiffness
    float spring = 0.3f;               ///< per second: how fast a partial is drawn back to its own place
    float dispersion = 0.0f;           ///< seconds: the clock of the lowest partials late (positive) or of the highest (negative)
    float pos[kFieldPoints] = {};      ///< log2 Hz: the scale's pitches ascending (Scale), or the attractors (Swarm, Scatter)
    float mass[kFieldAttractors] = {}; ///< the attractors' weights; Scatter's are negative
};
static_assert(sizeof(SpectralField) % 16 == 0, "SpectralField rides in the block header");

/** @brief How an engine is laid out; fixed between open() and close(). */
struct CloudConfig {
    double sampleRate = 48000.0;  ///< Hz
    int    blockSize = 128;       ///< samples per launch(), a power of two, 32 .. 4096
    int    voices = 16;           ///< voices the GPU tables hold
    int    partialsPerVoice = 20000;   ///< the most partials one voice can render
    bool   spinWait = true;       ///< wait() spins on the GPU's event (lowest latency) instead of sleeping on it
};

/**
 * @brief The GPU side: tables of partial state, the kernels and the graph that runs them.
 *
 * Use: open(), setShapes() (and setLandscape()), then per block edit voice(i), launch(), wait()
 * (or poll ready()). Not thread-safe: one thread drives it.
 */
class CloudEngine {
public:
    CloudEngine();
    ~CloudEngine();
    CloudEngine(const CloudEngine&) = delete;              ///< owns GPU memory: not copyable
    CloudEngine& operator=(const CloudEngine&) = delete;   ///< owns GPU memory: not copyable

    /**
     * @brief Picks the first CUDA device, allocates the tables and builds the graph.
     * @param config  the layout
     * @param error   receives the reason when it fails (may be null)
     * @return        true when the engine can render
     */
    bool open(const CloudConfig& config, std::string* error);

    /** @brief Frees everything; open() may be called again. */
    void close();

    /** @brief The layout open() was given. @return the config */
    const CloudConfig& config() const;

    /**
     * @brief Replaces the bank of envelope shapes; waits for a block in flight first.
     * @param shapes  the shapes
     * @param count   how many, at most kEnvShapes (more are dropped)
     * @return        true when uploaded
     */
    bool setShapes(const EnvShape* shapes, int count);

    /**
     * @brief Replaces the landscape: rows of time, each `columns` amplitudes from kLandLowHz to
     *        kLandHighHz; it loops over all its rows. Waits for a block in flight first.
     * @param amps        rows * columns amplitudes 0 .. 1, row by row
     * @param columns     2 .. kLandMaxColumns
     * @param rows        1 .. kLandMaxRows
     * @param rowSeconds  how long one row lasts at rate 1
     * @return            true when uploaded (false: outside the limits, nothing changed)
     */
    bool setLandscape(const float* amps, int columns, int rows, float rowSeconds);

    /**
     * @brief The host's copy of a voice, edited freely between wait() and the next launch().
     * @param i  0 .. voices-1
     * @return   the voice
     */
    CloudVoice& voice(int i);

    /** @brief The host's copy of the modal hall's settings, edited between wait() and the next launch(). @return the settings */
    ModalHall& modal();

    /** @brief The host's copy of the field and the dispersion, edited between wait() and the next launch(). @return the settings */
    SpectralField& field();

    /** @brief The host's copy of the binaural placement, edited between wait() and the next launch(). @return the settings */
    Binaural& binaural();

    /** @brief The host's copy of the spectral echo's settings, edited between wait() and the next launch(). @return the settings */
    SpectralEcho& spectralEcho();

    /** @brief The host's copy of the living landscape's settings, edited between wait() and the next launch(). @return the settings */
    LifeField& life();

    /** @brief The host's copy of the partial sync's settings, edited between wait() and the next launch(). @return the settings */
    PartialSync& sync();

    /** @brief The host's copy of the ripple, edited between wait() and the next launch(). @return the ripple */
    Ripple& ripple();

    /** @brief The host's copy of the reservoir's settings, edited between wait() and the next launch(). @return the settings */
    Reservoir& reservoir();

    /** @brief The host's copy of the melt's settings, edited between wait() and the next launch(). @return the settings */
    Melt& melt();

    /** @brief The host's copy of the spectral comb, edited between wait() and the next launch(). @return the settings */
    SpectralComb& comb();
    /** @brief The energy cascade's settings for the next block (the voices carry their drives). @return them */
    Turbulence& turbulence();

    /** @brief The reservoir's outputs after the last block, -1 .. 1. @param out receives kReservoirOutputs values */
    void reservoirOut(float* out) const;

    /**
     * @brief The living field's mean and mean square (of what the partials read, 0 .. 1), measured by
     *        the GPU in the last block.
     * @param mean    receives the mean
     * @param meanSq  receives the mean of the squares
     */
    void lifeStats(float& mean, float& meanSq) const;

    /**
     * @brief What the instrument heard of its own voices in the last block (an FFT of the last 1024
     *        samples, near and far together): its level, where its energy stands, how much it changed.
     * @param levelDb   receives the level, dB of full scale
     * @param centroid  receives the energy's centre on the log axis, 0 at 20 Hz .. 1 at 20 kHz
     * @param flux      receives the change: the rise of the spectrum's magnitudes over their sum, 0 .. 1
     */
    void heard(float& levelDb, float& centroid, float& flux) const;

    /**
     * @brief The living field as the last snapshot brought it down: kLifeW x kLifeH cells, each the value
     *        the partials read (0 .. 1), row after row. Valid after takeSnapshot() returned true.
     * @return  the cells, or null when the field is off
     */
    const float* lifePicture() const;

    /** @brief The next launch() also brings the partials' snapshot down (every kSnapStride-th partial's frequency and amplitude). */
    void requestSnapshot();

    /**
     * @brief The snapshot a launch brought down, once: after the wait() of a launch that copied it.
     * @param dots      receives the points, per voice snapshotPerVoice() of (Hz, amplitude), voices in order
     * @param perVoice  receives how many points each voice has
     * @return          true when a new snapshot is there (then valid until the next launch())
     */
    bool takeSnapshot(const float*& dots, int& perVoice);

    /**
     * @brief Sends the voices up and starts one block on the GPU; returns at once. Clears every
     *        voice's restart flag and advances the active voices' time by one block.
     */
    void launch();

    /** @brief Whether the launched block is done (never blocks). @return true when wait() would return at once */
    bool ready() const;

    /**
     * @brief Waits for the launched block.
     * @return  blockSize frames of kCloudOutChannels (near L, R, far L, R, modal hall L, R, spectral echo L, R, melt L, R), valid until the next launch()
     */
    const float* wait();

    /** @brief GPU time of the last finished block, from the events around the graph. @return milliseconds */
    float lastGpuMs() const;

    /** @brief The device's name and compute capability, for the logs. @return e.g. "NVIDIA GeForce RTX 5090 (sm_120)" */
    std::string deviceName() const;

private:
    struct Impl;                    ///< the CUDA side, defined in Cloud.cu
    std::unique_ptr<Impl> impl_;    ///< null while closed
};

} // namespace additiv
