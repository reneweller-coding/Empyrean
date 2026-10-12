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
 * stereo block (reduce kernel). The three kernels and the two copies around them are one CUDA
 * graph, so a block costs one launch.
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
    // Filled by the engine at launch(), read by the GPU.
    int   harmonics = 0;          ///< harmonics below the cutoff, at most `partials`
    int   members = 0;            ///< partials per harmonic: partials / harmonics
    float scale = 0.0f;           ///< amplitude of harmonic 1's every member, so that the voice meets `level`
};

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

    /**
     * @brief Sends the voices up and starts one block on the GPU; returns at once. Clears every
     *        voice's restart flag and advances the active voices' time by one block.
     */
    void launch();

    /** @brief Whether the launched block is done (never blocks). @return true when wait() would return at once */
    bool ready() const;

    /**
     * @brief Waits for the launched block.
     * @return  blockSize interleaved stereo frames, valid until the next launch()
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
