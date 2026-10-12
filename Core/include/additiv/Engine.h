/**
 * @file Engine.h
 * @brief The instrument: voices, the conductor, the tuning, the GPU cloud, the far reverb and the
 *        master -- rendered ahead of the audio callback by a worker thread.
 *
 * The audio callback never waits for the GPU (PLAN.md, measured in Phase 0). A worker renders the
 * timeline in GPU blocks (128 samples) into a ring and keeps it filled `latencySamples()` ahead of
 * what the callback has taken; the callback only stamps its notes and copies samples out. A note
 * played at callback offset o is stamped o + latency ahead and starts on exactly that sample
 * inside its GPU block (CloudVoice::onset), so everything the host hears is delayed by the same
 * reported latency and nothing jitters. Offline (a DAW's bounce, the renderer, the tests) the same
 * timeline is rendered synchronously from process() instead: same stamps, same samples.
 *
 * Per GPU block the renderer: applies the notes that are due, lets the conductor (Noctuary's
 * ClusterBrain, Free or Chords = Autoplay) decide, advances every voice's envelope, fills the
 * cloud voices (frequency from the scale, the cloud parameters, the motion, the distance), runs
 * the GPU, sends the far plane through the far reverb, adds the near plane, and finishes with
 * Mid/Side, the master gain and the soft clip -- all as in Noctuary's signal path, with the
 * sources replaced by the cloud.
 *
 * Threads: setParam() from any thread; noteOn(), noteOff(), allNotesOff() and process() from the
 * audio thread; prepare(), release() and setRealtime() from one control thread while process() is
 * not running.
 */
#pragma once

#include "additiv/Cloud.h"
#include "additiv/ClusterBrain.h"
#include "additiv/Dsp.h"
#include "additiv/Effects.h"
#include "additiv/Landscape.h"
#include "additiv/Modulation.h"
#include "additiv/Morph.h"
#include "additiv/Params.h"
#include "additiv/Tuning.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace additiv {

/** @brief How an engine is laid out; fixed between prepare() and release(). */
struct EngineConfig {
    double sampleRate = 48000.0;      ///< the host's rate; the GPU renders at it
    int    maxHostBlock = 1024;       ///< the largest block process() is asked for
    int    gpuBlock = 128;            ///< samples per GPU block, a power of two
    int    voices = 16;               ///< voices
    int    ghosts = 4;                ///< memory voices beside them: the ghosts of notes let go (MemoryLevel)
    int    partialsPerVoice = 20000;  ///< the most partials a voice can have
    float  lookaheadMs = 10.0f;       ///< how far beyond two host blocks the worker renders ahead
    bool   realtimeGpuClass = true;   ///< raise the process's GPU scheduling class to REALTIME (PLAN.md)
    std::uint64_t seed = 0x5EEDu;     ///< the conductor's and the notes' dice
};

/** @brief What a picture of the instrument needs to know about one voice (Engine::voiceView). */
struct VoiceView {
    float hz = 0.0f;         ///< the frequency it sounds at, 0 when silent
    float gain = 0.0f;       ///< its envelope, 0 .. 1
    float distance = 0.0f;   ///< where it stands, 0 at the ear .. 1 far
    float time = 0.0f;       ///< seconds since it began (the motions' clock)
    int source = 0;          ///< 0 played, 1 the conductor, 2 the second conductor, 3 a ghost of the memory
};

constexpr int kMaxViewVoices = 32;   ///< voices a picture can be given

/** @brief The instrument (see the file comment). */
class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;              ///< owns the GPU and a thread
    Engine& operator=(const Engine&) = delete;   ///< owns the GPU and a thread

    /**
     * @brief Opens the GPU, builds everything and, in real-time mode, starts the worker.
     * @param config  the layout
     * @param error   receives the reason when it fails (may be null)
     * @return        true when the engine can play
     */
    bool prepare(const EngineConfig& config, std::string* error);

    /** @brief Stops the worker and frees the GPU; prepare() may be called again. */
    void release();

    /** @brief Whether prepare() succeeded. @return true when process() makes sound */
    bool prepared() const { return prepared_; }

    /**
     * @brief Real time (a worker renders ahead) or offline (process() renders what it needs itself).
     * @param realtime  true for live playing, false for a bounce, the renderer and the tests
     */
    void setRealtime(bool realtime);

    /** @brief The mode setRealtime() chose. @return true in real-time mode */
    bool realtime() const { return realtime_; }

    /** @brief How many samples everything is delayed by (the host compensates). @return samples */
    int latencySamples() const { return latency_; }

    /** @brief The layout prepare() was given. @return the config */
    const EngineConfig& config() const { return cfg_; }

    /** @brief The GPU, for the logs. @return its name and compute capability ("" before prepare) */
    std::string deviceName() const;

    /**
     * @brief Sets a parameter, clamped to its range. Any thread.
     * @param id     the parameter
     * @param value  in its unit
     */
    void setParam(ParamId id, float value);

    /** @brief A parameter's value. @param id the parameter @return in its unit */
    float param(ParamId id) const { return params_[static_cast<size_t>(id)].load(std::memory_order_relaxed); }

    /** @brief Every parameter to its default. */
    void resetParams();

    /**
     * @brief A played note begins. Audio thread, before the process() it belongs to.
     * @param note      MIDI note
     * @param velocity  0 .. 1
     * @param offset    samples into the coming process() block
     */
    void noteOn(int note, float velocity, int offset);

    /**
     * @brief A played note is let go. Audio thread.
     * @param note    MIDI note
     * @param offset  samples into the coming process() block
     */
    void noteOff(int note, int offset);

    /** @brief Every played note is let go (the conductor's notes stay). Audio thread. */
    void allNotesOff();

    /**
     * @brief Sets one row of the modulation matrix. Any thread (a row is one atomic word).
     * @param i    0 .. kModRows-1
     * @param row  the row
     */
    void setModRow(int i, const ModRow& row);
    /** @brief One row of the matrix. @param i 0 .. kModRows-1 @return the row */
    ModRow modRow(int i) const;
    /** @brief The matrix as text (Modulation.h). @return the text */
    std::string modText() const;
    /** @brief The matrix from text; rows it does not name are emptied. @param text the text */
    void setModText(const std::string& text);

    /**
     * @name Morph (Morph.h)
     * A glide: from the next block the engine travels, over `seconds`, from what it played the
     * block before to whatever the parameters and the matrix are then -- so a host asks for the
     * glide and then sets the new preset (the knobs show where the sound is going; a knob turned on
     * the way changes where it goes). The A/B morph: with Morph on and both slots chosen, the sound
     * is the blend of the slots at A - B instead of the knobs; switching it on or off glides
     * (Glide seconds, at least one).
     * @{ */
    /**
     * @brief The player's own landscape, played when the Landscape choice is Your Own (until one is
     *        set, that choice plays Aurora). Message thread; the worker uploads it between blocks.
     *        Rows beyond the GPU's room are averaged down, the loop kept.
     * @param l  the landscape (a picture's or a recording's, Landscape.h)
     */
    void setUserLandscape(const Landscape& l);
    /** @brief Whether the player's own landscape has been set. Any thread. @return true then */
    bool hasUserLandscape() const { return userLandVersion_.load(std::memory_order_relaxed) > 0; }
    /** @brief Glide from what plays now to what is set next. Any thread. @param seconds how long (at least 0.05) */
    void beginMorph(double seconds);
    /** @brief How far the glide has come. Any thread. @return 0 .. 1; 1 when none runs */
    float morphProgress() const { return morphProgress_.load(std::memory_order_relaxed); }
    /**
     * @brief Chooses a slot of the A/B morph. Message thread.
     * @param slot     0 for A, 1 for B
     * @param values   kNumParams values
     * @param modText  its matrix (Modulation.h)
     */
    void setMorphSlot(int slot, const float* values, const std::string& modText);
    /** @brief Whether a slot has been chosen. Any thread. @param slot 0 or 1 @return true then */
    bool morphSlotSet(int slot) const { return slotSet_[static_cast<size_t>(slot & 1)].load(std::memory_order_relaxed); }
    /**
     * @brief A parameter where the sound stands, before the matrix: the knob's, or the morph's or
     *        the glide's way between. Any thread.
     * @param id  the parameter
     * @return    its value in its own units
     */
    float soundValue(ParamId id) const { return soundLive_[static_cast<size_t>(id)].load(std::memory_order_relaxed); }
    /** @brief The voice filter's share where the sound stands (Morph.h). Any thread. @return 0 .. 1 */
    float soundFilterMix() const { return soundMix_.load(std::memory_order_relaxed); }
    /**
     * @brief Loads a factory preset: every parameter but the listener's (keptByPresets) and the
     *        matrix, gliding into it when `glideSeconds` is above 0 (a host with its own parameters
     *        does this itself; the render tool and the tests use this).
     * @param index         the preset
     * @param glideSeconds  0: at once
     */
    void loadPreset(int index, double glideSeconds = 0.0);
    /** @} */
    /** @brief The mod wheel (CC 1). Any thread. @param v 0 .. 1 */
    void setModWheel(float v) { wheel_.store(v, std::memory_order_relaxed); }
    /** @brief Channel pressure. Any thread. @param v 0 .. 1 */
    void setPressure(float v) { pressure_.store(v, std::memory_order_relaxed); }
    /**
     * @brief Where a parameter plays after the matrix, normalised, for the panel's live rings.
     * @param id  the parameter
     * @return    0 .. 1, NaN when nothing modulates it
     */
    float playedNorm(ParamId id) const { return played_[static_cast<size_t>(id)].load(std::memory_order_relaxed); }

    /**
     * @brief A modulation source's value as the last block left it, for the panel. Any thread.
     * @param s  the source
     * @return   its value (-1 .. 1, or 0 .. 1 for a unipolar one)
     */
    float sourceValue(ModSource s) const
    {
        const int i = static_cast<int>(s);
        return i > 0 && i < kNumModSources ? sourceLive_[static_cast<size_t>(i)].load(std::memory_order_relaxed) : 0.0f;
    }

    /**
     * @brief The panel shows the sources: what is computed only for a row that reads it (the
     *        reservoir) runs while they are watched. Any thread.
     * @param on  watched
     */
    void setSourcesWatched(bool on) { sourcesWatched_.store(on, std::memory_order_relaxed); }

    /**
     * @brief The family jam, following: the leader's root, which the conductor's root is held on (its
     *        nearest note of that pitch class), as Noctuary follows. Any thread.
     * @param pitchClass  0 .. 11, -1 when no leader is heard
     */
    void setJamRoot(int pitchClass) { jamRoot_.store(pitchClass, std::memory_order_relaxed); }
    /**
     * @brief The family jam, following: the energy of the leader's section, which opens or closes the
     *        brightness and the filter (0.5 changes nothing). Any thread.
     * @param energy  0 .. 1
     */
    void setJamEnergy(float energy) { jamEnergy_.store(energy, std::memory_order_relaxed); }

    /** @brief Autoplay: exchange a voice of the cluster at the next block, whatever its timer says. Any thread. */
    void autoplayStep() { stepAsked_.store(true, std::memory_order_relaxed); }
    /**
     * @brief Both conductors fill their chords at speed (a note every third of a second) until they
     *        are full, as Noctuary after a preset change: the arriving preset is heard whole within
     *        seconds, not after minutes of its own event rate. Any thread.
     */
    void requestFill() { fillAsked_.store(true, std::memory_order_relaxed); }

    /**
     * @brief The audio callback: n samples of the timeline.
     * @param L  left output
     * @param R  right output
     * @param n  samples, at most config().maxHostBlock in real-time mode
     */
    void process(float* L, float* R, int n);

    /** @brief Voices sounding after the last rendered block. @return 0 .. voices */
    int soundingVoices() const { return sounding_.load(std::memory_order_relaxed); }

    /** @brief Voices sounding after the last rendered block, by whose notes they play. @param source 0 played, 1 the conductor, 2 the second conductor @return count */
    int soundingFrom(int source) const { return source >= 0 && source < 3 ? soundingBy_[source].load(std::memory_order_relaxed) : 0; }

    /** @brief The conductor's root after the last rendered block. @return MIDI note */
    int brainRoot() const { return brainRoot_.load(std::memory_order_relaxed); }

    /** @brief Callbacks that found the ring short (real time only). @return count since prepare() */
    std::uint64_t underruns() const { return underruns_.load(std::memory_order_relaxed); }

    /** @brief Notes that arrived after their stamp had been rendered. @return count since prepare() */
    std::uint64_t lateEvents() const { return late_.load(std::memory_order_relaxed); }

    /**
     * @brief A voice as the last rendered block left it, for pictures (any thread; the fields may come from two
     *        neighbouring blocks, which a picture does not mind).
     * @param i  the voice, 0 .. config().voices-1
     * @return   its view; hz 0 when it is silent
     */
    VoiceView voiceView(int i) const
    {
        VoiceView v;
        if (i < 0 || i >= kMaxViewVoices) return v;
        v.hz = viewHz_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
        v.gain = viewGain_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
        v.distance = viewDistance_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
        v.time = viewTime_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
        v.source = viewSource_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
        return v;
    }

    /**
     * @brief The partials as the picture draws them: every kSnapStride-th partial of every voice (and
     *        ghost), as (Hz, amplitude), brought from the GPU about thirty times a second. Message thread.
     * @param dots      receives the points, voice after voice, perVoice each
     * @param perVoice  receives how many points each voice has
     * @return          true when there was a new snapshot since the last call
     */
    bool partialSnapshot(std::vector<float>& dots, int& perVoice);
    /**
     * @brief The living field as the last snapshot brought it (kLifeW x kLifeH cells, 0 .. 1). Message thread.
     * @param cells  receives the cells, emptied when the field is off
     */
    void lifeSnapshot(std::vector<float>& cells);

    /** @brief The slowest block the renderer took since the last call, and starts over (diagnostics). @return milliseconds */
    float takeWorstBlockMs() { return worstBlockUs_.exchange(0, std::memory_order_relaxed) * 0.001f; }
    /**
     * @brief The slowest of each part of a block since the last call, and starts over (diagnostics): the
     *        preparation before the GPU is launched, the wait for it, the effects after.
     * @param prepMs    receives the slowest preparation, milliseconds
     * @param gpuMs     receives the longest wait for the GPU
     * @param effectsMs receives the slowest effects
     */
    void takeWorstParts(float& prepMs, float& gpuMs, float& effectsMs)
    {
        prepMs = worstPrepUs_.exchange(0, std::memory_order_relaxed) * 0.001f;
        gpuMs = worstWaitUs_.exchange(0, std::memory_order_relaxed) * 0.001f;
        effectsMs = worstPostUs_.exchange(0, std::memory_order_relaxed) * 0.001f;
    }
    /** @brief The emptiest the ring was when a callback came, since the last call, and starts over. @return samples, -1 if no callback came */
    int takeMinFill() { return minFill_.exchange(-1, std::memory_order_relaxed); }
    /** @brief The largest block a callback asked for since the last call, and starts over. @return samples */
    int takeMaxCallback() { return maxCallback_.exchange(0, std::memory_order_relaxed); }

    /** @brief GPU time of the last block. @return milliseconds */
    float lastGpuMs() const { return gpuMs_.load(std::memory_order_relaxed); }

private:
    /** @brief A note event on its way from the audio thread to the renderer. */
    struct Event {
        int type = 0;              ///< 0 note on, 1 note off, 2 all played notes off
        int note = 0;              ///< MIDI note
        float velocity = 0.0f;     ///< 0 .. 1 (note on)
        std::int64_t stamp = 0;    ///< the timeline sample it takes effect at
    };

    /** @brief One voice on the host side: the note, its envelope, its place, and what the GPU voice is told. */
    struct Voice {
        int note = -1;             ///< MIDI note, -1 when free
        int source = 0;            ///< 0 played, 1 the conductor, 2 the second conductor
        float velocity = 0.0f;     ///< 0 .. 1
        float distance = 0.0f;     ///< where it stands now, 0 at the ear .. 1 far (its place plus the breath)
        float place = 0.0f;        ///< where it was put
        float prevDistance = 0.0f; ///< the distance of the block before, for the doppler
        Drifter breath;            ///< the slow wander of its distance (Breath)
        Drifter cutoffDrift;       ///< the slow wander of its filter's cutoff (Filter Drift)
        double hz = 0.0;           ///< the frequency it is gliding at (purity and root changes glide)
        Envelope env;              ///< the amplitude envelope, stepped once per GPU block
        std::uint32_t seed = 1;    ///< the note's dice on the GPU
        bool restart = false;      ///< the GPU voice begins this note at the next block
        int onset = 0;             ///< and at this sample of it
        bool released = false;     ///< noteOff has come
        int tailBlocks = 0;        ///< blocks it stays active after the envelope fell silent (ramps to zero)
        bool tailArmed = false;    ///< the tail was lengthened by the dispersion when the envelope fell silent
        float tuneCents = 0.0f;    ///< self-tuning: how far the voice has moved from its pitch in the scale
        float lastGain = 0.0f;     ///< its envelope in the last block (self-tuning weighs by it)
        double age = 0.0;          ///< seconds since it began, for stealing
        bool stealing = false;     ///< fading out before `next` begins
        int stealBlocks = 0;       ///< blocks of fading still to go
        Event next;                ///< the note waiting for a stolen voice (type 0; -1 = cancelled)
        int nextSource = 0;        ///< whose note that is
        float nextDistance = 0.0f; ///< and where it will stand
    };

    void workerLoop();                        ///< the worker: renders while the ring is short of the latency
    void startWorker();                       ///< starts the worker (real-time mode)
    void stopWorker();                        ///< stops and joins it
    void renderBlock();                       ///< one GPU block of the timeline into the ring (renderer thread)
    void readParams();                        ///< the parameters the block uses, read once per block, the modulation applied
    /** @brief A parameter's effective value this block (after the matrix). @param id the parameter @return the value */
    float cur(ParamId id) const { return cur_[static_cast<size_t>(id)]; }
    void drainEvents();                       ///< moves the queued events into the pending list
    /**
     * @brief Starts a note on a free voice, or steals one (fading it over a block first).
     * @param note       MIDI note
     * @param velocity   0 .. 1
     * @param distance   0 at the ear .. 1 far
     * @param source     0 played, 1 the conductor, 2 the second conductor
     * @param onset      sample of the block it begins at
     */
    void startNote(int note, float velocity, float distance, int source, int onset);
    /**
     * @brief Begins a note on a voice now (the voice is free or has just faded).
     * @param i          the voice
     * @param note       MIDI note
     * @param velocity   0 .. 1
     * @param distance   0 .. 1
     * @param source     0 played, 1 the conductor, 2 the second conductor
     * @param onset      sample of the block it begins at
     */
    void beginOn(int i, int note, float velocity, float distance, int source, int onset);
    /**
     * @brief Lets a note go.
     * @param note    MIDI note
     * @param source  whose note: 0 played, 1 the conductor, 2 the second conductor
     */
    void stopNote(int note, int source);
    /** @brief Where a conductor's note stands: Noctuary's two planes, 40 % close and 60 % deep. @return 0 .. 1 */
    float brainDistance();
    /** @brief Loads the landscape the Landscape parameter names when it changed, and its RMS. */
    void refreshLandscape();
    /** @brief Fills the conductors' spectrum from the voice's shape, and rebuilds the Timbre scale when it changed. */
    void refreshSpectrum();
    /** @brief A note's frequency: the scale blended with 12-TET by Purity (Noctuary's frequencyOf). @param note MIDI note @return Hz */
    double frequencyOf(int note) const;
    /** @brief Rebuilds the scale when its choice, root or key mapping changed. */
    void refreshScale();

    EngineConfig cfg_;                                 ///< the layout
    bool prepared_ = false;                            ///< prepare() succeeded
    bool realtime_ = true;                             ///< worker mode
    int latency_ = 0;                                  ///< samples the timeline runs ahead
    int startReserve_ = 0;                             ///< samples rendered beyond it before the first callback, used up over as many
    /**
     * @brief Where an event from the host lands on the timeline: the latency ahead of what the
     *        callback has taken, and while the start reserve is not used up, its rest too (what is
     *        rendered already reaches that far; an event there would come late).
     * @param offset  samples into the host's block
     * @return        the event's sample on the timeline
     */
    std::int64_t stampFor(int offset) const
    {
        const std::int64_t out = outPos_.load(std::memory_order_relaxed);
        return out + std::max(offset, 0) + latency_ + std::max<std::int64_t>(0, startReserve_ - out);
    }
    std::array<std::atomic<float>, kNumParams> params_; ///< the parameters, any thread

    CloudEngine cloud_;                                ///< the GPU
    std::vector<Voice> voices_;                        ///< the host's voices
    ClusterBrain brain_;                               ///< the conductor
    BrainParams bp_;                                   ///< its knobs, filled per block
    ClusterBrain brain2_;                              ///< the second conductor, in the background (Noctuary's Brain 2)
    BrainParams bp2_;                                  ///< its knobs
    BrainSpectrum brainSpec_;                          ///< the cloud's spectrum for its Timbre judgement
    float specKey_[5] = { -1.0f, 0.0f, 0.0f, 0.0f, 1.0f };   ///< tilt, odd/even, brightness, inharmonic, stretch the spectrum was built for
    Rng rng_;                                          ///< the engine's dice (seeds, distances)
    Rng side_;                                         ///< a side stream for the drifters, so switching them on never moves the notes' dice
    Drifter tide_;                                     ///< the whole instrument's slow lean (Tide)
    int landChoice_ = -1;                              ///< the landscape on the GPU
    Landscape userLand_;                               ///< the player's own, as setUserLandscape left it (under userLandLock_)
    float userLandRms_ = 1.0f;                         ///< its RMS
    std::mutex userLandLock_;                          ///< guards userLand_ (the worker only tries it)
    std::atomic<int> userLandVersion_{ 0 };            ///< bumped by setUserLandscape
    int userLandSeen_ = 0;                             ///< the version on the GPU
    float landRms_ = 1.0f;                             ///< its RMS, for the level of voices that read it
    bool stepWas_ = false;                             ///< the Step switch's last value (it fires on the rising edge)
    Modulators mods_;                                  ///< the modulation sources
    // The morph (Morph.h). The worker owns sound_, from_, target_ and slots_; the editor hands
    // slots over through pendingSlots_ under slotLock_, which the worker only tries.
    /** @brief What plays this block before the matrix: the knobs, the A/B morph, or a glide between. */
    void updateSound();
    /** @brief A parameter where the sound stands this block (the worker's). @param id the parameter @return its value */
    float base(ParamId id) const { return sound_.values[static_cast<size_t>(id)]; }
    MorphState sound_;                                 ///< what plays this block, before the matrix
    MorphState from_;                                  ///< what played when the glide began
    MorphState target_;                                ///< where the glide goes: the knobs, or the A/B blend
    MorphState slots_[2];                              ///< the A/B morph's slots, as the worker has them
    MorphState pendingSlots_[2];                       ///< the slots as the editor last set them
    std::mutex slotLock_;                              ///< guards pendingSlots_ (the worker only tries it)
    std::atomic<int> slotVersion_{ 0 };                ///< bumped by setMorphSlot
    int slotSeen_ = 0;                                 ///< the version the worker has
    std::array<std::atomic<bool>, 2> slotSet_{};       ///< a slot has been chosen
    std::atomic<int> morphAsked_{ 0 };                 ///< bumped by beginMorph
    int morphSeen_ = 0;                                ///< the request the worker has served
    std::atomic<float> morphSeconds_{ 1.0f };          ///< the glide asked for
    float morphT_ = 1.0f;                              ///< 0 .. 1 of the glide's time; 1 when none runs
    float morphLen_ = 1.0f;                            ///< the glide's seconds
    bool abWas_ = false;                               ///< the A/B morph played the last block
    bool soundValid_ = false;                          ///< sound_ holds a block (else a glide has nothing to start from)
    std::atomic<float> morphProgress_{ 1.0f };         ///< morphT_, for any thread
    std::array<std::atomic<float>, kNumParams> soundLive_{};   ///< sound_'s values, for any thread
    std::atomic<float> soundMix_{ 1.0f };              ///< sound_'s filter share, for any thread
    std::array<std::atomic<std::uint64_t>, kModRows> modRows_{};   ///< the matrix, a packed row per word (packModRow)
    std::array<float, kNumParams> cur_{};              ///< every parameter's effective value this block
    std::array<std::atomic<float>, kNumModSources> sourceLive_{};   ///< every source's value after the last block, for the panel
    std::atomic<bool> sourcesWatched_{ false };        ///< the panel shows the sources (setSourcesWatched)
    std::array<std::atomic<float>, kNumParams> played_{};   ///< where each plays after the matrix, normalised (NaN: unmodulated)
    std::atomic<float> wheel_{ 0.0f };                 ///< the mod wheel
    std::atomic<float> pressure_{ 0.0f };              ///< channel pressure
    float lastVelocity_ = 0.0f;                        ///< the last note's velocity
    std::atomic<int> jamRoot_{ -1 };                   ///< the jam leader's root (pitch class), -1 none
    std::atomic<float> jamEnergy_{ 0.5f };             ///< the jam leader's section energy
    bool phraseStart_ = false;                         ///< a note arrived into silence: the envelopes start over
    int prevSounding_ = 0;                             ///< voices sounding after the block before
    FixedScale scale_;                                 ///< the current scale
    int scaleChoice_ = -1;                             ///< the choice scale_ was built from
    int rootPc_ = 0;                                   ///< the scale's root, a pitch class
    bool snapKeys_ = true;                             ///< key mapping
    Reverb far_;                                       ///< the far plane's reverb (Noctuary's FDN)
    Diffuser diffuser_;                                ///< the swell in front of the far reverb
    Unmask unmask_;                                    ///< the far hall stepping aside for the near plane
    StereoDelay delay_;                                ///< the first echo (Noctuary's)
    StereoDelay delay2_;                               ///< the second, in series after the first
    EarlyRoom early_;                                  ///< the near plane's early reflections
    Reverb nearRev_;                                   ///< the near plane's small reverb
    MidSide midSide_;                                  ///< bass mono and width
    std::vector<float> nearL_;                         ///< one GPU block's near plane, left (then the whole mix)
    std::vector<float> nearR_;                         ///< its near plane, right (then the whole mix)
    std::vector<float> farL_;                          ///< its far plane, left (then the far reverb)
    std::vector<float> farR_;                          ///< its far plane, right
    std::vector<float> dryL_;                          ///< the near plane before the echoes, left: the early reflections' input
    std::vector<float> dryR_;                          ///< and right
    std::vector<float> wetL_;                          ///< an echo's output, left
    std::vector<float> wetR_;                          ///< and right
    std::vector<float> modalL_;                        ///< the modal hall's block from the GPU, left
    std::vector<float> modalR_;                        ///< and right
    float prevDelayMix_ = 0.0f;                        ///< the first echo's level at the end of the last block (ramped across the next)
    float prevDelayToFar_ = 0.0f;                      ///< its send into the far plane, likewise
    float prevDelay2Mix_ = 0.0f;                       ///< the second echo's level, likewise
    float prevDelay2ToFar_ = 0.0f;                     ///< its send into the far plane, likewise
    float prevModalLevel_ = 0.0f;                      ///< the modal hall's level, likewise
    float modalTime_ = 0.0f;                           ///< the modal hall's drift clock, seconds, wrapped at 1000
    float srcDistance_ = 0.3f;                         ///< where the near room hears the voices: their mean distance, weighted by loudness
    double lorenz_[kFieldAttractors][3] = {};          ///< the field's attractors: eight Lorenz systems (x, y, z), wandering always
    bool flockWas_ = false;                            ///< the last block's field was the flock (else its state is taken afresh)
    float resMean_[kReservoirInputs] = {};             ///< the reservoir's inputs: their running means
    float resVar_[kReservoirInputs] = {};              ///< and running variances
    bool resStarted_ = false;                          ///< the running statistics have their first values
    float rippleNow_[kRippleCells] = {};               ///< the ripple's string now
    float ripplePrev_[kRippleCells] = {};              ///< and a step before (the wave equation's leapfrog)
    float rippleStrikes_[64] = {};                     ///< the fundamentals of the notes that started this block
    int rippleStrikeCount_ = 0;                        ///< how many
    /**
     * @brief One block of the ripple's string: struck at the harmonics of every note that started this
     *        block (rippleStrikes_, gathered by the voice loop, which clears the restart flags), then the damped wave equation stepped (leapfrog, reflecting ends, as many substeps
     *        as the speed needs to stay stable), and the make-up worked out.
     * @param dt  the block's length, seconds
     */
    void stepRipple(float dt);

    /** @brief A ghost of the memory: a voice as it was when it was let go, sounding on, fading, blurring. */
    struct Ghost {
        bool on = false;           ///< sounding
        bool restart = true;       ///< its first block on the GPU is next
        CloudVoice base;           ///< the voice as it was when it was let go
        std::uint32_t seed = 1;    ///< its own dice (new phases: it does not add to the voice it came from)
        float age = 0.0f;          ///< seconds since it formed
        float life = 60.0f;        ///< its fade (60 dB), from Memory Time when it formed
        float bloom = 4.0f;        ///< its rise, from Bloom when it formed
        float env = 0.0f;          ///< its level now, 0 .. 1 (the weakest is replaced first)
    };
    std::vector<Ghost> ghosts_;                        ///< the memory's ghosts, on the GPU voices after voices_
    Rng ghostRng_;                                     ///< the ghosts' dice: their own stream, so the memory never moves the conductor
    Rng noteRng_;                                      ///< the Random per Note source's dice (its own stream)
    float randomNote_ = 0.0f;                          ///< the last note's draw, -1 .. 1
    float roughness_ = 0.0f;                           ///< the chord's roughness as last worked out
    int roughTick_ = 0;                                ///< blocks until it is worked out again
    std::vector<float> specL_;                         ///< the spectral echo's block from the GPU, left
    std::vector<float> specR_;                         ///< and right
    float prevSpecMix_ = 0.0f;                         ///< the spectral echo's level at the end of the last block
    float prevSpecToFar_ = 0.0f;                       ///< its send into the far plane, likewise
    bool specWasOn_ = false;                           ///< it ran in the last block (switching it on clears its lines)
    std::vector<float> meltL_;                         ///< the melt's block from the GPU, left
    std::vector<float> meltR_;                         ///< and right
    float prevMeltMix_ = 0.0f;                         ///< the melt's level at the end of the last block
    float prevMeltToFar_ = 0.0f;                       ///< its send into the far plane, likewise
    bool meltWasOn_ = false;                           ///< it ran in the last block (switching it on clears it)
    float binAngle_ = 0.0f;                            ///< the binaural cloud's orbit, radians
    double combPhase_ = 0.0;                           ///< the spectral comb's phase, notches
    double pictureTime_ = 0.0;                         ///< the animated pictures' flow, in their units
    double pictureAngle_ = 0.0;                        ///< their turn, radians
    double pictureSeconds_ = 0.0;                      ///< their real seconds
    float paintTime_ = 0.0f;                           ///< seconds into the landscape's loop, for the painted hall
    float landLoop_ = 0.0f;                            ///< the landscape's loop, seconds
    std::mutex snapLock_;                              ///< guards snap_ (the worker only tries it: it never waits)
    std::vector<float> snap_;                          ///< the last snapshot of the partials, for the picture
    int snapPer_ = 0;                                  ///< its points per voice
    std::uint64_t snapSeq_ = 0;                        ///< snapshots published (under snapLock_)
    std::uint64_t snapSeen_ = 0;                       ///< the last one the picture took (message thread)
    int snapCountdown_ = 0;                            ///< blocks until the next snapshot is asked for
    std::vector<float> snapLife_;                      ///< the living field of the last snapshot, empty when off
    int lifeModeWas_ = 0;                              ///< the living field's mode in the last block (a change seeds it)
    float lifeAcc_ = 0.0f;                             ///< its steps owed, carried between blocks
    float lifeQuiet_ = 0.0f;                           ///< seconds since it was last seeded (it is not seeded again sooner than 2 s)
    float lifeMakeup_ = 1.0f;                          ///< the make-up as it glides (half a second), so a seeding does not jump
    /** @brief A released voice's ghost into a free slot, or over the faintest. @param i the voice */
    void captureGhost(int i);

    /** @brief The parameter values one block works with. */
    struct Block {
        float master = 1.0f;     ///< master gain, linear
        int partials = 20000;    ///< partials per voice
        float tilt = 1.2f;       ///< spectral tilt
        float width = 25.0f;     ///< cloud width, cents, after Purity's narrowing
        float widthSlope = 0.3f; ///< cloud widening
        float drift = 3.0f;      ///< cents
        float driftRate = 0.1f;  ///< Hz
        float shimmer = 0.4f;    ///< 0 .. 1
        float shimmerRate = 0.15f;   ///< Hz
        float stereo = 1.0f;     ///< 0 .. 1
        int motion = 0;          ///< MotionKind
        float motionRate = 1.0f; ///< times
        float motionDepth = 1.0f;///< times
        float attack = 6.0f;     ///< seconds
        float decay = 4.0f;      ///< seconds
        float sustain = 1.0f;    ///< 0 .. 1
        float release = 12.0f;   ///< seconds
        float refPitch = 440.0f; ///< Hz
        float purity = 1.0f;     ///< 0 .. 1
        float depth = 0.6f;      ///< conductor's depth
        float keysDepth = 0.2f;  ///< played notes' distance
        float farLevel = 0.8f;   ///< far reverb level
        float brightness = 1.0f; ///< brightness window
        float oddEven = 0.0f;    ///< odd/even
        float inharmonicB = 0.0f;///< the stiff string's B
        float stretch = 1.0f;    ///< the harmonics' stretch exponent (Stretch / 1200)
        float shift = 0.0f;      ///< Hz every partial is moved by
        float air = 0.0f;        ///< air band level
        float airCenter = 3.0f;  ///< its centre, times f0
        float airWidth = 600.0f; ///< its width, cents
        float airJitter = 8.0f;  ///< its flicker, Hz
        int filterModel = 0;     ///< FilterModel
        float cutoff = 2500.0f;  ///< Hz
        float resonance = 0.15f; ///< 0 .. 1
        float filterEnv = 0.0f;  ///< octaves
        float keyTrack = 0.5f;   ///< 0 .. 1
        float filterDrift = 0.3f;///< 0 .. 1
        float presence = 0.0f;   ///< dB
        float padLowCut = 0.0f;  ///< Hz
        float breath = 0.0f;     ///< 0 .. 1
        float breathRate = 0.03f;///< Hz
        float doppler = 0.0f;    ///< 0 .. 1
        float tide = 0.0f;       ///< cents
        float tidePeriod = 12.0f;///< minutes
        bool brain2 = false;     ///< the second conductor plays
        int brain2Interval = -12;///< semitones
        float brain2Depth = 0.9f;///< its plane
    } blk_;                      ///< read by readParams()

    std::vector<Event> queue_;                         ///< audio thread -> renderer, a single-producer single-consumer ring
    std::atomic<std::uint32_t> qHead_{ 0 };            ///< written by the audio thread
    std::atomic<std::uint32_t> qTail_{ 0 };            ///< written by the renderer
    std::vector<Event> pending_;                       ///< events drained but not yet due (renderer only)

    std::vector<float> ring_;                          ///< the rendered timeline, interleaved stereo
    std::int64_t ringMask_ = 0;                        ///< frames in the ring minus one (a power of two)
    std::atomic<std::int64_t> renderPos_{ 0 };         ///< timeline samples rendered (renderer writes)
    std::atomic<std::int64_t> outPos_{ 0 };            ///< timeline samples handed out (audio thread writes)

    std::thread worker_;                               ///< the renderer in real-time mode
    std::atomic<bool> running_{ false };               ///< the worker runs
    std::atomic<bool> stepAsked_{ false };             ///< autoplayStep() was asked for
    std::atomic<bool> fillAsked_{ false };             ///< requestFill() was asked for
    std::atomic<int> sounding_{ 0 };                   ///< voices sounding, for status
    std::atomic<int> soundingBy_[3] = {};               ///< voices sounding per source, for status
    std::atomic<int> brainRoot_{ 0 };                  ///< the conductor's root, for status
    std::atomic<std::uint64_t> underruns_{ 0 };        ///< callbacks short of samples
    std::atomic<std::uint64_t> late_{ 0 };             ///< notes applied after their stamp
    std::atomic<float> gpuMs_{ 0.0f };                 ///< GPU time of the last block
    std::atomic<int> worstBlockUs_{ 0 };               ///< the slowest renderBlock since takeWorstBlockMs(), microseconds
    std::atomic<int> worstPrepUs_{ 0 };                ///< its slowest preparation before the GPU, microseconds
    std::atomic<int> worstWaitUs_{ 0 };                ///< its longest wait for the GPU
    std::atomic<int> worstPostUs_{ 0 };                ///< its slowest effects after the GPU
    std::atomic<int> minFill_{ -1 };                   ///< the emptiest ring a callback found since takeMinFill(), samples
    std::atomic<int> maxCallback_{ 0 };                ///< the largest callback since takeMaxCallback(), samples
    std::array<std::atomic<float>, kMaxViewVoices> viewHz_{};        ///< per voice its frequency for pictures (0 silent)
    std::array<std::atomic<float>, kMaxViewVoices> viewGain_{};      ///< its envelope
    std::array<std::atomic<float>, kMaxViewVoices> viewDistance_{};  ///< its distance
    std::array<std::atomic<float>, kMaxViewVoices> viewTime_{};      ///< its time since the note began
    std::array<std::atomic<int>, kMaxViewVoices> viewSource_{};      ///< whose note it plays
};

} // namespace additiv
