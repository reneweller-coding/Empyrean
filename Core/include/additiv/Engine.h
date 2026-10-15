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
 * @author Rene Weller
 */
#pragma once

#include "additiv/Cloud.h"
#include "additiv/ClusterBrain.h"
#include "additiv/Dsp.h"
#include "additiv/Effects.h"
#include "additiv/Focus.h"
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

namespace additiv
{

/** @brief How an engine is laid out; fixed between prepare() and release(). */
struct EngineConfig
{
    double        m_sampleRate = 48000.0;       ///< the host's rate; the GPU renders at it
    int           m_maxHostBlock = 1024;        ///< the largest block process() is asked for
    int           m_gpuBlock = 128;             ///< samples per GPU block, a power of two
    int           m_voices = 16;                ///< voices
    int           m_ghosts = 4;                 ///< memory voices beside them: the ghosts of notes let go (MemoryLevel)
    int           m_partialsPerVoice = 20000;   ///< the most partials a voice can have
    float         m_lookaheadMs = 10.0f;        ///< how far beyond two host blocks the worker renders ahead
    bool          m_realtimeGpuClass = true;    ///< raise the process's GPU scheduling class to REALTIME (PLAN.md)
    std::uint64_t m_seed = 0x5EEDu;             ///< the conductor's and the notes' dice
};

/** @brief What a picture of the instrument needs to know about one voice (Engine::voiceView). */
struct VoiceView
{
    float m_hz = 0.0f;         ///< the frequency it sounds at, 0 when silent
    float m_gain = 0.0f;       ///< its envelope, 0 .. 1
    float m_distance = 0.0f;   ///< where it stands, 0 at the ear .. 1 far
    float m_time = 0.0f;       ///< seconds since it began (the motions' clock)
    int   m_source = 0;        ///< 0 played, 1 the conductor, 2 the second conductor, 3 a ghost of the memory
    int   m_note = -1;         ///< the key it sounds (MIDI), -1 for a ghost or when silent
    float m_press = 0.0f;      ///< MPE: its pressure, 0 .. 1
    float m_slide = 0.5f;      ///< MPE: its slide, 0 .. 1 (0.5 at rest)
    float m_bend = 0.0f;       ///< MPE: its own bend, semitones
    float m_tune = 0.0f;       ///< Self-Tune's pull on it, cents
};

constexpr int MaxViewVoices = 32;   ///< voices a picture can be given

/** @brief The instrument (see the file comment). */
class Engine
{
public:
    Engine();
    ~Engine();
    Engine( const Engine & ) = delete;              ///< owns the GPU and a thread
    Engine &operator=( const Engine & ) = delete;   ///< owns the GPU and a thread

    /**
     * @brief Opens the GPU, builds everything and, in real-time mode, starts the worker.
     * @param[in]  config  the layout
     * @param[out] error   receives the reason when it fails (may be null)
     * @return true when the engine can play
     */
    bool prepare( const EngineConfig &config, std::string *error );

    /** @brief Stops the worker and frees the GPU; prepare() may be called again. */
    void release();

    /** @brief Whether prepare() succeeded. @return true when process() makes sound */
    bool prepared() const { return m_prepared; }

    /**
     * @brief Real time (a worker renders ahead) or offline (process() renders what it needs itself).
     * @param[in] realtime  true for live playing, false for a bounce, the renderer and the tests
     */
    void setRealtime( bool realtime );

    /** @brief The mode setRealtime() chose. @return true in real-time mode */
    bool realtime() const { return m_realtime; }

    /** @brief How many samples everything is delayed by (the host compensates). @return samples */
    int latencySamples() const { return m_latency; }

    /** @brief The layout prepare() was given. @return the config */
    const EngineConfig &config() const { return m_cfg; }

    /** @brief The GPU, for the logs. @return its name and compute capability ("" before prepare) */
    std::string deviceName() const;

    /**
     * @brief Sets a parameter, clamped to its range. Any thread.
     * @param[in] id     the parameter
     * @param[in] value  in its unit
     */
    void setParam( ParamIdE id, float value );

    /**
     * @brief A parameter's value.
     * @param[in] id  the parameter
     * @return in its unit
     */
    float param( ParamIdE id ) const { return m_params[static_cast<size_t>( id )].load( std::memory_order_relaxed ); }

    /** @brief Every parameter to its default. */
    void resetParams();

    /**
     * @brief A played note begins. Audio thread, before the process() it belongs to.
     * @param[in] note      MIDI note
     * @param[in] velocity  0 .. 1
     * @param[in] offset    samples into the coming process() block
     * @param[in] channel   MPE: the member channel (0 .. 15) whose bend, pressure and slide are this note's; -1 none
     * @pre called on the audio thread
     */
    void noteOn( int note, float velocity, int offset, int channel = -1 );

    /**
     * @brief A played note is let go. Audio thread.
     * @param[in] note     MIDI note
     * @param[in] offset   samples into the coming process() block
     * @param[in] channel  MPE: its member channel, -1 any
     * @pre called on the audio thread
     */
    void noteOff( int note, int offset, int channel = -1 );

    /** @name MPE: what each member channel's note does after it has begun. Any thread.
     *  @{ */
    /**
     * @brief A channel's pitch bend.
     * @param[in] channel    0 .. 15, or -1 for the master (every played note)
     * @param[in] semitones  the bend
     */
    void setChannelBend( int channel, float semitones )
    {
        m_chanBend[static_cast<size_t>( channel < 0 ? 16 : std::min( channel, 15 ) )].store(
            semitones, std::memory_order_relaxed );
    }
    /**
     * @brief A channel's pressure.
     * @param[in] channel   0 .. 15
     * @param[in] pressure  0 .. 1
     */
    void setChannelPressure( int channel, float pressure )
    {
        m_chanPress[static_cast<size_t>( std::clamp( channel, 0, 15 ) )].store( pressure, std::memory_order_relaxed );
    }
    /**
     * @brief A channel's slide (CC 74).
     * @param[in] channel  0 .. 15
     * @param[in] slide    0 .. 1 (0.5 at rest)
     */
    void setChannelSlide( int channel, float slide )
    {
        m_chanSlide[static_cast<size_t>( std::clamp( channel, 0, 15 ) )].store( slide, std::memory_order_relaxed );
    }
    /** @} */

    /**
     * @brief Every played note is let go (the conductor's notes stay). Audio thread.
     * @pre called on the audio thread
     */
    void allNotesOff();

    /**
     * @brief Sets one row of the modulation matrix. Any thread (a row is one atomic word).
     * @param[in] i    0 .. ModRows-1
     * @param[in] row  the row
     */
    void setModRow( int i, const ModRow &row );
    /**
     * @brief One row of the matrix.
     * @param[in] i  0 .. ModRows-1
     * @return the row
     */
    ModRow modRow( int i ) const;
    /** @brief The matrix as text (Modulation.h). @return the text */
    std::string modText() const;
    /**
     * @brief The matrix from text; rows it does not name are emptied.
     * @param[in] text  the text
     */
    void setModText( const std::string &text );

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
     * @param[in] l  the landscape (a picture's or a recording's, Landscape.h)
     * @pre called on the message thread
     */
    void setUserLandscape( const Landscape &l );
    /** @brief Whether the player's own landscape has been set. Any thread. @return true then */
    bool hasUserLandscape() const { return m_userLandVersion.load( std::memory_order_relaxed ) > 0; }
    /**
     * @brief Glide from what plays now to what is set next. Any thread.
     * @param[in] seconds  how long (at least 0.05)
     */
    void beginMorph( double seconds );
    /** @brief How far the glide has come. Any thread. @return 0 .. 1; 1 when none runs */
    float morphProgress() const { return m_morphProgress.load( std::memory_order_relaxed ); }
    /**
     * @brief Chooses a slot of the A/B morph. Message thread.
     * @param[in] slot     0 for A, 1 for B
     * @param[in] values   NumParams values
     * @param[in] modText  its matrix (Modulation.h)
     * @pre called on the message thread
     */
    void setMorphSlot( int slot, const float *values, const std::string &modText );
    /**
     * @brief Whether a slot has been chosen. Any thread.
     * @param[in] slot  0 or 1
     * @return true then
     */
    bool morphSlotSet( int slot ) const
    {
        return m_slotSet[static_cast<size_t>( slot & 1 )].load( std::memory_order_relaxed );
    }
    /**
     * @brief The sound at the map's cursor (PresetMap.h's MapBlender), played while Map is on: the
     *        sound follows it over Map Glide. Message thread (the host blends when the cursor moves).
     * @param[in] s  the blend
     * @pre called on the message thread
     */
    void setMapTarget( const MorphState &s );
    /** @brief Whether the map has a target yet. Any thread. @return true then */
    bool mapTargetSet() const { return m_slotSet[2].load( std::memory_order_relaxed ); }
    /**
     * @brief A parameter where the sound stands, before the matrix: the knob's, or the morph's or
     *        the glide's way between. Any thread.
     * @param[in] id  the parameter
     * @return its value in its own units
     */
    float soundValue( ParamIdE id ) const
    {
        return m_soundLive[static_cast<size_t>( id )].load( std::memory_order_relaxed );
    }
    /** @brief The voice filter's share where the sound stands (Morph.h). Any thread. @return 0 .. 1 */
    float soundFilterMix() const { return m_soundMix.load( std::memory_order_relaxed ); }
    /**
     * @brief Loads a factory preset: every parameter but the listener's (keptByPresets) and the
     *        matrix, gliding into it when `glideSeconds` is above 0 (a host with its own parameters
     *        does this itself; the render tool and the tests use this).
     * @param[in] index         the preset
     * @param[in] glideSeconds  0: at once
     */
    void loadPreset( int index, double glideSeconds = 0.0 );
    /** @} */
    /**
     * @brief The mod wheel (CC 1). Any thread.
     * @param[in] v  0 .. 1
     */
    void setModWheel( float v ) { m_wheel.store( v, std::memory_order_relaxed ); }
    /**
     * @brief Channel pressure. Any thread.
     * @param[in] v  0 .. 1
     */
    void setPressure( float v ) { m_pressure.store( v, std::memory_order_relaxed ); }
    /**
     * @brief Where a parameter plays after the matrix, normalised, for the panel's live rings.
     * @param[in] id  the parameter
     * @return 0 .. 1, NaN when nothing modulates it
     */
    float playedNorm( ParamIdE id ) const
    {
        return m_played[static_cast<size_t>( id )].load( std::memory_order_relaxed );
    }

    /**
     * @brief A modulation source's value as the last block left it, for the panel. Any thread.
     * @param[in] s  the source
     * @return its value (-1 .. 1, or 0 .. 1 for a unipolar one)
     */
    float sourceValue( ModSourceE s ) const
    {
        const int i = static_cast<int>( s );
        return i > 0 && i < NumModSources ? m_sourceLive[static_cast<size_t>( i )].load( std::memory_order_relaxed )
                                          : 0.0f;
    }

    /**
     * @brief The panel shows the sources: what is computed only for a row that reads it (the
     *        reservoir) runs while they are watched. Any thread.
     * @param[in] on  watched
     */
    void setSourcesWatched( bool on ) { m_sourcesWatched.store( on, std::memory_order_relaxed ); }

    /**
     * @brief The family jam, following: the leader's root, which the conductor's root is held on (its
     *        nearest note of that pitch class), as Noctuary follows. Any thread.
     * @param[in] pitchClass  0 .. 11, -1 when no leader is heard
     */
    void setJamRoot( int pitchClass ) { m_jamRoot.store( pitchClass, std::memory_order_relaxed ); }
    /**
     * @brief The family jam, following: the energy of the leader's section, which opens or closes the
     *        brightness and the filter (0.5 changes nothing). Any thread.
     * @param[in] energy  0 .. 1
     */
    void setJamEnergy( float energy ) { m_jamEnergy.store( energy, std::memory_order_relaxed ); }

    /** @brief Autoplay: exchange a voice of the cluster at the next block, whatever its timer says. Any thread. */
    void autoplayStep() { m_stepAsked.store( true, std::memory_order_relaxed ); }
    /**
     * @brief Both conductors fill their chords at speed (a note every third of a second) until they
     *        are full, as Noctuary after a preset change: the arriving preset is heard whole within
     *        seconds, not after minutes of its own event rate. Any thread.
     */
    void requestFill() { m_fillAsked.store( true, std::memory_order_relaxed ); }

    /**
     * @brief The audio callback: n samples of the timeline.
     * @param[in,out] L  left output
     * @param[in,out] R  right output
     * @param[in]     n  samples, at most config().maxHostBlock in real-time mode
     */
    void process( float *L, float *R, int n );

    /** @brief Voices sounding after the last rendered block. @return 0 .. voices */
    int soundingVoices() const { return m_sounding.load( std::memory_order_relaxed ); }

    /**
     * @brief Voices sounding after the last rendered block, by whose notes they play.
     * @param[in] source  0 played, 1 the conductor, 2 the second conductor
     * @return count
     */
    int soundingFrom( int source ) const
    {
        return source >= 0 && source < 3 ? m_soundingBy[source].load( std::memory_order_relaxed ) : 0;
    }

    /** @brief The conductor's root after the last rendered block. @return MIDI note */
    int brainRoot() const { return m_brainRoot.load( std::memory_order_relaxed ); }
    /** @brief Seconds until the conductor's next decision, for the panel. Any thread. @return seconds */
    float conductorNext() const { return m_conductorNext.load( std::memory_order_relaxed ); }
    /** @brief The seconds that wait began with (the countdown's whole). Any thread. @return seconds */
    float conductorSpan() const { return m_conductorSpan.load( std::memory_order_relaxed ); }
    /** @brief The note the foundation holds, -1 none, for the picture. Any thread. @return MIDI */
    int foundationNow() const { return m_foundationView.load( std::memory_order_relaxed ); }
    /**
     * @brief The scale's pitches folded into one octave over the root, in cents above the root's equal-tempered
     *        pitch, for the panel (the scale itself, before Purity blends it with equal temperament). Any thread.
     * @param[out] out  receives them, ascending
     * @param[in]  max  room in out
     * @return how many
     */
    int scaleCents( float *out, int max ) const
    {
        const int n = std::min( max, m_viewScaleCount.load( std::memory_order_relaxed ) );
        for ( int i = 0; i < n; ++i ) out[i] = m_viewScale[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        return n;
    }
    /** @brief The root's equal-tempered pitch, the panel's zero for cents. Any thread. @return hertz */
    float rootHz() const { return m_viewRootHz.load( std::memory_order_relaxed ); }
    /**
     * @brief A key's pitch as the scale, the root, A4 and Purity tune it now (before Tide and Self-Tune), for the
     *        panel. Any thread.
     * @param[in] note  MIDI
     * @return hertz
     */
    float noteHz( int note ) const
    {
        return note >= 0 && note < 128 ? m_viewNoteHz[static_cast<size_t>( note )].load( std::memory_order_relaxed )
                                       : 0.0f;
    }

    /** @brief Callbacks that found the ring short (real time only). @return count since prepare() */
    std::uint64_t underruns() const { return m_underruns.load( std::memory_order_relaxed ); }

    /** @brief Notes that arrived after their stamp had been rendered. @return count since prepare() */
    std::uint64_t lateEvents() const { return m_late.load( std::memory_order_relaxed ); }

    /**
     * @brief A voice as the last rendered block left it, for pictures (any thread; the fields may come from two
     *        neighbouring blocks, which a picture does not mind).
     * @param[in] i  the voice, 0 .. config().voices-1
     * @return its view; hz 0 when it is silent
     */
    VoiceView voiceView( int i ) const
    {
        VoiceView v;
        if ( i < 0 || i >= MaxViewVoices ) return v;
        v.m_hz = m_viewHz[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_gain = m_viewGain[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_distance = m_viewDistance[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_time = m_viewTime[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_source = m_viewSource[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_note = m_viewNote[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_press = m_viewPress[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_slide = m_viewSlide[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_bend = m_viewBend[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        v.m_tune = m_viewTune[static_cast<size_t>( i )].load( std::memory_order_relaxed );
        return v;
    }

    /**
     * @brief The partials as the picture draws them: every SnapStride-th partial of every voice (and
     *        ghost), as (Hz, amplitude), brought from the GPU about thirty times a second. Message thread.
     * @param[out] dots      receives the points, voice after voice, perVoice each
     * @param[out] perVoice  receives how many points each voice has
     * @return true when there was a new snapshot since the last call
     * @pre called on the message thread
     */
    bool partialSnapshot( std::vector<float> &dots, int &perVoice );
    /**
     * @brief The living field as the last snapshot brought it (LifeW x LifeH cells, 0 .. 1). Message thread.
     * @param[out] cells  receives the cells, emptied when the field is off
     * @pre called on the message thread
     */
    void lifeSnapshot( std::vector<float> &cells );
    /**
     * @brief The hall's memory (Hall > Grow): per band over the octaves (HallBins, 20 Hz .. 20 kHz) how
     *        much it has heard there against the average, as the last snapshot brought it down -- or as
     *        it was last set, until the next block has it. Any thread but the worker's.
     * @param[out] out  receives HallBins values, 1 the average (a new hall: all 1)
     * @param[out] age  receives the seconds it has learnt
     * @return true when it has learnt anything (age above nothing)
     */
    bool hallMemory( std::vector<float> &out, double &age );
    /**
     * @brief A hall's memory to hear in from the next block on (a hall loaded, a state restored, or a
     *        new hall). Any thread but the worker's.
     * @param[in] in   HallBins values; empty or the wrong size: a new hall, every band 1
     * @param[in] age  its seconds learnt
     */
    void setHallMemory( const std::vector<float> &in, double age );
    /**
     * @brief The ripple's string as the last snapshot left it, for the picture (RippleCells cells, low to
     *        high; empty while the ripple rests). Any thread but the worker's.
     * @param[out] cells  receives the cells
     */
    void rippleSnapshot( std::vector<float> &cells );
    /**
     * @brief A macro taught by example (Focus.h): from the next block it moves the knobs it learnt, at its
     *        macro's value. Any thread but the worker's.
     * @param[in] macro   0 .. 3
     * @param[in] lesson  the lesson (fewer than two places: none)
     */
    void setMacroLesson( int macro, const MacroLesson &lesson );
    /** @brief Where the spectral comb's notches stand, in notches (SpectralComb::phase), for the picture. Any thread. @return 0 .. 1 */
    float combPhase() const { return m_combPhaseView.load( std::memory_order_relaxed ); }
    /**
     * @brief The ripple's string as it stands after the last block, low to high (tests, diagnostics).
     *        The worker's thread, or offline.
     * @param[out] out  receives RippleCells values
     */
    void rippleString( float *out ) const
    {
        for ( int i = 0; i < RippleCells; ++i ) out[i] = m_rippleNow[i];
    }

    /** @brief The slowest block the renderer took since the last call, and starts over (diagnostics). @return milliseconds */
    float takeWorstBlockMs() { return m_worstBlockUs.exchange( 0, std::memory_order_relaxed ) * 0.001f; }
    /**
     * @brief The slowest of each part of a block since the last call, and starts over (diagnostics): the
     *        preparation before the GPU is launched, the wait for it, the effects after.
     * @param[out] prepMs     receives the slowest preparation, milliseconds
     * @param[out] gpuMs      receives the longest wait for the GPU
     * @param[out] effectsMs  receives the slowest effects
     */
    void takeWorstParts( float &prepMs, float &gpuMs, float &effectsMs )
    {
        prepMs = m_worstPrepUs.exchange( 0, std::memory_order_relaxed ) * 0.001f;
        gpuMs = m_worstWaitUs.exchange( 0, std::memory_order_relaxed ) * 0.001f;
        effectsMs = m_worstPostUs.exchange( 0, std::memory_order_relaxed ) * 0.001f;
    }
    /** @brief The emptiest the ring was when a callback came, since the last call, and starts over. @return samples, -1 if no callback came */
    int takeMinFill() { return m_minFill.exchange( -1, std::memory_order_relaxed ); }
    /** @brief The largest block a callback asked for since the last call, and starts over. @return samples */
    int takeMaxCallback() { return m_maxCallback.exchange( 0, std::memory_order_relaxed ); }

    /** @brief GPU time of the last block. @return milliseconds */
    float lastGpuMs() const { return m_gpuMs.load( std::memory_order_relaxed ); }

private:
    /** @brief A note event on its way from the audio thread to the renderer. */
    struct Event
    {
        int          m_type = 0;          ///< 0 note on, 1 note off, 2 all played notes off
        int          m_note = 0;          ///< MIDI note
        float        m_velocity = 0.0f;   ///< 0 .. 1 (note on)
        std::int64_t m_stamp = 0;         ///< the timeline sample it takes effect at
        int          m_channel = -1;      ///< MPE: its member channel, -1 none
    };

    /** @brief One voice on the host side: the note, its envelope, its place, and what the GPU voice is told. */
    struct Voice
    {
        int           m_note = -1;      ///< MIDI note, -1 when free
        int           m_source = 0;     ///< 0 played, 1 the conductor, 2 the second conductor
        int           m_channel = -1;   ///< MPE: its member channel, -1 none
        float         m_bend = 0.0f;    ///< MPE: its own bend, semitones (its channel's while held, kept once let go)
        float         m_press = 0.0f;   ///< MPE: its pressure, 0 .. 1
        float         m_slide = 0.5f;   ///< MPE: its slide, 0 .. 1 (0.5 at rest)
        float         m_velocity = 0.0f;   ///< 0 .. 1
        float         m_distance = 0.0f;   ///< where it stands now, 0 at the ear .. 1 far (its place plus the breath)
        float         m_place = 0.0f;      ///< where it was put
        float         m_prevDistance = 0.0f;   ///< the distance of the block before, for the doppler
        Drifter       m_breath;                ///< the slow wander of its distance (Breath)
        Drifter       m_cutoffDrift;           ///< the slow wander of its filter's cutoff (Filter Drift)
        double        m_hz = 0.0;              ///< the frequency it is gliding at (purity and root changes glide)
        Envelope      m_env;                   ///< the amplitude envelope, stepped once per GPU block
        std::uint32_t m_seed = 1;              ///< the note's dice on the GPU
        bool          m_restart = false;       ///< the GPU voice begins this note at the next block
        int           m_onset = 0;             ///< and at this sample of it
        bool          m_released = false;      ///< noteOff has come
        int           m_tailBlocks = 0;   ///< blocks it stays active after the envelope fell silent (ramps to zero)
        bool   m_tailArmed = false;       ///< the tail was lengthened by the dispersion when the envelope fell silent
        float  m_tuneCents = 0.0f;        ///< self-tuning: how far the voice has moved from its pitch in the scale
        float  m_lastGain = 0.0f;         ///< its envelope in the last block (self-tuning weighs by it)
        double m_age = 0.0;               ///< seconds since it began, for stealing
        bool   m_stealing = false;        ///< fading out before `next` begins
        int    m_stealBlocks = 0;         ///< blocks of fading still to go
        Event  m_next;                    ///< the note waiting for a stolen voice (type 0; -1 = cancelled)
        int    m_nextSource = 0;          ///< whose note that is
        float  m_nextDistance = 0.0f;     ///< and where it will stand
    };

    void workerLoop();    ///< the worker: renders while the ring is short of the latency
    void startWorker();   ///< starts the worker (real-time mode)
    void stopWorker();    ///< stops and joins it
    void renderBlock();   ///< one GPU block of the timeline into the ring (renderer thread)
    void readParams();    ///< the parameters the block uses, read once per block, the modulation applied
    /**
     * @brief A parameter's effective value this block (after the matrix).
     * @param[in] id  the parameter
     * @return the value
     */
    float cur( ParamIdE id ) const { return m_cur[static_cast<size_t>( id )]; }
    void drainEvents();   ///< moves the queued events into the pending list
    /**
     * @brief Starts a note on a free voice, or steals one (fading it over a block first).
     * @param[in] note      MIDI note
     * @param[in] velocity  0 .. 1
     * @param[in] distance  0 at the ear .. 1 far
     * @param[in] source    0 played, 1 the conductor, 2 the second conductor
     * @param[in] onset     sample of the block it begins at
     * @param[in] channel   MPE: the played note's member channel, -1 none
     */
    void startNote( int note, float velocity, float distance, int source, int onset, int channel = -1 );
    /**
     * @brief Begins a note on a voice now (the voice is free or has just faded).
     * @param[in] i         the voice
     * @param[in] note      MIDI note
     * @param[in] velocity  0 .. 1
     * @param[in] distance  0 .. 1
     * @param[in] source    0 played, 1 the conductor, 2 the second conductor
     * @param[in] onset     sample of the block it begins at
     * @param[in] channel   MPE: the played note's member channel, -1 none
     */
    void beginOn( int i, int note, float velocity, float distance, int source, int onset, int channel = -1 );
    /**
     * @brief Lets a note go.
     * @param[in] note     MIDI note
     * @param[in] source   whose note: 0 played, 1 the conductor, 2 the second conductor
     * @param[in] channel  MPE: the played note's member channel, -1 any
     */
    void stopNote( int note, int source, int channel = -1 );
    /** @brief Where a conductor's note stands: Noctuary's two planes, 40 % close and 60 % deep. @return 0 .. 1 */
    float brainDistance();
    /** @brief Loads the landscape the Landscape parameter names when it changed, and its RMS. */
    void refreshLandscape();
    /** @brief Fills the conductors' spectrum from the voice's shape, and rebuilds the Timbre scale when it changed. */
    void refreshSpectrum();
    /**
     * @brief A note's frequency: the scale blended with 12-TET by Purity (Noctuary's frequencyOf).
     * @param[in] note  MIDI note
     * @return Hz
     */
    double frequencyOf( int note ) const;
    /** @brief Rebuilds the scale when its choice, root or key mapping changed. */
    void refreshScale();

    EngineConfig m_cfg;                ///< the layout
    bool         m_prepared = false;   ///< prepare() succeeded
    bool         m_realtime = true;    ///< worker mode
    int          m_latency = 0;        ///< samples the timeline runs ahead
    int          m_startReserve = 0;   ///< samples rendered beyond it before the first callback, used up over as many
    /**
     * @brief Where an event from the host lands on the timeline: the latency ahead of what the
     *        callback has taken, and while the start reserve is not used up, its rest too (what is
     *        rendered already reaches that far; an event there would come late).
     * @param[in] offset  samples into the host's block
     * @return the event's sample on the timeline
     */
    std::int64_t stampFor( int offset ) const
    {
        const std::int64_t out = m_outPos.load( std::memory_order_relaxed );
        return out + std::max( offset, 0 ) + m_latency + std::max<std::int64_t>( 0, m_startReserve - out );
    }
    std::array<std::atomic<float>, NumParams> m_params;   ///< the parameters, any thread

    CloudEngine        m_cloud;       ///< the GPU
    std::vector<Voice> m_voices;      ///< the host's voices
    ClusterBrain       m_brain;       ///< the conductor
    BrainParams        m_bp;          ///< its knobs, filled per block
    ClusterBrain       m_brain2;      ///< the second conductor, in the background (Noctuary's Brain 2)
    BrainParams        m_bp2;         ///< its knobs
    BrainSpectrum      m_brainSpec;   ///< the cloud's spectrum for its Timbre judgement
    float     m_specKey[5] = { -1.0f, 0.0f, 0.0f, 0.0f,
                               1.0f };   ///< tilt, odd/even, brightness, inharmonic, stretch the spectrum was built for
    Rng       m_rng;                     ///< the engine's dice (seeds, distances)
    Rng       m_side;              ///< a side stream for the drifters, so switching them on never moves the notes' dice
    Drifter   m_tide;              ///< the whole instrument's slow lean (Tide)
    int       m_landChoice = -1;   ///< the landscape on the GPU
    Landscape m_userLand;          ///< the player's own, as setUserLandscape left it (under m_userLandLock)
    float     m_userLandRms = 1.0f;            ///< its RMS
    std::mutex       m_userLandLock;           ///< guards m_userLand (the worker only tries it)
    std::atomic<int> m_userLandVersion{ 0 };   ///< bumped by setUserLandscape
    int              m_userLandSeen = 0;       ///< the version on the GPU
    float            m_landRms = 1.0f;         ///< its RMS, for the level of voices that read it
    bool             m_stepWas = false;        ///< the Step switch's last value (it fires on the rising edge)
    // The foundation (Foundation): one voice on the conductor's root in the octave from C2 up (from
    // Lowest when the register begins lower), held while the root stands; the chord is drawn from a
    // fifth above it.
    static constexpr int FoundationSource = 4;    ///< a voice's source for the foundation (shown as the conductor's)
    float                m_foundation = 0.0f;     ///< its level, 0 none
    int                  m_foundationLow = 36;    ///< the bottom of the foundation's octave: C2, or Lowest when lower
    int                  m_foundationNote = -1;   ///< the note it holds, -1 none
    float                m_foundationMatch =
        0.0f;   ///< half the chord voices' summed power (as an amplitude) over the foundation's sustain, followed over seconds: its level
    /**
     * @brief The foundation's note for a root: the root's pitch class in the octave from @p low up.
     * @param[in] root  the conductor's root (MIDI)
     * @param[in] low   the bottom of the register
     * @return the note
     */
    static int foundationNote( int root, int low ) { return low + ( ( ( root - low ) % 12 ) + 12 ) % 12; }
    /** @brief The foundation follows the root, comes and goes with the conductor and Foundation, and comes back when its voice was taken. */
    void updateFoundation();
    Modulators m_mods;   ///< the modulation sources
    /** @brief One setup of m_sound's own LFOs and envelopes, running (Morph.h: each row reads the modulators of the sound it came from). */
    struct OwnUnit
    {
        ModSetup      m_setup;          ///< whose settings
        OwnModulators m_mods;           ///< running
        bool          m_live = false;   ///< in use
    };
    static constexpr int OwnUnits =
        24;   ///< setups that run at once (a glide's start and its end hold sixteen each at most)
    std::array<OwnUnit, OwnUnits> m_own{};   ///< the running setups
    int m_ownOf[MorphSetups] = {};           ///< for each of m_sound's setups the unit that runs it this block, -1 none
    /**
     * @brief Runs a unit for every setup m_sound's rows read: the one already running it; else one no
     *        row reads any more takes the new settings on and runs on from where it is (a knob turned,
     *        a preset taken without a glide); else a new one starts where the strongest running one
     *        stands, so a sound joining a blend does not start its LFOs from nothing.
     * @param[in] dt  seconds of the block
     */
    void runOwnModulators( float dt );
    // The morph (Morph.h). The worker owns m_sound, m_from, m_target and m_slots; the editor hands
    // slots over through m_pendingSlots under m_slotLock, which the worker only tries.
    /** @brief What plays this block before the matrix: the knobs, the A/B morph, or a glide between. */
    void updateSound();
    /**
     * @brief A parameter where the sound stands this block (the worker's).
     * @param[in] id  the parameter
     * @return its value
     */
    float base( ParamIdE id ) const { return m_sound.m_values[static_cast<size_t>( id )]; }
    MorphState       m_sound;              ///< what plays this block, before the matrix
    MorphState       m_from;               ///< what played when the glide began
    MorphState       m_target;             ///< where the glide goes: the knobs, or the A/B blend
    MorphState       m_slots[3];           ///< the A/B morph's slots and the map's target, as the worker has them
    MorphState       m_pendingSlots[3];    ///< the same as the host last set them
    std::mutex       m_slotLock;           ///< guards m_pendingSlots (the worker only tries it)
    std::atomic<int> m_slotVersion{ 0 };   ///< bumped by setMorphSlot
    int              m_slotSeen = 0;       ///< the version the worker has
    std::array<std::atomic<bool>, 3> m_slotSet{};   ///< a slot has been chosen (2: the map's target)
    std::atomic<int>                 m_abVersion{
        0
    };   ///< bumped by setMorphSlot only (a new A or B glides; the map's cursor follows)
    int                m_abSeen = 0;              ///< the A/B version the worker has
    bool               m_mapWas = false;          ///< the map played the last block
    std::atomic<int>   m_morphAsked{ 0 };         ///< bumped by beginMorph
    int                m_morphSeen = 0;           ///< the request the worker has served
    std::atomic<float> m_morphSeconds{ 1.0f };    ///< the glide asked for
    float              m_morphT = 1.0f;           ///< 0 .. 1 of the glide's time; 1 when none runs
    float              m_morphLen = 1.0f;         ///< the glide's seconds
    bool               m_abWas = false;           ///< the A/B morph played the last block
    bool               m_soundValid = false;      ///< m_sound holds a block (else a glide has nothing to start from)
    std::atomic<float> m_morphProgress{ 1.0f };   ///< m_morphT, for any thread
    std::array<std::atomic<float>, NumParams>       m_soundLive{};        ///< m_sound's values, for any thread
    std::atomic<float>                              m_soundMix{ 1.0f };   ///< m_sound's filter share, for any thread
    std::array<std::atomic<std::uint64_t>, ModRows> m_modRows{};   ///< the matrix, a packed row per word (packModRow)
    std::array<float, NumParams>                    m_cur{};       ///< every parameter's effective value this block
    std::array<std::atomic<float>, NumModSources>
                      m_sourceLive{};              ///< every source's value after the last block, for the panel
    std::atomic<bool> m_sourcesWatched{ false };   ///< the panel shows the sources (setSourcesWatched)
    std::array<std::atomic<float>, NumParams>
                       m_played{};        ///< where each plays after the matrix, normalised (NaN: unmodulated)
    std::atomic<float> m_wheel{ 0.0f };   ///< the mod wheel
    std::array<std::atomic<float>, 17> m_chanBend{};    ///< MPE: each channel's bend in semitones, [16] the master's
    std::array<std::atomic<float>, 16> m_chanPress{};   ///< MPE: each channel's pressure
    std::array<std::atomic<float>, 16> m_chanSlide{};   ///< MPE: each channel's slide (set to 0.5 in the constructor)
    std::atomic<float>                 m_pressure{ 0.0f };      ///< channel pressure
    float                              m_lastVelocity = 0.0f;   ///< the last note's velocity
    std::atomic<int>                   m_jamRoot{ -1 };         ///< the jam leader's root (pitch class), -1 none
    std::atomic<float>                 m_jamEnergy{ 0.5f };     ///< the jam leader's section energy
    bool               m_phraseStart = false;   ///< a note arrived into silence: the envelopes start over
    int                m_prevSounding = 0;      ///< voices sounding after the block before
    FixedScale         m_scale;                 ///< the current scale
    int                m_scaleChoice = -1;      ///< the choice m_scale was built from
    int                m_rootPc = 0;            ///< the scale's root, a pitch class
    bool               m_snapKeys = true;       ///< key mapping
    Reverb             m_far;                   ///< the far plane's reverb (Noctuary's FDN)
    Diffuser           m_diffuser;              ///< the swell in front of the far reverb
    Unmask             m_unmask;                ///< the far hall stepping aside for the near plane
    StereoDelay        m_delay;                 ///< the first echo (Noctuary's)
    StereoDelay        m_delay2;                ///< the second, in series after the first
    EarlyRoom          m_early;                 ///< the near plane's early reflections
    Reverb             m_nearRev;               ///< the near plane's small reverb
    MidSide            m_midSide;               ///< bass mono and width
    std::vector<float> m_nearL;                 ///< one GPU block's near plane, left (then the whole mix)
    std::vector<float> m_nearR;                 ///< its near plane, right (then the whole mix)
    std::vector<float> m_farL;                  ///< its far plane, left (then the far reverb)
    std::vector<float> m_farR;                  ///< its far plane, right
    std::vector<float> m_dryL;         ///< the near plane before the echoes, left: the early reflections' input
    std::vector<float> m_dryR;         ///< and right
    std::vector<float> m_wetL;         ///< an echo's output, left
    std::vector<float> m_wetR;         ///< and right
    std::vector<float> m_modalL;       ///< the modal hall's block from the GPU, left
    std::vector<float> m_modalR;       ///< and right
    float  m_prevDelayMix = 0.0f;      ///< the first echo's level at the end of the last block (ramped across the next)
    float  m_prevDelayToFar = 0.0f;    ///< its send into the far plane, likewise
    float  m_prevDelay2Mix = 0.0f;     ///< the second echo's level, likewise
    float  m_prevDelay2ToFar = 0.0f;   ///< its send into the far plane, likewise
    float  m_prevModalLevel = 0.0f;    ///< the modal hall's level, likewise
    float  m_modalTime = 0.0f;         ///< the modal hall's drift clock, seconds, wrapped at 1000
    float  m_srcDistance = 0.3f;   ///< where the near room hears the voices: their mean distance, weighted by loudness
    double m_lorenz[FieldAttractors]
                   [3] = {};     ///< the field's attractors: eight Lorenz systems (x, y, z), wandering always
    bool   m_flockWas = false;   ///< the last block's field was the flock (else its state is taken afresh)
    float  m_resMean[ReservoirInputs] = {};   ///< the reservoir's inputs: their running means
    float  m_resVar[ReservoirInputs] = {};    ///< and running variances
    bool   m_resStarted = false;              ///< the running statistics have their first values
    float  m_rippleNow[RippleCells] = {};     ///< the ripple's string now
    float  m_ripplePrev[RippleCells] = {};    ///< and a step before (the wave equation's leapfrog)
    float  m_rippleStrikes[64] = {};          ///< the fundamentals of the notes that started this block
    int    m_rippleStrikeCount = 0;           ///< how many
    float  m_rippleHz
        [64] = {};   ///< the sounding voices' fundamentals this block (the ripple's bend reads where the spectrum is loud)
    float m_rippleAmp[64] = {};   ///< and their levels
    int   m_rippleVoices = 0;     ///< how many
    bool  m_orbitWas = false;     ///< the last block's field was the orbit (else its state is taken afresh)
    float m_light[128] = {};      ///< Follow Light: per MIDI note the conductors' factor
    int   m_lightCountdown = 0;   ///< blocks until the light is read again
    bool  m_lightOn = false;      ///< m_light holds factors (else the conductors ignore it)
    /**
     * @brief Follow Light: where the landscape is bright, read as the partials read it -- the living
     *        field's columns (averaged along), or, while the Aurora motion reads a landscape, its first
     *        eight seconds at the motion's rate (a new note reads it from its start) -- and per note how
     *        much of its harmonics' weight falls into the light against the conductor's register.
     *        Every eighth block; nothing while Follow Light is at nothing.
     */
    void updateLight();
    double m_hallAge = 0.0;   ///< seconds the hall's memory has learnt (worker)
    std::array<float, HallBins>
           m_snapHall{};              ///< the hall's memory as the last snapshot brought it down (under m_snapLock)
    double m_snapHallAge = 0.0;       ///< and its age
    bool   m_snapHallValid = false;   ///< m_snapHall holds what the card has (else a new hall)
    std::array<float, HallBins> m_hallPending{};   ///< a memory to send up with the next block (under m_snapLock)
    double                      m_hallPendingAge = 0.0;     ///< and its age
    bool                        m_hallPendingSet = false;   ///< m_hallPending waits for the next block
    std::vector<float>
        m_snapRipple;   ///< the ripple's string at the last snapshot (under m_snapLock), empty while it rests
    std::atomic<float>         m_combPhaseView{ 0.0f };   ///< the comb's phase after the last block, for the picture
    std::array<MacroLesson, 4> m_lessons{};               ///< the taught macros the worker plays
    std::array<MacroLesson, 4> m_lessonsPending{};   ///< taught macros waiting for the next block (under m_lessonLock)
    std::array<bool, 4>        m_lessonPendingSet{};   ///< which wait
    std::mutex                 m_lessonLock;           ///< guards the pending ones (the worker only tries it)
    /**
     * @brief One block of the ripple's string: struck at the harmonics of every note that started this
     *        block (m_rippleStrikes, gathered by the voice loop, which clears the restart flags), then the damped wave equation stepped (leapfrog, reflecting ends, as many substeps
     *        as the speed needs to stay stable), and the make-up worked out.
     * @param[in] dt  the block's length, seconds
     */
    void stepRipple( float dt );

    /** @brief A ghost of the memory: a voice as it was when it was let go, sounding on, fading, blurring. */
    struct Ghost
    {
        bool          m_on = false;       ///< sounding
        bool          m_restart = true;   ///< its first block on the GPU is next
        CloudVoice    m_base;             ///< the voice as it was when it was let go
        std::uint32_t m_seed = 1;         ///< its own dice (new phases: it does not add to the voice it came from)
        float         m_age = 0.0f;       ///< seconds since it formed
        float         m_life = 60.0f;     ///< its fade (60 dB), from Memory Time when it formed
        float         m_bloom = 4.0f;     ///< its rise, from Bloom when it formed
        float         m_env = 0.0f;       ///< its level now, 0 .. 1 (the weakest is replaced first)
    };
    std::vector<Ghost> m_ghosts;     ///< the memory's ghosts, on the GPU voices after m_voices
    Rng                m_ghostRng;   ///< the ghosts' dice: their own stream, so the memory never moves the conductor
    Rng                m_noteRng;    ///< the Random per Note source's dice (its own stream)
    float              m_randomNote = 0.0f;      ///< the last note's draw, -1 .. 1
    float              m_roughness = 0.0f;       ///< the chord's roughness as last worked out
    int                m_roughTick = 0;          ///< blocks until it is worked out again
    std::vector<float> m_specL;                  ///< the spectral echo's block from the GPU, left
    std::vector<float> m_specR;                  ///< and right
    float              m_prevSpecMix = 0.0f;     ///< the spectral echo's level at the end of the last block
    float              m_prevSpecToFar = 0.0f;   ///< its send into the far plane, likewise
    bool               m_specWasOn = false;      ///< it ran in the last block (switching it on clears its lines)
    std::vector<float> m_meltL;                  ///< the melt's block from the GPU, left
    std::vector<float> m_meltR;                  ///< and right
    float              m_prevMeltMix = 0.0f;     ///< the melt's level at the end of the last block
    float              m_prevMeltToFar = 0.0f;   ///< its send into the far plane, likewise
    bool               m_meltWasOn = false;      ///< it ran in the last block (switching it on clears it)
    float              m_binAngle = 0.0f;        ///< the binaural cloud's orbit, radians
    double             m_combPhase = 0.0;        ///< the spectral comb's phase, notches
    double             m_pictureTime = 0.0;      ///< the animated pictures' flow, in their units
    double             m_pictureAngle = 0.0;     ///< their turn, radians
    double             m_pictureSeconds = 0.0;   ///< their real seconds
    float              m_paintTime = 0.0f;       ///< seconds into the landscape's loop, for the painted hall
    float              m_landLoop = 0.0f;        ///< the landscape's loop, seconds
    std::mutex         m_snapLock;               ///< guards m_snap (the worker only tries it: it never waits)
    std::vector<float> m_snap;                   ///< the last snapshot of the partials, for the picture
    int                m_snapPer = 0;            ///< its points per voice
    std::uint64_t      m_snapSeq = 0;            ///< snapshots published (under m_snapLock)
    std::uint64_t      m_snapSeen = 0;           ///< the last one the picture took (message thread)
    int                m_snapCountdown = 0;      ///< blocks until the next snapshot is asked for
    std::vector<float> m_snapLife;               ///< the living field of the last snapshot, empty when off
    int                m_lifeModeWas = 0;        ///< the living field's mode in the last block (a change seeds it)
    float              m_lifeAcc = 0.0f;         ///< its steps owed, carried between blocks
    double             m_lifeSeconds = 0.0;      ///< seconds the field has lived (the Fluid's jets lean by it)
    double             m_islandDrift = 0.0;      ///< the Islands' drift across, cells of their grid
    float m_turbDrive = 0.0f;   ///< Turbulence: the cascade's drive this block, per second, before each note's strength
    float m_turbAfter = 0.0f;   ///< s the cascade keeps settling after Turbulence went to nothing
    float m_cutoffPush =
        0.0f;   ///< the matrix's push on the cutoff this block, normalised (each of the morph's filters takes it)
    float m_resonancePush = 0.0f;   ///< and on the resonance
    float m_jamCutoff = 1.0f;       ///< the family jam's factor on the cutoff this block
    float m_lifeQuiet = 0.0f;       ///< seconds since it was last seeded (it is not seeded again sooner than 2 s)
    float m_lifeMakeup = 1.0f;      ///< the make-up as it glides (half a second), so a seeding does not jump
    /**
     * @brief A released voice's ghost into a free slot, or over the faintest.
     * @param[in] i  the voice
     */
    void captureGhost( int i );

    /** @brief The parameter values one block works with. */
    struct Block
    {
        float m_master = 1.0f;          ///< master gain, linear
        int   m_partials = 20000;       ///< partials per voice
        float m_tilt = 1.2f;            ///< spectral tilt
        float m_width = 25.0f;          ///< cloud width, cents, after Purity's narrowing
        float m_widthSlope = 0.3f;      ///< cloud widening
        float m_drift = 3.0f;           ///< cents
        float m_driftRate = 0.1f;       ///< Hz
        float m_shimmer = 0.4f;         ///< 0 .. 1
        float m_shimmerRate = 0.15f;    ///< Hz
        float m_stereo = 1.0f;          ///< 0 .. 1
        int   m_motion = 0;             ///< MotionKind
        float m_motionRate = 1.0f;      ///< times
        float m_motionDepth = 1.0f;     ///< times
        float m_attack = 6.0f;          ///< seconds
        float m_decay = 4.0f;           ///< seconds
        float m_sustain = 1.0f;         ///< 0 .. 1
        float m_release = 12.0f;        ///< seconds
        float m_refPitch = 440.0f;      ///< Hz
        float m_purity = 1.0f;          ///< 0 .. 1
        float m_depth = 0.6f;           ///< conductor's depth
        float m_keysDepth = 0.2f;       ///< played notes' distance
        float m_farLevel = 0.8f;        ///< far reverb level
        float m_brightness = 1.0f;      ///< brightness window
        float m_oddEven = 0.0f;         ///< odd/even
        float m_inharmonicB = 0.0f;     ///< the stiff string's B
        float m_stretch = 1.0f;         ///< the harmonics' stretch exponent (Stretch / 1200)
        float m_shift = 0.0f;           ///< Hz every partial is moved by
        float m_air = 0.0f;             ///< air band level
        float m_airCenter = 3.0f;       ///< its centre, times f0
        float m_airWidth = 600.0f;      ///< its width, cents
        float m_airJitter = 8.0f;       ///< its flicker, Hz
        int   m_filterModel = 0;        ///< FilterModel
        float m_cutoff = 2500.0f;       ///< Hz
        float m_resonance = 0.15f;      ///< 0 .. 1
        float m_filterEnv = 0.0f;       ///< octaves
        float m_keyTrack = 0.5f;        ///< 0 .. 1
        float m_filterDrift = 0.3f;     ///< 0 .. 1
        float m_presence = 0.0f;        ///< dB
        float m_padLowCut = 0.0f;       ///< Hz
        float m_breath = 0.0f;          ///< 0 .. 1
        float m_breathRate = 0.03f;     ///< Hz
        float m_doppler = 0.0f;         ///< 0 .. 1
        float m_tide = 0.0f;            ///< cents
        float m_tidePeriod = 12.0f;     ///< minutes
        bool  m_brain2 = false;         ///< the second conductor plays
        int   m_brain2Interval = -12;   ///< semitones
        float m_brain2Depth = 0.9f;     ///< its plane
    } m_blk;                            ///< read by readParams()

    std::vector<Event>         m_queue;        ///< audio thread -> renderer, a single-producer single-consumer ring
    std::atomic<std::uint32_t> m_qHead{ 0 };   ///< written by the audio thread
    std::atomic<std::uint32_t> m_qTail{ 0 };   ///< written by the renderer
    std::vector<Event>         m_pending;      ///< events drained but not yet due (renderer only)

    std::vector<float>        m_ring;             ///< the rendered timeline, interleaved stereo
    std::int64_t              m_ringMask = 0;     ///< frames in the ring minus one (a power of two)
    std::atomic<std::int64_t> m_renderPos{ 0 };   ///< timeline samples rendered (renderer writes)
    std::atomic<std::int64_t> m_outPos{ 0 };      ///< timeline samples handed out (audio thread writes)

    std::thread        m_worker;                   ///< the renderer in real-time mode
    std::atomic<bool>  m_running{ false };         ///< the worker runs
    std::atomic<bool>  m_stepAsked{ false };       ///< autoplayStep() was asked for
    std::atomic<bool>  m_fillAsked{ false };       ///< requestFill() was asked for
    std::atomic<int>   m_sounding{ 0 };            ///< voices sounding, for status
    std::atomic<int>   m_soundingBy[3] = {};       ///< voices sounding per source, for status
    std::atomic<int>   m_brainRoot{ 0 };           ///< the conductor's root, for status
    std::atomic<float> m_conductorNext{ 0.0f };    ///< conductorNext()
    std::atomic<float> m_conductorSpan{ 1.0f };    ///< conductorSpan()
    std::atomic<int>   m_foundationView{ -1 };     ///< foundationNow()
    double             m_conductorWas = 0.0;       ///< the countdown at the block before (a jump up begins a new wait)
    std::atomic<std::uint64_t> m_underruns{ 0 };   ///< callbacks short of samples
    std::atomic<std::uint64_t> m_late{ 0 };        ///< notes applied after their stamp
    std::atomic<float>         m_gpuMs{ 0.0f };    ///< GPU time of the last block
    std::atomic<int> m_worstBlockUs{ 0 };          ///< the slowest renderBlock since takeWorstBlockMs(), microseconds
    std::atomic<int> m_worstPrepUs{ 0 };           ///< its slowest preparation before the GPU, microseconds
    std::atomic<int> m_worstWaitUs{ 0 };           ///< its longest wait for the GPU
    std::atomic<int> m_worstPostUs{ 0 };           ///< its slowest effects after the GPU
    std::atomic<int> m_minFill{ -1 };              ///< the emptiest ring a callback found since takeMinFill(), samples
    std::atomic<int> m_maxCallback{ 0 };           ///< the largest callback since takeMaxCallback(), samples
    std::array<std::atomic<float>, MaxViewVoices> m_viewHz{};     ///< per voice its frequency for pictures (0 silent)
    std::array<std::atomic<float>, MaxViewVoices> m_viewGain{};   ///< its envelope
    std::array<std::atomic<float>, MaxViewVoices> m_viewDistance{};   ///< its distance
    std::array<std::atomic<float>, MaxViewVoices> m_viewTime{};       ///< its time since the note began
    std::array<std::atomic<int>, MaxViewVoices>   m_viewSource{};     ///< whose note it plays
    std::array<std::atomic<int>, MaxViewVoices>   m_viewNote{};       ///< the key it sounds, -1 none
    std::array<std::atomic<float>, MaxViewVoices> m_viewPress{};      ///< its MPE pressure
    std::array<std::atomic<float>, MaxViewVoices> m_viewSlide{};      ///< its MPE slide
    std::array<std::atomic<float>, MaxViewVoices> m_viewBend{};       ///< its MPE bend, semitones
    std::array<std::atomic<float>, MaxViewVoices> m_viewTune{};       ///< Self-Tune's pull on it, cents
    std::array<std::atomic<float>, 48>
                       m_viewScale{};             ///< the scale folded into an octave, cents over the root (scaleCents)
    std::atomic<int>   m_viewScaleCount{ 0 };     ///< how many
    std::atomic<float> m_viewRootHz{ 261.63f };   ///< the root's equal-tempered pitch (rootHz)
    std::array<std::atomic<float>, 128> m_viewNoteHz{};   ///< every key's pitch (noteHz)
    double m_noteHzKey[5] = { -1, -1, -1, -1, -1 };   ///< what they were worked out for: scale, root, A4, Purity, keys
};

}   // namespace additiv
