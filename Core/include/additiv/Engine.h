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
    double m_sunClock = 0.0;   ///< the night arc's clock at prepare, Unix seconds (0: the computer's); it runs on
                               ///< with the time rendered
};

/**
 * @brief The headset as the Quest app sends it (Phase 30): the head's turn and both hands, in the room as it stood when
 *        the app began (x to the right, y up, z ahead), from the head.
 */
struct HeadsetPose
{
    float m_yaw = 0.0f;     ///< rad: the head turned, positive to the right
    float m_pitch = 0.0f;   ///< rad: the head tilted, up positive
    float m_hand[2][3] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };   ///< left, right: x, y, z, m from the head
    float m_pinch[2] = { 0.0f, 0.0f };       ///< left, right: 0 open .. 1 the fingers touch
    bool  m_tracked[2] = { false, false };   ///< left, right: the headset sees the hand
    bool  m_valid = false;                   ///< a pose (false: none, or none for a second)
};

constexpr int DreamJournalMost = 1024;   ///< memories the dream journal keeps (Phase 30)

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
     * @brief What the weather, the bloom, the swarm and the physarum did in the last block (Phase 28; for tests and
     *        tools, offline: read on the thread that renders).
     * @param[out] out  receives 10 values: drops sounding, drops born, the wind's mean speed, the leaves' rustle, the
     *                  sea's foam, the bloom's share above 2 kHz, the swarm's orders R and S, the physarum's trail mean
     *                  and its spread
     */
    void natureStats( float *out ) const
    {
        m_cloud.weatherStats( out );
        m_cloud.physarumStats( out[8], out[9] );
    }

    /**
     * @brief The halo's carriers as the last block left them (Phase 29; tests and tools, offline).
     * @param[out] energy  receives HaloCarriers energies
     * @param[out] angle   receives where round the disc each one's energy stands, radians
     * @param[out] rim     receives each one's share on the rim
     */
    void haloStats( float *energy, float *angle, float *rim ) const { m_cloud.haloStats( energy, angle, rim ); }

    /**
     * @brief The quantum walk's moves (Phase 29; tests, offline).
     * @param[out] lastStep  receives the last move's step, semitones within the octave (0 .. 11)
     * @return the roots it moved since prepare()
     */
    long long quantumMoves( int &lastStep ) const
    {
        lastStep = m_qwLastStep;
        return m_qwMoves;
    }

    /**
     * @brief What the night did in the last block (Phase 30; tests and tools, offline).
          * @param[out] out  receives 6 values: the harp's strings' mean wake amplitude, the harmonic they sound, the ice's
     *                  last crack's size, the chirps begun, the strikes since prepare(), the last strike's distance (m)
     */
    void nightStats( float *out ) const
    {
        m_cloud.nightStats( out );
        out[4] = static_cast<float>( m_strikes );
        out[5] = m_lastStrikeM;
    }

    /**
     * @brief Where the night arc reads the sun (Phase 30): a place, from now on. Any thread.
     * @param[in] latitude   degrees, north positive
     * @param[in] longitude  degrees, east positive
     */
    void setLocation( double latitude, double longitude );
    /**
     * @brief Where the night arc reads the sun. Any thread.
     * @param[out] latitude   receives degrees, north positive
     * @param[out] longitude  receives degrees, east positive
     */
    void location( double &latitude, double &longitude ) const;
    /**
     * @brief The sun as the last block saw it, for the panel. Any thread.
     * @param[out] elevation  receives its elevation, degrees
     * @return the night it makes, 0 .. 1
     */
    float nightNow( float &elevation ) const
    {
        elevation = m_sunLive.load( std::memory_order_relaxed );
        return m_nightLive.load( std::memory_order_relaxed );
    }
    /**
     * @brief The headset's pose, from now on (Phase 30). Any thread.
     * @param[in] p  the head's turn and the hands (m_valid false: none)
     */
    void setHeadsetPose( const HeadsetPose &p );
    /**
     * @brief The dream journal (Phase 30): the dream's newest memories (up to DreamJournalMost), quantised to a byte a
     *        band, as the worker last brought them down (every half minute while the dream dreams) or as they were last
     *        set. Any thread but the worker's.
     * @param[out] out  receives the journal: "DRJ1", the count and the bands (32-bit little-endian), then per memory
     *                  its scale (a 32-bit float) and a byte a band
     * @return true when it holds any memory
     */
    bool dreamJournal( std::vector<std::uint8_t> &out );
    /**
     * @brief A dream journal to dream from (a state restored): the worker writes it to the card before its next block.
     *        Any thread but the worker's.
     * @param[in] in  a journal as dreamJournal() gives it (anything else is ignored)
     */
    void setDreamJournal( const std::vector<std::uint8_t> &in );
    /**
     * @brief What the night chorus did (Phase 31; tests and tools, offline).
     * @param[out] out  receives 4 values: the calls begun since prepare(), and as the last block left them the callers
     *                  calling, the most numerous kind's order (0 .. 1), the share waiting for a gap
     */
    void chorusStats( float *out ) const
    {
        m_cloud.chorusStats( out );
        out[0] = static_cast<float>( m_chorusCalls );
    }
    /** @brief Risset's pulse as the last block left it (Phase 30; tests). @return the voices' gain factor */
    float rissetGain() const { return m_rissetGain; }

    /** @brief Chimes that struck since the engine was prepared (Phase 27; tests, diagnostics). @return the count */
    long long chimeStrikes() const { return m_chimeStrikes.load( std::memory_order_relaxed ); }
    /** @brief Memories the dream keeps, as the last block left them (Phase 27). @return 0 .. DreamMemories */
    int dreamMemories() const { return m_cloud.dreamMemories(); }

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
    /** @brief Begins a merger at the next block, when Merger is up (Phase 33: for the tests). */
    void beginMerger() { m_mergerAsk.store( true, std::memory_order_relaxed ); }
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
     * @brief The pitches the phantom layer carries after the last block (Phase 27): its GPU voice's targets.
     * @param[out] hz      receives the pitches, Hz
     * @param[out] weight  receives each one's share of the full turn
     * @return how many, 0 .. PhantomTargets
     */
    int phantomPitches( float *hz, float *weight )
    {
        const CloudVoice &cv = m_cloud.voice( static_cast<int>( m_voices.size() + m_ghosts.size() ) );
        const int         n = cv.m_active ? std::clamp( cv.m_phantomCount, 0, PhantomTargets ) : 0;
        for ( int k = 0; k < n; ++k )
        {
            hz[k] = cv.m_phantomHz[k];
            weight[k] = cv.m_phantomWeight[k];
        }
        return n;
    }
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
        // Phase 26: the harmonics' own lives (stepHarmonics)
        std::array<float, 4> m_share{ 1.0f, 0.0f, 0.0f, 0.0f };        ///< Overblow: the registers' shares (1, 2, 3, 4)
        Drifter              m_blowWander;                             ///< Overblow: the breath's own slow unrest
        Rng                  m_pileDice;                               ///< Avalanche: the pile's dice, from the note's
        std::array<std::uint8_t, AvalancheSites> m_slope{};            ///< Avalanche: the pile's slope at each harmonic
        std::array<std::uint8_t, AvalancheSites> m_threshold{};        ///< the slope it topples over there (1 or 2)
        std::array<float, AvalancheSites>        m_glowTarget{};       ///< the glow a toppling lit, fading over Glow
        std::array<float, AvalancheSites>        m_glow{};             ///< the glow heard (the target, softened)
        float                                    m_grainDue = 0.0f;    ///< s until the next grain falls on the pile
        float                                    m_pileClock = 0.0f;   ///< s owed to the pile's generations
        float m_pileDepth = 0.0f;               ///< Avalanche as it glides (turning it off fades the glow)
        bool  m_pileLive = false;               ///< the pile is laid out for this note
        bool  m_subOn = false;                  ///< Subharmonics: clouds reserved for this note (decided at its start)
        std::array<float, SubClouds> m_sub{};   ///< their amplitudes, against harmonic 1's at rest
        float m_pre = 0.0f;        ///< Premonition: seconds before the note sounds (its swell meanwhile), 0 sounding
        float m_preTotal = 0.0f;   ///< the premonition's whole length for this note
        float m_suck = 0.0f;       ///< 1 .. 0 over 0.15 s after the note began: the swell drawn into it
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
    float              m_specKey[7] = {
        -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f
    };   ///< tilt, odd/even, brightness, inharmonic, stretch, horizon and its ratio the spectrum was built for
    Rng              m_rng;        ///< the engine's dice (seeds, distances)
    Rng              m_eventRng;   ///< the dice of the mergers (Phase 33), apart: they never shift the conductor's
    Rng              m_side;       ///< a side stream for the drifters, so switching them on never moves the notes' dice
    Drifter          m_tide;       ///< the whole instrument's slow lean (Tide)
    int              m_landChoice = -1;        ///< the landscape on the GPU
    Landscape        m_userLand;               ///< the player's own, as setUserLandscape left it (under m_userLandLock)
    float            m_userLandRms = 1.0f;     ///< its RMS
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
    // Phase 32: the floor
    float m_floorShare =
        0.0f;   ///< Floor: the share of the power under 150 Hz the foundation holds at the least (0: off)
    float  m_floorBoost = 1.0f;     ///< the foundation's depth (its partials under 2.2 f0) raised for it, 1 .. 8
    double m_floorLow = 0.0;        ///< the mix's power under 150 Hz, followed over two seconds
    double m_floorAll = 0.0;        ///< the mix's power, likewise
    double m_floorZ[2][2] = {};     ///< the low pass's state, left and right
    float  m_pedal = 0.0f;          ///< Pedal
    int    m_pedalPc = -1;          ///< the pitch class the foundation holds (-1: it follows the root)
    float  m_pedalWait = 0.0f;      ///< s the root has stood elsewhere
    float  m_floorSub = 0.0f;       ///< Sub Octave
    float  m_floorFifth = 0.0f;     ///< Fifth
    float  m_floorBreath = 0.0f;    ///< Breath
    double m_breathClock = 0.0;     ///< s, the breath's clock
    int    m_floorSubNote = -1;     ///< the sub octave's note, -1 none
    int    m_floorFifthNote = -1;   ///< the fifth's note, -1 none
    /** @brief The root the foundation stands on: the conductor's, or the pitch class Pedal holds. @return a MIDI note or pitch class (only its class counts) */
    int floorRoot() const { return m_pedalPc >= 0 ? m_pedalPc : m_brain.root(); }
    /**
     * @brief Pedal: the pitch class the foundation holds, against the root's moves for up to three minutes, or on the
     *        key's tonic at 1.
     * @param[in] dt  seconds since the last block
     */
    void stepPedal( float dt );
    /**
     * @brief Starts, moves or stops one of the foundation's voices beside it (the sub octave, the fifth).
     * @param[in,out] note  the note it holds (-1 none), updated
     * @param[in]     want  the note it should hold (-1 none)
     */
    void holdFloorVoice( int &note, int want );
    /**
     * @brief Floor: the mix's power under 150 Hz against all of it, followed; the foundation raised while the rest
     *        covers it, let down as it comes through.
     * @param[in] L  the mix, left
     * @param[in] R  the mix, right
     * @param[in] n  samples
     */
    void followFloor( const float *L, const float *R, int n );
    /**
     * @brief A foundation voice's level against the foundation's: 1 for the foundation, Sub Octave or Fifth for the
     *        voices beside it, each breathing in its own time (Breath).
     * @param[in] note  the voice's note
     * @return the factor
     */
    float floorVoiceLevel( int note ) const
    {
        if ( note == m_floorSubNote && note != m_foundationNote )
            return m_floorSub * ( 1.0f - 0.5f * m_floorBreath +
                                  0.5f * m_floorBreath * static_cast<float>( std::sin( 0.2416609 * m_breathClock ) ) );
        if ( note == m_floorFifthNote && note != m_foundationNote )
            return m_floorFifth *
                   ( 1.0f - 0.5f * m_floorBreath +
                     0.5f * m_floorBreath * static_cast<float>( std::sin( 0.1493547 * m_breathClock + 1.0 ) ) );
        return 1.0f;
    }
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
    /**
     * @brief One block of the seasons (Phase 26): five families of harmonics in a cyclic tournament --
     *        each beats the next two and loses more to the two before (replicator dynamics, Hofbauer and
     *        Sigmund), so the heteroclinic cycle draws them in, and a little mutation keeps it a regular
     *        round that neither dies out nor stalls: one round in Year seconds.
     * @param[in] dt  the block's length, seconds
     */
    void stepSeasons( float dt );
    /**
     * @brief One block of a voice's harmonics' own lives (Phase 26), written into its GPU copy: its
     *        registers (Overblow), the avalanches on its rice pile, the subharmonics its loud harmonics
     *        pump, the seasons' factors, and the make-up that keeps its power.
     * @param[in,out] v         the voice
     * @param[in,out] cv        its GPU copy (tilt, odd/even, brightness and pitch already set)
     * @param[in]     dt        the block's length, seconds
     * @param[in]     gain      its envelope this block
     * @param[in]     breath    MPE's Breath on it, 0 .. 1
     */
    void stepHarmonics( Voice &v, CloudVoice &cv, float dt, float gain, float breath );
    /**
     * @brief The Talbot timbre's eigen system for the mode and period (Phase 27), made again only when they change:
     *        Revival -- the cosine basis over the first TalbotHarmonics harmonics, its n-th mode turning at n^2 times
     *        2 pi over the period, so the amplitudes come back exactly after it (half way they stand mirrored, at a
     *        quarter both at once: Talbot's effect); Bloch -- a chain of harmonics with hopping and a tilt of 2 pi over
     *        the period from one to the next (a Wannier-Stark ladder), so the energy swings up and down the harmonics
     *        and comes back after the period (Bloch's oscillation). Jacobi's rotations for the chain.
     */
    void talbotSystem();
    /**
     * @brief The layers that are not notes, one block (Phase 27): the phantom layer's voice (the pitches it carries,
     *        folded into 220 .. 880 Hz), the dream's voice and its memory, the chime field's wind, gusts and pitches.
     * @param[in] dt  the block's length, seconds
     */
    void stepLayers( float dt );
    /**
     * @brief The weather, the bloom and the swarmalators, one block (Phase 28): their levels gliding, the wind
     *        blowing for the chimes or the weather (its gusts, its push), the rain's showers and pitches, the sea's
     *        height by the wind, the plate's size, the swarm's couplings.
     * @param[in] dt  the block's length, seconds
     */
    void stepNature( float dt );
    /**
     * @brief The halo, the wave room and the quantum walk, one block (Phase 29): the halo's carriers given to the
     *        sounding notes (a free slot, or the faintest) and driven at the rim where each stands, its speed and
     *        topology; the room's size, walls, listener and the voices standing in it; the walk evolving on the
     *        Tonnetz and, every so often, looked at -- the conductor's root moves to where it is found.
     * @param[in] dt  the block's length, seconds
     */
    void stepRooms( float dt );
    /**
     * @brief The night, one block (Phase 30): the harp's strings tuned to the chord (thick to thin, gliding), thunder's
     *        strikes drawn as they come (a Poisson stream; each its distance, direction, height, energy and channel),
     *        the ice's load from the cold and its thickness.
     * @param[in] dt  the block's length, seconds
     */
    void stepNight( float dt );
    /**
     * @brief Space, one block (Phase 33): the star's table rebuilt when its kind, age, spin, pitch or the root moved;
     *        the black hole's fall; a merger begun when its time comes (a Poisson stream) and its clock run on.
     * @param[in] dt  the block's length, seconds
     */
    void stepSpace( float dt );
    /**
     * @brief A merger, sample by sample (Phase 33), added to the mix before the master: the wave's tone in the bass
     *        once it is heard (its plus polarization in the middle, its cross on the sides), and the field's width
     *        stretched and squeezed by it while it is still too slow to hear.
     * @param[in,out] L  left
     * @param[in,out] R  right
     * @param[in]     n  samples
     */
    void renderMerger( float *L, float *R, int n );
    /**
     * @brief Light and depth, one block (Phase 34): the scintillation's screen drifting, the lines' and the absorption's
     *        settings to the card.
     * @param[in] dt  the block's length, seconds
     */
    void stepLight( float dt );
    /**
     * @brief The wreck, sample by sample (Phase 34), added to the near plane before the effects: its modes (Wreck.inc)
     *        rung by micrometeorites (a Poisson stream, each at one of its struck places) and thermal snaps (creaking
     *        runs of tiny slips, most at dusk and dawn).
     * @param[in,out] L  left
     * @param[in,out] R  right
     * @param[in]     n  samples
     */
    void renderWreck( float *L, float *R, int n );
    /**
     * @brief The deep channel, sample by sample (Phase 34): the mix heard again through the arrivals of rays across an
     *        ocean's sound channel (traced anew when Range or Off Axis moved), darkened by the water.
     * @param[in,out] L  left
     * @param[in,out] R  right
     * @param[in]     n  samples
     */
    void renderSofar( float *L, float *R, int n );
    /**
     * @brief Traces the deep channel's rays (Phase 34): Munk's profile c(z) = 1500 (1 + 0.00737 (eta + e^-eta - 1)),
     *        eta = 2 (z - 1300 m) / 1300 m; rays from the source by Snell's invariant, reflected at the surface (inverted)
     *        and the bottom (lossy); the arrivals at the range where neighbouring rays bracket the receiver on the axis,
     *        each as loud as its ray tube is narrow. Fills m_sofarTaps.
     * @param[in] rangeKm  the range
     * @param[in] offAxis  0 .. 1: the source on the axis .. 1500 m below it
     */
    void traceSofar( float rangeKm, float offAxis );
    /**
     * @brief The magnetosphere, sample by sample (Phase 35), added to the near plane before the effects: whistlers (a
     *        sferic's click dispersed along a field line, f(t) = (D / t)^2, with its echoes at 3 D and 5 D), the dawn
     *        chorus's rising elements in bursts (each sweeping faster the louder it is: Omura's nonlinear growth), and
     *        plasmaspheric hiss.
     * @param[in,out] L  left
     * @param[in,out] R  right
     * @param[in]     n  samples
     */
    void renderMagneto( float *L, float *R, int n );
    /**
     * @brief The throat of the abyss, sample by sample (Phase 35), added to the near plane before the effects: up to
     *        four Kelly-Lochbaum vocal tracts run at half the rate, each driven by a Rosenberg glottal pulse on the root
     *        (its fifth, its octave), the ventricular folds closing every other cycle for the octave below (kargyraa);
     *        their area along the tract a vowel's (u o a e i), slowly drifting.
     * @param[in,out] L  left
     * @param[in,out] R  right
     * @param[in]     n  samples
     */
    void renderThroat( float *L, float *R, int n );
    /**
     * @brief The headset, one block (Phase 30): the head's turn held off the binaural cloud and the thunder, the hands
     *        in the wind, the halo, the wave room and the physarum's food (after stepRooms, so its places are known).
     * @param[in] dt  the block's length, seconds
     */
    void stepHeadset( float dt );
    /**
     * @brief The dream journal, one block (Phase 30): a journal set goes onto the card first; while the dream dreams,
     *        its newest memories come down every half minute, a byte a band.
     * @param[in] dt  the block's length, seconds
     */
    void stepJournal( float dt );

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
    std::array<float, SeasonFamilies> m_seasonShare{ 0.6f, 0.1f, 0.1f, 0.1f, 0.1f };   ///< Seasons: each family's share
    float                             m_seasonDepth = 0.0f;   ///< Seasons as it glides (half a second)
    Rng m_blowRng;       ///< Overblow's dice (its own stream: switching it on never moves the other drifters)
    Rng m_breatherRng;   ///< the breathers' dice (their drift)
    // Phase 27
    std::array<float, TalbotHarmonics * TalbotHarmonics> m_talbotVec{};   ///< Talbot: the eigenvectors, one a row
    std::array<double, TalbotHarmonics>                  m_talbotVal{};   ///< their frequencies, radians a second
    float m_talbotKey[2] = { -1.0f, -1.0f };             ///< the mode and the period they were made for
    float m_gust = 0.0f;                                 ///< the chimes' gust, a slow random (3 s memory)
    Rng   m_windRng;                                     ///< its dice
    float m_chimeLevel = 0.0f;                           ///< the chime field's level as it glides
    bool  m_chimesOn = false;                            ///< the field ran in the last block
    float m_phantomGain = 0.0f;                          ///< the phantom layer's level as it glides
    Rng   m_phantomRng;                                  ///< the phantom melody's dice
    float m_melodyHz[2] = { 0.0f, 0.0f };                ///< the phantom melody: its note now and the one before
    float m_melodyWeight[2] = { 0.0f, 0.0f };            ///< their shares of the turn, one rising, one falling
    int   m_melodyAt = -1;                               ///< the note now, among the scale's pitches in 220 .. 880 Hz
    float m_melodyNext = 0.0f;                           ///< s until its next note
    std::array<float, PhantomTargets> m_echoHz{};        ///< the phantom echo: notes that ended lately, Hz
    std::array<float, PhantomTargets> m_echoAge{};       ///< and seconds since (fading over 30 s)
    int                               m_echoCount = 0;   ///< how many
    float                             m_dreamGain = 0.0f;    ///< the dream's level as it glides
    int                               m_dreamBlocks = 0;     ///< blocks since the dream last kept what it heard
    std::atomic<long long>            m_chimeStrikes{ 0 };   ///< chimes struck since prepare (chimeStrikes)
    std::vector<float>                m_phantomL;            ///< the phantom layer's block from the GPU, left
    std::vector<float>                m_phantomR;            ///< and right
    // Phase 28
    float              m_rainLevel = 0.0f;     ///< the rain's level as it glides
    float              m_leavesLevel = 0.0f;   ///< the leaves' level as it glides
    float              m_surfLevel = 0.0f;     ///< the sea's level as it glides
    float              m_bloomLevel = 0.0f;    ///< the bloom's level as it glides
    float              m_swarmDepth = 0.0f;    ///< the swarmalators' depth as it glides
    bool               m_rainWas = false;      ///< the rain fell in the last block
    bool               m_windWas = false;      ///< the wind blew in the last block
    bool               m_swarmWas = false;     ///< the swarm moved in the last block
    bool               m_physWas = false;      ///< the field was the physarum in the last block
    float              m_shower[4] = {};       ///< the showers' windward feed, slow randoms (10 s memory)
    Rng                m_showerRng;            ///< their dice
    std::vector<float> m_bloomL;               ///< the bloom's block from the GPU, left
    std::vector<float> m_bloomR;               ///< and right
    float              m_prevBloom = 0.0f;     ///< unused: the bloom's level is applied on the card
    // Phase 29
    float                           m_haloGain = 0.0f;    ///< the halo's level as it glides
    std::array<int, HaloCarriers>   m_haloNote{};         ///< the note each carrier carries (-1 free)
    std::array<float, HaloCarriers> m_haloAge{};          ///< seconds since it was given its note
    std::array<bool, HaloCarriers>  m_haloHeld{};         ///< its note still sounds
    std::array<short, HaloSites>    m_haloRim{};          ///< the rim's sites (their index), first m_haloRimCount
    std::array<float, HaloSites>    m_haloRimAngle{};     ///< and their angles round the disc
    int                             m_haloRimCount = 0;   ///< how many
    float                           m_roomLevel = 0.0f;   ///< the wave room's level as it glides
    int                    m_roomDims[3] = { 0, 0, 0 };   ///< its size the last block (a change clears the air)
    std::array<double, 24> m_qwPsi{};         ///< the quantum walk: 12 complex amplitudes (re, im) on the pitch classes
    float                  m_qwNext = 0.0f;   ///< s until it is looked at
    bool                   m_qwWas = false;   ///< it walked in the last block
    long long              m_qwMoves = 0;     ///< roots it moved (since prepare)
    int                    m_qwLastStep = 0;   ///< the last move's step, semitones within the octave
    int                    m_qwRootPc = -1;    ///< the root's pitch class as the walk last saw or set it
    Rng                    m_qwRng;            ///< its dice
    std::vector<float>     m_roomL;            ///< the wave room's block from the GPU, left
    std::vector<float>     m_roomR;            ///< and right
    // Phase 30
    float                          m_harpLevel = 0.0f;      ///< the harp's level as it glides
    bool                           m_harpWas = false;       ///< the harp sang in the last block
    std::array<float, HarpStrings> m_harpHz{};              ///< each string's tuning as it glides (0: not yet tuned)
    float                          m_thunderLevel = 0.0f;   ///< thunder's level as it glides
    float                          m_strikeIn = 0.0f;       ///< s until the next strike
    long long                      m_strikes = 0;           ///< strikes since prepare
    float                          m_lastStrikeM = 0.0f;    ///< the last strike's distance, m
    Rng                            m_stormRng;              ///< the storm's dice
    float                          m_iceLevel = 0.0f;       ///< the ice's level as it glides
    bool                           m_iceWas = false;        ///< the ice sang in the last block
    float                          m_crackIn = 0.0f;        ///< s until the cold brings the next crack
    double                         m_sunTime = 0.0;         ///< the night arc's clock, Unix seconds
    std::atomic<double>            m_latitude{ 45.0 };      ///< where the sun is read: degrees north
    std::atomic<double>            m_longitude{ 0.0 };      ///< and east
    float                          m_night = 0.0f;          ///< the night the sun makes, 0 .. 1 (this block)
    std::atomic<float>             m_nightLive{ 0.0f };     ///< the same, for the panel
    std::atomic<float>             m_sunLive{ 0.0f };       ///< the sun's elevation, degrees, for the panel
    double                         m_rissetPhase = 0.0;     ///< Risset: the slowest layer's beats, 0 .. 2^5
    float                          m_rissetU = 0.0f;        ///< its way through the octave, 0 .. 1
    float                          m_rissetGain = 1.0f;     ///< the voices' gain by its pulse this block
    float                          m_rissetPulse = 0.0f;    ///< the pulse, 0 .. 1 (the source)
    float                          m_lightning = 0.0f;      ///< the last strike's flash, 0 .. 1 (the source)
    // Phase 33: space
    float             m_starLevel = 0.0f;       ///< the star's level as it glides
    bool              m_starWas = false;        ///< the star sounded in the last block
    float             m_starKey[5] = {};        ///< what its table was built from: kind, age, spin, pitch, root (Hz)
    float             m_holeLevel = 0.0f;       ///< the black hole's amount as it glides
    double            m_holePhase = 0.0;        ///< its fall's clock, 0 .. 1
    double            m_redPhase = 0.0;         ///< the redshift hall's clock, 0 .. 1
    float             m_mergerWait = 0.0f;      ///< s until the next merger begins
    bool              m_mergerOn = false;       ///< a merger sounds
    double            m_mergerT = 0.0;          ///< s into it
    double            m_mergerTc = 0.0;         ///< s from its beginning to the coalescence
    double            m_mergerTauIsco = 0.0;    ///< s before the coalescence at the last stable orbit
    double            m_mergerPhase = 0.0;      ///< the wave's phase, radians
    double            m_mergerPhase3 = 0.0;     ///< the second ringdown mode's phase
    float             m_mergerIsco = 20.0f;     ///< Hz: the wave at the last stable orbit
    float             m_mergerRing = 70.0f;     ///< Hz: the ringdown's l = m = 2 mode
    float             m_mergerQ = 3.2f;         ///< its quality
    float             m_mergerRing3 = 110.0f;   ///< Hz: the l = m = 3 mode
    float             m_mergerQ3 = 5.0f;        ///< its quality
    float             m_mergerA3 = 0.1f;        ///< its share
    float             m_mergerEnv = 0.0f;       ///< the wave's amplitude, 0 .. 1 (the source)
    std::atomic<bool> m_mergerAsk{ false };     ///< beginMerger(): the next block begins one
    // Phase 34: light and depth
    double m_scintOffset = 0.0;   ///< the scintillation's screen, points
    /** @brief One of the wreck's modes: its complex state and its turn a sample. */
    struct WreckMode
    {
        double m_re = 0.0;    ///< state, real
        double m_im = 0.0;    ///< state, imaginary (heard)
        double m_cr = 1.0;    ///< the turn and the decay a sample, real
        double m_ci = 0.0;    ///< likewise, imaginary
        float  m_gl = 0.0f;   ///< the left pick-up's gain
        float  m_gr = 0.0f;   ///< the right's
    };
    std::vector<WreckMode> m_wreck;               ///< its modes (sized at prepare)
    float                  m_wreckKey[5] = {};    ///< what they were tuned for: shape, size, decay, root, level
    float                  m_wreckWait = 1.0f;    ///< s until the next micrometeorite
    float                  m_snapWait = 5.0f;     ///< s until the next run of creaks
    int                    m_creaks = 0;          ///< creaks left in this run
    float                  m_creakWait = 0.0f;    ///< s until its next creak
    int                    m_slips = 0;           ///< slips left in this creak
    int                    m_slipGap = 0;         ///< samples until the next slip
    int                    m_slipAt = 0;          ///< where it creaks (a struck place)
    float                  m_slipEnergy = 0.0f;   ///< how hard
    /** @brief One arrival of the deep channel: its delay and its gains. */
    struct SofarTap
    {
        int   m_delay = 0;   ///< samples after the first arrival
        float m_gl = 0.0f;   ///< left
        float m_gr = 0.0f;   ///< right
    };
    std::vector<SofarTap> m_sofarTaps;      ///< the arrivals
    std::vector<float>    m_sofarRing;      ///< the mix, mono, as long as the longest spread (sized at prepare)
    int                   m_sofarPos = 0;   ///< where the ring is written
    float                 m_sofarKey[2] = { -1.0f, -1.0f };   ///< what the taps were traced for: range, off axis
    float                 m_sofarCut = 1000.0f;               ///< Hz: the water's darkening at the range
    float                 m_sofarZ[2][2] = {};                ///< its low pass's state, left and right (two poles)
    // Phase 35: plasma and voice
    /** @brief A whistler (or one of its echoes): its time since the sferic, its dispersion, its phase. */
    struct Whistler
    {
        bool   m_on = false;    ///< sounding
        double m_t = 0.0;       ///< s since its sferic (from where it is at 10 kHz)
        double m_D = 50.0;      ///< s sqrt(Hz): its dispersion
        double m_phase = 0.0;   ///< radians
        float  m_amp = 0.0f;    ///< its loudness
        float  m_pan = 0.0f;    ///< -1 .. 1
    };
    std::array<Whistler, 12> m_whistlers{};   ///< the whistlers sounding
    /** @brief One element of the dawn chorus: a rising tone. */
    struct ChorusTone
    {
        bool   m_on = false;    ///< sounding
        double m_f = 0.0;       ///< Hz now
        double m_rate = 0.0;    ///< Hz a second it rises
        double m_t = 0.0;       ///< s into it
        double m_len = 0.3;     ///< s it lasts
        double m_phase = 0.0;   ///< radians
        float  m_amp = 0.0f;    ///< its loudness
        float  m_pan = 0.0f;    ///< -1 .. 1
    };
    std::array<ChorusTone, 24> m_tones{};               ///< the chorus's elements sounding
    float                      m_whistlerWait = 5.0f;   ///< s until the next whistler
    float                      m_burstWait = 2.0f;      ///< s until the next burst of the chorus
    int                        m_burstLeft = 0;         ///< elements left in this burst
    float                      m_toneWait = 0.0f;       ///< s until its next element
    float                      m_hissZ[2][2] = {};      ///< the hiss's band pass, left and right
    int                        m_strikesSeen = 0;       ///< thunder's strikes already given a whistler
    long long                  m_lightningCount = 0;    ///< thunder's strikes since prepare
    /** @brief One throat: its tract's waves and its voice. */
    struct Throat
    {
        std::vector<float> m_fwd;           ///< the waves going to the lips, a section each
        std::vector<float> m_bwd;           ///< the waves going back to the glottis
        std::vector<float> m_k;             ///< the reflection at each junction
        double             m_phase = 0.0;   ///< the glottal cycle, 0 .. 2 (two cycles: the folds' period doubling)
        double             m_hz = 0.0;      ///< its pitch now (gliding)
        float              m_last = 0.0f;   ///< the lips' last output (for the radiation's derivative)
        float              m_prev = 0.0f;   ///< the output at the half-rate step before (for the interpolation)
        float              m_now = 0.0f;    ///< and at this one
        float              m_pan = 0.0f;    ///< -1 .. 1
    };
    std::array<Throat, 4>     m_throats{};                         ///< up to four
    float                     m_throatKey[2] = { -1.0f, -1.0f };   ///< what the tracts were shaped for: length, vowel
    double                    m_throatClock = 0.0;                 ///< s, for the vowels' drift and the vibrato
    bool                      m_throatOdd = false;                 ///< the half-rate tract steps every other sample
    std::mutex                m_poseLock;                          ///< guards m_posePending
    HeadsetPose               m_posePending;                       ///< the pose as it arrived
    HeadsetPose               m_pose;                              ///< the pose this block works with
    std::vector<std::uint8_t> m_journal;                           ///< the dream journal brought down last (m_snapLock)
    std::vector<std::uint8_t> m_journalPending;                    ///< a journal to write to the card (m_snapLock)
    bool                      m_journalPendingSet = false;         ///< journalPending waits for the worker
    float                     m_journalClock = 0.0f;               ///< s until the journal is brought down again
    std::vector<float>        m_journalScratch;                    ///< the worker's memories on their way (no lock)
    float                     m_headYaw = 0.0f;                    ///< rad: the head's turn held off the binaural cloud
    float                     m_handFood[2] = { -1.0f, -1.0f };    ///< the hands' food lines, trail cells (-1 none)
    float                     m_handFoodWeight = 1.0f;             ///< their weight
                                                                   // Phase 31
    float     m_chorusLevel = 0.0f;                                ///< the night chorus's level as it glides
    long long m_chorusCalls = 0;                                   ///< the chorus's calls begun since prepare
    bool      m_chorusWas = false;                                 ///< the chorus called in the last block
    int       m_mirrorPos = 0;    ///< the time mirror: where in its window the next block begins (the room's samples)
    int       m_mirrorBuf = 0;    ///< which of its recordings records
    float     m_roomDc[4] = {};   ///< the wave room's output high pass (Phase 36): last in, left and right; last out
    double    m_mirrorDrySmooth = 0.0;   ///< the mirror's governor (Phase 36): the voices' energy a block, smoothed
    double    m_mirrorDry = 0.0;         ///< ... held over the last seconds (MirrorHold), what it weighs by
    double    m_mirrorWet = 0.0;         ///< ... the room's at its ears, its level divided out, smoothed
    float     m_mirrorGovern = 1.0f;   ///< the share of its gain the mirror may send back (1 unless it would run away)
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
        float m_master = 1.0f;            ///< master gain, linear
        int   m_partials = 20000;         ///< partials per voice
        float m_tilt = 1.2f;              ///< spectral tilt
        float m_width = 25.0f;            ///< cloud width, cents, after Purity's narrowing
        float m_widthSlope = 0.3f;        ///< cloud widening
        float m_drift = 3.0f;             ///< cents
        float m_driftRate = 0.1f;         ///< Hz
        float m_shimmer = 0.4f;           ///< 0 .. 1
        float m_shimmerRate = 0.15f;      ///< Hz
        int   m_streams = 0;              ///< Streams: how the partials part (StreamsE)
        float m_streamDepth = 0.0f;       ///< Streams: their motion, 0 .. 1
        float m_streamRate = 2.0f;        ///< Streams: Hz
        float m_fusion = 0.0f;            ///< Streams: 0 apart .. 1 as one
        float m_pressure = 0.0f;          ///< Pressure: 0 .. 1 (Phase 32)
        float m_pressureRate = 0.12f;     ///< Pressure: Hz
        float m_stereo = 1.0f;            ///< 0 .. 1
        int   m_motion = 0;               ///< MotionKind
        float m_motionRate = 1.0f;        ///< times
        float m_motionDepth = 1.0f;       ///< times
        float m_attack = 6.0f;            ///< seconds
        float m_decay = 4.0f;             ///< seconds
        float m_sustain = 1.0f;           ///< 0 .. 1
        float m_release = 12.0f;          ///< seconds
        float m_refPitch = 440.0f;        ///< Hz
        float m_purity = 1.0f;            ///< 0 .. 1
        float m_depth = 0.6f;             ///< conductor's depth
        float m_keysDepth = 0.2f;         ///< played notes' distance
        float m_farLevel = 0.8f;          ///< far reverb level
        float m_brightness = 1.0f;        ///< brightness window
        float m_oddEven = 0.0f;           ///< odd/even
        float m_inharmonicB = 0.0f;       ///< the stiff string's B
        float m_stretch = 1.0f;           ///< the harmonics' stretch exponent (Stretch / 1200)
        float m_shift = 0.0f;             ///< Hz every partial is moved by
        float m_air = 0.0f;               ///< air band level
        float m_airCenter = 3.0f;         ///< its centre, times f0
        float m_airWidth = 600.0f;        ///< its width, cents
        float m_airJitter = 8.0f;         ///< its flicker, Hz
        int   m_filterModel = 0;          ///< FilterModel
        float m_cutoff = 2500.0f;         ///< Hz
        float m_resonance = 0.15f;        ///< 0 .. 1
        float m_filterEnv = 0.0f;         ///< octaves
        float m_keyTrack = 0.5f;          ///< 0 .. 1
        float m_filterDrift = 0.3f;       ///< 0 .. 1
        float m_presence = 0.0f;          ///< dB
        float m_padLowCut = 0.0f;         ///< Hz
        float m_breath = 0.0f;            ///< 0 .. 1
        float m_breathRate = 0.03f;       ///< Hz
        float m_doppler = 0.0f;           ///< 0 .. 1
        float m_tide = 0.0f;              ///< cents
        float m_tidePeriod = 12.0f;       ///< minutes
        bool  m_brain2 = false;           ///< the second conductor plays
        int   m_brain2Interval = -12;     ///< semitones
        float m_brain2Depth = 0.9f;       ///< its plane
        float m_overblow = 0.0f;          ///< Overblow, 0 .. 1
        float m_overblowTime = 0.6f;      ///< s
        float m_seasons = 0.0f;           ///< Seasons, 0 .. 1
        float m_seasonTime = 240.0f;      ///< s, a whole round
        float m_avalanche = 0.0f;         ///< Avalanche, 0 .. 1
        float m_avalancheRate = 0.3f;     ///< grains a second
        float m_avalancheGlow = 0.6f;     ///< s
        float m_subharmonics = 0.0f;      ///< Subharmonics, 0 .. 1
        float m_subTime = 1.0f;           ///< s
        float m_horizon = 0.0f;           ///< Horizon, 0 .. 1
        float m_horizonRatio = 8.0f;      ///< the horizon over the fundamental
        float m_grain = 0.0f;             ///< Grain, 0 .. 1
        float m_grainLength = 0.08f;      ///< s
        float m_grainDuty = 0.3f;         ///< Grain Density, 0 .. 1
        float m_talbot = 0.0f;            ///< Talbot, 0 .. 1
        float m_talbotMode = 0.0f;        ///< 0 Revival, 1 Bloch
        float m_talbotTime = 60.0f;       ///< s
        float m_premonition = 0.0f;       ///< Premonition, 0 .. 1
        float m_premonitionTime = 5.0f;   ///< s
    } m_blk;                              ///< read by readParams()

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
