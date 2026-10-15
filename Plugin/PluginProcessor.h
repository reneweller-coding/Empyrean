/**
 * @file PluginProcessor.h
 * @brief The JUCE shell around the engine: parameters from the one table, MIDI into stamped notes
 *        (with the family's controllers: 64 holds the notes, 74 moves the filter, and MIDI learn for
 *        the rest), the engine's latency reported to the host, the state as key=value pairs.
 *
 * The GPU opens in prepareToPlay, never in the constructor: the VST3 build step loads the plugin to
 * read its description, and a scan must not wake the GPU.
 * @author Rene Weller
 */
#pragma once

#include "Jam.h"

#include "additiv/Engine.h"
#include "additiv/Journey.h"
#include "additiv/Landscape.h"
#include "additiv/PresetMap.h"
#include "additiv/Route.h"

#include <juce_audio_utils/juce_audio_utils.h>

#include <array>
#include <atomic>
#include <bitset>
#include <mutex>
#include <string>
#include <vector>

/** @brief The plugin and the standalone: the engine behind JUCE's processor interface. */
class AdditivProcessor : public juce::AudioProcessor, private juce::AsyncUpdater, private juce::Timer
{
public:
    AdditivProcessor();
    ~AdditivProcessor() override;

    /**
     * @brief Opens the engine at the host's rate and block size.
     * @param[in] sampleRate       Hz
     * @param[in] samplesPerBlock  the block the host announced
     */
    void prepareToPlay( double sampleRate, int samplesPerBlock ) override;
    /** @brief Stops the engine's worker and frees the GPU. */
    void releaseResources() override;
    /**
     * @brief Stereo out, nothing in.
     * @param[in] layouts  the layout asked for
     * @return true for stereo out
     */
    bool isBusesLayoutSupported( const BusesLayout &layouts ) const override;
    /**
     * @brief Parameters, controllers and notes into the engine, its samples out.
     * @param[in,out] buffer  the output
     * @param[in,out] midi    the notes of this block
     */
    void processBlock( juce::AudioBuffer<float> &buffer, juce::MidiBuffer &midi ) override;

    /** @brief The family's panel (PluginEditor.h). @return a new editor */
    juce::AudioProcessorEditor *createEditor() override;
    /** @brief There is an editor. @return true */
    bool hasEditor() const override { return true; }

    /** @brief The plugin's name. @return the product name */
    const juce::String getName() const override { return JucePlugin_Name; }
    /** @brief It plays notes. @return true */
    bool acceptsMidi() const override { return true; }
    /** @brief It sends none. @return false */
    bool producesMidi() const override { return false; }
    /** @brief MPE: every played note may come on its own channel with its own bend, pressure and slide. @return true */
    bool supportsMPE() const override { return true; }
    /** @brief The far reverb rings on. @return seconds */
    double getTailLengthSeconds() const override { return 30.0; }

    /** @brief The factory presets are the programs. @return their count */
    int getNumPrograms() override;
    /** @brief The preset last applied. @return its index (0 when none was) */
    int getCurrentProgram() override { return std::max( m_current.load( std::memory_order_relaxed ), 0 ); }
    /**
     * @brief Applies a factory preset (message thread).
     * @param[in] index  the preset
     * @pre called on the message thread
     */
    void setCurrentProgram( int index ) override
    {
        choosePreset( index, false );
    }   // the host's: not one the player played
    /**
     * @brief A preset's name with its group.
     * @param[in] index  the preset
     * @return "Group: Name"
     */
    const juce::String getProgramName( int index ) override;
    /**
     * @brief Nothing to rename.
     * @param[in] index  ignored
     * @param[in] name   ignored
     */
    void changeProgramName( int index, const juce::String &name ) override
    {
        (void)index;
        (void)name;
    }

    /**
     * @brief Every parameter as key=value, and the learned controllers.
     * @param[out] dest  receives the XML
     */
    void getStateInformation( juce::MemoryBlock &dest ) override;
    /**
     * @brief Reads key=value; a parameter the state does not name goes to its default.
     * @param[in] data  the XML
     * @param[in] size  its bytes
     */
    void setStateInformation( const void *data, int size ) override;

    /** @brief The engine, for the editor's status and pictures. @return the engine */
    additiv::Engine &engine() { return m_engine; }
    /**
     * @brief A parameter by its ParamId.
     * @param[in] id  the parameter's index
     * @return the host parameter, null outside the table
     */
    juce::RangedAudioParameter *parameter( int id ) const
    {
        return id >= 0 && id < static_cast<int>( m_params.size() ) ? m_params[static_cast<size_t>( id )] : nullptr;
    }
    /**
     * @brief A parameter's value in its unit.
     * @param[in] id  ParamId
     * @return the value
     */
    float value( int id ) const;
    /**
     * @brief Sets a parameter in its unit, telling the host.
     * @param[in] id  ParamId
     * @param[in] v   the value
     */
    void setValue( int id, float v );

    /**
     * @brief MIDI learn: the next controller that moves is bound to parameter @p id.
     * @param[in] id  ParamId
     */
    void learn( int id ) { m_learning.store( id, std::memory_order_relaxed ); }
    /**
     * @brief The controller bound to a parameter.
     * @param[in] id  ParamId
     * @return 0 .. 127, -1 none
     */
    int controllerFor( int id ) const;
    /**
     * @brief Unbinds a parameter's controller.
     * @param[in] id  ParamId
     */
    void forget( int id );
    /**
     * @brief A parameter back to its default.
     * @param[in] id  ParamId
     */
    void resetToDefault( int id );
    /** @brief Whether a learn is waiting for its controller. @return the parameter waiting, -1 none */
    int learning() const { return m_learning.load( std::memory_order_relaxed ); }

    /**
     * @brief Applies a factory preset: every parameter to the preset's value (its default where the
     *        preset says nothing), the whole matrix replaced. Message thread.
     * @param[in] index  the preset
     * @pre called on the message thread
     */
    void applyPreset( int index ) { applyPreset( index, -1.0 ); }
    /**
     * @brief Applies a factory preset, gliding into it (Morph.h): the knobs go to the preset at once,
     *        the sound follows over `glide` seconds.
     * @param[in] index  the preset
     * @param[in] glide  seconds; below 0 the Glide knob's; 0 at once (and the conductor fills its chord)
     */
    void applyPreset( int index, double glide );
    /**
     * @brief The player -- or the host, or a hand on the headset -- chooses a preset: that preset is
     *        what is heard. Whatever was choosing the sound lets go (a journey, a route, the A/B morph,
     *        the map, whose cursor goes to the preset's point so the map shows where the sound is),
     *        and the sound glides into the preset as applyPreset does. Under the A/B morph or the map
     *        a preset changed only the knobs, which they do not play: the menu seemed broken.
     *        Message thread.
     * @param[in] index   the preset
     * @param[in] recent  it joins the presets played lately (the panel's choices, MIDI; not the host's)
     */
    void choosePreset( int index, bool recent = true );

    /** @name The A/B morph's slots (Morph.h)
     *  @{ */
    /**
     * @brief A slot takes the sound as it plays now: every knob and the matrix.
     * @param[in] slot  0 (A) or 1 (B)
     */
    void setSlotFromSound( int slot );
    /**
     * @brief A slot takes a factory preset.
     * @param[in] slot   0 or 1
     * @param[in] index  the preset
     */
    void setSlotFromPreset( int slot, int index );
    /**
     * @brief What a slot holds, in words.
     * @param[in] slot  0 or 1
     * @return a preset's name, "the sound of ...", or "empty"
     */
    juce::String slotName( int slot ) const
    {
        return m_slotNames[slot & 1].isEmpty() ? juce::String( "empty" ) : m_slotNames[slot & 1];
    }
    /** @} */

    /** @name The player's own landscape
     *  @{ */
    /**
     * @brief Reads a picture (PNG, JPEG, BMP, TIFF, GIF: time left to right over a minute, low at the
     *        bottom) or a recording (WAV, FLAC: its own motion) as the landscape, and chooses it.
     * @param[in]  file        the file
     * @param[in]  letItSound  also switch the motion to Aurora when nothing reads the landscape yet
     * @param[out] error       receives why it could not be read
     * @return true when it plays
     */
    bool loadLandscape( const juce::File &file, bool letItSound, juce::String *error );
    /**
     * @brief The player's own landscape (empty when none). Message thread.
     * @return it
     * @pre called on the message thread
     */
    const additiv::Landscape &userLandscape() const { return m_userLand; }
    /** @brief Bumped whenever the player's landscape changes (for the picture). @return the count */
    int userLandscapeVersion() const { return m_userLandVersion; }
    /** @brief The file the player's landscape came from. @return it, or an empty File */
    juce::File userLandscapeFile() const { return m_userLandFile; }
    /** @} */

    /** @name The preset map (PresetMap.h) and its routes (Route.h)
     *  @{ */
    /**
     * @brief The map's cursor moves there (the map switched on, a route stopped: the hand has it).
     * @param[in] x  across
     * @param[in] y  up
     */
    void moveCursor( float x, float y );
    /**
     * @brief The preset nearest a place on the map.
     * @param[in] x  across
     * @param[in] y  up
     * @return its index
     */
    int nearestPreset( float x, float y ) const;
    /** @brief The route being played or written. @return it */
    const additiv::Route &route() const { return m_route; }
    /** @brief Its name (a built-in's, a file's, or empty). @return the name */
    juce::String routeName() const { return m_routeName; }
    /** @brief A route to choose: one that comes with the instrument, or a file of the player's. */
    struct RouteEntry
    {
        juce::String m_name;           ///< as the list shows it
        int          m_builtin = -1;   ///< the built-in route, -1 for a file
        juce::File   m_file;           ///< the file, for the player's own
    };
    /** @brief Where the player's routes live: Documents/Empyrean/Routes. @return the folder */
    static juce::File routeFolder();
    /** @name The player's own presets (Documents/Empyrean/Presets)
     *  @{ */
    /** @brief One of the player's presets. */
    struct UserPreset
    {
        juce::String       m_name;     ///< its name (its file's)
        std::vector<float> m_values;   ///< its knobs, NumParams values (those it does not name at their defaults)
        std::string        m_mod;      ///< its matrix
        juce::File         m_file;     ///< its file
    };
    /** @brief Where the player's presets live: Documents/Empyrean/Presets. @return the folder */
    static juce::File presetFolder();
    /** @brief The player's presets as last read from the folder, by name. @return them */
    const std::vector<UserPreset> &userPresets() const { return m_userPresets; }
    /** @brief Reads the folder again. */
    void refreshUserPresets();
    /**
     * @brief Saves the sound now (every knob a preset sets, and the matrix) as one of the player's
     *        presets; it becomes the preset playing.
     * @param[in]  name  its name (a factory preset's name is refused: journeys find presets by name)
     * @param[out] why   receives the reason when it cannot
     * @return true when written
     */
    bool saveUserPreset( const juce::String &name, juce::String &why );
    /**
     * @brief The player chooses one of their presets (a journey and the A/B morph let go).
     * @param[in] i  its index in userPresets()
     */
    void chooseUserPreset( int i );
    /**
     * @brief One of the player's presets by name.
     * @param[in] name  its name
     * @return its index, -1 none
     */
    int findUserPreset( const juce::String &name ) const;
    /** @brief The player's preset playing, by name ("" when a factory one is, or none). @return it */
    juce::String currentUserPreset() const { return m_currentUser; }
    /** @brief The knobs of the preset the sound came from (what "changed" is measured against). @return NumParams values, empty before any */
    const std::vector<float> &reference() const { return m_reference; }
    /** @brief Whether the sound has moved from its preset: a knob a preset sets, or the matrix. @return true then */
    bool modified() const;
    /** @brief Back to the preset the sound came from, gliding over a second. */
    void revert();
    /**
     * @brief One knob back to its preset's value.
     * @param[in] id  the knob
     */
    void revertKnob( int id );
    /** @} */

    /** @name Snapshots (the Focus page)
     *  @{ */
    /** @brief A moment of the sound kept: its knobs, its matrix, its picture. */
    struct Snapshot
    {
        std::vector<float> m_values;    ///< the knobs, NumParams values
        std::string        m_mod;       ///< the matrix
        juce::Image        m_picture;   ///< the partials as they looked (none after a session was reloaded)
    };
    /** @brief The snapshots, the oldest first (at most eight). @return them */
    const std::vector<Snapshot> &snapshots() const { return m_snapshots; }
    /**
     * @brief Keeps the sound now (the oldest gives way to a ninth).
     * @param[in] picture  what the partials look like
     */
    void takeSnapshot( const juce::Image &picture );
    /**
     * @brief Back to a snapshot, gliding over a second.
     * @param[in] i  which
     */
    void recallSnapshot( int i );
    /**
     * @brief Forgets a snapshot.
     * @param[in] i  which
     */
    void forgetSnapshot( int i );
    /**
     * @brief A snapshot into the A/B morph's slot.
     * @param[in] i     which
     * @param[in] slot  0 (A) or 1 (B)
     */
    void snapshotToSlot( int i, int slot );
    /** @} */

    /** @name Browsing (the preset browser)
     *  @{ */
    /**
     * @brief Listens to a factory preset: the sound glides there, until endPreview() (or it is chosen).
     * @param[in] index  the preset
     */
    void previewPreset( int index );
    /**
     * @brief Listens to one of the player's presets.
     * @param[in] i  its index in userPresets()
     */
    void previewUserPreset( int i );
    /** @brief Back to the sound before the listening, gliding. */
    void endPreview();
    /** @brief Whether a preset is being listened to. @return true then */
    bool previewingPreset() const { return m_browsing; }
    /**
     * @brief Whether a preset is one of the player's favourites.
     * @param[in] name  its name
     * @return true then
     */
    bool favorite( const juce::String &name ) const;
    /**
     * @brief Marks or unmarks a favourite.
     * @param[in] name  its name
     * @param[in] on    marked
     */
    void setFavorite( const juce::String &name, bool on );
    /** @brief The presets chosen lately, the newest first (at most 40). @return their names */
    juce::StringArray recentPresets() const;
    /** @brief The player's own settings (favourites, recent, the night mode, the tour), in their application data. @return them */
    juce::PropertiesFile &userSettings();
    /** @} */

    /** @name The Focus page (Focus.h, Find.h)
     *  @{ */
    /** @brief The knobs as they stand. @return NumParams values */
    std::vector<float> knobValues() const;
    /**
     * @brief Sets the knobs a preset sets (the listener's stay), the engine gliding there.
     * @param[in] v      NumParams values
     * @param[in] glide  seconds (0: at once)
     */
    void applyValues( const std::vector<float> &v, double glide );
    /**
     * @brief A nudge: every heard shaping knob a little towards brighter, livelier, wider or airier (or
     *        the other way), the level held.
     * @param[in] axis  0 .. NudgeAxes-1
     * @param[in] sign  +1 or -1
     * @return the predicted change, in spreads of the measure
     */
    float nudgeSound( int axis, float sign );
    /**
     * @brief Six variations around the sound now (Focus.h vary).
     * @param[in] far  how far, 0 .. 1
     */
    void makeVariations( float far );
    /** @brief How many variations there are. @return 0 or 6 */
    int variationCount() const { return static_cast<int>( m_variations.size() ); }
    /**
     * @brief Listens to a variation, gliding there (-1: back to the sound before, gliding).
     * @param[in] k  which
     */
    void previewVariation( int k );
    /** @brief Which variation is being listened to. @return it, -1 none */
    int previewing() const { return m_previewing; }
    /**
     * @brief Keeps a variation: it becomes the sound, and the next six are made around it.
     * @param[in] k  which
     */
    void takeVariation( int k );
    /** @brief Back to the sound before the last variation kept, and six around it. @return false when there is none */
    bool variationBack();
    /**
     * @brief What a variation changes most, in words.
     * @param[in] k  which
     * @return the text
     */
    juce::String variationSummary( int k ) const;
    /** @brief The sound the variations were made around (empty before the first). @return its values */
    const std::vector<float> &variationParent() const { return m_parent; }
    /** @brief How often the variations were made anew (a page sees new ones by it). @return the count */
    int variationGeneration() const { return m_variationGeneration; }
    /** @brief Whether they can be made now without leaving a blend (the map, the morph). @return true then */
    bool variationsFree() const;
    /**
     * @brief Compare: while held, the preset as it was (its knobs and matrix, gliding there); let go, the sound
     *        as it is again. No undo step: nothing changes.
     * @param[in] on  held
     */
    void compareWithPreset( bool on );
    /** @brief Whether the preset is being compared with. @return true then */
    bool comparing() const { return m_comparing; }
    /**
     * @brief The output's peaks since the last look, per side (the meter in the header); they reset as they are read.
     * @param[out] left   receives the left side's
     * @param[out] right  receives the right side's
     */
    void takePeaks( float &left, float &right )
    {
        left = m_peak[0].exchange( 0.0f, std::memory_order_relaxed );
        right = m_peak[1].exchange( 0.0f, std::memory_order_relaxed );
    }
    /**
     * @brief Begins to teach a macro: the sound now is its place at its value now.
     * @param[in] m  0 .. 3
     */
    void teachBegin( int m );
    /**
     * @brief The sound now becomes the macro's place at its value now; the knobs go back to where the
     *        teaching began (the macro carries the change from now on, so nothing is heard to move).
     * @param[in] m  0 .. 3
     * @return false when the macro is not being taught
     */
    bool teachHere( int m );
    /**
     * @brief Forgets what a macro was taught.
     * @param[in] m  0 .. 3
     */
    void teachForget( int m );
    /**
     * @brief Ends a teaching without a new place: the knobs back where it began, the lesson as it was.
     * @param[in] m  0 .. 3
     */
    void teachCancel( int m );
    /**
     * @brief Whether a macro has a lesson (two places at least).
     * @param[in] m  0 .. 3
     * @return true then
     */
    bool taught( int m ) const { return m >= 0 && m < 4 && m_lessons[static_cast<size_t>( m )].m_count >= 2; }
    /**
     * @brief Whether a macro is being taught.
     * @param[in] m  0 .. 3
     * @return true then
     */
    bool teaching( int m ) const { return m >= 0 && m < 4 && !m_teachBase[static_cast<size_t>( m )].empty(); }
    /**
     * @brief A macro's name: what it moves most -- the knob its lesson moves furthest, or the target of
     *        its strongest matrix row ("Motion Depth"), "and n more" when it moves several; "Macro n" when
     *        it moves nothing.
     * @param[in] m  0 .. 3
     * @return the name
     */
    juce::String macroName( int m ) const;
    /**
     * @brief What a macro was taught, in words.
     * @param[in] m  0 .. 3
     * @return the text
     */
    juce::String teachStatus( int m ) const;
    /**
     * @brief The presets that sound most like a recording (Find.h).
     * @param[in]  file  WAV or FLAC
     * @param[out] why   receives the reason when it cannot be read
     * @return false when it cannot; found() holds them, nearest first
     */
    bool findLike( const juce::File &file, juce::String &why );
    /** @brief The presets the last search found, nearest first. @return them */
    const std::vector<int> &found() const { return m_found; }
    /** @brief Their distances, in spreads. @return them */
    const std::vector<float> &foundDistance() const { return m_foundDistance; }
    /** @} */

    /** @brief Where the player's halls live: Documents/Empyrean/Halls. @return the folder */
    static juce::File hallFolder();
    /**
     * @brief Writes the hall's memory (Hall page) to a file.
     * @param[in] file  where
     * @return false when it cannot
     */
    bool saveHall( const juce::File &file );
    /**
     * @brief Hears a hall's memory in from a file (from the next block).
     * @param[in] file  which
     * @return false when it holds no hall
     */
    bool loadHall( const juce::File &file );
    /** @brief A new hall: everything it has heard forgotten. */
    void newHall();
    /** @brief The hall's name: the file it was last loaded from or saved to, empty for one never saved. @return the name */
    juce::String hallName() const { return m_hallName; }
    /** @brief The routes to choose from: the built-ins, then the player's files. @return them */
    std::vector<RouteEntry> routes() const;
    /**
     * @brief Loads a route (it does not start).
     * @param[in] entry  which
     * @return false when it cannot be read
     */
    bool loadRoute( const RouteEntry &entry );
    /** @brief Starts the route from where the cursor is (the map on; a journey stops). @return false when it is empty */
    bool startRoute();
    /** @brief Stops the route; the cursor stays. */
    void stopRoute() { m_route.stop(); }
    /** @brief Whether a route is being walked. @return true then */
    bool routeRunning() const { return m_route.running(); }
    /** @brief Whether the route starts over after its last point. @return true then */
    bool routeLoop() const { return m_routeLoop; }
    /**
     * @brief Sets whether it loops (from the next start).
     * @param[in] on  loop
     */
    void setRouteLoop( bool on ) { m_routeLoop = on; }
    /**
     * @brief Appends a waypoint: a preset's place (index >= 0) or a free one.
     * @param[in] x       across
     * @param[in] y       up
     * @param[in] preset  the preset, -1 none
     */
    void addWaypoint( float x, float y, int preset );
    /**
     * @brief Changes a waypoint (its times, its radius).
     * @param[in] i  which
     * @param[in] w  the point
     */
    void setWaypoint( int i, const additiv::Waypoint &w ) { m_route.set( i, w ); }
    /**
     * @brief Removes a waypoint.
     * @param[in] i  which
     */
    void removeWaypoint( int i ) { m_route.remove( i ); }
    /** @brief Empties the route. */
    void clearRoute()
    {
        m_route.clear();
        m_routeName.clear();
    }
    /**
     * @brief Writes the route to a file (one waypoint a line).
     * @param[in] file  where
     * @return false when it cannot
     */
    bool saveRoute( const juce::File &file );
    /** @brief The route under way in words, empty when none is. @return the text */
    juce::String routeStatus() const;
    /** @} */

    /** @name Journeys (Journey.h)
     *  @{ */
    /** @brief A journey to choose: one that comes with the instrument, or a file of the player's. */
    struct JourneyEntry
    {
        juce::String m_name;           ///< as the box shows it
        int          m_builtin = -1;   ///< the built-in journey, -1 for a file
        juce::File   m_file;           ///< the file, for the player's own
    };
    /** @brief Where the player's journeys live: Documents/Empyrean/Journeys. @return the folder */
    static juce::File journeyFolder();
    /** @brief The journeys to choose from: the ones that come with the instrument, then the player's files. @return them */
    std::vector<JourneyEntry> journeys() const;
    /**
     * @brief Starts a journey from its first step (the A/B morph lets go: it would hold the sound).
     * @param[in] entry  which
     * @return false when it cannot be read or names no known preset
     */
    bool startJourney( const JourneyEntry &entry );
    /** @brief Stops the journey; the preset playing stays. */
    void stopJourney() { m_journeyPlayer.stop(); }
    /** @brief The journey's clock: running, its step, the seconds left in it, the journey (the timeline). @return it */
    const additiv::JourneyPlayer &journeyPlayer() const { return m_journeyPlayer; }
    /**
     * @brief Reads a journey without starting it (the timeline's preview).
     * @param[in]  entry  which
     * @param[out] out    receives it
     * @return false when it cannot be read
     */
    static bool readJourney( const JourneyEntry &entry, additiv::Journey &out )
    {
        const bool read = entry.m_builtin >= 0 ? out.parse( additiv::builtinJourneyText( entry.m_builtin ) )
                                               : out.load( entry.m_file.getFullPathName().toRawUTF8() );
        if ( read && out.m_name.empty() ) out.m_name = entry.m_name.toStdString();
        return read && !out.m_steps.empty();
    }
    /** @brief The journey under way in words (name, step, preset, time left), empty when none is. @return the text */
    juce::String journeyStatus() const;
    /** @brief Appends the preset playing to the journey being written. */
    void journeyAddCurrent();
    /** @brief How many presets the journey being written has. @return the count */
    int journeyDraftSteps() const { return static_cast<int>( m_draft.m_steps.size() ); }
    /** @brief The journey being written (the timeline edits it). @return it */
    const additiv::Journey &journeyDraft() const { return m_draft; }
    /**
     * @brief The journey being written, replaced (a step moved, longer, gone; or a copy of another to edit).
     * @param[in] j  the journey
     */
    void setJourneyDraft( const additiv::Journey &j ) { m_draft = j; }
    /**
     * @brief The journey under way goes on from another step, gliding into it as into any.
     * @param[in] step  its index
     */
    void skipJourneyTo( int step );
    /**
     * @brief Writes the journey being written to a file (named after it when it has no name).
     * @param[in] file  where
     * @return false when it cannot
     */
    bool saveJourneyDraft( const juce::File &file );
    /** @} */
    /** @brief The preset last applied. @return its index, -1 none */
    int currentPreset() const { return m_current.load( std::memory_order_relaxed ); }

    /** @brief The family jam's line for the settings menu. @return who leads, who follows, what is heard */
    juce::String jamStatus() const { return m_jam.status(); }
    /** @brief Why the GPU engine did not open, for the status line. @return the reason, empty when it runs */
    juce::String engineError() const
    {
        const std::lock_guard<std::mutex> lock( m_errorLock );
        return juce::String( m_engineError );
    }
    /** @brief The hands arriving from a Quest app in bridge mode. @return the headset */
    frame::Headset &headset() { return m_headset; }
    /**
     * @brief A key played from the panel (the canvas's spectrum), as if from a keyboard on channel 1.
     *        Message thread.
     * @param[in] note      MIDI note
     * @param[in] velocity  0 .. 1; 0 lets it go
     */
    void tapNote( int note, float velocity )
    {
        if ( velocity > 0.0f )
            m_taps.noteOn( 1, note, velocity );
        else
            m_taps.noteOff( 1, note, 0.0f );
    }
    static constexpr const char *App = "Empyrean";   ///< the settings' and the jam's name

private:
    additiv::Engine                           m_engine;   ///< the instrument
    std::vector<juce::RangedAudioParameter *> m_params;   ///< the host's parameters, index = ParamId
    std::vector<float>                        m_pushed;   ///< the value last handed to the engine, per parameter
    juce::AudioBuffer<float>                  m_spare;    ///< a right channel when the host gives only one
    std::array<std::atomic<int>, 128>         m_ccMap;    ///< per controller the parameter it moves, -1 none
    std::atomic<int>                          m_learning{ -1 };   ///< the parameter waiting for a controller, -1 none
    /**
     * @brief Applies a program change that came from MIDI (message thread).
     * @pre called on the message thread
     */
    void handleAsyncUpdate() override;
    /**
     * @brief The jam's role from the settings, and the headset's hands, thirty times a second (message thread).
     * @pre called on the message thread
     */
    void timerCallback() override;
    /** @brief The hands' moves in the family's grammar, on the macros every preset shares. */
    void pollHeadset();
    /** @brief The journey's pump: the seconds since the last tick, and the preset a step glides into. */
    void journeyTick();
    /** @brief The map's pump: the route moves the cursor, a moved cursor is blended anew. */
    void mapTick();
    /**
     * @brief Leaves the map: the knobs go to the preset nearest the cursor (so the sound stays near
     *        where the map left it), the route stops.
     * @param[in] keepSound  apply that preset (false: a journey takes over at once)
     */
    void leaveMap( bool keepSound );
    /** @brief The map and the A/B morph let go (the map onto its nearest preset): a nudge, a variation or a lesson starts from knobs that sound. */
    void stepOutOfBlends();

    frame::JamBus  m_jam{ App };            ///< the family jam's bus
    frame::Headset m_headset;               ///< the hands from the Quest app (bridge mode)
    int            m_jamPostedRoot = -1;    ///< leading: the root posted last
    int            m_jamPostedScale = -1;   ///< leading: the scale posted last

    std::atomic<int>                    m_current{ -1 };   ///< the preset last applied, -1 none
    juce::String                        m_slotNames[2];    ///< what the morph's slots hold, in words
    std::vector<float>                  m_slotValues[2];   ///< the slots' parameters (for the state), empty when unset
    std::string                         m_slotMods[2];     ///< the slots' matrices (for the state)
    additiv::Journey                    m_journey;         ///< the journey under way
    additiv::JourneyPlayer              m_journeyPlayer;   ///< its clock and step
    double                              m_journeyLastTick = 0.0;   ///< seconds of the last tick, 0 before the first
    additiv::Journey                    m_draft;                   ///< the journey being written
    additiv::Landscape                  m_userLand;                ///< the player's own landscape, as read
    additiv::MapBlender                 m_blender;                 ///< the map's blend (keeps the presets it has read)
    float                               m_mapX = -1.0f;        ///< the cursor last blended at, across (-1: none yet)
    float                               m_mapY = -1.0f;        ///< likewise, up
    float                               m_mapR = -1.0f;        ///< likewise, the radius
    bool                                m_mapOnWas = false;    ///< the map was on at the last tick
    double                              m_mapLastTick = 0.0;   ///< seconds of the last map tick
    additiv::Route                      m_route;               ///< the route being played or written
    bool                                m_routeLoop = true;    ///< it starts over after its last point
    juce::String                        m_routeName;           ///< its name
    juce::String                        m_hallName;            ///< the hall's name (hallName())
    std::vector<std::vector<float>>     m_variations;          ///< the six variations, each NumParams values
    std::vector<float>                  m_parent;              ///< the sound they were made around
    std::vector<std::vector<float>>     m_parents;             ///< the sounds before, for Back
    std::vector<float>                  m_previewFrom;         ///< the sound before a variation was listened to
    int                                 m_previewing = -1;     ///< the variation listened to, -1 none
    float                               m_varFar = 0.3f;       ///< how far the last variations went
    std::uint64_t                       m_varSeed = 1;         ///< their dice
    std::array<additiv::MacroLesson, 4> m_lessons{};           ///< what each macro was taught
    std::array<std::vector<float>, 4>
                            m_teachBase{};   ///< per macro the knobs when its teaching began (empty: not teaching)
    std::vector<UserPreset> m_userPresets;   ///< the player's presets (refreshUserPresets)
    std::vector<Snapshot>   m_snapshots;     ///< the snapshots, the oldest first
    std::unique_ptr<juce::PropertiesFile> m_userSettings;   ///< favourites, recent, night mode, the tour (userSettings)
    std::vector<float>                    m_browseFrom;     ///< the knobs before a preset was listened to
    std::string                           m_browseFromMod;             ///< and the matrix
    bool                                  m_browsing = false;          ///< a preset is being listened to
    int                                   m_variationGeneration = 0;   ///< variationGeneration()
    std::vector<float>                    m_compareFrom;               ///< the knobs while the preset is compared with
    std::string                           m_compareFromMod;            ///< and the matrix
    bool                                  m_comparing = false;         ///< the preset is being compared with
    std::array<std::atomic<float>, 2>     m_peak{};   ///< the output's peaks since the last look (takePeaks)
    /**
     * @brief A preset chosen by the player goes on the recent list.
     * @param[in] name  its name
     */
    void noteRecent( const juce::String &name );
    /**
     * @brief Knobs and matrix applied for listening (no reference, no recent).
     * @param[in] v    the knobs
     * @param[in] mod  the matrix
     */
    void previewValues( const std::vector<float> &v, const std::string &mod );
    juce::String       m_currentUser;    ///< the player's preset playing, "" none
    std::vector<float> m_reference;      ///< the knobs of the preset the sound came from
    std::string        m_referenceMod;   ///< and its matrix, as the engine writes it
    /** @brief The sound now is what "changed" is measured against (after a preset arrives or is saved). */
    void setReference();
    /**
     * @brief One of the player's presets applied (no journey or morph let go).
     * @param[in] i      its index
     * @param[in] glide  seconds, -1: Preset Glide
     */
    void applyUserPreset( int i, double glide );
    std::vector<int>   m_found;                 ///< the last search's presets
    std::vector<float> m_foundDistance;         ///< and their distances
    juce::File         m_userLandFile;          ///< where it came from
    int                m_userLandVersion = 0;   ///< bumped when it changes
    mutable std::mutex m_errorLock;             ///< guards m_engineError (set by prepareToPlay, read by the editor)
    std::string        m_engineError;           ///< why the engine did not open, empty when it runs
    const bool         m_muted =
        juce::SystemStats::getEnvironmentVariable( "ADDITIV_MUTE", {} ).isNotEmpty();   ///< ADDITIV_MUTE: silent output
    std::atomic<int>        m_pendingProgram{ -1 };   ///< a MIDI program change waiting for the message thread
    int                     m_bankGroup = 0;          ///< the group bank select (CC 0) chose
    bool                    m_sustain = false;        ///< the hold pedal (CC 64) is down
    juce::MidiKeyboardState m_taps;   ///< keys the picture plays (the canvas), merged into the block's MIDI
    std::bitset<128>        m_held;   ///< notes let go while the pedal was down, released with it
    float                   m_masterBend = 2.0f;   ///< MPE: the master channel's bend range, semitones (its RPN 0)
    int                     m_rpn[16] = {
                            0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF,
                            0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF, 0x3FFF
    };   ///< each channel's chosen RPN (CC 101 and 100), 0x3FFF none
    /**
     * @brief MPE's controllers on a channel: the RPN chosen (CC 101 / 100), and its value (CC 6):
     *        RPN 0 a bend range (the master's on channel 1, the member channels' elsewhere), RPN 6 on
     *        channel 1 the zone (MPE on with members, off without). Audio thread.
     * @param[in] ch     the channel, 0 .. 15
     * @param[in] cc     the controller
     * @param[in] value  its value, 0 .. 127
     * @return true when it was one of these
     * @pre called on the audio thread
     */
    bool mpeController( int ch, int cc, int value );

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR( AdditivProcessor )
};
