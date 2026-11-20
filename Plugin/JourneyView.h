/**
 * @file JourneyView.h
 * @brief The Journey page: the glide between presets, the A/B morph and its two slots, and the
 *        journeys -- presets in a row, each glided into the next, for a whole night (Journey.h).
 * @author Rene Weller
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <functional>
#include <vector>

/**
 * @brief The night director's controls (NightDirector.h, 20.11.2026): how long the night is (hours, or until a time
 *        to wake), which groups it draws from, whether the favourites are preferred, new dice and Make. What it makes
 *        becomes the journey being written -- the timeline shows it, the disk saves it; nothing starts by itself. Open,
 *        it lies over the morph slots, so the page does not grow.
 */
class NightPanel final : public juce::Component
{
public:
    /**
     * @brief The controls for @p proc.
     * @param[in,out] proc  the processor (its favourites, the journey being written)
     */
    explicit NightPanel( AdditivProcessor &proc );
    /** @brief The two rows. */
    void resized() override;
    /**
     * @brief Its ground and its heading.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override;
    /** @brief The lengths offered anew from the clock, whenever it is shown. */
    void visibilityChanged() override;
    /**
     * @brief Composes a night and makes it the journey being written.
     * @param[in] newDice  draw a new seed first
     */
    void make( bool newDice );
    /**
     * @brief Sets the controls as the screenshot mode asks (ADDITIV_SHOT_DIRECTOR): a length, no favourites, seed 1.
     * @param[in] hours  the night's length
     */
    void forShot( double hours );
    std::function<void()> m_onClose;   ///< what the close button does (the view hides the panel)

private:
    /** @brief Fills the length box: until 5:30 .. 9:00 from now (when 1 to 14 hours away), then 3 .. 10 hours. */
    void refreshLengths();

    AdditivProcessor   &m_proc;                                   ///< whose favourites and journey being written
    juce::ComboBox      m_length;                                 ///< how long the night is
    juce::ComboBox      m_family;                                 ///< which groups it draws from
    juce::ToggleButton  m_favourites{ "Prefer my favourites" };   ///< the favourites' bonus
    juce::TextButton    m_dice{ "New dice" };                     ///< another night of the same kind
    juce::TextButton    m_make{ "Make" };                         ///< composes it
    juce::TextButton    m_close{ "Close" };                       ///< hides the panel
    juce::Label         m_status;                                 ///< what was made
    std::vector<double> m_hours;                                  ///< the length box's hours (item id = index + 1)
    std::vector<bool>   m_until;                                  ///< whether an item is a time to wake
    std::uint64_t       m_seed = 1;                               ///< the dice
};

/** @brief The slots and the journeys (below the page's knobs). */
class JourneyView final : public juce::Component, private juce::Timer
{
public:
    /**
     * @brief The slots and journeys of @p proc.
     * @param[in,out] proc  the processor
     */
    explicit JourneyView( AdditivProcessor &proc );
    /** @brief The rows. */
    void resized() override;
    /**
     * @brief The boxes and their headings.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override;
    /** @brief The height it needs. @return pixels */
    static int idealHeight() { return 2 * 34 + 30 + 2 * 34 + 30 + 20; }
    /** @brief What it shows, in words, for the help. @return the text */
    juce::String describe() const;
    /**
     * @brief The journey chosen in the box, read (the timeline's preview).
     * @param[out] out  receives it
     * @return false when none is chosen
     */
    bool chosen( additiv::Journey &out ) const
    {
        const int id = m_journeys.getSelectedId();
        return id > 0 && id <= static_cast<int>( m_entries.size() ) &&
               AdditivProcessor::readJourney( m_entries[static_cast<size_t>( id - 1 )], out );
    }

private:
    /** @brief One slot's row. */
    struct SlotRow
    {
        juce::Label      m_name;                         ///< what the slot holds
        juce::TextButton m_fromSound{ "This sound" };    ///< the slot takes what plays
        juce::TextButton m_fromPreset{ "Preset ..." };   ///< the slot takes a preset from the bank
    };
    /** @brief The status, the slots' names and the journey list now and then. */
    void timerCallback() override;
    /** @brief Fills the journey box: the ones that come with the instrument, then the player's files. */
    void refreshJourneys();
    /**
     * @brief The bank as a menu; the chosen preset goes to a slot.
     * @param[in] slot  0 or 1
     */
    void choosePreset( int slot );
    /** @brief Starts the journey chosen in the box. */
    void startChosen();
    /** @brief Asks for a name and writes the journey being written into the journeys folder. */
    void saveDraft();

    AdditivProcessor                           &m_proc;       ///< whose slots and journeys
    SlotRow                                     m_slots[2];   ///< A and B
    juce::ComboBox                              m_journeys;   ///< the journeys to choose from
    std::vector<AdditivProcessor::JourneyEntry> m_entries;    ///< what the box's items are (item id = index + 1)
    /** @brief Starts the journey chosen in the box. */
    frame::IconButton m_start{
        frame::IconButton::IconE::PLAY,
        "Starts the journey chosen beside it: each preset held for its time, then glided into the next"
    };
    /** @brief Stops the journey under way. */
    frame::IconButton m_stop{ frame::IconButton::IconE::STOP, "Stops the journey; the preset playing stays" };
    juce::Label       m_status;                   ///< the journey under way
    juce::TextButton  m_add{ "+ This preset" };   ///< appends the preset playing to the journey being written
    /** @brief Saves the journey being written. */
    frame::IconButton m_save{
        frame::IconButton::IconE::SAVE,
        "Writes the journey being written into Documents/Empyrean/Journeys under a name, and offers it in the box"
    };
    /** @brief Opens the journeys folder. */
    frame::IconButton m_folder{
        frame::IconButton::IconE::OPEN,
        "Opens the journeys folder (Documents/Empyrean/Journeys): journeys are plain text, one preset a line"
    };
    juce::Label                        m_draft;   ///< how many presets the journey being written has
    juce::TextButton                   m_nightButton{ "Night ..." };   ///< opens the night director
    NightPanel                         m_night;                        ///< the night director, over the slots
    std::unique_ptr<juce::FileChooser> m_chooser;                      ///< the save dialog while it is open
    int                                m_tick = 0;                     ///< timer ticks, for the slower refreshes
};

/**
 * @brief The journey's timeline: its steps as a band over time, each as wide as it lasts (its glide a ramp into it),
 *        the preset's name on it; the step under way lit and filling as its time runs, and a click on another goes
 *        there. Before a journey runs, the one chosen in the box, faint. Under it, once one is being written, the
 *        player's own journey to edit by the hand: a block's right edge dragged sets how long it stays, its ramp how
 *        long it glides in, the block itself dragged moves it, a right-click removes it. "Edit a copy" takes the
 *        journey above into it.
 */
class JourneyTimeline final : public juce::Component, private juce::Timer
{
public:
    /**
     * @brief The timeline of @p proc's journeys.
     * @param[in,out] proc  the processor
     * @param[in]     view  the box it previews
     */
    JourneyTimeline( AdditivProcessor &proc, const JourneyView &view );
    /**
     * @brief The bands, the steps, where the journey is.
     * @param[in,out] g  where
     */
    void paint( juce::Graphics &g ) override;
    /** @brief The copy button. */
    void resized() override;
    /**
     * @brief A step clicked (skip there), or a stroke on the one being written begins.
     * @param[in] e  the press
     */
    void mouseDown( const juce::MouseEvent &e ) override;
    /**
     * @brief The edge, the ramp or the block follows.
     * @param[in] e  the drag
     */
    void mouseDrag( const juce::MouseEvent &e ) override;
    /**
     * @brief A moved block lands where it was let go.
     * @param[in] e  the release
     */
    void mouseUp( const juce::MouseEvent &e ) override;
    /**
     * @brief The cursor says what a drag would do.
     * @param[in] e  the move
     */
    void mouseMove( const juce::MouseEvent &e ) override;

private:
    /** @brief One step's place in a band: where its glide begins, where it stands, where it ends. */
    struct Block
    {
        float m_x0 = 0.0f;   ///< the glide's start
        float m_xg = 0.0f;   ///< the step's own start
        float m_x1 = 0.0f;   ///< its end
    };
    /** @brief What a drag on the journey being written does. */
    enum class DragE
    {
        NONE,
        DWELL,
        GLIDE,
        MOVE
    };
    /** @brief Twice a second while it is seen (ten while a drag is on). */
    void timerCallback() override;
    /**
     * @brief The journey above: the one under way, else the one chosen.
     * @param[out] out  receives it
     * @return false when there is none
     */
    bool upper( additiv::Journey &out ) const;
    /**
     * @brief A band's rectangle.
     * @param[in] which  0 the journey above, 1 the one being written
     * @return pixels
     */
    juce::Rectangle<float> band( int which ) const;
    /**
     * @brief The blocks of a journey in a band, on its own time scale.
     * @param[in] j  the journey
     * @param[in] r  the band
     * @return them
     */
    static std::vector<Block> blocks( const additiv::Journey &j, juce::Rectangle<float> r );
    /**
     * @brief Draws a journey in a band.
     * @param[in,out] g      where
     * @param[in]     j      the journey
     * @param[in]     r      the band
     * @param[in]     lit    the step lit, -1 none
     * @param[in]     done   how far it has gone, 0 .. 1
     * @param[in]     faint  drawn faint
     */
    void drawBand( juce::Graphics &g, const additiv::Journey &j, juce::Rectangle<float> r, int lit, double done,
                   bool faint ) const;
    /**
     * @brief The block of the one being written under a point.
     * @param[in]  p     the point
     * @param[out] drag  receives what a drag there does
     * @return its index, -1 none
     */
    int draftAt( juce::Point<float> p, DragE &drag ) const;

    AdditivProcessor    &m_proc;                    ///< whose journeys
    const JourneyView   &m_view;                    ///< the box
    juce::TextButton     m_copy{ "Edit a copy" };   ///< the journey above into the one being written
    DragE                m_drag = DragE::NONE;      ///< the drag under way
    int                  m_dragStep = -1;           ///< its step
    float                m_dragX = 0.0f;            ///< where it began
    double               m_perPixel = 1.0;          ///< seconds a pixel when it began (the scale held while dragging)
    additiv::JourneyStep m_before;                  ///< the step as it was when it began
    int                  m_dropAt = -1;             ///< where a moved block would land, -1 nowhere
};

/** @brief The Journey page: its knobs (the glide, the morph) above, the slots and journeys below, the timeline under them. */
class JourneyPage final : public juce::Component
{
public:
    /**
     * @brief Builds the page.
     * @param[in,out] proc     the processor
     * @param[in]     actions  the controls' right-click menu
     * @param[in,out] live     the live rings
     */
    JourneyPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live );
    /**
     * @brief The height the page needs at a width.
     * @param[in] w  pixels
     * @return pixels
     */
    int heightFor( int w ) const
    {
        return m_params.heightFor( w ) + JourneyView::idealHeight() + 10 + TimelineMin + 10;
    }
    /** @brief The knobs above, the rest below. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const { return m_params.describe() + m_view.describe(); }

private:
    static constexpr int TimelineMin =
        110;   ///< the timeline's least height (it takes what the page leaves; two bands once a journey is being written)
    ParamPage       m_params;     ///< the glide and the morph's knobs
    JourneyView     m_view;       ///< the slots and the journeys
    JourneyTimeline m_timeline;   ///< the journey over time
};
