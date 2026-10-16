/**
 * @file JourneysComponent.h
 * @brief Empyrean Journeys' window: the journeys in a table (their steps, their length, its range, their kind), a search
 *        over them, and what to render them into -- the format, the passes, the dice, the fade, the night, the folder --
 *        with the rendering's progress, speed and time left. Several chosen are rendered one after another.
 * @author Rene Weller
 */
#pragma once

#include "JourneyRender.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <deque>
#include <mutex>

namespace journeys
{

/** @brief The renderings asked for, done one after another on a thread of their own. */
class RenderQueue : public juce::Thread
{
public:
    RenderQueue() : juce::Thread( "Empyrean Journeys render" ) {}
    /** @brief Stops what is under way and waits for the thread. */
    ~RenderQueue() override;
    /**
     * @brief Adds renderings to the queue (and starts the thread when it rests).
     * @param[in] jobs  the renderings
     */
    void add( const std::vector<RenderJob> &jobs );
    /** @brief Cancels the rendering under way and those still waiting. */
    void cancelAll();
    /**
     * @brief What the queue is doing, for the window.
     * @param[out] name      receives the journey under way ("" when none)
     * @param[out] waiting   receives how many wait after it
     * @param[out] finished  receives how many were written so far
     * @return true while one is under way
     */
    bool status( juce::String &name, int &waiting, int &finished );
    /**
     * @brief The last message (a file written, a failure), once.
     * @return it, or "" when there is none new
     */
    juce::String takeMessage();
    Renderer m_renderer;   ///< the rendering under way: its progress

private:
    /** @brief Renders the queue, one after another. */
    void run() override;
    std::mutex            m_lock;              ///< guards the queue and the messages
    std::deque<RenderJob> m_jobs;              ///< waiting
    juce::String          m_current;           ///< the journey under way
    juce::String          m_message;           ///< the last message
    int                   m_finished = 0;      ///< written so far
    std::atomic<bool>     m_cancel{ false };   ///< the one under way stops
    juce::WaitableEvent   m_wake;              ///< work was added (or the queue closes)
};

/** @brief The window's content. */
class JourneysComponent : public juce::Component, private juce::TableListBoxModel, private juce::Timer
{
public:
    JourneysComponent();
    /** @brief Keeps the settings. */
    ~JourneysComponent() override;
    /**
     * @brief Paints the background and the headings.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override;
    /** @brief Lays out the table and the panel beside it. */
    void resized() override;
    /**
     * @brief Chooses a journey in the table (the manual's picture).
     * @param[in] name  its name
     * @return false when the table has none of that name
     */
    bool choose( const juce::String &name );

private:
    /** @brief The table's columns. */
    enum ColumnE
    {
        NAME = 1,   ///< the journey
        STEPS,      ///< its steps
        LENGTH,     ///< a pass at the middle of its ranges
        RANGE,      ///< its shortest .. longest
        KIND        ///< whole night, night, journey, yours
    };
    /** @brief The rows: the journeys the search lets through, in the order asked for. */
    void refilter();
    /** @brief The panel's text for the journey chosen first: its length with these dice and passes. */
    void updateChosen();
    /** @brief Queues every chosen journey. */
    void renderChosen();
    /**
     * @brief The journeys the table has chosen.
     * @return them, in the table's order
     */
    std::vector<const JourneyItem *> chosen() const;
    /**
     * @brief A file for a rendering, not overwriting one that is there.
     * @param[in] item  the journey
     * @param[in] len   its length, s
     * @return the file in the folder
     */
    juce::File fileFor( const JourneyItem &item, double len ) const;
    /** @brief Progress, speed and time left, ten times a second. */
    void timerCallback() override;
    /**
     * @brief The rows. @return their count
     */
    int getNumRows() override;
    /**
     * @brief A row's background.
     * @param[in,out] g         the graphics context
     * @param[in]     row       the row
     * @param[in]     w         its width
     * @param[in]     h         its height
     * @param[in]     selected  it is chosen
     */
    void paintRowBackground( juce::Graphics &g, int row, int w, int h, bool selected ) override;
    /**
     * @brief A cell.
     * @param[in,out] g         the graphics context
     * @param[in]     row       the row
     * @param[in]     column    the column (ColumnE)
     * @param[in]     w         its width
     * @param[in]     h         its height
     * @param[in]     selected  its row is chosen
     */
    void paintCell( juce::Graphics &g, int row, int column, int w, int h, bool selected ) override;
    /**
     * @brief The order changed (a click on a heading).
     * @param[in] column   the column sorted by
     * @param[in] forward  ascending
     */
    void sortOrderChanged( int column, bool forward ) override;
    /**
     * @brief The choice changed.
     * @param[in] row  the last row clicked
     */
    void selectedRowsChanged( int row ) override;
    /**
     * @brief A double click renders the row.
     * @param[in] row     the row
     * @param[in] column  the column
     * @param[in] e       the mouse event
     */
    void cellDoubleClicked( int row, int column, const juce::MouseEvent &e ) override;

    juce::LookAndFeel_V4                  m_look;   ///< the instrument's colours (before every child: it outlives them)
    std::vector<JourneyItem>              m_items;                ///< every journey
    std::vector<int>                      m_rows;                 ///< the rows: indices into m_items
    int                                   m_sortColumn = NAME;    ///< the order's column
    bool                                  m_sortForward = true;   ///< ascending
    std::uint64_t                         m_seed = 1;             ///< the dice
    juce::File                            m_folder;               ///< where the files go
    std::unique_ptr<juce::PropertiesFile> m_settings;         ///< the window's settings (Empyrean/journeys.settings)
    RenderQueue                           m_queue;            ///< the renderings
    double                                m_progress = 0.0;   ///< the progress bar's value
    juce::Label                           m_title;            ///< the heading
    juce::Label                           m_search;           ///< the search's label
    juce::TextEditor                      m_filter;           ///< the search
    juce::TableListBox                    m_table;            ///< the journeys
    juce::Label                           m_chosenName;       ///< the journey chosen first
    juce::Label                           m_chosenInfo;       ///< its steps and this rendering's length
    juce::Label                           m_formatLabel;      ///< "Format"
    juce::Label                           m_passesLabel;      ///< "Passes"
    juce::Label                           m_folderLabel;      ///< "Folder"
    juce::ComboBox                        m_format;           ///< the format
    juce::ComboBox                        m_passes;           ///< passes of a cyclic journey
    juce::TextButton                      m_dice{ "New dice" };                            ///< new dwells and glides
    juce::ToggleButton                    m_fade{ "Fade out over the last 20 seconds" };   ///< the end faded
    juce::ToggleButton                    m_night{ "As at night (the night arc from 23:00)" };   ///< the night arc
    juce::ToggleButton                 m_exact{ "Exactly as the instrument plays (slower)" };    ///< GPU blocks of 128
    juce::Label                        m_folderText;                                             ///< the folder
    juce::TextButton                   m_chooseFolder{ "Choose ..." };                           ///< another folder
    juce::TextButton                   m_openFolder{ "Open folder" };   ///< the folder in Explorer
    juce::TextButton                   m_render{ "Render" };            ///< renders the journeys chosen
    juce::TextButton                   m_cancel{ "Cancel" };            ///< stops the renderings
    juce::ProgressBar                  m_bar{ m_progress };             ///< the progress
    juce::Label                        m_status;                        ///< what it is doing
    std::unique_ptr<juce::FileChooser> m_chooser;                       ///< the folder dialog while it is open
};

}   // namespace journeys
