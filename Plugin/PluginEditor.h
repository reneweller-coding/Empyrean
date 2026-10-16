/**
 * @file PluginEditor.h
 * @brief The panel, on the family's frame (Frame.h): the header (logo and name; Autoplay's play and
 *        stop, its mode and Step; scale, root, motion, landscape; the status; undo, redo, help,
 *        settings), the cloud picture (CloudView), the pages in their strip (PageStrip) -- Focus; Cloud &
 *        Filter, Motion; Tuning, Conductor; Space, Effects, Hall; Modulation, Matrix; Journey, Map -- and
 *        the modulation dock along the bottom (ModDock). The family's keys (Space, Ctrl+Z/Y, F1, F11, Esc)
 *        and its right-click menu on every control (MIDI learn, forget, default). The window starts as
 *        large as the screen allows (three quarters of its width), at least 1280 x 860, and comes back at
 *        the size it was left at.
 * @author Rene Weller
 */
#pragma once

#include "Browser.h"
#include "CloudView.h"
#include "InfoLine.h"
#include "ModDock.h"
#include "PageStrip.h"
#include "Palette.h"
#include "Tour.h"
#include "Frame.h"
#include "HeaderParts.h"
#include "MatrixView.h"
#include "ParamPage.h"
#include "PluginProcessor.h"
#include "Theme.h"

#include <memory>
#include <vector>

/** @brief The instrument's name, drawn in the skin's title style. */
class TitleComponent final : public juce::Component
{
public:
    /**
     * @brief Draws the name.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override
    {
        frame::drawTitle( g, additivui::skin(), getLocalBounds().toFloat(), static_cast<float>( getHeight() ) * 0.62f );
    }
};

/** @brief The panel (see the file comment). */
class AdditivEditor final : public juce::AudioProcessorEditor,
                            public juce::DragAndDropContainer,
                            public ParamPage::DropListener,
                            private juce::Timer,
                            private juce::AudioProcessorParameter::Listener
{
public:
    /**
     * @brief The panel of @p proc.
     * @param[in,out] proc  the processor
     */
    explicit AdditivEditor( AdditivProcessor &proc );
    ~AdditivEditor() override;

    /**
     * @brief The window's background (the frame's backdrop).
     * @param[in,out] g  where
     */
    void paint( juce::Graphics &g ) override;
    /**
     * @brief The conductor's countdown: a thin line under Step that runs down to the next exchange.
     * @param[in,out] g  where
     */
    void paintOverChildren( juce::Graphics &g ) override;
    /** @brief The header, the picture, the pages, the help. */
    void resized() override;
    /**
     * @brief The family's keys.
     * @param[in] key  the key
     * @return true when it was one
     */
    bool keyPressed( const juce::KeyPress &key ) override;
    /** @brief In the standalone, the window gets a maximise button beside the other two (Noctuary's, Totality's) and comes back where it was left. */
    void parentHierarchyChanged() override;
    /**
     * @brief The host's scale (a screen's density), the zoom on top of it.
     * @param[in] newScale  the host's
     */
    void setScaleFactor( float newScale ) override;

private:
    /** @brief The status line, the play button and the picture, fifteen times a second. */
    void timerCallback() override;
    /**
     * @brief Not used: values are read when a gesture ends.
     * @param[in] index  ignored
     * @param[in] v      ignored
     */
    void parameterValueChanged( int index, float v ) override
    {
        (void)index;
        (void)v;
    }
    /**
     * @brief A knob's gesture opens and closes an undo step.
     * @param[in] index     the parameter
     * @param[in] starting  begins or ends
     */
    void parameterGestureChanged( int index, bool starting ) override;

    /** @brief Every parameter's value in its unit, in ParamId order. @return the values */
    std::vector<float> values() const;
    /**
     * @brief Applies an undo step's values.
     * @param[in] step   the step (null: nothing)
     * @param[in] after  the after values (redo) or the before
     */
    void apply( const frame::UndoStep *step, bool after );
    /**
     * @brief Opens or closes the help.
     * @param[in] on  open
     */
    void showHelp( bool on );
    /** @brief The settings menu under its button. */
    void showSettings();
    /** @brief Asks for the night arc's place (latitude and longitude) and keeps it (Phase 30). */
    void askPlace();
    /** @brief Whether this is the standalone. @return true there */
    bool standalone() const { return m_proc.wrapperType == juce::AudioProcessor::wrapperType_Standalone; }
    /** @brief The standalone's full screen on or off. */
    void toggleFullScreen();
    /**
     * @brief The canvas on or off: the picture takes the whole window, the pages step aside.
     * @param[in] on  canvas
     */
    void setCanvas( bool on );
    /** @brief Opens the command palette (Ctrl+K). */
    void openPalette();
    /**
     * @brief The night mode on or off (kept in the player's settings).
     * @param[in] on  night
     */
    void setNight( bool on );
    /** @brief The first steps' tour, from its first step. */
    void startTour();
    /** @brief Everything the palette can do: the actions, every knob, every preset. @return the commands */
    std::vector<CommandPalette::Command> paletteCommands();
    /**
     * @brief Shows a parameter's control: its page in front, scrolled to it, lighting up.
     * @param[in] id  the parameter
     * @return false when no page shows it
     */
    bool revealParam( int id );
    /**
     * @brief A header chooser bound to a parameter.
     * @param[in,out] box  the chooser
     * @param[in]     id   the parameter
     */
    void bindChoice( frame::Choice &box, additiv::ParamIdE id );
    /**
     * @brief The window's first size: the size it was left at, where that still fits a screen (any of them: it
     *        may have been on a larger one than the main screen); else three quarters of the screen's width at
     *        the window's proportions, 1280 x 860 at least (less only on a screen smaller than that). 1280 x 860
     *        in the screenshot and layout modes.
     * @return pixels
     */
    juce::Point<int> startSize();
    /**
     * @brief The standalone's window where it was left -- its place, its size, maximised or not -- and from then on its
     *        moves kept.
     * @param[in,out] window  the window
     */
    void restoreWindow( juce::ResizableWindow &window );
    /** @brief Keeps the standalone window's state (where, how large, maximised) in the player's settings whenever it moves. */
    struct WindowKeeper final : public juce::ComponentListener
    {
        AdditivEditor                                      *m_owner = nullptr;   ///< whose settings
        juce::Component::SafePointer<juce::ResizableWindow> m_window;   ///< the standalone's window (none in a host)
        /**
         * @brief The window moved or was resized: its state written.
         * @param[in,out] c        the window
         * @param[in]     moved    ignored
         * @param[in]     resized  ignored
         */
        void componentMovedOrResized( juce::Component &c, bool moved, bool resized ) override;
    };
    WindowKeeper m_windowKeeper;   ///< the standalone window's state, kept
    /**
     * @brief A page's state for its strip: its groups on and off, the matrix's rows, a journey or route running.
     * @param[in] tab  the page
     * @return its state
     */
    PageStrip::Status pageStatus( int tab );
    /**
     * @brief A source chosen in the dock (or dropped on a knob): the inspector shows it, open.
     * @param[in] s       the source
     * @param[in] toggle  a second click on it shuts the inspector
     */
    void chooseSource( additiv::ModSourceE s, bool toggle );
    /**
     * @brief The inspector open or shut (docked: remembered; over the pages: until shut).
     * @param[in] open  open
     */
    void setInspector( bool open );

    AdditivProcessor      &m_proc;                       ///< the instrument
    frame::LookAndFeel     m_lnf{ additivui::skin() };   ///< first, so it outlives every component that uses it
    juce::TooltipWindow    m_tooltips{ nullptr, 700 };   ///< tooltips after 700 ms
    frame::ControlActions  m_actions;                    ///< the controls' right-click menu
    frame::Backdrop        m_backdrop;                   ///< the window's background
    juce::Rectangle<float> m_logo;                       ///< where the logo is drawn
    TitleComponent         m_title;                      ///< the name
    frame::IconButton      m_play{ frame::IconButton::IconE::PLAY,
                                   "Autoplay: start the conductor" };   ///< Autoplay on and off
    juce::TextButton       m_step{ "Step" };                            ///< Autoplay: exchange a voice now
    juce::TextButton       m_prevPreset{ "<" };                         ///< the preset before
    juce::TextButton       m_presetName;                                ///< the preset playing; a click opens the bank
    juce::TextButton       m_nextPreset{ ">" };                         ///< the preset after
    /** @brief The bank as a menu of its groups. */
    void showPresets();
    /** @brief The Save menu: save, save as, back to the preset, the folder. */
    void showSaveMenu();
    /** @brief Saves the sound under its own name when it is one of the player's presets, else asks for one. */
    void quickSave();
    /** @brief Asks for a name and saves the sound as one of the player's presets. */
    void saveAs();
    /**
     * @brief Applies a preset as one undo step.
     * @param[in] index  the preset
     */
    void loadPreset( int index );
    frame::Choice m_mode{ &m_actions, static_cast<int>( additiv::ParamIdE::AUTO_MODE ) };        ///< Free or Chords
    frame::Choice m_scale{ &m_actions, static_cast<int>( additiv::ParamIdE::SCALE ) };           ///< the scale
    frame::Choice m_root{ &m_actions, static_cast<int>( additiv::ParamIdE::ROOT_NOTE ) };        ///< its root
    frame::Choice m_motion{ &m_actions, static_cast<int>( additiv::ParamIdE::MOTION ) };         ///< the motion
    frame::Choice m_landscape{ &m_actions, static_cast<int>( additiv::ParamIdE::LANDSCAPE ) };   ///< the landscape
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>>
                      m_headerAttach;   ///< the header's choosers to their parameters
    juce::Label       m_status;         ///< voices, root, GPU, latency, device
    frame::IconButton m_undo{ frame::IconButton::IconE::UNDO, "Undo (Ctrl+Z)" };   ///< undo
    frame::IconButton m_redo{ frame::IconButton::IconE::REDO, "Redo (Ctrl+Y)" };   ///< redo
    frame::IconButton m_help{ frame::IconButton::IconE::HELP, "Help (F1)" };       ///< the help
    /** @brief Save the sound as one's own preset (Ctrl+S), back to the preset, the presets' folder. */
    frame::IconButton m_save{ frame::IconButton::IconE::SAVE,
                              "Your presets: save the sound (Ctrl+S), save it as ..., back to the preset" };
    std::unique_ptr<juce::AlertWindow> m_askName;   ///< the name dialog while it is open
    frame::IconButton                  m_settings{ frame::IconButton::IconE::SETTINGS, "Settings" };   ///< the settings
    CloudView                          m_cloud{ m_proc };                                              ///< the picture
    ViewButton                         m_canvasButton{
        ViewButton::GlyphE::CANVAS,
        "Canvas: the partials and the landscape alone, the whole window (again, or Esc: back)"
    };   ///< the picture alone
    ViewButton m_searchButton{
        ViewButton::GlyphE::SEARCH, "Search (Ctrl+K): any knob, preset or action by a few letters"
    };   ///< the command palette
    ViewButton        m_compare{ ViewButton::GlyphE::COMPARE,
                                 "Hold to hear the preset as it was; let go for your sound" };   ///< hold: the preset
    frame::IconButton m_midi{ frame::IconButton::IconE::MIDI_LEARN,
                              "Show the MIDI controllers learned on the knobs" };   ///< MIDI marks on the knobs
    OutputMeter       m_meter;                                                      ///< the output
    CommandPalette    m_palette;             ///< the command palette, over everything while open
    PresetBrowser     m_browser{ m_proc };   ///< the preset browser, over everything while open
    std::unique_ptr<InfoLine> m_info;        ///< the info line at the bottom: what the control under the mouse does
    ViewButton                m_nightButton{
        ViewButton::GlyphE::NIGHT, "Night: the window dim and warm for a dark room, the picture calmer"
    };   ///< the night mode
    bool   m_night = false;        ///< the night mode is on: the window dim and warm
    bool   m_tourAsked = false;    ///< the first start's tour was considered
    double m_lastActivity = 0.0;   ///< when the mouse or a key last moved (ms)
    int    m_refreshTick = 0;      ///< timer ticks (the night refreshes the picture half as often)
    /** @brief Notes every move of the mouse in the window (the night's canvas fades when left alone). */
    struct Activity final : public juce::MouseListener
    {
        double *m_last = nullptr;   ///< where the time goes
        /**
         * @brief A move.
         */
        void mouseMove( const juce::MouseEvent & ) override { *m_last = juce::Time::getMillisecondCounterHiRes(); }
        /**
         * @brief A press.
         */
        void mouseDown( const juce::MouseEvent & ) override { *m_last = juce::Time::getMillisecondCounterHiRes(); }
        /**
         * @brief A wheel.
         */
        void mouseWheelMove( const juce::MouseEvent &, const juce::MouseWheelDetails & ) override
        {
            *m_last = juce::Time::getMillisecondCounterHiRes();
        }
    };
    Activity m_activity;   ///< the mouse's moves
    /** @brief The night's veil: over the window, under the palette, the browser and the tour; the mouse goes through it. */
    struct Veil final : public juce::Component
    {
        float m_alpha = 0.0f;   ///< how dark
        /**
         * @brief Dark and warm: little blue for eyes used to the dark.
         * @param[in,out] g  where
         */
        void paint( juce::Graphics &g ) override { g.fillAll( juce::Colour( 0xff0c0703 ).withAlpha( m_alpha ) ); }
    };
    Veil             m_veil;             ///< the night's veil
    Tour             m_tour;             ///< the first steps
    bool             m_canvas = false;   ///< the canvas is shown
    frame::LiveRings m_live{
        [this]( int id ) { return m_proc.engine().playedNorm( static_cast<additiv::ParamIdE>( id ) ); }
    };   ///< where modulated knobs play
    juce::TabbedComponent                      m_tabs{ juce::TabbedButtonBar::TabsAtTop };   ///< the pages
    std::vector<std::function<juce::String()>> m_describe;           ///< per page its parameters in words, for the help
    std::unique_ptr<frame::HelpView>           m_helpView;           ///< the help, while it is open
    frame::UndoHistory                         m_history;            ///< undo and redo
    bool                                       m_applying = false;   ///< an undo is being applied: record nothing
    int                                        m_headerBottom = 0;   ///< where the header ends, for the backdrop
    juce::String
        m_shotPrefix;   ///< ADDITIV_SHOT: every page as \<prefix\>-\<n\>-\<page\>.png (ADDITIV_PRESET first), then the standalone quits
    int m_shotTick = 0;   ///< timer ticks since the screenshots were asked for
    juce::MemoryBlock
        m_shotSaved;   ///< the state the screenshots found, given back before the standalone quits (it saves its state then)
    int  m_logTick = 0;             ///< timer ticks, for the ADDITIV_LOG diagnostics
    bool m_layoutChecked = false;   ///< ADDITIV_LAYOUT: the pages were measured
    bool m_rememberSize = false;    ///< the window's size is kept for the next start (not in the shot or layout modes)
    PageStrip m_strip;              ///< the pages' strip
    /**
     * @brief A source dropped on a knob: the inspector shows it (or all rows are taken).
     * @param[in] s    the source
     * @param[in] id   the parameter
     * @param[in] row  the row, -1 none
     */
    void modDropped( additiv::ModSourceE s, int id, int row ) override;
    /**
     * @brief The parameter pages on a page, as they are now.
     * @param[in] tab  the page
     * @return them
     */
    std::vector<ParamPage *> pagesOn( int tab );
    std::unique_ptr<ModDock>         m_dock;        ///< the sources along the bottom
    std::unique_ptr<SourceInspector> m_inspector;   ///< one source seen whole
    juce::Component *m_matrixPage = nullptr;      ///< the Modulation page (owned by its tab), to mark the source chosen
    bool             m_inspectorOpen = true;      ///< the docked inspector is wanted (remembered)
    bool             m_overlayOpen = false;       ///< the inspector is open over the pages (a small window)
    bool             m_inspectorDocked = false;   ///< the window is tall enough to dock it
    float            m_hostScale = 1.0f;          ///< the scale the host asked for
    float            m_zoom = 1.0f;               ///< the player's zoom: the controls larger (player.settings "zoom")
    /**
     * @brief The zoom anew (kept in the player's settings).
     * @param[in] z  1 .. 1.5
     */
    void setZoom( float z );
    int   m_altRow = -1;            ///< the matrix row an Alt-drag on a knob sets, -1 none
    float m_altDepth = 0.0f;        ///< its depth when the drag began
    float m_stepLeft = -1.0f;       ///< the share of the conductor's wait still to come, -1 none (the line under Step)
    juce::String m_lastMatrix;      ///< the matrix as the undo history last knew it
    juce::String m_pendingBefore;   ///< a matrix edit outside an undo step: the matrix before it
    double       m_pendingAt = 0.0;   ///< when it last moved (ms); a step once it rests
    juce::String m_presetKey;         ///< the preset playing, as the matrix's watch last saw it
    /** @brief The matrix as text, the rest of an undo step's state. @return it */
    juce::String matrixText() const { return juce::String( m_proc.engine().modText() ); }
    /** @brief Closes an undo step with the matrix as it is now. */
    void endStep();
    /** @brief The matrix watched: an edit made outside an undo step (a dropped chip, the inspector, the matrix page, a knob's menu) becomes one once it rests. */
    void watchMatrix();
    /** @brief Records a matrix edit still pending, before an undo or redo. */
    void flushMatrix();
    /** @brief One step of the screenshot mode (ADDITIV_SHOT), from the timer. */
    void shootStep();
    /**
     * @brief ADDITIV_LAYOUT (a file): every page's height against its room at a few window sizes, written there; the
     *        standalone then quits with the number of pages that would scroll at the window's usual size.
     * @param[in] out  the report
     */
    void layoutCheck( const juce::File &out );
};
