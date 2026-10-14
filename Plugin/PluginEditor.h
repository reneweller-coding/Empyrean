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
class TitleComponent final : public juce::Component {
public:
    /** @brief Draws the name. */
    void paint(juce::Graphics& g) override { frame::drawTitle(g, additivui::skin(), getLocalBounds().toFloat(), static_cast<float>(getHeight()) * 0.62f); }
};

/** @brief The panel (see the file comment). */
class AdditivEditor final : public juce::AudioProcessorEditor,
                            public juce::DragAndDropContainer,
                            public ParamPage::DropListener,
                            private juce::Timer,
                            private juce::AudioProcessorParameter::Listener {
public:
    /** @brief The panel of @p proc. @param proc the processor */
    explicit AdditivEditor(AdditivProcessor& proc);
    ~AdditivEditor() override;

    /** @brief The window's background (the frame's backdrop). @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The conductor's countdown: a thin line under Step that runs down to the next exchange. @param g where */
    void paintOverChildren(juce::Graphics& g) override;
    /** @brief The header, the picture, the pages, the help. */
    void resized() override;
    /** @brief The family's keys. @param key the key @return true when it was one */
    bool keyPressed(const juce::KeyPress& key) override;
    /** @brief In the standalone, the window gets a maximise button beside the other two (Noctuary's, Totality's). */
    void parentHierarchyChanged() override;
    /** @brief The host's scale (a screen's density), the zoom on top of it. @param newScale the host's */
    void setScaleFactor(float newScale) override;

private:
    /** @brief The status line, the play button and the picture, fifteen times a second. */
    void timerCallback() override;
    /** @brief Not used: values are read when a gesture ends. @param index ignored @param v ignored */
    void parameterValueChanged(int index, float v) override { (void)index; (void)v; }
    /** @brief A knob's gesture opens and closes an undo step. @param index the parameter @param starting begins or ends */
    void parameterGestureChanged(int index, bool starting) override;

    /** @brief Every parameter's value in its unit, in ParamId order. @return the values */
    std::vector<float> values() const;
    /** @brief Applies an undo step's values. @param step the step (null: nothing) @param after the after values (redo) or the before */
    void apply(const frame::UndoStep* step, bool after);
    /** @brief Opens or closes the help. @param on open */
    void showHelp(bool on);
    /** @brief The settings menu under its button. */
    void showSettings();
    /** @brief Whether this is the standalone. @return true there */
    bool standalone() const { return proc_.wrapperType == juce::AudioProcessor::wrapperType_Standalone; }
    /** @brief The standalone's full screen on or off. */
    void toggleFullScreen();
    /** @brief The canvas on or off: the picture takes the whole window, the pages step aside. @param on canvas */
    void setCanvas(bool on);
    /** @brief Opens the command palette (Ctrl+K). */
    void openPalette();
    /** @brief The night mode on or off (kept in the player's settings). @param on night */
    void setNight(bool on);
    /** @brief The first steps' tour, from its first step. */
    void startTour();
    /** @brief Everything the palette can do: the actions, every knob, every preset. @return the commands */
    std::vector<CommandPalette::Command> paletteCommands();
    /**
     * @brief Shows a parameter's control: its page in front, scrolled to it, lighting up.
     * @param id  the parameter
     * @return    false when no page shows it
     */
    bool revealParam(int id);
    /** @brief A header chooser bound to a parameter. @param box the chooser @param id the parameter */
    void bindChoice(frame::Choice& box, additiv::ParamId id);
    /**
     * @brief The window's first size: the size it was left at, where that still fits the screen; else three
     *        quarters of the screen's width at the window's proportions, 1280 x 860 at least (less only on
     *        a screen smaller than that). 1280 x 860 in the screenshot and layout modes.
     * @return pixels
     */
    juce::Point<int> startSize();
    /**
     * @brief A page's state for its strip: its groups on and off, the matrix's rows, a journey or route running.
     * @param tab  the page
     * @return     its state
     */
    PageStrip::Status pageStatus(int tab);
    /** @brief A source chosen in the dock (or dropped on a knob): the inspector shows it, open. @param s the source @param toggle a second click on it shuts the inspector */
    void chooseSource(additiv::ModSource s, bool toggle);
    /** @brief The inspector open or shut (docked: remembered; over the pages: until shut). @param open open */
    void setInspector(bool open);

    AdditivProcessor& proc_;                                   ///< the instrument
    frame::LookAndFeel lnf_{ additivui::skin() };              ///< first, so it outlives every component that uses it
    juce::TooltipWindow tooltips_{ nullptr, 700 };             ///< tooltips after 700 ms
    frame::ControlActions actions_;                            ///< the controls' right-click menu
    frame::Backdrop backdrop_;                                 ///< the window's background
    juce::Rectangle<float> logo_;                              ///< where the logo is drawn
    TitleComponent title_;                                     ///< the name
    frame::IconButton play_{ frame::IconButton::Icon::Play, "Autoplay: start the conductor" };   ///< Autoplay on and off
    juce::TextButton step_{ "Step" };                          ///< Autoplay: exchange a voice now
    juce::TextButton prevPreset_{ "<" };                       ///< the preset before
    juce::TextButton presetName_;                              ///< the preset playing; a click opens the bank
    juce::TextButton nextPreset_{ ">" };                       ///< the preset after
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
     * @param index  the preset
     */
    void loadPreset(int index);
    frame::Choice mode_{ &actions_, static_cast<int>(additiv::ParamId::AutoMode) };      ///< Free or Chords
    frame::Choice scale_{ &actions_, static_cast<int>(additiv::ParamId::Scale) };        ///< the scale
    frame::Choice root_{ &actions_, static_cast<int>(additiv::ParamId::RootNote) };      ///< its root
    frame::Choice motion_{ &actions_, static_cast<int>(additiv::ParamId::Motion) };      ///< the motion
    frame::Choice landscape_{ &actions_, static_cast<int>(additiv::ParamId::Landscape) };///< the landscape
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>> headerAttach_;      ///< the header's choosers to their parameters
    juce::Label status_;                                       ///< voices, root, GPU, latency, device
    frame::IconButton undo_{ frame::IconButton::Icon::Undo, "Undo (Ctrl+Z)" };          ///< undo
    frame::IconButton redo_{ frame::IconButton::Icon::Redo, "Redo (Ctrl+Y)" };          ///< redo
    frame::IconButton help_{ frame::IconButton::Icon::Help, "Help (F1)" };              ///< the help
    /** @brief Save the sound as one's own preset (Ctrl+S), back to the preset, the presets' folder. */
    frame::IconButton save_{ frame::IconButton::Icon::Save, "Your presets: save the sound (Ctrl+S), save it as ..., back to the preset" };
    std::unique_ptr<juce::AlertWindow> askName_;               ///< the name dialog while it is open
    frame::IconButton settings_{ frame::IconButton::Icon::Settings, "Settings" };       ///< the settings
    CloudView cloud_{ proc_ };                                 ///< the picture
    ViewButton canvasButton_{ ViewButton::Glyph::Canvas, "Canvas: the partials and the landscape alone, the whole window (again, or Esc: back)" };   ///< the picture alone
    ViewButton searchButton_{ ViewButton::Glyph::Search, "Search (Ctrl+K): any knob, preset or action by a few letters" };   ///< the command palette
    ViewButton compare_{ ViewButton::Glyph::Compare, "Hold to hear the preset as it was; let go for your sound" };   ///< hold: the preset
    frame::IconButton midi_{ frame::IconButton::Icon::MidiLearn, "Show the MIDI controllers learned on the knobs" };   ///< MIDI marks on the knobs
    OutputMeter meter_;                                        ///< the output
    CommandPalette palette_;                                   ///< the command palette, over everything while open
    PresetBrowser browser_{ proc_ };                           ///< the preset browser, over everything while open
    std::unique_ptr<InfoLine> info_;                           ///< the info line at the bottom: what the control under the mouse does
    ViewButton nightButton_{ ViewButton::Glyph::Night, "Night: the window dim and warm for a dark room, the picture calmer" };   ///< the night mode
    bool night_ = false;                                       ///< the night mode is on: the window dim and warm
    bool tourAsked_ = false;                                   ///< the first start's tour was considered
    double lastActivity_ = 0.0;                                ///< when the mouse or a key last moved (ms)
    int refreshTick_ = 0;                                      ///< timer ticks (the night refreshes the picture half as often)
    /** @brief Notes every move of the mouse in the window (the night's canvas fades when left alone). */
    struct Activity final : public juce::MouseListener {
        double* last = nullptr;   ///< where the time goes
        /** @brief A move. @param e ignored */
        void mouseMove(const juce::MouseEvent&) override { *last = juce::Time::getMillisecondCounterHiRes(); }
        /** @brief A press. @param e ignored */
        void mouseDown(const juce::MouseEvent&) override { *last = juce::Time::getMillisecondCounterHiRes(); }
        /** @brief A wheel. @param e ignored @param w ignored */
        void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails&) override { *last = juce::Time::getMillisecondCounterHiRes(); }
    };
    Activity activity_;                                        ///< the mouse's moves
    /** @brief The night's veil: over the window, under the palette, the browser and the tour; the mouse goes through it. */
    struct Veil final : public juce::Component {
        float alpha = 0.0f;   ///< how dark
        /** @brief Dark and warm: little blue for eyes used to the dark. @param g where */
        void paint(juce::Graphics& g) override { g.fillAll(juce::Colour(0xff0c0703).withAlpha(alpha)); }
    };
    Veil veil_;                                                ///< the night's veil
    Tour tour_;                                                ///< the first steps
    bool canvas_ = false;                                      ///< the canvas is shown
    frame::LiveRings live_{ [this](int id) { return proc_.engine().playedNorm(static_cast<additiv::ParamId>(id)); } };   ///< where modulated knobs play
    juce::TabbedComponent tabs_{ juce::TabbedButtonBar::TabsAtTop };   ///< the pages
    std::vector<std::function<juce::String()>> describe_;     ///< per page its parameters in words, for the help
    std::unique_ptr<frame::HelpView> helpView_;                ///< the help, while it is open
    frame::UndoHistory history_;                               ///< undo and redo
    bool applying_ = false;                                    ///< an undo is being applied: record nothing
    int headerBottom_ = 0;                                     ///< where the header ends, for the backdrop
    juce::String shotPrefix_;                                  ///< ADDITIV_SHOT: every page as <prefix>-<n>-<page>.png (ADDITIV_PRESET first), then the standalone quits
    int shotTick_ = 0;                                         ///< timer ticks since the screenshots were asked for
    juce::MemoryBlock shotSaved_;                              ///< the state the screenshots found, given back before the standalone quits (it saves its state then)
    int logTick_ = 0;                                          ///< timer ticks, for the ADDITIV_LOG diagnostics
    bool layoutChecked_ = false;                               ///< ADDITIV_LAYOUT: the pages were measured
    bool rememberSize_ = false;                                ///< the window's size is kept for the next start (not in the shot or layout modes)
    PageStrip strip_;                                          ///< the pages' strip
    /** @brief A source dropped on a knob: the inspector shows it (or all rows are taken). @param s the source @param id the parameter @param row the row, -1 none */
    void modDropped(additiv::ModSource s, int id, int row) override;
    /** @brief The parameter pages on a page, as they are now. @param tab the page @return them */
    std::vector<ParamPage*> pagesOn(int tab);
    std::unique_ptr<ModDock> dock_;                            ///< the sources along the bottom
    std::unique_ptr<SourceInspector> inspector_;               ///< one source seen whole
    juce::Component* matrixPage_ = nullptr;                    ///< the Modulation page (owned by its tab), to mark the source chosen
    bool inspectorOpen_ = true;                                ///< the docked inspector is wanted (remembered)
    bool overlayOpen_ = false;                                 ///< the inspector is open over the pages (a small window)
    bool inspectorDocked_ = false;                             ///< the window is tall enough to dock it
    float hostScale_ = 1.0f;                                   ///< the scale the host asked for
    float zoom_ = 1.0f;                                        ///< the player's zoom: the controls larger (player.settings "zoom")
    /** @brief The zoom anew (kept in the player's settings). @param z 1 .. 1.5 */
    void setZoom(float z);
    int altRow_ = -1;                                          ///< the matrix row an Alt-drag on a knob sets, -1 none
    float altDepth_ = 0.0f;                                    ///< its depth when the drag began
    float stepLeft_ = -1.0f;                                   ///< the share of the conductor's wait still to come, -1 none (the line under Step)
    juce::String lastMatrix_;                                  ///< the matrix as the undo history last knew it
    juce::String pendingBefore_;                               ///< a matrix edit outside an undo step: the matrix before it
    double pendingAt_ = 0.0;                                   ///< when it last moved (ms); a step once it rests
    juce::String presetKey_;                                   ///< the preset playing, as the matrix's watch last saw it
    /** @brief The matrix as text, the rest of an undo step's state. @return it */
    juce::String matrixText() const { return juce::String(proc_.engine().modText()); }
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
     * @param out the report
     */
    void layoutCheck(const juce::File& out);
};
