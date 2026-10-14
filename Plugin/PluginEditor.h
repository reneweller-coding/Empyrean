/**
 * @file PluginEditor.h
 * @brief The panel, on the family's frame (Frame.h): the header (logo and name; Autoplay's play and
 *        stop, its mode and Step; scale, root, motion, landscape; the status; undo, redo, help,
 *        settings), the cloud picture (CloudView), and the pages -- Cloud, Motion, Filter, Tuning,
 *        Conductor, Space, Effects, Modulation, Journey, Map. The family's keys (Space, Ctrl+Z/Y, F1, F11, Esc), its right-click menu
 *        on every control (MIDI learn, forget, default) and its window size (1280 x 860).
 */
#pragma once

#include "CloudView.h"
#include "Frame.h"
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
                            private juce::Timer,
                            private juce::AudioProcessorParameter::Listener {
public:
    /** @brief The panel of @p proc. @param proc the processor */
    explicit AdditivEditor(AdditivProcessor& proc);
    ~AdditivEditor() override;

    /** @brief The window's background (the frame's backdrop). @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The header, the picture, the pages, the help. */
    void resized() override;
    /** @brief The family's keys. @param key the key @return true when it was one */
    bool keyPressed(const juce::KeyPress& key) override;
    /** @brief In the standalone, the window gets a maximise button beside the other two (Noctuary's, Totality's). */
    void parentHierarchyChanged() override;

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
    /** @brief A header chooser bound to a parameter. @param box the chooser @param id the parameter */
    void bindChoice(frame::Choice& box, additiv::ParamId id);

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
    frame::IconButton settings_{ frame::IconButton::Icon::Settings, "Settings" };       ///< the settings
    CloudView cloud_{ proc_ };                                 ///< the picture
    juce::TextButton canvasButton_{ "Canvas" };                ///< the picture alone, the whole window (and back)
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
    /** @brief One step of the screenshot mode (ADDITIV_SHOT), from the timer. */
    void shootStep();
};
