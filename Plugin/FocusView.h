/**
 * @file FocusView.h
 * @brief The Focus page: the simple ways in (Focus.h, Find.h).
 *
 *        Nudge: four pairs of buttons -- darker or brighter, calmer or livelier, narrower or wider,
 *        purer or airier -- each moving every heard knob a little the measured way, the level held.
 *        Focus: the knobs that do the most for this sound, all else left on the pages. Variations: six
 *        near the sound now, each listened to while its tile is held and kept with Keep, the next six
 *        then made around it (interactive evolution); Back returns. Teach: a macro learns a gesture by
 *        example -- Teach, move the knobs, set the macro, Here. Find: a recording, opened or dropped,
 *        and the presets that sound most like it (also ringed on the map).
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <array>
#include <functional>
#include <memory>
#include <vector>

/** @brief One variation's tile: what it changes; held, it is listened to. */
class VariationTile final : public juce::Component {
public:
    /** @brief Tile @p k of @p proc's variations. @param proc the processor @param k 0 .. 5 */
    VariationTile(AdditivProcessor& proc, int k) : proc_(proc), k_(k) {}
    /** @brief Its number and what it changes. */
    void paint(juce::Graphics& g) override;
    /** @brief Listens to it. @param e the press */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief Back to the sound before. @param e the release */
    void mouseUp(const juce::MouseEvent& e) override;

private:
    AdditivProcessor& proc_;   ///< whose variations
    int k_;                    ///< which
};

/** @brief The Focus page (see the file comment). */
class FocusPage final : public juce::Component, public juce::FileDragAndDropTarget, private juce::Timer {
public:
    /** @brief A change made from the page as one undo step: its name, and the change. */
    using Undoable = std::function<void(const juce::String&, std::function<void()>)>;
    /**
     * @brief Builds the page.
     * @param proc      the processor
     * @param actions   the knobs' right-click menu
     * @param live      the live rings
     * @param undoable  wraps a change into one undo step (the editor's history)
     */
    FocusPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live, Undoable undoable);
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const;
    /** @brief The boxes from the top. */
    void resized() override;
    /** @brief The boxes and their titles. */
    void paint(juce::Graphics& g) override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;
    /** @brief A recording dragged over the page. @param files the files @return true for one WAV or FLAC */
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    /** @brief A recording dropped: the presets like it. @param files the files @param x where @param y where */
    void filesDropped(const juce::StringArray& files, int x, int y) override;

private:
    /** @brief The focus anew when what matters changed; the statuses. */
    void timerCallback() override;
    /** @brief The focus box's knobs anew. @param ids the knobs */
    void rebuildFocus(const std::vector<additiv::ParamId>& ids);
    /** @brief Searches the presets like a recording. @param f the file */
    void find(const juce::File& f);
    /** @brief The search's results into their buttons. */
    void showFound();
    /** @brief The boxes' rectangles at a width. @param w pixels @return nudge, focus, variations, teach, find */
    std::array<juce::Rectangle<int>, 5> boxes(int w) const;

    AdditivProcessor& proc_;                       ///< the processor
    const frame::ControlActions& actions_;         ///< the knobs' menu
    frame::LiveRings* live_;                       ///< the live rings
    Undoable undoable_;                            ///< one undo step
    std::array<juce::TextButton, 4> lower_;        ///< darker, calmer, narrower, purer
    std::array<juce::TextButton, 4> higher_;       ///< brighter, livelier, wider, airier
    juce::Label nudgeStatus_;                      ///< what the last nudge did
    std::unique_ptr<ParamPage> focus_;             ///< the knobs that matter
    std::vector<additiv::ParamId> focusIds_;       ///< which
    juce::TextButton make_{ "Variations" };        ///< six new ones
    juce::TextButton back_{ "Back" };              ///< the sound before the last kept
    juce::Slider far_;                             ///< how far
    juce::Label farName_;                          ///< its name
    std::vector<std::unique_ptr<VariationTile>> tiles_;   ///< six tiles
    std::array<juce::TextButton, 6> keep_;         ///< keep that one
    std::vector<std::unique_ptr<frame::Knob>> macro_;   ///< the macros, large knobs
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> macroLinks_;   ///< to their parameters
    std::array<juce::Label, 4> macroName_;         ///< their names
    std::array<juce::TextButton, 4> teach_;        ///< begin teaching
    std::array<juce::TextButton, 4> here_;         ///< the sound now is the macro's place here
    std::array<juce::TextButton, 4> forget_;       ///< forget the lesson
    std::array<juce::Label, 4> teachStatus_;       ///< what each was taught
    juce::TextButton open_{ "Open a recording ..." };   ///< the search's file
    juce::Label findStatus_;                       ///< what was searched
    std::array<juce::TextButton, 8> results_;      ///< the presets found, nearest first
    std::unique_ptr<juce::FileChooser> chooser_;   ///< the open dialog while it is open
    int tick_ = 0;                                 ///< timer ticks
};
