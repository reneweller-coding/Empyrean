/**
 * @file FocusView.h
 * @brief The Focus page: the simple ways in (Focus.h, Find.h).
 *
 *        Nudge: four pairs of buttons -- darker or brighter, calmer or livelier, narrower or wider,
 *        purer or airier -- each moving every heard knob a little the measured way, the level held.
 *        Focus: the knobs that do the most for this sound, all else left on the pages. Variations: six
 *        near the sound now, made on their own whenever the sound has moved (not while a blend plays),
 *        each listened to while its tile is held and kept with Keep (shown once it was heard), the next
 *        six then made around it (interactive evolution); Back returns. Snapshots: the first free tile
 *        is a "+" that keeps the sound. Teach: a macro learns a gesture by example -- Teach, move the
 *        knobs, set the macro, Set here (or Cancel); a taught macro has a small x to forget it. Find: a
 *        recording, opened or dropped, and the presets that sound most like it (also ringed on the map).
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
    VariationTile(AdditivProcessor& proc, int k) : m_proc(proc), m_k(k) {}
    /** @brief Its number and what it changes. */
    void paint(juce::Graphics& g) override;
    /** @brief Listens to it. @param e the press */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief Back to the sound before. @param e the release */
    void mouseUp(const juce::MouseEvent& e) override;
    /** @brief It was listened to (its Keep is shown). */
    std::function<void(int)> m_onHeard;

private:
    AdditivProcessor& m_proc;   ///< whose variations
    int m_k;                    ///< which
};

/** @brief The Focus page (see the file comment). */
class FocusPage final : public juce::Component, public juce::FileDragAndDropTarget, private juce::Timer {
public:
    /** @brief A change made from the page as one undo step: its name, and the change. */
    using UndoableT = std::function<void(const juce::String&, std::function<void()>)>;
    /**
     * @brief Builds the page.
     * @param proc      the processor
     * @param actions   the knobs' right-click menu
     * @param live      the live rings
     * @param undoable  wraps a change into one undo step (the editor's history)
     * @param picture   the partials as the picture shows them now, small (a snapshot's)
     */
    FocusPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live, UndoableT undoable,
              std::function<juce::Image()> picture);
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
    void rebuildFocus(const std::vector<additiv::ParamIdE>& ids);
    /** @brief Searches the presets like a recording. @param f the file */
    void find(const juce::File& f);
    /** @brief The search's results into their buttons. */
    void showFound();
    /** @brief The boxes' rectangles at a width. @param w pixels @return nudge, snapshots, focus, variations, teach, find */
    std::array<juce::Rectangle<int>, 6> boxes(int w) const;
    /** @brief A snapshot's tile: its picture; a click recalls it, a right-click offers A, B, forget. */
    class SnapshotTile;

    AdditivProcessor& m_proc;                       ///< the processor
    const frame::ControlActions& m_actions;         ///< the knobs' menu
    frame::LiveRings* m_live;                       ///< the live rings
    UndoableT m_undoable;                            ///< one undo step
    std::function<juce::Image()> m_picture;         ///< the partials now, small
    juce::TextButton m_snap{ "Snapshot" };          ///< keeps the sound now
    std::vector<std::unique_ptr<SnapshotTile>> m_snapTiles;   ///< eight tiles
    std::array<juce::TextButton, 4> m_lower;        ///< darker, calmer, narrower, purer
    std::array<juce::TextButton, 4> m_higher;       ///< brighter, livelier, wider, airier
    juce::Label m_nudgeStatus;                      ///< what the last nudge did
    std::unique_ptr<ParamPage> m_focus;             ///< the knobs that matter
    std::vector<additiv::ParamIdE> m_focusIds;       ///< which
    juce::TextButton m_make{ "Variations" };        ///< six new ones
    juce::TextButton m_back{ "Back" };              ///< the sound before the last kept
    juce::Slider m_far;                             ///< how far
    juce::Label m_farName;                          ///< its name
    std::vector<std::unique_ptr<VariationTile>> m_tiles;   ///< six tiles
    std::array<juce::TextButton, 6> m_keep;         ///< keep that one
    std::vector<std::unique_ptr<frame::Knob>> m_macro;   ///< the macros, large knobs
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> m_macroLinks;   ///< to their parameters
    std::array<juce::Label, 4> m_macroName;         ///< their names
    std::array<juce::TextButton, 4> m_teach;        ///< begin teaching
    std::array<juce::TextButton, 4> m_here;         ///< the sound now is the macro's place here
    std::array<juce::TextButton, 4> m_forget;       ///< forget the lesson (a small x, once taught)
    std::array<juce::TextButton, 4> m_cancel;       ///< end the teaching without a new place
    std::array<bool, 6> m_heard{};                  ///< which variations were listened to (their Keep shown)
    int m_generation = -1;                          ///< the variations' generation the tiles show
    double m_pulse = 0.0;                           ///< the teaching's pulse, seconds
    std::array<juce::Label, 4> m_teachStatus;       ///< what each was taught
    juce::TextButton m_open{ "Open a recording ..." };   ///< the search's file
    juce::Label m_findStatus;                       ///< what was searched
    std::array<juce::TextButton, 8> m_results;      ///< the presets found, nearest first
    std::unique_ptr<juce::FileChooser> m_chooser;   ///< the open dialog while it is open
    int m_tick = 0;                                 ///< timer ticks
};
