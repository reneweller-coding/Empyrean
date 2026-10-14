/**
 * @file ModDock.h
 * @brief The modulation dock along the bottom of the window, as Vital and Pigments keep their sources
 *        in sight: every source a chip with its last four seconds drawn small. Drag a chip onto any
 *        knob of any page and the source modulates it (a new row of the matrix, its arc round the knob
 *        at once); click a chip and the inspector shows the source -- its trace large, its own knobs,
 *        and where it goes: every row's target (a click shows the knob), its depth, a button that
 *        removes it. The inspector is docked under the pages where the window is tall enough; in a
 *        smaller one it opens over the pages' bottom while it is wanted.
 *
 *        The chips: the LFOs, envelopes and macros, the mod wheel, pressure, velocity and the random
 *        curve; then every other source a row of the matrix reads; then one chosen with "+".
 */
#pragma once

#include "Frame.h"
#include "PluginProcessor.h"

#include "additiv/Modulation.h"

#include <array>
#include <functional>
#include <memory>
#include <vector>

/** @brief The sources as chips (see the file comment). */
class ModDock final : public juce::Component, public juce::SettableTooltipClient, private juce::Timer {
public:
    static constexpr int kHistory = 120;   ///< samples a trace holds: four seconds at 30 a second
    static constexpr int kHeight = 30;     ///< the dock's height

    /** @brief The sources of @p proc's engine. @param proc the processor */
    explicit ModDock(AdditivProcessor& proc);

    /** @brief A chip was clicked: its source. */
    std::function<void(additiv::ModSource)> onChoose;
    /** @brief The dock's button: the inspector open or shut. */
    std::function<void()> onToggle;
    /** @brief The source the inspector shows (its chip is marked). @param s the source */
    void setChosen(additiv::ModSource s);
    /** @brief Whether the inspector is open (the button's arrow). @param open open */
    void setOpen(bool open);
    /**
     * @brief A source's trace, oldest first.
     * @param s  the source
     * @param k  0 .. kHistory-1
     * @return   its value then
     */
    float traceAt(additiv::ModSource s, int k) const;

    /** @brief The chips. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The button at the left. */
    void resized() override;
    /** @brief Remembers the chip pressed. @param e the press */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief Drags the chip's source towards a knob. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override;
    /** @brief A click without a drag chooses the chip; "+" offers every source. @param e the release */
    void mouseUp(const juce::MouseEvent& e) override;
    /** @brief The chip under the mouse lights up. @param e the move */
    void mouseMove(const juce::MouseEvent& e) override;
    /** @brief Nothing lights up. @param e the exit */
    void mouseExit(const juce::MouseEvent& e) override;
    /** @brief What the chip under the mouse is and does (the info line shows it). @return the text */
    juce::String getTooltip() override;

private:
    /** @brief Samples every source; repaints while on screen. */
    void timerCallback() override;
    /** @brief The chips to show now: the classic sources, those the matrix reads, the one chosen with "+". @return them */
    std::vector<additiv::ModSource> wanted() const;
    /** @brief The chips' area (between the button and "+"). @return pixels */
    juce::Rectangle<int> chipArea() const;
    /** @brief A chip's rectangle. @param k its place @param n the chips @return pixels */
    juce::Rectangle<int> chipBounds(size_t k, size_t n) const;
    /** @brief The "+" chip. @return pixels */
    juce::Rectangle<int> plusBounds() const;
    /** @brief The chip at a point. @param p the point @return its place, -1 none, -2 the "+" */
    int chipAt(juce::Point<int> p) const;
    /** @brief The menu of every source: the one chosen gets a chip. */
    void showMore();
    /** @brief The rows that read a source. @param s the source @return how many */
    int rowsOf(additiv::ModSource s) const;

    AdditivProcessor& proc_;                                                         ///< whose engine
    juce::TextButton toggle_;                                                        ///< the inspector open or shut
    std::vector<additiv::ModSource> shown_;                                          ///< the chips as last laid out
    additiv::ModSource pinned_ = additiv::ModSource::Off;                           ///< the source chosen with "+"
    additiv::ModSource chosen_ = additiv::ModSource::Off;                           ///< the source the inspector shows
    int pressed_ = -1;                                                               ///< the chip pressed, -1 none, -2 "+"
    bool dragging_ = false;                                                          ///< a chip is being dragged
    int hover_ = -1;                                                                 ///< the chip under the mouse
    std::array<std::array<float, kHistory>, additiv::kNumModSources> history_{};    ///< per source its last values, a ring
    int head_ = 0;                                                                   ///< where the next sample goes
};

/** @brief One source seen whole: its trace, its own knobs, its rows (see ModDock.h's file comment). */
class SourceInspector final : public juce::Component, private juce::Timer {
public:
    static constexpr int kHeight = 112;   ///< the inspector's height

    /**
     * @brief The inspector.
     * @param proc     the processor
     * @param actions  the knobs' right-click menu
     * @param dock     the dock, whose traces it draws
     */
    SourceInspector(AdditivProcessor& proc, const frame::ControlActions& actions, const ModDock& dock);
    /** @brief Shows a source. @param s the source */
    void show(additiv::ModSource s);
    /** @brief The source shown. @return it */
    additiv::ModSource source() const { return source_; }
    /** @brief Over the pages (with a button that shuts it) or docked under them. @param on over the pages */
    void setOverlay(bool on);
    /** @brief The shut button was pressed (over the pages). */
    std::function<void()> onClose;
    /** @brief A row's target was clicked: its parameter, to show its knob. */
    std::function<void(int)> onReveal;
    /** @brief The Matrix button: the matrix page. */
    std::function<void()> onMatrix;
    /** @brief A source's own parameters (an LFO's shape and rate, a macro, the attractors' period ...). @param s the source @return them */
    static std::vector<additiv::ParamId> ownParams(additiv::ModSource s);
    /** @brief The source whose own knob a parameter is (the first, for knobs shared by a family). @param id the parameter @return it, Off none */
    static additiv::ModSource sourceOf(int id);

    /** @brief The box, the name, the trace. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The knobs and the rows. */
    void resized() override;

private:
    /** @brief The trace anew; the rows anew when the matrix changed. */
    void timerCallback() override;
    /** @brief The rows' controls anew from the matrix. */
    void rebuildRows();
    /** @brief The source's own knobs anew. */
    void rebuildControls();
    /** @brief The rows on the source, as a key: changes when one is added, removed or retargeted. @return the key */
    juce::String rowsKey() const;
    /** @brief Where the trace is drawn. @return pixels */
    juce::Rectangle<int> traceArea() const;

    /** @brief One row's controls: its target, its depth, its removal. */
    struct RowUi {
        int row = -1;                          ///< the matrix row
        juce::TextButton target;               ///< the target's name; a click shows its knob
        juce::Slider depth;                    ///< how far
        juce::TextButton remove{ "x" };        ///< removes the row
    };

    AdditivProcessor& proc_;                                                    ///< whose engine
    const frame::ControlActions& actions_;                                      ///< the knobs' menu
    const ModDock& dock_;                                                       ///< the traces
    additiv::ModSource source_ = additiv::ModSource::Lfo1;                     ///< the source shown
    bool overlay_ = false;                                                      ///< over the pages
    juce::String key_;                                                          ///< the rows as last built
    juce::Label hint_;                                                          ///< what to do when no row reads it
    juce::TextButton close_{ "x" };                                             ///< shuts it (over the pages)
    juce::TextButton matrix_{ "Matrix ..." };                                   ///< the matrix page
    std::vector<std::unique_ptr<RowUi>> rows_;                                  ///< the rows' controls
    std::vector<std::unique_ptr<juce::Label>> labels_;                          ///< the own knobs' names
    std::vector<std::unique_ptr<juce::Component>> controls_;                    ///< the own knobs and choosers
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> sliders_;    ///< knobs to parameters (before the knobs go)
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>> combos_;   ///< choosers to parameters
};
