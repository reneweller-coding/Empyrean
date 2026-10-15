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
 * @author Rene Weller
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
    static constexpr int History = 120;   ///< samples a trace holds: four seconds at 30 a second
    static constexpr int Height = 30;     ///< the dock's height

    /**
     * @brief The sources of @p proc's engine.
     * @param[in,out] proc  the processor
     */
    explicit ModDock(AdditivProcessor& proc);

    /** @brief A chip was clicked: its source. */
    std::function<void(additiv::ModSourceE)> m_onChoose;
    /** @brief The dock's button: the inspector open or shut. */
    std::function<void()> m_onToggle;
    /**
     * @brief The source the inspector shows (its chip is marked).
     * @param[in] s  the source
     */
    void setChosen(additiv::ModSourceE s);
    /**
     * @brief Whether the inspector is open (the button's arrow).
     * @param[in] open  open
     */
    void setOpen(bool open);
    /**
     * @brief A source's trace, oldest first.
     * @param[in] s  the source
     * @param[in] k  0 .. History-1
     * @return its value then
     */
    float traceAt(additiv::ModSourceE s, int k) const;

    /**
     * @brief The chips.
     * @param[in,out] g  where
     */
    void paint(juce::Graphics& g) override;
    /** @brief The button at the left. */
    void resized() override;
    /**
     * @brief Remembers the chip pressed.
     * @param[in] e  the press
     */
    void mouseDown(const juce::MouseEvent& e) override;
    /**
     * @brief Drags the chip's source towards a knob.
     * @param[in] e  the drag
     */
    void mouseDrag(const juce::MouseEvent& e) override;
    /**
     * @brief A click without a drag chooses the chip; "+" offers every source.
     * @param[in] e  the release
     */
    void mouseUp(const juce::MouseEvent& e) override;
    /**
     * @brief The chip under the mouse lights up.
     * @param[in] e  the move
     */
    void mouseMove(const juce::MouseEvent& e) override;
    /**
     * @brief Nothing lights up.
     * @param[in] e  the exit
     */
    void mouseExit(const juce::MouseEvent& e) override;
    /** @brief What the chip under the mouse is and does (the info line shows it). @return the text */
    juce::String getTooltip() override;

private:
    /** @brief Samples every source; repaints while on screen. */
    void timerCallback() override;
    /** @brief The chips to show now: the classic sources, those the matrix reads, the one chosen with "+". @return them */
    std::vector<additiv::ModSourceE> wanted() const;
    /** @brief The chips' area (between the button and "+"). @return pixels */
    juce::Rectangle<int> chipArea() const;
    /**
     * @brief A chip's rectangle.
     * @param[in] k  its place
     * @param[in] n  the chips
     * @return pixels
     */
    juce::Rectangle<int> chipBounds(size_t k, size_t n) const;
    /** @brief The "+" chip. @return pixels */
    juce::Rectangle<int> plusBounds() const;
    /**
     * @brief The chip at a point.
     * @param[in] p  the point
     * @return its place, -1 none, -2 the "+"
     */
    int chipAt(juce::Point<int> p) const;
    /** @brief The menu of every source: the one chosen gets a chip. */
    void showMore();
    /**
     * @brief The rows that read a source.
     * @param[in] s  the source
     * @return how many
     */
    int rowsOf(additiv::ModSourceE s) const;

    AdditivProcessor& m_proc;                                                         ///< whose engine
    juce::TextButton m_toggle;                                                        ///< the inspector open or shut
    std::vector<additiv::ModSourceE> m_shown;                                          ///< the chips as last laid out
    additiv::ModSourceE m_pinned = additiv::ModSourceE::OFF;                           ///< the source chosen with "+"
    additiv::ModSourceE m_chosen = additiv::ModSourceE::OFF;                           ///< the source the inspector shows
    int m_pressed = -1;                                                               ///< the chip pressed, -1 none, -2 "+"
    bool m_dragging = false;                                                          ///< a chip is being dragged
    int m_hover = -1;                                                                 ///< the chip under the mouse
    std::array<std::array<float, History>, additiv::NumModSources> m_history{};    ///< per source its last values, a ring
    int m_head = 0;                                                                   ///< where the next sample goes
};

/** @brief One source seen whole: its trace, its own knobs, its rows (see ModDock.h's file comment). */
class SourceInspector final : public juce::Component, private juce::Timer {
public:
    static constexpr int Height = 112;   ///< the inspector's height

    /**
     * @brief The inspector.
     * @param[in,out] proc     the processor
     * @param[in]     actions  the knobs' right-click menu
     * @param[in]     dock     the dock, whose traces it draws
     */
    SourceInspector(AdditivProcessor& proc, const frame::ControlActions& actions, const ModDock& dock);
    /**
     * @brief Shows a source.
     * @param[in] s  the source
     */
    void show(additiv::ModSourceE s);
    /** @brief The source shown. @return it */
    additiv::ModSourceE source() const { return m_source; }
    /**
     * @brief Over the pages (with a button that shuts it) or docked under them.
     * @param[in] on  over the pages
     */
    void setOverlay(bool on);
    /** @brief The shut button was pressed (over the pages). */
    std::function<void()> m_onClose;
    /** @brief A row's target was clicked: its parameter, to show its knob. */
    std::function<void(int)> m_onReveal;
    /** @brief The Matrix button: the matrix page. */
    std::function<void()> m_onMatrix;
    /**
     * @brief A source's own parameters (an LFO's shape and rate, a macro, the attractors' period ...).
     * @param[in] s  the source
     * @return them
     */
    static std::vector<additiv::ParamIdE> ownParams(additiv::ModSourceE s);
    /**
     * @brief The source whose own knob a parameter is (the first, for knobs shared by a family).
     * @param[in] id  the parameter
     * @return it, Off none
     */
    static additiv::ModSourceE sourceOf(int id);

    /**
     * @brief The box, the name, the trace.
     * @param[in,out] g  where
     */
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
        int m_row = -1;                          ///< the matrix row
        juce::TextButton m_target;               ///< the target's name; a click shows its knob
        juce::Slider m_depth;                    ///< how far
        juce::TextButton m_remove{ "x" };        ///< removes the row
    };

    AdditivProcessor& m_proc;                                                    ///< whose engine
    const frame::ControlActions& m_actions;                                      ///< the knobs' menu
    const ModDock& m_dock;                                                       ///< the traces
    additiv::ModSourceE m_source = additiv::ModSourceE::LFO1;                     ///< the source shown
    bool m_overlay = false;                                                      ///< over the pages
    juce::String m_key;                                                          ///< the rows as last built
    juce::Label m_hint;                                                          ///< what to do when no row reads it
    juce::TextButton m_close{ "x" };                                             ///< shuts it (over the pages)
    juce::TextButton m_matrix{ "Matrix ..." };                                   ///< the matrix page
    std::vector<std::unique_ptr<RowUi>> m_rows;                                  ///< the rows' controls
    std::vector<std::unique_ptr<juce::Label>> m_labels;                          ///< the own knobs' names
    std::vector<std::unique_ptr<juce::Component>> m_controls;                    ///< the own knobs and choosers
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> m_sliders;    ///< knobs to parameters (before the knobs go)
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>> m_combos;   ///< choosers to parameters
};
