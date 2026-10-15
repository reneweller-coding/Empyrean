/**
 * @file SourcesView.h
 * @brief The modulation sources, live: every source of the matrix a small trace of its last four
 *        seconds, coloured by its kind (the classic ones, fields and chaos, the piece, what the
 *        instrument hears, the reservoir) -- so one sees which moves how before routing it.
 */
#pragma once

#include "PluginProcessor.h"

#include "additiv/Modulation.h"

#include <array>
#include <functional>
#include <utility>
#include <vector>

/** @brief The live sources (see the file comment). */
class SourcesView final : public juce::Component, private juce::Timer {
public:
    /**
     * @brief The sources of @p proc's engine, all or those listed; while it lives, the engine computes them all.
     * @param proc   the processor
     * @param shown  the sources drawn, in order (empty: every one)
     */
    explicit SourcesView(AdditivProcessor& proc, std::vector<additiv::ModSourceE> shown = {});
    /** @brief Lets the engine stop computing what no row reads. */
    ~SourcesView() override;
    /** @brief The traces in a grid. */
    void paint(juce::Graphics& g) override;
    /** @brief A click on a trace chooses its source. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief A source clicked. */
    std::function<void(additiv::ModSourceE)> m_onChoose;
    /** @brief The source marked (the inspector's). @param s the source */
    void setChosen(additiv::ModSourceE s)
    {
        m_chosen = s;
        repaint();
    }
    /**
     * @brief The least height the grid needs at a width: as many columns as fit, the cells at their lowest. Given more,
     *        the cells grow.
     * @param w      pixels
     * @param count  the sources drawn
     * @return       pixels
     */
    static int heightFor(int w, int count = additiv::NumModSources - 1);

private:
    static constexpr int History = 120;          ///< samples a trace holds: four seconds at 30 a second
    static constexpr int CellW = 150;            ///< a cell's least width
    static constexpr int CellH = 42;             ///< a cell's height, as it likes it
    static constexpr int CellMinW = 120;         ///< a cell's least width, when the grid is pressed for height
    static constexpr int CellMinH = 32;          ///< a cell's least height
    static constexpr int CellMaxH = 60;          ///< a cell's greatest height
    static constexpr int Gap = 6;                ///< between the cells
    static constexpr int Head = 26;              ///< the heading's height

    /** @brief Samples every source (while the window is open, so a trace is full when the page is opened); repaints when on screen. */
    void timerCallback() override;
    /** @brief The columns at a width, the cells as wide as they like. @param w pixels @return columns */
    static int columns(int w);
    /** @brief The most columns at a width, the cells at their narrowest. @param w pixels @return columns */
    static int mostColumns(int w);
    /**
     * @brief The grid at a size: as wide as the cells like, more columns only while the cells would be too low.
     * @param w      pixels
     * @param h      pixels
     * @param count  cells
     * @return       the columns and the cells' height
     */
    static std::pair<int, int> grid(int w, int h, int count);

    AdditivProcessor& m_proc;                                                            ///< whose engine
    std::array<std::array<float, History>, additiv::NumModSources> m_history{};       ///< per source its last values, a ring
    int m_head = 0;                                                                      ///< where the next sample goes
    std::vector<additiv::ModSourceE> m_shown;                                             ///< the sources drawn, in order
    additiv::ModSourceE m_chosen = additiv::ModSourceE::OFF;                               ///< the source marked
};
