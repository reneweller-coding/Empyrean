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

/** @brief The live sources (see the file comment). */
class SourcesView final : public juce::Component, private juce::Timer {
public:
    /** @brief The sources of @p proc's engine; while it lives, the engine computes them all. @param proc the processor */
    explicit SourcesView(AdditivProcessor& proc);
    /** @brief Lets the engine stop computing what no row reads. */
    ~SourcesView() override;
    /** @brief The traces in a grid. */
    void paint(juce::Graphics& g) override;
    /** @brief The height the grid needs at a width. @param w pixels @return pixels */
    static int heightFor(int w);

private:
    static constexpr int kHistory = 120;          ///< samples a trace holds: four seconds at 30 a second
    static constexpr int kCellW = 150;            ///< a cell's least width
    static constexpr int kCellH = 42;             ///< a cell's height
    static constexpr int kGap = 6;                ///< between the cells
    static constexpr int kHead = 26;              ///< the heading's height

    /** @brief Samples every source (while the window is open, so a trace is full when the page is opened); repaints when on screen. */
    void timerCallback() override;
    /** @brief The columns at a width. @param w pixels @return columns */
    static int columns(int w);

    AdditivProcessor& proc_;                                                            ///< whose engine
    std::array<std::array<float, kHistory>, additiv::kNumModSources> history_{};       ///< per source its last values, a ring
    int head_ = 0;                                                                      ///< where the next sample goes
};
