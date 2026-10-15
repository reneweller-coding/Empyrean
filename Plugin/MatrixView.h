/**
 * @file MatrixView.h
 * @brief The modulation matrix on the panel: sixteen rows of source, target and depth, written
 *        straight into the engine (the rows are data, not host parameters -- Modulation.h).
 * @author Rene Weller
 */
#pragma once

#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** @brief The matrix's rows (see the file comment). */
class MatrixView final : public juce::Component, private juce::Timer
{
public:
    /**
     * @brief The rows of @p proc's engine.
     * @param[in,out] proc  the processor
     */
    explicit MatrixView( AdditivProcessor &proc );
    /** @brief One column of sixteen rows where it is high enough (heightFor(1)), else two of eight. */
    void resized() override;
    /**
     * @brief The rows' numbers and the column heads.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override;
    /**
     * @brief The height the rows need in one or two columns.
     * @param[in] columns  1 or 2
     * @return pixels
     */
    static int heightFor( int columns )
    {
        return columns == 1 ? 28 + additiv::ModRows * 28 + 4 : 28 + additiv::ModRows / 2 * 30;
    }
    /** @brief The rows in words, for the help. @return the text */
    juce::String describe() const;

private:
    /** @brief The columns at the view's height. @return 1 or 2 */
    int columns() const { return getHeight() >= heightFor( 1 ) ? 1 : 2; }
    /** @brief One row's controls. */
    struct Row
    {
        juce::ComboBox m_source;   ///< where from
        juce::ComboBox m_target;   ///< what it moves
        juce::ComboBox m_via;      ///< a second source scaling the depth (an amount 0 .. 1), "-" none
        juce::Slider   m_depth;    ///< how far, -1 .. 1
    };
    /** @brief Reads the engine's rows into the controls (a preset or a state may have changed them). */
    void timerCallback() override;
    /**
     * @brief Writes one row's controls into the engine.
     * @param[in] i  the row
     */
    void push( int i );

    AdditivProcessor                 &m_proc;      ///< whose engine
    std::vector<std::unique_ptr<Row>> m_rows;      ///< ModRows of them
    std::vector<int>                  m_targets;   ///< the target menu's ParamIds, in its order (item id = index + 2)
    bool m_updating = false;                       ///< the controls are being set from the engine: push nothing
};
