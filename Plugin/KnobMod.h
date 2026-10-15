/**
 * @file KnobMod.h
 * @brief Modulation at the knob: what moves a knob is seen and changed where the knob is, not only on
 *        the Modulation page. Every matrix row on a knob draws an arc round it in its source's colour
 *        (the span it can move the knob over, and a dot where it moves it now); the knob's right-click
 *        menu lists those rows, each with its depth to drag and a button to remove it, and offers every
 *        source to modulate it with (a new row in the first free one). The rows are the engine's, as
 *        the Modulation page writes them (Modulation.h): the page follows on its own.
 */
#pragma once

#include "PluginProcessor.h"

#include "additiv/Modulation.h"

namespace knobmod {

/**
 * @brief A source's colour: by its kind, as the matrix's menu sections them (the classic ones, fields
 *        and chaos, the piece, what the instrument hears, the reservoir) -- the same everywhere.
 * @param s  the source
 * @return   the colour
 */
juce::Colour sourceColour(additiv::ModSourceE s);

/**
 * @brief Whether a parameter can be a matrix row's target.
 * @param id  the parameter (ParamId)
 * @return    true then
 */
bool canTarget(int id);

/**
 * @brief A new row in the first free one: @p source on parameter @p id at @p depth -- or the row that
 *        already has that source on that parameter (without a via source).
 * @param proc    the processor
 * @param source  where from
 * @param id      the parameter (ParamId)
 * @param depth   how far, of the knob's range (a fifth: heard, and easy to change)
 * @return        the row, -1 when all sixteen are taken or the parameter is no target
 */
int addRow(AdditivProcessor& proc, additiv::ModSourceE source, int id, float depth = 0.2f);

/**
 * @brief What a source's chip carries while it is dragged onto a knob (DragAndDropContainer).
 * @param s  the source
 * @return   the drag's description
 */
juce::var dragOf(additiv::ModSourceE s);

/**
 * @brief The source a drag carries.
 * @param description  the drag's description
 * @return             the source, Off when it carries none
 */
additiv::ModSourceE sourceOfDrag(const juce::var& description);

/**
 * @brief Adds the modulation items for parameter @p id to a knob's menu: the rows on it (depth, remove)
 *        and "Modulate with". Nothing for a parameter that cannot be a target.
 * @param proc  the processor (its engine holds the rows)
 * @param m     the menu
 * @param id    the parameter (ParamId)
 */
void addMenuItems(AdditivProcessor& proc, juce::PopupMenu& m, int id);

/**
 * @brief Draws the rows on a knob as arcs outside its ring: per row the span it moves the knob over
 *        (both ways for a source that swings either way), faint, and a dot where it moves it now.
 * @param g       where (the knob's parent's graphics)
 * @param proc    the processor
 * @param knob    the knob
 * @param id      its parameter (ParamId)
 * @return        true when any row is on it (the parent repaints it while the sources move)
 */
bool paintArcs(juce::Graphics& g, AdditivProcessor& proc, juce::Slider& knob, int id);

} // namespace knobmod
