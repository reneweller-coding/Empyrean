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
juce::Colour sourceColour(additiv::ModSource s);

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
