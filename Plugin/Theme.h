/**
 * @file Theme.h
 * @brief The instrument's skin for the family's frame (Frame.h): an aurora at night -- deep blue,
 *        the green-teal of the light, violet at its edges -- and a logo made of a ring of partials.
 */
#pragma once

#include "Frame.h"

namespace additivui {

namespace colour {
const juce::Colour bg     { 0xff07090f };   ///< the window: the night
const juce::Colour panel  { 0xa00c1018 };   ///< a page (translucent)
const juce::Colour group  { 0xff111725 };   ///< a group box
const juce::Colour raised { 0xff1a2132 };   ///< buttons, menus, the knobs' bodies
const juce::Colour edge   { 0xff283149 };   ///< hairlines
const juce::Colour ink    { 0xffe2ecf2 };   ///< text
const juce::Colour dim    { 0xff8794a8 };   ///< names, secondary text
const juce::Colour faint  { 0xff3c4559 };   ///< tracks, axes, the off state
const juce::Colour accent { 0xff7fe0c4 };   ///< the aurora's green: the title, the tab in front
const juce::Colour violet { 0xffb69cf0 };   ///< its violet edge: the conductor's notes
const juce::Colour onset  { 0xffd7799b };   ///< a playhead, a mute
const juce::Colour root   { 0xffe8b56a };   ///< the conductor's root on the keys
const juce::Colour good   { 0xff7fd49a };   ///< a meter in range
const juce::Colour bad    { 0xffe06a5f };   ///< a meter over
} // namespace colour

/** @brief The skin (built once). @return the skin */
const frame::Skin& skin();

/** @brief Draws the logo: a ring of partials of different sizes, brightest at the top. @param g where @param r the square */
void drawLogo(juce::Graphics& g, juce::Rectangle<float> r);

} // namespace additivui
