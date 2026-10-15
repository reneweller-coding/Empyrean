/**
 * @file Theme.h
 * @brief The instrument's skin for the family's frame (Frame.h): an aurora at night -- deep blue,
 *        the green-teal of the light, violet at its edges -- and a logo made of a ring of partials.
 * @author Rene Weller
 */
#pragma once

#include "Frame.h"

namespace additivui {

namespace colour {
const juce::Colour Bg     { 0xff07090f };   ///< the window: the night
const juce::Colour Panel  { 0xa00c1018 };   ///< a page (translucent)
const juce::Colour GroupColour  { 0xff111725 };   ///< a group box
const juce::Colour Raised { 0xff1a2132 };   ///< buttons, menus, the knobs' bodies
const juce::Colour Edge   { 0xff283149 };   ///< hairlines
const juce::Colour Ink    { 0xffe2ecf2 };   ///< text
const juce::Colour Dim    { 0xff8794a8 };   ///< names, secondary text
const juce::Colour Faint  { 0xff3c4559 };   ///< tracks, axes, the off state
const juce::Colour Accent { 0xff7fe0c4 };   ///< the aurora's green: the title, the tab in front
const juce::Colour Violet { 0xffb69cf0 };   ///< its violet edge: the conductor's notes
const juce::Colour Onset  { 0xffd7799b };   ///< a playhead, a mute
const juce::Colour Root   { 0xffe8b56a };   ///< the conductor's root on the keys
const juce::Colour Good   { 0xff7fd49a };   ///< a meter in range
const juce::Colour Bad    { 0xffe06a5f };   ///< a meter over
} // namespace colour

/** @brief The skin (built once). @return the skin */
const frame::Skin& skin();

/**
 * @brief Draws the logo: a ring of partials of different sizes, brightest at the top.
 * @param[in,out] g  where
 * @param[in]     r  the square
 */
void drawLogo(juce::Graphics& g, juce::Rectangle<float> r);

} // namespace additivui
