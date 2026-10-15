/**
 * @file Shapes.h
 * @brief The factory bank of envelope shapes: what the lanes of a CloudVoice share among their partials.
 * @author Rene Weller
 */
#pragma once

#include "additiv/Cloud.h"

#include <initializer_list>
#include <vector>

namespace additiv {

/** @brief The factory shapes, by their place in factoryShapes(). */
enum FactoryShapeE : int {
    SWELL = 0,          ///< 0 -> 1 -> 0 over 8 s as a raised cosine, looped: a slow breath
    PULSE,              ///< a quick rise and a long fall, looped every 4 s
    BLOOM,              ///< rises once over 3 s, then breathes between 1 and 0.4 every 6 s; at release it falls to 0 over 4 s
    FLICKER,            ///< six uneven steps, looped every 3 s
    WOBBLE,             ///< bipolar: 0 -> 1 -> -1 -> 0 over 8 s, looped (for pitch and pan)
    RISE_IN,             ///< bipolar, once: from -1 to 0 over 12 s, fast at first (a glide in from below)
    FALL_IN,             ///< bipolar, once: from +1 to 0 over 12 s, fast at first (a glide in from above)
    SAW_UP,              ///< 0 -> 1 over 4 s, then back at once, looped
    FACTORY_SHAPE_COUNT   ///< how many factory shapes there are
};

/**
 * @brief A shape from its points.
 * @param[in] points     the breakpoints (at most EnvPoints are taken)
 * @param[in] loopStart  first point of the loop, -1 = none
 * @param[in] loopEnd    last point of the loop
 * @return the shape
 */
EnvShape makeShape(std::initializer_list<EnvPoint> points, int loopStart, int loopEnd);

/** @brief The factory bank, in FactoryShape order. @return FACTORY_SHAPE_COUNT shapes */
std::vector<EnvShape> factoryShapes();

/**
 * @brief A factory shape's name.
 * @param[in] i  its FactoryShape
 * @return the name ("?" outside the bank)
 */
const char* factoryShapeName(int i);

} // namespace additiv
