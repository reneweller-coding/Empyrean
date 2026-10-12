/**
 * @file Shapes.h
 * @brief The factory bank of envelope shapes: what the lanes of a CloudVoice share among their partials.
 */
#pragma once

#include "additiv/Cloud.h"

#include <initializer_list>
#include <vector>

namespace additiv {

/** @brief The factory shapes, by their place in factoryShapes(). */
enum FactoryShape : int {
    kSwell = 0,          ///< 0 -> 1 -> 0 over 8 s as a raised cosine, looped: a slow breath
    kPulse,              ///< a quick rise and a long fall, looped every 4 s
    kBloom,              ///< rises once over 3 s, then breathes between 1 and 0.4 every 6 s; at release it falls to 0 over 4 s
    kFlicker,            ///< six uneven steps, looped every 3 s
    kWobble,             ///< bipolar: 0 -> 1 -> -1 -> 0 over 8 s, looped (for pitch and pan)
    kRiseIn,             ///< bipolar, once: from -1 to 0 over 12 s, fast at first (a glide in from below)
    kFallIn,             ///< bipolar, once: from +1 to 0 over 12 s, fast at first (a glide in from above)
    kSawUp,              ///< 0 -> 1 over 4 s, then back at once, looped
    kFactoryShapeCount   ///< how many factory shapes there are
};

/**
 * @brief A shape from its points.
 * @param points     the breakpoints (at most kEnvPoints are taken)
 * @param loopStart  first point of the loop, -1 = none
 * @param loopEnd    last point of the loop
 * @return           the shape
 */
EnvShape makeShape(std::initializer_list<EnvPoint> points, int loopStart, int loopEnd);

/** @brief The factory bank, in FactoryShape order. @return kFactoryShapeCount shapes */
std::vector<EnvShape> factoryShapes();

/** @brief A factory shape's name. @param i its FactoryShape @return the name ("?" outside the bank) */
const char* factoryShapeName(int i);

} // namespace additiv
