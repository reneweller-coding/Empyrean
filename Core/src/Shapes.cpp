/**
 * @file Shapes.cpp
 * @brief The factory bank of envelope shapes.
 */
#include "additiv/Shapes.h"

namespace additiv {

EnvShape makeShape(std::initializer_list<EnvPoint> points, int loopStart, int loopEnd)
{
    EnvShape s;
    for (const EnvPoint& p : points) {
        if (s.count == kEnvPoints) break;
        s.p[s.count++] = p;
    }
    s.loopStart = loopStart;
    s.loopEnd = loopEnd;
    return s;
}

std::vector<EnvShape> factoryShapes()
{
    std::vector<EnvShape> bank(kFactoryShapeCount);
    // A raised cosine in eight straight pieces: smooth enough that nobody hears the corners.
    bank[kSwell] = makeShape({ { 0, 0 }, { 1, 0.146f }, { 2, 0.5f }, { 3, 0.854f }, { 4, 1 },
                               { 5, 0.854f }, { 6, 0.5f }, { 7, 0.146f }, { 8, 0 } }, 0, 8);
    bank[kPulse] = makeShape({ { 0, 0 }, { 0.25f, 1, -0.5f }, { 4, 0, -0.6f } }, 0, 2);
    bank[kBloom] = makeShape({ { 0, 0 }, { 3, 1, 0.3f }, { 6, 0.4f }, { 9, 1 }, { 13, 0, -0.5f } }, 1, 3);
    bank[kFlicker] = makeShape({ { 0, 0.2f }, { 0.4f, 1 }, { 0.9f, 0.35f }, { 1.5f, 0.8f }, { 2.1f, 0.15f }, { 3, 0.2f } }, 0, 5);
    bank[kWobble] = makeShape({ { 0, 0 }, { 2, 1 }, { 6, -1 }, { 8, 0 } }, 0, 3);
    bank[kRiseIn] = makeShape({ { 0, -1 }, { 12, 0, -0.4f } }, -1, -1);
    bank[kFallIn] = makeShape({ { 0, 1 }, { 12, 0, -0.4f } }, -1, -1);
    bank[kSawUp] = makeShape({ { 0, 0 }, { 4, 1 }, { 4.02f, 0 } }, 0, 2);
    return bank;
}

const char* factoryShapeName(int i)
{
    static const char* const kNames[kFactoryShapeCount] = { "Swell", "Pulse", "Bloom", "Flicker", "Wobble", "Rise In", "Fall In", "Saw Up" };
    return i >= 0 && i < kFactoryShapeCount ? kNames[i] : "?";
}

} // namespace additiv
