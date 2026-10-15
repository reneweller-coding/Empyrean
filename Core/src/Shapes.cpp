/**
 * @file Shapes.cpp
 * @brief The factory bank of envelope shapes.
 * @author Rene Weller
 */
#include "additiv/Shapes.h"

namespace additiv
{

EnvShape makeShape( std::initializer_list<EnvPoint> points, int loopStart, int loopEnd )
{
    EnvShape s;
    for ( const EnvPoint &p : points )
    {
        if ( s.m_count == EnvPoints ) break;
        s.m_p[s.m_count++] = p;
    }
    s.m_loopStart = loopStart;
    s.m_loopEnd = loopEnd;
    return s;
}

std::vector<EnvShape> factoryShapes()
{
    std::vector<EnvShape> bank( FACTORY_SHAPE_COUNT );
    // A raised cosine in eight straight pieces: smooth enough that nobody hears the corners.
    bank[SWELL] = makeShape( { { 0, 0 },
                               { 1, 0.146f },
                               { 2, 0.5f },
                               { 3, 0.854f },
                               { 4, 1 },
                               { 5, 0.854f },
                               { 6, 0.5f },
                               { 7, 0.146f },
                               { 8, 0 } },
                             0, 8 );
    bank[PULSE] = makeShape( { { 0, 0 }, { 0.25f, 1, -0.5f }, { 4, 0, -0.6f } }, 0, 2 );
    bank[BLOOM] = makeShape( { { 0, 0 }, { 3, 1, 0.3f }, { 6, 0.4f }, { 9, 1 }, { 13, 0, -0.5f } }, 1, 3 );
    bank[FLICKER] =
        makeShape( { { 0, 0.2f }, { 0.4f, 1 }, { 0.9f, 0.35f }, { 1.5f, 0.8f }, { 2.1f, 0.15f }, { 3, 0.2f } }, 0, 5 );
    bank[WOBBLE] = makeShape( { { 0, 0 }, { 2, 1 }, { 6, -1 }, { 8, 0 } }, 0, 3 );
    bank[RISE_IN] = makeShape( { { 0, -1 }, { 12, 0, -0.4f } }, -1, -1 );
    bank[FALL_IN] = makeShape( { { 0, 1 }, { 12, 0, -0.4f } }, -1, -1 );
    bank[SAW_UP] = makeShape( { { 0, 0 }, { 4, 1 }, { 4.02f, 0 } }, 0, 2 );
    return bank;
}

const char *factoryShapeName( int i )
{
    static const char *const Names[FACTORY_SHAPE_COUNT] = { "Swell",  "Pulse",   "Bloom",   "Flicker",
                                                            "Wobble", "Rise In", "Fall In", "Saw Up" };
    return i >= 0 && i < FACTORY_SHAPE_COUNT ? Names[i] : "?";
}

}   // namespace additiv
