/**
 * @file Scene.cpp
 * @brief The scenes' knobs (Scene.h), Empyrean's.
 * @author Rene Weller
 */
#include "additiv/Scene.h"

namespace additiv
{

const char *const SceneNames[NumScenes] = { "Off", "Forest", "Canyon", "Desert", "Sky", "Ocean" };

namespace
{
/** @brief Forest: a short, very diffuse, dark room; the far plane low and short. */
const SceneSetting Forest[] = {
    { "near_mix", 0.35f },   { "near_decay", 1.2f },  { "near_damp", 0.75f },   { "early_level", 0.2f },
    { "early_size", 5.0f },  { "early_absorb", 0.8f }, { "far_level", 0.3f },   { "far_size", 1.0f },
    { "far_decay", 5.0f },   { "far_damp", 0.8f },     { "far_diffuse", 0.9f }, { "far_predelay", 8.0f },
    { "delay_mix", 0.0f },   { "delay2_mix", 0.0f },
};
/** @brief Canyon: walls at 85 m and 140 m -- 496 and 816 ms there and back, damped, a little feedback, out of the tempo. */
const SceneSetting Canyon[] = {
    { "delay_mix", 0.25f },     { "delay_time_l", 0.496f }, { "delay_time_r", 0.816f }, { "delay_feedback", 0.15f },
    { "delay_cross", 0.0f },    { "delay_damp", 0.55f },    { "delay_absorb", 0.3f },   { "delay_to_far", 0.25f },
    { "delay2_mix", 0.0f },     { "near_mix", 0.1f },       { "early_level", 0.0f },    { "far_level", 0.45f },
    { "far_decay", 4.0f },      { "far_damp", 0.6f },       { "far_predelay", 40.0f },  { "far_diffuse", 0.5f },
};
/** @brief Desert: nearly dry, one far echo. */
const SceneSetting Desert[] = {
    { "near_mix", 0.05f },      { "early_level", 0.0f },    { "far_level", 0.12f },     { "far_decay", 3.0f },
    { "far_damp", 0.7f },       { "far_predelay", 60.0f },  { "delay_mix", 0.0f },      { "delay2_mix", 0.12f },
    { "delay2_time_l", 1.1f },  { "delay2_time_r", 1.3f },  { "delay2_feedback", 0.0f }, { "delay2_damp", 0.8f },
    { "delay2_to_far", 0.0f },
};
/** @brief Sky: a very long, diffuse reverb without pre-delay, mixed quietly. */
const SceneSetting Sky[] = {
    { "far_level", 0.65f }, { "far_size", 3.0f },  { "far_decay", 12.0f },  { "far_damp", 0.35f },
    { "far_predelay", 0.0f }, { "far_diffuse", 1.0f }, { "near_mix", 0.05f }, { "early_level", 0.0f },
    { "delay_mix", 0.0f },  { "delay2_mix", 0.0f },
};
/** @brief Ocean: a long dark reverb. */
const SceneSetting Ocean[] = {
    { "far_level", 0.6f }, { "far_size", 2.5f },  { "far_decay", 10.0f }, { "far_damp", 0.85f },
    { "far_predelay", 5.0f }, { "far_diffuse", 0.9f }, { "near_mix", 0.1f },  { "early_level", 0.0f },
    { "delay_mix", 0.0f }, { "delay2_mix", 0.0f },
};

/**
 * @brief A recipe and its length.
 * @tparam N  how many settings
 * @param[in]  s      the recipe
 * @param[out] count  receives N
 * @return the recipe
 */
template <int N> const SceneSetting *give( const SceneSetting ( &s )[N], int &count )
{
    count = N;
    return s;
}
}   // namespace

const SceneSetting *sceneSettings( int scene, int &count )
{
    count = 0;
    switch ( scene )
    {
    case 1: return give( Forest, count );
    case 2: return give( Canyon, count );
    case 3: return give( Desert, count );
    case 4: return give( Sky, count );
    case 5: return give( Ocean, count );
    default: return nullptr;
    }
}

}   // namespace additiv
