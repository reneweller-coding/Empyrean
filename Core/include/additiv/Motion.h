/**
 * @file Motion.h
 * @brief The motions: ready settings of a voice's three envelope lanes (amplitude, pitch, pan),
 *        the scenes of Phase 0b made into a choice of the instrument.
 *
 * A motion only sets lanes; the shapes come from the factory bank (Shapes.h) and Aurora reads the
 * landscape the engine uploads. Rate and depth scale whatever the motion set, so one knob slows
 * every lane and one knob fades the motion out to the bare cloud.
 * @author Rene Weller
 */
#pragma once

#include "additiv/Cloud.h"

namespace additiv {

/** @brief The motions, in the order of MotionNames (Params.h). */
enum class MotionKindE : int {
    NONE = 0,    ///< the cloud alone
    RANDOM,      ///< every partial for itself: random shape, rate and phase (it averages out: a fine glitter)
    WAVE,        ///< a wave climbing through the spectrum; the attack sweeps upwards
    PENDULUM,    ///< a pendulum wave across the harmonics, over a bass that stays
    BREATHE,     ///< loudness, place and pitch travelling through each cloud
    GLISSANDO,   ///< every partial glides into its note from up to 15 semitones away
    AURORA,      ///< the amplitude lane reads the landscape
};

/**
 * @brief Whether a motion reads the landscape.
 * @param[in] kind  the motion
 * @return true for Aurora
 */
inline bool motionUsesLandscape(MotionKindE kind) { return kind == MotionKindE::AURORA; }

/**
 * @brief Sets a voice's three lanes to a motion, scaled by rate and depth; the other fields stay.
 * @param[in]     kind   the motion
 * @param[in,out] v      the voice
 * @param[in]     rate   times the motion's own speed (0.1 .. 4)
 * @param[in]     depth  times the motion's own depth (0 .. 1); 0 leaves the cloud bare
 */
void applyMotion(MotionKindE kind, CloudVoice& v, float rate, float depth);

} // namespace additiv
