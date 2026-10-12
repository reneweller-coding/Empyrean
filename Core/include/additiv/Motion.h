/**
 * @file Motion.h
 * @brief The motions: ready settings of a voice's three envelope lanes (amplitude, pitch, pan),
 *        the scenes of Phase 0b made into a choice of the instrument.
 *
 * A motion only sets lanes; the shapes come from the factory bank (Shapes.h) and Aurora reads the
 * landscape the engine uploads. Rate and depth scale whatever the motion set, so one knob slows
 * every lane and one knob fades the motion out to the bare cloud.
 */
#pragma once

#include "additiv/Cloud.h"

namespace additiv {

/** @brief The motions, in the order of kMotionNames (Params.h). */
enum class MotionKind : int {
    None = 0,    ///< the cloud alone
    Random,      ///< every partial for itself: random shape, rate and phase (it averages out: a fine glitter)
    Wave,        ///< a wave climbing through the spectrum; the attack sweeps upwards
    Pendulum,    ///< a pendulum wave across the harmonics, over a bass that stays
    Breathe,     ///< loudness, place and pitch travelling through each cloud
    Glissando,   ///< every partial glides into its note from up to 15 semitones away
    Aurora,      ///< the amplitude lane reads the landscape
};

/** @brief Whether a motion reads the landscape. @param kind the motion @return true for Aurora */
inline bool motionUsesLandscape(MotionKind kind) { return kind == MotionKind::Aurora; }

/**
 * @brief Sets a voice's three lanes to a motion, scaled by rate and depth; the other fields stay.
 * @param kind   the motion
 * @param v      the voice
 * @param rate   times the motion's own speed (0.1 .. 4)
 * @param depth  times the motion's own depth (0 .. 1); 0 leaves the cloud bare
 */
void applyMotion(MotionKind kind, CloudVoice& v, float rate, float depth);

} // namespace additiv
