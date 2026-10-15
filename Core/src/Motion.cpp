/**
 * @file Motion.cpp
 * @brief The motions' lane settings (the scenes of Phase 0b, probe scenes).
 */
#include "additiv/Motion.h"

#include "additiv/Shapes.h"

namespace additiv {

void applyMotion(MotionKindE kind, CloudVoice& v, float rate, float depth)
{
    v.m_amp = EnvLane{};
    v.m_pitch = EnvLane{};
    v.m_pan = EnvLane{};
    switch (kind) {
    case MotionKindE::NONE:
        break;
    case MotionKindE::RANDOM:
        v.m_amp.m_amount = 1.0f;
        v.m_amp.m_shape = SWELL;
        v.m_amp.m_shapeSpread = 4;
        v.m_amp.m_shapeBy = MapSourceE::RANDOM;
        v.m_amp.m_chaos = 1.0f;
        break;
    case MotionKindE::WAVE:
        v.m_amp.m_amount = 0.8f;   // a floor under the wave: motion on a carrying sound (12.10.2026)
        v.m_amp.m_shape = SWELL;
        v.m_amp.m_phase = -2.0f;
        v.m_amp.m_phaseBy = MapSourceE::PITCH;
        v.m_amp.m_delay = 3.0f;
        v.m_amp.m_delayBy = MapSourceE::PITCH;
        v.m_amp.m_chaos = 0.03f;
        break;
    case MotionKindE::PENDULUM:
        // After the user's verdict (12.10.2026: "wird zwischendurch eher dünn"): the depth grows
        // with pitch -- a quarter at 110 Hz, a half near 1 kHz, three quarters at 3.5 kHz -- so
        // the bass carries while the top pulses. (By harmonic number it was tried first: then the
        // strong low harmonics got a fifth of the depth and the pendulum all but vanished.)
        v.m_amp.m_amount = 1.0f;
        v.m_amp.m_shape = PULSE;
        v.m_amp.m_rate = 0.5f;
        v.m_amp.m_rateSpread = 6.0f;
        v.m_amp.m_rateBy = MapSourceE::HARMONIC;
        v.m_amp.m_depthTilt = 1.0f;
        v.m_amp.m_depthBy = MapSourceE::PITCH;
        break;
    case MotionKindE::BREATHE:
        v.m_amp.m_amount = 0.8f;
        v.m_amp.m_shape = SWELL;
        v.m_amp.m_rate = 0.7f;
        v.m_amp.m_phase = 1.0f;
        v.m_amp.m_phaseBy = MapSourceE::MEMBER;
        v.m_pan.m_amount = 0.8f;
        v.m_pan.m_shape = WOBBLE;
        v.m_pan.m_rate = 0.5f;
        v.m_pan.m_phase = 1.0f;
        v.m_pan.m_phaseBy = MapSourceE::MEMBER;
        v.m_pitch.m_amount = 6.0f;
        v.m_pitch.m_shape = WOBBLE;
        v.m_pitch.m_chaos = 1.0f;
        break;
    case MotionKindE::GLISSANDO:
        v.m_pitch.m_amount = 1500.0f;
        v.m_pitch.m_shape = RISE_IN;
        v.m_pitch.m_shapeSpread = 2;
        v.m_pitch.m_shapeBy = MapSourceE::RANDOM;
        v.m_pitch.m_depthTilt = 1.0f;
        v.m_pitch.m_depthBy = MapSourceE::RANDOM;
        v.m_pitch.m_chaos = 0.35f;
        break;
    case MotionKindE::AURORA:
        v.m_amp.m_amount = 1.0f;
        v.m_amp.m_landscape = 1;
        break;
    }
    for (EnvLane* lane : { &v.m_amp, &v.m_pitch, &v.m_pan }) {
        lane->m_rate *= rate;
        lane->m_amount *= depth;
    }
}

} // namespace additiv
