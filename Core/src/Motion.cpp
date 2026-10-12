/**
 * @file Motion.cpp
 * @brief The motions' lane settings (the scenes of Phase 0b, probe scenes).
 */
#include "additiv/Motion.h"

#include "additiv/Shapes.h"

namespace additiv {

void applyMotion(MotionKind kind, CloudVoice& v, float rate, float depth)
{
    v.amp = EnvLane{};
    v.pitch = EnvLane{};
    v.pan = EnvLane{};
    switch (kind) {
    case MotionKind::None:
        break;
    case MotionKind::Random:
        v.amp.amount = 1.0f;
        v.amp.shape = kSwell;
        v.amp.shapeSpread = 4;
        v.amp.shapeBy = MapSource::Random;
        v.amp.chaos = 1.0f;
        break;
    case MotionKind::Wave:
        v.amp.amount = 0.8f;   // a floor under the wave: motion on a carrying sound (12.10.2026)
        v.amp.shape = kSwell;
        v.amp.phase = -2.0f;
        v.amp.phaseBy = MapSource::Pitch;
        v.amp.delay = 3.0f;
        v.amp.delayBy = MapSource::Pitch;
        v.amp.chaos = 0.03f;
        break;
    case MotionKind::Pendulum:
        // After the user's verdict (12.10.2026: "wird zwischendurch eher dünn"): the depth grows
        // with pitch -- a quarter at 110 Hz, a half near 1 kHz, three quarters at 3.5 kHz -- so
        // the bass carries while the top pulses. (By harmonic number it was tried first: then the
        // strong low harmonics got a fifth of the depth and the pendulum all but vanished.)
        v.amp.amount = 1.0f;
        v.amp.shape = kPulse;
        v.amp.rate = 0.5f;
        v.amp.rateSpread = 6.0f;
        v.amp.rateBy = MapSource::Harmonic;
        v.amp.depthTilt = 1.0f;
        v.amp.depthBy = MapSource::Pitch;
        break;
    case MotionKind::Breathe:
        v.amp.amount = 0.8f;
        v.amp.shape = kSwell;
        v.amp.rate = 0.7f;
        v.amp.phase = 1.0f;
        v.amp.phaseBy = MapSource::Member;
        v.pan.amount = 0.8f;
        v.pan.shape = kWobble;
        v.pan.rate = 0.5f;
        v.pan.phase = 1.0f;
        v.pan.phaseBy = MapSource::Member;
        v.pitch.amount = 6.0f;
        v.pitch.shape = kWobble;
        v.pitch.chaos = 1.0f;
        break;
    case MotionKind::Glissando:
        v.pitch.amount = 1500.0f;
        v.pitch.shape = kRiseIn;
        v.pitch.shapeSpread = 2;
        v.pitch.shapeBy = MapSource::Random;
        v.pitch.depthTilt = 1.0f;
        v.pitch.depthBy = MapSource::Random;
        v.pitch.chaos = 0.35f;
        break;
    case MotionKind::Aurora:
        v.amp.amount = 1.0f;
        v.amp.landscape = 1;
        break;
    }
    for (EnvLane* lane : { &v.amp, &v.pitch, &v.pan }) {
        lane->rate *= rate;
        lane->amount *= depth;
    }
}

} // namespace additiv
