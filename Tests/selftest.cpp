/**
 * @file selftest.cpp
 * @brief additiv_selftest: the instrument around the cloud -- the parameter table, the tuning, notes
 *        that begin on their sample, silence, determinism, the conductor's Autoplay, stealing,
 *        release, the real-time path without underruns, the echoes and the modal hall. Needs the
 *        GPU; the cloud's own measurements are probe's selftest.
 */
#include "additiv/Engine.h"
#include "additiv/Journey.h"
#include "additiv/Describe.h"
#include "additiv/Find.h"
#include "additiv/Focus.h"
#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Presets.h"
#include "additiv/Route.h"
#include "additiv/PresetMap.h"
#include "additiv/Spectrum.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace additiv;

namespace {

int g_failures = 0;   ///< checks that failed

/**
 * @brief Prints one check's result and counts a failure.
 * @param good    whether it passed
 * @param what    what was checked
 * @param detail  the measured values
 */
void check(bool good, const char* what, const std::string& detail)
{
    std::printf("  %s  %-52s %s\n", good ? "ok  " : "FAIL", what, detail.c_str());
    if (!good) ++g_failures;
}

/**
 * @brief A string from a printf format.
 * @param fmt   the format
 * @param ...   its values
 * @return      the text
 */
std::string fmt(const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return buf;
}

/**
 * @brief An offline engine at 48 kHz with 256-sample host blocks.
 * @param e      the engine
 * @param brain  whether the conductor plays
 * @param seed   the dice
 * @return       true when it is prepared
 */
bool openOffline(Engine& e, bool brain, std::uint64_t seed = 0x5EEDu)
{
    e.setRealtime(false);
    e.setParam(ParamId::BrainOn, brain ? 1.0f : 0.0f);
    EngineConfig c;
    c.maxHostBlock = 256;
    c.realtimeGpuClass = false;
    c.seed = seed;
    std::string error;
    if (e.prepare(c, &error)) return true;
    std::printf("  FAIL  the engine did not prepare: %s\n", error.c_str());
    ++g_failures;
    return false;
}

/**
 * @brief Renders seconds of an engine into interleaved stereo.
 * @param e        the engine
 * @param seconds  how long
 * @return         the samples
 */
std::vector<float> render(Engine& e, double seconds)
{
    const int n = 256;
    const int blocks = static_cast<int>(seconds * 48000.0 / n);
    std::vector<float> lr;
    lr.reserve(static_cast<size_t>(blocks) * n * 2);
    std::vector<float> L(n), R(n);
    for (int b = 0; b < blocks; ++b) {
        e.process(L.data(), R.data(), n);
        for (int i = 0; i < n; ++i) { lr.push_back(L[i]); lr.push_back(R[i]); }
    }
    return lr;
}

/** @brief RMS of a stretch of interleaved stereo, in dBFS. @param lr samples @param from first frame @param to end frame @return dBFS */
double rmsDb(const std::vector<float>& lr, size_t from, size_t to)
{
    double sq = 0.0;
    to = std::min(to, lr.size() / 2);
    for (size_t i = from * 2; i < to * 2; ++i) sq += static_cast<double>(lr[i]) * lr[i];
    const double rms = std::sqrt(sq / std::max<double>(1.0, static_cast<double>((to - from) * 2)));
    return 20.0 * std::log10(std::max(rms, 1e-12));
}

/**
 * @brief Every effect off: what is heard is the voices alone.
 * @param e  the engine (before or after prepare)
 */
void noEffects(Engine& e)
{
    for (ParamId id : { ParamId::FarLevel, ParamId::EarlyLevel, ParamId::NearMix, ParamId::DelayMix, ParamId::DelayToFar,
                        ParamId::Delay2Mix, ParamId::Delay2ToFar, ParamId::ModalLevel, ParamId::FarUnmask })
        e.setParam(id, 0.0f);
}

/**
 * @brief Every effect off, and a short envelope: what is heard is the voice and then only what a test opens.
 * @param e  the engine (before or after prepare)
 */
void quietEffects(Engine& e)
{
    noEffects(e);
    e.setParam(ParamId::Attack, 0.01f);
    e.setParam(ParamId::Release, 0.05f);
    e.setParam(ParamId::Sustain, 1.0f);
}

/**
 * @brief The level of one frequency in a stretch of the mono sum (a Goertzel filter under a Hann window).
 * @param lr    interleaved stereo
 * @param from  first frame
 * @param to    one past the last frame
 * @param hz    the frequency
 * @return      its level in dB (relative; only differences mean anything)
 */
double toneDb(const std::vector<float>& lr, size_t from, size_t to, double hz)
{
    const double w = 2.0 * 3.14159265358979 * hz / 48000.0, c = 2.0 * std::cos(w);
    double s1 = 0.0, s2 = 0.0;
    const size_t len = to - from;
    for (size_t i = 0; i < len; ++i) {
        const double win = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979 * static_cast<double>(i) / static_cast<double>(len));
        const double x = win * (lr[2 * (from + i)] + lr[2 * (from + i) + 1]);
        const double s = x + c * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    return 10.0 * std::log10(std::max(s1 * s1 + s2 * s2 - c * s1 * s2, 1e-30));
}

/**
 * @brief One frequency in one channel of a stretch, as a complex value (Goertzel under a Hann window).
 * @param lr       interleaved stereo
 * @param from     first frame
 * @param to       one past the last frame
 * @param hz       the frequency
 * @param channel  0 left, 1 right
 * @return         its complex amplitude (relative; the phase and the ratio between channels mean something)
 */
std::complex<double> toneAt(const std::vector<float>& lr, size_t from, size_t to, double hz, int channel)
{
    const double w = 2.0 * 3.14159265358979 * hz / 48000.0;
    std::complex<double> acc(0.0, 0.0);
    const size_t len = to - from;
    for (size_t i = 0; i < len; ++i) {
        const double win = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979 * static_cast<double>(i) / static_cast<double>(len));
        acc += win * static_cast<double>(lr[2 * (from + i) + static_cast<size_t>(channel)]) * std::polar(1.0, -w * static_cast<double>(from + i));
    }
    return acc;
}

/**
 * @brief The frequencies of the sounding voices, lowest first.
 * @param e  the engine
 * @return   Hz of each voice that sounds
 */
std::vector<double> soundingAll(const Engine& e)
{
    std::vector<double> hz;
    for (int i = 0; i < 16; ++i)
        if (e.voiceView(i).hz > 0.0f) hz.push_back(e.voiceView(i).hz);
    std::sort(hz.begin(), hz.end());
    return hz;
}

/**
 * @brief The frequency of the first sounding voice.
 * @param e  the engine
 * @return   Hz, 0 when none sounds
 */
double soundingHz(const Engine& e)
{
    for (int i = 0; i < 16; ++i)
        if (e.voiceView(i).hz > 0.0f) return e.voiceView(i).hz;
    return 0.0;
}

} // namespace

/** @brief Runs every check. @return 0 when all passed */
int main()
{
    // 1. The parameter table: in ParamId order, defaults inside their ranges, keys found.
    {
        bool ordered = true, inRange = true;
        for (int i = 0; i < kNumParams; ++i) {
            const ParamDesc& d = paramTable()[static_cast<size_t>(i)];
            ordered = ordered && static_cast<int>(d.id) == i;
            inRange = inRange && d.def >= d.min && d.def <= d.max;
        }
        const ParamDesc* p = findParam("purity");
        check(ordered && inRange && p && p->id == ParamId::Purity, "parameter table: order, defaults, keys",
              fmt("%d parameters", kNumParams));
    }

    // 2. The tuning (Noctuary's): the root sounds its tempered pitch, the fifth above it 3/2 of that.
    {
        FixedScale s;
        const bool built = makeBuiltinScale(3, s);   // JI 7-limit
        const double root = scaleFrequency(s, 62, 62, 440.0, true);
        const double fifth = scaleFrequency(s, 69, 62, 440.0, true);
        const double et = 440.0 * std::pow(2.0, -7.0 / 12.0);
        check(built && std::fabs(root / et - 1.0) < 1e-12 && std::fabs(fifth / root - 1.5) < 1e-12, "tuning: JI 7-limit root and fifth",
              fmt("D %.4f Hz, A %.4f Hz, ratio %.9f", root, fifth, fifth / root));
    }

    // 3. Without the conductor and without notes the instrument is silent to the last bit.
    {
        Engine e;
        if (!openOffline(e, false)) return 1;
        const std::vector<float> lr = render(e, 2.0);
        const bool silent = std::all_of(lr.begin(), lr.end(), [](float x) { return x == 0.0f; });
        check(silent, "silence: no conductor, no notes", silent ? "every sample 0" : "noise");
    }

    // 4. A played note begins on its sample: stamped offset + latency, the GPU ramp starts there.
    {
        Engine e;
        if (!openOffline(e, false)) return 1;
        e.setParam(ParamId::Attack, 0.01f);
        e.noteOn(69, 1.0f, 37);
        const std::vector<float> lr = render(e, 1.0);
        size_t first = lr.size();
        for (size_t i = 0; i < lr.size(); ++i) if (lr[i] != 0.0f) { first = i / 2; break; }
        const size_t expect = static_cast<size_t>(e.latencySamples()) + 37 + 1;
        check(first == expect, "a note begins on its sample (offset + latency)",
              fmt("first sound at %zu, expected %zu (latency %d)", first, expect, e.latencySamples()));
        check(e.lateEvents() == 0, "offline: no note is late", fmt("%llu late", static_cast<unsigned long long>(e.lateEvents())));
    }

    // 5. The same seed renders the same bits, conductor and all.
    {
        std::vector<float> a, b;
        for (std::vector<float>* dst : { &a, &b }) {
            Engine e;
            if (!openOffline(e, true, 42)) return 1;
            e.setParam(ParamId::AutoRate, 5.0f);
            *dst = render(e, 20.0);
        }
        const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), sizeof(float) * a.size()) == 0;
        check(same, "determinism: the conductor's 20 s twice", same ? "bit for bit" : "differs");
    }

    // 6. Autoplay (Chords): the cluster fills to its density and the piece sounds at a sane level.
    {
        Engine e;
        if (!openOffline(e, true, 7)) return 1;
        e.setParam(ParamId::AutoMode, 1.0f);
        e.setParam(ParamId::AutoRate, 5.0f);
        e.setParam(ParamId::BrainDensity, 5.0f);
        const std::vector<float> lr = render(e, 60.0);
        const bool finite = std::all_of(lr.begin(), lr.end(), [](float x) { return std::isfinite(x) && std::fabs(x) <= 1.0f; });
        const double db = rmsDb(lr, 40 * 48000, 60 * 48000);
        check(finite && e.soundingVoices() >= 5 && db > -45.0 && db < -6.0, "autoplay: the cluster fills, the level is sane",
              fmt("%d voices, %.1f dBFS RMS over 40-60 s, root %d", e.soundingVoices(), db, e.brainRoot()));
    }

    // 7. Twenty notes on sixteen voices: the oldest are stolen (faded over a block), nothing breaks.
    {
        Engine e;
        if (!openOffline(e, false)) return 1;
        e.setParam(ParamId::Attack, 0.01f);
        for (int k = 0; k < 20; ++k) e.noteOn(48 + k, 0.8f, k * 10);
        const std::vector<float> lr = render(e, 1.5);
        const bool finite = std::all_of(lr.begin(), lr.end(), [](float x) { return std::isfinite(x); });
        check(finite && e.soundingVoices() == 16, "stealing: 20 notes, 16 voices", fmt("%d voices sounding", e.soundingVoices()));
    }

    // 8. Release: a note let go falls silent and its voice is free again.
    {
        Engine e;
        if (!openOffline(e, false)) return 1;
        e.setParam(ParamId::Attack, 0.05f);
        e.setParam(ParamId::Release, 0.3f);
        e.noteOn(60, 1.0f, 0);
        render(e, 1.0);
        const int during = e.soundingVoices();
        e.noteOff(60, 0);
        render(e, 2.0);
        check(during == 1 && e.soundingVoices() == 0, "release: the voice falls silent and is freed",
              fmt("%d voice while held, %d after", during, e.soundingVoices()));
    }

    // 9. Real time: a paced callback of 256 samples for 6 s with notes coming and going -- the
    //    worker keeps ahead, nothing underruns, no note is late.
    {
        Engine e;
        e.setRealtime(true);
        EngineConfig c;
        c.maxHostBlock = 256;
        c.realtimeGpuClass = true;
        std::string error;
        if (!e.prepare(c, &error)) { check(false, "real time: prepare", error); return 1; }
        std::vector<float> L(256), R(256);
        LARGE_INTEGER f, t;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t);
        const double period = 256.0 / 48000.0;
        const long long start = t.QuadPart;
        double peak = 0.0;
        for (int b = 0; b < static_cast<int>(6.0 / period); ++b) {
            const long long due = start + static_cast<long long>((b + 1) * period * static_cast<double>(f.QuadPart));
            do { std::this_thread::yield(); QueryPerformanceCounter(&t); } while (t.QuadPart < due);
            if (b % 40 == 0) e.noteOn(50 + (b / 40) % 24, 0.7f, (b * 7) % 256);
            if (b % 40 == 20) e.noteOff(50 + (b / 40) % 24, (b * 3) % 256);
            e.process(L.data(), R.data(), 256);
            for (int i = 0; i < 256; ++i) peak = std::max(peak, static_cast<double>(std::fabs(L[i])));
        }
        check(e.underruns() == 0 && e.lateEvents() == 0 && peak > 0.0, "real time: 6 s paced, notes coming and going",
              fmt("underruns %llu, late %llu, latency %d samples (%.1f ms), GPU %.3f ms/block",
                  static_cast<unsigned long long>(e.underruns()), static_cast<unsigned long long>(e.lateEvents()),
                  e.latencySamples(), e.latencySamples() / 48.0, e.lastGpuMs()));
    }

    // 10. Brain 2: the second conductor plays its own notes in the background.
    {
        Engine e;
        if (!openOffline(e, true, 11)) return 1;
        e.setParam(ParamId::AutoRate, 5.0f);
        e.setParam(ParamId::Brain2On, 1.0f);
        e.setParam(ParamId::Brain2Rate, 4.0f);
        const std::vector<float> lr = render(e, 60.0);
        const bool finite = std::all_of(lr.begin(), lr.end(), [](float x) { return std::isfinite(x); });
        check(finite && e.soundingFrom(1) >= 3 && e.soundingFrom(2) >= 1, "brain 2: a second conductor in the background",
              fmt("conductor %d, second conductor %d voices", e.soundingFrom(1), e.soundingFrom(2)));
    }

    // 11. The Timbre scale (Sethares) is made from the cloud's own spectrum.
    {
        BrainSpectrum sp;
        sp.count = BrainSpectrum::kMax;
        for (int h = 1; h <= sp.count; ++h) { sp.ratio[h - 1] = h * stretchOf(h, 0.02f); sp.amp[h - 1] = harmonicShape(h, 1.2f, 0.0f, 1.0f); }
        FixedScale s;
        const bool built = makeTimbreScale(sp, s);
        Engine e;
        if (!openOffline(e, false)) return 1;
        e.setParam(ParamId::Scale, static_cast<float>(kTimbreScaleIndex));
        e.setParam(ParamId::Attack, 0.05f);
        e.noteOn(60, 0.8f, 0);
        e.noteOn(67, 0.8f, 0);
        const std::vector<float> lr = render(e, 2.0);
        const bool finite = std::all_of(lr.begin(), lr.end(), [](float x) { return std::isfinite(x); });
        check(built && s.count >= 2 && finite && rmsDb(lr, 48000, 96000) > -60.0, "timbre scale: from the stretched spectrum",
              fmt("%d degrees per period, plays at %.1f dBFS", s.count, rmsDb(lr, 48000, 96000)));
    }

    // 12. The landscape's make-up: the same chord in Aurora sits within 4 dB of the bare cloud;
    //     a low pass at 60 Hz, under every fundamental, takes it far down.
    {
        auto chordDb = [](float motion, float filterModel) {
            Engine e;
            if (!openOffline(e, false)) return -200.0;
            e.setParam(ParamId::Motion, motion);
            e.setParam(ParamId::FilterModel, filterModel);
            e.setParam(ParamId::Cutoff, 60.0f);   // below every fundamental: at tilt 1.2 they carry two thirds of the power
            e.setParam(ParamId::KeyTrack, 0.0f);
            e.setParam(ParamId::FilterDrift, 0.0f);
            e.setParam(ParamId::Attack, 0.5f);
            for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
            const std::vector<float> lr = render(e, 20.0);
            return rmsDb(lr, 5 * 48000, 20 * 48000);
        };
        const double bare = chordDb(0.0f, 0.0f);
        const double aurora = chordDb(6.0f, 0.0f);
        const double lp = chordDb(0.0f, 2.0f);
        check(std::fabs(aurora - bare) < 4.0 && lp < bare - 10.0, "landscape make-up, and the filter's reach",
              fmt("bare %.1f, Aurora %.1f, LP 12 at 60 Hz %.1f dBFS", bare, aurora, lp));
    }

    // 13. Everything that moves at once -- breath, doppler, tide, air, presence, a resonant filter
    //     with drift and envelope, glissando -- stays finite and sane.
    {
        Engine e;
        if (!openOffline(e, true, 5)) return 1;
        e.setParam(ParamId::AutoRate, 4.0f);
        e.setParam(ParamId::Breath, 1.0f);
        e.setParam(ParamId::BreathRate, 0.2f);
        e.setParam(ParamId::Doppler, 1.0f);
        e.setParam(ParamId::Tide, 30.0f);
        e.setParam(ParamId::TidePeriod, 1.0f);
        e.setParam(ParamId::Air, 0.7f);
        e.setParam(ParamId::Presence, 6.0f);
        e.setParam(ParamId::PadLowCut, 120.0f);
        e.setParam(ParamId::FilterModel, 8.0f);   // Formant
        e.setParam(ParamId::Resonance, 0.8f);
        e.setParam(ParamId::FilterEnv, 2.0f);
        e.setParam(ParamId::FilterDrift, 1.0f);
        e.setParam(ParamId::Motion, 5.0f);       // Glissando
        e.setParam(ParamId::Inharmonic, 0.5f);
        const std::vector<float> lr = render(e, 40.0);
        const bool finite = std::all_of(lr.begin(), lr.end(), [](float x) { return std::isfinite(x) && std::fabs(x) <= 1.0f; });
        const double db = rmsDb(lr, 20 * 48000, 40 * 48000);
        check(finite && db > -50.0 && db < -6.0, "everything moving at once: finite and sane", fmt("%.1f dBFS RMS over 20-40 s", db));
    }

    // 14. The matrix as text: written by one engine, read by another, the same rows.
    {
        Engine a, b;
        a.setModRow(0, { ModSource::Lfo1, static_cast<int>(ParamId::Cutoff), 0.35f });
        a.setModRow(3, { ModSource::Macro2, static_cast<int>(ParamId::MotionDepth), -0.5f });
        a.setModRow(5, { ModSource::Env1, static_cast<int>(ParamId::Scale), 0.2f });   // a choice: not a target, dropped
        const std::string text = a.modText();
        b.setModText(text);
        const ModRow r0 = b.modRow(0), r1 = b.modRow(1), r2 = b.modRow(2);
        const bool same = r0.source == ModSource::Lfo1 && r0.target == static_cast<int>(ParamId::Cutoff) && std::fabs(r0.depth - 0.35f) < 1e-6f
                       && r1.source == ModSource::Macro2 && r1.target == static_cast<int>(ParamId::MotionDepth) && std::fabs(r1.depth + 0.5f) < 1e-6f
                       && r2.source == ModSource::Off;
        check(same, "modulation: the matrix as text and back", "\"" + text + "\"");
    }

    // 15. No source ever steps: every LFO shape, a hundred cycles at 1 Hz in millisecond steps.
    {
        double worst = 0.0;
        for (int shape = 0; shape < kNumLfoShapes; ++shape) {
            Modulators m;
            m.reset(3);
            ModInputs in;
            for (int i = 0; i < 4; ++i) { in.lfoShape[i] = shape; in.lfoRate[i] = 1.0f; }
            m.advance(0.001f, in);
            float prev = m.value(ModSource::Lfo1);
            for (int s = 0; s < 100000; ++s) {
                m.advance(0.001f, in);
                const float v = m.value(ModSource::Lfo1);
                worst = std::max(worst, static_cast<double>(std::fabs(v - prev)));
                prev = v;
            }
        }
        // The steepest a shape may be: a ramp's return over 7 % of a cycle, 2 / 0.07 per cycle.
        check(worst < 0.03, "modulation: no LFO shape steps", fmt("largest move per millisecond at 1 Hz %.4f (a return over 7 %%: 0.029)", worst));
    }

    // 16. An LFO on the cutoff moves the level; the live value is where the cutoff plays.
    {
        auto swing = [](bool modulated, float* played) {
            Engine e;
            if (!openOffline(e, false)) return 0.0;
            noEffects(e);   // the level's swing as the voices make it: a room or a hall would smooth it
            e.setParam(ParamId::FilterModel, 2.0f);   // LP 12
            e.setParam(ParamId::Cutoff, 400.0f);
            e.setParam(ParamId::KeyTrack, 0.0f);
            e.setParam(ParamId::FilterDrift, 0.0f);
            e.setParam(ParamId::Attack, 0.1f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Lfo1Shape, 0.0f);
            e.setParam(ParamId::Lfo1Rate, 0.5f);
            if (modulated) e.setModRow(0, { ModSource::Lfo1, static_cast<int>(ParamId::Cutoff), 0.4f });
            for (int note : { 45, 52, 57 }) e.noteOn(note, 0.8f, 0);
            const std::vector<float> lr = render(e, 8.0);
            double lo = 1e9, hi = -1e9;
            for (int w = 8; w < 32; ++w) {   // quarter seconds from 2 s on
                const double db = rmsDb(lr, static_cast<size_t>(w) * 12000, static_cast<size_t>(w + 1) * 12000);
                lo = std::min(lo, db);
                hi = std::max(hi, db);
            }
            if (played) *played = e.playedNorm(ParamId::Cutoff);
            return hi - lo;
        };
        float played = 0.0f, still = 0.0f;
        const double moved = swing(true, &played);
        const double steady = swing(false, &still);
        // Without the row the level still breathes: behind a 400 Hz low pass only the fundamentals' clouds
        // remain, and their members beat. The row must move it far more than that.
        check(moved > steady + 6.0 && std::isfinite(played) && played >= 0.0f && played <= 1.0f && std::isnan(still),
              "modulation: an LFO on the cutoff, and its live value",
              fmt("level swings %.1f dB with the row, %.1f dB without; cutoff plays at %.2f", moved, steady, played));
    }

    // 17. The factory bank: 64 groups of 64, unique names, every key known and every value in its
    //     range, every matrix row read, and bank select plus program finding each preset again.
    {
        int bad = 0, outOfRange = 0, rowsLost = 0, perGroup[kNumPresetGroups] = {};
        std::vector<std::string> names;
        for (int i = 0; i < numPresets(); ++i) {
            const Preset& p = preset(i);
            ++perGroup[std::clamp(p.group, 0, kNumPresetGroups - 1)];
            names.push_back(p.name);
            int entries = 0;
            for (const char* s = p.params; *s != 0; ++s) if (*s == '=') ++entries;
            float values[kNumParams];
            std::string mod;
            if (presetValues(i, values, &mod) != entries) ++bad;
            // In range before the clamp: read each value again and compare.
            for (const char* s = p.params; *s != 0;) {
                while (*s == ' ') ++s;
                const char* eq = std::strchr(s, '=');
                if (eq == nullptr) break;
                const std::string key(s, eq);
                const float v = static_cast<float>(std::atof(eq + 1));
                if (const ParamDesc* d = findParam(key.c_str()); d != nullptr && (v < d->min || v > d->max)) ++outOfRange;
                const char* end = std::strchr(eq, ' ');
                s = end != nullptr ? end : eq + std::strlen(eq);
            }
            ModRow rows[kModRows];
            const int want = mod.empty() ? 0 : 1 + static_cast<int>(std::count(mod.begin(), mod.end(), ';'));
            if (modRowsFromText(mod, rows, kModRows) != want) ++rowsLost;
            if (presetAt(p.group, presetIndexInGroup(i)) != i) ++bad;
        }
        std::sort(names.begin(), names.end());
        const bool unique = std::adjacent_find(names.begin(), names.end()) == names.end();
        bool full = true;
        for (int g = 0; g < kNumPresetGroups; ++g) full = full && perGroup[g] == 64;
        check(numPresets() == 6144 && full && unique && bad == 0 && outOfRange == 0 && rowsLost == 0, "presets: the factory bank",
              fmt("%d presets, %s, %d unreadable, %d out of range, %d with lost rows", numPresets(), unique ? "names unique" : "NAMES REPEAT",
                  bad, outOfRange, rowsLost));
    }

    // 18. The family jam, following: the conductor's root comes to the leader's pitch class and stays
    //     there while the conductor plays; the energy's neutral 0.5 changes nothing.
    {
        Engine e;
        if (!openOffline(e, true, 21)) return 1;
        e.setParam(ParamId::AutoRate, 4.0f);
        e.setParam(ParamId::AutoRootMove, 1.0f);   // the conductor wants to wander; the jam holds it
        e.setJamRoot(7);
        e.setJamEnergy(0.5f);
        bool held = true;
        for (int s = 0; s < 30; ++s) {
            render(e, 1.0);
            if (s >= 1 && ((e.brainRoot() % 12) + 12) % 12 != 7) held = false;
        }
        check(held, "family jam: the root held on the leader's", fmt("root %d (pitch class %d) after 30 s", e.brainRoot(), ((e.brainRoot() % 12) + 12) % 12));
    }

    // 19. A preset change fills the chord at speed (requestFill, Noctuary's way): with an event every
    //     two minutes the conductor alone has one or two notes after five seconds, filled it has all five.
    {
        auto conducted = [](bool fill, int& count) {
            Engine e;
            if (!openOffline(e, true, 23)) return false;
            e.setParam(ParamId::AutoMode, 1.0f);
            e.setParam(ParamId::BrainRate, 120.0f);
            e.setParam(ParamId::AutoRate, 300.0f);
            e.setParam(ParamId::BrainDensity, 5.0f);
            render(e, 1.0);
            if (fill) e.requestFill();
            render(e, 4.0);
            count = e.soundingFrom(1);
            return true;
        };
        int slow = 0, filled = 0;
        if (!conducted(false, slow) || !conducted(true, filled)) return 1;
        check(slow <= 2 && filled >= 5, "preset change: the chord fills within seconds", fmt("%d notes alone, %d filled, after 5 s", slow, filled));
    }

    // 20. The modal hall decays at its time: with a decay of 2 s and no damping, 30 dB in a second.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::ModalLevel, 1.0f);
        e.setParam(ParamId::ModalDecay, 2.0f);
        e.setParam(ParamId::ModalDamp, 0.0f);
        e.setParam(ParamId::ModalTune, 0.0f);
        e.setParam(ParamId::ModalDrift, 0.0f);
        if (!openOffline(e, false, 31)) return 1;
        e.noteOn(62, 0.8f, 0);
        render(e, 1.0);
        e.noteOff(62, 0);
        const std::vector<float> lr = render(e, 2.0);
        const double a = rmsDb(lr, 14400, 24000), b = rmsDb(lr, 62400, 72000);
        check(a - b > 22.0 && a - b < 38.0, "modal hall: it dies at its decay time", fmt("%.1f dB in 1 s at 2 s to -60 dB (30 expected); tail at %.1f dBFS", a - b, a));
    }

    // 21. Tuned, it rings on the scale: 15 cents beside the voice's fourth harmonic the tail is
    //     far quieter than on it -- spread over the range it is not.
    {
        double ratio[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::ModalLevel, 1.0f);
            e.setParam(ParamId::ModalDecay, 4.0f);
            e.setParam(ParamId::ModalDamp, 0.0f);
            e.setParam(ParamId::ModalTune, static_cast<float>(k));
            e.setParam(ParamId::ModalSpread, 0.0f);
            e.setParam(ParamId::ModalDrift, 0.0f);
            if (!openOffline(e, false, 37)) return 1;
            e.noteOn(62, 0.8f, 0);
            render(e, 1.5);
            const double f4 = 4.0 * soundingHz(e);
            e.noteOff(62, 0);
            const std::vector<float> lr = render(e, 1.5);
            ratio[k] = toneDb(lr, 9600, 57600, f4 * std::pow(2.0, 15.0 / 1200.0)) - toneDb(lr, 9600, 57600, f4);
        }
        check(ratio[1] < ratio[0] - 10.0, "modal hall: tuned, it rings on the scale",
              fmt("15 ct beside the 4th harmonic: %.1f dB spread, %.1f dB tuned", ratio[0], ratio[1]));
    }

    // 22. Frozen, it holds: the level three seconds on within 1.5 dB of where it was.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::ModalLevel, 1.0f);
        e.setParam(ParamId::ModalDecay, 3.0f);
        e.setParam(ParamId::ModalDrift, 0.0f);
        if (!openOffline(e, false, 41)) return 1;
        e.noteOn(62, 0.8f, 0);
        render(e, 1.0);
        e.noteOff(62, 0);
        e.setParam(ParamId::ModalFreeze, 1.0f);
        const std::vector<float> lr = render(e, 4.0);
        const double a = rmsDb(lr, 24000, 48000), b = rmsDb(lr, 168000, 192000);
        check(std::fabs(a - b) < 1.5 && a > -80.0, "modal hall: frozen, it holds", fmt("%.1f dBFS after 0.5 s, %.1f dBFS after 3.5 s", a, b));
    }

    // 23. The echo arrives at its time: a tenth of a second of a note, then half a second of
    //     delay; the window around the echo is far louder than the gap before it.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Attack, 0.005f);
        e.setParam(ParamId::Release, 0.03f);
        e.setParam(ParamId::DelayMix, 1.0f);
        e.setParam(ParamId::DelayTimeL, 0.5f);
        e.setParam(ParamId::DelayTimeR, 0.5f);
        e.setParam(ParamId::DelayFeedback, 0.0f);
        if (!openOffline(e, false, 43)) return 1;
        e.noteOn(62, 0.8f, 0);
        const std::vector<float> on = render(e, 0.1);
        e.noteOff(62, 0);
        const std::vector<float> lr = render(e, 1.0);
        const double gap = rmsDb(lr, 9600, 16800), echo = rmsDb(lr, 19200, 26400);
        check(echo > gap + 10.0, "echo: it arrives at its time", fmt("%.1f dBFS in the gap, %.1f dBFS half a second after the note", gap, echo));
    }

    // 24. Everything at once, with the conductor: finite, inside the rails, at a sane level.
    {
        Engine e;
        for (const auto& [id, v] : std::initializer_list<std::pair<ParamId, float>>{
                 { ParamId::DelayMix, 0.4f }, { ParamId::Delay2Mix, 0.3f }, { ParamId::DelayFeedback, 0.7f }, { ParamId::EarlyLevel, 0.5f },
                 { ParamId::NearMix, 0.4f }, { ParamId::FarDiffuse, 0.8f }, { ParamId::FarUnmask, 0.6f }, { ParamId::FarMode, 3.0f },
                 { ParamId::ModalLevel, 0.6f }, { ParamId::ModalTune, 0.7f }, { ParamId::ModalDecay, 30.0f }, { ParamId::AutoRate, 6.0f },
                 { ParamId::ModalBreath, 0.7f }, { ParamId::FieldMode, 2.0f }, { ParamId::FieldStrength, 0.8f }, { ParamId::Dispersion, 0.4f } })
            e.setParam(id, v);
        if (!openOffline(e, true, 47)) return 1;
        const std::vector<float> lr = render(e, 30.0);
        const bool finite = std::all_of(lr.begin(), lr.end(), [](float x) { return std::isfinite(x) && std::fabs(x) <= 1.0f; });
        const double db = rmsDb(lr, 10 * 48000, 30 * 48000);
        check(finite && db > -45.0 && db < -6.0, "effects: everything at once is finite and sane", fmt("%.1f dBFS RMS over 10-30 s, GPU %.3f ms/block", db, e.lastGpuMs()));
    }

    // 25. The Scale field gathers the cloud onto the key: 20 cents beside the fourth harmonic (a
    //     pitch of the scale) a wide cloud has energy, gathered it has far less -- at the same level.
    {
        double ratio[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.5f);
            e.setParam(ParamId::CloudWidth, 60.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::FieldMode, k == 1 ? 1.0f : 0.0f);
            e.setParam(ParamId::FieldStrength, 0.8f);
            e.setParam(ParamId::FieldSpring, 0.0f);
            if (!openOffline(e, false, 53)) return 1;
            e.noteOn(62, 0.8f, 0);
            const std::vector<float> lr = render(e, 4.0);
            const double f4 = 4.0 * soundingHz(e);
            ratio[k] = toneDb(lr, 96000, 192000, f4 * std::pow(2.0, 20.0 / 1200.0)) - toneDb(lr, 96000, 192000, f4);
            level[k] = rmsDb(lr, 96000, 192000);
        }
        check(ratio[1] < ratio[0] - 8.0 && std::fabs(level[1] - level[0]) < 1.5, "field: Scale gathers the cloud onto the key",
              fmt("20 ct beside the 4th harmonic: %.1f dB free, %.1f dB gathered; level %.1f / %.1f dBFS", ratio[0], ratio[1], level[0], level[1]));
    }

    // 26. Swarm and Scatter: the level stays where it was, and the same seed plays the same (the
    //     attractors' Lorenz systems run on the host, from the seed).
    {
        auto play = [](int mode) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.5f);
            e.setParam(ParamId::FieldMode, static_cast<float>(mode));
            e.setParam(ParamId::FieldStrength, 0.9f);
            e.setParam(ParamId::FieldRadius, 200.0f);
            e.setParam(ParamId::FieldRate, 0.5f);
            if (!openOffline(e, false, 59)) return std::vector<float>();
            for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
            return render(e, 6.0);
        };
        const std::vector<float> off = play(0), swarm = play(2), again = play(2), scatter = play(3);
        if (off.empty() || swarm.empty() || again.empty() || scatter.empty()) return 1;
        const double l0 = rmsDb(off, 96000, 288000), l1 = rmsDb(swarm, 96000, 288000), l2 = rmsDb(scatter, 96000, 288000);
        check(std::fabs(l1 - l0) < 2.0 && std::fabs(l2 - l0) < 2.0 && swarm == again && swarm != off,
              "field: Swarm and Scatter keep the level, the seed decides",
              fmt("%.1f dBFS free, %.1f swarm, %.1f scatter; the same seed %s", l0, l1, l2, swarm == again ? "identical" : "DIFFERENT"));
    }

    // 27. Dispersion: with a second of it the high partials arrive first and the low ones after --
    //     the eighth harmonic sounds while the fundamental is still to come, and the other way round.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Attack, 0.005f);
        e.setParam(ParamId::Release, 0.03f);
        e.setParam(ParamId::Dispersion, 1.0f);
        if (!openOffline(e, false, 61)) return 1;
        e.noteOn(62, 0.8f, 0);
        std::vector<float> lr = render(e, 0.3);
        const double f1 = soundingHz(e), f8 = 8.0 * f1;
        e.noteOff(62, 0);
        const std::vector<float> rest = render(e, 1.2);
        lr.insert(lr.end(), rest.begin(), rest.end());
        // Late by the share of the range above them: the 8th harmonic by about 0.31 s, the fundamental by 0.61 s.
        const double a1 = toneDb(lr, 17280, 26880, f1), b1 = toneDb(lr, 31680, 41280, f1);
        const double a8 = toneDb(lr, 17280, 26880, f8), b8 = toneDb(lr, 31680, 41280, f8);
        check(b1 > a1 + 15.0 && a8 > b8 + 15.0, "dispersion: the highs arrive first, the lows after",
              fmt("fundamental %.1f -> %.1f dB, 8th harmonic %.1f -> %.1f dB (0.36-0.56 s -> 0.66-0.86 s)", a1, b1, a8, b8));
    }

    // 28. Binaural: in front the ears hear the same; turned to the right (an orbit of a quarter turn
    //     a second, a second on) the right ear hears the highs louder and everything sooner.
    {
        double ild[2] = { 0.0, 0.0 }, itdMs[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Stereo, 0.0f);
            e.setParam(ParamId::Binaural, 1.0f);
            e.setParam(ParamId::BinauralHeight, 0.0f);
            e.setParam(ParamId::BinauralOrbit, k == 1 ? 0.25f : 0.0f);
            if (!openOffline(e, false, 67)) return 1;
            e.noteOn(62, 0.8f, 0);
            const std::vector<float> lr = render(e, 1.2);
            const double f1 = soundingHz(e), f8 = 8.0 * f1;
            const size_t a = 43200, b = 52800;   // 0.9 .. 1.1 s
            ild[k] = 20.0 * std::log10(std::abs(toneAt(lr, a, b, f8, 1)) / std::abs(toneAt(lr, a, b, f8, 0)));
            itdMs[k] = std::arg(toneAt(lr, a, b, f1, 1) * std::conj(toneAt(lr, a, b, f1, 0))) / (2.0 * 3.14159265358979 * f1) * 1000.0;
        }
        check(std::fabs(ild[0]) < 1.0 && std::fabs(itdMs[0]) < 0.05 && ild[1] > 6.0 && itdMs[1] > 0.3 && itdMs[1] < 0.9,
              "binaural: in front alike, to the right louder and sooner",
              fmt("front: %.1f dB, %.2f ms; right: the 8th harmonic %.1f dB louder, the fundamental %.2f ms sooner", ild[0], itdMs[0], ild[1], itdMs[1]));
    }

    // 29. The memory: a released chord's ghosts rise over their bloom and are heard long after the
    //     chord; at the moment of release they add almost nothing.
    {
        double early[2] = { 0.0, 0.0 }, late[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.5f);
            e.setParam(ParamId::Release, 2.0f);
            e.setParam(ParamId::MemoryLevel, k == 1 ? 0.8f : 0.0f);
            e.setParam(ParamId::MemoryBloom, 4.0f);
            e.setParam(ParamId::MemorySink, 0.0f);
            if (!openOffline(e, false, 71)) return 1;
            for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
            render(e, 4.0);
            for (int note : { 50, 57, 62, 66 }) e.noteOff(note, 0);
            const std::vector<float> lr = render(e, 12.0);
            early[k] = rmsDb(lr, 0, 24000);
            late[k] = rmsDb(lr, 9 * 48000, 11 * 48000);
        }
        check(early[1] - early[0] < 3.0 && late[1] - late[0] > 10.0, "memory: the ghosts bloom after the chord and linger",
              fmt("first 0.5 s %.1f -> %.1f dBFS, 9-11 s after %.1f -> %.1f dBFS", early[0], early[1], late[0], late[1]));
    }

    // 30. Self-tuning: a C major triad in equal temperament settles towards just intervals -- the
    //     third from 400 towards 386 cents, the fifth from 700 towards 702.
    {
        double third[2] = { 0.0, 0.0 }, fifth[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.3f);
            e.setParam(ParamId::Scale, 0.0f);   // 12-TET
            e.setParam(ParamId::TuneGravity, k == 1 ? 1.0f : 0.0f);
            e.setParam(ParamId::TuneAnchor, 0.02f);
            if (!openOffline(e, false, 73)) return 1;
            for (int note : { 60, 64, 67 }) e.noteOn(note, 0.8f, 0);
            render(e, 6.0);
            const std::vector<double> hz = soundingAll(e);
            if (hz.size() != 3) { check(false, "self-tune: three voices", fmt("%d sound", static_cast<int>(hz.size()))); break; }
            third[k] = 1200.0 * std::log2(hz[1] / hz[0]);
            fifth[k] = 1200.0 * std::log2(hz[2] / hz[0]);
        }
        check(std::fabs(third[0] - 400.0) < 0.5 && std::fabs(third[1] - 386.3) < 5.0 && std::fabs(fifth[1] - 702.0) < 3.0,
              "self-tune: an equal-tempered triad finds just intervals",
              fmt("third %.1f -> %.1f ct (386.3), fifth %.1f -> %.1f ct (702.0)", third[0], third[1], fifth[0], fifth[1]));
    }

    // 31. The spectral echo: its echo comes back at its time; tilted, the low band's echo comes
    //     back before the high band's.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Attack, 0.005f);
        e.setParam(ParamId::Release, 0.03f);
        e.setParam(ParamId::SpecMix, 1.0f);
        e.setParam(ParamId::SpecToFar, 0.0f);
        e.setParam(ParamId::SpecTime, 0.6f);
        e.setParam(ParamId::SpecTilt, 1.0f);
        e.setParam(ParamId::SpecScatter, 0.0f);
        e.setParam(ParamId::SpecFeedback, 0.0f);
        if (!openOffline(e, false, 79)) return 1;
        e.noteOn(62, 0.8f, 0);
        std::vector<float> lr = render(e, 0.1);
        const double f1 = soundingHz(e), f8 = 8.0 * f1;
        e.noteOff(62, 0);
        const std::vector<float> rest = render(e, 1.4);
        lr.insert(lr.end(), rest.begin(), rest.end());
        // Tilted: the fundamental's band back after about 0.52 s, the 8th harmonic's after 0.77 s.
        const double gap = rmsDb(lr, 12000, 20000);
        const double a1 = toneDb(lr, 24480, 31680, f1), b1 = toneDb(lr, 37440, 45120, f1);
        const double a8 = toneDb(lr, 24480, 31680, f8), b8 = toneDb(lr, 37440, 45120, f8);
        check(a1 > b1 + 10.0 && b8 > a8 + 10.0 && rmsDb(lr, 24480, 45120) > gap + 10.0, "spectral echo: tilted, the lows come back first",
              fmt("fundamental %.1f -> %.1f dB, 8th harmonic %.1f -> %.1f dB (0.51-0.66 s -> 0.78-0.94 s); gap %.1f dBFS", a1, b1, a8, b8, gap));
    }

    // 32. The painted hall keeps its level (within 3 dB of the unpainted one) while the landscape moves it.
    {
        double level[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.5f);
            e.setParam(ParamId::ModalLevel, 1.0f);
            e.setParam(ParamId::ModalPaint, static_cast<float>(k));
            e.setParam(ParamId::Landscape, 0.0f);   // Aurora
            if (!openOffline(e, false, 83)) return 1;
            for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
            const std::vector<float> lr = render(e, 8.0);
            level[k] = rmsDb(lr, 4 * 48000, 8 * 48000);
        }
        check(std::fabs(level[1] - level[0]) < 3.0, "modal hall: painted, it keeps its level", fmt("%.1f dBFS plain, %.1f painted", level[0], level[1]));
    }

    // 33. The picture's snapshot: the partials of a sounding note come down from the GPU, at its
    //     harmonics, with their amplitudes; the silent voices' points are empty.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Attack, 0.05f);
        if (!openOffline(e, false, 89)) return 1;
        e.noteOn(62, 0.8f, 0);
        render(e, 1.0);
        std::vector<float> dots;
        int per = 0;
        const bool got = e.partialSnapshot(dots, per);
        const double f0 = soundingHz(e);
        int voice = -1;
        for (int i = 0; i < 16; ++i)
            if (e.voiceView(i).hz > 0.0f) voice = i;
        int onHarmonic = 0, loud = 0, elsewhere = 0;
        if (got && per > 0 && voice >= 0) {
            for (int v = 0; v < static_cast<int>(dots.size() / 2) / per; ++v)
                for (int i = 0; i < per; ++i) {
                    const size_t at = 2 * (static_cast<size_t>(v) * static_cast<size_t>(per) + static_cast<size_t>(i));
                    const float hz = dots[at], a = dots[at + 1];
                    if (v != voice) { if (a > 0.0f) ++elsewhere; continue; }
                    if (a <= 0.0f) continue;
                    ++loud;
                    const double h = hz / f0;   // a harmonic's cloud lies within 100 cents of it (the air band is off)
                    if (std::fabs(1200.0 * std::log2(h / std::max(1.0, std::round(h)))) < 100.0) ++onHarmonic;
                }
        }
        check(got && loud > per / 2 && onHarmonic > loud * 9 / 10 && elsewhere == 0, "picture: the partials come down from the GPU",
              fmt("%d of %d points of the sounding voice heard, %d at its harmonics; %d elsewhere", loud, per, onHarmonic, elsewhere));
    }

    // 34. Noctuary's sources move and keep to their ranges: each routed to the brightness, its live
    //     value sampled every half second over 40 s of the conductor.
    {
        const ModSource kSources[] = { ModSource::Lenia1, ModSource::LorenzX, ModSource::RosslerZ, ModSource::Kura1,
                                       ModSource::RootAge, ModSource::Note, ModSource::RandomNote };
        std::string still;
        for (ModSource s : kSources) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::AutoRate, 5.0f);
            e.setParam(ParamId::ChaosPeriod, 10.0f);
            e.setParam(ParamId::Brightness, 0.5f);
            e.setModRow(0, { s, static_cast<int>(ParamId::Brightness), 0.5f });
            if (!openOffline(e, true, 97)) return 1;
            float lo = 1e9f, hi = -1e9f;
            bool sane = true;
            for (int k = 0; k < 80; ++k) {
                render(e, 0.5);
                const float v = e.playedNorm(ParamId::Brightness);
                if (!std::isfinite(v) || v < -0.001f || v > 1.001f) sane = false;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            // Root Age climbs over ten minutes: in forty seconds a fifteenth of the way, at half depth 0.033.
            const float need = s == ModSource::RootAge ? 0.02f : 0.05f;
            if (!sane || hi - lo < need) still += std::string(kModSourceNames[static_cast<int>(s)]) + fmt(" (%.3f) ", hi - lo);
        }
        check(still.empty(), "modulation: Lenia, chaos, Kuramoto and the piece's sources move", still.empty() ? std::string("all seven swing, all inside the range") : still);
    }

    // 35. The Beat: in equal temperament a fifth beats and the source turns; in just intonation it stands.
    {
        float swing[2] = { 0.0f, 0.0f };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.1f);
            e.setParam(ParamId::Scale, k == 0 ? 0.0f : 3.0f);   // 12-TET, JI 7-limit
            e.setParam(ParamId::RootNote, 0.0f);                 // C: C and G a pure fifth in the just scale
            e.setParam(ParamId::Purity, 1.0f);
            e.setParam(ParamId::Brightness, 0.5f);
            e.setModRow(0, { ModSource::Beat, static_cast<int>(ParamId::Brightness), 0.5f });
            if (!openOffline(e, false, 101)) return 1;
            e.noteOn(60, 0.8f, 0);
            e.noteOn(67, 0.8f, 0);
            render(e, 6.0);
            float lo = 1e9f, hi = -1e9f;
            for (int i = 0; i < 40; ++i) {
                render(e, 0.1);
                lo = std::min(lo, e.playedNorm(ParamId::Brightness));
                hi = std::max(hi, e.playedNorm(ParamId::Brightness));
            }
            swing[k] = hi - lo;
        }
        check(swing[0] > 0.2f && swing[1] < 0.02f, "modulation: the Beat turns when the fifth beats, stands when it is pure",
              fmt("swing %.3f in 12-TET, %.3f in just intonation", swing[0], swing[1]));
    }

    // 36. The via source scales the depth, and survives the text form.
    {
        float swing[2] = { 0.0f, 0.0f };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Lfo1Rate, 1.0f);
            e.setParam(ParamId::Brightness, 0.5f);
            e.setParam(ParamId::Macro1, static_cast<float>(k));
            ModRow r{ ModSource::Lfo1, static_cast<int>(ParamId::Brightness), 0.5f };
            r.via = ModSource::Macro1;
            e.setModRow(0, r);
            if (!openOffline(e, false, 103)) return 1;
            float lo = 1e9f, hi = -1e9f;
            for (int i = 0; i < 30; ++i) {
                render(e, 0.1);
                lo = std::min(lo, e.playedNorm(ParamId::Brightness));
                hi = std::max(hi, e.playedNorm(ParamId::Brightness));
            }
            swing[k] = hi - lo;
        }
        ModRow rows[2];
        modRowsFromText("lfo1>cutoff:0.35:lenia2;kura3>drift:-0.2", rows, 2);
        const std::string back = modRowsToText(rows, 2);
        check(swing[0] < 0.01f && swing[1] > 0.5f && rows[0].via == ModSource::Lenia2 && rows[1].via == ModSource::Off
                  && back == "lfo1>cutoff:0.35:lenia2;kura3>drift:-0.2",
              "modulation: a via source scales the depth", fmt("swing %.3f at Macro 1 = 0, %.3f at 1; text '%s'", swing[0], swing[1], back.c_str()));
    }

    // 37. The living landscapes: Lenia and the Turing field live (the field at 6 s is not the field
    //     at 3 s), keep the level within 2 dB of no field, and the same seed plays the same.
    {
        std::string report;
        bool good = true;
        double level0 = 0.0;
        for (int mode = 0; mode < 3; ++mode) {
            auto play = [&](std::vector<float>& f3, std::vector<float>& f6) {
                Engine e;
                quietEffects(e);
                e.setParam(ParamId::Attack, 0.5f);
                e.setParam(ParamId::LifeMode, static_cast<float>(mode));
                e.setParam(ParamId::LifeDepth, 0.8f);
                e.setParam(ParamId::LifeRate, 0.5f);
                if (!openOffline(e, false, 107)) return std::vector<float>();
                for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
                std::vector<float> lr = render(e, 3.0);
                e.lifeSnapshot(f3);
                const std::vector<float> rest = render(e, 3.0);
                e.lifeSnapshot(f6);
                lr.insert(lr.end(), rest.begin(), rest.end());
                return lr;
            };
            std::vector<float> a3, a6, b3, b6;
            const std::vector<float> a = play(a3, a6), b = play(b3, b6);
            if (a.empty()) return 1;
            const double level = rmsDb(a, 3 * 48000, 6 * 48000);
            if (mode == 0) { level0 = level; continue; }
            double moved = 0.0;
            for (size_t i = 0; i < a3.size() && i < a6.size(); ++i) moved += std::fabs(a6[i] - a3[i]);
            moved /= std::max<size_t>(1, a3.size());
            const bool ok = a3.size() == static_cast<size_t>(kLifeW * kLifeH) && moved > 0.01 && std::fabs(level - level0) < 2.0 && a == b;
            good = good && ok;
            report += fmt("%s: moved %.3f a cell, %.1f dBFS (free %.1f)%s; ", kLifeModeNames[mode], moved, level, level0, a == b ? "" : ", NOT identical");
        }
        check(good, "life: Lenia and Turing live, keep the level, repeat", report);
    }

    // 38. Partial sync: coupled, the members of a harmonic fall into step and the harmonic pulses as
    //     one -- its level in tenths of a second swings far more than uncoupled -- at the same level.
    {
        double sd[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::SyncDepth, 1.0f);
            e.setParam(ParamId::SyncRate, 1.0f);
            e.setParam(ParamId::SyncCoupling, static_cast<float>(k));
            if (!openOffline(e, false, 109)) return 1;
            e.noteOn(62, 0.8f, 0);
            const std::vector<float> lr = render(e, 12.0);
            const double f8 = 8.0 * soundingHz(e);
            double s = 0.0, s2 = 0.0;
            const int windows = 60;
            for (int w = 0; w < windows; ++w) {
                const size_t a = static_cast<size_t>(6 * 48000 + w * 4800);
                const double db = toneDb(lr, a, a + 4800, f8);
                s += db;
                s2 += db * db;
            }
            sd[k] = std::sqrt(std::max(0.0, s2 / windows - (s / windows) * (s / windows)));
            level[k] = rmsDb(lr, 6 * 48000, 12 * 48000);
        }
        check(sd[1] > sd[0] + 2.0 && std::fabs(level[1] - level[0]) < 1.5, "sync: coupled, a harmonic's members pulse as one",
              fmt("8th harmonic's swing %.1f dB free, %.1f dB coupled; level %.1f / %.1f dBFS", sd[0], sd[1], level[0], level[1]));
    }

    // 39. What it hears of itself: louder with a note than in silence, brighter for a bright spectrum,
    //     more change at an attack than in a held note, rougher for a minor second than a pure fifth.
    {
        // A source's value through a row onto the brightness (depth 1 from 0): the live value is the source.
        auto heard = [](ModSource s, const std::vector<int>& notes, float tilt, double seconds, float attack, float scale = 3.0f) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, attack);
            e.setParam(ParamId::Tilt, tilt);
            e.setParam(ParamId::Scale, scale);     // JI 7-limit on C (C and G pure); 12-TET where a semitone must be a semitone
            e.setParam(ParamId::RootNote, 0.0f);
            e.setParam(ParamId::Brightness, 0.0f);
            e.setModRow(0, { s, static_cast<int>(ParamId::Brightness), 1.0f });
            if (!openOffline(e, false, 113)) return -1.0f;
            for (int note : notes) e.noteOn(note, 0.8f, 0);
            render(e, seconds);
            return e.playedNorm(ParamId::Brightness);
        };
        const float quiet = heard(ModSource::Loudness, {}, 1.2f, 2.0, 0.05f), loud = heard(ModSource::Loudness, { 60, 64, 67 }, 1.2f, 2.0, 0.05f);
        const float dark = heard(ModSource::Centroid, { 48 }, 3.0f, 2.0, 0.05f), bright = heard(ModSource::Centroid, { 48 }, 0.3f, 2.0, 0.05f);
        // Change right after the attack (the attack's rise lasts a frame or two, the source follows over a fifth of a second).
        const float onset = heard(ModSource::Flux, { 60 }, 1.2f, 0.06, 0.005f), held = heard(ModSource::Flux, { 60 }, 1.2f, 4.0, 0.005f);
        // In 12-TET: the just scale has no C sharp of its own, so C and C sharp on the keys would be a unison.
        const float pure = heard(ModSource::Roughness, { 60, 67 }, 1.2f, 3.0, 0.05f, 0.0f), rough = heard(ModSource::Roughness, { 60, 61 }, 1.2f, 3.0, 0.05f, 0.0f);
        check(loud > quiet + 0.3f && bright > dark + 0.1f && onset > held + 0.03f && rough > pure + 0.2f, "listening: loudness, brightness, change, roughness",
              fmt("loudness %.2f -> %.2f, brightness %.2f -> %.2f, change %.2f at the attack / %.2f held, roughness %.2f fifth / %.2f second",
                  quiet, loud, dark, bright, onset, held, pure, rough));
    }

    // 40. The flock: a cloud's members move together -- the centre of the 4th harmonic's cloud wanders
    //     over ten seconds by several cents -- but keep their spread (no collapse), at the same level.
    {
        double moved[2] = { 0.0, 0.0 }, spread[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::FieldMode, k == 1 ? 4.0f : 0.0f);   // Off, Flock
            e.setParam(ParamId::FieldStrength, 0.8f);
            e.setParam(ParamId::FieldRate, 0.3f);
            if (!openOffline(e, false, 127)) return 1;
            e.noteOn(62, 0.8f, 0);
            // The 4th harmonic's points in the snapshot: their mean and spread in cents from it.
            auto cloud4 = [&](double& mean, double& sd) {
                std::vector<float> dots;
                int per = 0;
                e.partialSnapshot(dots, per);
                const double f4 = 4.0 * soundingHz(e);
                int voice = -1;
                for (int i = 0; i < 16; ++i) if (e.voiceView(i).hz > 0.0f) voice = i;
                double s = 0.0, s2 = 0.0;
                int count = 0;
                for (int i = 0; voice >= 0 && i < per; ++i) {
                    const size_t at = 2 * (static_cast<size_t>(voice) * per + i);
                    if (dots[at + 1] <= 0.0f) continue;
                    const double c = 1200.0 * std::log2(dots[at] / f4);
                    if (std::fabs(c) > 300.0) continue;   // another harmonic's
                    s += c;
                    s2 += c * c;
                    ++count;
                }
                mean = count ? s / count : 0.0;
                sd = count ? std::sqrt(std::max(0.0, s2 / count - mean * mean)) : 0.0;
            };
            // The centre twenty times over ten seconds: its span is how far the cloud has wandered.
            std::vector<float> lr = render(e, 2.0);
            double lo = 1e9, hi = -1e9, sd = 0.0;
            for (int w = 0; w < 20; ++w) {
                const std::vector<float> part = render(e, 0.5);
                lr.insert(lr.end(), part.begin(), part.end());
                double m = 0.0;
                cloud4(m, sd);
                lo = std::min(lo, m);
                hi = std::max(hi, m);
            }
            moved[k] = hi - lo;
            spread[k] = sd;
            level[k] = rmsDb(lr, 2 * 48000, 12 * 48000);
        }
        check(moved[1] > 5.0 && moved[1] > moved[0] + 4.0 && spread[1] > 0.5 * spread[0] && std::fabs(level[1] - level[0]) < 1.5,
              "field: Flock moves the clouds together, keeping them wide",
              fmt("4th harmonic's centre wandered over %.1f ct (still %.1f), spread %.1f ct (still %.1f), level %.1f / %.1f dBFS",
                  moved[1], moved[0], spread[1], spread[0], level[1], level[0]));
    }

    // 41. The ripple: notes struck one after another send waves along the spectrum, and a high
    //     harmonic's loudness rides them -- it swings far more than without -- at the same level.
    {
        double sd[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::CloudWidth, 0.0f);   // one steady sine per harmonic: whatever moves is the ripple
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::RippleDepth, static_cast<float>(k));
            e.setParam(ParamId::RippleSpeed, 1.5f);
            e.setParam(ParamId::RippleDecay, 8.0f);
            if (!openOffline(e, false, 131)) return 1;
            e.noteOn(50, 0.8f, 0);
            std::vector<float> lr = render(e, 1.0);
            const double f4 = 4.0 * soundingHz(e);
            for (int note : { 57, 62, 66 }) {   // more strikes, a second apart
                e.noteOn(note, 0.8f, 0);
                const std::vector<float> part = render(e, 1.0);
                lr.insert(lr.end(), part.begin(), part.end());
            }
            const std::vector<float> rest = render(e, 4.0);
            lr.insert(lr.end(), rest.begin(), rest.end());
            // Quarter seconds: a passing wave takes about that long; the cloud's own beating, shorter, averages out.
            double s = 0.0, s2 = 0.0;
            const int windows = 24;
            for (int w = 0; w < windows; ++w) {
                const size_t a = static_cast<size_t>(1 * 48000 + w * 12000);
                const double db = toneDb(lr, a, a + 12000, f4);
                s += db;
                s2 += db * db;
            }
            sd[k] = std::sqrt(std::max(0.0, s2 / windows - (s / windows) * (s / windows)));
            level[k] = rmsDb(lr, 4 * 48000, 8 * 48000);
        }
        check(sd[1] > sd[0] + 1.5 && std::fabs(level[1] - level[0]) < 1.5, "ripple: the strikes' waves move the partials",
              fmt("4th harmonic of the first note swings %.1f dB still, %.1f dB on the ripple; level %.1f / %.1f dBFS", sd[0], sd[1], level[0], level[1]));
    }

    // 42. The reservoir: listening to the conductor its readout moves, inside its range, the same on
    //     every run; left alone it comes to rest when its chaos is low and runs on by itself when high.
    {
        auto run = [&](float chaos, float input, bool conductor, std::vector<float>& seq) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::AutoRate, 5.0f);
            e.setParam(ParamId::Brightness, 0.5f);
            e.setParam(ParamId::ReservoirChaos, chaos);
            e.setParam(ParamId::ReservoirInput, input);
            e.setParam(ParamId::ReservoirMemory, 1.0f);
            e.setModRow(0, { ModSource::Reservoir1, static_cast<int>(ParamId::Brightness), 0.5f });
            if (!openOffline(e, conductor, 137)) return false;
            render(e, 10.0);
            for (int k = 0; k < 60; ++k) {
                render(e, 0.5);
                seq.push_back(e.playedNorm(ParamId::Brightness));
            }
            return true;
        };
        auto swing = [](const std::vector<float>& s) {
            float lo = 1e9f, hi = -1e9f;
            for (float v : s) { lo = std::min(lo, v); hi = std::max(hi, v); }
            return hi - lo;
        };
        std::vector<float> a, b, calm, wild;
        if (!run(0.95f, 1.0f, true, a) || !run(0.95f, 1.0f, true, b) || !run(0.6f, 0.0f, false, calm) || !run(1.4f, 0.0f, false, wild)) return 1;
        bool sane = true;
        for (float v : a) if (!std::isfinite(v) || v < -0.001f || v > 1.001f) sane = false;
        check(sane && a == b && swing(a) > 0.05f && swing(calm) < 0.01f && swing(wild) > 0.05f,
              "modulation: the reservoir listens, rests and runs on",
              fmt("swing %.3f with the conductor (%s on a second run); alone %.4f at chaos 0.6, %.3f at 1.4",
                  swing(a), a == b ? "the same" : "DIFFERENT", swing(calm), swing(wild)));
    }

    // 43. The melt: a note let go lingers in it, about as loud as it was; without spread it stays on
    //     its harmonics, with spread it melts into the space between them.
    {
        double dry = 0.0, tail[2] = { 0.0, 0.0 }, between[2] = { 0.0, 0.0 }, silent = 0.0;
        for (int k = 0; k < 3; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.005f);
            e.setParam(ParamId::Release, 0.03f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::MeltMix, k == 0 ? 0.0f : 1.0f);
            e.setParam(ParamId::MeltToFar, 0.0f);
            e.setParam(ParamId::MeltTime, 20.0f);
            e.setParam(ParamId::MeltSpread, k == 2 ? 1.5f : 0.0f);
            if (!openOffline(e, false, 139)) return 1;
            e.noteOn(50, 0.8f, 0);
            std::vector<float> lr = render(e, 3.0);
            const double f1 = soundingHz(e);
            e.noteOff(50, 0);
            const std::vector<float> rest = render(e, 2.0);
            lr.insert(lr.end(), rest.begin(), rest.end());
            if (k == 0) {
                dry = rmsDb(lr, 2 * 48000, 3 * 48000);
                silent = rmsDb(lr, 3 * 48000 + 9600, 4 * 48000);
                continue;
            }
            const size_t a = 3 * 48000 + 9600, b = 4 * 48000;   // 0.2 .. 1 s after the release
            tail[k - 1] = rmsDb(lr, a, b);
            // Between the 3rd and 4th harmonics against on them.
            between[k - 1] = toneDb(lr, a, b, 3.5 * f1) - 0.5 * (toneDb(lr, a, b, 3.0 * f1) + toneDb(lr, a, b, 4.0 * f1));
        }
        check(tail[0] > silent + 30.0 && std::fabs(tail[0] - dry) < 4.0 && std::fabs(tail[1] - dry) < 4.0 && between[1] > between[0] + 10.0,
              "melt: a note lingers, and melts between its harmonics",
              fmt("held %.1f dBFS, after the release %.1f / %.1f dBFS spread (without the melt %.1f); between the harmonics %.1f dB still, %.1f dB spread",
                  dry, tail[0], tail[1], silent, between[0], between[1]));
    }

    // 44. The glide: one sound becomes another. Every number on its way in its own domain, the
    //     filter fading in by its share, the motion through nothing where it changes, the level
    //     stepping no more than either sound does by itself; and at the end exactly the new sound.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Attack, 0.2f);
        e.setParam(ParamId::Tilt, 0.6f);
        e.setParam(ParamId::Brightness, 0.9f);
        e.setParam(ParamId::CloudWidth, 10.0f);
        e.setParam(ParamId::Motion, 4.0f);   // Breathe
        e.setParam(ParamId::MotionDepth, 0.8f);
        e.setParam(ParamId::Scale, 0.0f);    // 12-TET
        if (!openOffline(e, false, 141)) return 1;
        e.noteOn(50, 0.8f, 0);
        e.noteOn(57, 0.8f, 0);
        render(e, 1.0);
        // The level in windows of a tenth of a second: its largest step from one to the next.
        double prevDb = 0.0;
        bool first = true;
        auto window = [&](double& worst) {
            const std::vector<float> part = render(e, 0.1);
            const double db = rmsDb(part, 0, part.size() / 2);
            if (!first) worst = std::max(worst, std::fabs(db - prevDb));
            first = false;
            prevDb = db;
            return db;
        };
        double stepA = 0.0, stepGlide = 0.0, stepB = 0.0;
        for (int k = 0; k < 10; ++k) window(stepA);
        const ParamDesc& tilt = paramDesc(ParamId::Tilt);
        e.beginMorph(4.0);
        e.setParam(ParamId::Tilt, 2.0f);
        e.setParam(ParamId::Brightness, 0.4f);
        e.setParam(ParamId::CloudWidth, 40.0f);
        e.setParam(ParamId::FilterModel, 3.0f);   // LP 24
        e.setParam(ParamId::Cutoff, 600.0f);
        e.setParam(ParamId::Motion, 6.0f);        // Aurora
        e.setParam(ParamId::MotionDepth, 0.6f);
        e.setParam(ParamId::Scale, 3.0f);         // JI 7-limit
        double worstTilt = 0.0, worstMix = 0.0, depthMid = 1.0;
        bool switchedAtMiddle = true;
        for (int k = 0; k < 45; ++k) {   // until it arrives (the request is served by the next block)
            window(stepGlide);
            const float w = morphCurve(e.morphProgress());
            const float want = fromNorm(tilt, toNorm(tilt, 0.6f) + (toNorm(tilt, 2.0f) - toNorm(tilt, 0.6f)) * w);
            worstTilt = std::max(worstTilt, static_cast<double>(std::fabs(e.soundValue(ParamId::Tilt) - want)));
            worstMix = std::max(worstMix, static_cast<double>(std::fabs(e.soundFilterMix() - (e.morphProgress() < 1.0f ? w : 1.0f))));
            if (std::fabs(w - 0.5f) < 0.06f) depthMid = std::min(depthMid, static_cast<double>(e.soundValue(ParamId::MotionDepth)));
            const bool aurora = std::lround(e.soundValue(ParamId::Motion)) == 6;
            const bool pure = std::lround(e.soundValue(ParamId::Scale)) == 3;
            if (e.morphProgress() < 1.0f && (aurora != (w >= 0.5f) || pure != (w >= 0.5f))) switchedAtMiddle = false;
            if (e.morphProgress() >= 1.0f) break;
        }
        const bool arrived = e.morphProgress() >= 1.0f && e.soundValue(ParamId::Tilt) == 2.0f && e.soundValue(ParamId::Cutoff) == 600.0f
                          && e.soundFilterMix() == 1.0f && e.soundValue(ParamId::Motion) == 6.0f && e.soundValue(ParamId::MotionDepth) == 0.6f
                          && e.soundValue(ParamId::Scale) == 3.0f;
        for (int k = 0; k < 60; ++k) window(stepB);
        check(arrived && worstTilt < 1.0e-4 && worstMix < 1.0e-4 && depthMid < 0.1 && switchedAtMiddle && stepGlide < std::max(stepA, stepB) + 1.5,
              "morph: a glide from one sound into another",
              fmt("tilt on its way within %.1e, the filter's share within %.1e, motion depth %.2f at the middle (motion and scale switched there: %s); "
                  "largest step between tenths of a second %.1f dB gliding, %.1f before, %.1f after; arrived: %s", worstTilt, worstMix, depthMid,
                  switchedAtMiddle ? "yes" : "NO", stepGlide, stepA, stepB, arrived ? "yes" : "NO"));
    }

    // 45. The A/B morph: the sound is the blend of two slots at A - B (matrix included); switching
    //     it on and off glides; off, the knobs play again.
    {
        Engine e;
        quietEffects(e);
        if (!openOffline(e, false, 143)) return 1;
        float a[kNumParams], b[kNumParams];
        for (const ParamDesc& d : paramTable()) a[static_cast<size_t>(d.id)] = b[static_cast<size_t>(d.id)] = d.def;
        a[static_cast<size_t>(ParamId::Tilt)] = 0.5f;
        b[static_cast<size_t>(ParamId::Tilt)] = 2.5f;
        b[static_cast<size_t>(ParamId::FilterModel)] = 2.0f;   // LP 12
        b[static_cast<size_t>(ParamId::Cutoff)] = 800.0f;
        e.setMorphSlot(0, a, "lfo1>cutoff:0.3");
        e.setMorphSlot(1, b, "lfo2>brightness:0.4");
        e.setParam(ParamId::MorphPos, 0.25f);
        e.setParam(ParamId::MorphActive, 1.0f);
        e.noteOn(55, 0.8f, 0);
        render(e, 1.5);   // switching on glides a second
        const ParamDesc& tilt = paramDesc(ParamId::Tilt);
        const float want = fromNorm(tilt, toNorm(tilt, 0.5f) + (toNorm(tilt, 2.5f) - toNorm(tilt, 0.5f)) * 0.25f);
        const float tiltQuarter = e.soundValue(ParamId::Tilt), mixQuarter = e.soundFilterMix();
        const bool quarter = std::fabs(tiltQuarter - want) < 1.0e-4f && std::fabs(mixQuarter - 0.25f) < 1.0e-4f
                          && e.soundValue(ParamId::FilterModel) == 2.0f;
        e.setParam(ParamId::MorphPos, 1.0f);
        render(e, 0.1);
        const bool atB = e.soundValue(ParamId::Tilt) == 2.5f && e.soundFilterMix() == 1.0f;
        e.setParam(ParamId::MorphActive, 0.0f);
        render(e, 0.5);
        const float halfway = e.soundValue(ParamId::Tilt);
        render(e, 1.0);
        const bool back = e.soundValue(ParamId::Tilt) == e.param(ParamId::Tilt) && halfway != e.param(ParamId::Tilt) && halfway != 2.5f;
        check(quarter && atB && back, "morph: A/B between two slots, gliding on and off",
              fmt("at 0.25 tilt %.3f (want %.3f), filter share %.2f; at B %s; off: %.3f half a second later, then the knob's %.3f",
                  tiltQuarter, want, mixQuarter, atB ? "exact" : "NOT", halfway, e.param(ParamId::Tilt)));
    }

    // 46. The journeys: every one that comes with the instrument reads, names presets the bank has
    //     and reads back from its text; a seeded player repeats; a journey played offline glides into
    //     each of its presets in turn.
    {
        int bad = 0, unknown = 0, steps = 0;
        for (int i = 0; i < numBuiltinJourneys(); ++i) {
            Journey j, k;
            if (!j.parse(builtinJourneyText(i)) || j.steps.empty() || !k.parse(j.text().c_str()) || k.steps.size() != j.steps.size()) ++bad;
            for (const JourneyStep& st : j.steps) {
                ++steps;
                if (findPreset(st.preset.c_str()) < 0) ++unknown;
            }
        }
        Journey j;
        j.parse("journey test\ncyclic on\nVelvet Dreaming | 0:03 | 0:01\nTeeming Garden | 0:03 | 0:01\nFading Frost | 0:03-0:04 | 0:01-0:02\n");
        JourneyPlayer p1, p2;
        p1.start(j, 7);
        p2.start(j, 7);
        bool same = true;
        for (int k = 0; k < 40; ++k) {
            JourneyStep s1, s2;
            double f1 = 0.0, f2 = 0.0;
            const bool b1 = p1.advance(0.5, s1, f1), b2 = p2.advance(0.5, s2, f2);
            if (b1 != b2 || (b1 && (s1.preset != s2.preset || f1 != f2))) same = false;
        }
        // Played: each step's preset reached (its tilt and its master gain) before the next begins.
        Engine e;
        if (!openOffline(e, true, 149)) return 1;
        JourneyPlayer p;
        p.start(j, 11);
        int reached = 0, begun = 0, current = -1;
        bool checked = true;
        for (int k = 0; k < 140; ++k) {
            JourneyStep st;
            double fade = 0.0;
            if (p.advance(0.1, st, fade)) {
                if (current >= 0 && checked) ++reached;
                current = findPreset(st.preset.c_str());
                e.loadPreset(current, fade);
                ++begun;
                checked = false;
            }
            render(e, 0.1);
            if (!checked && current >= 0 && e.morphProgress() >= 1.0f) {
                float values[kNumParams];
                presetValues(current, values, nullptr);
                checked = e.soundValue(ParamId::Tilt) == values[static_cast<size_t>(ParamId::Tilt)]
                       && e.soundValue(ParamId::MasterGain) == values[static_cast<size_t>(ParamId::MasterGain)];
            }
        }
        check(bad == 0 && unknown == 0 && numBuiltinJourneys() >= 68 && same && begun >= 4 && reached >= 3 && p.passes() >= 1,
              "journeys: read, repeatable, played as glides",
              fmt("%d journeys, %d steps, %d unreadable, %d unknown presets; seeded players %s; played: %d steps begun, %d reached, %d passes",
                  numBuiltinJourneys(), steps, bad, unknown, same ? "agree" : "DIFFER", begun, reached, p.passes()));
    }

    // 47. Stretch and Shift: stretched by 60 cents an octave, the 4th harmonic sits at 4^1.05 of the
    //     fundamental, not at 4; shifted by 50 Hz, the fundamental sits 50 Hz up.
    {
        double stretched[2] = { 0.0, 0.0 }, shifted[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::CloudWidth, 4.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::Scale, 0.0f);
            e.setParam(ParamId::Stretch, k == 0 ? 1200.0f : 1260.0f);
            if (!openOffline(e, false, 151)) return 1;
            e.noteOn(48, 0.8f, 0);
            const std::vector<float> lr = render(e, 2.0);
            const double f0 = soundingHz(e);
            stretched[k] = toneDb(lr, 48000, 96000, f0 * std::pow(4.0, 1.05)) - toneDb(lr, 48000, 96000, 4.0 * f0);
        }
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::CloudWidth, 4.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::Scale, 0.0f);
            e.setParam(ParamId::Shift, k == 0 ? 0.0f : 50.0f);
            if (!openOffline(e, false, 153)) return 1;
            e.noteOn(48, 0.8f, 0);
            const std::vector<float> lr = render(e, 2.0);
            const double f0 = soundingHz(e);
            shifted[k] = toneDb(lr, 48000, 96000, f0 + 50.0) - toneDb(lr, 48000, 96000, f0);
        }
        check(stretched[1] > 10.0 && stretched[0] < -10.0 && shifted[1] > 10.0 && shifted[0] < -10.0, "spectrum: Stretch and Shift move the partials",
              fmt("4th harmonic at 4^1.05 against 4: %.1f dB harmonic, %.1f dB stretched; fundamental +50 Hz against itself: %.1f dB, %.1f dB shifted",
                  stretched[0], stretched[1], shifted[0], shifted[1]));
    }

    // 48. The spectral comb: standing, it cuts some harmonics deep and leaves others; sweeping, a
    //     harmonic dives and surfaces as the notches pass; over whole sweeps the level stays (one
    //     note standing is louder or softer as its fundamental meets a crest or a notch).
    {
        double spread = 0.0, swing = 0.0, level[2] = { 0.0, 0.0 }, swept = 0.0;
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::CloudWidth, 4.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::CombDepth, static_cast<float>(k));
            e.setParam(ParamId::CombDensity, 1.0f);
            e.setParam(ParamId::CombSweep, 0.0f);
            e.setParam(ParamId::CombStereo, 0.0f);
            if (!openOffline(e, false, 157)) return 1;
            e.noteOn(48, 0.8f, 0);
            const std::vector<float> lr = render(e, 2.0);
            const double f0 = soundingHz(e);
            level[k] = rmsDb(lr, 48000, 96000);   // standing: for the message
            if (k == 1) {
                // Against the same note without the comb: how far apart the harmonics are cut.
                Engine o;
                quietEffects(o);
                o.setParam(ParamId::Motion, 0.0f);
                o.setParam(ParamId::CloudWidth, 4.0f);
                o.setParam(ParamId::Shimmer, 0.0f);
                if (!openOffline(o, false, 157)) return 1;
                o.noteOn(48, 0.8f, 0);
                const std::vector<float> ref = render(o, 2.0);
                double lo = 1e9, hi = -1e9;
                for (int h = 2; h <= 9; ++h) {
                    const double d = toneDb(lr, 48000, 96000, h * f0) - toneDb(ref, 48000, 96000, h * f0);
                    lo = std::min(lo, d);
                    hi = std::max(hi, d);
                }
                spread = hi - lo;
            }
        }
        double plain = 0.0;
        for (int k = 0; k < 2; ++k) {   // the same note, the same seed, with and without a sweeping comb
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::CloudWidth, 4.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::CombDepth, static_cast<float>(k));
            e.setParam(ParamId::CombDensity, 1.0f);
            e.setParam(ParamId::CombSweep, 0.25f);
            e.setParam(ParamId::CombStereo, 0.0f);
            if (!openOffline(e, false, 159)) return 1;
            e.noteOn(48, 0.8f, 0);
            const std::vector<float> lr = render(e, 9.0);
            (k == 0 ? plain : swept) = rmsDb(lr, 48000, 9 * 48000);   // two whole sweeps (a notch an octave, a quarter notch a second)
            if (k == 0) continue;
            const double f4 = 4.0 * soundingHz(e);
            double lo = 1e9, hi = -1e9;
            for (int w = 0; w < 40; ++w) {
                const size_t a = static_cast<size_t>(48000 + w * 9600);
                const double db = toneDb(lr, a, a + 4800, f4);
                lo = std::min(lo, db);
                hi = std::max(hi, db);
            }
            swing = hi - lo;
        }
        check(spread > 15.0 && swing > 15.0 && std::fabs(swept - plain) < 1.0, "comb: notches across the harmonics, travelling",
              fmt("standing: harmonics 2-9 cut %.1f dB apart (one note %.1f dBFS against %.1f); sweeping: the 4th swings %.1f dB; "
                  "over two sweeps %.1f dBFS against %.1f", spread, level[1], level[0], swing, swept, plain));
    }

    // 49. The player's own landscape: one that is bright only around 1 kHz, chosen as Your Own and
    //     read by Aurora, lifts the 8th harmonic (near 1 kHz) far above the 2nd.
    {
        double tilt[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::CloudWidth, 4.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::Motion, k == 0 ? 0.0f : 6.0f);   // None, Aurora
            e.setParam(ParamId::MotionDepth, 1.0f);
            e.setParam(ParamId::Landscape, static_cast<float>(kUserLandscape));
            Landscape l;
            l.columns = 256;
            l.rows = 64;
            l.rowSeconds = 0.5f;
            l.amps.assign(static_cast<size_t>(l.columns) * l.rows, 0.01f);
            for (int r = 0; r < l.rows; ++r)
                for (int c = 0; c < l.columns; ++c)
                    if (std::fabs(std::log2(landscapeColumnHz(c, l.columns) / 1000.0f)) < 0.25f) l.amps[static_cast<size_t>(r) * l.columns + c] = 1.0f;
            e.setUserLandscape(l);
            if (!openOffline(e, false, 161)) return 1;
            e.noteOn(48, 0.8f, 0);
            const std::vector<float> lr = render(e, 3.0);
            const double f0 = soundingHz(e);
            tilt[k] = toneDb(lr, 96000, 144000, 8.0 * f0) - toneDb(lr, 96000, 144000, 2.0 * f0);
        }
        check(tilt[1] > tilt[0] + 15.0, "landscape: the player's own is read",
              fmt("8th harmonic against the 2nd: %.1f dB without, %.1f dB reading a landscape bright at 1 kHz", tilt[0], tilt[1]));
    }

    // 50. The animated pictures: every one moves (the field at 6 s is not the field at 3 s), stays
    //     finite, keeps the level within 2 dB of no field, and the same seed plays the same.
    {
        std::string report;
        bool good = true;
        double level0 = 0.0;
        for (int mode : { 0, 3, 4, 5, 6, 7, 8, 9, 10 }) {
            auto play = [&](std::vector<float>& f3, std::vector<float>& f6) {
                Engine e;
                quietEffects(e);
                e.setParam(ParamId::Attack, 0.5f);
                e.setParam(ParamId::LifeMode, static_cast<float>(mode));
                e.setParam(ParamId::LifeDepth, 0.7f);
                e.setParam(ParamId::LifeRate, 0.5f);
                e.setParam(ParamId::LifeTurn, 1.0f);
                if (!openOffline(e, false, 163)) return std::vector<float>();
                for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
                std::vector<float> lr = render(e, 3.0);
                e.lifeSnapshot(f3);
                const std::vector<float> rest = render(e, 3.0);
                e.lifeSnapshot(f6);
                lr.insert(lr.end(), rest.begin(), rest.end());
                return lr;
            };
            std::vector<float> a3, a6, b3, b6;
            const std::vector<float> a = play(a3, a6), b = play(b3, b6);
            if (a.empty()) return 1;
            const double level = rmsDb(a, 3 * 48000, 6 * 48000);
            if (mode == 0) { level0 = level; continue; }
            double moved = 0.0;
            bool finite = true;
            for (size_t i = 0; i < a3.size() && i < a6.size(); ++i) {
                moved += std::fabs(a6[i] - a3[i]);
                finite = finite && std::isfinite(a6[i]);
            }
            moved /= std::max<size_t>(1, a3.size());
            const bool ok = a3.size() == static_cast<size_t>(kLifeW * kLifeH) && finite && moved > 0.005 && std::fabs(level - level0) < 2.0 && a == b;
            good = good && ok;
            report += fmt("%s %.3f %.1f%s%s; ", kLifeModeNames[mode], moved, level, a == b ? "" : " NOT identical", ok ? "" : " (FAILED)");
        }
        check(good, "life: the animated pictures move, keep the level, repeat", fmt("free %.1f dBFS; moved a cell, dBFS: ", level0) + report);
    }

    // 51. The preset map: on a preset's own point, at the default radius, the blend is that preset;
    //     half way to a neighbour a number lies between the two; the engine follows the map's target
    //     over Map Glide; the routes that come with it read, write back and walk to their points.
    {
        MapBlender blender;
        float x = 0.0f, y = 0.0f;
        presetMapPosition(100, x, y);
        MorphState at;
        MapNeighbours nb;
        blender.blend(x, y, paramDesc(ParamId::MapRadius).def, at, &nb);
        float values[kNumParams];
        presetValues(100, values, nullptr);
        double worst = 0.0;   // what carries the sound (an amount whose choice is off may differ: it is not heard)
        for (ParamId id : { ParamId::Tilt, ParamId::Brightness, ParamId::CloudWidth, ParamId::Shimmer, ParamId::Attack, ParamId::Release,
                            ParamId::MasterGain, ParamId::Depth, ParamId::FarLevel, ParamId::Drift }) {
            const ParamDesc& d = paramDesc(id);
            worst = std::max(worst, static_cast<double>(std::fabs(toNorm(d, at.values[static_cast<size_t>(id)]) - toNorm(d, values[static_cast<size_t>(id)]))));
        }
        const bool onPoint = nb.index[0] == 100 && nb.weight[0] > 0.95f && worst < 0.03;
        // Half way to the nearest other preset, a wide blend: the tilt between the two (or equal).
        const int other = nb.index[1];
        float ox = 0.0f, oy = 0.0f;
        presetMapPosition(other, ox, oy);
        MorphState mid;
        blender.blend(0.5f * (x + ox), 0.5f * (y + oy), 1.0f, mid);
        float vo[kNumParams];
        presetValues(other, vo, nullptr);
        const float ta = values[static_cast<size_t>(ParamId::Tilt)], tb = vo[static_cast<size_t>(ParamId::Tilt)], tm = mid.values[static_cast<size_t>(ParamId::Tilt)];
        const bool between = tm >= std::min(ta, tb) - 1.0e-3f && tm <= std::max(ta, tb) + 1.0e-3f;
        // The engine follows the target.
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::MapGlide, 0.3f);
        if (!openOffline(e, false, 165)) return 1;
        e.noteOn(55, 0.8f, 0);
        render(e, 0.5);
        e.setMapTarget(mid);
        e.setParam(ParamId::MapActive, 1.0f);
        render(e, 3.0);
        const ParamDesc& tilt = paramDesc(ParamId::Tilt);
        const bool follows = std::fabs(toNorm(tilt, e.soundValue(ParamId::Tilt)) - toNorm(tilt, tm)) < 1.0e-3f;
        // The routes.
        int bad = 0;
        for (int i = 0; i < numRoutePresets(); ++i) {
            Route r, back;
            char buf[4096];
            if (!r.parse(routePreset(i).points) || r.count() < 3 || r.write(buf, sizeof(buf)) <= 0 || !back.parse(buf) || back.count() != r.count()) ++bad;
        }
        Route r;
        r.parse("0.2,0.2|1|1|0.5;0.8,0.6|2|1|1.5");
        r.start(0.5f, 0.5f, 0.4f, true);
        float cx = 0.0f, cy = 0.0f, cr = 0.0f;
        bool reachedA = false, reachedB = false, looped = false;
        for (int k = 0; k < 160; ++k) {   // 8 s at 0.05 s: A after 1 s, B after 4 s, A again after 6 s
            r.update(0.05f, cx, cy, cr);
            if (std::fabs(cx - 0.2f) < 1.0e-4f && std::fabs(cy - 0.2f) < 1.0e-4f) { reachedA = true; if (reachedB) looped = true; }
            if (std::fabs(cx - 0.8f) < 1.0e-4f && std::fabs(cy - 0.6f) < 1.0e-4f && std::fabs(cr - 1.5f) < 1.0e-5f) reachedB = true;
        }
        check(onPoint && between && follows && bad == 0 && reachedA && reachedB && looped && r.running(), "map: blends, follows, and routes walk",
              fmt("on its point: preset %d weight %.3f, off by %.1e; half way the tilt %.2f between %.2f and %.2f; followed: %s; "
                  "%d routes, %d unreadable; walked A %s, B %s, round again %s",
                  nb.index[0], nb.weight[0], worst, tm, ta, tb, follows ? "yes" : "NO", numRoutePresets(), bad,
                  reachedA ? "yes" : "NO", reachedB ? "yes" : "NO", looped ? "yes" : "NO"));
    }

    // 52. A glide with modulation has no steps. Two sounds use the same LFO differently (a slow sine
    //     on the brightness, quick steps on the width) and modulate the motion's depth while the
    //     motion changes: every modulated number moves per block no more than the modulation
    //     itself does, the motion changes where its depth -- modulation included -- is nothing,
    //     the map's cursor jumping far does not make the partials or the motion jump, and a
    //     picture's landscape changes where the picture is at nothing.
    {
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Attack, 0.2f);
        e.setParam(ParamId::Motion, 4.0f);        // Breathe
        e.setParam(ParamId::MotionDepth, 0.5f);
        e.setParam(ParamId::Lfo1Shape, 0.0f);     // sine
        e.setParam(ParamId::Lfo1Rate, 0.3f);
        e.setParam(ParamId::Lfo2Shape, 1.0f);     // triangle
        e.setParam(ParamId::Lfo2Rate, 0.2f);
        e.setParam(ParamId::Macro1, 1.0f);       // a constant push on the depth: at the change it must be faded too
        e.setModText("lfo1>brightness:0.4;macro1>motion_depth:0.3");
        if (!openOffline(e, false, 167)) return 1;
        e.noteOn(50, 0.8f, 0);
        e.noteOn(57, 0.8f, 0);
        std::vector<float> L(128), R(128);
        // The played value of a number: modulated, or as it stands.
        auto played = [&](ParamId id) {
            const float n = e.playedNorm(id);
            return std::isnan(n) ? toNorm(paramDesc(id), e.soundValue(id)) : n;
        };
        // Each number's largest move per block: before (A's slow sine on the brightness), gliding,
        // after (B's quick steps on the width).
        const ParamId watched[] = { ParamId::Brightness, ParamId::CloudWidth };
        float last[2] = {};
        double before[2] = {}, gliding[2] = {}, after[2] = {}, ignore[2] = {};
        auto step = [&](double* worst) {
            e.process(L.data(), R.data(), 128);
            for (int k = 0; k < 2; ++k) {
                const float v = played(watched[k]);
                worst[k] = std::max(worst[k], static_cast<double>(std::fabs(v - last[k])));
                last[k] = v;
            }
        };
        for (int k = 0; k < 1500; ++k) step(k < 10 ? ignore : before);   // 4 s
        e.beginMorph(4.0);
        e.setParam(ParamId::Motion, 6.0f);        // Aurora
        e.setParam(ParamId::MotionDepth, 0.6f);
        e.setParam(ParamId::Lfo1Shape, 6.0f);     // steps
        e.setParam(ParamId::Lfo1Rate, 2.0f);
        e.setModText("lfo1>cloud_width:0.4;macro1>motion_depth:0.3");
        double depthAtSwitch = 0.0;
        int prevMotion = 4;
        for (int k = 0; k < 1700; ++k) {          // the glide and a little
            step(gliding);
            const int motion = static_cast<int>(std::lround(e.soundValue(ParamId::Motion)));
            if (motion != prevMotion) depthAtSwitch = std::max(depthAtSwitch, static_cast<double>(played(ParamId::MotionDepth)));
            prevMotion = motion;
        }
        for (int k = 0; k < 1500; ++k) step(after);
        // The map: on, then its target far away -- the other end of the bank.
        float va[kNumParams], vb[kNumParams];
        std::string ma, mb;
        presetValues(0, va, &ma);
        presetValues(numPresets() - 1, vb, &mb);
        ModRow ra[kModRows], rb[kModRows];
        MorphState here, yonder;
        here.set(va, ra, modRowsFromText(ma, ra, kModRows));
        yonder.set(vb, rb, modRowsFromText(mb, rb, kModRows));
        here.values[static_cast<size_t>(ParamId::Partials)] = 2000.0f;   // far apart in what can only step
        yonder.values[static_cast<size_t>(ParamId::Partials)] = 20000.0f;
        e.setParam(ParamId::MapGlide, 2.0f);
        e.setMapTarget(here);
        e.setParam(ParamId::MapActive, 1.0f);
        for (int k = 0; k < 2000; ++k) e.process(L.data(), R.data(), 128);   // the map takes over, gliding
        e.setMapTarget(yonder);
        double partialsJump = 0.0, amountAtSwitch = 0.0;
        float prevPartials = e.soundValue(ParamId::Partials);
        int prevKind[3] = { static_cast<int>(std::lround(e.soundValue(ParamId::Motion))), static_cast<int>(std::lround(e.soundValue(ParamId::LifeMode))),
                            static_cast<int>(std::lround(e.soundValue(ParamId::FieldMode))) };
        const ParamId amounts[3] = { ParamId::MotionDepth, ParamId::LifeDepth, ParamId::FieldStrength };
        const ParamId kinds[3] = { ParamId::Motion, ParamId::LifeMode, ParamId::FieldMode };
        const float partialsSpan = paramDesc(ParamId::Partials).max - paramDesc(ParamId::Partials).min;
        for (int k = 0; k < 4000; ++k) {
            const float amountBefore[3] = { played(amounts[0]), played(amounts[1]), played(amounts[2]) };
            e.process(L.data(), R.data(), 128);
            const float p = e.soundValue(ParamId::Partials);
            partialsJump = std::max(partialsJump, static_cast<double>(std::fabs(p - prevPartials) / partialsSpan));
            prevPartials = p;
            for (int j = 0; j < 3; ++j) {
                const int kind = static_cast<int>(std::lround(e.soundValue(kinds[j])));
                if (kind != prevKind[j] && prevKind[j] != 0) amountAtSwitch = std::max(amountAtSwitch, static_cast<double>(amountBefore[j]));
                prevKind[j] = kind;
            }
        }
        // A picture over one landscape glides into the same picture over another: where the
        // landscape changes, the picture -- pushed by the matrix too -- is at nothing.
        double lifeAtSwitch = 1.0;
        {
            Engine f;
            quietEffects(f);
            f.setParam(ParamId::LifeMode, 3.0f);   // Kaleidoscope
            f.setParam(ParamId::LifeDepth, 0.8f);
            f.setParam(ParamId::Landscape, 0.0f);
            f.setParam(ParamId::Macro1, 1.0f);
            f.setModText("macro1>life_depth:0.2");
            if (!openOffline(f, false, 169)) return 1;
            f.noteOn(55, 0.8f, 0);
            for (int k = 0; k < 400; ++k) f.process(L.data(), R.data(), 128);
            f.beginMorph(2.0);
            f.setParam(ParamId::Landscape, 2.0f);
            int land = 0;
            for (int k = 0; k < 1000; ++k) {
                f.process(L.data(), R.data(), 128);
                const int now = static_cast<int>(std::lround(f.soundValue(ParamId::Landscape)));
                if (now != land) {
                    const float n = f.playedNorm(ParamId::LifeDepth);
                    lifeAtSwitch = std::isnan(n) ? f.soundValue(ParamId::LifeDepth) : n;
                }
                land = now;
            }
        }
        // The brightness is moved only by A's sine, the width only by B's steps: gliding, neither may
        // move faster than its own modulator does.
        const bool smooth = gliding[0] < 1.5 * before[0] + 1.0e-3 && gliding[1] < 1.5 * after[1] + 1.0e-3;
        check(smooth && depthAtSwitch < 0.02 && partialsJump < 0.02 && amountAtSwitch < 0.02 && lifeAtSwitch < 0.02,
              "morph: modulation glides without steps",
              fmt("largest move per block: brightness %.4f gliding (%.4f by A's LFO), width %.4f gliding (%.4f by B's); motion depth %.3f where "
                  "the motion changed; the map's far jump: partials %.3f of their range in a block, an amount %.3f where its kind changed; "
                  "a picture's depth %.3f where its landscape changed",
                  gliding[0], before[0], gliding[1], after[1], depthAtSwitch, partialsJump, amountAtSwitch, lifeAtSwitch));
    }

    // 53. Chimeras and swarms (Phase 16a). One high note, so that its first harmonic's cloud has a
    //     thousand members, every eighth of them watched through the picture's snapshot. Uncoupled
    //     each pulses on its own; as a chimera an ordered part lies beside a disordered one: some
    //     neighbours' loudness moves alike, others' does not. As a swarm their pitch moves on their
    //     circles by no more than the reach: when like phases gather they all stand as far from the
    //     middle (one place, or two opposite -- the ring model counts them alike, and they pulse
    //     against each other), when they part they scatter round the circle.
    {
        struct Watch {
            std::vector<std::vector<float>> amp;   // per watched member its loudness over time
            std::vector<float> hz;                 // its last frequency
        };
        auto watch = [&](int mode, float coupling, float spread, float swarm, float reach, Watch& w) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.05f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::SyncDepth, 1.0f);
            e.setParam(ParamId::SyncRate, 0.5f);
            e.setParam(ParamId::SyncSpread, spread);
            e.setParam(ParamId::SyncCoupling, coupling);
            e.setParam(ParamId::SyncMode, static_cast<float>(mode));
            e.setParam(ParamId::SyncSwarm, swarm);
            e.setParam(ParamId::SyncReach, reach);
            e.setParam(ParamId::SyncLag, 0.47f);
            if (!openOffline(e, false, 171)) return false;
            e.noteOn(84, 0.8f, 0);
            std::vector<float> L(128), R(128), dots;
            int per = 0, members = 0;
            for (int k = 0; k < 375 * 24; ++k) {   // 24 s
                e.process(L.data(), R.data(), 128);
                if (k < 375 * 4 || !e.partialSnapshot(dots, per)) continue;   // after four seconds, as the snapshots come
                // The voice that sounds; its first cloud: the watched members up to where the frequency jumps an octave.
                for (int v = 0; per > 0 && v * per * 2 < static_cast<int>(dots.size()); ++v) {
                    const float* d = dots.data() + static_cast<size_t>(v) * per * 2;
                    if (d[0] <= 0.0f) continue;
                    if (members == 0) {
                        while (members + 1 < per && d[2 * (members + 1)] < 1.5f * d[0]) ++members;
                        ++members;
                        w.amp.assign(static_cast<size_t>(members), {});
                    }
                    w.hz.assign(static_cast<size_t>(members), 0.0f);
                    for (int j = 0; j < members; ++j) {
                        w.amp[static_cast<size_t>(j)].push_back(d[2 * j + 1]);
                        w.hz[static_cast<size_t>(j)] = d[2 * j];
                    }
                    break;
                }
            }
            return members > 8;
        };
        // The share of neighbouring watched members whose loudness correlates over 0.8 (and under 0.5).
        auto together = [](const Watch& w, double* apart = nullptr) {
            int pairs = 0, alike = 0, unlike = 0;
            for (size_t j = 0; j + 1 < w.amp.size(); ++j) {
                const std::vector<float>& a = w.amp[j];
                const std::vector<float>& b = w.amp[j + 1];
                const size_t n = std::min(a.size(), b.size());
                double ma = 0.0, mb = 0.0;
                for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
                ma /= std::max<double>(1.0, static_cast<double>(n));
                mb /= std::max<double>(1.0, static_cast<double>(n));
                double ab = 0.0, aa = 0.0, bb = 0.0;
                for (size_t i = 0; i < n; ++i) { ab += (a[i] - ma) * (b[i] - mb); aa += (a[i] - ma) * (a[i] - ma); bb += (b[i] - mb) * (b[i] - mb); }
                ++pairs;
                const double r = aa > 0.0 && bb > 0.0 ? ab / std::sqrt(aa * bb) : 0.0;
                if (r > 0.8) ++alike;
                if (r < 0.5) ++unlike;
            }
            if (apart != nullptr) *apart = pairs > 0 ? static_cast<double>(unlike) / pairs : -1.0;
            return pairs > 0 ? static_cast<double>(alike) / pairs : -1.0;
        };
        Watch chimera, free, base, gather, part;
        const bool ran = watch(1, 1.0f, 0.0f, 0.0f, 0.0f, chimera) && watch(0, 0.0f, 0.3f, 0.0f, 0.0f, free)
                      && watch(2, 1.0f, 0.1f, 1.0f, 0.0f, base) && watch(2, 1.0f, 0.1f, 1.0f, 40.0f, gather) && watch(2, 1.0f, 0.1f, -1.0f, 40.0f, part);
        double cApart = 0.0;
        const double c = together(chimera, &cApart), f = together(free);
        // The swarm's pitch against the same cloud without it: cents per watched member.
        auto offsets = [&](const Watch& w, double& spread, double& widest) {
            std::vector<double> d;
            for (size_t j = 0; j < w.hz.size() && j < base.hz.size(); ++j)
                if (w.hz[j] > 0.0f && base.hz[j] > 0.0f) d.push_back(1200.0 * std::log2(static_cast<double>(w.hz[j]) / base.hz[j]));
            double m = 0.0;
            for (double x : d) m += x;
            m /= std::max<double>(1.0, static_cast<double>(d.size()));
            // How far from the middle each stands: the spread of |offset| (the ring counts opposite places alike).
            double ma = 0.0;
            for (double x : d) ma += std::fabs(x);
            ma /= std::max<double>(1.0, static_cast<double>(d.size()));
            spread = 0.0;
            widest = 0.0;
            for (double x : d) { spread += (std::fabs(x) - ma) * (std::fabs(x) - ma); widest = std::max(widest, std::fabs(x)); }
            spread = std::sqrt(spread / std::max<double>(1.0, static_cast<double>(d.size())));
            (void)m;
        };
        double gSpread = 0.0, gWide = 0.0, pSpread = 0.0, pWide = 0.0;
        offsets(gather, gSpread, gWide);
        offsets(part, pSpread, pWide);
        check(ran && f < 0.3 && c > 0.15 && c < 0.9 && cApart > 0.05 && gSpread < 0.15 * 40.0 && pSpread > 0.2 * 40.0 && gWide < 41.0 && pWide < 41.0,
              "sync: chimeras and swarms",
              fmt("neighbours pulsing alike: %.2f as a chimera (%.2f not at all), %.2f free; a swarm of reach 40 ct: distance from the middle "
                  "scattered by %.1f ct (widest %.1f) when like phases gather, %.1f ct (widest %.1f) when they part", c, cApart, f, gSpread, gWide, pSpread, pWide));
    }

    // 54. The Fluid and the Islands (Phase 16b). The Fluid: the ink comes where the notes blow it
    //     (their columns hold more of it than the rest after a few seconds), the flow carries it on,
    //     the level holds, and two runs are the same bit for bit. The Islands: the field gathers --
    //     fewer borders along its rows after forty seconds than after five -- and the level holds.
    {
        auto run = [&](int mode, float character, double seconds, std::vector<float>& early, std::vector<float>& late,
                       std::vector<float>& out) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::LifeMode, static_cast<float>(mode));
            e.setParam(ParamId::LifeDepth, 0.8f);
            e.setParam(ParamId::LifeRate, 1.0f);
            e.setParam(ParamId::LifeCharacter, character);
            if (!openOffline(e, false, 173)) return false;
            for (int note : { 45, 57, 64 }) e.noteOn(note, 0.8f, 0);
            std::vector<float> L(128), R(128), dots;
            int per = 0;
            out.clear();
            const int blocks = static_cast<int>(seconds * 375.0);
            for (int k = 0; k < blocks; ++k) {
                e.process(L.data(), R.data(), 128);
                for (int i = 0; i < 128; ++i) { out.push_back(L[static_cast<size_t>(i)]); out.push_back(R[static_cast<size_t>(i)]); }
                if (e.partialSnapshot(dots, per)) {   // the field comes with the snapshot
                    if (k < static_cast<int>(5.0 * 375.0)) e.lifeSnapshot(early);
                    e.lifeSnapshot(late);
                }
            }
            return early.size() == static_cast<size_t>(kLifeW * kLifeH) && late.size() == early.size();
        };
        // The Fluid.
        std::vector<float> early, late, out1, out2, e2, l2;
        const bool fluidRan = run(11, 0.6f, 20.0, early, late, out1) && run(11, 0.6f, 20.0, e2, l2, out2);
        const bool same = out1 == out2;
        // The ink atNotes the notes' columns (a fundamental and its octave each, +-3 cells) against the rest, at five seconds.
        double atNotes = 0.0, rest = 0.0;
        int atNotesCells = 0, restCells = 0;
        std::vector<bool> column(static_cast<size_t>(kLifeW), false);
        for (int note : { 45, 57, 64 }) {
            const double hz = 440.0 * std::pow(2.0, (note - 69) / 12.0);
            for (double f : { hz, 2.0 * hz }) {
                const int x = static_cast<int>(std::log2(f / 20.0) / std::log2(1000.0) * kLifeW);
                for (int d = -3; d <= 3; ++d) column[static_cast<size_t>((x + d) & (kLifeW - 1))] = true;
            }
        }
        for (int i = 0; fluidRan && i < kLifeW * kLifeH; ++i) {
            if (column[static_cast<size_t>(i & (kLifeW - 1))]) { atNotes += early[static_cast<size_t>(i)]; ++atNotesCells; }
            else { rest += early[static_cast<size_t>(i)]; ++restCells; }
        }
        atNotes /= std::max(1, atNotesCells);
        rest /= std::max(1, restCells);
        double moved = 0.0;
        for (size_t i = 0; fluidRan && i < early.size(); ++i) moved += std::fabs(late[i] - early[i]);
        moved /= std::max<size_t>(1, early.size());
        const size_t frames = out1.size() / 2;
        const double fluidDb = rmsDb(out1, frames / 2, frames);
        // The Islands: borders along the rows (where the field crosses its middle).
        auto borders = [](const std::vector<float>& f) {
            int n = 0;
            for (int y = 0; y < kLifeH; ++y)
                for (int x = 0; x < kLifeW; ++x)
                    n += (f[static_cast<size_t>(y * kLifeW + x)] > 0.5f) != (f[static_cast<size_t>(y * kLifeW + ((x + 1) & (kLifeW - 1)))] > 0.5f);
            return n;
        };
        std::vector<float> ie, il, iout;
        const bool islesRan = run(12, 0.8f, 40.0, ie, il, iout);
        const int bEarly = islesRan ? borders(ie) : 0, bLate = islesRan ? borders(il) : 0;
        const size_t iframes = iout.size() / 2;
        const double islesDb = rmsDb(iout, iframes / 2, iframes);
        // Without a living field, for the level.
        std::vector<float> oe, ol, plain;
        run(0, 0.5f, 10.0, oe, ol, plain);
        const double plainDb = rmsDb(plain, plain.size() / 4, plain.size() / 2);
        // The level within 4 dB: the field's make-up is measured over the whole field, and the Fluid's ink lies
        // where the loud low harmonics read it (about 3 dB over); a preset is levelled by measurement anyway.
        check(fluidRan && same && atNotes > 2.0 * rest && moved > 0.01 && islesRan && bLate < bEarly && bLate > 0 && std::fabs(fluidDb - plainDb) < 4.0
                  && std::fabs(islesDb - plainDb) < 4.0,
              "life: the Fluid and the Islands",
              fmt("Fluid: ink %.3f at the notes, %.3f elsewhere after 5 s, %.3f moved by 20 s, twice %s, %.1f dBFS; Islands: %d borders "
                  "after 5 s, %d after 40 s, %.1f dBFS; without a field %.1f dBFS",
                  atNotes, rest, moved, same ? "the same" : "DIFFERENT", fluidDb, bEarly, bLate, islesDb, plainDb));
    }

    // 55. Turbulence (Phase 16c): a loud chord's energy climbs its harmonics -- its partials' centroid
    //     (power-weighted, from the picture's snapshot) rises -- without the chord growing louder; a soft
    //     chord's climbs far less; turned off, the spectrum settles back.
    {
        // The centroid in octaves above 20 Hz, averaged over the snapshots of a stretch, and the level.
        struct Heard { double centroid = 0.0, db = 0.0, back = 0.0; };
        auto listen = [&](float turbulence, float velocity, bool thenOff, Heard& h) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.3f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Tilt, 1.2f);
            e.setParam(ParamId::Turbulence, turbulence);
            e.setParam(ParamId::TurbulenceTime, 4.0f);
            if (!openOffline(e, false, 175)) return false;
            for (int note : { 48, 55, 60, 64 }) e.noteOn(note, velocity, 0);
            std::vector<float> L(128), R(128), dots, lr;
            int per = 0;
            double sum = 0.0, back = 0.0;
            int count = 0, backCount = 0;
            for (int k = 0; k < 375 * 30; ++k) {   // 30 s; off at 10 s when asked
                if (thenOff && k == 375 * 10) e.setParam(ParamId::Turbulence, 0.0f);
                e.process(L.data(), R.data(), 128);
                if (k >= 375 * 8 && k < 375 * 10)
                    for (int i = 0; i < 128; ++i) { lr.push_back(L[static_cast<size_t>(i)]); lr.push_back(R[static_cast<size_t>(i)]); }
                if (!e.partialSnapshot(dots, per)) continue;
                double w = 0.0, m = 0.0;
                for (size_t i = 0; i + 1 < dots.size(); i += 2)
                    if (dots[i] > 20.0f && dots[i + 1] > 0.0f) {
                        const double p = static_cast<double>(dots[i + 1]) * dots[i + 1];
                        w += p;
                        m += p * std::log2(dots[i] / 20.0);
                    }
                if (w <= 0.0) continue;
                if (k >= 375 * 8 && k < 375 * 10) { sum += m / w; ++count; }
                if (k >= 375 * 28) { back += m / w; ++backCount; }
            }
            h.centroid = count > 0 ? sum / count : 0.0;
            h.back = backCount > 0 ? back / backCount : 0.0;
            h.db = rmsDb(lr, 0, lr.size() / 2);
            return count > 0;
        };
        Heard plain, loud, soft, softPlain, settled;
        const bool ran = listen(0.0f, 0.9f, false, plain) && listen(1.0f, 0.9f, false, loud) && listen(1.0f, 0.15f, false, soft)
                      && listen(0.0f, 0.15f, false, softPlain) && listen(1.0f, 0.9f, true, settled);
        const double rise = loud.centroid - plain.centroid, softRise = soft.centroid - softPlain.centroid;
        check(ran && rise > 0.3 && std::fabs(loud.db - plain.db) < 1.5 && softRise < 0.6 * rise && std::fabs(settled.back - plain.centroid) < 0.1,
              "turbulence: a loud chord's energy climbs, a soft one's hardly",
              fmt("centroid %.2f octaves higher loud (%.1f against %.1f dBFS), %.2f soft; turned off, %.2f octaves from the plain chord after 20 s",
                  rise, loud.db, plain.db, softRise, settled.back - plain.centroid));
    }

    // 56. The filter morphs by transport (Phase 16d): a glide from a low pass at 300 Hz into a high
    //     pass at 3 kHz. Half way the filter is neither faded out nor both at half: its mass has
    //     travelled into the middle -- the partials between 600 Hz and 1.5 kHz pass more than four
    //     times as strongly as those under 100 Hz and over 8 kHz (a fade would leave them alike).
    //     The filter's gain at a partial: its amplitude against the same voice without a filter.
    {
        // Every watched partial's (Hz, amplitude) at the start, half way and at the end of the glide.
        auto take = [&](bool filtered, std::vector<float>& start, std::vector<float>& middle, std::vector<float>& end) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.05f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::KeyTrack, 0.0f);
            e.setParam(ParamId::FilterEnv, 0.0f);
            e.setParam(ParamId::FilterDrift, 0.0f);
            e.setParam(ParamId::FilterModel, filtered ? 3.0f : 0.0f);   // LP 24
            e.setParam(ParamId::Cutoff, 300.0f);
            e.setParam(ParamId::Resonance, 0.2f);
            if (!openOffline(e, false, 177)) return false;
            e.noteOn(31, 0.8f, 0);
            std::vector<float> L(128), R(128), dots;
            int per = 0;
            for (int k = 0; k < 375 * 2; ++k) {
                e.process(L.data(), R.data(), 128);
                if (e.partialSnapshot(dots, per)) start = dots;
            }
            e.beginMorph(6.0);
            e.setParam(ParamId::FilterModel, filtered ? 4.0f : 0.0f);   // HP 12
            e.setParam(ParamId::Cutoff, 3000.0f);
            bool half = false;
            for (int k = 0; k < 375 * 8; ++k) {
                e.process(L.data(), R.data(), 128);
                if (!e.partialSnapshot(dots, per)) continue;
                if (!half && e.morphProgress() >= 0.5f) { middle = dots; half = true; }
                end = dots;
            }
            return half && !start.empty();
        };
        std::vector<float> s0, m0, e0, s1, m1, e1;
        const bool ran = take(false, s0, m0, e0) && take(true, s1, m1, e1);
        // The filter's gain in a band, the power mean of the partials in it.
        auto band = [](const std::vector<float>& plain, const std::vector<float>& filtered, double lo, double hi) {
            double sum = 0.0;
            int n = 0;
            for (size_t i = 0; i + 1 < plain.size() && i + 1 < filtered.size(); i += 2)
                if (plain[i] >= lo && plain[i] < hi && plain[i + 1] > 1.0e-9f) {
                    const double g = filtered[i + 1] / plain[i + 1];
                    sum += g * g;
                    ++n;
                }
            return n > 0 ? std::sqrt(sum / n) : 0.0;
        };
        double lo[3] = {}, mid[3] = {}, hi[3] = {};
        const std::vector<float>* plain[3] = { &s0, &m0, &e0 };
        const std::vector<float>* filt[3] = { &s1, &m1, &e1 };
        for (int k = 0; ran && k < 3; ++k) {
            lo[k] = band(*plain[k], *filt[k], 40.0, 100.0);
            mid[k] = band(*plain[k], *filt[k], 600.0, 1500.0);
            hi[k] = band(*plain[k], *filt[k], 8000.0, 16000.0);
        }
        auto db = [](double g) { return 20.0 * std::log10(std::max(g, 1.0e-9)); };
        check(ran && lo[0] > 4.0 * hi[0] && mid[1] > 4.0 * lo[1] && mid[1] > 4.0 * hi[1] && hi[2] > 4.0 * lo[2],
              "filter: a morph moves the pass band through the middle",
              fmt("low / middle / high, dB: start %.1f / %.1f / %.1f, half way %.1f / %.1f / %.1f, end %.1f / %.1f / %.1f",
                  db(lo[0]), db(mid[0]), db(hi[0]), db(lo[1]), db(mid[1]), db(hi[1]), db(lo[2]), db(mid[2]), db(hi[2])));
    }

    // 57. MPE (Phase 16e): two notes on two member channels. A bend on one channel moves its note
    //     alone; a note let go keeps its bend when its channel is bent again for the next; the
    //     master's bend moves every played note. Pressure on Level lifts a note by up to 2^depth
    //     (3 dB at 0.5), slide on Brightness makes it brighter.
    {
        auto pitches = [](const Engine& e) {
            std::vector<double> hz;
            for (int i = 0; i < 16; ++i)
                if (e.voiceView(i).hz > 0.0f && e.voiceView(i).source == 0) hz.push_back(e.voiceView(i).hz);
            std::sort(hz.begin(), hz.end());
            return hz;
        };
        Engine e;
        quietEffects(e);
        e.setParam(ParamId::Motion, 0.0f);
        e.setParam(ParamId::Release, 2.0f);
        bool good = openOffline(e, false, 179);
        e.noteOn(60, 0.8f, 0, 1);
        e.noteOn(67, 0.8f, 0, 2);
        render(e, 0.5);
        const std::vector<double> before = pitches(e);
        e.setChannelBend(1, 2.0f);
        render(e, 0.2);
        const std::vector<double> bent = pitches(e);
        e.noteOff(67, 0, 2);
        render(e, 0.05);
        e.setChannelBend(2, 5.0f);   // the channel's next note would be bent; the released one stays
        render(e, 0.2);
        const std::vector<double> released = pitches(e);
        e.setChannelBend(-1, -1.0f);   // the master
        render(e, 0.2);
        const std::vector<double> master = pitches(e);
        auto cents = [](double a, double b) { return 1200.0 * std::log2(a / b); };
        const bool sizes = before.size() == 2 && bent.size() == 2 && released.size() == 2 && master.size() == 2;
        const bool pitch = sizes && std::fabs(cents(bent[0], before[0]) - 200.0) < 2.0 && std::fabs(cents(bent[1], before[1])) < 2.0
                        && std::fabs(cents(released[1], before[1])) < 2.0 && std::fabs(cents(master[0], before[0]) - 100.0) < 2.0
                        && std::fabs(cents(master[1], before[1]) + 100.0) < 2.0;
        // Pressure on Level, slide on Brightness: a note on channel 3 pressed fully against not at all,
        // slid up against at rest.
        auto play = [&](float press, float slide, double& db, double& centroid) {
            Engine x;
            quietEffects(x);
            x.setParam(ParamId::Motion, 0.0f);
            x.setParam(ParamId::MpePress, static_cast<float>(MpeTarget::Level));
            x.setParam(ParamId::MpePressDepth, 0.5f);
            x.setParam(ParamId::MpeSlide, static_cast<float>(MpeTarget::Brightness));
            x.setParam(ParamId::MpeSlideDepth, 1.0f);
            x.setParam(ParamId::Brightness, 0.4f);
            if (!openOffline(x, false, 181)) return false;
            x.setChannelPressure(3, press);
            x.setChannelSlide(3, slide);
            x.noteOn(57, 0.8f, 0, 3);
            render(x, 1.0);
            const std::vector<float> lr = render(x, 1.0);
            db = rmsDb(lr, 0, lr.size() / 2);
            std::vector<float> dots;
            int per = 0;
            centroid = 0.0;
            double w = 0.0;
            for (int k = 0; k < 40 && !x.partialSnapshot(dots, per); ++k) render(x, 0.01);
            for (size_t i = 0; i + 1 < dots.size(); i += 2)
                if (dots[i] > 20.0f) { const double pw = static_cast<double>(dots[i + 1]) * dots[i + 1]; w += pw; centroid += pw * std::log2(dots[i] / 20.0); }
            centroid = w > 0.0 ? centroid / w : 0.0;
            return true;
        };
        double dbRest = 0.0, dbPress = 0.0, dbSlide = 0.0, cRest = 0.0, cPress = 0.0, cSlide = 0.0;
        good = good && play(0.0f, 0.5f, dbRest, cRest) && play(1.0f, 0.5f, dbPress, cPress) && play(0.0f, 1.0f, dbSlide, cSlide);
        check(good && pitch && std::fabs(dbPress - dbRest - 3.01) < 0.5 && cSlide > cRest + 0.2,
              "MPE: each note its own bend, pressure and slide",
              fmt("bent %.1f ct (the other %.1f); released, its channel bent again: %.1f ct; the master -1: %.1f / %.1f ct; "
                  "pressure %+.2f dB; slide: centroid %+.2f octaves",
                  sizes ? cents(bent[0], before[0]) : 0.0, sizes ? cents(bent[1], before[1]) : 0.0, sizes ? cents(released[1], before[1]) : 0.0,
                  sizes ? cents(master[0], before[0]) : 0.0, sizes ? cents(master[1], before[1]) : 0.0, dbPress - dbRest, cSlide - cRest));
    }

    // 58. The growing hall (Phase 17a): a chord held for a minute with Grow on teaches the hall its
    //     pitches. Growing, the hall's tail after the chord dies more slowly than a new hall's; eroding,
    //     faster. The memory brought out of the engine and put into a new one rings as the trained hall.
    {
        struct Tail { double slope = 0.0, level = 0.0; };   // dB a second, dBFS
        std::vector<float> learnt;
        double age = 0.0;
        double readBack = 1.0;   // the largest difference of the memory read back after a restore, of its value
        auto play = [&](float grow, double train, const std::vector<float>* memory, bool keep, Tail& t) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::ModalLevel, 1.0f);
            e.setParam(ParamId::ModalDecay, 6.0f);
            e.setParam(ParamId::ModalBreath, 0.0f);   // its decays the same at every time: only the memory changes them
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::HallGrow, grow);
            e.setParam(ParamId::HallMemory, memory != nullptr ? 14400.0f : 60.0f);
            if (!openOffline(e, false, 177)) return false;
            if (memory != nullptr) e.setHallMemory(*memory, 60.0);
            for (int note : { 48, 55, 60, 64 }) e.noteOn(note, 0.8f, 0);
            render(e, train);
            if (keep && !e.hallMemory(learnt, age)) return false;
            if (memory != nullptr) {
                std::vector<float> back;
                double backAge = 0.0;
                e.hallMemory(back, backAge);
                readBack = 0.0;
                for (size_t b = 0; b < back.size() && b < memory->size(); ++b)
                    readBack = std::max(readBack, std::fabs(static_cast<double>(back[b]) - (*memory)[b]) / std::max(0.1, static_cast<double>((*memory)[b])));
            }
            for (int note : { 48, 55, 60, 64 }) e.noteOff(note, 0);
            const std::vector<float> lr = render(e, 2.5);
            const double a = rmsDb(lr, 12000, 36000), b = rmsDb(lr, 84000, 108000);   // 0.25 .. 0.75 s, 1.75 .. 2.25 s after
            t.slope = (a - b) / 1.5;
            t.level = a;
            return true;
        };
        Tail plain, grown, eroded, restored;
        const bool ran = play(0.0f, 6.0, nullptr, false, plain) && play(1.0f, 60.0, nullptr, true, grown) && play(-1.0f, 60.0, nullptr, false, eroded)
                      && play(1.0f, 6.0, &learnt, false, restored);
        // The memory: the chord's root (C3, 130.8 Hz) heard far above the average, the most of the bands below it.
        const int root = static_cast<int>(std::log2(130.81 / 20.0) / 9.96578428 * kHallBins);
        double below = 0.0;
        for (int b = 0; b < root - 40; ++b) below = std::max(below, static_cast<double>(learnt.empty() ? 1.0f : learnt[static_cast<size_t>(b)]));
        const double atRoot = learnt.empty() ? 0.0 : *std::max_element(learnt.begin() + root - 3, learnt.begin() + root + 4);
        check(ran && atRoot > 2.0 && below < 0.5 && std::fabs(age - 60.0) < 2.0 && grown.slope < 0.7 * plain.slope && eroded.slope > 1.25 * plain.slope
                  && restored.slope < 0.7 * plain.slope && readBack < 0.02,
              "hall: it grows into what it hears, erodes, and is kept",
              fmt("memory %.1f at the root, %.2f below it, %.0f s learnt; the tail falls %.1f dB/s new, %.1f grown, %.1f eroded; "
                  "restored into a new engine %.1f, read back within %.2f %%",
                  atRoot, below, age, plain.slope, grown.slope, eroded.slope, restored.slope, 100.0 * readBack));
    }

    // 59. The partials' own time (Phase 17b): where the spectrum is dense their drift slows (Time Bend
    //     positive) or quickens (negative), at the same level. The drift as heard: each partial's move
    //     weighted by its amplitude.
    {
        double speed[3] = { 0.0, 0.0, 0.0 }, level[3] = { 0.0, 0.0, 0.0 };
        bool ran = true;
        const float bends[3] = { 0.0f, 1.0f, -1.0f };
        for (int k = 0; k < 3; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::Drift, 30.0f);
            e.setParam(ParamId::DriftRate, 0.3f);
            e.setParam(ParamId::Doppler, 0.0f);
            e.setParam(ParamId::Breath, 0.0f);
            e.setParam(ParamId::TimeBend, bends[k]);
            if (!openOffline(e, false, 179)) return 1;
            for (int note : { 48, 55, 60, 64 }) e.noteOn(note, 0.8f, 0);
            render(e, 2.0);
            std::vector<float> L(256), R(256), dots, last, lr;
            int per = 0;
            double moved = 0.0, weight = 0.0;
            long count = 0;
            for (int b = 0; b < 750; ++b) {   // 4 s
                e.process(L.data(), R.data(), 256);
                for (int i = 0; i < 256; ++i) { lr.push_back(L[static_cast<size_t>(i)]); lr.push_back(R[static_cast<size_t>(i)]); }
                if (!e.partialSnapshot(dots, per)) continue;
                if (last.size() == dots.size())
                    for (size_t i = 0; i + 1 < dots.size(); i += 2)
                        if (dots[i] > 20.0f && last[i] > 20.0f && dots[i + 1] > 0.0f) {
                            moved += dots[i + 1] * std::fabs(1200.0 * std::log2(static_cast<double>(dots[i]) / last[i]));
                            weight += dots[i + 1];
                            ++count;
                        }
                last = dots;
            }
            ran = ran && count > 100;
            speed[k] = weight > 0.0 ? moved / weight : 0.0;
            level[k] = rmsDb(lr, 0, lr.size() / 2);
        }
        check(ran && speed[1] < 0.75 * speed[0] && speed[2] > 1.3 * speed[0] && std::fabs(level[1] - level[0]) < 1.0 && std::fabs(level[2] - level[0]) < 1.0,
              "time bend: the partials' drift slower where the spectrum is dense, or faster",
              fmt("%.2f cents a snapshot plain, %.2f bent slow, %.2f bent fast; %.1f / %.1f / %.1f dBFS", speed[0], speed[1], speed[2], level[0], level[1], level[2]));
    }

    // 60. The Orbit field (Phase 17c): a partial circles its well -- its pitch swings about the
    //     scale's pitch, back and forth many times, within the radius -- at the level of no field.
    {
        double sd[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 }, mean[2] = { 0.0, 0.0 }, widest = 0.0;
        int crossings = 0;
        bool ran = true;
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Attack, 0.2f);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::Shimmer, 0.0f);
            e.setParam(ParamId::Drift, 0.0f);
            e.setParam(ParamId::CloudWidth, 0.0f);
            e.setParam(ParamId::Doppler, 0.0f);
            e.setParam(ParamId::Breath, 0.0f);
            e.setParam(ParamId::FieldMode, k == 0 ? 0.0f : static_cast<float>(FieldMode::Orbit));
            e.setParam(ParamId::FieldStrength, 0.5f);
            e.setParam(ParamId::FieldRadius, 20.0f);   // orbits of about 10 cents: within the well (wider ones hop to the next pitch)
            e.setParam(ParamId::FieldSpring, 0.2f);
            e.setParam(ParamId::FieldRate, 0.2f);
            if (!openOffline(e, false, 181)) return 1;
            e.noteOn(60, 0.8f, 0);
            render(e, 2.0);
            std::vector<float> L(256), R(256), dots, lr;
            std::vector<double> cents;
            int per = 0;
            for (int b = 0; b < 1875; ++b) {   // 10 s
                e.process(L.data(), R.data(), 256);
                for (int i = 0; i < 256; ++i) { lr.push_back(L[static_cast<size_t>(i)]); lr.push_back(R[static_cast<size_t>(i)]); }
                if (e.partialSnapshot(dots, per) && dots.size() >= 2 && dots[0] > 20.0f) cents.push_back(1200.0 * std::log2(dots[0] / 261.6256));   // the tuning's C sets the mean
            }
            ran = ran && cents.size() > 200;
            double m = 0.0, m2 = 0.0;
            for (double c : cents) { m += c; m2 += c * c; }
            m /= std::max<size_t>(cents.size(), 1);
            sd[k] = std::sqrt(std::max(0.0, m2 / std::max<size_t>(cents.size(), 1) - m * m));
            level[k] = rmsDb(lr, 0, lr.size() / 2);
            mean[k] = m;
            if (k == 1)
                for (size_t i = 1; i < cents.size(); ++i) {
                    if ((cents[i] - m) * (cents[i - 1] - m) < 0.0) ++crossings;
                    widest = std::max(widest, std::fabs(cents[i] - mean[0]));
                }
        }
        check(ran && sd[0] < 1.0 && sd[1] > 3.0 && sd[1] < 25.0 && widest < 60.0 && std::fabs(mean[1] - mean[0]) < 10.0 && crossings >= 10
                  && std::fabs(level[1] - level[0]) < 1.5,
              "orbit: the partials circle the scale's pitches",
              fmt("the fundamental swings %.1f cents (%.2f without a field), at most %.0f from its pitch, mean %+.1f, %d crossings in 10 s; %.1f / %.1f dBFS",
                  sd[1], sd[0], widest, mean[1] - mean[0], crossings, level[1], level[0]));
    }

    // 61. The ripple's bend (Phase 17b): a struck note's wave lingers where it sounds -- with Bend, 0.6 s
    //     later the string still stands far higher at the note's fundamental; without, the wave has run on.
    //     And the waves leaving the slow zone are no louder for it (the whole string holds no more).
    {
        double kept[2] = { 0.0, 0.0 }, whole[2] = { 0.0, 0.0 };
        bool ran = true;
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Motion, 0.0f);
            e.setParam(ParamId::RippleDepth, 1.0f);
            e.setParam(ParamId::RippleSpeed, 1.5f);
            e.setParam(ParamId::RippleDecay, 8.0f);
            e.setParam(ParamId::RippleBend, static_cast<float>(k));
            if (!openOffline(e, false, 183)) return 1;
            e.noteOn(50, 0.8f, 0);
            float u[kRippleCells];
            constexpr float perOctave = static_cast<float>(kRippleCells - 1) / 9.96578428f;
            const float c1 = std::log2(146.83f / 20.0f) * perOctave;
            auto atRoot = [&](double& energy) {   // the string's height at the fundamental, and its whole energy
                e.rippleString(u);
                double peak = 0.0;
                energy = 0.0;
                for (int i = 0; i < kRippleCells; ++i) {
                    if (std::fabs(static_cast<float>(i) - c1) <= 3.0f) peak = std::max(peak, std::fabs(static_cast<double>(u[i])));
                    energy += static_cast<double>(u[i]) * u[i];
                }
                return peak;
            };
            render(e, 0.05);   // past the latency: the strike is on the string
            double energy0 = 0.0, energy1 = 0.0;
            const double start = atRoot(energy0);
            render(e, 0.6);
            const double later = atRoot(energy1);
            ran = ran && start > 0.0 && energy0 > 0.0;
            kept[k] = start > 0.0 ? later / start : 0.0;
            whole[k] = energy0 > 0.0 ? energy1 / energy0 : 0.0;
        }
        check(ran && kept[1] > 2.0 * kept[0] && whole[1] < 1.0, "ripple: with Bend its waves linger at the notes",
              fmt("0.6 s after the strike the string stands at %.0f %% of its height at the fundamental plain, %.0f %% bent; "
                  "the whole string %.0f / %.0f %% of its energy", 100.0 * kept[0], 100.0 * kept[1], 100.0 * whole[0], 100.0 * whole[1]));
    }

    // 62. Follow Light (Phase 19a): a landscape bright only between 200 and 400 Hz, read by the Aurora
    //     motion. Following the light, the conductor puts far more of its notes there than without
    //     (without, the band is already the middle of its register, which it favours; and its rules
    //     against doublings and crowding keep spreading the chord).
    {
        Landscape band;
        band.columns = 128;
        band.rows = 64;
        band.rowSeconds = 0.5f;
        band.amps.assign(static_cast<size_t>(band.columns) * band.rows, 0.02f);
        for (int r = 0; r < band.rows; ++r)
            for (int c = 0; c < band.columns; ++c) {
                const float hz = landscapeColumnHz(c, band.columns);
                if (hz >= 200.0f && hz <= 400.0f) band.amps[static_cast<size_t>(r) * band.columns + c] = 1.0f;
            }
        double share[2] = { 0.0, 0.0 };
        int seen[2] = { 0, 0 };
        for (int k = 0; k < 2; ++k) {
            Engine e;
            quietEffects(e);
            e.setParam(ParamId::Motion, static_cast<float>(MotionKind::Aurora));
            e.setParam(ParamId::MotionDepth, 1.0f);
            e.setParam(ParamId::Landscape, static_cast<float>(kUserLandscape));
            e.setParam(ParamId::AutoMode, 0.0f);   // Free: notes come and go on their own timers
            e.setParam(ParamId::BrainRate, 2.0f);
            e.setParam(ParamId::BrainHoldMin, 1.0f);
            e.setParam(ParamId::BrainHoldMax, 3.0f);
            e.setParam(ParamId::BrainLow, 36.0f);
            e.setParam(ParamId::BrainHigh, 84.0f);
            e.setParam(ParamId::BrainLight, static_cast<float>(k));
            e.setUserLandscape(band);
            if (!openOffline(e, true, 191)) return 1;
            std::vector<float> L(256), R(256);
            int inBand = 0, all = 0;
            for (int b = 0; b < 187 * 120; ++b) {   // two minutes, a look every second
                e.process(L.data(), R.data(), 256);
                if (b % 187 != 0 || b < 187 * 10) continue;
                for (int i = 0; i < kMaxViewVoices; ++i) {
                    const VoiceView v = e.voiceView(i);
                    if (v.source != 1 || v.hz <= 0.0f || v.gain <= 0.0f) continue;
                    ++all;
                    if (v.hz >= 190.0f && v.hz <= 420.0f) ++inBand;
                }
            }
            seen[k] = all;
            share[k] = all > 0 ? static_cast<double>(inBand) / all : 0.0;
        }
        check(seen[0] > 100 && seen[1] > 100 && share[1] > 0.6 && share[1] > share[0] + 0.2, "conductor: Follow Light puts the notes where the landscape is bright",
              fmt("fundamentals in the bright band (200 .. 400 Hz): %.0f %% without, %.0f %% following the light (%d / %d voices seen)",
                  100.0 * share[0], 100.0 * share[1], seen[0], seen[1]));
    }

    // The measuring chord as additiv_presets plays it (conductor off, attack at most half a second,
    // master at 0 dB, 512-sample blocks, 14 s): the description of the last eight seconds and the
    // level of the loudest second -- so that what is measured here is what the bank was measured by.
    auto measureLike = [&](const float* values, const std::string& mod, Description& d, double& db) {
        Engine e;
        e.setRealtime(false);
        for (int i = 0; i < kNumParams; ++i) e.setParam(static_cast<ParamId>(i), values[i]);
        e.setModText(mod);
        e.setParam(ParamId::BrainOn, 0.0f);
        e.setParam(ParamId::Attack, std::min(values[static_cast<int>(ParamId::Attack)], 0.5f));
        e.setParam(ParamId::MasterGain, 0.0f);
        EngineConfig c;
        c.maxHostBlock = 512;
        c.realtimeGpuClass = false;
        std::string error;
        if (!e.prepare(c, &error)) return false;
        for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
        const int blocks = static_cast<int>(14.0 * 48000.0 / 512), keepFrom = blocks - static_cast<int>(8.0 * 48000.0 / 512);
        std::vector<float> L(512), R(512), keptL, keptR;
        double sq = 0.0, loudest = 0.0;
        int inWindow = 0;
        for (int b = 0; b < blocks; ++b) {
            e.process(L.data(), R.data(), 512);
            if (b >= keepFrom) {
                keptL.insert(keptL.end(), L.begin(), L.end());
                keptR.insert(keptR.end(), R.begin(), R.end());
            }
            for (int i = 0; i < 512; ++i) sq += 0.5 * (static_cast<double>(L[static_cast<size_t>(i)]) * L[static_cast<size_t>(i)] + static_cast<double>(R[static_cast<size_t>(i)]) * R[static_cast<size_t>(i)]);
            if (++inWindow == 93) {   // 48000 / 512: about a second
                loudest = std::max(loudest, sq / (93.0 * 512.0));
                sq = 0.0;
                inWindow = 0;
            }
        }
        d = describe(keptL, keptR, 48000.0);
        db = 10.0 * std::log10(std::max(loudest, 1e-20));
        return true;
    };

    // 63. Nudges (Phase 20): around a preset, Brighter raises the measured brightness and Wider the
    //     width, each by a good part of what the measured sensitivities predict, the level held; Calmer
    //     (a rule: the motion measured over eight seconds is too unsteady to steer by) slows every rate
    //     that moves something by 1.23 a quarter step and touches no knob at nothing.
    {
        const int index = 64;   // Choirs' first
        float values[kNumParams], up[kNumParams], wide[kNumParams], calm[kNumParams];
        std::string mod;
        presetValues(index, values, &mod);
        const float wantB = nudge(values, 0, 1.0f, 0.5f, up);
        const float wantW = nudge(values, 2, 1.0f, 0.5f, wide);
        nudge(values, 1, -1.0f, 0.25f, calm);
        Description d0, d1, d2;
        double l0 = 0.0, l1 = 0.0, l2 = 0.0;
        const bool ran = measureLike(values, mod, d0, l0) && measureLike(up, mod, d1, l1) && measureLike(wide, mod, d2, l2);
        const double gotB = (d1.brightness - d0.brightness) / measureSpread(Measure::Brightness);
        const double gotW = (d2.width - d0.width) / measureSpread(Measure::Width);
        // The rule: rates down by 1.23 (within their ranges), nothing that was at nothing moved.
        bool rule = values[static_cast<int>(ParamId::MotionRate)] > 0.0f;
        for (ParamId id : { ParamId::MotionRate, ParamId::DriftRate, ParamId::ShimmerRate }) {
            const float v = values[static_cast<int>(id)], c = calm[static_cast<int>(id)];
            if (v > 0.0f && paramHeard(id, values)) rule = rule && (std::fabs(c - std::max(v / std::exp2(0.3f), paramDesc(id).min)) < 1.0e-4f);
        }
        for (int i = 0; i < kNumParams; ++i)
            if (values[i] == 0.0f && paramDesc(static_cast<ParamId>(i)).min == 0.0f) rule = rule && calm[i] == 0.0f && up[i] == 0.0f && wide[i] == 0.0f;
        std::string movedB, movedW;   // the knobs each moved, for the report
        for (int pass = 0; pass < 2; ++pass) {
            const float* to = pass == 0 ? up : wide;
            std::string& s = pass == 0 ? movedB : movedW;
            for (int i = 0; i < kNumParams; ++i)
                if (to[i] != values[i]) s += fmt("%s %+.2f ", paramTable()[static_cast<size_t>(i)].key,
                                                 toNorm(paramTable()[static_cast<size_t>(i)], to[i]) - toNorm(paramTable()[static_cast<size_t>(i)], values[i]));
        }
        check(ran && rule && wantB > 0.2f && gotB > 0.4 * wantB && wantW > 0.2f && gotW > 0.4 * wantW && std::fabs(l1 - l0) < 1.5 && std::fabs(l2 - l0) < 1.5,
              "nudge: brighter is brighter, wider is wider, calmer is slower",
              fmt("brighter: predicted %.2f, measured %.2f spreads (%+.1f dB; %s); wider: predicted %.2f, measured %.2f (%+.1f dB; %s); calmer by its rule: %s",
                  wantB, gotB, l1 - l0, movedB.c_str(), wantW, gotW, l2 - l0, movedW.c_str(), rule ? "yes" : "NO"));
    }

    // 64. Variations: the same dice give the same variation, other dice another; a few heard shaping
    //     knobs move (four to eight), nothing else but the master that holds the level.
    {
        float values[kNumParams], a[kNumParams], b[kNumParams], c[kNumParams];
        presetValues(64, values, nullptr);
        vary(values, 0.4f, 7, a);
        vary(values, 0.4f, 7, b);
        vary(values, 0.4f, 8, c);
        int count = 0;
        const ParamId* shaping = nullptr;
        int shapingCount = 0;
        shaping = shapingParams(shapingCount);
        bool onlyShaping = true, onlyHeard = true;
        for (int i = 0; i < kNumParams; ++i) {
            if (a[i] == values[i] || static_cast<ParamId>(i) == ParamId::MasterGain) continue;
            ++count;
            onlyShaping = onlyShaping && std::find(shaping, shaping + shapingCount, static_cast<ParamId>(i)) != shaping + shapingCount;
            onlyHeard = onlyHeard && paramHeard(static_cast<ParamId>(i), values);
        }
        const bool same = std::equal(a, a + kNumParams, b), differ = !std::equal(a, a + kNumParams, c);
        check(same && differ && count >= 1 && count <= 8 && onlyShaping && onlyHeard, "variations: repeatable, different, a few heard knobs",
              fmt("%d knobs moved, %s, %s, %s", count, same ? "the same dice the same" : "NOT repeatable", differ ? "other dice another" : "other dice the SAME",
                  onlyShaping && onlyHeard ? "all heard shaping knobs" : "some NOT heard or not shaping"));
    }

    // 65. A macro taught by example: at its places the knobs stand where they were taught, between them
    //     half way, and the macro's value glides them.
    {
        Engine e;
        quietEffects(e);
        if (!openOffline(e, false, 199)) return 1;
        MacroLesson lesson;
        float zero[kNumParams] = {}, moved[kNumParams] = {};
        moved[static_cast<int>(ParamId::Tilt)] = -0.2f;
        moved[static_cast<int>(ParamId::Shimmer)] = 0.1f;
        lesson.set(0.0f, zero);
        lesson.set(1.0f, moved);
        e.setMacroLesson(0, lesson);
        const float tilt0 = toNorm(paramDesc(ParamId::Tilt), e.param(ParamId::Tilt));
        e.setParam(ParamId::Macro1, 0.5f);
        render(e, 0.5);
        const float half = e.playedNorm(ParamId::Tilt);
        e.setParam(ParamId::Macro1, 1.0f);
        render(e, 0.5);
        const float whole = e.playedNorm(ParamId::Tilt);
        e.setParam(ParamId::Macro1, 0.0f);
        render(e, 0.5);
        const float none = e.playedNorm(ParamId::Tilt);
        check(std::fabs(half - (tilt0 - 0.1f)) < 0.005f && std::fabs(whole - (tilt0 - 0.2f)) < 0.005f && (std::isnan(none) || std::fabs(none - tilt0) < 0.005f)
                  && lesson.knobs() == 2,
              "macro: taught by example, it glides between its places",
              fmt("tilt %.3f untaught, %.3f at Macro 0.5, %.3f at 1, %.3f at 0 again (%d knobs taught)", tilt0, half, whole, std::isnan(none) ? tilt0 : none, lesson.knobs()));
    }

    // 66. Search by example: a preset measured as the bank was measured is found, nearest first.
    {
        const int index = 100;
        float values[kNumParams];
        std::string mod;
        presetValues(index, values, &mod);
        Description d;
        double db = 0.0;
        int found[5] = { -1, -1, -1, -1, -1 };
        float dist[5] = {};
        const bool ran = measureLike(values, mod, d, db);
        const int n = ran ? findLike(d, found, 5, dist) : 0;
        int rank = -1;
        for (int i = 0; i < n; ++i)
            if (found[i] == index) rank = i;
        check(ran && rank >= 0 && rank < 3 && dist[rank] < 0.05f, "find: a preset is found by its own sound",
              fmt("%s found %s (distance %.3f); nearest: %s", preset(index).name, rank >= 0 ? fmt("at %d", rank + 1).c_str() : "NOT", rank >= 0 ? dist[rank] : -1.0f,
                  n > 0 ? preset(found[0]).name : "-"));
    }

    std::printf(g_failures == 0 ? "selftest passed\n" : "selftest: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
