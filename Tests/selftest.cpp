/**
 * @file selftest.cpp
 * @brief additiv_selftest: the instrument around the cloud -- the parameter table, the tuning, notes
 *        that begin on their sample, silence, determinism, the conductor's Autoplay, stealing,
 *        release, the real-time path without underruns, the echoes and the modal hall. Needs the
 *        GPU; the cloud's own measurements are probe's selftest.
 */
#include "additiv/Engine.h"
#include "additiv/Presets.h"
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

    // 17. The factory bank: 32 groups of 64, unique names, every key known and every value in its
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
        check(numPresets() == 2048 && full && unique && bad == 0 && outOfRange == 0 && rowsLost == 0, "presets: the factory bank",
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

    std::printf(g_failures == 0 ? "selftest passed\n" : "selftest: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
