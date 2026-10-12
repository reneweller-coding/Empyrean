/**
 * @file selftest.cpp
 * @brief additiv_selftest: the instrument around the cloud -- the parameter table, the tuning, notes
 *        that begin on their sample, silence, determinism, the conductor's Autoplay, stealing,
 *        release, and the real-time path without underruns. Needs the GPU; the cloud's own
 *        measurements are probe's selftest.
 */
#include "additiv/Engine.h"
#include "additiv/Spectrum.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

    std::printf(g_failures == 0 ? "selftest passed\n" : "selftest: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
