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

    // 17. The factory bank: 16 groups of 64, unique names, every key known and every value in its
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
        check(numPresets() == 1024 && full && unique && bad == 0 && outOfRange == 0 && rowsLost == 0, "presets: the factory bank",
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

    std::printf(g_failures == 0 ? "selftest passed\n" : "selftest: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
