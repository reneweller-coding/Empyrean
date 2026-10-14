/**
 * @file main.cpp
 * @brief additiv_presets: measures every factory preset's level, so make_presets.py can bring the
 *        bank to one level (Noctuary's level matching, done once for the bank).
 *
 *     additiv_presets [--from N] [--to M] [--names FILE] [--seconds S (14)] [--out FILE (work/presets/levels.json)]
 *                     [--describe FILE] [--sensitivity FILE]
 *
 * --names measures only the presets a file names, one a line (those a change of the bank touched);
 * what levels.json held for the others stays.
 *
 * --sensitivity measures, for each named preset, how its five measures change when each knob that
 * shapes the sound moves (Tools/presets/make_nudges.py makes the nudges and the focus from it).
 *
 * --describe also writes what each preset sounds like over its last eight seconds, for the preset
 * map (Tools/presets/make_map.py): the brightness (the spectral centroid in octaves above 40 Hz),
 * the motion (how much the spectrum changes over a quarter of a second, a second and three seconds,
 * as a share of itself), the width (1 - the correlation of left and right) and the noisiness (the
 * spectral flatness of the bands). The spectrum is the power of left and right added, never of
 * their sum: decorrelated partials cancel in L+R and would fake motion.
 *
 * Each preset plays the same chord (D3 A3 D4 F#4) with the conductor off, its attack shortened to
 * at most half a second and the master at 0 dB; the level is the RMS of its loudest second.
 */
#include "additiv/Describe.h"
#include "additiv/Engine.h"
#include "additiv/Focus.h"
#include "additiv/Presets.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace additiv;

namespace {

/** @brief The repository root: the first folder above the program that holds CMakeLists.txt and Core. @return the root */
fs::path repoRoot()
{
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    fs::path p = fs::path(buf).parent_path();
    for (int up = 0; up < 5 && !p.empty(); ++up) {
        if (fs::exists(p / "CMakeLists.txt") && fs::exists(p / "Core")) return p;
        p = p.parent_path();
    }
    return fs::current_path();
}

/**
 * @brief One preset's level, and what it sounds like.
 * @param index    the preset
 * @param seconds  how long to play
 * @param error    receives the reason when the engine does not open
 * @param desc     receives the description of the last eight seconds, when not null
 * @param overrides  knobs set after the preset's (the sensitivity mode)
 * @return         dBFS RMS of the loudest second; NaN on failure
 */
double measure(int index, double seconds, std::string* error, Description* desc, const std::vector<std::pair<ParamId, float>>& overrides = {})
{
    Engine e;
    e.setRealtime(false);
    float values[kNumParams];
    std::string mod;
    presetValues(index, values, &mod);
    for (const auto& [id, v] : overrides) values[static_cast<int>(id)] = v;
    for (int i = 0; i < kNumParams; ++i) e.setParam(static_cast<ParamId>(i), values[i]);
    e.setModText(mod);
    e.setParam(ParamId::BrainOn, 0.0f);
    e.setParam(ParamId::Attack, std::min(values[static_cast<int>(ParamId::Attack)], 0.5f));
    e.setParam(ParamId::MasterGain, 0.0f);
    EngineConfig c;
    c.maxHostBlock = 512;
    c.realtimeGpuClass = false;
    if (!e.prepare(c, error)) return std::nan("");
    for (int note : { 50, 57, 62, 66 }) e.noteOn(note, 0.8f, 0);
    // The loudest second: a pad's sustain, a bell's strike -- what a listener levels by. (The
    // last eight seconds measured the bells' and the glass's decay, 20 dB under their strike.)
    const int n = 512;
    const int blocks = static_cast<int>(seconds * c.sampleRate / n);
    const int perSecond = static_cast<int>(c.sampleRate / n);
    std::vector<float> L(n), R(n);
    double sq = 0.0, loudest = 0.0;
    int inWindow = 0;
    // The last eight seconds kept for the description.
    const int keepFrom = std::max(0, blocks - static_cast<int>(8.0 * c.sampleRate / n));
    std::vector<float> keptL, keptR;
    for (int b = 0; b < blocks; ++b) {
        e.process(L.data(), R.data(), n);
        if (desc != nullptr && b >= keepFrom) {
            keptL.insert(keptL.end(), L.begin(), L.end());
            keptR.insert(keptR.end(), R.begin(), R.end());
        }
        for (int i = 0; i < n; ++i) sq += 0.5 * (static_cast<double>(L[i]) * L[i] + static_cast<double>(R[i]) * R[i]);
        if (++inWindow == perSecond) {
            loudest = std::max(loudest, sq / (static_cast<double>(perSecond) * n));
            sq = 0.0;
            inWindow = 0;
        }
    }
    if (desc != nullptr) *desc = describe(keptL, keptR, c.sampleRate);
    return 10.0 * std::log10(std::max(loudest, 1e-20));
}

/** @brief The knobs the sensitivity mode nudges: those that shape the sound and that a preset sets. */
const ParamId kSensed[] = {
    ParamId::Tilt, ParamId::Brightness, ParamId::OddEven, ParamId::Inharmonic, ParamId::CloudWidth, ParamId::WidthSlope, ParamId::Drift,
    ParamId::DriftRate, ParamId::Shimmer, ParamId::ShimmerRate, ParamId::Stereo, ParamId::Air, ParamId::AirCenter, ParamId::AirWidth,
    ParamId::AirJitter, ParamId::Turbulence, ParamId::MotionRate, ParamId::MotionDepth, ParamId::FieldStrength, ParamId::Dispersion,
    ParamId::TimeBend, ParamId::LifeDepth, ParamId::LifeRate, ParamId::SyncDepth, ParamId::SyncCoupling, ParamId::SyncRate,
    ParamId::RippleDepth, ParamId::Cutoff, ParamId::Resonance, ParamId::FilterEnv, ParamId::CombDepth, ParamId::CombSweep,
    ParamId::Presence, ParamId::PadLowCut, ParamId::FarLevel, ParamId::Depth, ParamId::Breath, ParamId::StereoWidth, ParamId::DelayMix,
    ParamId::ModalLevel, ParamId::SpecMix, ParamId::MeltMix, ParamId::MemoryLevel, ParamId::EarlyLevel, ParamId::NearMix,
    ParamId::Purity, ParamId::TuneGravity,
};

/**
 * @brief The sensitivity mode: per preset its measure, then each knob of kSensed moved by 0.15 of its
 *        travel (down when it stands high) and the preset measured again; per knob the change of the
 *        five measures (brightness, motion, width, noise, level) for a whole travel, one JSON line each.
 * @param which    the presets
 * @param seconds  how long each plays
 * @param out      the file
 * @return         0 on success
 */
int sensitivity(const std::vector<int>& which, double seconds, const fs::path& out)
{
    std::ofstream f(out);
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t k = 0; k < which.size(); ++k) {
        const int i = which[k];
        std::string error;
        Description d0;
        const double l0 = measure(i, seconds, &error, &d0);
        if (std::isnan(l0)) { std::fprintf(stderr, "additiv_presets: %s\n", error.c_str()); return 1; }
        float values[kNumParams];
        presetValues(i, values, nullptr);
        f << "{\"preset\": \"" << preset(i).name << "\", \"group\": " << preset(i).group << ", \"base\": [" << d0.brightness << ", " << d0.motion << ", "
          << d0.width << ", " << d0.noise << ", " << l0 << "], \"d\": {";
        bool first = true;
        for (ParamId id : kSensed) {
            const ParamDesc& pd = paramDesc(id);
            const float n0 = toNorm(pd, values[static_cast<int>(id)]);
            const float step = n0 <= 0.85f ? 0.15f : -0.15f;
            Description d1;
            const double l1 = measure(i, seconds, &error, &d1, { { id, fromNorm(pd, n0 + step) } });
            if (std::isnan(l1)) { std::fprintf(stderr, "additiv_presets: %s\n", error.c_str()); return 1; }
            f << (first ? "" : ", ") << "\"" << pd.key << "\": [" << (d1.brightness - d0.brightness) / step << ", " << (d1.motion - d0.motion) / step << ", "
              << (d1.width - d0.width) / step << ", " << (d1.noise - d0.noise) / step << ", " << (l1 - l0) / step << ", " << n0 << "]";
            first = false;
        }
        f << "}}\n";
        f.flush();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  %2zu / %zu  %-24s %5.0f s\n", k + 1, which.size(), preset(i).name, s);
        std::fflush(stdout);
    }
    return 0;
}

/**
 * @brief The nudge check: around each preset every nudge (four measures, both ways, a quarter of a
 *        spread), the change the sensitivities predict against the change measured, and the level's.
 * @param which    the presets
 * @param seconds  how long each plays
 * @return         0 on success
 */
int nudgeCheck(const std::vector<int>& which, double seconds)
{
    static const char* const kNames[4] = { "brightness", "motion", "width", "noise" };
    for (int i : which) {
        std::string error;
        Description d0;
        const double l0 = measure(i, seconds, &error, &d0);
        if (std::isnan(l0)) { std::fprintf(stderr, "additiv_presets: %s\n", error.c_str()); return 1; }
        float values[kNumParams], out[kNumParams];
        presetValues(i, values, nullptr);
        std::printf("%s\n", preset(i).name);
        for (int a = 0; a < kNudgeAxes; ++a)
            for (float sign : { -1.0f, 1.0f }) {
                const float want = nudge(values, a, sign, 0.25f, out);
                std::vector<std::pair<ParamId, float>> set;
                for (int k = 0; k < kNumParams; ++k)
                    if (out[k] != values[k]) set.emplace_back(static_cast<ParamId>(k), out[k]);
                Description d1;
                const double l1 = measure(i, seconds, &error, &d1, set);
                const double got[4] = { d1.brightness - d0.brightness, d1.motion - d0.motion, d1.width - d0.width, d1.noise - d0.noise };
                std::printf("  %-10s %+.0f  predicted %+.2f  measured %+.2f spreads  level %+.1f dB  (%zu knobs)\n", kNames[a], sign, want,
                            got[a] / measureSpread(static_cast<Measure>(a)), l1 - l0, set.size());
                std::fflush(stdout);
            }
    }
    return 0;
}

} // namespace

/**
 * @brief Measures the presets asked for and writes their levels as JSON (name: dBFS).
 * @param argc  argument count
 * @param argv  arguments
 * @return      0 on success
 */
int main(int argc, char** argv)
{
    int from = 0, to = numPresets() - 1;
    std::string namesFile, describeFile, sensitivityFile;
    bool checkNudges = false;
    double seconds = 14.0;
    fs::path out = repoRoot() / "work" / "presets" / "levels.json";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--from" && has) from = std::atoi(argv[++i]);
        else if (a == "--to" && has) to = std::atoi(argv[++i]);
        else if (a == "--seconds" && has) seconds = std::atof(argv[++i]);
        else if (a == "--out" && has) out = argv[++i];
        else if (a == "--names" && has) namesFile = argv[++i];
        else if (a == "--describe" && has) describeFile = argv[++i];
        else if (a == "--sensitivity" && has) sensitivityFile = argv[++i];
        else if (a == "--nudgecheck") checkNudges = true;
        else { std::fprintf(stderr, "additiv_presets [--from N] [--to M] [--names FILE] [--seconds S] [--out FILE] [--describe FILE] [--sensitivity FILE]\n"); return 1; }
    }
    from = std::clamp(from, 0, numPresets() - 1);
    to = std::clamp(to, from, numPresets() - 1);
    std::vector<int> which;
    if (!namesFile.empty()) {
        std::ifstream in(namesFile);
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            const int index = findPreset(line.c_str());
            if (index < 0) { std::fprintf(stderr, "additiv_presets: no preset %s\n", line.c_str()); return 1; }
            which.push_back(index);
        }
    } else {
        for (int i = from; i <= to; ++i) which.push_back(i);
    }
    if (!sensitivityFile.empty()) return sensitivity(which, seconds, sensitivityFile);
    if (checkNudges) return nudgeCheck(which, seconds);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, double>> levels;
    std::vector<std::pair<std::string, Description>> descriptions;
    double lo = 1e9, hi = -1e9;
    for (size_t k = 0; k < which.size(); ++k) {
        const int i = which[k];
        std::string error;
        Description desc;
        const double db = measure(i, seconds, &error, describeFile.empty() ? nullptr : &desc);
        if (!describeFile.empty()) descriptions.emplace_back(preset(i).name, desc);
        if (std::isnan(db)) { std::fprintf(stderr, "additiv_presets: %s\n", error.c_str()); return 1; }
        levels.emplace_back(preset(i).name, db);
        lo = std::min(lo, db);
        hi = std::max(hi, db);
        if (k % 64 == 63 || k + 1 == which.size())
            std::printf("  %4zu / %zu  %-14s %-24s %6.1f dBFS (so far %.1f .. %.1f)\n", k + 1, which.size(), kPresetGroupNames[preset(i).group],
                        preset(i).name, db, lo, hi);
        std::fflush(stdout);
    }
    // Merged into what the file already holds, so one group can be measured again on its own.
    {
        std::ifstream in(out);
        std::string line;
        while (std::getline(in, line)) {
            const size_t a = line.find('"'), b = line.find('"', a + 1), colon = line.find(':', b);
            if (a == std::string::npos || b == std::string::npos || colon == std::string::npos) continue;
            const std::string name = line.substr(a + 1, b - a - 1);
            if (std::none_of(levels.begin(), levels.end(), [&](const auto& l) { return l.first == name; }))
                levels.emplace_back(name, std::atof(line.c_str() + colon + 1));
        }
    }
    fs::create_directories(out.parent_path());
    std::ofstream f(out);
    f << "{\n";
    for (size_t i = 0; i < levels.size(); ++i)
        f << "  \"" << levels[i].first << "\": " << levels[i].second << (i + 1 < levels.size() ? ",\n" : "\n");
    f << "}\n";
    if (!describeFile.empty()) {   // name: [brightness, motion, width, noise, level], merged into what the file holds
        const fs::path dp(describeFile);
        std::vector<std::string> kept;
        {
            std::ifstream in(dp);
            std::string line;
            while (std::getline(in, line)) {
                const size_t a = line.find('"'), b = line.find('"', a + 1), open = line.find('[', b == std::string::npos ? 0 : b);
                if (a == std::string::npos || b == std::string::npos || open == std::string::npos) continue;
                const std::string name = line.substr(a + 1, b - a - 1);
                if (std::any_of(descriptions.begin(), descriptions.end(), [&](const auto& d) { return d.first == name; })) continue;
                const size_t close = line.find(']', open);
                if (close != std::string::npos) kept.push_back(line.substr(0, close + 1));
            }
        }
        if (dp.has_parent_path()) fs::create_directories(dp.parent_path());
        std::ofstream df(dp);
        df << "{\n";
        for (size_t i = 0; i < descriptions.size(); ++i) {
            const Description& d = descriptions[i].second;
            char line[256];
            std::snprintf(line, sizeof line, "  \"%s\": [%.5f, %.5f, %.5f, %.5f, %.2f]%s\n", descriptions[i].first.c_str(), d.brightness, d.motion, d.width,
                          d.noise, levels[i].second, i + 1 < descriptions.size() || !kept.empty() ? "," : "");
            df << line;
        }
        for (size_t i = 0; i < kept.size(); ++i) df << kept[i] << (i + 1 < kept.size() ? ",\n" : "\n");
        df << "}\n";
    }
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu presets measured in %.0f s, %.1f .. %.1f dBFS -> %s\n", levels.size(), took, lo, hi, out.string().c_str());
    return 0;
}
