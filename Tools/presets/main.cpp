/**
 * @file main.cpp
 * @brief additiv_presets: measures every factory preset's level, so make_presets.py can bring the
 *        bank to one level (Noctuary's level matching, done once for the bank).
 *
 *     additiv_presets [--from N] [--to M] [--seconds S (14)] [--out FILE (work/presets/levels.json)]
 *
 * Each preset plays the same chord (D3 A3 D4 F#4) with the conductor off, its attack shortened to
 * at most half a second and the master at 0 dB; the level is the RMS of its loudest second.
 */
#include "additiv/Engine.h"
#include "additiv/Presets.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
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
 * @brief One preset's level.
 * @param index    the preset
 * @param seconds  how long to play
 * @param error    receives the reason when the engine does not open
 * @return         dBFS RMS over the last eight seconds (or the second half, if shorter); NaN on failure
 */
double measure(int index, double seconds, std::string* error)
{
    Engine e;
    e.setRealtime(false);
    float values[kNumParams];
    std::string mod;
    presetValues(index, values, &mod);
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
    for (int b = 0; b < blocks; ++b) {
        e.process(L.data(), R.data(), n);
        for (int i = 0; i < n; ++i) sq += 0.5 * (static_cast<double>(L[i]) * L[i] + static_cast<double>(R[i]) * R[i]);
        if (++inWindow == perSecond) {
            loudest = std::max(loudest, sq / (static_cast<double>(perSecond) * n));
            sq = 0.0;
            inWindow = 0;
        }
    }
    return 10.0 * std::log10(std::max(loudest, 1e-20));
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
    double seconds = 14.0;
    fs::path out = repoRoot() / "work" / "presets" / "levels.json";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--from" && has) from = std::atoi(argv[++i]);
        else if (a == "--to" && has) to = std::atoi(argv[++i]);
        else if (a == "--seconds" && has) seconds = std::atof(argv[++i]);
        else if (a == "--out" && has) out = argv[++i];
        else { std::fprintf(stderr, "additiv_presets [--from N] [--to M] [--seconds S] [--out FILE]\n"); return 1; }
    }
    from = std::clamp(from, 0, numPresets() - 1);
    to = std::clamp(to, from, numPresets() - 1);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, double>> levels;
    double lo = 1e9, hi = -1e9;
    for (int i = from; i <= to; ++i) {
        std::string error;
        const double db = measure(i, seconds, &error);
        if (std::isnan(db)) { std::fprintf(stderr, "additiv_presets: %s\n", error.c_str()); return 1; }
        levels.emplace_back(preset(i).name, db);
        lo = std::min(lo, db);
        hi = std::max(hi, db);
        if ((i - from) % 64 == 63 || i == to)
            std::printf("  %4d / %d  %-14s %-24s %6.1f dBFS (so far %.1f .. %.1f)\n", i + 1, numPresets(), kPresetGroupNames[preset(i).group],
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
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu presets measured in %.0f s, %.1f .. %.1f dBFS -> %s\n", levels.size(), took, lo, hi, out.string().c_str());
    return 0;
}
