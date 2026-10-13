/**
 * @file main.cpp
 * @brief additiv_presets: measures every factory preset's level, so make_presets.py can bring the
 *        bank to one level (Noctuary's level matching, done once for the bank).
 *
 *     additiv_presets [--from N] [--to M] [--names FILE] [--seconds S (14)] [--out FILE (work/presets/levels.json)]
 *                     [--describe FILE]
 *
 * --names measures only the presets a file names, one a line (those a change of the bank touched);
 * what levels.json held for the others stays.
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
#include "additiv/Engine.h"
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

/** @brief What a preset sounds like (see the file comment). */
struct Description {
    double brightness = 0.0;   ///< the spectral centroid, octaves above 40 Hz
    double motion = 0.0;       ///< the spectrum's change as a share of itself, over three lags
    double width = 0.0;        ///< 1 - the correlation of left and right
    double noise = 0.0;        ///< the bands' spectral flatness, 0 (lines) .. 1 (noise)
};

constexpr int kFft = 4096;     ///< the analysis frame
constexpr int kHop = 2048;     ///< frames start this far apart
constexpr int kBands = 36;     ///< bands from 40 Hz to 16 kHz, a quarter of an octave each nearly

/** @brief An in-place radix-2 FFT. @param x kFft values */
void fft(std::vector<std::complex<double>>& x)
{
    const int n = static_cast<int>(x.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[static_cast<size_t>(i)], x[static_cast<size_t>(j)]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double a = -2.0 * 3.14159265358979323846 / len;
        const std::complex<double> wl(std::cos(a), std::sin(a));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int k = 0; k < len / 2; ++k) {
                const std::complex<double> u = x[static_cast<size_t>(i + k)], v = x[static_cast<size_t>(i + k + len / 2)] * w;
                x[static_cast<size_t>(i + k)] = u + v;
                x[static_cast<size_t>(i + k + len / 2)] = u - v;
                w *= wl;
            }
        }
    }
}

/**
 * @brief The description of a stretch of stereo sound (the file comment).
 * @param L   left
 * @param R   right
 * @param sr  the sample rate
 * @return    it
 */
Description describe(const std::vector<float>& L, const std::vector<float>& R, double sr)
{
    Description d;
    const size_t n = L.size();
    if (n < static_cast<size_t>(kFft) * 4) return d;
    std::vector<double> win(kFft);
    for (int i = 0; i < kFft; ++i) win[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / kFft);
    // The bands' edges in bins.
    std::array<int, kBands + 1> edge{};
    for (int b = 0; b <= kBands; ++b)
        edge[static_cast<size_t>(b)] = std::clamp(static_cast<int>(std::lround(40.0 * std::pow(400.0, static_cast<double>(b) / kBands) * kFft / sr)), 1, kFft / 2);
    std::vector<std::array<double, kBands>> frames;
    std::vector<std::complex<double>> x(kFft);
    double centroid = 0.0, flat = 0.0;
    int counted = 0;
    for (size_t at = 0; at + kFft <= n; at += kHop) {
        // Left and right in one transform: left the real part, right the imaginary.
        for (int i = 0; i < kFft; ++i)
            x[static_cast<size_t>(i)] = { L[at + static_cast<size_t>(i)] * win[static_cast<size_t>(i)], R[at + static_cast<size_t>(i)] * win[static_cast<size_t>(i)] };
        fft(x);
        std::array<double, kBands> band{};
        double sum = 0.0, moment = 0.0;
        for (int b = 0; b < kBands; ++b) {
            double p = 0.0;
            for (int k = edge[static_cast<size_t>(b)]; k < std::max(edge[static_cast<size_t>(b) + 1], edge[static_cast<size_t>(b)] + 1); ++k) {
                const std::complex<double> xk = x[static_cast<size_t>(k)], xn = std::conj(x[static_cast<size_t>(kFft - k)]);
                const std::complex<double> l = 0.5 * (xk + xn), r = std::complex<double>(0.0, -0.5) * (xk - xn);
                p += std::norm(l) + std::norm(r);
            }
            band[static_cast<size_t>(b)] = p;
            sum += p;
            moment += p * (b + 0.5) / kBands * std::log2(400.0);
        }
        frames.push_back(band);
        if (sum > 1e-18) {
            centroid += moment / sum;
            double lg = 0.0;
            for (double p : band) lg += std::log(p + sum * 1e-9);
            flat += std::exp(lg / kBands) / (sum / kBands);
            ++counted;
        }
    }
    if (counted > 0) {
        d.brightness = centroid / counted;
        d.noise = flat / counted;
    }
    // The motion: the bands' magnitudes against themselves a lag before, as a share of them.
    const double frameSeconds = kHop / sr;
    double motion = 0.0;
    int lags = 0;
    for (double lagSeconds : { 0.25, 1.0, 3.0 }) {
        const size_t lag = static_cast<size_t>(std::max(1L, std::lround(lagSeconds / frameSeconds)));
        if (lag >= frames.size()) continue;
        double change = 0.0, total = 0.0;
        for (size_t f = lag; f < frames.size(); ++f)
            for (int b = 0; b < kBands; ++b) {
                const double m = std::sqrt(frames[f][static_cast<size_t>(b)]), m0 = std::sqrt(frames[f - lag][static_cast<size_t>(b)]);
                change += std::fabs(m - m0);
                total += 0.5 * (m + m0);
            }
        if (total > 1e-12) { motion += change / total; ++lags; }
    }
    d.motion = lags > 0 ? motion / lags : 0.0;
    double ll = 0.0, rr = 0.0, lr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ll += static_cast<double>(L[i]) * L[i];
        rr += static_cast<double>(R[i]) * R[i];
        lr += static_cast<double>(L[i]) * R[i];
    }
    d.width = ll > 1e-18 && rr > 1e-18 ? 1.0 - lr / std::sqrt(ll * rr) : 0.0;
    return d;
}

/**
 * @brief One preset's level, and what it sounds like.
 * @param index    the preset
 * @param seconds  how long to play
 * @param error    receives the reason when the engine does not open
 * @param desc     receives the description of the last eight seconds, when not null
 * @return         dBFS RMS of the loudest second; NaN on failure
 */
double measure(int index, double seconds, std::string* error, Description* desc)
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
    std::string namesFile, describeFile;
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
        else { std::fprintf(stderr, "additiv_presets [--from N] [--to M] [--names FILE] [--seconds S] [--out FILE] [--describe FILE]\n"); return 1; }
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
    if (!describeFile.empty()) {   // name: [brightness, motion, width, noise, level]
        const fs::path dp(describeFile);
        if (dp.has_parent_path()) fs::create_directories(dp.parent_path());
        std::ofstream df(dp);
        df << "{\n";
        for (size_t i = 0; i < descriptions.size(); ++i) {
            const Description& d = descriptions[i].second;
            char line[256];
            std::snprintf(line, sizeof line, "  \"%s\": [%.5f, %.5f, %.5f, %.5f, %.2f]%s\n", descriptions[i].first.c_str(), d.brightness, d.motion, d.width,
                          d.noise, levels[i].second, i + 1 < descriptions.size() ? "," : "");
            df << line;
        }
        df << "}\n";
    }
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu presets measured in %.0f s, %.1f .. %.1f dBFS -> %s\n", levels.size(), took, lo, hi, out.string().c_str());
    return 0;
}
