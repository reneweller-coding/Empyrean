/**
 * @file Find.cpp
 * @brief Search by example (Find.h).
 */
#include "additiv/Find.h"

#include "additiv/Landscape.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace additiv {

namespace {
#include "Find.inc"
constexpr int kFound = static_cast<int>(sizeof(kFindFeatures) / sizeof(kFindFeatures[0]));   ///< presets in the table
} // namespace

int findLike(const Description& d, int* out, int n, float* distance)
{
    const float raw[4] = { static_cast<float>(d.brightness), std::log(static_cast<float>(std::max(d.motion, 0.0)) + 0.01f), static_cast<float>(d.width),
                           static_cast<float>(d.noise) };
    float f[4];
    for (int k = 0; k < 4; ++k) f[k] = (raw[k] - kFindMean[k]) / kFindSpread[k];
    std::vector<std::pair<float, int>> all;
    all.reserve(static_cast<size_t>(std::min(kFound, numPresets())));
    for (int i = 0; i < kFound && i < numPresets(); ++i) {
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) s += (kFindFeatures[i][k] - f[k]) * (kFindFeatures[i][k] - f[k]);
        all.emplace_back(std::sqrt(s), i);
    }
    const int m = std::clamp(n, 0, static_cast<int>(all.size()));
    std::partial_sort(all.begin(), all.begin() + m, all.end());
    for (int i = 0; i < m; ++i) {
        out[i] = all[static_cast<size_t>(i)].second;
        if (distance != nullptr) distance[i] = all[static_cast<size_t>(i)].first;
    }
    return m;
}

bool presetFeatures(int index, float* out)
{
    if (index < 0 || index >= kFound) return false;
    for (int k = 0; k < 4; ++k) out[k] = kFindFeatures[index][k];
    return true;
}

bool describeFile(const std::filesystem::path& path, Description& out, double* seconds, std::string* error)
{
    std::vector<float> L, R;
    double rate = 0.0;
    if (!loadStereo(path, L, R, rate, error)) return false;
    if (seconds != nullptr) *seconds = rate > 0.0 ? static_cast<double>(L.size()) / rate : 0.0;
    if (rate <= 0.0 || L.size() < static_cast<size_t>(rate)) {
        if (error != nullptr) *error = "the recording is shorter than a second";
        return false;
    }
    // At most two minutes, from the middle: a long piece is heard where it has arrived.
    const size_t most = static_cast<size_t>(120.0 * rate);
    if (L.size() > most) {
        const size_t from = (L.size() - most) / 2;
        L = std::vector<float>(L.begin() + static_cast<std::ptrdiff_t>(from), L.begin() + static_cast<std::ptrdiff_t>(from + most));
        R = std::vector<float>(R.begin() + static_cast<std::ptrdiff_t>(from), R.begin() + static_cast<std::ptrdiff_t>(from + most));
    }
    out = describe(L, R, rate);
    return true;
}

} // namespace additiv
