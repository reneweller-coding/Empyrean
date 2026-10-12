/**
 * @file PresetMap.cpp
 * @brief The preset map (PresetMap.h); the places themselves are generated (PresetMap.inc).
 */
#include "additiv/PresetMap.h"

#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace additiv {

namespace {
#include "PresetMap.inc"
constexpr int kPlaces = static_cast<int>(sizeof(kPresetMap) / sizeof(kPresetMap[0]));   ///< presets on the map
} // namespace

void presetMapPosition(int index, float& x, float& y)
{
    const int i = std::clamp(index, 0, kPlaces - 1);
    x = kPresetMap[i][0];
    y = kPresetMap[i][1];
}

MapNeighbours mapNeighbours(float x, float y, float radius)
{
    MapNeighbours b;
    const int n = std::min(kPlaces, numPresets());
    float dist[MapNeighbours::kMax] = {};
    for (int i = 0; i < n; ++i) {   // the closest kMax by squared distance, in a small sorted list
        const float dx = kPresetMap[i][0] - x, dy = kPresetMap[i][1] - y;
        const float d = dx * dx + dy * dy;
        int pos = b.count;
        while (pos > 0 && dist[pos - 1] > d) --pos;
        if (pos >= MapNeighbours::kMax) continue;
        for (int k = std::min(b.count, MapNeighbours::kMax - 1); k > pos; --k) { dist[k] = dist[k - 1]; b.index[k] = b.index[k - 1]; }
        dist[pos] = d;
        b.index[pos] = i;
        if (b.count < MapNeighbours::kMax) ++b.count;
    }
    // In a gap (the nearest further than the radius) the width grows to reach the third neighbour;
    // on a preset's own point it is that preset alone.
    const float sigma = std::max(radius, 1.0e-3f);
    const float d0 = b.count > 0 ? std::sqrt(dist[0]) : 0.0f, d3 = std::sqrt(dist[b.count >= 3 ? 2 : 0]);
    const float sig = (b.count > 0 && d0 > sigma) ? std::max(sigma, 0.7f * d3) : sigma;
    float sum = 0.0f;
    for (int k = 0; k < b.count; ++k) {
        b.weight[k] = std::exp(-dist[k] / (2.0f * sig * sig));
        sum += b.weight[k];
    }
    for (int k = 0; k < b.count; ++k) b.weight[k] = sum > 1.0e-12f ? b.weight[k] / sum : (k == 0 ? 1.0f : 0.0f);
    return b;
}

const MorphState& MapBlender::state(int index)
{
    auto it = cache_.find(index);
    if (it != cache_.end()) return it->second;
    float values[kNumParams];
    std::string mod;
    presetValues(index, values, &mod);
    ModRow rows[kModRows];
    const int n = modRowsFromText(mod, rows, kModRows);
    MorphState s;
    s.set(values, rows, n);
    return cache_.emplace(index, s).first->second;
}

void MapBlender::blend(float x, float y, float radius, MorphState& out, MapNeighbours* around)
{
    const MapNeighbours b = mapNeighbours(x, y, radius);
    if (around != nullptr) *around = b;
    if (b.count == 0) return;
    // One neighbour after another, each by its share of the weight so far: the running blend is
    // the weighted mean (the numbers exactly, the choices by the heavier side). A neighbour under
    // 2 % is left out: it would change nothing heard, but could switch on a living landscape or a
    // filter at a thousandth of its depth.
    out = state(b.index[0]);
    float acc = b.weight[0];
    MorphState next;
    for (int k = 1; k < b.count; ++k) {
        if (b.weight[k] < 0.02f) continue;
        acc += b.weight[k];
        blendMorph(out, state(b.index[k]), b.weight[k] / acc, next);
        out = next;
    }
}

} // namespace additiv
