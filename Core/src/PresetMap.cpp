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
    // The closest kMax + 1 by squared distance, in a small sorted list (the last only to fade the others by).
    constexpr int kList = MapNeighbours::kMax + 1;
    float dist[kList] = {};
    int index[kList] = {};
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float dx = kPresetMap[i][0] - x, dy = kPresetMap[i][1] - y;
        const float d = dx * dx + dy * dy;
        int pos = count;
        while (pos > 0 && dist[pos - 1] > d) --pos;
        if (pos >= kList) continue;
        for (int k = std::min(count, kList - 1); k > pos; --k) { dist[k] = dist[k - 1]; index[k] = index[k - 1]; }
        dist[pos] = d;
        index[pos] = i;
        if (count < kList) ++count;
    }
    b.count = std::min(count, MapNeighbours::kMax);
    for (int k = 0; k < b.count; ++k) b.index[k] = index[k];
    if (b.count == 0) return b;
    // In a gap (the nearest further than the radius) the width grows towards the third neighbour,
    // smoothly over one more radius; on a preset's own point it is that preset alone.
    const float sigma = std::max(radius, 1.0e-3f);
    const float d0 = std::sqrt(dist[0]), d3 = std::sqrt(dist[b.count >= 3 ? 2 : 0]);
    const float g = std::clamp((d0 - sigma) / sigma, 0.0f, 1.0f);
    const float sig = sigma + g * g * (3.0f - 2.0f * g) * std::max(0.0f, 0.7f * d3 - sigma);
    // Relative to the nearest (no underflow in a gap), less what the seventh would weigh.
    const float floor = count > MapNeighbours::kMax ? std::exp(-(dist[MapNeighbours::kMax] - dist[0]) / (2.0f * sig * sig)) : 0.0f;
    float sum = 0.0f;
    for (int k = 0; k < b.count; ++k) {
        b.weight[k] = std::max(0.0f, std::exp(-(dist[k] - dist[0]) / (2.0f * sig * sig)) - floor);
        sum += b.weight[k];
    }
    if (sum <= 1.0e-12f) {   // the six as far as the seventh: the nearest alone
        for (int k = 0; k < b.count; ++k) b.weight[k] = k == 0 ? 1.0f : 0.0f;
        return b;
    }
    // A neighbour under 2 % changes nothing heard but could switch on a living landscape or a
    // filter at a thousandth of its depth: 2 % come off each, so it enters at nothing.
    float rest = 0.0f;
    for (int k = 0; k < b.count; ++k) {
        b.weight[k] = std::max(0.0f, b.weight[k] / sum - 0.02f);
        rest += b.weight[k];
    }
    for (int k = 0; k < b.count; ++k) b.weight[k] /= rest;
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
    // the weighted mean (the numbers exactly, the choices by the heavier side). A neighbour of no
    // weight (mapNeighbours took 2 % off each) is left out.
    out = state(b.index[0]);
    float acc = b.weight[0];
    MorphState next;
    for (int k = 1; k < b.count; ++k) {
        if (b.weight[k] <= 0.0f) continue;
        acc += b.weight[k];
        blendMorph(out, state(b.index[k]), b.weight[k] / acc, next);
        out = next;
    }
}

} // namespace additiv
