/**
 * @file PresetMap.cpp
 * @brief The preset map (PresetMap.h); the places themselves are generated (PresetMap.inc).
 * @author Rene Weller
 */
#include "additiv/PresetMap.h"

#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace additiv {

namespace {
#include "PresetMap.inc"
constexpr int Places = static_cast<int>(sizeof(PresetMap) / sizeof(PresetMap[0]));   ///< presets on the map
} // namespace

void presetMapPosition(int index, float& x, float& y)
{
    const int i = std::clamp(index, 0, Places - 1);
    x = PresetMap[i][0];
    y = PresetMap[i][1];
}

MapNeighbours mapNeighbours(float x, float y, float radius)
{
    MapNeighbours b;
    const int n = std::min(Places, numPresets());
    // The closest Max + 1 by squared distance, in a small sorted list (the last only to fade the others by).
    constexpr int List = MapNeighbours::Max + 1;
    float dist[List] = {};
    int index[List] = {};
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const float dx = PresetMap[i][0] - x, dy = PresetMap[i][1] - y;
        const float d = dx * dx + dy * dy;
        int pos = count;
        while (pos > 0 && dist[pos - 1] > d) --pos;
        if (pos >= List) continue;
        for (int k = std::min(count, List - 1); k > pos; --k) { dist[k] = dist[k - 1]; index[k] = index[k - 1]; }
        dist[pos] = d;
        index[pos] = i;
        if (count < List) ++count;
    }
    b.m_count = std::min(count, MapNeighbours::Max);
    for (int k = 0; k < b.m_count; ++k) b.m_index[k] = index[k];
    if (b.m_count == 0) return b;
    // The width: the radius in the spacing there, the distance to the second nearest (on a
    // preset's own point its neighbour's; between two, half their distance; in a gap, the gap's).
    const float spacing = std::sqrt(dist[count >= 2 ? 1 : 0]);
    const float sig = std::max(std::max(radius, 0.01f) * spacing, 1.0e-4f);
    b.m_sigma = sig;
    // Relative to the nearest (no underflow in a gap), less what the seventh would weigh.
    const float floor = count > MapNeighbours::Max ? std::exp(-(dist[MapNeighbours::Max] - dist[0]) / (2.0f * sig * sig)) : 0.0f;
    float sum = 0.0f;
    for (int k = 0; k < b.m_count; ++k) {
        b.m_weight[k] = std::max(0.0f, std::exp(-(dist[k] - dist[0]) / (2.0f * sig * sig)) - floor);
        sum += b.m_weight[k];
    }
    if (sum <= 1.0e-12f) {   // the six as far as the seventh: the nearest alone
        for (int k = 0; k < b.m_count; ++k) b.m_weight[k] = k == 0 ? 1.0f : 0.0f;
        return b;
    }
    // A neighbour under 2 % changes nothing heard but could switch on a living landscape or a
    // filter at a thousandth of its depth: 2 % come off each, so it enters at nothing.
    float rest = 0.0f;
    for (int k = 0; k < b.m_count; ++k) {
        b.m_weight[k] = std::max(0.0f, b.m_weight[k] / sum - 0.02f);
        rest += b.m_weight[k];
    }
    for (int k = 0; k < b.m_count; ++k) b.m_weight[k] /= rest;
    return b;
}

const MorphState& MapBlender::state(int index)
{
    auto it = m_cache.find(index);
    if (it != m_cache.end()) return it->second;
    float values[NumParams];
    std::string mod;
    presetValues(index, values, &mod);
    ModRow rows[ModRows];
    const int n = modRowsFromText(mod, rows, ModRows);
    MorphState s;
    s.set(values, rows, n);
    return m_cache.emplace(index, s).first->second;
}

void MapBlender::blend(float x, float y, float radius, MorphState& out, MapNeighbours* around)
{
    const MapNeighbours b = mapNeighbours(x, y, radius);
    if (around != nullptr) *around = b;
    if (b.m_count == 0) return;
    // One neighbour after another, each by its share of the weight so far: the running blend is
    // the weighted mean (the numbers exactly, the choices by the heavier side). A neighbour of no
    // weight (mapNeighbours took 2 % off each) is left out.
    out = state(b.m_index[0]);
    float acc = b.m_weight[0];
    MorphState next;
    for (int k = 1; k < b.m_count; ++k) {
        if (b.m_weight[k] <= 0.0f) continue;
        acc += b.m_weight[k];
        blendMorph(out, state(b.m_index[k]), b.m_weight[k] / acc, next);
        out = next;
    }
}

} // namespace additiv
