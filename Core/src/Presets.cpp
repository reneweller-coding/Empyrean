/**
 * @file Presets.cpp
 * @brief The factory presets (Presets.h); the table itself is generated (Presets.inc).
 */
#include "additiv/Presets.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace additiv {

const char* const kPresetGroupNames[kNumPresetGroups] = {
    "Sleep Concert", "Choirs", "Glass", "Bells", "Strings", "Organs", "Air & Noise", "Deep Space",
    "Glissandi", "Pendulums", "Waves", "Shepard & Tides", "Shimmer", "Dark Drones", "Microtonal", "Keys",
    "Living Fields", "Turing Gardens", "Synchrony", "Flocks", "Ripples", "Melting Spectra", "Spectral Cascades", "Resonant Halls",
    "Ghosts & Memory", "Strange Attractors", "Reservoir Minds", "Self-Listening", "Gravity Wells", "Smear & Dispersion",
    "Lattices & Beats", "Night Gardens",
};

namespace {
#include "Presets.inc"
constexpr int kCount = static_cast<int>(sizeof(kPresets) / sizeof(kPresets[0]));   ///< presets in the table
} // namespace

bool keptByPresets(ParamId id) { return id == ParamId::Binaural; }

int numPresets() { return kCount; }

const Preset& preset(int i) { return kPresets[std::clamp(i, 0, kCount - 1)]; }

int presetValues(int i, float* values, std::string* modText)
{
    for (const ParamDesc& d : paramTable()) values[static_cast<size_t>(d.id)] = d.def;
    const Preset& p = preset(i);
    int known = 0;
    const char* s = p.params;
    while (*s != 0) {
        while (*s == ' ') ++s;
        const char* eq = std::strchr(s, '=');
        if (eq == nullptr) break;
        const char* end = std::strchr(eq, ' ');
        if (end == nullptr) end = eq + std::strlen(eq);
        char key[64] = {};
        std::memcpy(key, s, static_cast<size_t>(std::min<std::ptrdiff_t>(eq - s, 63)));
        if (const ParamDesc* d = findParam(key)) {
            values[static_cast<size_t>(d->id)] = std::clamp(static_cast<float>(std::atof(eq + 1)), d->min, d->max);
            ++known;
        }
        s = end;
    }
    if (modText != nullptr) *modText = p.mod;
    return known;
}

int presetIndexInGroup(int i)
{
    const int g = preset(i).group;
    int index = 0;
    for (int k = 0; k < std::clamp(i, 0, kCount - 1); ++k)
        if (kPresets[k].group == g) ++index;
    return index;
}

int presetAt(int group, int index)
{
    int seen = 0;
    for (int k = 0; k < kCount; ++k)
        if (kPresets[k].group == group && seen++ == index) return k;
    return -1;
}

} // namespace additiv
