/**
 * @file PresetMap.h
 * @brief The preset map: every factory preset a point on a square -- across dark to bright, up
 *        still to moving, as they were measured (Tools/presets/make_map.py lays it out) -- and a
 *        cursor anywhere on it that blends the presets around it, weighted by a Gaussian of the
 *        distance whose width is the radius in the spacing of the presets there.
 *
 * After Noctuary's (Core/include/ambient/PresetMap.h). On a preset's point the blend is that
 * preset; between points it is a sound that was never saved. The blend is made with the morph's
 * rules (Morph.h) -- the numbers in their own domains, a choice with an amount faded through
 * nothing where its neighbours disagree, the filter by its share, the matrices weighted -- one
 * neighbour after another, so the weights come out as a weighted mean.
 */
#pragma once

#include "additiv/Morph.h"

#include <unordered_map>

namespace additiv {

/**
 * @brief A factory preset's place on the map.
 * @param index  the preset (clamped)
 * @param x      receives across, 0 .. 1 (dark to bright)
 * @param y      receives up, 0 .. 1 (still to moving)
 */
void presetMapPosition(int index, float& x, float& y);

/** @brief The presets around a place and their weights. */
struct MapNeighbours {
    static constexpr int kMax = 6;   ///< the closest presets that take part
    int count = 0;                   ///< how many are valid
    int index[kMax] = {};            ///< presets, closest first
    float weight[kMax] = {};         ///< normalised weights, the closest heaviest
    float sigma = 0.0f;              ///< the Gaussian's width there, in map units (the radius times the spacing)
};

/**
 * @brief The presets closest to a place and their Gaussian weights. The Gaussian's width is the
 *        radius times the distance to the second nearest preset -- the spacing of the map there:
 *        the measured map is dense in its core and empty at its edges, and a width in map units
 *        would blend six unlike sounds in the one and reach nothing in the other. On a preset's
 *        own point the spacing is its neighbour's distance, so at a radius under a half the preset
 *        plays (nearly) alone; half way to a neighbour the two share; in a gap the width grows with
 *        the distances. Every weight is continuous in the place, so a cursor that moves never makes
 *        the sound jump: the weight the seventh nearest would have is taken off the six (a preset
 *        enters and leaves the six at nothing), and so are 2 % of the whole (a neighbour too faint
 *        to be heard enters at nothing rather than at 2 %).
 * @param x       across, 0 .. 1
 * @param y       up, 0 .. 1
 * @param radius  the Gaussian's width in the spacing of the presets there (MapRadius, 0.1 .. 4)
 * @return        the neighbours
 */
MapNeighbours mapNeighbours(float x, float y, float radius);

/**
 * @brief Blends the presets around a place into one sound (MorphState). Keeps every preset it has
 *        read (as a sound: its values and matrix), so moving the cursor costs only the blend.
 *        Not for the audio thread: reading a preset allocates.
 */
class MapBlender {
public:
    /**
     * @brief The sound at a place on the map.
     * @param x       across
     * @param y       up
     * @param radius  the blend's width in the spacing of the presets there (0.1 .. 4)
     * @param out     receives the blend
     * @param around  receives the neighbours (may be null)
     */
    void blend(float x, float y, float radius, MorphState& out, MapNeighbours* around = nullptr);
    /** @brief One preset as a sound. @param index the preset @return its state (kept) */
    const MorphState& state(int index);

private:
    std::unordered_map<int, MorphState> cache_;   ///< the presets read so far
};

} // namespace additiv
