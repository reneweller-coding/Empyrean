/**
 * @file Describe.h
 * @brief What a stretch of stereo sound sounds like, in four numbers: the brightness (the spectral
 *        centroid in octaves above 40 Hz), the motion (how much the spectrum changes over a quarter
 *        of a second, a second and three seconds, as a share of itself), the width (1 - the
 *        correlation of left and right) and the noisiness (the spectral flatness of the bands). The
 *        spectrum is the power of left and right added, never of their sum: decorrelated partials
 *        cancel in L+R and would fake motion.
 *
 *        The preset map is made of these (Tools/presets: additiv_presets --describe, make_map.py), the
 *        nudges are measured with them, and a recording the player brings is described the same way
 *        to find the presets that sound like it.
 */
#pragma once

#include <vector>

namespace additiv {

/** @brief What a sound sounds like (see the file comment). */
struct Description {
    double brightness = 0.0;   ///< the spectral centroid, octaves above 40 Hz
    double motion = 0.0;       ///< the spectrum's change as a share of itself, over three lags
    double width = 0.0;        ///< 1 - the correlation of left and right
    double noise = 0.0;        ///< the bands' spectral flatness, 0 (lines) .. 1 (noise)
};

/**
 * @brief The description of a stretch of stereo sound: at least a second or so (four analysis
 *        frames of 4096 samples); a shorter one describes as all nought.
 * @param L   left
 * @param R   right
 * @param sr  the sample rate, Hz
 * @return    it
 */
Description describe(const std::vector<float>& L, const std::vector<float>& R, double sr);

} // namespace additiv
