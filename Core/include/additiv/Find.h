/**
 * @file Find.h
 * @brief Search by example: a recording the player brings (or anything described as Describe.h does)
 *        and the presets that sound most like it -- the nearest in the four measures the preset map
 *        is made of, each scaled by its spread over the bank (Tools/presets/make_find.py).
 * @author Rene Weller
 */
#pragma once

#include "additiv/Describe.h"

#include <filesystem>
#include <string>

namespace additiv
{

/**
 * @brief The presets nearest a description, nearest first.
 * @param[in]  d         the description
 * @param[out] out       receives preset indices
 * @param[in]  n         how many at most
 * @param[out] distance  receives each one's distance (in spreads), when not null
 * @return how many it wrote
 */
int findLike( const Description &d, int *out, int n, float *distance = nullptr );

/**
 * @brief A preset's measured traits, as the search compares them: brightness, motion, width and noise,
 *        each in spreads of the bank about its mean (the preset browser filters by them).
 * @param[in]  index  the preset
 * @param[out] out    receives four values
 * @return false for an index the table does not hold
 */
bool presetFeatures( int index, float *out );

/**
 * @brief A recording (WAV or FLAC) described: at most two minutes from its middle, left and right
 *        (a mono file is its own left and right).
 * @param[in]  path     the file
 * @param[out] out      receives the description
 * @param[out] seconds  receives how long the file is, when not null
 * @param[out] error    receives the reason when it fails
 * @return true when read and long enough (a second at least)
 */
bool describeFile( const std::filesystem::path &path, Description &out, double *seconds, std::string *error );

}   // namespace additiv
