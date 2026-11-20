/**
 * @file NightDirector.h
 * @brief The night director (20.11.2026, plan D3): a night of so many hours as a journey with a form -- falling
 *        asleep, a deep middle in waves of about ninety minutes, a morning -- composed from the bank along a curve of
 *        stillness and brightness, every glide a step to something that sounds alike.
 *
 * A shipped Whole Night is twenty-four presets in a circle, about three and a quarter hours, the same all through. A
 * night has eight hours and a course; Robert Rich's sleep concerts are the model. The director draws it as a curve over
 * the night's position x (0 .. 1) and two measured quantities, each as a percentile of the presets it may choose from
 * (so the same curve serves any family of groups):
 *
 *  - **falling asleep**, the first 8 % of the night (30 to 45 minutes, at most a quarter): the stillness eases from
 *    the 55th percentile down to the 6th, the brightness from the 50th to the 12th (a half cosine each);
 *  - **the middle**: deep, with waves -- as many as fit at about ninety minutes each (round( middle / 90 min ), one at
 *    the least), each rising from the deep and falling back, ((1 - cos 2 pi f) / 2)^2 so most of a wave is near the
 *    deep, up to 30 points of stillness and 15 of brightness at the crest, the later waves a little higher (0.6 of
 *    that at the first, the whole at the last: sleep grows lighter towards the morning);
 *  - **the morning**, the last 8 % (30 to 45 minutes): from the deep to the 65th percentile of stillness and the 85th
 *    of brightness -- brighter and a little more moving; the dawn groups (Dawn Choruses) are drawn from as well and
 *    weigh in more the nearer the end.
 *
 * Stillness is the order make_journeys.py chooses a group's Night by, without the brightness in it (that is the
 * curve's other axis): half the measured motion, an eighth the noise, three eighths the parameters' stillness -- the
 * motion, shimmer, ripple and sync, the slowest attacks and conductor (Tools/presets/make_night.py, NightDirector.inc);
 * brightness is the measured spectral centroid (Find.inc).
 *
 * The night is walked step by step. At each, among the presets not yet played, the one with the lowest cost is
 * taken: its distance from the curve's target (in percentiles, the brightness weighing 0.7) plus 0.3 times its
 * distance in sound from the step before (the four measures in spreads, divided by 3, about a random pair's), less 0.1
 * for a preferred preset (a favourite), less up to 0.25 for a dawn preset in the morning, plus a die of up to 0.04 (the
 * seed's: another seed, another night). A step stays as make_journeys.py's nights do -- by how much the preset moves by
 * itself: the third of the candidates that moves least 3-6 minutes, the third that moves most 10-15, the rest 6-10,
 * never two short or two long ones in a row -- and the target is read at the middle of that stay. When the night is
 * full, every stay is scaled by the same factor (near one) so the night lasts what was asked at the middle of its
 * ranges. The glides are those of the nights: 1-2, 1:30-2:30 and 2-3:30 minutes.
 *
 * Pure and deterministic: no JUCE, no card, the same request gives the same night.
 * @author Rene Weller
 */
#pragma once

#include "additiv/Journey.h"

#include <cstdint>
#include <string>
#include <vector>

namespace additiv
{

constexpr int NightFamilyAll = -1;    ///< every group a shipped Whole Night draws from (the default)
constexpr int NightFamilyBank = -2;   ///< the whole bank

/** @brief What the director is asked for. */
struct NightRequest
{
    double        m_hours = 8.0;               ///< how long the night is, 1 .. 14 hours
    int           m_family = NightFamilyAll;   ///< 0 .. numNightFamilies()-1, NightFamilyAll or NightFamilyBank
    std::uint64_t m_seed = 1;                  ///< the dice; another seed, another night
    std::vector<std::string> m_prefer;         ///< presets preferred (the favourites), by name
    std::vector<std::string> m_exclude;        ///< presets never taken, by name
    bool                     m_dawn = true;    ///< the morning draws from the dawn groups too
    std::string              m_name;           ///< the journey's name; empty: "Night 8 h"
};

/** @brief One step of a composed night, with what the director saw (night_plot.py and the selftest read it). */
struct NightStep
{
    int    m_preset = -1;          ///< the bank's index
    double m_at = 0.0;             ///< seconds into the night it begins, at the middle of the ranges
    double m_stay = 0.0;           ///< seconds it stays, at the middle of its range (its glide included)
    double m_targetStill = 0.0;    ///< the curve's stillness at the middle of the stay, a percentile 0 .. 1
    double m_targetBright = 0.0;   ///< the curve's brightness there
    double m_still = 0.0;          ///< the preset's stillness, a percentile of the candidates (0 the stillest)
    double m_bright = 0.0;         ///< its brightness, a percentile of the candidates (0 the darkest)
    int    m_class = 1;            ///< 0 a short stay, 1 a middle one, 2 a long one
    bool   m_preferred = false;    ///< one of the preferred
};

/** @brief A composed night: the journey and how it came about. */
struct NightPlan
{
    Journey                m_journey;          ///< not cyclic: once through, the last preset stays
    std::vector<NightStep> m_steps;            ///< one per step of the journey
    int                    m_candidates = 0;   ///< presets it could choose from
    double                 m_scale = 1.0;      ///< the factor every stay was scaled by to fill the night
};

/** @brief How many families of groups there are to choose from (the shipped Whole Nights'). @return the count */
int numNightFamilies();

/**
 * @brief A family's name.
 * @param[in] family  0 .. numNightFamilies()-1, NightFamilyAll or NightFamilyBank
 * @return "Sleep Concert", ..., "All whole-night groups", "The whole bank"
 */
const char *nightFamilyName( int family );

/**
 * @brief The lengths of the night's first and last phase.
 * @param[in]  hours    the night's length
 * @param[out] asleep   receives the falling asleep's share of the night, 0 .. 1
 * @param[out] morning  receives the morning's share
 * @param[out] waves    receives the middle's waves
 */
void nightPhases( double hours, double &asleep, double &morning, int &waves );

/**
 * @brief The curve: where the night wants to be at a point of it (see the file comment).
 * @param[in]  x       the position in the night, 0 .. 1
 * @param[in]  hours   the night's length
 * @param[out] still   receives the stillness as a percentile, 0 the stillest
 * @param[out] bright  receives the brightness as a percentile, 0 the darkest
 */
void nightCurve( double x, double hours, double &still, double &bright );

/**
 * @brief A preset's stillness and its own motion as the director reads them (z-scores over the bank).
 * @param[in]  preset  the bank's index
 * @param[out] still   receives its stillness (lower is stiller)
 * @param[out] moving  receives how much it moves by itself (higher moves more)
 * @return false for an index the table does not hold
 */
bool nightMeasures( int preset, float &still, float &moving );

/**
 * @brief How unalike two presets sound: the distance of their four measures (brightness, motion, width, noise, each
 *        in spreads of the bank), make_journeys.py's measured part.
 * @param[in] a  a preset
 * @param[in] b  another
 * @return the distance; 0 when either is unknown
 */
float nightDistance( int a, int b );

/**
 * @brief Composes a night.
 * @param[in] request  what is asked
 * @return the night; its journey is empty when no preset could be chosen
 */
NightPlan composeNight( const NightRequest &request );

}   // namespace additiv
