/**
 * @file Sun.h
 * @brief Where the sun stands (Phase 30, the night arc): its elevation for a place and a moment, the night it makes,
 *        and a place to begin with from the computer's time zone.
 * @author Rene Weller
 */
#pragma once

#include <string>

namespace additiv
{

/**
 * @brief The sun's elevation above the horizon, after the Astronomical Almanac's low-precision formulas (good to about a
 *        hundredth of a degree between 1950 and 2050; no refraction).
 * @param[in] unixSeconds  the moment, seconds since 1970-01-01 00:00 UTC
 * @param[in] latitude     degrees, north positive
 * @param[in] longitude    degrees, east positive
 * @return degrees, -90 .. 90
 */
double sunElevation( double unixSeconds, double latitude, double longitude );

/**
 * @brief The night the sun makes: 0 while it is up, 1 from astronomical night on (18 degrees below), smoothly between
 *        through the civil and nautical twilights.
 * @param[in] elevation  the sun's elevation, degrees
 * @return 0 .. 1
 */
float nightOf( double elevation );

/**
 * @brief A place to begin with: the main city of the computer's time zone (Windows' zone names, about sixty of them),
 *        or the zone's offset as a longitude at 45 degrees north.
 * @param[out] latitude   receives degrees, north positive
 * @param[out] longitude  receives degrees, east positive
 * @param[out] where      receives the city's name, or "the time zone" (may be null)
 */
void defaultLocation( double &latitude, double &longitude, std::string *where = nullptr );

}   // namespace additiv
