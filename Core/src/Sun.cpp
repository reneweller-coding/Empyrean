/**
 * @file Sun.cpp
 * @brief Where the sun stands (Sun.h): the Astronomical Almanac's approximate solar position, the twilights as a night
 *        of 0 .. 1, and a starting place from the time zone.
 * @author Rene Weller
 */
#include "additiv/Sun.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>

#if defined( _WIN32 )
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace additiv
{
namespace
{

constexpr double Rad = 3.14159265358979323846 / 180.0;   ///< radians a degree

/** @brief A time zone's main city. */
struct ZoneCity
{
    const char *m_zone;   ///< Windows' name of the zone
    const char *m_city;   ///< its city
    double      m_lat;    ///< degrees north
    double      m_lon;    ///< degrees east
};

/** @brief Windows' time zones and a city in each (the most common ones). */
constexpr ZoneCity Zones[] = {
    { "W. Europe Standard Time", "Berlin", 52.52, 13.40 },
    { "Central Europe Standard Time", "Budapest", 47.50, 19.04 },
    { "Romance Standard Time", "Paris", 48.86, 2.35 },
    { "Central European Standard Time", "Warsaw", 52.23, 21.01 },
    { "GMT Standard Time", "London", 51.51, -0.13 },
    { "Greenwich Standard Time", "Reykjavik", 64.15, -21.94 },
    { "E. Europe Standard Time", "Chisinau", 47.01, 28.86 },
    { "FLE Standard Time", "Kyiv", 50.45, 30.52 },
    { "GTB Standard Time", "Athens", 37.98, 23.73 },
    { "Russian Standard Time", "Moscow", 55.76, 37.62 },
    { "Turkey Standard Time", "Istanbul", 41.01, 28.98 },
    { "Israel Standard Time", "Jerusalem", 31.77, 35.21 },
    { "Egypt Standard Time", "Cairo", 30.04, 31.24 },
    { "South Africa Standard Time", "Johannesburg", -26.20, 28.05 },
    { "W. Central Africa Standard Time", "Lagos", 6.52, 3.38 },
    { "E. Africa Standard Time", "Nairobi", -1.29, 36.82 },
    { "Morocco Standard Time", "Casablanca", 33.57, -7.59 },
    { "Arabian Standard Time", "Dubai", 25.20, 55.27 },
    { "Iran Standard Time", "Tehran", 35.69, 51.39 },
    { "Pakistan Standard Time", "Karachi", 24.86, 67.01 },
    { "India Standard Time", "New Delhi", 28.61, 77.21 },
    { "Nepal Standard Time", "Kathmandu", 27.72, 85.32 },
    { "Bangladesh Standard Time", "Dhaka", 23.81, 90.41 },
    { "Myanmar Standard Time", "Yangon", 16.84, 96.17 },
    { "SE Asia Standard Time", "Bangkok", 13.76, 100.50 },
    { "Singapore Standard Time", "Singapore", 1.35, 103.82 },
    { "China Standard Time", "Beijing", 39.90, 116.40 },
    { "Taipei Standard Time", "Taipei", 25.03, 121.57 },
    { "Korea Standard Time", "Seoul", 37.57, 126.98 },
    { "Tokyo Standard Time", "Tokyo", 35.68, 139.69 },
    { "Ekaterinburg Standard Time", "Yekaterinburg", 56.84, 60.61 },
    { "North Asia Standard Time", "Krasnoyarsk", 56.01, 92.85 },
    { "W. Australia Standard Time", "Perth", -31.95, 115.86 },
    { "Cen. Australia Standard Time", "Adelaide", -34.93, 138.60 },
    { "E. Australia Standard Time", "Brisbane", -27.47, 153.03 },
    { "AUS Eastern Standard Time", "Sydney", -33.87, 151.21 },
    { "Tasmania Standard Time", "Hobart", -42.88, 147.33 },
    { "New Zealand Standard Time", "Auckland", -36.85, 174.76 },
    { "Hawaiian Standard Time", "Honolulu", 21.31, -157.86 },
    { "Alaskan Standard Time", "Anchorage", 61.22, -149.90 },
    { "Pacific Standard Time", "Los Angeles", 34.05, -118.24 },
    { "US Mountain Standard Time", "Phoenix", 33.45, -112.07 },
    { "Mountain Standard Time", "Denver", 39.74, -104.99 },
    { "Canada Central Standard Time", "Regina", 50.45, -104.62 },
    { "Central Standard Time", "Chicago", 41.88, -87.63 },
    { "Central Standard Time (Mexico)", "Mexico City", 19.43, -99.13 },
    { "Eastern Standard Time", "New York", 40.71, -74.01 },
    { "SA Pacific Standard Time", "Bogota", 4.71, -74.07 },
    { "Atlantic Standard Time", "Halifax", 44.65, -63.57 },
    { "Newfoundland Standard Time", "St. John's", 47.56, -52.71 },
    { "Pacific SA Standard Time", "Santiago", -33.45, -70.67 },
    { "E. South America Standard Time", "Sao Paulo", -23.55, -46.63 },
    { "Argentina Standard Time", "Buenos Aires", -34.60, -58.38 },
    { "UTC", "Greenwich", 51.48, 0.0 },
};

}   // namespace

double sunElevation( double unixSeconds, double latitude, double longitude )
{
    const double n = unixSeconds / 86400.0 - 10957.5;   // days since J2000.0 (2000-01-01 12:00 UTC)
    const double L = std::fmod( 280.460 + 0.9856474 * n, 360.0 );
    const double g = std::fmod( 357.528 + 0.9856003 * n, 360.0 ) * Rad;
    const double lambda = ( L + 1.915 * std::sin( g ) + 0.020 * std::sin( 2.0 * g ) ) * Rad;
    const double eps = ( 23.439 - 0.0000004 * n ) * Rad;
    const double ra = std::atan2( std::cos( eps ) * std::sin( lambda ), std::cos( lambda ) );
    const double dec = std::asin( std::sin( eps ) * std::sin( lambda ) );
    double       gmst = std::fmod( 18.697374558 + 24.06570982441908 * n, 24.0 );   // hours
    if ( gmst < 0.0 ) gmst += 24.0;
    const double ha = ( gmst * 15.0 + longitude ) * Rad - ra;
    const double phi = latitude * Rad;
    const double s = std::sin( phi ) * std::sin( dec ) + std::cos( phi ) * std::cos( dec ) * std::cos( ha );
    return std::asin( std::clamp( s, -1.0, 1.0 ) ) / Rad;
}

float nightOf( double elevation )
{
    const double x = std::clamp( -elevation / 18.0, 0.0, 1.0 );
    return static_cast<float>( x * x * ( 3.0 - 2.0 * x ) );
}

void defaultLocation( double &latitude, double &longitude, std::string *where )
{
    latitude = 45.0;
    longitude = 0.0;
    if ( where ) *where = "the time zone";
#if defined( _WIN32 )
    DYNAMIC_TIME_ZONE_INFORMATION tz{};
    if ( GetDynamicTimeZoneInformation( &tz ) != TIME_ZONE_ID_INVALID )
    {
        char key[128] = {};
        WideCharToMultiByte( CP_UTF8, 0, tz.TimeZoneKeyName, -1, key, static_cast<int>( sizeof( key ) ) - 1, nullptr,
                             nullptr );
        for ( const ZoneCity &z : Zones )
            if ( std::strcmp( z.m_zone, key ) == 0 )
            {
                latitude = z.m_lat;
                longitude = z.m_lon;
                if ( where ) *where = z.m_city;
                return;
            }
        longitude = std::clamp( -static_cast<double>( tz.Bias ) * 0.25, -180.0, 180.0 );   // 15 degrees an hour
    }
#else
    const std::time_t now = std::time( nullptr );
    std::tm           local{};
    if ( localtime_r( &now, &local ) != nullptr )
        longitude = std::clamp( static_cast<double>( local.tm_gmtoff ) / 240.0, -180.0, 180.0 );
    (void)Zones;
#endif
}

}   // namespace additiv
