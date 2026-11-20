/**
 * @file NightDirector.cpp
 * @brief The night director (NightDirector.h); its table of the bank is generated (NightDirector.inc).
 * @author Rene Weller
 */
#include "additiv/NightDirector.h"

#include "additiv/Find.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <unordered_set>

namespace additiv
{

namespace
{
#include "NightDirector.inc"
constexpr int Measured =
    static_cast<int>( sizeof( NightMeasures ) / sizeof( NightMeasures[0] ) );   ///< presets in the table
constexpr int Families =
    static_cast<int>( sizeof( NightFamilies ) / sizeof( NightFamilies[0] ) );   ///< families of groups

constexpr double Pi = 3.14159265358979323846;   ///< pi
// The curve (NightDirector.h), percentiles of the candidates.
constexpr double StillEvening = 0.55;    ///< the stillness the night begins at
constexpr double StillDeep = 0.06;       ///< the deep
constexpr double StillWave = 0.30;       ///< a wave's crest above the deep (the last wave's; the first 0.6 of it)
constexpr double StillMorning = 0.65;    ///< the stillness the night ends at
constexpr double BrightEvening = 0.50;   ///< the brightness the night begins at
constexpr double BrightDeep = 0.12;      ///< the deep's
constexpr double BrightWave = 0.15;      ///< a wave's crest above it
constexpr double BrightMorning = 0.85;   ///< the brightness the night ends at
constexpr double PhaseShare = 0.08;      ///< falling asleep and the morning, each a share of the night ...
constexpr double PhaseLeast = 30.0;      ///< ... 30 minutes at the least ...
constexpr double PhaseMost = 45.0;       ///< ... and 45 at the most (and a quarter of the night at the most)
constexpr double WaveMinutes = 90.0;     ///< a wave of the middle, about
// The choice.
constexpr double BrightWeight = 0.7;   ///< the brightness's distance from the curve counts this much of the stillness's
constexpr double SoundWeight = 0.3;    ///< the distance in sound from the step before, against the curve's
constexpr double SoundScale = 3.0;     ///< the distance in sound divided by this (about a random pair's)
constexpr double PreferBonus = 0.1;    ///< a preferred preset's cost lowered by this
constexpr double DawnBonus = 0.1;      ///< a dawn preset's at the end of the morning (growing through it)
constexpr double Dice = 0.04;          ///< a die of up to this on every candidate's cost
// The stays (make_journeys.py's nights): short, middle, long -- dwell and glide ranges in seconds.
constexpr double DwellLo[3] = { 180.0, 360.0, 600.0 };   ///< the shortest dwell of each class
constexpr double DwellHi[3] = { 360.0, 600.0, 900.0 };   ///< the longest
constexpr double GlideLo[3] = { 60.0, 90.0, 120.0 };     ///< the shortest glide of each class
constexpr double GlideHi[3] = { 120.0, 150.0, 210.0 };   ///< the longest

/**
 * @brief Half a cosine from a to b.
 * @param[in] a  the value at u = 0
 * @param[in] b  the value at u = 1
 * @param[in] u  0 .. 1 (clamped)
 * @return the eased value
 */
double ease( double a, double b, double u )
{
    u = std::clamp( u, 0.0, 1.0 );
    return a + ( b - a ) * ( 0.5 - 0.5 * std::cos( Pi * u ) );
}

/**
 * @brief The groups of a family.
 * @param[in] family  0 .. Families-1, NightFamilyAll or NightFamilyBank
 * @return a flag per group
 */
std::vector<bool> familyGroups( int family )
{
    std::vector<bool> out( NumPresetGroups, family == NightFamilyBank );
    if ( family == NightFamilyBank ) return out;
    for ( int f = 0; f < Families; ++f )
    {
        if ( family >= 0 && family < Families && f != family ) continue;
        const std::string list = NightFamilies[f][1];
        size_t            start = 0;
        while ( start <= list.size() )
        {
            const size_t      bar = list.find( '|', start );
            const std::string name = list.substr( start, bar == std::string::npos ? std::string::npos : bar - start );
            for ( int g = 0; g < NumPresetGroups; ++g )
                if ( name == PresetGroupNames[g] ) out[static_cast<size_t>( g )] = true;
            if ( bar == std::string::npos ) break;
            start = bar + 1;
        }
    }
    return out;
}

/**
 * @brief Percentiles of values: (rank + 0.5) / n, ties broken by the index.
 * @param[in] v  the values
 * @return one percentile per value
 */
std::vector<double> percentiles( const std::vector<double> &v )
{
    std::vector<size_t> order( v.size() );
    std::iota( order.begin(), order.end(), size_t{ 0 } );
    std::stable_sort( order.begin(), order.end(), [&]( size_t a, size_t b ) { return v[a] < v[b]; } );
    std::vector<double> out( v.size() );
    for ( size_t r = 0; r < order.size(); ++r )
        out[order[r]] = ( static_cast<double>( r ) + 0.5 ) / static_cast<double>( order.size() );
    return out;
}
}   // namespace

int numNightFamilies()
{
    return Families;
}

const char *nightFamilyName( int family )
{
    if ( family == NightFamilyBank ) return "The whole bank";
    if ( family < 0 || family >= Families ) return "All whole-night groups";
    return NightFamilies[family][0];
}

void nightPhases( double hours, double &asleep, double &morning, int &waves )
{
    const double minutes = std::max( hours, 0.25 ) * 60.0;
    const double phase = std::min( std::clamp( PhaseShare * minutes, PhaseLeast, PhaseMost ), 0.25 * minutes );
    asleep = morning = phase / minutes;
    waves = std::max( 1, static_cast<int>( std::lround( ( minutes - 2.0 * phase ) / WaveMinutes ) ) );
}

void nightCurve( double x, double hours, double &still, double &bright )
{
    double asleep = 0.0, morning = 0.0;
    int    waves = 1;
    nightPhases( hours, asleep, morning, waves );
    x = std::clamp( x, 0.0, 1.0 );
    if ( x < asleep )
    {
        still = ease( StillEvening, StillDeep, x / asleep );
        bright = ease( BrightEvening, BrightDeep, x / asleep );
        return;
    }
    const double end = 1.0 - morning;
    if ( x > end )
    {
        still = ease( StillDeep, StillMorning, ( x - end ) / morning );
        bright = ease( BrightDeep, BrightMorning, ( x - end ) / morning );
        return;
    }
    // The middle: waves from the deep and back, mostly near the deep, the later ones a little higher.
    const double u = ( x - asleep ) / std::max( end - asleep, 1e-9 ) * waves;
    const double f = u - std::floor( u );
    const double w = std::pow( 0.5 - 0.5 * std::cos( 2.0 * Pi * f ), 2.0 );
    const double grow = 0.6 + 0.4 * std::clamp( u / waves, 0.0, 1.0 );
    still = StillDeep + grow * StillWave * w;
    bright = BrightDeep + grow * BrightWave * w;
}

bool nightMeasures( int preset, float &still, float &moving )
{
    if ( preset < 0 || preset >= Measured ) return false;
    still = NightMeasures[preset][0];
    moving = NightMeasures[preset][1];
    return true;
}

float nightDistance( int a, int b )
{
    float fa[4], fb[4];
    if ( !presetFeatures( a, fa ) || !presetFeatures( b, fb ) ) return 0.0f;
    float s = 0.0f;
    for ( int k = 0; k < 4; ++k ) s += ( fa[k] - fb[k] ) * ( fa[k] - fb[k] );
    return std::sqrt( s );
}

NightPlan composeNight( const NightRequest &request )
{
    NightPlan    plan;
    const double hours = std::clamp( request.m_hours, 1.0, 14.0 );
    const double total = hours * 3600.0;
    double       asleep = 0.0, morning = 0.0;
    int          waves = 1;
    nightPhases( hours, asleep, morning, waves );

    // The candidates: the family's groups, the dawn's for the morning; none excluded.
    const std::vector<bool> inFamily = familyGroups( request.m_family );
    std::vector<bool>       dawn( NumPresetGroups, false );
    for ( const char *g : NightMorningGroups )
        for ( int k = 0; k < NumPresetGroups; ++k )
            if ( std::strcmp( g, PresetGroupNames[k] ) == 0 ) dawn[static_cast<size_t>( k )] = true;
    const std::unordered_set<std::string> excluded( request.m_exclude.begin(), request.m_exclude.end() );
    const std::unordered_set<std::string> preferred( request.m_prefer.begin(), request.m_prefer.end() );
    struct Candidate
    {
        int    m_preset;         // the bank's index
        double m_still = 0.0;    // stillness, then its percentile
        double m_bright = 0.0;   // brightness, then its percentile
        double m_moving = 0.0;   // its own motion
        double m_dice = 0.0;     // the seed's die
        int    m_class = 1;      // its stay
        bool   m_preferred = false;
        bool   m_dawn = false;          // of a dawn group
        bool   m_morningOnly = false;   // of a dawn group outside the family: the morning's only
        bool   m_used = false;
    };
    std::vector<Candidate> cand;
    const int              n = std::min( numPresets(), Measured );
    for ( int i = 0; i < n; ++i )
    {
        const Preset &p = preset( i );
        const size_t  g = static_cast<size_t>( p.m_group );
        const bool    morningOnly = !inFamily[g] && request.m_dawn && dawn[g];
        if ( ( !inFamily[g] && !morningOnly ) || excluded.count( p.m_name ) ) continue;
        float f[4];
        if ( !presetFeatures( i, f ) ) continue;
        Candidate c{ i };
        c.m_still = NightMeasures[i][0];
        c.m_moving = NightMeasures[i][1];
        c.m_bright = f[0];
        c.m_preferred = preferred.count( p.m_name ) > 0;
        c.m_dawn = request.m_dawn && dawn[g];
        c.m_morningOnly = morningOnly;
        cand.push_back( c );
    }
    plan.m_candidates = static_cast<int>( cand.size() );
    if ( cand.empty() ) return plan;
    {   // percentiles over the candidates; the stays by the thirds of their own motion
        std::vector<double> s( cand.size() ), b( cand.size() ), m( cand.size() );
        for ( size_t k = 0; k < cand.size(); ++k )
        {
            s[k] = cand[k].m_still;
            b[k] = cand[k].m_bright;
            m[k] = cand[k].m_moving;
        }
        const std::vector<double> ps = percentiles( s ), pb = percentiles( b ), pm = percentiles( m );
        std::uint64_t             rng = request.m_seed ? request.m_seed : 0x9E3779B97F4A7C15ull;
        for ( size_t k = 0; k < cand.size(); ++k )
        {
            cand[k].m_still = ps[k];
            cand[k].m_bright = pb[k];
            cand[k].m_class = pm[k] < 1.0 / 3.0 ? 0 : ( pm[k] >= 2.0 / 3.0 ? 2 : 1 );
            rng ^= rng << 13;   // xorshift64, as the journey player's
            rng ^= rng >> 7;
            rng ^= rng << 17;
            cand[k].m_dice = Dice * static_cast<double>( rng >> 11 ) * ( 1.0 / 9007199254740992.0 );
        }
    }

    // The walk: at each step the candidate whose cost is lowest.
    double t = 0.0;
    int    prevClass = -1, prev = -1;
    while ( total - t > 0.5 * 0.5 * ( DwellLo[1] + DwellHi[1] ) )
    {
        int    best = -1, bestClass = 1;
        double bestCost = 1e30;
        for ( size_t k = 0; k < cand.size(); ++k )
        {
            const Candidate &c = cand[k];
            if ( c.m_used ) continue;
            int cls = c.m_class;
            if ( cls == prevClass && cls != 1 ) cls = 1;   // never two short or two long ones in a row
            const double stay = 0.5 * ( DwellLo[cls] + DwellHi[cls] );
            const double x = std::min( 1.0, ( t + 0.5 * stay ) / total );
            const double intoMorning = ( x - ( 1.0 - morning ) ) / morning;   // below 0 before the morning
            if ( c.m_morningOnly && intoMorning < 0.0 ) continue;
            double ts = 0.0, tb = 0.0;
            nightCurve( x, hours, ts, tb );
            double cost = std::hypot( c.m_still - ts, BrightWeight * ( c.m_bright - tb ) ) + c.m_dice;
            if ( prev >= 0 ) cost += SoundWeight * nightDistance( prev, c.m_preset ) / SoundScale;
            if ( c.m_preferred ) cost -= PreferBonus;
            if ( c.m_dawn && intoMorning > 0.0 ) cost -= DawnBonus * std::min( 1.0, intoMorning );
            if ( cost < bestCost )
            {
                bestCost = cost;
                best = static_cast<int>( k );
                bestClass = cls;
            }
        }
        if ( best < 0 ) break;   // every candidate played
        Candidate &c = cand[static_cast<size_t>( best )];
        c.m_used = true;
        NightStep st;
        st.m_preset = c.m_preset;
        st.m_at = t;
        st.m_stay = 0.5 * ( DwellLo[bestClass] + DwellHi[bestClass] );
        nightCurve( std::min( 1.0, ( t + 0.5 * st.m_stay ) / total ), hours, st.m_targetStill, st.m_targetBright );
        st.m_still = c.m_still;
        st.m_bright = c.m_bright;
        st.m_class = bestClass;
        st.m_preferred = c.m_preferred;
        plan.m_steps.push_back( st );
        t += st.m_stay;
        prev = c.m_preset;
        prevClass = bestClass;
    }
    if ( plan.m_steps.empty() ) return plan;

    // Every stay scaled alike, so the night lasts what was asked.
    plan.m_scale = total / t;
    Journey &j = plan.m_journey;
    char     name[64];
    std::snprintf( name, sizeof( name ), "Night %g h", std::round( hours * 4.0 ) / 4.0 );
    j.m_name = request.m_name.empty() ? std::string( name ) : request.m_name;
    j.m_cyclic = false;
    for ( NightStep &st : plan.m_steps )
    {
        st.m_at *= plan.m_scale;
        st.m_stay *= plan.m_scale;
        JourneyStep js;
        js.m_preset = preset( st.m_preset ).m_name;
        js.m_dwellLo = std::round( DwellLo[st.m_class] * plan.m_scale );
        js.m_dwellHi = std::round( DwellHi[st.m_class] * plan.m_scale );
        js.m_fadeLo = GlideLo[st.m_class];
        js.m_fadeHi = GlideHi[st.m_class];
        j.m_steps.push_back( js );
    }
    return plan;
}

}   // namespace additiv
