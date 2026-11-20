/**
 * @file night.cpp
 * @brief additiv_night: the night director's night as a table (plan 20.11.2026, D3) -- what Tools/presets/night_plot.py
 *        prints and draws; no card, no sound.
 *
 *     additiv_night [--hours H (8)] [--seed N (1)] [--family N|NAME] [--prefer FILE] [--exclude FILE] [--no-dawn]
 *                   [--journey] [--families]
 *
 * Prints a line `# night <hours> <seed> <family> <candidates> <scale>`, a line `# phases <asleep> <morning> <waves>`
 * (shares of the night), a header and one tab-separated line per step: its number, where it begins (seconds), how long
 * it stays (seconds, at the middle of its range), the preset, its group, its stay class (0 short, 1 middle, 2 long),
 * its stillness and brightness as percentiles of the candidates, the curve's two targets at the middle of its stay,
 * whether it is preferred, its distance in sound from the step before, and its measured motion and brightness (in
 * spreads of the bank). Then `# whole_nights <mean neighbour distance>`: the shipped Whole Nights' steps, measured
 * the same way, and 201 lines `# curve <x> <stillness> <brightness>`: the curve over the night (x 0 .. 1).
 * `--journey` prints the journey file instead; `--families` the families to choose from.
 * --prefer and --exclude read preset names, one a line.
 * @author Rene Weller
 */
#include "additiv/Find.h"
#include "additiv/Journey.h"
#include "additiv/NightDirector.h"
#include "additiv/Presets.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace additiv;

namespace
{

/**
 * @brief The lines of a file, trimmed, empty ones left out.
 * @param[in]  path  the file
 * @param[out] out   receives them
 * @return false when it cannot be read
 */
bool readNames( const char *path, std::vector<std::string> &out )
{
    std::ifstream f( path );
    if ( !f ) return false;
    std::string line;
    while ( std::getline( f, line ) )
    {
        line = Journey::trim( line );
        if ( !line.empty() ) out.push_back( line );
    }
    return true;
}

/**
 * @brief The mean distance in sound between neighbouring steps of the shipped Whole Nights.
 * @return the mean (nightDistance), 0 when there are none
 */
double wholeNightsDistance()
{
    double sum = 0.0;
    int    count = 0;
    for ( int i = 0; i < numBuiltinJourneys(); ++i )
    {
        Journey j;
        if ( !j.parse( builtinJourneyText( i ) ) || j.m_name.rfind( "Whole Night", 0 ) != 0 ) continue;
        for ( size_t k = 1; k < j.m_steps.size(); ++k )
        {
            sum += nightDistance( findPreset( j.m_steps[k - 1].m_preset.c_str() ),
                                  findPreset( j.m_steps[k].m_preset.c_str() ) );
            ++count;
        }
    }
    return count ? sum / count : 0.0;
}

}   // namespace

/**
 * @brief Reads the arguments, composes the night and prints it.
 * @param[in] argc  the argument count
 * @param[in] argv  the arguments
 * @return 0, 2 on a bad argument
 */
int main( int argc, char **argv )
{
    NightRequest r;
    bool         journey = false;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string a = argv[i];
        const bool        has = i + 1 < argc;
        if ( a == "--hours" && has )
            r.m_hours = std::atof( argv[++i] );
        else if ( a == "--seed" && has )
            r.m_seed = std::strtoull( argv[++i], nullptr, 10 );
        else if ( a == "--family" && has )
        {
            const std::string f = argv[++i];
            r.m_family = NightFamilyAll;
            if ( f == "bank" ) r.m_family = NightFamilyBank;
            for ( int k = 0; k < numNightFamilies(); ++k )
                if ( f == nightFamilyName( k ) || f == std::to_string( k ) ) r.m_family = k;
        }
        else if ( ( a == "--prefer" || a == "--exclude" ) && has )
        {
            if ( !readNames( argv[++i], a == "--prefer" ? r.m_prefer : r.m_exclude ) )
            {
                std::fprintf( stderr, "additiv_night: cannot read %s\n", argv[i] );
                return 2;
            }
        }
        else if ( a == "--no-dawn" )
            r.m_dawn = false;
        else if ( a == "--journey" )
            journey = true;
        else if ( a == "--families" )
        {
            for ( int k = 0; k < numNightFamilies(); ++k ) std::printf( "%d\t%s\n", k, nightFamilyName( k ) );
            std::printf( "all\t%s\nbank\t%s\n", nightFamilyName( NightFamilyAll ), nightFamilyName( NightFamilyBank ) );
            return 0;
        }
        else
        {
            std::fprintf( stderr, "additiv_night [--hours H] [--seed N] [--family N|NAME|bank] [--prefer FILE] "
                                  "[--exclude FILE] [--no-dawn] [--journey] [--families]\n" );
            return 2;
        }
    }
    const NightPlan plan = composeNight( r );
    if ( journey )
    {
        std::printf( "%s", plan.m_journey.text().c_str() );
        return 0;
    }
    double asleep = 0.0, morning = 0.0;
    int    waves = 1;
    nightPhases( r.m_hours, asleep, morning, waves );
    std::printf( "# night %g %llu %s %d %.4f\n", r.m_hours, static_cast<unsigned long long>( r.m_seed ),
                 nightFamilyName( r.m_family ), plan.m_candidates, plan.m_scale );
    std::printf( "# phases %.4f %.4f %d\n", asleep, morning, waves );
    std::printf(
        "step\tat\tstay\tpreset\tgroup\tclass\tstill\tbright\ttarget_still\ttarget_bright\tpreferred\tdistance\t"
        "motion_z\tbright_z\n" );
    for ( size_t k = 0; k < plan.m_steps.size(); ++k )
    {
        const NightStep &s = plan.m_steps[k];
        float            f[4] = {};
        presetFeatures( s.m_preset, f );
        std::printf( "%zu\t%.0f\t%.0f\t%s\t%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\n", k + 1, s.m_at,
                     s.m_stay, preset( s.m_preset ).m_name, PresetGroupNames[preset( s.m_preset ).m_group], s.m_class,
                     s.m_still, s.m_bright, s.m_targetStill, s.m_targetBright, s.m_preferred ? 1 : 0,
                     k ? static_cast<double>( nightDistance( plan.m_steps[k - 1].m_preset, s.m_preset ) ) : 0.0,
                     static_cast<double>( f[1] ), static_cast<double>( f[0] ) );
    }
    std::printf( "# whole_nights %.4f\n", wholeNightsDistance() );
    for ( int k = 0; k <= 200; ++k )   // the curve itself, for the plot
    {
        double still = 0.0, bright = 0.0;
        nightCurve( k / 200.0, r.m_hours, still, bright );
        std::printf( "# curve %.4f %.4f %.4f\n", k / 200.0, still, bright );
    }
    return 0;
}
