/**
 * @file brain_audit.cpp
 * @brief additiv_brain_audit: what the conductors, the foundation and the figures play, without the card -- an hour
 *        of a preset's notes in a fraction of a second (plan 20.11.2026, A2; Noctuary's ambient_brain_audit).
 *
 *     additiv_brain_audit [--preset NAME|N]... [--presets FILE] [--hours H (0.5)] [--seconds S] [--set key=value]...
 *                         [--seed N] [--dt S] [--param-every K]
 *
 * Prints, per preset, a line `# preset <number> <name>`, a line `A,<attack>,<figure attack>` (seconds, as the
 * preset sets them: what Tools/presets/music_audit.py measures "too short" against), then one line per note
 *
 *     E,<seconds>,<source>,<on 1|0>,<note>,<velocity>,<Hz>,<figure>,<step>
 *
 * with the source as Engine::M_NoteTrace numbers it (1 and 2 the conductors, 4 the foundation, 5 a figure), the pitch
 * as the tuning plays it, and for a figure's onset which figure (0 .. 3) and which of its notes in the round (-1
 * otherwise), and one line per change of the conductor's root, `R,<seconds>,<root>`.
 *
 * No audio is rendered and the GPU is never opened: Engine::auditConductor steps the engine's own conductor code with
 * the parameters as a render reads them, seeded as a render seeds them. With the default `--dt` (one GPU block,
 * 128 / 48000 s) the notes are the ones `empy_render --preset X --note-log` logs with the same seed; what it cannot
 * see is listed at Engine::auditConductor: presets that use Follow Light over a living field (brain_light with
 * life_mode) are audited without the field, a modulation route from what the instrument hears reads silence, and
 * Night Arc has no night (a render follows the real sun at the hour it runs). Checked against the render on ten
 * presets of every kind, five minutes each (20.11.2026): 1012 notes of 1012 the same, at the same sample.
 *
 * The parameters are read again only when the matrix moves one the notes hear (`--param-every 0` reads them every
 * step whatever the matrix does, the reference; 40 presets x 10 minutes gave the same bytes). `--dt 0.0213
 * --param-every 4` (eight blocks a step, a modulated knob read every fourth) is about eight times faster and no
 * longer note for note: what Tools/presets/music_audit.py uses over a whole bank.
 *
 * `--presets FILE` audits every preset in the file (a name or a number per line) in turn, each on a fresh engine:
 * one process for a whole bank, as Noctuary found a process per preset spends its time starting.
 * @author Rene Weller
 */
#include "additiv/Engine.h"
#include "additiv/Params.h"
#include "additiv/Presets.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace additiv;

namespace
{

/**
 * @brief Applies one --set key=value: a number, or for a choice also its name (as empy_render).
 * @param[in,out] e    the engine
 * @param[in]     arg  "key=value"
 * @return true when the key exists and the value was read
 */
bool applySet( Engine &e, const std::string &arg )
{
    const size_t eq = arg.find( '=' );
    if ( eq == std::string::npos ) return false;
    const std::string key = arg.substr( 0, eq ), val = arg.substr( eq + 1 );
    const ParamDesc  *d = findParam( key.c_str() );
    if ( !d ) return false;
    if ( d->m_kind == ParamKindE::CHOICE )
        for ( int i = 0; i < d->m_numChoices; ++i )
            if ( val == d->m_choices[i] )
            {
                e.setParam( d->m_id, static_cast<float>( i ) );
                return true;
            }
    if ( d->m_kind == ParamKindE::BOOL && ( val == "on" || val == "off" ) )
    {
        e.setParam( d->m_id, val == "on" ? 1.0f : 0.0f );
        return true;
    }
    char       *end = nullptr;
    const float x = std::strtof( val.c_str(), &end );
    if ( end == val.c_str() ) return false;
    e.setParam( d->m_id, x );
    return true;
}

/**
 * @brief A preset by its name or its number.
 * @param[in] what  the name, or 0 .. numPresets() - 1
 * @return its index, -1 when there is none
 */
int presetIndex( const std::string &what )
{
    int index = findPreset( what.c_str() );
    if ( index < 0 && !what.empty() && what.find_first_not_of( "0123456789" ) == std::string::npos )
        index = std::atoi( what.c_str() );
    return index >= 0 && index < numPresets() ? index : -1;
}

}   // namespace

/**
 * @brief Reads the options and audits each preset asked for (the defaults when none), printing the lines the file
 *        header describes to stdout.
 * @param[in] argc  the arguments' count
 * @param[in] argv  the options in the file header
 * @return 0 when it ran, 2 for an unknown option, preset or parameter, or a file it cannot read
 */
int main( int argc, char **argv )
{
    double                   seconds = 1800.0, dt = 0.0;
    int                      paramEvery = 1;
    std::uint64_t            seed = EngineConfig{}.m_seed;
    std::vector<std::string> names, sets;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string a = argv[i];
        const bool        has = i + 1 < argc;
        if ( a == "--preset" && has )
            names.push_back( argv[++i] );
        else if ( a == "--presets" && has )
        {
            std::ifstream in( argv[++i] );
            if ( !in )
            {
                std::fprintf( stderr, "additiv_brain_audit: cannot read %s\n", argv[i] );
                return 2;
            }
            for ( std::string line; std::getline( in, line ); )
            {
                while ( !line.empty() && ( line.back() == '\r' || line.back() == '\n' ) ) line.pop_back();
                if ( !line.empty() ) names.push_back( line );
            }
        }
        else if ( a == "--hours" && has )
            seconds = 3600.0 * std::atof( argv[++i] );
        else if ( a == "--seconds" && has )
            seconds = std::atof( argv[++i] );
        else if ( a == "--dt" && has )
            dt = std::atof( argv[++i] );
        else if ( a == "--param-every" && has )
            paramEvery = std::max( 0, std::atoi( argv[++i] ) );
        else if ( a == "--seed" && has )
            seed = std::strtoull( argv[++i], nullptr, 10 );
        else if ( a == "--set" && has )
            sets.push_back( argv[++i] );
        else
        {
            std::fprintf( stderr, "additiv_brain_audit [--preset NAME|N]... [--presets FILE] [--hours H] [--seconds S] "
                                  "[--set key=value]... [--seed N] [--dt S] [--param-every K]\n" );
            return 2;
        }
    }
    EngineConfig config;
    config.m_sampleRate = DefaultSampleRate;
    config.m_realtimeGpuClass = false;
    config.m_seed = seed;
    if ( dt <= 0.0 ) dt = config.m_gpuBlock / config.m_sampleRate;   // one GPU block: as the render steps
    if ( seconds <= 0.0 )
    {
        std::fprintf( stderr, "additiv_brain_audit: --hours and --seconds must be positive\n" );
        return 2;
    }
    std::vector<int> indices;
    for ( const std::string &n : names )
    {
        const int index = presetIndex( n );
        if ( index < 0 )
        {
            std::fprintf( stderr, "additiv_brain_audit: no preset %s\n", n.c_str() );
            return 2;
        }
        indices.push_back( index );
    }
    if ( indices.empty() ) indices.push_back( -1 );   // none: the defaults
    std::vector<char> out( 1 << 20 );
    std::setvbuf( stdout, out.data(), _IOFBF, out.size() );
    for ( const int index : indices )
    {
        auto e = std::make_unique<Engine>();   // a fresh one each: an Engine is large, the heap and not the stack
        e->setRealtime( false );
        if ( index >= 0 ) e->loadPreset( index );
        for ( const std::string &s : sets )
            if ( !applySet( *e, s ) )
            {
                std::fprintf( stderr, "additiv_brain_audit: cannot apply --set %s\n", s.c_str() );
                return 2;
            }
        std::printf( "# preset %d %s\n", index, index >= 0 ? preset( index ).m_name : "(defaults)" );
        std::printf( "A,%.4f,%.4f\n", static_cast<double>( e->param( ParamIdE::ATTACK ) ),
                     static_cast<double>( e->param( ParamIdE::FIGURE_ATTACK ) ) );
        e->auditConductor(
            seconds, config, dt, paramEvery,
            []( const Engine::AuditNote &n )
            {
                std::printf( "E,%.4f,%d,%d,%d,%.3f,%.3f,%d,%d\n", n.m_seconds, n.m_source, n.m_on ? 1 : 0, n.m_note,
                             static_cast<double>( n.m_velocity ), n.m_hz, n.m_figure, n.m_step );
            },
            []( double t, int root ) { std::printf( "R,%.4f,%d\n", t, root ); } );
    }
    std::fflush( stdout );
    return 0;
}
