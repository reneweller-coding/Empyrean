/**
 * @file brain_audit.cpp
 * @brief additiv_brain_audit: what the conductors, the foundation and the figures play, without the card -- an hour
 *        of a preset's notes in a fraction of a second (plan 20.11.2026, A2; Noctuary's ambient_brain_audit).
 *
 *     additiv_brain_audit [--preset NAME|N]... [--presets FILE] [--hours H (0.5)] [--seconds S] [--set key=value]...
 *                         [--seed N] [--dt S] [--param-every K] [--pairs FILE] [--tuning]
 *
 * Prints, per preset, a line `# preset <number> <name>`, a line `A,<attack>,<figure attack>,<figure release>` (seconds,
 * as the preset sets them: what Tools/presets/music_audit.py measures "too short" and "still ringing" against), then
 * one line per note
 *
 *     E,<seconds>,<source>,<on 1|0>,<note>,<velocity>,<Hz>,<figure>,<step>
 *
 * with the source as Engine::M_NoteTrace numbers it (1 and 2 the conductors, 4 the foundation, 5 a figure), the pitch
 * as the tuning plays it, and for a figure's onset which figure (0 .. 3) and which of its notes in the round (-1
 * otherwise), and one line per change of the conductor's root, `R,<seconds>,<root>`; last, `F,<moved>,<left out>`: the
 * figure notes moved off a pitch a figure still sounded, and those left out where nothing near was free
 * (Engine::figureMoved).
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
 *
 * A journey's transition (plan D2, 20.11.2026; Tools/presets/journey_transitions.py): `--pairs FILE` takes lines
 * `<from>|<to>|<at>|<glide>|<seconds>` and audits each on a fresh engine: the first preset from the start, the second
 * loaded at `at` seconds with a glide of `glide` seconds (Engine::loadPreset, as the journey player does), `seconds` in
 * all. Each pair prints `# pair <from> | <to>`, `P,<0|1>,<attack>,<release>` for the two presets as they set them,
 * `G,<at>,<glide>`, the note lines and the root lines as above, and `T,<seconds>,<Hz of key 0>,...,<Hz of key 127>`
 * -- every key's pitch as the tuning gives it (Engine::keyHz) -- at the start and whenever a key's pitch has moved by
 * more than half a cent (looked at every tenth of a second): a glide changes the scale and the root at its middle, and
 * what sounds is retuned. `--tuning` prints, per preset asked for, `# preset ...` and one T line of its own tuning.
 * @author Rene Weller
 */
#include "additiv/Engine.h"
#include "additiv/Params.h"
#include "additiv/Presets.h"

#include <cmath>
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

/**
 * @brief Prints a T line: every key's pitch as the engine's tuning gives it now.
 * @param[in] e  the engine
 * @param[in] t  the audit's clock, seconds
 * @param[out] hz  receives the 128 pitches printed
 */
void printTuning( const Engine &e, double t, double *hz )
{
    std::printf( "T,%.4f", t );
    for ( int k = 0; k < 128; ++k )
    {
        hz[k] = e.keyHz( k );
        std::printf( ",%.4f", hz[k] );
    }
    std::printf( "\n" );
}

/** @brief One journey transition to audit (--pairs): two presets, when the second is loaded, its glide, the length. */
struct Pair
{
    int    m_from = -1;       ///< the preset that plays first
    int    m_to = -1;         ///< the preset glided into
    double m_at = 0.0;        ///< seconds after the start when the glide begins
    double m_glide = 0.0;     ///< the glide's seconds
    double m_seconds = 0.0;   ///< the audit's length
};

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
    std::vector<Pair>        pairs;
    bool                     tuning = false;
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
        else if ( a == "--tuning" )
            tuning = true;
        else if ( a == "--pairs" && has )
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
                if ( line.empty() || line[0] == '#' ) continue;
                std::vector<std::string> f;
                for ( size_t p = 0;; )
                {
                    const size_t bar = line.find( '|', p );
                    f.push_back( line.substr( p, bar == std::string::npos ? std::string::npos : bar - p ) );
                    if ( bar == std::string::npos ) break;
                    p = bar + 1;
                }
                Pair pr;
                if ( f.size() == 5 )
                {
                    pr.m_from = presetIndex( f[0] );
                    pr.m_to = presetIndex( f[1] );
                    pr.m_at = std::atof( f[2].c_str() );
                    pr.m_glide = std::atof( f[3].c_str() );
                    pr.m_seconds = std::atof( f[4].c_str() );
                }
                if ( pr.m_from < 0 || pr.m_to < 0 || pr.m_seconds <= 0.0 )
                {
                    std::fprintf( stderr, "additiv_brain_audit: bad pair line %s\n", line.c_str() );
                    return 2;
                }
                pairs.push_back( pr );
            }
        }
        else
        {
            std::fprintf( stderr, "additiv_brain_audit [--preset NAME|N]... [--presets FILE] [--hours H] [--seconds S] "
                                  "[--set key=value]... [--seed N] [--dt S] [--param-every K] [--pairs FILE] "
                                  "[--tuning]\n" );
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
    auto noteLine = []( const Engine::AuditNote &n )
    {
        std::printf( "E,%.4f,%d,%d,%d,%.3f,%.3f,%d,%d\n", n.m_seconds, n.m_source, n.m_on ? 1 : 0, n.m_note,
                     static_cast<double>( n.m_velocity ), n.m_hz, n.m_figure, n.m_step );
    };
    auto rootLine = []( double t, int root ) { std::printf( "R,%.4f,%d\n", t, root ); };
    for ( const Pair &pr : pairs )
    {   // a journey's transition: the first preset, the second glided into at m_at (as JourneyPlayer and empy_render)
        auto e = std::make_unique<Engine>();
        e->setRealtime( false );
        std::printf( "# pair %s | %s\n", preset( pr.m_from ).m_name, preset( pr.m_to ).m_name );
        for ( int k = 0; k < 2; ++k )
        {
            e->loadPreset( k == 0 ? pr.m_to : pr.m_from );   // read the second's envelope, then stand on the first
            std::printf( "P,%d,%.4f,%.4f\n", 1 - k, static_cast<double>( e->param( ParamIdE::ATTACK ) ),
                         static_cast<double>( e->param( ParamIdE::RELEASE ) ) );
        }
        e->setParam( ParamIdE::BRAIN_ON, 1.0f );   // a journey keeps the conductor playing
        std::printf( "G,%.4f,%.4f\n", pr.m_at, pr.m_glide );
        double hz[128] = {}, lastLook = -1.0;
        bool   loaded = false, first = true;
        Engine *eng = e.get();
        e->auditConductor( pr.m_seconds, config, dt, paramEvery, noteLine, rootLine,
                           [&]( double t )
                           {
                               if ( !loaded && t >= pr.m_at )
                               {
                                   eng->loadPreset( pr.m_to, std::max( pr.m_glide, 0.5 ) );
                                   eng->setParam( ParamIdE::BRAIN_ON, 1.0f );
                                   loaded = true;
                               }
                               if ( t - lastLook < 0.1 ) return;
                               lastLook = t;
                               bool moved = first;
                               for ( int k = 0; k < 128 && !moved; ++k )
                                   moved = std::fabs( 1200.0 * std::log2( eng->keyHz( k ) / hz[k] ) ) > 0.5;
                               if ( moved ) printTuning( *eng, t, hz );
                               first = false;
                           } );
    }
    if ( !pairs.empty() )
    {
        std::fflush( stdout );
        return 0;
    }
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
        if ( tuning )
        {   // one step sets the scale (the Timbre scale from the spectrum), then the keys' pitches
            double hz[128];
            e->auditConductor( dt, config, dt, paramEvery, []( const Engine::AuditNote & ) {}, nullptr );
            printTuning( *e, 0.0, hz );
            continue;
        }
        std::printf( "A,%.4f,%.4f,%.4f\n", static_cast<double>( e->param( ParamIdE::ATTACK ) ),
                     static_cast<double>( e->param( ParamIdE::FIGURE_ATTACK ) ),
                     static_cast<double>( e->param( ParamIdE::FIGURE_RELEASE ) ) );
        e->auditConductor( seconds, config, dt, paramEvery, noteLine, rootLine );
        int       left = 0;
        const int moved = e->figureMoved( left );
        std::printf( "F,%d,%d\n", moved, left );
    }
    std::fflush( stdout );
    return 0;
}
