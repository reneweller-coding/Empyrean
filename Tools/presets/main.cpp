/**
 * @file main.cpp
 * @brief additiv_presets: measures every factory preset's level, so make_presets.py can bring the
 *        bank to one level (Noctuary's level matching, done once for the bank).
 *
  *     additiv_presets [--from N] [--to M] [--names FILE] [--seconds S (14)] [--out FILE (work/presets/levels.json)]
 *                     [--describe FILE] [--sensitivity FILE] [--register FILE [--step K] [--set "key=value;..."]]
 *                     [--evaluate FILE --out FILE] [--jobs N]
 *
 * --evaluate measures any sounds as the presets are measured (Phase 31, for Tools/presets/grow.py): a line a sound,
 * its knobs as "key=value key=value ..." (the others at their defaults), a '|' and its matrix (Modulation.h's text
 * form); --out receives a JSON list, a row a line in their order: the loudness (LUFS) and the four descriptors
 * (brightness, motion, width, noise), or null where one failed.
 *
 * --register lets every K-th preset (or the named ones) play as its conductor plays it, alone, for
 * the seconds asked (240 by default) -- the chord arriving within seconds, as a preset chosen on the
 * panel does -- and writes per preset where the conductor's notes stand: the
 * lowest note's median, the share of the time it is at B2 or below, the share it is the root, how
 * many notes sound, the share of the power below 150 Hz -- whether the music has a floor -- and the
 * level (dBFS RMS).
 * --set changes knobs after the preset's (to hear a change against the bank as it is).
 *
 * --names measures only the presets a file names, one a line (those a change of the bank touched);
 * what levels.json held for the others stays.
 *
 * --sensitivity measures, for each named preset, how its five measures change when each knob that
 * shapes the sound moves (Tools/presets/make_nudges.py makes the nudges and the focus from it).
 *
 * --describe also writes what each preset sounds like over its last eight seconds, for the preset
 * map (Tools/presets/make_map.py): the brightness (the spectral centroid in octaves above 40 Hz),
 * the motion (how much the spectrum changes over a quarter of a second, a second and three seconds,
 * as a share of itself), the width (1 - the correlation of left and right) and the noisiness (the
 * spectral flatness of the bands). The spectrum is the power of left and right added, never of
 * their sum: decorrelated partials cancel in L+R and would fake motion.
 *
 * Each preset plays the same chord (D3 A3 D4 F#4) with the conductor off, its attack shortened to
 * at most half a second and the master at 0 dB; the level is the RMS of its loudest second.
 * @author Rene Weller
 */
#include "additiv/Describe.h"
#include "additiv/Engine.h"
#include "additiv/Focus.h"
#include "additiv/Presets.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace additiv;

namespace
{
/** @brief Noon in Berlin on the June solstice 2026 (Unix seconds): the clock the presets are measured by. */
constexpr double DaySun = 1782040080.0;

/** @brief The repository root: the first folder above the program that holds CMakeLists.txt and Core. @return the root */
fs::path repoRoot()
{
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW( nullptr, buf, MAX_PATH );
    fs::path p = fs::path( buf ).parent_path();
    for ( int up = 0; up < 5 && !p.empty(); ++up )
    {
        if ( fs::exists( p / "CMakeLists.txt" ) && fs::exists( p / "Core" ) ) return p;
        p = p.parent_path();
    }
    return fs::current_path();
}

/**
 * @brief A sound's loudness as a preset's is measured (see the file comment): the chord with the conductor off, the
 *        attack at most half a second, the master at 0 dB; and what it sounds like when asked.
 * @param[in]  values   every knob (NumParams)
 * @param[in]  mod      its matrix (Modulation.h's text form)
 * @param[in]  seconds  how long it plays
 * @param[out] error    receives why the engine failed
 * @param[out] desc     receives what it sounds like (null: not asked)
 * @return LUFS of the loudest three seconds; NaN on failure
 */
double measureSound( const float *values, const std::string &mod, double seconds, std::string *error,
                     Description *desc )
{
    Engine e;
    e.setRealtime( false );
    for ( int i = 0; i < NumParams; ++i ) e.setParam( static_cast<ParamIdE>( i ), values[i] );
    e.setModText( mod );
    e.setParam( ParamIdE::BRAIN_ON, 0.0f );
    e.setParam( ParamIdE::ATTACK, std::min( values[static_cast<int>( ParamIdE::ATTACK )], 0.5f ) );
    e.setParam( ParamIdE::MASTER_GAIN, 0.0f );
    EngineConfig c;
    c.m_maxHostBlock = 512;
    c.m_realtimeGpuClass = false;
    // By day (Phase 30): the night arc moves nothing then, so a preset measures the same at any hour.
    c.m_sunClock = DaySun;
    e.setLocation( 52.52, 13.40 );
    if ( !e.prepare( c, error ) ) return std::nan( "" );
    for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
    // The loudest three seconds, K-weighted (ITU-R BS.1770's short-term loudness, LUFS): a pad's sustain, a bell's
    // strike -- what a listener levels by. (The last eight seconds measured the bells' and the glass's decay, 20 dB
    // under their strike. The loudest second's RMS weighed a wave room's bass as much as a rain's glints, which the ear
    // does not: so the loudness's weighting, 16.10.2026.)
    const int          n = 512;
    const int          blocks = static_cast<int>( seconds * c.m_sampleRate / n );
    const int          perSecond = static_cast<int>( c.m_sampleRate / n );
    std::vector<float> L( n ), R( n );
    double             sq = 0.0, loudest = 0.0;
    int                inWindow = 0;
    // The K-weighting at 48 kHz (BS.1770-4: the head's shelf, then the RLB high pass), a pair of biquads a channel.
    struct Biquad
    {
        double b0, b1, b2, a1, a2, z1 = 0.0, z2 = 0.0;
        double run( double x )
        {
            const double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };
    Biquad shelf[2], rlb[2];
    for ( int ch = 0; ch < 2; ++ch )
    {
        shelf[ch] = { 1.53512485958697, -2.69169618940638, 1.19839281085285, -1.69065929318241, 0.73248077421585 };
        rlb[ch] = { 1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621 };
    }
    std::vector<double> seconds3;   // per second the K-weighted power, both channels summed
    // The last eight seconds kept for the description.
    const int          keepFrom = std::max( 0, blocks - static_cast<int>( 8.0 * c.m_sampleRate / n ) );
    std::vector<float> keptL, keptR;
    for ( int b = 0; b < blocks; ++b )
    {
        e.process( L.data(), R.data(), n );
        if ( desc != nullptr && b >= keepFrom )
        {
            keptL.insert( keptL.end(), L.begin(), L.end() );
            keptR.insert( keptR.end(), R.begin(), R.end() );
        }
        for ( int i = 0; i < n; ++i )
        {
            const double l = rlb[0].run( shelf[0].run( L[i] ) ), r = rlb[1].run( shelf[1].run( R[i] ) );
            sq += l * l + r * r;
        }
        if ( ++inWindow == perSecond )
        {
            seconds3.push_back( sq / ( static_cast<double>( perSecond ) * n ) );
            sq = 0.0;
            inWindow = 0;
            if ( seconds3.size() >= 3 )
            {
                const size_t k = seconds3.size();
                loudest = std::max( loudest, ( seconds3[k - 1] + seconds3[k - 2] + seconds3[k - 3] ) / 3.0 );
            }
        }
    }
    if ( desc != nullptr ) *desc = describe( keptL, keptR, c.m_sampleRate );
    return -0.691 + 10.0 * std::log10( std::max( loudest, 1e-20 ) );
}

/**
 * @brief A factory preset's loudness (measureSound), with knobs set after its own.
 * @param[in]  index      the preset
 * @param[in]  seconds    how long it plays
 * @param[out] error      receives why the engine failed
 * @param[out] desc       receives what it sounds like (null: not asked)
 * @param[in]  overrides  knobs set after the preset's (the sensitivity mode)
 * @return LUFS of the loudest three seconds; NaN on failure
 */
double measure( int index, double seconds, std::string *error, Description *desc,
                const std::vector<std::pair<ParamIdE, float>> &overrides = {} )
{
    float       values[NumParams];
    std::string mod;
    presetValues( index, values, &mod );
    for ( const auto &[id, v] : overrides ) values[static_cast<int>( id )] = v;
    return measureSound( values, mod, seconds, error, desc );
}

/** @brief The knobs the sensitivity mode nudges: those that shape the sound and that a preset sets. */
const ParamIdE Sensed[] = {
    ParamIdE::TILT,          ParamIdE::BRIGHTNESS,   ParamIdE::ODD_EVEN,       ParamIdE::INHARMONIC,
    ParamIdE::CLOUD_WIDTH,   ParamIdE::WIDTH_SLOPE,  ParamIdE::DRIFT,          ParamIdE::DRIFT_RATE,
    ParamIdE::SHIMMER,       ParamIdE::SHIMMER_RATE, ParamIdE::STEREO,         ParamIdE::AIR,
    ParamIdE::AIR_CENTER,    ParamIdE::AIR_WIDTH,    ParamIdE::AIR_JITTER,     ParamIdE::TURBULENCE,
    ParamIdE::MOTION_RATE,   ParamIdE::MOTION_DEPTH, ParamIdE::FIELD_STRENGTH, ParamIdE::DISPERSION,
    ParamIdE::TIME_BEND,     ParamIdE::LIFE_DEPTH,   ParamIdE::LIFE_RATE,      ParamIdE::SYNC_DEPTH,
    ParamIdE::SYNC_COUPLING, ParamIdE::SYNC_RATE,    ParamIdE::RIPPLE_DEPTH,   ParamIdE::CUTOFF,
    ParamIdE::RESONANCE,     ParamIdE::FILTER_ENV,   ParamIdE::COMB_DEPTH,     ParamIdE::COMB_SWEEP,
    ParamIdE::PRESENCE,      ParamIdE::PAD_LOW_CUT,  ParamIdE::FAR_LEVEL,      ParamIdE::DEPTH,
    ParamIdE::BREATH,        ParamIdE::STEREO_WIDTH, ParamIdE::DELAY_MIX,      ParamIdE::MODAL_LEVEL,
    ParamIdE::SPEC_MIX,      ParamIdE::MELT_MIX,     ParamIdE::MEMORY_LEVEL,   ParamIdE::EARLY_LEVEL,
    ParamIdE::NEAR_MIX,      ParamIdE::PURITY,       ParamIdE::TUNE_GRAVITY,
};

/**
 * @brief The sensitivity mode: per preset its measure, then each knob of Sensed moved by 0.15 of its
 *        travel (down when it stands high) and the preset measured again; per knob the change of the
 *        five measures (brightness, motion, width, noise, level) for a whole travel, one JSON line each.
 * @param[in] which    the presets
 * @param[in] seconds  how long each plays
 * @param[in] out      the file
 * @return 0 on success
 */
int sensitivity( const std::vector<int> &which, double seconds, const fs::path &out )
{
    std::ofstream f( out );
    const auto    t0 = std::chrono::steady_clock::now();
    for ( size_t k = 0; k < which.size(); ++k )
    {
        const int    i = which[k];
        std::string  error;
        Description  d0;
        const double l0 = measure( i, seconds, &error, &d0 );
        if ( std::isnan( l0 ) )
        {
            std::fprintf( stderr, "additiv_presets: %s\n", error.c_str() );
            return 1;
        }
        float values[NumParams];
        presetValues( i, values, nullptr );
        f << "{\"preset\": \"" << preset( i ).m_name << "\", \"group\": " << preset( i ).m_group << ", \"base\": ["
          << d0.m_brightness << ", " << d0.m_motion << ", " << d0.m_width << ", " << d0.m_noise << ", " << l0
          << "], \"d\": {";
        bool first = true;
        for ( ParamIdE id : Sensed )
        {
            const ParamDesc &pd = paramDesc( id );
            const float      n0 = toNorm( pd, values[static_cast<int>( id )] );
            const float      step = n0 <= 0.85f ? 0.15f : -0.15f;
            Description      d1;
            const double     l1 = measure( i, seconds, &error, &d1, { { id, fromNorm( pd, n0 + step ) } } );
            if ( std::isnan( l1 ) )
            {
                std::fprintf( stderr, "additiv_presets: %s\n", error.c_str() );
                return 1;
            }
            f << ( first ? "" : ", " ) << "\"" << pd.m_key << "\": [" << ( d1.m_brightness - d0.m_brightness ) / step
              << ", " << ( d1.m_motion - d0.m_motion ) / step << ", " << ( d1.m_width - d0.m_width ) / step << ", "
              << ( d1.m_noise - d0.m_noise ) / step << ", " << ( l1 - l0 ) / step << ", " << n0 << "]";
            first = false;
        }
        f << "}}\n";
        f.flush();
        const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
        std::printf( "  %2zu / %zu  %-24s %5.0f s\n", k + 1, which.size(), preset( i ).m_name, s );
        std::fflush( stdout );
    }
    return 0;
}

/**
 * @brief The nudge check: around each preset every nudge (four measures, both ways, a quarter of a
 *        spread), the change the sensitivities predict against the change measured, and the level's.
 * @param[in] which    the presets
 * @param[in] seconds  how long each plays
 * @return 0 on success
 */
int nudgeCheck( const std::vector<int> &which, double seconds )
{
    static const char *const Names[4] = { "brightness", "motion", "width", "noise" };
    for ( int i : which )
    {
        std::string  error;
        Description  d0;
        const double l0 = measure( i, seconds, &error, &d0 );
        if ( std::isnan( l0 ) )
        {
            std::fprintf( stderr, "additiv_presets: %s\n", error.c_str() );
            return 1;
        }
        float values[NumParams], out[NumParams];
        presetValues( i, values, nullptr );
        std::printf( "%s\n", preset( i ).m_name );
        for ( int a = 0; a < NudgeAxes; ++a )
            for ( float sign : { -1.0f, 1.0f } )
            {
                const float                             want = nudge( values, a, sign, 0.25f, out );
                std::vector<std::pair<ParamIdE, float>> set;
                for ( int k = 0; k < NumParams; ++k )
                    if ( out[k] != values[k] ) set.emplace_back( static_cast<ParamIdE>( k ), out[k] );
                Description  d1;
                const double l1 = measure( i, seconds, &error, &d1, set );
                const double got[4] = { d1.m_brightness - d0.m_brightness, d1.m_motion - d0.m_motion,
                                        d1.m_width - d0.m_width, d1.m_noise - d0.m_noise };
                std::printf( "  %-10s %+.0f  predicted %+.2f  measured %+.2f spreads  level %+.1f dB  (%zu knobs)\n",
                             Names[a], sign, want, got[a] / measureSpread( static_cast<MeasureE>( a ) ), l1 - l0,
                             set.size() );
                std::fflush( stdout );
            }
    }
    return 0;
}

/**
 * @brief The register mode (see the file comment): one JSON line per preset.
 * @param[in] which      the presets
 * @param[in] seconds    how long each plays
 * @param[in] out        the file
 * @param[in] overrides  knobs set after the preset's
 * @return 0 on success
 */
int registerMode( const std::vector<int> &which, double seconds, const fs::path &out,
                  const std::vector<std::pair<ParamIdE, float>> &overrides )
{
    std::ofstream    f( out );
    const auto       t0 = std::chrono::steady_clock::now();
    double           sumLowBass = 0.0, sumRoot = 0.0, sumBand = 0.0;
    std::vector<int> medians;
    for ( size_t k = 0; k < which.size(); ++k )
    {
        const int index = which[k];
        Engine    e;
        e.setRealtime( false );
        float       values[NumParams];
        std::string mod, error;
        presetValues( index, values, &mod );
        for ( const auto &[id, v] : overrides ) values[static_cast<int>( id )] = v;
        for ( int i = 0; i < NumParams; ++i ) e.setParam( static_cast<ParamIdE>( i ), values[i] );
        e.setModText( mod );
        e.setParam( ParamIdE::BRAIN_ON, 1.0f );
        e.setParam( ParamIdE::MASTER_GAIN, 0.0f );
        EngineConfig c;
        c.m_maxHostBlock = 2048;
        c.m_realtimeGpuClass = false;
        c.m_sunClock = DaySun;   // by day, as the levels are measured (Phase 30)
        e.setLocation( 52.52, 13.40 );
        if ( !e.prepare( c, &error ) )
        {
            std::fprintf( stderr, "additiv_presets: %s\n", error.c_str() );
            return 1;
        }
        e.requestFill();   // as a preset chosen on the panel: the chord within seconds
        const int          n = 2048, blocks = static_cast<int>( seconds * c.m_sampleRate / n ),
                           warm = static_cast<int>( 20.0 * c.m_sampleRate / n );
        std::vector<float> L( n ), R( n );
        // The power below 150 Hz: a second-order low pass on each side (RBJ, Q 0.707).
        const double w0 = 2.0 * 3.14159265358979 * 150.0 / c.m_sampleRate, alpha = std::sin( w0 ) / ( 2.0 * 0.7071 ),
                     cw = std::cos( w0 );
        const double a0 = 1.0 + alpha, b0 = ( 1.0 - cw ) / 2.0 / a0, b1 = ( 1.0 - cw ) / a0, b2 = b0,
                     a1 = -2.0 * cw / a0, a2 = ( 1.0 - alpha ) / a0;
        double       zl[2] = { 0.0, 0.0 }, zr[2] = { 0.0, 0.0 };
        double       low = 0.0, all = 0.0;
        std::vector<int> lowest;
        int              bassTicks = 0, rootTicks = 0, ticks = 0, notesSum = 0;
        for ( int b = 0; b < blocks; ++b )
        {
            e.process( L.data(), R.data(), n );
            for ( int i = 0; i < n; ++i )
            {
                const double xl = L[static_cast<size_t>( i )], xr = R[static_cast<size_t>( i )];
                const double yl = b0 * xl + zl[0];
                zl[0] = b1 * xl - a1 * yl + zl[1];
                zl[1] = b2 * xl - a2 * yl;
                const double yr = b0 * xr + zr[0];
                zr[0] = b1 * xr - a1 * yr + zr[1];
                zr[1] = b2 * xr - a2 * yr;
                if ( b >= warm )
                {
                    low += yl * yl + yr * yr;
                    all += xl * xl + xr * xr;
                }
            }
            if ( b < warm ) continue;
            // The conductors' notes that sound (a block is 43 ms: every sixth one, a quarter second).
            if ( b % 6 != 0 ) continue;
            int lo = 128, count = 0;
            for ( int v = 0; v < MaxViewVoices; ++v )
            {
                const VoiceView vv = e.voiceView( v );
                if ( vv.m_hz <= 0.0f || vv.m_gain < 0.25f || ( vv.m_source != 1 && vv.m_source != 2 ) || vv.m_note < 0 )
                    continue;
                lo = std::min( lo, vv.m_note );
                ++count;
            }
            if ( count == 0 ) continue;
            ++ticks;
            notesSum += count;
            lowest.push_back( lo );
            if ( lo <= 47 ) ++bassTicks;
            if ( ( ( lo - e.brainRoot() ) % 12 + 12 ) % 12 == 0 ) ++rootTicks;
        }
        std::sort( lowest.begin(), lowest.end() );
        const int    median = lowest.empty() ? -1 : lowest[lowest.size() / 2];
        const double bass = ticks > 0 ? static_cast<double>( bassTicks ) / ticks : 0.0,
                     root = ticks > 0 ? static_cast<double>( rootTicks ) / ticks : 0.0;
        const double band = all > 0.0 ? low / all : 0.0;
        sumLowBass += bass;
        sumRoot += root;
        sumBand += band;
        medians.push_back( median );
        char line[512];
        std::snprintf(
            line, sizeof line,
            "{\"preset\": \"%s\", \"group\": %d, \"low\": %d, \"high\": %d, \"median_lowest\": %d, \"bass_share\": %.3f, \"root_share\": %.3f, "
            "\"notes\": %.2f, \"below150\": %.4f, \"level\": %.2f}\n",
            preset( index ).m_name, preset( index ).m_group,
            static_cast<int>( values[static_cast<int>( ParamIdE::BRAIN_LOW )] ),
            static_cast<int>( values[static_cast<int>( ParamIdE::BRAIN_HIGH )] ), median, bass, root,
            ticks > 0 ? static_cast<double>( notesSum ) / ticks : 0.0, band,
            10.0 * std::log10(
                       std::max( all / ( 2.0 * std::max( 1.0, static_cast<double>( blocks - warm ) ) * n ), 1e-20 ) ) );
        f << line;
        f.flush();
        const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
        std::printf( "  %3zu / %zu  %-24s lowest %3d  bass %4.0f %%  root %4.0f %%  below 150 Hz %5.1f %%   %5.0f s\n",
                     k + 1, which.size(), preset( index ).m_name, median, 100.0 * bass, 100.0 * root, 100.0 * band, s );
        std::fflush( stdout );
    }
    std::sort( medians.begin(), medians.end() );
    const double m = static_cast<double>( which.size() );
    std::printf(
        "%zu presets: lowest note's median %d, at B2 or below %.0f %% of the time, the root %.0f %%, below 150 Hz %.1f %% of the power\n",
        which.size(), medians.empty() ? -1 : medians[medians.size() / 2], 100.0 * sumLowBass / m, 100.0 * sumRoot / m,
        100.0 * sumBand / m );
    return 0;
}

}   // namespace

/**
 * @brief Measures the presets asked for and writes their levels as JSON (name: dBFS).
 * @param[in]     argc  argument count
 * @param[in,out] argv  arguments
 * @return 0 on success
 */
int main( int argc, char **argv )
{
    int         from = 0, to = numPresets() - 1;
    std::string namesFile, describeFile, sensitivityFile, registerFile, setText, evaluateFile;
    bool        checkNudges = false;
    int         step = 1, jobs = 10;
    double      seconds = -1.0;
    fs::path    out = repoRoot() / "work" / "presets" / "levels.json";
    for ( int i = 1; i < argc; ++i )
    {
        const std::string a = argv[i];
        const bool        has = i + 1 < argc;
        if ( a == "--from" && has )
            from = std::atoi( argv[++i] );
        else if ( a == "--to" && has )
            to = std::atoi( argv[++i] );
        else if ( a == "--seconds" && has )
            seconds = std::atof( argv[++i] );
        else if ( a == "--out" && has )
            out = argv[++i];
        else if ( a == "--names" && has )
            namesFile = argv[++i];
        else if ( a == "--describe" && has )
            describeFile = argv[++i];
        else if ( a == "--sensitivity" && has )
            sensitivityFile = argv[++i];
        else if ( a == "--nudgecheck" )
            checkNudges = true;
        else if ( a == "--register" && has )
            registerFile = argv[++i];
        else if ( a == "--step" && has )
            step = std::max( 1, std::atoi( argv[++i] ) );
        else if ( a == "--set" && has )
            setText = argv[++i];
        else if ( a == "--jobs" && has )
            jobs = std::clamp( std::atoi( argv[++i] ), 1, 16 );
        else if ( a == "--evaluate" && has )
            evaluateFile = argv[++i];
        else
        {
            std::fprintf(
                stderr,
                "additiv_presets [--from N] [--to M] [--step K] [--names FILE] [--seconds S] [--out FILE] [--describe FILE] [--sensitivity FILE] "
                "[--register FILE [--set \"key=value;...\"]] [--evaluate FILE] [--jobs N]\n" );
            return 1;
        }
    }
    if ( !evaluateFile.empty() )
    {   // Phase 31: any sounds, a line each, measured in parallel, written in their order
        std::ifstream            in( evaluateFile );
        std::vector<std::string> lines;
        for ( std::string line; std::getline( in, line ); )
            if ( !line.empty() ) lines.push_back( line );
        if ( seconds <= 0.0 ) seconds = 14.0;
        std::vector<double>      lufs( lines.size(), std::nan( "" ) );
        std::vector<Description> descs( lines.size() );
        std::atomic<size_t>      next{ 0 };
        auto                     work = [&]()
        {
            for ( ;; )
            {
                const size_t k = next++;
                if ( k >= lines.size() ) return;
                float values[NumParams];
                for ( const ParamDesc &d : paramTable() ) values[static_cast<int>( d.m_id )] = d.m_def;
                const std::string &line = lines[k];
                const size_t       bar = line.find( '|' );
                std::istringstream knobs( line.substr( 0, bar ) );
                for ( std::string kv; knobs >> kv; )
                {
                    const size_t eq = kv.find( '=' );
                    if ( eq == std::string::npos ) continue;
                    if ( const ParamDesc *d = findParam( kv.substr( 0, eq ).c_str() ) )
                        values[static_cast<int>( d->m_id )] =
                            std::clamp( static_cast<float>( std::atof( kv.c_str() + eq + 1 ) ), d->m_min, d->m_max );
                }
                std::string error;
                lufs[k] = measureSound( values, bar == std::string::npos ? std::string() : line.substr( bar + 1 ),
                                        seconds, &error, &descs[k] );
            }
        };
        {
            std::vector<std::thread> pool;
            for ( int j = 0; j < std::min<int>( jobs, static_cast<int>( lines.size() ) ); ++j )
                pool.emplace_back( work );
            for ( std::thread &th : pool ) th.join();
        }
        std::ofstream o( out );
        o << "[\n";
        for ( size_t k = 0; k < lines.size(); ++k )
        {
            if ( std::isnan( lufs[k] ) )
                o << "  null";
            else
                o << "  [" << lufs[k] << ", " << descs[k].m_brightness << ", " << descs[k].m_motion << ", "
                  << descs[k].m_width << ", " << descs[k].m_noise << "]";
            o << ( k + 1 < lines.size() ? ",\n" : "\n" );
        }
        o << "]\n";
        std::printf( "%zu sounds evaluated -> %s\n", lines.size(), out.string().c_str() );
        return 0;
    }
    if ( seconds <= 0.0 ) seconds = registerFile.empty() ? 14.0 : 240.0;
    from = std::clamp( from, 0, numPresets() - 1 );
    to = std::clamp( to, from, numPresets() - 1 );
    std::vector<int> which;
    if ( !namesFile.empty() )
    {
        std::ifstream in( namesFile );
        std::string   line;
        while ( std::getline( in, line ) )
        {
            while ( !line.empty() && ( line.back() == '\r' || line.back() == ' ' ) ) line.pop_back();
            if ( line.empty() ) continue;
            const int index = findPreset( line.c_str() );
            if ( index < 0 )
            {
                std::fprintf( stderr, "additiv_presets: no preset %s\n", line.c_str() );
                return 1;
            }
            which.push_back( index );
        }
    }
    else
    {
        for ( int i = from; i <= to; i += step ) which.push_back( i );
    }
    if ( !registerFile.empty() )
    {
        std::vector<std::pair<ParamIdE, float>> overrides;
        size_t                                  at = 0;
        while ( at < setText.size() )
        {   // "key=value;key=value"
            const size_t      end = std::min( setText.find( ';', at ), setText.size() );
            const std::string item = setText.substr( at, end - at );
            const size_t      eq = item.find( '=' );
            if ( eq != std::string::npos )
                if ( const ParamDesc *d = findParam( item.substr( 0, eq ).c_str() ) )
                    overrides.emplace_back( d->m_id, static_cast<float>( std::atof( item.c_str() + eq + 1 ) ) );
            at = end + 1;
        }
        return registerMode( which, seconds, registerFile, overrides );
    }
    if ( !sensitivityFile.empty() ) return sensitivity( which, seconds, sensitivityFile );
    if ( checkNudges ) return nudgeCheck( which, seconds );
    const auto                                       t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, double>>      levels;
    std::vector<std::pair<std::string, Description>> descriptions;
    double                                           lo = 1e9, hi = -1e9;
    // Several engines at once (each its own stream and tables on the card, its own thread for the effects): the
    // card is mostly idle while one engine renders. Kept in the order asked for, so the files read the same.
    std::vector<double>      dbs( which.size(), std::nan( "" ) );
    std::vector<Description> descs( which.size() );
    std::atomic<size_t>      next{ 0 }, done{ 0 };
    std::atomic<bool>        failed{ false };
    std::mutex               lock;
    std::string              failure;
    auto                     work = [&]()
    {
        for ( ;; )
        {
            const size_t k = next++;
            if ( k >= which.size() || failed ) return;
            const int    i = which[k];
            std::string  error;
            Description  desc;
            const double db = measure( i, seconds, &error, describeFile.empty() ? nullptr : &desc );
            const std::lock_guard<std::mutex> hold( lock );
            if ( std::isnan( db ) )
            {
                failure = error;
                failed = true;
                return;
            }
            dbs[k] = db;
            descs[k] = desc;
            lo = std::min( lo, db );
            hi = std::max( hi, db );
            const size_t count = ++done;
            if ( count % 64 == 0 || count == which.size() )
            {
                std::printf( "  %4zu / %zu  %-14s %-24s %6.1f LUFS (so far %.1f .. %.1f)\n", count, which.size(),
                             PresetGroupNames[preset( i ).m_group], preset( i ).m_name, db, lo, hi );
                std::fflush( stdout );
            }
        }
    };
    {
        std::vector<std::thread> pool;
        for ( int j = 0; j < std::min<int>( jobs, static_cast<int>( which.size() ) ); ++j ) pool.emplace_back( work );
        for ( std::thread &th : pool ) th.join();
    }
    if ( failed )
    {
        std::fprintf( stderr, "additiv_presets: %s\n", failure.c_str() );
        return 1;
    }
    for ( size_t k = 0; k < which.size(); ++k )
    {
        levels.emplace_back( preset( which[k] ).m_name, dbs[k] );
        if ( !describeFile.empty() ) descriptions.emplace_back( preset( which[k] ).m_name, descs[k] );
    }
    // Merged into what the file already holds, so one group can be measured again on its own.
    {
        std::ifstream in( out );
        std::string   line;
        while ( std::getline( in, line ) )
        {
            const size_t a = line.find( '"' ), b = line.find( '"', a + 1 ), colon = line.find( ':', b );
            if ( a == std::string::npos || b == std::string::npos || colon == std::string::npos ) continue;
            const std::string name = line.substr( a + 1, b - a - 1 );
            if ( std::none_of( levels.begin(), levels.end(), [&]( const auto &l ) { return l.first == name; } ) )
                levels.emplace_back( name, std::atof( line.c_str() + colon + 1 ) );
        }
    }
    fs::create_directories( out.parent_path() );
    std::ofstream f( out );
    f << "{\n";
    for ( size_t i = 0; i < levels.size(); ++i )
        f << "  \"" << levels[i].first << "\": " << levels[i].second << ( i + 1 < levels.size() ? ",\n" : "\n" );
    f << "}\n";
    if ( !describeFile.empty() )
    {   // name: [brightness, motion, width, noise, level], merged into what the file holds
        const fs::path           dp( describeFile );
        std::vector<std::string> kept;
        {
            std::ifstream in( dp );
            std::string   line;
            while ( std::getline( in, line ) )
            {
                const size_t a = line.find( '"' ), b = line.find( '"', a + 1 ),
                             open = line.find( '[', b == std::string::npos ? 0 : b );
                if ( a == std::string::npos || b == std::string::npos || open == std::string::npos ) continue;
                const std::string name = line.substr( a + 1, b - a - 1 );
                if ( std::any_of( descriptions.begin(), descriptions.end(),
                                  [&]( const auto &d ) { return d.first == name; } ) )
                    continue;
                const size_t close = line.find( ']', open );
                if ( close != std::string::npos ) kept.push_back( line.substr( 0, close + 1 ) );
            }
        }
        if ( dp.has_parent_path() ) fs::create_directories( dp.parent_path() );
        std::ofstream df( dp );
        df << "{\n";
        for ( size_t i = 0; i < descriptions.size(); ++i )
        {
            const Description &d = descriptions[i].second;
            char               line[256];
            std::snprintf( line, sizeof line, "  \"%s\": [%.5f, %.5f, %.5f, %.5f, %.2f]%s\n",
                           descriptions[i].first.c_str(), d.m_brightness, d.m_motion, d.m_width, d.m_noise,
                           levels[i].second, i + 1 < descriptions.size() || !kept.empty() ? "," : "" );
            df << line;
        }
        for ( size_t i = 0; i < kept.size(); ++i ) df << kept[i] << ( i + 1 < kept.size() ? ",\n" : "\n" );
        df << "}\n";
    }
    const double took = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    std::printf( "%zu presets measured in %.0f s, %.1f .. %.1f LUFS -> %s\n", levels.size(), took, lo, hi,
                 out.string().c_str() );
    return 0;
}
