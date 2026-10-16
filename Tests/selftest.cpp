/**
 * @file selftest.cpp
 * @brief additiv_selftest: the instrument around the cloud -- the parameter table, the tuning, notes
 *        that begin on their sample, silence, determinism, the conductor's Autoplay, stealing,
 *        release, the real-time path without underruns, the echoes and the modal hall. Needs the
 *        GPU; the cloud's own measurements are probe's selftest.
 * @author Rene Weller
 */
#include "additiv/Engine.h"
#include "additiv/Sun.h"
#include "additiv/Journey.h"
#include "additiv/Describe.h"
#include "additiv/Find.h"
#include "additiv/Focus.h"
#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Presets.h"
#include "additiv/Route.h"
#include "additiv/PresetMap.h"
#include "additiv/Spectrum.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <memory>
#include <vector>

using namespace additiv;

namespace
{

int g_failures = 0;   ///< checks that failed

/**
 * @brief Prints one check's result and counts a failure.
 * @param[in] good    whether it passed
 * @param[in] what    what was checked
 * @param[in] detail  the measured values
 */
void check( bool good, const char *what, const std::string &detail )
{
    std::printf( "  %s  %-52s %s\n", good ? "ok  " : "FAIL", what, detail.c_str() );
    if ( !good ) ++g_failures;
}

/**
 * @brief A string from a printf format.
 * @param[in] fmt  the format
 * @return the text
 * @param[in] ...   its values
 */
std::string fmt( const char *fmt, ... )
{
    char    buf[512];
    va_list ap;
    va_start( ap, fmt );
    std::vsnprintf( buf, sizeof buf, fmt, ap );
    va_end( ap );
    return buf;
}

/**
 * @brief An offline engine at 48 kHz with 256-sample host blocks.
 * @param[in,out] e      the engine
 * @param[in]     brain  whether the conductor plays
 * @param[in]     seed   the dice
 * @return true when it is prepared
 */
bool openOffline( Engine &e, bool brain, std::uint64_t seed = 0x5EEDu )
{
    e.setRealtime( false );
    e.setParam( ParamIdE::BRAIN_ON, brain ? 1.0f : 0.0f );
    EngineConfig c;
    c.m_maxHostBlock = 256;
    c.m_realtimeGpuClass = false;
    c.m_seed = seed;
    std::string error;
    if ( e.prepare( c, &error ) ) return true;
    std::printf( "  FAIL  the engine did not prepare: %s\n", error.c_str() );
    ++g_failures;
    return false;
}

/**
 * @brief Renders seconds of an engine into interleaved stereo.
 * @param[in,out] e        the engine
 * @param[in]     seconds  how long
 * @return the samples
 */
std::vector<float> render( Engine &e, double seconds )
{
    const int          n = 256;
    const int          blocks = static_cast<int>( seconds * 48000.0 / n );
    std::vector<float> lr;
    lr.reserve( static_cast<size_t>( blocks ) * n * 2 );
    std::vector<float> L( n ), R( n );
    for ( int b = 0; b < blocks; ++b )
    {
        e.process( L.data(), R.data(), n );
        for ( int i = 0; i < n; ++i )
        {
            lr.push_back( L[i] );
            lr.push_back( R[i] );
        }
    }
    return lr;
}

/**
 * @brief RMS of a stretch of interleaved stereo, in dBFS.
 * @param[in] lr    samples
 * @param[in] from  first frame
 * @param[in] to    end frame
 * @return dBFS
 */
double rmsDb( const std::vector<float> &lr, size_t from, size_t to )
{
    double sq = 0.0;
    to = std::min( to, lr.size() / 2 );
    for ( size_t i = from * 2; i < to * 2; ++i ) sq += static_cast<double>( lr[i] ) * lr[i];
    const double rms = std::sqrt( sq / std::max<double>( 1.0, static_cast<double>( ( to - from ) * 2 ) ) );
    return 20.0 * std::log10( std::max( rms, 1e-12 ) );
}

/**
 * @brief Every effect off: what is heard is the voices alone.
 * @param[in,out] e  the engine (before or after prepare)
 */
void noEffects( Engine &e )
{
    for ( ParamIdE id :
          { ParamIdE::FAR_LEVEL, ParamIdE::EARLY_LEVEL, ParamIdE::NEAR_MIX, ParamIdE::DELAY_MIX, ParamIdE::DELAY_TO_FAR,
            ParamIdE::DELAY2_MIX, ParamIdE::DELAY2_TO_FAR, ParamIdE::MODAL_LEVEL, ParamIdE::FAR_UNMASK } )
        e.setParam( id, 0.0f );
}

/**
 * @brief Every effect off, and a short envelope: what is heard is the voice and then only what a test opens.
 * @param[in,out] e  the engine (before or after prepare)
 */
void quietEffects( Engine &e )
{
    noEffects( e );
    e.setParam( ParamIdE::ATTACK, 0.01f );
    e.setParam( ParamIdE::RELEASE, 0.05f );
    e.setParam( ParamIdE::SUSTAIN, 1.0f );
}

/**
 * @brief The level of one frequency in a stretch of the mono sum (a Goertzel filter under a Hann window).
 * @param[in] lr    interleaved stereo
 * @param[in] from  first frame
 * @param[in] to    one past the last frame
 * @param[in] hz    the frequency
 * @return its level in dB (relative; only differences mean anything)
 */
double toneDb( const std::vector<float> &lr, size_t from, size_t to, double hz )
{
    const double w = 2.0 * 3.14159265358979 * hz / 48000.0, c = 2.0 * std::cos( w );
    double       s1 = 0.0, s2 = 0.0;
    const size_t len = to - from;
    for ( size_t i = 0; i < len; ++i )
    {
        const double win =
            0.5 - 0.5 * std::cos( 2.0 * 3.14159265358979 * static_cast<double>( i ) / static_cast<double>( len ) );
        const double x = win * ( lr[2 * ( from + i )] + lr[2 * ( from + i ) + 1] );
        const double s = x + c * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    return 10.0 * std::log10( std::max( s1 * s1 + s2 * s2 - c * s1 * s2, 1e-30 ) );
}

/**
 * @brief One frequency in one channel of a stretch, as a complex value (Goertzel under a Hann window).
 * @param[in] lr       interleaved stereo
 * @param[in] from     first frame
 * @param[in] to       one past the last frame
 * @param[in] hz       the frequency
 * @param[in] channel  0 left, 1 right
 * @return its complex amplitude (relative; the phase and the ratio between channels mean something)
 */
std::complex<double> toneAt( const std::vector<float> &lr, size_t from, size_t to, double hz, int channel )
{
    const double         w = 2.0 * 3.14159265358979 * hz / 48000.0;
    std::complex<double> acc( 0.0, 0.0 );
    const size_t         len = to - from;
    for ( size_t i = 0; i < len; ++i )
    {
        const double win =
            0.5 - 0.5 * std::cos( 2.0 * 3.14159265358979 * static_cast<double>( i ) / static_cast<double>( len ) );
        acc += win * static_cast<double>( lr[2 * ( from + i ) + static_cast<size_t>( channel )] ) *
               std::polar( 1.0, -w * static_cast<double>( from + i ) );
    }
    return acc;
}

/**
 * @brief The frequencies of the sounding voices, lowest first.
 * @param[in] e  the engine
 * @return Hz of each voice that sounds
 */
std::vector<double> soundingAll( const Engine &e )
{
    std::vector<double> hz;
    for ( int i = 0; i < 16; ++i )
        if ( e.voiceView( i ).m_hz > 0.0f ) hz.push_back( e.voiceView( i ).m_hz );
    std::sort( hz.begin(), hz.end() );
    return hz;
}

/**
 * @brief The frequency of the first sounding voice.
 * @param[in] e  the engine
 * @return Hz, 0 when none sounds
 */
double soundingHz( const Engine &e )
{
    for ( int i = 0; i < 16; ++i )
        if ( e.voiceView( i ).m_hz > 0.0f ) return e.voiceView( i ).m_hz;
    return 0.0;
}

}   // namespace

/**
 * @brief Runs every check, or those from a number on.
 * @param[in] argc  the arguments' count
 * @param[in] argv  "--from N": only checks N and after
 * @return 0 when all passed
 */
int main( int argc, char **argv )
{
    int from = 1;
    for ( int a = 1; a + 1 < argc; ++a )
        if ( std::strcmp( argv[a], "--from" ) == 0 ) from = std::max( 1, std::atoi( argv[a + 1] ) );
    // 1. The parameter table: in ParamId order, defaults inside their ranges, keys found.
    if ( from <= 1 )
    {
        bool ordered = true, inRange = true;
        for ( int i = 0; i < NumParams; ++i )
        {
            const ParamDesc &d = paramTable()[static_cast<size_t>( i )];
            ordered = ordered && static_cast<int>( d.m_id ) == i;
            inRange = inRange && d.m_def >= d.m_min && d.m_def <= d.m_max;
        }
        const ParamDesc *p = findParam( "purity" );
        check( ordered && inRange && p && p->m_id == ParamIdE::PURITY, "parameter table: order, defaults, keys",
               fmt( "%d parameters", NumParams ) );
    }

    // 2. The tuning (Noctuary's): the root sounds its tempered pitch, the fifth above it 3/2 of that.
    if ( from <= 2 )
    {
        FixedScale   s;
        const bool   built = makeBuiltinScale( 3, s );   // JI 7-limit
        const double root = scaleFrequency( s, 62, 62, 440.0, true );
        const double fifth = scaleFrequency( s, 69, 62, 440.0, true );
        const double et = 440.0 * std::pow( 2.0, -7.0 / 12.0 );
        check( built && std::fabs( root / et - 1.0 ) < 1e-12 && std::fabs( fifth / root - 1.5 ) < 1e-12,
               "tuning: JI 7-limit root and fifth",
               fmt( "D %.4f Hz, A %.4f Hz, ratio %.9f", root, fifth, fifth / root ) );
    }

    // 3. Without the conductor and without notes the instrument is silent to the last bit.
    if ( from <= 3 )
    {
        Engine e;
        if ( !openOffline( e, false ) ) return 1;
        const std::vector<float> lr = render( e, 2.0 );
        const bool               silent = std::all_of( lr.begin(), lr.end(), []( float x ) { return x == 0.0f; } );
        check( silent, "silence: no conductor, no notes", silent ? "every sample 0" : "noise" );
    }

    // 4. A played note begins on its sample: stamped offset + latency, the GPU ramp starts there.
    if ( from <= 4 )
    {
        Engine e;
        if ( !openOffline( e, false ) ) return 1;
        e.setParam( ParamIdE::ATTACK, 0.01f );
        e.noteOn( 69, 1.0f, 37 );
        const std::vector<float> lr = render( e, 1.0 );
        size_t                   first = lr.size();
        for ( size_t i = 0; i < lr.size(); ++i )
            if ( lr[i] != 0.0f )
            {
                first = i / 2;
                break;
            }
        const size_t expect = static_cast<size_t>( e.latencySamples() ) + 37 + 1;
        check( first == expect, "a note begins on its sample (offset + latency)",
               fmt( "first sound at %zu, expected %zu (latency %d)", first, expect, e.latencySamples() ) );
        check( e.lateEvents() == 0, "offline: no note is late",
               fmt( "%llu late", static_cast<unsigned long long>( e.lateEvents() ) ) );
    }

    // 5. The same seed renders the same bits, conductor and all.
    if ( from <= 5 )
    {
        std::vector<float> a, b;
        for ( std::vector<float> *dst : { &a, &b } )
        {
            Engine e;
            if ( !openOffline( e, true, 42 ) ) return 1;
            e.setParam( ParamIdE::AUTO_RATE, 5.0f );
            *dst = render( e, 20.0 );
        }
        const bool same = a.size() == b.size() && std::memcmp( a.data(), b.data(), sizeof( float ) * a.size() ) == 0;
        check( same, "determinism: the conductor's 20 s twice", same ? "bit for bit" : "differs" );
    }

    // 6. Autoplay (Chords): the cluster fills to its density and the piece sounds at a sane level.
    if ( from <= 6 )
    {
        Engine e;
        if ( !openOffline( e, true, 7 ) ) return 1;
        e.setParam( ParamIdE::AUTO_MODE, 1.0f );
        e.setParam( ParamIdE::AUTO_RATE, 5.0f );
        e.setParam( ParamIdE::BRAIN_DENSITY, 5.0f );
        const std::vector<float> lr = render( e, 60.0 );
        const bool               finite =
            std::all_of( lr.begin(), lr.end(), []( float x ) { return std::isfinite( x ) && std::fabs( x ) <= 1.0f; } );
        const double db = rmsDb( lr, 40 * 48000, 60 * 48000 );
        check( finite && e.soundingVoices() >= 5 && db > -45.0 && db < -6.0,
               "autoplay: the cluster fills, the level is sane",
               fmt( "%d voices, %.1f dBFS RMS over 40-60 s, root %d", e.soundingVoices(), db, e.brainRoot() ) );
    }

    // 7. Twenty notes on sixteen voices: the oldest are stolen (faded over a block), nothing breaks.
    if ( from <= 7 )
    {
        Engine e;
        if ( !openOffline( e, false ) ) return 1;
        e.setParam( ParamIdE::ATTACK, 0.01f );
        for ( int k = 0; k < 20; ++k ) e.noteOn( 48 + k, 0.8f, k * 10 );
        const std::vector<float> lr = render( e, 1.5 );
        const bool finite = std::all_of( lr.begin(), lr.end(), []( float x ) { return std::isfinite( x ); } );
        check( finite && e.soundingVoices() == 16, "stealing: 20 notes, 16 voices",
               fmt( "%d voices sounding", e.soundingVoices() ) );
    }

    // 8. Release: a note let go falls silent and its voice is free again.
    if ( from <= 8 )
    {
        Engine e;
        if ( !openOffline( e, false ) ) return 1;
        e.setParam( ParamIdE::ATTACK, 0.05f );
        e.setParam( ParamIdE::RELEASE, 0.3f );
        e.noteOn( 60, 1.0f, 0 );
        render( e, 1.0 );
        const int during = e.soundingVoices();
        e.noteOff( 60, 0 );
        render( e, 2.0 );
        check( during == 1 && e.soundingVoices() == 0, "release: the voice falls silent and is freed",
               fmt( "%d voice while held, %d after", during, e.soundingVoices() ) );
    }

    // 9. Real time: a paced callback of 256 samples for 6 s with notes coming and going -- the
    //    worker keeps ahead, nothing underruns, no note is late.
    if ( from <= 9 )
    {
        Engine e;
        e.setRealtime( true );
        EngineConfig c;
        c.m_maxHostBlock = 256;
        c.m_realtimeGpuClass = true;
        std::string error;
        if ( !e.prepare( c, &error ) )
        {
            check( false, "real time: prepare", error );
            return 1;
        }
        std::vector<float> L( 256 ), R( 256 );
        LARGE_INTEGER      f, t;
        QueryPerformanceFrequency( &f );
        QueryPerformanceCounter( &t );
        const double    period = 256.0 / 48000.0;
        const long long start = t.QuadPart;
        double          peak = 0.0;
        for ( int b = 0; b < static_cast<int>( 6.0 / period ); ++b )
        {
            const long long due =
                start + static_cast<long long>( ( b + 1 ) * period * static_cast<double>( f.QuadPart ) );
            do
            {
                std::this_thread::yield();
                QueryPerformanceCounter( &t );
            } while ( t.QuadPart < due );
            if ( b % 40 == 0 ) e.noteOn( 50 + ( b / 40 ) % 24, 0.7f, ( b * 7 ) % 256 );
            if ( b % 40 == 20 ) e.noteOff( 50 + ( b / 40 ) % 24, ( b * 3 ) % 256 );
            e.process( L.data(), R.data(), 256 );
            for ( int i = 0; i < 256; ++i ) peak = std::max( peak, static_cast<double>( std::fabs( L[i] ) ) );
        }
        check( e.underruns() == 0 && e.lateEvents() == 0 && peak > 0.0, "real time: 6 s paced, notes coming and going",
               fmt( "underruns %llu, late %llu, latency %d samples (%.1f ms), GPU %.3f ms/block",
                    static_cast<unsigned long long>( e.underruns() ), static_cast<unsigned long long>( e.lateEvents() ),
                    e.latencySamples(), e.latencySamples() / 48.0, e.lastGpuMs() ) );
    }

    // 10. Brain 2: the second conductor plays its own notes in the background.
    if ( from <= 10 )
    {
        Engine e;
        if ( !openOffline( e, true, 11 ) ) return 1;
        e.setParam( ParamIdE::AUTO_RATE, 5.0f );
        e.setParam( ParamIdE::BRAIN2_ON, 1.0f );
        e.setParam( ParamIdE::BRAIN2_RATE, 4.0f );
        const std::vector<float> lr = render( e, 60.0 );
        const bool finite = std::all_of( lr.begin(), lr.end(), []( float x ) { return std::isfinite( x ); } );
        check( finite && e.soundingFrom( 1 ) >= 3 && e.soundingFrom( 2 ) >= 1,
               "brain 2: a second conductor in the background",
               fmt( "conductor %d, second conductor %d voices", e.soundingFrom( 1 ), e.soundingFrom( 2 ) ) );
    }

    // 11. The Timbre scale (Sethares) is made from the cloud's own spectrum.
    if ( from <= 11 )
    {
        BrainSpectrum sp;
        sp.m_count = BrainSpectrum::Max;
        for ( int h = 1; h <= sp.m_count; ++h )
        {
            sp.m_ratio[h - 1] = h * stretchOf( h, 0.02f );
            sp.m_amp[h - 1] = harmonicShape( h, 1.2f, 0.0f, 1.0f );
        }
        FixedScale s;
        const bool built = makeTimbreScale( sp, s );
        Engine     e;
        if ( !openOffline( e, false ) ) return 1;
        e.setParam( ParamIdE::SCALE, static_cast<float>( TimbreScaleIndex ) );
        e.setParam( ParamIdE::ATTACK, 0.05f );
        e.noteOn( 60, 0.8f, 0 );
        e.noteOn( 67, 0.8f, 0 );
        const std::vector<float> lr = render( e, 2.0 );
        const bool finite = std::all_of( lr.begin(), lr.end(), []( float x ) { return std::isfinite( x ); } );
        check( built && s.m_count >= 2 && finite && rmsDb( lr, 48000, 96000 ) > -60.0,
               "timbre scale: from the stretched spectrum",
               fmt( "%d degrees per period, plays at %.1f dBFS", s.m_count, rmsDb( lr, 48000, 96000 ) ) );
    }

    // 12. The landscape's make-up: the same chord in Aurora sits within 4 dB of the bare cloud;
    //     a low pass at 60 Hz, under every fundamental, takes it far down.
    if ( from <= 12 )
    {
        auto chordDb = []( float motion, float filterModel )
        {
            Engine e;
            if ( !openOffline( e, false ) ) return -200.0;
            e.setParam( ParamIdE::MOTION, motion );
            e.setParam( ParamIdE::FILTER_MODEL, filterModel );
            e.setParam( ParamIdE::CUTOFF,
                        60.0f );   // below every fundamental: at tilt 1.2 they carry two thirds of the power
            e.setParam( ParamIdE::KEY_TRACK, 0.0f );
            e.setParam( ParamIdE::FILTER_DRIFT, 0.0f );
            e.setParam( ParamIdE::ATTACK, 0.5f );
            for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
            const std::vector<float> lr = render( e, 20.0 );
            return rmsDb( lr, 5 * 48000, 20 * 48000 );
        };
        const double bare = chordDb( 0.0f, 0.0f );
        const double aurora = chordDb( 6.0f, 0.0f );
        const double lp = chordDb( 0.0f, 2.0f );
        check( std::fabs( aurora - bare ) < 4.0 && lp < bare - 10.0, "landscape make-up, and the filter's reach",
               fmt( "bare %.1f, Aurora %.1f, LP 12 at 60 Hz %.1f dBFS", bare, aurora, lp ) );
    }

    // 13. Everything that moves at once -- breath, doppler, tide, air, presence, a resonant filter
    //     with drift and envelope, glissando -- stays finite and sane.
    if ( from <= 13 )
    {
        Engine e;
        if ( !openOffline( e, true, 5 ) ) return 1;
        e.setParam( ParamIdE::AUTO_RATE, 4.0f );
        e.setParam( ParamIdE::BREATH, 1.0f );
        e.setParam( ParamIdE::BREATH_RATE, 0.2f );
        e.setParam( ParamIdE::DOPPLER, 1.0f );
        e.setParam( ParamIdE::TIDE, 30.0f );
        e.setParam( ParamIdE::TIDE_PERIOD, 1.0f );
        e.setParam( ParamIdE::AIR, 0.7f );
        e.setParam( ParamIdE::PRESENCE, 6.0f );
        e.setParam( ParamIdE::PAD_LOW_CUT, 120.0f );
        e.setParam( ParamIdE::FILTER_MODEL, 8.0f );   // Formant
        e.setParam( ParamIdE::RESONANCE, 0.8f );
        e.setParam( ParamIdE::FILTER_ENV, 2.0f );
        e.setParam( ParamIdE::FILTER_DRIFT, 1.0f );
        e.setParam( ParamIdE::MOTION, 5.0f );   // Glissando
        e.setParam( ParamIdE::INHARMONIC, 0.5f );
        const std::vector<float> lr = render( e, 40.0 );
        const bool               finite =
            std::all_of( lr.begin(), lr.end(), []( float x ) { return std::isfinite( x ) && std::fabs( x ) <= 1.0f; } );
        const double db = rmsDb( lr, 20 * 48000, 40 * 48000 );
        check( finite && db > -50.0 && db < -6.0, "everything moving at once: finite and sane",
               fmt( "%.1f dBFS RMS over 20-40 s", db ) );
    }

    // 14. The matrix as text: written by one engine, read by another, the same rows.
    if ( from <= 14 )
    {
        Engine a, b;
        a.setModRow( 0, { ModSourceE::LFO1, static_cast<int>( ParamIdE::CUTOFF ), 0.35f } );
        a.setModRow( 3, { ModSourceE::MACRO2, static_cast<int>( ParamIdE::MOTION_DEPTH ), -0.5f } );
        a.setModRow(
            5, { ModSourceE::ENV1, static_cast<int>( ParamIdE::SCALE ), 0.2f } );   // a choice: not a target, dropped
        const std::string text = a.modText();
        b.setModText( text );
        const ModRow r0 = b.modRow( 0 ), r1 = b.modRow( 1 ), r2 = b.modRow( 2 );
        const bool   same = r0.m_source == ModSourceE::LFO1 && r0.m_target == static_cast<int>( ParamIdE::CUTOFF ) &&
                            std::fabs( r0.m_depth - 0.35f ) < 1e-6f && r1.m_source == ModSourceE::MACRO2 &&
                            r1.m_target == static_cast<int>( ParamIdE::MOTION_DEPTH ) &&
                            std::fabs( r1.m_depth + 0.5f ) < 1e-6f && r2.m_source == ModSourceE::OFF;
        check( same, "modulation: the matrix as text and back", "\"" + text + "\"" );
    }

    // 15. No source ever steps: every LFO shape, a hundred cycles at 1 Hz in millisecond steps.
    if ( from <= 15 )
    {
        double worst = 0.0;
        for ( int shape = 0; shape < NumLfoShapes; ++shape )
        {
            Modulators m;
            m.reset( 3 );
            ModInputs in;
            for ( int i = 0; i < 4; ++i )
            {
                in.m_lfoShape[i] = shape;
                in.m_lfoRate[i] = 1.0f;
            }
            m.advance( 0.001f, in );
            float prev = m.value( ModSourceE::LFO1 );
            for ( int s = 0; s < 100000; ++s )
            {
                m.advance( 0.001f, in );
                const float v = m.value( ModSourceE::LFO1 );
                worst = std::max( worst, static_cast<double>( std::fabs( v - prev ) ) );
                prev = v;
            }
        }
        // The steepest a shape may be: a ramp's return over 7 % of a cycle, 2 / 0.07 per cycle.
        check( worst < 0.03, "modulation: no LFO shape steps",
               fmt( "largest move per millisecond at 1 Hz %.4f (a return over 7 %%: 0.029)", worst ) );
    }

    // 16. An LFO on the cutoff moves the level; the live value is where the cutoff plays.
    if ( from <= 16 )
    {
        auto swing = []( bool modulated, float *played )
        {
            Engine e;
            if ( !openOffline( e, false ) ) return 0.0;
            noEffects( e );   // the level's swing as the voices make it: a room or a hall would smooth it
            e.setParam( ParamIdE::FILTER_MODEL, 2.0f );   // LP 12
            e.setParam( ParamIdE::CUTOFF, 400.0f );
            e.setParam( ParamIdE::KEY_TRACK, 0.0f );
            e.setParam( ParamIdE::FILTER_DRIFT, 0.0f );
            e.setParam( ParamIdE::ATTACK, 0.1f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::LFO1_SHAPE, 0.0f );
            e.setParam( ParamIdE::LFO1_RATE, 0.5f );
            if ( modulated ) e.setModRow( 0, { ModSourceE::LFO1, static_cast<int>( ParamIdE::CUTOFF ), 0.4f } );
            for ( int note : { 45, 52, 57 } ) e.noteOn( note, 0.8f, 0 );
            const std::vector<float> lr = render( e, 8.0 );
            double                   lo = 1e9, hi = -1e9;
            for ( int w = 8; w < 32; ++w )
            {   // quarter seconds from 2 s on
                const double db = rmsDb( lr, static_cast<size_t>( w ) * 12000, static_cast<size_t>( w + 1 ) * 12000 );
                lo = std::min( lo, db );
                hi = std::max( hi, db );
            }
            if ( played ) *played = e.playedNorm( ParamIdE::CUTOFF );
            return hi - lo;
        };
        float        played = 0.0f, still = 0.0f;
        const double moved = swing( true, &played );
        const double steady = swing( false, &still );
        // Without the row the level still breathes: behind a 400 Hz low pass only the fundamentals' clouds
        // remain, and their members beat. The row must move it far more than that.
        check(
            moved > steady + 6.0 && std::isfinite( played ) && played >= 0.0f && played <= 1.0f && std::isnan( still ),
            "modulation: an LFO on the cutoff, and its live value",
            fmt( "level swings %.1f dB with the row, %.1f dB without; cutoff plays at %.2f", moved, steady, played ) );
    }

    // 17. The factory bank: 64 groups of 64, unique names, every key known and every value in its
    //     range, every matrix row read, and bank select plus program finding each preset again.
    if ( from <= 17 )
    {
        int                      bad = 0, outOfRange = 0, rowsLost = 0, perGroup[NumPresetGroups] = {};
        std::vector<std::string> names;
        for ( int i = 0; i < numPresets(); ++i )
        {
            const Preset &p = preset( i );
            ++perGroup[std::clamp( p.m_group, 0, NumPresetGroups - 1 )];
            names.push_back( p.m_name );
            int entries = 0;
            for ( const char *s = p.m_params; *s != 0; ++s )
                if ( *s == '=' ) ++entries;
            float       values[NumParams];
            std::string mod;
            if ( presetValues( i, values, &mod ) != entries ) ++bad;
            // In range before the clamp: read each value again and compare.
            for ( const char *s = p.m_params; *s != 0; )
            {
                while ( *s == ' ' ) ++s;
                const char *eq = std::strchr( s, '=' );
                if ( eq == nullptr ) break;
                const std::string key( s, eq );
                const float       v = static_cast<float>( std::atof( eq + 1 ) );
                if ( const ParamDesc *d = findParam( key.c_str() ); d != nullptr && ( v < d->m_min || v > d->m_max ) )
                    ++outOfRange;
                const char *end = std::strchr( eq, ' ' );
                s = end != nullptr ? end : eq + std::strlen( eq );
            }
            ModRow    rows[ModRows];
            const int want = mod.empty() ? 0 : 1 + static_cast<int>( std::count( mod.begin(), mod.end(), ';' ) );
            if ( modRowsFromText( mod, rows, ModRows ) != want ) ++rowsLost;
            if ( presetAt( p.m_group, presetIndexInGroup( i ) ) != i ) ++bad;
        }
        std::sort( names.begin(), names.end() );
        const bool unique = std::adjacent_find( names.begin(), names.end() ) == names.end();
        bool       full = true;
        for ( int g = 0; g < NumPresetGroups; ++g ) full = full && perGroup[g] == 64;
        check( numPresets() == 14336 && full && unique && bad == 0 && outOfRange == 0 && rowsLost == 0,
               "presets: the factory bank",
               fmt( "%d presets, %s, %d unreadable, %d out of range, %d with lost rows", numPresets(),
                    unique ? "names unique" : "NAMES REPEAT", bad, outOfRange, rowsLost ) );
    }

    // 18. The family jam, following: the conductor's root comes to the leader's pitch class and stays
    //     there while the conductor plays; the energy's neutral 0.5 changes nothing.
    if ( from <= 18 )
    {
        Engine e;
        if ( !openOffline( e, true, 21 ) ) return 1;
        e.setParam( ParamIdE::AUTO_RATE, 4.0f );
        e.setParam( ParamIdE::AUTO_ROOT_MOVE, 1.0f );   // the conductor wants to wander; the jam holds it
        e.setJamRoot( 7 );
        e.setJamEnergy( 0.5f );
        bool held = true;
        for ( int s = 0; s < 30; ++s )
        {
            render( e, 1.0 );
            if ( s >= 1 && ( ( e.brainRoot() % 12 ) + 12 ) % 12 != 7 ) held = false;
        }
        check( held, "family jam: the root held on the leader's",
               fmt( "root %d (pitch class %d) after 30 s", e.brainRoot(), ( ( e.brainRoot() % 12 ) + 12 ) % 12 ) );
    }

    // 19. A preset change fills the chord at speed (requestFill, Noctuary's way): with an event every
    //     two minutes the conductor alone has one or two notes after five seconds, filled it has all five.
    if ( from <= 19 )
    {
        auto conducted = []( bool fill, int &count )
        {
            Engine e;
            if ( !openOffline( e, true, 23 ) ) return false;
            e.setParam( ParamIdE::AUTO_MODE, 1.0f );
            e.setParam( ParamIdE::BRAIN_RATE, 120.0f );
            e.setParam( ParamIdE::AUTO_RATE, 300.0f );
            e.setParam( ParamIdE::BRAIN_DENSITY, 5.0f );
            render( e, 1.0 );
            if ( fill ) e.requestFill();
            render( e, 4.0 );
            count = e.soundingFrom( 1 );
            return true;
        };
        int slow = 0, filled = 0;
        if ( !conducted( false, slow ) || !conducted( true, filled ) ) return 1;
        check( slow <= 2 && filled >= 5, "preset change: the chord fills within seconds",
               fmt( "%d notes alone, %d filled, after 5 s", slow, filled ) );
    }

    // 20. The modal hall decays at its time: with a decay of 2 s and no damping, 30 dB in a second.
    if ( from <= 20 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::MODAL_LEVEL, 1.0f );
        e.setParam( ParamIdE::MODAL_DECAY, 2.0f );
        e.setParam( ParamIdE::MODAL_DAMP, 0.0f );
        e.setParam( ParamIdE::MODAL_TUNE, 0.0f );
        e.setParam( ParamIdE::MODAL_DRIFT, 0.0f );
        if ( !openOffline( e, false, 31 ) ) return 1;
        e.noteOn( 62, 0.8f, 0 );
        render( e, 1.0 );
        e.noteOff( 62, 0 );
        const std::vector<float> lr = render( e, 2.0 );
        const double             a = rmsDb( lr, 14400, 24000 ), b = rmsDb( lr, 62400, 72000 );
        check( a - b > 22.0 && a - b < 38.0, "modal hall: it dies at its decay time",
               fmt( "%.1f dB in 1 s at 2 s to -60 dB (30 expected); tail at %.1f dBFS", a - b, a ) );
    }

    // 21. Tuned, it rings on the scale: 15 cents beside the voice's fourth harmonic the tail is
    //     far quieter than on it -- spread over the range it is not.
    if ( from <= 21 )
    {
        double ratio[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MODAL_LEVEL, 1.0f );
            e.setParam( ParamIdE::MODAL_DECAY, 4.0f );
            e.setParam( ParamIdE::MODAL_DAMP, 0.0f );
            e.setParam( ParamIdE::MODAL_TUNE, static_cast<float>( k ) );
            e.setParam( ParamIdE::MODAL_SPREAD, 0.0f );
            e.setParam( ParamIdE::MODAL_DRIFT, 0.0f );
            if ( !openOffline( e, false, 37 ) ) return 1;
            e.noteOn( 62, 0.8f, 0 );
            render( e, 1.5 );
            const double f4 = 4.0 * soundingHz( e );
            e.noteOff( 62, 0 );
            const std::vector<float> lr = render( e, 1.5 );
            ratio[k] = toneDb( lr, 9600, 57600, f4 * std::pow( 2.0, 15.0 / 1200.0 ) ) - toneDb( lr, 9600, 57600, f4 );
        }
        check( ratio[1] < ratio[0] - 10.0, "modal hall: tuned, it rings on the scale",
               fmt( "15 ct beside the 4th harmonic: %.1f dB spread, %.1f dB tuned", ratio[0], ratio[1] ) );
    }

    // 22. Frozen, it holds: the level three seconds on within 1.5 dB of where it was.
    if ( from <= 22 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::MODAL_LEVEL, 1.0f );
        e.setParam( ParamIdE::MODAL_DECAY, 3.0f );
        e.setParam( ParamIdE::MODAL_DRIFT, 0.0f );
        if ( !openOffline( e, false, 41 ) ) return 1;
        e.noteOn( 62, 0.8f, 0 );
        render( e, 1.0 );
        e.noteOff( 62, 0 );
        e.setParam( ParamIdE::MODAL_FREEZE, 1.0f );
        const std::vector<float> lr = render( e, 4.0 );
        const double             a = rmsDb( lr, 24000, 48000 ), b = rmsDb( lr, 168000, 192000 );
        check( std::fabs( a - b ) < 1.5 && a > -80.0, "modal hall: frozen, it holds",
               fmt( "%.1f dBFS after 0.5 s, %.1f dBFS after 3.5 s", a, b ) );
    }

    // 23. The echo arrives at its time: a tenth of a second of a note, then half a second of
    //     delay; the window around the echo is far louder than the gap before it.
    if ( from <= 23 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::ATTACK, 0.005f );
        e.setParam( ParamIdE::RELEASE, 0.03f );
        e.setParam( ParamIdE::DELAY_MIX, 1.0f );
        e.setParam( ParamIdE::DELAY_TIME_L, 0.5f );
        e.setParam( ParamIdE::DELAY_TIME_R, 0.5f );
        e.setParam( ParamIdE::DELAY_FEEDBACK, 0.0f );
        if ( !openOffline( e, false, 43 ) ) return 1;
        e.noteOn( 62, 0.8f, 0 );
        const std::vector<float> on = render( e, 0.1 );
        e.noteOff( 62, 0 );
        const std::vector<float> lr = render( e, 1.0 );
        const double             gap = rmsDb( lr, 9600, 16800 ), echo = rmsDb( lr, 19200, 26400 );
        check( echo > gap + 10.0, "echo: it arrives at its time",
               fmt( "%.1f dBFS in the gap, %.1f dBFS half a second after the note", gap, echo ) );
    }

    // 24. Everything at once, with the conductor: finite, inside the rails, at a sane level.
    if ( from <= 24 )
    {
        Engine e;
        for ( const auto &[id, v] :
              std::initializer_list<std::pair<ParamIdE, float>>{ { ParamIdE::DELAY_MIX, 0.4f },
                                                                 { ParamIdE::DELAY2_MIX, 0.3f },
                                                                 { ParamIdE::DELAY_FEEDBACK, 0.7f },
                                                                 { ParamIdE::EARLY_LEVEL, 0.5f },
                                                                 { ParamIdE::NEAR_MIX, 0.4f },
                                                                 { ParamIdE::FAR_DIFFUSE, 0.8f },
                                                                 { ParamIdE::FAR_UNMASK, 0.6f },
                                                                 { ParamIdE::FAR_MODE, 3.0f },
                                                                 { ParamIdE::MODAL_LEVEL, 0.6f },
                                                                 { ParamIdE::MODAL_TUNE, 0.7f },
                                                                 { ParamIdE::MODAL_DECAY, 30.0f },
                                                                 { ParamIdE::AUTO_RATE, 6.0f },
                                                                 { ParamIdE::MODAL_BREATH, 0.7f },
                                                                 { ParamIdE::FIELD_MODE, 2.0f },
                                                                 { ParamIdE::FIELD_STRENGTH, 0.8f },
                                                                 { ParamIdE::DISPERSION, 0.4f } } )
            e.setParam( id, v );
        if ( !openOffline( e, true, 47 ) ) return 1;
        const std::vector<float> lr = render( e, 30.0 );
        const bool               finite =
            std::all_of( lr.begin(), lr.end(), []( float x ) { return std::isfinite( x ) && std::fabs( x ) <= 1.0f; } );
        const double db = rmsDb( lr, 10 * 48000, 30 * 48000 );
        check( finite && db > -45.0 && db < -6.0, "effects: everything at once is finite and sane",
               fmt( "%.1f dBFS RMS over 10-30 s, GPU %.3f ms/block", db, e.lastGpuMs() ) );
    }

    // 25. The Scale field gathers the cloud onto the key: 20 cents beside the fourth harmonic (a
    //     pitch of the scale) a wide cloud has energy, gathered it has far less -- at the same level.
    if ( from <= 25 )
    {
        double ratio[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.5f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 60.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::FIELD_MODE, k == 1 ? 1.0f : 0.0f );
            e.setParam( ParamIdE::FIELD_STRENGTH, 0.8f );
            e.setParam( ParamIdE::FIELD_SPRING, 0.0f );
            if ( !openOffline( e, false, 53 ) ) return 1;
            e.noteOn( 62, 0.8f, 0 );
            const std::vector<float> lr = render( e, 4.0 );
            const double             f4 = 4.0 * soundingHz( e );
            ratio[k] =
                toneDb( lr, 96000, 192000, f4 * std::pow( 2.0, 20.0 / 1200.0 ) ) - toneDb( lr, 96000, 192000, f4 );
            level[k] = rmsDb( lr, 96000, 192000 );
        }
        check( ratio[1] < ratio[0] - 8.0 && std::fabs( level[1] - level[0] ) < 1.5,
               "field: Scale gathers the cloud onto the key",
               fmt( "20 ct beside the 4th harmonic: %.1f dB free, %.1f dB gathered; level %.1f / %.1f dBFS", ratio[0],
                    ratio[1], level[0], level[1] ) );
    }

    // 26. Swarm and Scatter: the level stays where it was, and the same seed plays the same (the
    //     attractors' Lorenz systems run on the host, from the seed).
    if ( from <= 26 )
    {
        auto play = []( int mode )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.5f );
            e.setParam( ParamIdE::FIELD_MODE, static_cast<float>( mode ) );
            e.setParam( ParamIdE::FIELD_STRENGTH, 0.9f );
            e.setParam( ParamIdE::FIELD_RADIUS, 200.0f );
            e.setParam( ParamIdE::FIELD_RATE, 0.5f );
            if ( !openOffline( e, false, 59 ) ) return std::vector<float>();
            for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
            return render( e, 6.0 );
        };
        const std::vector<float> off = play( 0 ), swarm = play( 2 ), again = play( 2 ), scatter = play( 3 );
        if ( off.empty() || swarm.empty() || again.empty() || scatter.empty() ) return 1;
        const double l0 = rmsDb( off, 96000, 288000 ), l1 = rmsDb( swarm, 96000, 288000 ),
                     l2 = rmsDb( scatter, 96000, 288000 );
        check( std::fabs( l1 - l0 ) < 2.0 && std::fabs( l2 - l0 ) < 2.0 && swarm == again && swarm != off,
               "field: Swarm and Scatter keep the level, the seed decides",
               fmt( "%.1f dBFS free, %.1f swarm, %.1f scatter; the same seed %s", l0, l1, l2,
                    swarm == again ? "identical" : "DIFFERENT" ) );
    }

    // 27. Dispersion: with a second of it the high partials arrive first and the low ones after --
    //     the eighth harmonic sounds while the fundamental is still to come, and the other way round.
    if ( from <= 27 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::ATTACK, 0.005f );
        e.setParam( ParamIdE::RELEASE, 0.03f );
        e.setParam( ParamIdE::DISPERSION, 1.0f );
        if ( !openOffline( e, false, 61 ) ) return 1;
        e.noteOn( 62, 0.8f, 0 );
        std::vector<float> lr = render( e, 0.3 );
        const double       f1 = soundingHz( e ), f8 = 8.0 * f1;
        e.noteOff( 62, 0 );
        const std::vector<float> rest = render( e, 1.2 );
        lr.insert( lr.end(), rest.begin(), rest.end() );
        // Late by the share of the range above them: the 8th harmonic by about 0.31 s, the fundamental by 0.61 s.
        const double a1 = toneDb( lr, 17280, 26880, f1 ), b1 = toneDb( lr, 31680, 41280, f1 );
        const double a8 = toneDb( lr, 17280, 26880, f8 ), b8 = toneDb( lr, 31680, 41280, f8 );
        check( b1 > a1 + 15.0 && a8 > b8 + 15.0, "dispersion: the highs arrive first, the lows after",
               fmt( "fundamental %.1f -> %.1f dB, 8th harmonic %.1f -> %.1f dB (0.36-0.56 s -> 0.66-0.86 s)", a1, b1,
                    a8, b8 ) );
    }

    // 28. Binaural: in front the ears hear the same; turned to the right (an orbit of a quarter turn
    //     a second, a second on) the right ear hears the highs louder and everything sooner.
    if ( from <= 28 )
    {
        double ild[2] = { 0.0, 0.0 }, itdMs[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::STEREO, 0.0f );
            e.setParam( ParamIdE::BINAURAL, 1.0f );
            e.setParam( ParamIdE::BINAURAL_HEIGHT, 0.0f );
            e.setParam( ParamIdE::BINAURAL_ORBIT, k == 1 ? 0.25f : 0.0f );
            if ( !openOffline( e, false, 67 ) ) return 1;
            e.noteOn( 62, 0.8f, 0 );
            const std::vector<float> lr = render( e, 1.2 );
            const double             f1 = soundingHz( e ), f8 = 8.0 * f1;
            const size_t             a = 43200, b = 52800;   // 0.9 .. 1.1 s
            ild[k] = 20.0 * std::log10( std::abs( toneAt( lr, a, b, f8, 1 ) ) / std::abs( toneAt( lr, a, b, f8, 0 ) ) );
            itdMs[k] = std::arg( toneAt( lr, a, b, f1, 1 ) * std::conj( toneAt( lr, a, b, f1, 0 ) ) ) /
                       ( 2.0 * 3.14159265358979 * f1 ) * 1000.0;
        }
        check( std::fabs( ild[0] ) < 1.0 && std::fabs( itdMs[0] ) < 0.05 && ild[1] > 6.0 && itdMs[1] > 0.3 &&
                   itdMs[1] < 0.9,
               "binaural: in front alike, to the right louder and sooner",
               fmt( "front: %.1f dB, %.2f ms; right: the 8th harmonic %.1f dB louder, the fundamental %.2f ms sooner",
                    ild[0], itdMs[0], ild[1], itdMs[1] ) );
    }

    // 29. The memory: a released chord's ghosts rise over their bloom and are heard long after the
    //     chord; at the moment of release they add almost nothing.
    if ( from <= 29 )
    {
        double early[2] = { 0.0, 0.0 }, late[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.5f );
            e.setParam( ParamIdE::RELEASE, 2.0f );
            e.setParam( ParamIdE::MEMORY_LEVEL, k == 1 ? 0.8f : 0.0f );
            e.setParam( ParamIdE::MEMORY_BLOOM, 4.0f );
            e.setParam( ParamIdE::MEMORY_SINK, 0.0f );
            if ( !openOffline( e, false, 71 ) ) return 1;
            for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
            render( e, 4.0 );
            for ( int note : { 50, 57, 62, 66 } ) e.noteOff( note, 0 );
            const std::vector<float> lr = render( e, 12.0 );
            early[k] = rmsDb( lr, 0, 24000 );
            late[k] = rmsDb( lr, 9 * 48000, 11 * 48000 );
        }
        check( early[1] - early[0] < 3.0 && late[1] - late[0] > 10.0,
               "memory: the ghosts bloom after the chord and linger",
               fmt( "first 0.5 s %.1f -> %.1f dBFS, 9-11 s after %.1f -> %.1f dBFS", early[0], early[1], late[0],
                    late[1] ) );
    }

    // 30. Self-tuning: a C major triad in equal temperament settles towards just intervals -- the
    //     third from 400 towards 386 cents, the fifth from 700 towards 702.
    if ( from <= 30 )
    {
        double third[2] = { 0.0, 0.0 }, fifth[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.3f );
            e.setParam( ParamIdE::SCALE, 0.0f );   // 12-TET
            e.setParam( ParamIdE::TUNE_GRAVITY, k == 1 ? 1.0f : 0.0f );
            e.setParam( ParamIdE::TUNE_ANCHOR, 0.02f );
            if ( !openOffline( e, false, 73 ) ) return 1;
            for ( int note : { 60, 64, 67 } ) e.noteOn( note, 0.8f, 0 );
            render( e, 6.0 );
            const std::vector<double> hz = soundingAll( e );
            if ( hz.size() != 3 )
            {
                check( false, "self-tune: three voices", fmt( "%d sound", static_cast<int>( hz.size() ) ) );
                break;
            }
            third[k] = 1200.0 * std::log2( hz[1] / hz[0] );
            fifth[k] = 1200.0 * std::log2( hz[2] / hz[0] );
        }
        check( std::fabs( third[0] - 400.0 ) < 0.5 && std::fabs( third[1] - 386.3 ) < 5.0 &&
                   std::fabs( fifth[1] - 702.0 ) < 3.0,
               "self-tune: an equal-tempered triad finds just intervals",
               fmt( "third %.1f -> %.1f ct (386.3), fifth %.1f -> %.1f ct (702.0)", third[0], third[1], fifth[0],
                    fifth[1] ) );
    }

    // 31. The spectral echo: its echo comes back at its time; tilted, the low band's echo comes
    //     back before the high band's.
    if ( from <= 31 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::ATTACK, 0.005f );
        e.setParam( ParamIdE::RELEASE, 0.03f );
        e.setParam( ParamIdE::SPEC_MIX, 1.0f );
        e.setParam( ParamIdE::SPEC_TO_FAR, 0.0f );
        e.setParam( ParamIdE::SPEC_TIME, 0.6f );
        e.setParam( ParamIdE::SPEC_TILT, 1.0f );
        e.setParam( ParamIdE::SPEC_SCATTER, 0.0f );
        e.setParam( ParamIdE::SPEC_FEEDBACK, 0.0f );
        if ( !openOffline( e, false, 79 ) ) return 1;
        e.noteOn( 62, 0.8f, 0 );
        std::vector<float> lr = render( e, 0.1 );
        const double       f1 = soundingHz( e ), f8 = 8.0 * f1;
        e.noteOff( 62, 0 );
        const std::vector<float> rest = render( e, 1.4 );
        lr.insert( lr.end(), rest.begin(), rest.end() );
        // Tilted: the fundamental's band back after about 0.52 s, the 8th harmonic's after 0.77 s.
        const double gap = rmsDb( lr, 12000, 20000 );
        const double a1 = toneDb( lr, 24480, 31680, f1 ), b1 = toneDb( lr, 37440, 45120, f1 );
        const double a8 = toneDb( lr, 24480, 31680, f8 ), b8 = toneDb( lr, 37440, 45120, f8 );
        check(
            a1 > b1 + 10.0 && b8 > a8 + 10.0 && rmsDb( lr, 24480, 45120 ) > gap + 10.0,
            "spectral echo: tilted, the lows come back first",
            fmt(
                "fundamental %.1f -> %.1f dB, 8th harmonic %.1f -> %.1f dB (0.51-0.66 s -> 0.78-0.94 s); gap %.1f dBFS",
                a1, b1, a8, b8, gap ) );
    }

    // 32. The painted hall keeps its level (within 3 dB of the unpainted one) while the landscape moves it.
    if ( from <= 32 )
    {
        double level[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.5f );
            e.setParam( ParamIdE::MODAL_LEVEL, 1.0f );
            e.setParam( ParamIdE::MODAL_PAINT, static_cast<float>( k ) );
            e.setParam( ParamIdE::LANDSCAPE, 0.0f );   // Aurora
            if ( !openOffline( e, false, 83 ) ) return 1;
            for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
            const std::vector<float> lr = render( e, 8.0 );
            level[k] = rmsDb( lr, 4 * 48000, 8 * 48000 );
        }
        check( std::fabs( level[1] - level[0] ) < 3.0, "modal hall: painted, it keeps its level",
               fmt( "%.1f dBFS plain, %.1f painted", level[0], level[1] ) );
    }

    // 33. The picture's snapshot: the partials of a sounding note come down from the GPU, at its
    //     harmonics, with their amplitudes; the silent voices' points are empty.
    if ( from <= 33 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::ATTACK, 0.05f );
        if ( !openOffline( e, false, 89 ) ) return 1;
        e.noteOn( 62, 0.8f, 0 );
        render( e, 1.0 );
        std::vector<float> dots;
        int                per = 0;
        const bool         got = e.partialSnapshot( dots, per );
        const double       f0 = soundingHz( e );
        int                voice = -1;
        for ( int i = 0; i < 16; ++i )
            if ( e.voiceView( i ).m_hz > 0.0f ) voice = i;
        int onHarmonic = 0, loud = 0, elsewhere = 0;
        if ( got && per > 0 && voice >= 0 )
        {
            for ( int v = 0; v < static_cast<int>( dots.size() / 2 ) / per; ++v )
                for ( int i = 0; i < per; ++i )
                {
                    const size_t at =
                        2 * ( static_cast<size_t>( v ) * static_cast<size_t>( per ) + static_cast<size_t>( i ) );
                    const float hz = dots[at], a = dots[at + 1];
                    if ( v != voice )
                    {
                        if ( a > 0.0f ) ++elsewhere;
                        continue;
                    }
                    if ( a <= 0.0f ) continue;
                    ++loud;
                    const double h = hz / f0;   // a harmonic's cloud lies within 100 cents of it (the air band is off)
                    if ( std::fabs( 1200.0 * std::log2( h / std::max( 1.0, std::round( h ) ) ) ) < 100.0 ) ++onHarmonic;
                }
        }
        check( got && loud > per / 2 && onHarmonic > loud * 9 / 10 && elsewhere == 0,
               "picture: the partials come down from the GPU",
               fmt( "%d of %d points of the sounding voice heard, %d at its harmonics; %d elsewhere", loud, per,
                    onHarmonic, elsewhere ) );
    }

    // 34. Noctuary's sources move and keep to their ranges: each routed to the brightness, its live
    //     value sampled every half second over 40 s of the conductor.
    if ( from <= 34 )
    {
        const ModSourceE Sources[] = { ModSourceE::LENIA1,     ModSourceE::LORENZ_X, ModSourceE::ROSSLER_Z,
                                       ModSourceE::KURA1,      ModSourceE::ROOT_AGE, ModSourceE::NOTE,
                                       ModSourceE::RANDOM_NOTE };
        std::string      still;
        for ( ModSourceE s : Sources )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::AUTO_RATE, 5.0f );
            e.setParam( ParamIdE::CHAOS_PERIOD, 10.0f );
            e.setParam( ParamIdE::BRIGHTNESS, 0.5f );
            e.setModRow( 0, { s, static_cast<int>( ParamIdE::BRIGHTNESS ), 0.5f } );
            if ( !openOffline( e, true, 97 ) ) return 1;
            float lo = 1e9f, hi = -1e9f;
            bool  sane = true;
            for ( int k = 0; k < 80; ++k )
            {
                render( e, 0.5 );
                const float v = e.playedNorm( ParamIdE::BRIGHTNESS );
                if ( !std::isfinite( v ) || v < -0.001f || v > 1.001f ) sane = false;
                lo = std::min( lo, v );
                hi = std::max( hi, v );
            }
            // Root Age climbs over ten minutes: in forty seconds a fifteenth of the way, at half depth 0.033.
            const float need = s == ModSourceE::ROOT_AGE ? 0.02f : 0.05f;
            if ( !sane || hi - lo < need )
                still += std::string( ModSourceNames[static_cast<int>( s )] ) + fmt( " (%.3f) ", hi - lo );
        }
        check( still.empty(), "modulation: Lenia, chaos, Kuramoto and the piece's sources move",
               still.empty() ? std::string( "all seven swing, all inside the range" ) : still );
    }

    // 35. The Beat: in equal temperament a fifth beats and the source turns; in just intonation it stands.
    if ( from <= 35 )
    {
        float swing[2] = { 0.0f, 0.0f };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.1f );
            e.setParam( ParamIdE::SCALE, k == 0 ? 0.0f : 3.0f );   // 12-TET, JI 7-limit
            e.setParam( ParamIdE::ROOT_NOTE, 0.0f );               // C: C and G a pure fifth in the just scale
            e.setParam( ParamIdE::PURITY, 1.0f );
            e.setParam( ParamIdE::BRIGHTNESS, 0.5f );
            e.setModRow( 0, { ModSourceE::BEAT, static_cast<int>( ParamIdE::BRIGHTNESS ), 0.5f } );
            if ( !openOffline( e, false, 101 ) ) return 1;
            e.noteOn( 60, 0.8f, 0 );
            e.noteOn( 67, 0.8f, 0 );
            render( e, 6.0 );
            float lo = 1e9f, hi = -1e9f;
            for ( int i = 0; i < 40; ++i )
            {
                render( e, 0.1 );
                lo = std::min( lo, e.playedNorm( ParamIdE::BRIGHTNESS ) );
                hi = std::max( hi, e.playedNorm( ParamIdE::BRIGHTNESS ) );
            }
            swing[k] = hi - lo;
        }
        check( swing[0] > 0.2f && swing[1] < 0.02f,
               "modulation: the Beat turns when the fifth beats, stands when it is pure",
               fmt( "swing %.3f in 12-TET, %.3f in just intonation", swing[0], swing[1] ) );
    }

    // 36. The via source scales the depth, and survives the text form.
    if ( from <= 36 )
    {
        float swing[2] = { 0.0f, 0.0f };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::LFO1_RATE, 1.0f );
            e.setParam( ParamIdE::BRIGHTNESS, 0.5f );
            e.setParam( ParamIdE::MACRO1, static_cast<float>( k ) );
            ModRow r{ ModSourceE::LFO1, static_cast<int>( ParamIdE::BRIGHTNESS ), 0.5f };
            r.m_via = ModSourceE::MACRO1;
            e.setModRow( 0, r );
            if ( !openOffline( e, false, 103 ) ) return 1;
            float lo = 1e9f, hi = -1e9f;
            for ( int i = 0; i < 30; ++i )
            {
                render( e, 0.1 );
                lo = std::min( lo, e.playedNorm( ParamIdE::BRIGHTNESS ) );
                hi = std::max( hi, e.playedNorm( ParamIdE::BRIGHTNESS ) );
            }
            swing[k] = hi - lo;
        }
        ModRow rows[2];
        modRowsFromText( "lfo1>cutoff:0.35:lenia2;kura3>drift:-0.2", rows, 2 );
        const std::string back = modRowsToText( rows, 2 );
        check( swing[0] < 0.01f && swing[1] > 0.5f && rows[0].m_via == ModSourceE::LENIA2 &&
                   rows[1].m_via == ModSourceE::OFF && back == "lfo1>cutoff:0.35:lenia2;kura3>drift:-0.2",
               "modulation: a via source scales the depth",
               fmt( "swing %.3f at Macro 1 = 0, %.3f at 1; text '%s'", swing[0], swing[1], back.c_str() ) );
    }

    // 37. The living landscapes: Lenia and the Turing field live (the field at 6 s is not the field
    //     at 3 s), keep the level within 2 dB of no field, and the same seed plays the same.
    if ( from <= 37 )
    {
        std::string report;
        bool        good = true;
        double      level0 = 0.0;
        for ( int mode = 0; mode < 3; ++mode )
        {
            auto play = [&]( std::vector<float> &f3, std::vector<float> &f6 )
            {
                Engine e;
                quietEffects( e );
                e.setParam( ParamIdE::ATTACK, 0.5f );
                e.setParam( ParamIdE::LIFE_MODE, static_cast<float>( mode ) );
                e.setParam( ParamIdE::LIFE_DEPTH, 0.8f );
                e.setParam( ParamIdE::LIFE_RATE, 0.5f );
                if ( !openOffline( e, false, 107 ) ) return std::vector<float>();
                for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
                std::vector<float> lr = render( e, 3.0 );
                e.lifeSnapshot( f3 );
                const std::vector<float> rest = render( e, 3.0 );
                e.lifeSnapshot( f6 );
                lr.insert( lr.end(), rest.begin(), rest.end() );
                return lr;
            };
            std::vector<float>       a3, a6, b3, b6;
            const std::vector<float> a = play( a3, a6 ), b = play( b3, b6 );
            if ( a.empty() ) return 1;
            const double level = rmsDb( a, 3 * 48000, 6 * 48000 );
            if ( mode == 0 )
            {
                level0 = level;
                continue;
            }
            double moved = 0.0;
            for ( size_t i = 0; i < a3.size() && i < a6.size(); ++i ) moved += std::fabs( a6[i] - a3[i] );
            moved /= std::max<size_t>( 1, a3.size() );
            const bool ok = a3.size() == static_cast<size_t>( LifeW * LifeH ) && moved > 0.01 &&
                            std::fabs( level - level0 ) < 2.0 && a == b;
            good = good && ok;
            report += fmt( "%s: moved %.3f a cell, %.1f dBFS (free %.1f)%s; ", LifeModeNames[mode], moved, level,
                           level0, a == b ? "" : ", NOT identical" );
        }
        check( good, "life: Lenia and Turing live, keep the level, repeat", report );
    }

    // 38. Partial sync: coupled, the members of a harmonic fall into step and the harmonic pulses as
    //     one -- its level in tenths of a second swings far more than uncoupled -- at the same level.
    if ( from <= 38 )
    {
        double sd[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::SYNC_DEPTH, 1.0f );
            e.setParam( ParamIdE::SYNC_RATE, 1.0f );
            e.setParam( ParamIdE::SYNC_COUPLING, static_cast<float>( k ) );
            if ( !openOffline( e, false, 109 ) ) return 1;
            e.noteOn( 62, 0.8f, 0 );
            const std::vector<float> lr = render( e, 12.0 );
            const double             f8 = 8.0 * soundingHz( e );
            double                   s = 0.0, s2 = 0.0;
            const int                windows = 60;
            for ( int w = 0; w < windows; ++w )
            {
                const size_t a = static_cast<size_t>( 6 * 48000 + w * 4800 );
                const double db = toneDb( lr, a, a + 4800, f8 );
                s += db;
                s2 += db * db;
            }
            sd[k] = std::sqrt( std::max( 0.0, s2 / windows - ( s / windows ) * ( s / windows ) ) );
            level[k] = rmsDb( lr, 6 * 48000, 12 * 48000 );
        }
        check( sd[1] > sd[0] + 2.0 && std::fabs( level[1] - level[0] ) < 1.5,
               "sync: coupled, a harmonic's members pulse as one",
               fmt( "8th harmonic's swing %.1f dB free, %.1f dB coupled; level %.1f / %.1f dBFS", sd[0], sd[1],
                    level[0], level[1] ) );
    }

    // 39. What it hears of itself: louder with a note than in silence, brighter for a bright spectrum,
    //     more change at an attack than in a held note, rougher for a minor second than a pure fifth.
    if ( from <= 39 )
    {
        // A source's value through a row onto the brightness (depth 1 from 0): the live value is the source.
        auto heard = []( ModSourceE s, const std::vector<int> &notes, float tilt, double seconds, float attack,
                         float scale = 3.0f )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, attack );
            e.setParam( ParamIdE::TILT, tilt );
            e.setParam( ParamIdE::SCALE,
                        scale );   // JI 7-limit on C (C and G pure); 12-TET where a semitone must be a semitone
            e.setParam( ParamIdE::ROOT_NOTE, 0.0f );
            e.setParam( ParamIdE::BRIGHTNESS, 0.0f );
            e.setModRow( 0, { s, static_cast<int>( ParamIdE::BRIGHTNESS ), 1.0f } );
            if ( !openOffline( e, false, 113 ) ) return -1.0f;
            for ( int note : notes ) e.noteOn( note, 0.8f, 0 );
            render( e, seconds );
            return e.playedNorm( ParamIdE::BRIGHTNESS );
        };
        const float quiet = heard( ModSourceE::LOUDNESS, {}, 1.2f, 2.0, 0.05f ),
                    loud = heard( ModSourceE::LOUDNESS, { 60, 64, 67 }, 1.2f, 2.0, 0.05f );
        const float dark = heard( ModSourceE::CENTROID, { 48 }, 3.0f, 2.0, 0.05f ),
                    bright = heard( ModSourceE::CENTROID, { 48 }, 0.3f, 2.0, 0.05f );
        // Change right after the attack (the attack's rise lasts a frame or two, the source follows over a fifth of a second).
        const float onset = heard( ModSourceE::FLUX, { 60 }, 1.2f, 0.06, 0.005f ),
                    held = heard( ModSourceE::FLUX, { 60 }, 1.2f, 4.0, 0.005f );
        // In 12-TET: the just scale has no C sharp of its own, so C and C sharp on the keys would be a unison.
        const float pure = heard( ModSourceE::ROUGHNESS, { 60, 67 }, 1.2f, 3.0, 0.05f, 0.0f ),
                    rough = heard( ModSourceE::ROUGHNESS, { 60, 61 }, 1.2f, 3.0, 0.05f, 0.0f );
        check(
            loud > quiet + 0.3f && bright > dark + 0.1f && onset > held + 0.03f && rough > pure + 0.2f,
            "listening: loudness, brightness, change, roughness",
            fmt(
                "loudness %.2f -> %.2f, brightness %.2f -> %.2f, change %.2f at the attack / %.2f held, roughness %.2f fifth / %.2f second",
                quiet, loud, dark, bright, onset, held, pure, rough ) );
    }

    // 40. The flock: a cloud's members move together -- the centre of the 4th harmonic's cloud wanders
    //     over ten seconds by several cents -- but keep their spread (no collapse), at the same level.
    if ( from <= 40 )
    {
        double moved[2] = { 0.0, 0.0 }, spread[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::FIELD_MODE, k == 1 ? 4.0f : 0.0f );   // Off, Flock
            e.setParam( ParamIdE::FIELD_STRENGTH, 0.8f );
            e.setParam( ParamIdE::FIELD_RATE, 0.3f );
            if ( !openOffline( e, false, 127 ) ) return 1;
            e.noteOn( 62, 0.8f, 0 );
            // The 4th harmonic's points in the snapshot: their mean and spread in cents from it.
            auto cloud4 = [&]( double &mean, double &sd )
            {
                std::vector<float> dots;
                int                per = 0;
                e.partialSnapshot( dots, per );
                const double f4 = 4.0 * soundingHz( e );
                int          voice = -1;
                for ( int i = 0; i < 16; ++i )
                    if ( e.voiceView( i ).m_hz > 0.0f ) voice = i;
                double s = 0.0, s2 = 0.0;
                int    count = 0;
                for ( int i = 0; voice >= 0 && i < per; ++i )
                {
                    const size_t at = 2 * ( static_cast<size_t>( voice ) * per + i );
                    if ( dots[at + 1] <= 0.0f ) continue;
                    const double c = 1200.0 * std::log2( dots[at] / f4 );
                    if ( std::fabs( c ) > 300.0 ) continue;   // another harmonic's
                    s += c;
                    s2 += c * c;
                    ++count;
                }
                mean = count ? s / count : 0.0;
                sd = count ? std::sqrt( std::max( 0.0, s2 / count - mean * mean ) ) : 0.0;
            };
            // The centre twenty times over ten seconds: its span is how far the cloud has wandered.
            std::vector<float> lr = render( e, 2.0 );
            double             lo = 1e9, hi = -1e9, sd = 0.0;
            for ( int w = 0; w < 20; ++w )
            {
                const std::vector<float> part = render( e, 0.5 );
                lr.insert( lr.end(), part.begin(), part.end() );
                double m = 0.0;
                cloud4( m, sd );
                lo = std::min( lo, m );
                hi = std::max( hi, m );
            }
            moved[k] = hi - lo;
            spread[k] = sd;
            level[k] = rmsDb( lr, 2 * 48000, 12 * 48000 );
        }
        check(
            moved[1] > 5.0 && moved[1] > moved[0] + 4.0 && spread[1] > 0.5 * spread[0] &&
                std::fabs( level[1] - level[0] ) < 1.5,
            "field: Flock moves the clouds together, keeping them wide",
            fmt(
                "4th harmonic's centre wandered over %.1f ct (still %.1f), spread %.1f ct (still %.1f), level %.1f / %.1f dBFS",
                moved[1], moved[0], spread[1], spread[0], level[1], level[0] ) );
    }

    // 41. The ripple: notes struck one after another send waves along the spectrum, and a high
    //     harmonic's loudness rides them -- it swings far more than without -- at the same level.
    if ( from <= 41 )
    {
        double sd[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 0.0f );   // one steady sine per harmonic: whatever moves is the ripple
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::RIPPLE_DEPTH, static_cast<float>( k ) );
            e.setParam( ParamIdE::RIPPLE_SPEED, 1.5f );
            e.setParam( ParamIdE::RIPPLE_DECAY, 8.0f );
            if ( !openOffline( e, false, 131 ) ) return 1;
            e.noteOn( 50, 0.8f, 0 );
            std::vector<float> lr = render( e, 1.0 );
            const double       f4 = 4.0 * soundingHz( e );
            for ( int note : { 57, 62, 66 } )
            {   // more strikes, a second apart
                e.noteOn( note, 0.8f, 0 );
                const std::vector<float> part = render( e, 1.0 );
                lr.insert( lr.end(), part.begin(), part.end() );
            }
            const std::vector<float> rest = render( e, 4.0 );
            lr.insert( lr.end(), rest.begin(), rest.end() );
            // Quarter seconds: a passing wave takes about that long; the cloud's own beating, shorter, averages out.
            double    s = 0.0, s2 = 0.0;
            const int windows = 24;
            for ( int w = 0; w < windows; ++w )
            {
                const size_t a = static_cast<size_t>( 1 * 48000 + w * 12000 );
                const double db = toneDb( lr, a, a + 12000, f4 );
                s += db;
                s2 += db * db;
            }
            sd[k] = std::sqrt( std::max( 0.0, s2 / windows - ( s / windows ) * ( s / windows ) ) );
            level[k] = rmsDb( lr, 4 * 48000, 8 * 48000 );
        }
        check(
            sd[1] > sd[0] + 1.5 && std::fabs( level[1] - level[0] ) < 1.5,
            "ripple: the strikes' waves move the partials",
            fmt( "4th harmonic of the first note swings %.1f dB still, %.1f dB on the ripple; level %.1f / %.1f dBFS",
                 sd[0], sd[1], level[0], level[1] ) );
    }

    // 42. The reservoir: listening to the conductor its readout moves, inside its range, the same on
    //     every run; left alone it comes to rest when its chaos is low and runs on by itself when high.
    if ( from <= 42 )
    {
        auto run = [&]( float chaos, float input, bool conductor, std::vector<float> &seq )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::AUTO_RATE, 5.0f );
            e.setParam( ParamIdE::BRIGHTNESS, 0.5f );
            e.setParam( ParamIdE::RESERVOIR_CHAOS, chaos );
            e.setParam( ParamIdE::RESERVOIR_INPUT, input );
            e.setParam( ParamIdE::RESERVOIR_MEMORY, 1.0f );
            e.setModRow( 0, { ModSourceE::RESERVOIR1, static_cast<int>( ParamIdE::BRIGHTNESS ), 0.5f } );
            if ( !openOffline( e, conductor, 137 ) ) return false;
            render( e, 10.0 );
            for ( int k = 0; k < 60; ++k )
            {
                render( e, 0.5 );
                seq.push_back( e.playedNorm( ParamIdE::BRIGHTNESS ) );
            }
            return true;
        };
        auto swing = []( const std::vector<float> &s )
        {
            float lo = 1e9f, hi = -1e9f;
            for ( float v : s )
            {
                lo = std::min( lo, v );
                hi = std::max( hi, v );
            }
            return hi - lo;
        };
        std::vector<float> a, b, calm, wild;
        if ( !run( 0.95f, 1.0f, true, a ) || !run( 0.95f, 1.0f, true, b ) || !run( 0.6f, 0.0f, false, calm ) ||
             !run( 1.4f, 0.0f, false, wild ) )
            return 1;
        bool sane = true;
        for ( float v : a )
            if ( !std::isfinite( v ) || v < -0.001f || v > 1.001f ) sane = false;
        check( sane && a == b && swing( a ) > 0.05f && swing( calm ) < 0.01f && swing( wild ) > 0.05f,
               "modulation: the reservoir listens, rests and runs on",
               fmt( "swing %.3f with the conductor (%s on a second run); alone %.4f at chaos 0.6, %.3f at 1.4",
                    swing( a ), a == b ? "the same" : "DIFFERENT", swing( calm ), swing( wild ) ) );
    }

    // 43. The melt: a note let go lingers in it, about as loud as it was; without spread it stays on
    //     its harmonics, with spread it melts into the space between them.
    if ( from <= 43 )
    {
        double dry = 0.0, tail[2] = { 0.0, 0.0 }, between[2] = { 0.0, 0.0 }, silent = 0.0;
        for ( int k = 0; k < 3; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.005f );
            e.setParam( ParamIdE::RELEASE, 0.03f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::MELT_MIX, k == 0 ? 0.0f : 1.0f );
            e.setParam( ParamIdE::MELT_TO_FAR, 0.0f );
            e.setParam( ParamIdE::MELT_TIME, 20.0f );
            e.setParam( ParamIdE::MELT_SPREAD, k == 2 ? 1.5f : 0.0f );
            if ( !openOffline( e, false, 139 ) ) return 1;
            e.noteOn( 50, 0.8f, 0 );
            std::vector<float> lr = render( e, 3.0 );
            const double       f1 = soundingHz( e );
            e.noteOff( 50, 0 );
            const std::vector<float> rest = render( e, 2.0 );
            lr.insert( lr.end(), rest.begin(), rest.end() );
            if ( k == 0 )
            {
                dry = rmsDb( lr, 2 * 48000, 3 * 48000 );
                silent = rmsDb( lr, 3 * 48000 + 9600, 4 * 48000 );
                continue;
            }
            const size_t a = 3 * 48000 + 9600, b = 4 * 48000;   // 0.2 .. 1 s after the release
            tail[k - 1] = rmsDb( lr, a, b );
            // Between the 3rd and 4th harmonics against on them.
            between[k - 1] =
                toneDb( lr, a, b, 3.5 * f1 ) - 0.5 * ( toneDb( lr, a, b, 3.0 * f1 ) + toneDb( lr, a, b, 4.0 * f1 ) );
        }
        check(
            tail[0] > silent + 30.0 && std::fabs( tail[0] - dry ) < 4.0 && std::fabs( tail[1] - dry ) < 4.0 &&
                between[1] > between[0] + 10.0,
            "melt: a note lingers, and melts between its harmonics",
            fmt(
                "held %.1f dBFS, after the release %.1f / %.1f dBFS spread (without the melt %.1f); between the harmonics %.1f dB still, %.1f dB spread",
                dry, tail[0], tail[1], silent, between[0], between[1] ) );
    }

    // 44. The glide: one sound becomes another. Every number on its way in its own domain, the
    //     filter fading in by its share, the motion through nothing where it changes, the level
    //     stepping no more than either sound does by itself; and at the end exactly the new sound.
    if ( from <= 44 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::ATTACK, 0.2f );
        e.setParam( ParamIdE::TILT, 0.6f );
        e.setParam( ParamIdE::BRIGHTNESS, 0.9f );
        e.setParam( ParamIdE::CLOUD_WIDTH, 10.0f );
        e.setParam( ParamIdE::MOTION, 4.0f );   // Breathe
        e.setParam( ParamIdE::MOTION_DEPTH, 0.8f );
        e.setParam( ParamIdE::SCALE, 0.0f );   // 12-TET
        if ( !openOffline( e, false, 141 ) ) return 1;
        e.noteOn( 50, 0.8f, 0 );
        e.noteOn( 57, 0.8f, 0 );
        render( e, 1.0 );
        // The level in windows of a tenth of a second: its largest step from one to the next.
        double prevDb = 0.0;
        bool   first = true;
        auto   window = [&]( double &worst )
        {
            const std::vector<float> part = render( e, 0.1 );
            const double             db = rmsDb( part, 0, part.size() / 2 );
            if ( !first ) worst = std::max( worst, std::fabs( db - prevDb ) );
            first = false;
            prevDb = db;
            return db;
        };
        double stepA = 0.0, stepGlide = 0.0, stepB = 0.0;
        for ( int k = 0; k < 10; ++k ) window( stepA );
        const ParamDesc &tilt = paramDesc( ParamIdE::TILT );
        e.beginMorph( 4.0 );
        e.setParam( ParamIdE::TILT, 2.0f );
        e.setParam( ParamIdE::BRIGHTNESS, 0.4f );
        e.setParam( ParamIdE::CLOUD_WIDTH, 40.0f );
        e.setParam( ParamIdE::FILTER_MODEL, 3.0f );   // LP 24
        e.setParam( ParamIdE::CUTOFF, 600.0f );
        e.setParam( ParamIdE::MOTION, 6.0f );   // Aurora
        e.setParam( ParamIdE::MOTION_DEPTH, 0.6f );
        e.setParam( ParamIdE::SCALE, 3.0f );   // JI 7-limit
        double worstTilt = 0.0, worstMix = 0.0, depthMid = 1.0;
        bool   switchedAtMiddle = true;
        for ( int k = 0; k < 45; ++k )
        {   // until it arrives (the request is served by the next block)
            window( stepGlide );
            const float w = morphCurve( e.morphProgress() );
            const float want =
                fromNorm( tilt, toNorm( tilt, 0.6f ) + ( toNorm( tilt, 2.0f ) - toNorm( tilt, 0.6f ) ) * w );
            worstTilt =
                std::max( worstTilt, static_cast<double>( std::fabs( e.soundValue( ParamIdE::TILT ) - want ) ) );
            worstMix = std::max( worstMix, static_cast<double>( std::fabs(
                                               e.soundFilterMix() - ( e.morphProgress() < 1.0f ? w : 1.0f ) ) ) );
            if ( std::fabs( w - 0.5f ) < 0.06f )
                depthMid = std::min( depthMid, static_cast<double>( e.soundValue( ParamIdE::MOTION_DEPTH ) ) );
            const bool aurora = std::lround( e.soundValue( ParamIdE::MOTION ) ) == 6;
            const bool pure = std::lround( e.soundValue( ParamIdE::SCALE ) ) == 3;
            if ( e.morphProgress() < 1.0f && ( aurora != ( w >= 0.5f ) || pure != ( w >= 0.5f ) ) )
                switchedAtMiddle = false;
            if ( e.morphProgress() >= 1.0f ) break;
        }
        const bool arrived = e.morphProgress() >= 1.0f && e.soundValue( ParamIdE::TILT ) == 2.0f &&
                             e.soundValue( ParamIdE::CUTOFF ) == 600.0f && e.soundFilterMix() == 1.0f &&
                             e.soundValue( ParamIdE::MOTION ) == 6.0f &&
                             e.soundValue( ParamIdE::MOTION_DEPTH ) == 0.6f && e.soundValue( ParamIdE::SCALE ) == 3.0f;
        for ( int k = 0; k < 60; ++k ) window( stepB );
        check(
            arrived && worstTilt < 1.0e-4 && worstMix < 1.0e-4 && depthMid < 0.1 && switchedAtMiddle &&
                stepGlide < std::max( stepA, stepB ) + 1.5,
            "morph: a glide from one sound into another",
            fmt(
                "tilt on its way within %.1e, the filter's share within %.1e, motion depth %.2f at the middle (motion and scale switched there: %s); "
                "largest step between tenths of a second %.1f dB gliding, %.1f before, %.1f after; arrived: %s",
                worstTilt, worstMix, depthMid, switchedAtMiddle ? "yes" : "NO", stepGlide, stepA, stepB,
                arrived ? "yes" : "NO" ) );
    }

    // 45. The A/B morph: the sound is the blend of two slots at A - B (matrix included); switching
    //     it on and off glides; off, the knobs play again.
    if ( from <= 45 )
    {
        Engine e;
        quietEffects( e );
        if ( !openOffline( e, false, 143 ) ) return 1;
        float a[NumParams], b[NumParams];
        for ( const ParamDesc &d : paramTable() )
            a[static_cast<size_t>( d.m_id )] = b[static_cast<size_t>( d.m_id )] = d.m_def;
        a[static_cast<size_t>( ParamIdE::TILT )] = 0.5f;
        b[static_cast<size_t>( ParamIdE::TILT )] = 2.5f;
        b[static_cast<size_t>( ParamIdE::FILTER_MODEL )] = 2.0f;   // LP 12
        b[static_cast<size_t>( ParamIdE::CUTOFF )] = 800.0f;
        e.setMorphSlot( 0, a, "lfo1>cutoff:0.3" );
        e.setMorphSlot( 1, b, "lfo2>brightness:0.4" );
        e.setParam( ParamIdE::MORPH_POS, 0.25f );
        e.setParam( ParamIdE::MORPH_ACTIVE, 1.0f );
        e.noteOn( 55, 0.8f, 0 );
        render( e, 1.5 );   // switching on glides a second
        const ParamDesc &tilt = paramDesc( ParamIdE::TILT );
        const float      want =
            fromNorm( tilt, toNorm( tilt, 0.5f ) + ( toNorm( tilt, 2.5f ) - toNorm( tilt, 0.5f ) ) * 0.25f );
        const float tiltQuarter = e.soundValue( ParamIdE::TILT ), mixQuarter = e.soundFilterMix();
        const bool  quarter = std::fabs( tiltQuarter - want ) < 1.0e-4f && std::fabs( mixQuarter - 0.25f ) < 1.0e-4f &&
                              e.soundValue( ParamIdE::FILTER_MODEL ) == 2.0f;
        e.setParam( ParamIdE::MORPH_POS, 1.0f );
        render( e, 0.1 );
        const bool atB = e.soundValue( ParamIdE::TILT ) == 2.5f && e.soundFilterMix() == 1.0f;
        e.setParam( ParamIdE::MORPH_ACTIVE, 0.0f );
        render( e, 0.5 );
        const float halfway = e.soundValue( ParamIdE::TILT );
        render( e, 1.0 );
        const bool back = e.soundValue( ParamIdE::TILT ) == e.param( ParamIdE::TILT ) &&
                          halfway != e.param( ParamIdE::TILT ) && halfway != 2.5f;
        check(
            quarter && atB && back, "morph: A/B between two slots, gliding on and off",
            fmt(
                "at 0.25 tilt %.3f (want %.3f), filter share %.2f; at B %s; off: %.3f half a second later, then the knob's %.3f",
                tiltQuarter, want, mixQuarter, atB ? "exact" : "NOT", halfway, e.param( ParamIdE::TILT ) ) );
    }

    // 46. The journeys: every one that comes with the instrument reads, names presets the bank has
    //     and reads back from its text; a seeded player repeats; a journey played offline glides into
    //     each of its presets in turn.
    if ( from <= 46 )
    {
        int bad = 0, unknown = 0, steps = 0;
        for ( int i = 0; i < numBuiltinJourneys(); ++i )
        {
            Journey j, k;
            if ( !j.parse( builtinJourneyText( i ) ) || j.m_steps.empty() || !k.parse( j.text().c_str() ) ||
                 k.m_steps.size() != j.m_steps.size() )
                ++bad;
            for ( const JourneyStep &st : j.m_steps )
            {
                ++steps;
                if ( findPreset( st.m_preset.c_str() ) < 0 ) ++unknown;
            }
        }
        Journey j;
        j.parse(
            "journey test\ncyclic on\nVelvet Dreaming | 0:03 | 0:01\nTeeming Garden | 0:03 | 0:01\nFading Frost | 0:03-0:04 | 0:01-0:02\n" );
        JourneyPlayer p1, p2;
        p1.start( j, 7 );
        p2.start( j, 7 );
        bool same = true;
        for ( int k = 0; k < 40; ++k )
        {
            JourneyStep s1, s2;
            double      f1 = 0.0, f2 = 0.0;
            const bool  b1 = p1.advance( 0.5, s1, f1 ), b2 = p2.advance( 0.5, s2, f2 );
            if ( b1 != b2 || ( b1 && ( s1.m_preset != s2.m_preset || f1 != f2 ) ) ) same = false;
        }
        // Played: each step's preset reached (its tilt and its master gain) before the next begins.
        Engine e;
        if ( !openOffline( e, true, 149 ) ) return 1;
        JourneyPlayer p;
        p.start( j, 11 );
        int  reached = 0, begun = 0, current = -1;
        bool checked = true;
        for ( int k = 0; k < 140; ++k )
        {
            JourneyStep st;
            double      fade = 0.0;
            if ( p.advance( 0.1, st, fade ) )
            {
                if ( current >= 0 && checked ) ++reached;
                current = findPreset( st.m_preset.c_str() );
                e.loadPreset( current, fade );
                ++begun;
                checked = false;
            }
            render( e, 0.1 );
            if ( !checked && current >= 0 && e.morphProgress() >= 1.0f )
            {
                float values[NumParams];
                presetValues( current, values, nullptr );
                checked = e.soundValue( ParamIdE::TILT ) == values[static_cast<size_t>( ParamIdE::TILT )] &&
                          e.soundValue( ParamIdE::MASTER_GAIN ) == values[static_cast<size_t>( ParamIdE::MASTER_GAIN )];
            }
        }
        check(
            bad == 0 && unknown == 0 && numBuiltinJourneys() >= 68 && same && begun >= 4 && reached >= 3 &&
                p.passes() >= 1,
            "journeys: read, repeatable, played as glides",
            fmt(
                "%d journeys, %d steps, %d unreadable, %d unknown presets; seeded players %s; played: %d steps begun, %d reached, %d passes",
                numBuiltinJourneys(), steps, bad, unknown, same ? "agree" : "DIFFER", begun, reached, p.passes() ) );
    }

    // 47. Stretch and Shift: stretched by 60 cents an octave, the 4th harmonic sits at 4^1.05 of the
    //     fundamental, not at 4; shifted by 50 Hz, the fundamental sits 50 Hz up.
    if ( from <= 47 )
    {
        double stretched[2] = { 0.0, 0.0 }, shifted[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::SCALE, 0.0f );
            e.setParam( ParamIdE::STRETCH, k == 0 ? 1200.0f : 1260.0f );
            if ( !openOffline( e, false, 151 ) ) return 1;
            e.noteOn( 48, 0.8f, 0 );
            const std::vector<float> lr = render( e, 2.0 );
            const double             f0 = soundingHz( e );
            stretched[k] =
                toneDb( lr, 48000, 96000, f0 * std::pow( 4.0, 1.05 ) ) - toneDb( lr, 48000, 96000, 4.0 * f0 );
        }
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::SCALE, 0.0f );
            e.setParam( ParamIdE::SHIFT, k == 0 ? 0.0f : 50.0f );
            if ( !openOffline( e, false, 153 ) ) return 1;
            e.noteOn( 48, 0.8f, 0 );
            const std::vector<float> lr = render( e, 2.0 );
            const double             f0 = soundingHz( e );
            shifted[k] = toneDb( lr, 48000, 96000, f0 + 50.0 ) - toneDb( lr, 48000, 96000, f0 );
        }
        check(
            stretched[1] > 10.0 && stretched[0] < -10.0 && shifted[1] > 10.0 && shifted[0] < -10.0,
            "spectrum: Stretch and Shift move the partials",
            fmt(
                "4th harmonic at 4^1.05 against 4: %.1f dB harmonic, %.1f dB stretched; fundamental +50 Hz against itself: %.1f dB, %.1f dB shifted",
                stretched[0], stretched[1], shifted[0], shifted[1] ) );
    }

    // 48. The spectral comb: standing, it cuts some harmonics deep and leaves others; sweeping, a
    //     harmonic dives and surfaces as the notches pass; over whole sweeps the level stays (one
    //     note standing is louder or softer as its fundamental meets a crest or a notch).
    if ( from <= 48 )
    {
        double spread = 0.0, swing = 0.0, level[2] = { 0.0, 0.0 }, swept = 0.0;
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::COMB_DEPTH, static_cast<float>( k ) );
            e.setParam( ParamIdE::COMB_DENSITY, 1.0f );
            e.setParam( ParamIdE::COMB_SWEEP, 0.0f );
            e.setParam( ParamIdE::COMB_STEREO, 0.0f );
            if ( !openOffline( e, false, 157 ) ) return 1;
            e.noteOn( 48, 0.8f, 0 );
            const std::vector<float> lr = render( e, 2.0 );
            const double             f0 = soundingHz( e );
            level[k] = rmsDb( lr, 48000, 96000 );   // standing: for the message
            if ( k == 1 )
            {
                // Against the same note without the comb: how far apart the harmonics are cut.
                Engine o;
                quietEffects( o );
                o.setParam( ParamIdE::MOTION, 0.0f );
                o.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
                o.setParam( ParamIdE::SHIMMER, 0.0f );
                if ( !openOffline( o, false, 157 ) ) return 1;
                o.noteOn( 48, 0.8f, 0 );
                const std::vector<float> ref = render( o, 2.0 );
                double                   lo = 1e9, hi = -1e9;
                for ( int h = 2; h <= 9; ++h )
                {
                    const double d = toneDb( lr, 48000, 96000, h * f0 ) - toneDb( ref, 48000, 96000, h * f0 );
                    lo = std::min( lo, d );
                    hi = std::max( hi, d );
                }
                spread = hi - lo;
            }
        }
        double plain = 0.0;
        for ( int k = 0; k < 2; ++k )
        {   // the same note, the same seed, with and without a sweeping comb
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::COMB_DEPTH, static_cast<float>( k ) );
            e.setParam( ParamIdE::COMB_DENSITY, 1.0f );
            e.setParam( ParamIdE::COMB_SWEEP, 0.25f );
            e.setParam( ParamIdE::COMB_STEREO, 0.0f );
            if ( !openOffline( e, false, 159 ) ) return 1;
            e.noteOn( 48, 0.8f, 0 );
            const std::vector<float> lr = render( e, 9.0 );
            ( k == 0 ? plain : swept ) =
                rmsDb( lr, 48000, 9 * 48000 );   // two whole sweeps (a notch an octave, a quarter notch a second)
            if ( k == 0 ) continue;
            const double f4 = 4.0 * soundingHz( e );
            double       lo = 1e9, hi = -1e9;
            for ( int w = 0; w < 40; ++w )
            {
                const size_t a = static_cast<size_t>( 48000 + w * 9600 );
                const double db = toneDb( lr, a, a + 4800, f4 );
                lo = std::min( lo, db );
                hi = std::max( hi, db );
            }
            swing = hi - lo;
        }
        check(
            spread > 15.0 && swing > 15.0 && std::fabs( swept - plain ) < 1.0,
            "comb: notches across the harmonics, travelling",
            fmt(
                "standing: harmonics 2-9 cut %.1f dB apart (one note %.1f dBFS against %.1f); sweeping: the 4th swings %.1f dB; "
                "over two sweeps %.1f dBFS against %.1f",
                spread, level[1], level[0], swing, swept, plain ) );
    }

    // 49. The player's own landscape: one that is bright only around 1 kHz, chosen as Your Own and
    //     read by Aurora, lifts the 8th harmonic (near 1 kHz) far above the 2nd.
    if ( from <= 49 )
    {
        double tilt[2] = { 0.0, 0.0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::MOTION, k == 0 ? 0.0f : 6.0f );   // None, Aurora
            e.setParam( ParamIdE::MOTION_DEPTH, 1.0f );
            e.setParam( ParamIdE::LANDSCAPE, static_cast<float>( UserLandscape ) );
            Landscape l;
            l.m_columns = 256;
            l.m_rows = 64;
            l.m_rowSeconds = 0.5f;
            l.m_amps.assign( static_cast<size_t>( l.m_columns ) * l.m_rows, 0.01f );
            for ( int r = 0; r < l.m_rows; ++r )
                for ( int c = 0; c < l.m_columns; ++c )
                    if ( std::fabs( std::log2( landscapeColumnHz( c, l.m_columns ) / 1000.0f ) ) < 0.25f )
                        l.m_amps[static_cast<size_t>( r ) * l.m_columns + c] = 1.0f;
            e.setUserLandscape( l );
            if ( !openOffline( e, false, 161 ) ) return 1;
            e.noteOn( 48, 0.8f, 0 );
            const std::vector<float> lr = render( e, 3.0 );
            const double             f0 = soundingHz( e );
            tilt[k] = toneDb( lr, 96000, 144000, 8.0 * f0 ) - toneDb( lr, 96000, 144000, 2.0 * f0 );
        }
        check( tilt[1] > tilt[0] + 15.0, "landscape: the player's own is read",
               fmt( "8th harmonic against the 2nd: %.1f dB without, %.1f dB reading a landscape bright at 1 kHz",
                    tilt[0], tilt[1] ) );
    }

    // 50. The animated pictures: every one moves (the field at 6 s is not the field at 3 s), stays
    //     finite, keeps the level within 2 dB of no field, and the same seed plays the same.
    if ( from <= 50 )
    {
        std::string report;
        bool        good = true;
        double      level0 = 0.0;
        for ( int mode : { 0, 3, 4, 5, 6, 7, 8, 9, 10 } )
        {
            auto play = [&]( std::vector<float> &f3, std::vector<float> &f6 )
            {
                Engine e;
                quietEffects( e );
                e.setParam( ParamIdE::ATTACK, 0.5f );
                e.setParam( ParamIdE::LIFE_MODE, static_cast<float>( mode ) );
                e.setParam( ParamIdE::LIFE_DEPTH, 0.7f );
                e.setParam( ParamIdE::LIFE_RATE, 0.5f );
                e.setParam( ParamIdE::LIFE_TURN, 1.0f );
                if ( !openOffline( e, false, 163 ) ) return std::vector<float>();
                for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
                std::vector<float> lr = render( e, 3.0 );
                e.lifeSnapshot( f3 );
                const std::vector<float> rest = render( e, 3.0 );
                e.lifeSnapshot( f6 );
                lr.insert( lr.end(), rest.begin(), rest.end() );
                return lr;
            };
            std::vector<float>       a3, a6, b3, b6;
            const std::vector<float> a = play( a3, a6 ), b = play( b3, b6 );
            if ( a.empty() ) return 1;
            const double level = rmsDb( a, 3 * 48000, 6 * 48000 );
            if ( mode == 0 )
            {
                level0 = level;
                continue;
            }
            double moved = 0.0;
            bool   finite = true;
            for ( size_t i = 0; i < a3.size() && i < a6.size(); ++i )
            {
                moved += std::fabs( a6[i] - a3[i] );
                finite = finite && std::isfinite( a6[i] );
            }
            moved /= std::max<size_t>( 1, a3.size() );
            const bool ok = a3.size() == static_cast<size_t>( LifeW * LifeH ) && finite && moved > 0.005 &&
                            std::fabs( level - level0 ) < 2.0 && a == b;
            good = good && ok;
            report += fmt( "%s %.3f %.1f%s%s; ", LifeModeNames[mode], moved, level, a == b ? "" : " NOT identical",
                           ok ? "" : " (FAILED)" );
        }
        check( good, "life: the animated pictures move, keep the level, repeat",
               fmt( "free %.1f dBFS; moved a cell, dBFS: ", level0 ) + report );
    }

    // 51. The preset map: on a preset's own point, at the default radius, the blend is that preset;
    //     half way to a neighbour a number lies between the two; the engine follows the map's target
    //     over Map Glide; the routes that come with it read, write back and walk to their points.
    if ( from <= 51 )
    {
        MapBlender blender;
        float      x = 0.0f, y = 0.0f;
        presetMapPosition( 100, x, y );
        MorphState    at;
        MapNeighbours nb;
        blender.blend( x, y, paramDesc( ParamIdE::MAP_RADIUS ).m_def, at, &nb );
        float values[NumParams];
        presetValues( 100, values, nullptr );
        double worst = 0.0;   // what carries the sound (an amount whose choice is off may differ: it is not heard)
        for ( ParamIdE id :
              { ParamIdE::TILT, ParamIdE::BRIGHTNESS, ParamIdE::CLOUD_WIDTH, ParamIdE::SHIMMER, ParamIdE::ATTACK,
                ParamIdE::RELEASE, ParamIdE::MASTER_GAIN, ParamIdE::DEPTH, ParamIdE::FAR_LEVEL, ParamIdE::DRIFT } )
        {
            const ParamDesc &d = paramDesc( id );
            worst =
                std::max( worst, static_cast<double>( std::fabs( toNorm( d, at.m_values[static_cast<size_t>( id )] ) -
                                                                 toNorm( d, values[static_cast<size_t>( id )] ) ) ) );
        }
        const bool onPoint = nb.m_index[0] == 100 && nb.m_weight[0] > 0.95f && worst < 0.03;
        // Half way to the nearest other preset, a wide blend: the tilt between the two (or equal).
        const int other = nb.m_index[1];
        float     ox = 0.0f, oy = 0.0f;
        presetMapPosition( other, ox, oy );
        MorphState mid;
        blender.blend( 0.5f * ( x + ox ), 0.5f * ( y + oy ), 1.0f, mid );
        float vo[NumParams];
        presetValues( other, vo, nullptr );
        const float ta = values[static_cast<size_t>( ParamIdE::TILT )], tb = vo[static_cast<size_t>( ParamIdE::TILT )],
                    tm = mid.m_values[static_cast<size_t>( ParamIdE::TILT )];
        const bool  between = tm >= std::min( ta, tb ) - 1.0e-3f && tm <= std::max( ta, tb ) + 1.0e-3f;
        // The engine follows the target.
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::MAP_GLIDE, 0.3f );
        if ( !openOffline( e, false, 165 ) ) return 1;
        e.noteOn( 55, 0.8f, 0 );
        render( e, 0.5 );
        e.setMapTarget( mid );
        e.setParam( ParamIdE::MAP_ACTIVE, 1.0f );
        render( e, 3.0 );
        const ParamDesc &tilt = paramDesc( ParamIdE::TILT );
        const bool follows = std::fabs( toNorm( tilt, e.soundValue( ParamIdE::TILT ) ) - toNorm( tilt, tm ) ) < 1.0e-3f;
        // The routes.
        int bad = 0;
        for ( int i = 0; i < numRoutePresets(); ++i )
        {
            Route r, back;
            char  buf[4096];
            if ( !r.parse( routePreset( i ).m_points ) || r.count() < 3 || r.write( buf, sizeof( buf ) ) <= 0 ||
                 !back.parse( buf ) || back.count() != r.count() )
                ++bad;
        }
        Route r;
        r.parse( "0.2,0.2|1|1|0.5;0.8,0.6|2|1|1.5" );
        r.start( 0.5f, 0.5f, 0.4f, true );
        float cx = 0.0f, cy = 0.0f, cr = 0.0f;
        bool  reachedA = false, reachedB = false, looped = false;
        for ( int k = 0; k < 160; ++k )
        {   // 8 s at 0.05 s: A after 1 s, B after 4 s, A again after 6 s
            r.update( 0.05f, cx, cy, cr );
            if ( std::fabs( cx - 0.2f ) < 1.0e-4f && std::fabs( cy - 0.2f ) < 1.0e-4f )
            {
                reachedA = true;
                if ( reachedB ) looped = true;
            }
            if ( std::fabs( cx - 0.8f ) < 1.0e-4f && std::fabs( cy - 0.6f ) < 1.0e-4f &&
                 std::fabs( cr - 1.5f ) < 1.0e-5f )
                reachedB = true;
        }
        check(
            onPoint && between && follows && bad == 0 && reachedA && reachedB && looped && r.running(),
            "map: blends, follows, and routes walk",
            fmt(
                "on its point: preset %d weight %.3f, off by %.1e; half way the tilt %.2f between %.2f and %.2f; followed: %s; "
                "%d routes, %d unreadable; walked A %s, B %s, round again %s",
                nb.m_index[0], nb.m_weight[0], worst, tm, ta, tb, follows ? "yes" : "NO", numRoutePresets(), bad,
                reachedA ? "yes" : "NO", reachedB ? "yes" : "NO", looped ? "yes" : "NO" ) );
    }

    // 52. A glide with modulation has no steps. Two sounds use the same LFO differently (a slow sine
    //     on the brightness, quick steps on the width) and modulate the motion's depth while the
    //     motion changes: every modulated number moves per block no more than the modulation
    //     itself does, the motion changes where its depth -- modulation included -- is nothing,
    //     the map's cursor jumping far does not make the partials or the motion jump, and a
    //     picture's landscape changes where the picture is at nothing.
    if ( from <= 52 )
    {
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::ATTACK, 0.2f );
        e.setParam( ParamIdE::MOTION, 4.0f );   // Breathe
        e.setParam( ParamIdE::MOTION_DEPTH, 0.5f );
        e.setParam( ParamIdE::LFO1_SHAPE, 0.0f );   // sine
        e.setParam( ParamIdE::LFO1_RATE, 0.3f );
        e.setParam( ParamIdE::LFO2_SHAPE, 1.0f );   // triangle
        e.setParam( ParamIdE::LFO2_RATE, 0.2f );
        e.setParam( ParamIdE::MACRO1, 1.0f );   // a constant push on the depth: at the change it must be faded too
        e.setModText( "lfo1>brightness:0.4;macro1>motion_depth:0.3" );
        if ( !openOffline( e, false, 167 ) ) return 1;
        e.noteOn( 50, 0.8f, 0 );
        e.noteOn( 57, 0.8f, 0 );
        std::vector<float> L( 128 ), R( 128 );
        // The played value of a number: modulated, or as it stands.
        auto played = [&]( ParamIdE id )
        {
            const float n = e.playedNorm( id );
            return std::isnan( n ) ? toNorm( paramDesc( id ), e.soundValue( id ) ) : n;
        };
        // Each number's largest move per block: before (A's slow sine on the brightness), gliding,
        // after (B's quick steps on the width).
        const ParamIdE watched[] = { ParamIdE::BRIGHTNESS, ParamIdE::CLOUD_WIDTH };
        float          last[2] = {};
        double         before[2] = {}, gliding[2] = {}, after[2] = {}, ignore[2] = {};
        auto           step = [&]( double *worst )
        {
            e.process( L.data(), R.data(), 128 );
            for ( int k = 0; k < 2; ++k )
            {
                const float v = played( watched[k] );
                worst[k] = std::max( worst[k], static_cast<double>( std::fabs( v - last[k] ) ) );
                last[k] = v;
            }
        };
        for ( int k = 0; k < 1500; ++k ) step( k < 10 ? ignore : before );   // 4 s
        e.beginMorph( 4.0 );
        e.setParam( ParamIdE::MOTION, 6.0f );   // Aurora
        e.setParam( ParamIdE::MOTION_DEPTH, 0.6f );
        e.setParam( ParamIdE::LFO1_SHAPE, 6.0f );   // steps
        e.setParam( ParamIdE::LFO1_RATE, 2.0f );
        e.setModText( "lfo1>cloud_width:0.4;macro1>motion_depth:0.3" );
        double depthAtSwitch = 0.0;
        int    prevMotion = 4;
        for ( int k = 0; k < 1700; ++k )
        {   // the glide and a little
            step( gliding );
            const int motion = static_cast<int>( std::lround( e.soundValue( ParamIdE::MOTION ) ) );
            if ( motion != prevMotion )
                depthAtSwitch = std::max( depthAtSwitch, static_cast<double>( played( ParamIdE::MOTION_DEPTH ) ) );
            prevMotion = motion;
        }
        for ( int k = 0; k < 1500; ++k ) step( after );
        // The map: on, then its target far away -- the other end of the bank.
        float       va[NumParams], vb[NumParams];
        std::string ma, mb;
        presetValues( 0, va, &ma );
        presetValues( numPresets() - 1, vb, &mb );
        ModRow     ra[ModRows], rb[ModRows];
        MorphState here, yonder;
        here.set( va, ra, modRowsFromText( ma, ra, ModRows ) );
        yonder.set( vb, rb, modRowsFromText( mb, rb, ModRows ) );
        here.m_values[static_cast<size_t>( ParamIdE::PARTIALS )] = 2000.0f;   // far apart in what can only step
        yonder.m_values[static_cast<size_t>( ParamIdE::PARTIALS )] = 20000.0f;
        e.setParam( ParamIdE::MAP_GLIDE, 2.0f );
        e.setMapTarget( here );
        e.setParam( ParamIdE::MAP_ACTIVE, 1.0f );
        for ( int k = 0; k < 2000; ++k ) e.process( L.data(), R.data(), 128 );   // the map takes over, gliding
        e.setMapTarget( yonder );
        double         partialsJump = 0.0, amountAtSwitch = 0.0;
        float          prevPartials = e.soundValue( ParamIdE::PARTIALS );
        int            prevKind[3] = { static_cast<int>( std::lround( e.soundValue( ParamIdE::MOTION ) ) ),
                                       static_cast<int>( std::lround( e.soundValue( ParamIdE::LIFE_MODE ) ) ),
                                       static_cast<int>( std::lround( e.soundValue( ParamIdE::FIELD_MODE ) ) ) };
        const ParamIdE amounts[3] = { ParamIdE::MOTION_DEPTH, ParamIdE::LIFE_DEPTH, ParamIdE::FIELD_STRENGTH };
        const ParamIdE kinds[3] = { ParamIdE::MOTION, ParamIdE::LIFE_MODE, ParamIdE::FIELD_MODE };
        const float    partialsSpan = paramDesc( ParamIdE::PARTIALS ).m_max - paramDesc( ParamIdE::PARTIALS ).m_min;
        for ( int k = 0; k < 4000; ++k )
        {
            const float amountBefore[3] = { played( amounts[0] ), played( amounts[1] ), played( amounts[2] ) };
            e.process( L.data(), R.data(), 128 );
            const float p = e.soundValue( ParamIdE::PARTIALS );
            partialsJump =
                std::max( partialsJump, static_cast<double>( std::fabs( p - prevPartials ) / partialsSpan ) );
            prevPartials = p;
            for ( int j = 0; j < 3; ++j )
            {
                const int kind = static_cast<int>( std::lround( e.soundValue( kinds[j] ) ) );
                if ( kind != prevKind[j] && prevKind[j] != 0 )
                    amountAtSwitch = std::max( amountAtSwitch, static_cast<double>( amountBefore[j] ) );
                prevKind[j] = kind;
            }
        }
        // A picture over one landscape glides into the same picture over another: where the
        // landscape changes, the picture -- pushed by the matrix too -- is at nothing.
        double lifeAtSwitch = 1.0;
        {
            Engine f;
            quietEffects( f );
            f.setParam( ParamIdE::LIFE_MODE, 3.0f );   // Kaleidoscope
            f.setParam( ParamIdE::LIFE_DEPTH, 0.8f );
            f.setParam( ParamIdE::LANDSCAPE, 0.0f );
            f.setParam( ParamIdE::MACRO1, 1.0f );
            f.setModText( "macro1>life_depth:0.2" );
            if ( !openOffline( f, false, 169 ) ) return 1;
            f.noteOn( 55, 0.8f, 0 );
            for ( int k = 0; k < 400; ++k ) f.process( L.data(), R.data(), 128 );
            f.beginMorph( 2.0 );
            f.setParam( ParamIdE::LANDSCAPE, 2.0f );
            int land = 0;
            for ( int k = 0; k < 1000; ++k )
            {
                f.process( L.data(), R.data(), 128 );
                const int now = static_cast<int>( std::lround( f.soundValue( ParamIdE::LANDSCAPE ) ) );
                if ( now != land )
                {
                    const float n = f.playedNorm( ParamIdE::LIFE_DEPTH );
                    lifeAtSwitch = std::isnan( n ) ? f.soundValue( ParamIdE::LIFE_DEPTH ) : n;
                }
                land = now;
            }
        }
        // The brightness is moved only by A's sine, the width only by B's steps: gliding, neither may
        // move faster than its own modulator does.
        const bool smooth = gliding[0] < 1.5 * before[0] + 1.0e-3 && gliding[1] < 1.5 * after[1] + 1.0e-3;
        check(
            smooth && depthAtSwitch < 0.02 && partialsJump < 0.02 && amountAtSwitch < 0.02 && lifeAtSwitch < 0.02,
            "morph: modulation glides without steps",
            fmt(
                "largest move per block: brightness %.4f gliding (%.4f by A's LFO), width %.4f gliding (%.4f by B's); motion depth %.3f where "
                "the motion changed; the map's far jump: partials %.3f of their range in a block, an amount %.3f where its kind changed; "
                "a picture's depth %.3f where its landscape changed",
                gliding[0], before[0], gliding[1], after[1], depthAtSwitch, partialsJump, amountAtSwitch,
                lifeAtSwitch ) );
    }

    // 53. Chimeras and swarms (Phase 16a). One high note, so that its first harmonic's cloud has a
    //     thousand members, every eighth of them watched through the picture's snapshot. Uncoupled
    //     each pulses on its own; as a chimera an ordered part lies beside a disordered one: some
    //     neighbours' loudness moves alike, others' does not. As a swarm their pitch moves on their
    //     circles by no more than the reach: when like phases gather they all stand as far from the
    //     middle (one place, or two opposite -- the ring model counts them alike, and they pulse
    //     against each other), when they part they scatter round the circle.
    if ( from <= 53 )
    {
        struct Watch
        {
            std::vector<std::vector<float>> m_amp;   // per watched member its loudness over time
            std::vector<float>              m_hz;    // its last frequency
        };
        auto watch = [&]( int mode, float coupling, float spread, float swarm, float reach, Watch &w )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.05f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::SYNC_DEPTH, 1.0f );
            e.setParam( ParamIdE::SYNC_RATE, 0.5f );
            e.setParam( ParamIdE::SYNC_SPREAD, spread );
            e.setParam( ParamIdE::SYNC_COUPLING, coupling );
            e.setParam( ParamIdE::SYNC_MODE, static_cast<float>( mode ) );
            e.setParam( ParamIdE::SYNC_SWARM, swarm );
            e.setParam( ParamIdE::SYNC_REACH, reach );
            e.setParam( ParamIdE::SYNC_LAG, 0.47f );
            if ( !openOffline( e, false, 171 ) ) return false;
            e.noteOn( 84, 0.8f, 0 );
            std::vector<float> L( 128 ), R( 128 ), dots;
            int                per = 0, members = 0;
            for ( int k = 0; k < 375 * 24; ++k )
            {   // 24 s
                e.process( L.data(), R.data(), 128 );
                if ( k < 375 * 4 || !e.partialSnapshot( dots, per ) )
                    continue;   // after four seconds, as the snapshots come
                // The voice that sounds; its first cloud: the watched members up to where the frequency jumps an octave.
                for ( int v = 0; per > 0 && v * per * 2 < static_cast<int>( dots.size() ); ++v )
                {
                    const float *d = dots.data() + static_cast<size_t>( v ) * per * 2;
                    if ( d[0] <= 0.0f ) continue;
                    if ( members == 0 )
                    {
                        while ( members + 1 < per && d[2 * ( members + 1 )] < 1.5f * d[0] ) ++members;
                        ++members;
                        w.m_amp.assign( static_cast<size_t>( members ), {} );
                    }
                    w.m_hz.assign( static_cast<size_t>( members ), 0.0f );
                    for ( int j = 0; j < members; ++j )
                    {
                        w.m_amp[static_cast<size_t>( j )].push_back( d[2 * j + 1] );
                        w.m_hz[static_cast<size_t>( j )] = d[2 * j];
                    }
                    break;
                }
            }
            return members > 8;
        };
        // The share of neighbouring watched members whose loudness correlates over 0.8 (and under 0.5).
        auto together = []( const Watch &w, double *apart = nullptr )
        {
            int pairs = 0, alike = 0, unlike = 0;
            for ( size_t j = 0; j + 1 < w.m_amp.size(); ++j )
            {
                const std::vector<float> &a = w.m_amp[j];
                const std::vector<float> &b = w.m_amp[j + 1];
                const size_t              n = std::min( a.size(), b.size() );
                double                    ma = 0.0, mb = 0.0;
                for ( size_t i = 0; i < n; ++i )
                {
                    ma += a[i];
                    mb += b[i];
                }
                ma /= std::max<double>( 1.0, static_cast<double>( n ) );
                mb /= std::max<double>( 1.0, static_cast<double>( n ) );
                double ab = 0.0, aa = 0.0, bb = 0.0;
                for ( size_t i = 0; i < n; ++i )
                {
                    ab += ( a[i] - ma ) * ( b[i] - mb );
                    aa += ( a[i] - ma ) * ( a[i] - ma );
                    bb += ( b[i] - mb ) * ( b[i] - mb );
                }
                ++pairs;
                const double r = aa > 0.0 && bb > 0.0 ? ab / std::sqrt( aa * bb ) : 0.0;
                if ( r > 0.8 ) ++alike;
                if ( r < 0.5 ) ++unlike;
            }
            if ( apart != nullptr ) *apart = pairs > 0 ? static_cast<double>( unlike ) / pairs : -1.0;
            return pairs > 0 ? static_cast<double>( alike ) / pairs : -1.0;
        };
        Watch        chimera, free, base, gather, part;
        const bool   ran = watch( 1, 1.0f, 0.0f, 0.0f, 0.0f, chimera ) && watch( 0, 0.0f, 0.3f, 0.0f, 0.0f, free ) &&
                           watch( 2, 1.0f, 0.1f, 1.0f, 0.0f, base ) && watch( 2, 1.0f, 0.1f, 1.0f, 40.0f, gather ) &&
                           watch( 2, 1.0f, 0.1f, -1.0f, 40.0f, part );
        double       cApart = 0.0;
        const double c = together( chimera, &cApart ), f = together( free );
        // The swarm's pitch against the same cloud without it: cents per watched member.
        auto offsets = [&]( const Watch &w, double &spread, double &widest )
        {
            std::vector<double> d;
            for ( size_t j = 0; j < w.m_hz.size() && j < base.m_hz.size(); ++j )
                if ( w.m_hz[j] > 0.0f && base.m_hz[j] > 0.0f )
                    d.push_back( 1200.0 * std::log2( static_cast<double>( w.m_hz[j] ) / base.m_hz[j] ) );
            double m = 0.0;
            for ( double x : d ) m += x;
            m /= std::max<double>( 1.0, static_cast<double>( d.size() ) );
            // How far from the middle each stands: the spread of |offset| (the ring counts opposite places alike).
            double ma = 0.0;
            for ( double x : d ) ma += std::fabs( x );
            ma /= std::max<double>( 1.0, static_cast<double>( d.size() ) );
            spread = 0.0;
            widest = 0.0;
            for ( double x : d )
            {
                spread += ( std::fabs( x ) - ma ) * ( std::fabs( x ) - ma );
                widest = std::max( widest, std::fabs( x ) );
            }
            spread = std::sqrt( spread / std::max<double>( 1.0, static_cast<double>( d.size() ) ) );
            (void)m;
        };
        double gSpread = 0.0, gWide = 0.0, pSpread = 0.0, pWide = 0.0;
        offsets( gather, gSpread, gWide );
        offsets( part, pSpread, pWide );
        check(
            ran && f < 0.3 && c > 0.15 && c < 0.9 && cApart > 0.05 && gSpread < 0.15 * 40.0 && pSpread > 0.2 * 40.0 &&
                gWide < 41.0 && pWide < 41.0,
            "sync: chimeras and swarms",
            fmt(
                "neighbours pulsing alike: %.2f as a chimera (%.2f not at all), %.2f free; a swarm of reach 40 ct: distance from the middle "
                "scattered by %.1f ct (widest %.1f) when like phases gather, %.1f ct (widest %.1f) when they part",
                c, cApart, f, gSpread, gWide, pSpread, pWide ) );
    }

    // 54. The Fluid and the Islands (Phase 16b). The Fluid: the ink comes where the notes blow it
    //     (their columns hold more of it than the rest after a few seconds), the flow carries it on,
    //     the level holds, and two runs are the same bit for bit. The Islands: the field gathers --
    //     fewer borders along its rows after forty seconds than after five -- and the level holds.
    if ( from <= 54 )
    {
        auto run = [&]( int mode, float character, double seconds, std::vector<float> &early, std::vector<float> &late,
                        std::vector<float> &out )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::LIFE_MODE, static_cast<float>( mode ) );
            e.setParam( ParamIdE::LIFE_DEPTH, 0.8f );
            e.setParam( ParamIdE::LIFE_RATE, 1.0f );
            e.setParam( ParamIdE::LIFE_CHARACTER, character );
            if ( !openOffline( e, false, 173 ) ) return false;
            for ( int note : { 45, 57, 64 } ) e.noteOn( note, 0.8f, 0 );
            std::vector<float> L( 128 ), R( 128 ), dots;
            int                per = 0;
            out.clear();
            const int blocks = static_cast<int>( seconds * 375.0 );
            for ( int k = 0; k < blocks; ++k )
            {
                e.process( L.data(), R.data(), 128 );
                for ( int i = 0; i < 128; ++i )
                {
                    out.push_back( L[static_cast<size_t>( i )] );
                    out.push_back( R[static_cast<size_t>( i )] );
                }
                if ( e.partialSnapshot( dots, per ) )
                {   // the field comes with the snapshot
                    if ( k < static_cast<int>( 5.0 * 375.0 ) ) e.lifeSnapshot( early );
                    e.lifeSnapshot( late );
                }
            }
            return early.size() == static_cast<size_t>( LifeW * LifeH ) && late.size() == early.size();
        };
        // The Fluid.
        std::vector<float> early, late, out1, out2, e2, l2;
        const bool         fluidRan = run( 11, 0.6f, 20.0, early, late, out1 ) && run( 11, 0.6f, 20.0, e2, l2, out2 );
        const bool         same = out1 == out2;
        // The ink atNotes the notes' columns (a fundamental and its octave each, +-3 cells) against the rest, at five seconds.
        double            atNotes = 0.0, rest = 0.0;
        int               atNotesCells = 0, restCells = 0;
        std::vector<bool> column( static_cast<size_t>( LifeW ), false );
        for ( int note : { 45, 57, 64 } )
        {
            const double hz = 440.0 * std::pow( 2.0, ( note - 69 ) / 12.0 );
            for ( double f : { hz, 2.0 * hz } )
            {
                const int x = static_cast<int>( std::log2( f / 20.0 ) / std::log2( 1000.0 ) * LifeW );
                for ( int d = -3; d <= 3; ++d ) column[static_cast<size_t>( ( x + d ) & ( LifeW - 1 ) )] = true;
            }
        }
        for ( int i = 0; fluidRan && i < LifeW * LifeH; ++i )
        {
            if ( column[static_cast<size_t>( i & ( LifeW - 1 ) )] )
            {
                atNotes += early[static_cast<size_t>( i )];
                ++atNotesCells;
            }
            else
            {
                rest += early[static_cast<size_t>( i )];
                ++restCells;
            }
        }
        atNotes /= std::max( 1, atNotesCells );
        rest /= std::max( 1, restCells );
        double moved = 0.0;
        for ( size_t i = 0; fluidRan && i < early.size(); ++i ) moved += std::fabs( late[i] - early[i] );
        moved /= std::max<size_t>( 1, early.size() );
        const size_t frames = out1.size() / 2;
        const double fluidDb = rmsDb( out1, frames / 2, frames );
        // The Islands: borders along the rows (where the field crosses its middle).
        auto borders = []( const std::vector<float> &f )
        {
            int n = 0;
            for ( int y = 0; y < LifeH; ++y )
                for ( int x = 0; x < LifeW; ++x )
                    n += ( f[static_cast<size_t>( y * LifeW + x )] > 0.5f ) !=
                         ( f[static_cast<size_t>( y * LifeW + ( ( x + 1 ) & ( LifeW - 1 ) ) )] > 0.5f );
            return n;
        };
        std::vector<float> ie, il, iout;
        const bool         islesRan = run( 12, 0.8f, 40.0, ie, il, iout );
        const int          bEarly = islesRan ? borders( ie ) : 0, bLate = islesRan ? borders( il ) : 0;
        const size_t       iframes = iout.size() / 2;
        const double       islesDb = rmsDb( iout, iframes / 2, iframes );
        // Without a living field, for the level.
        std::vector<float> oe, ol, plain;
        run( 0, 0.5f, 10.0, oe, ol, plain );
        const double plainDb = rmsDb( plain, plain.size() / 4, plain.size() / 2 );
        // The level within 4 dB: the field's make-up is measured over the whole field, and the Fluid's ink lies
        // where the loud low harmonics read it (about 3 dB over); a preset is levelled by measurement anyway.
        check(
            fluidRan && same && atNotes > 2.0 * rest && moved > 0.01 && islesRan && bLate < bEarly && bLate > 0 &&
                std::fabs( fluidDb - plainDb ) < 4.0 && std::fabs( islesDb - plainDb ) < 4.0,
            "life: the Fluid and the Islands",
            fmt(
                "Fluid: ink %.3f at the notes, %.3f elsewhere after 5 s, %.3f moved by 20 s, twice %s, %.1f dBFS; Islands: %d borders "
                "after 5 s, %d after 40 s, %.1f dBFS; without a field %.1f dBFS",
                atNotes, rest, moved, same ? "the same" : "DIFFERENT", fluidDb, bEarly, bLate, islesDb, plainDb ) );
    }

    // 55. Turbulence (Phase 16c): a loud chord's energy climbs its harmonics -- its partials' centroid
    //     (power-weighted, from the picture's snapshot) rises -- without the chord growing louder; a soft
    //     chord's climbs far less; turned off, the spectrum settles back.
    if ( from <= 55 )
    {
        // The centroid in octaves above 20 Hz, averaged over the snapshots of a stretch, and the level.
        struct Heard
        {
            double m_centroid = 0.0, m_db = 0.0, m_back = 0.0;
        };
        auto listen = [&]( float turbulence, float velocity, bool thenOff, Heard &h )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.3f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::TILT, 1.2f );
            e.setParam( ParamIdE::TURBULENCE, turbulence );
            e.setParam( ParamIdE::TURBULENCE_TIME, 4.0f );
            if ( !openOffline( e, false, 175 ) ) return false;
            for ( int note : { 48, 55, 60, 64 } ) e.noteOn( note, velocity, 0 );
            std::vector<float> L( 128 ), R( 128 ), dots, lr;
            int                per = 0;
            double             sum = 0.0, back = 0.0;
            int                count = 0, backCount = 0;
            for ( int k = 0; k < 375 * 30; ++k )
            {   // 30 s; off at 10 s when asked
                if ( thenOff && k == 375 * 10 ) e.setParam( ParamIdE::TURBULENCE, 0.0f );
                e.process( L.data(), R.data(), 128 );
                if ( k >= 375 * 8 && k < 375 * 10 )
                    for ( int i = 0; i < 128; ++i )
                    {
                        lr.push_back( L[static_cast<size_t>( i )] );
                        lr.push_back( R[static_cast<size_t>( i )] );
                    }
                if ( !e.partialSnapshot( dots, per ) ) continue;
                double w = 0.0, m = 0.0;
                for ( size_t i = 0; i + 1 < dots.size(); i += 2 )
                    if ( dots[i] > 20.0f && dots[i + 1] > 0.0f )
                    {
                        const double p = static_cast<double>( dots[i + 1] ) * dots[i + 1];
                        w += p;
                        m += p * std::log2( dots[i] / 20.0 );
                    }
                if ( w <= 0.0 ) continue;
                if ( k >= 375 * 8 && k < 375 * 10 )
                {
                    sum += m / w;
                    ++count;
                }
                if ( k >= 375 * 28 )
                {
                    back += m / w;
                    ++backCount;
                }
            }
            h.m_centroid = count > 0 ? sum / count : 0.0;
            h.m_back = backCount > 0 ? back / backCount : 0.0;
            h.m_db = rmsDb( lr, 0, lr.size() / 2 );
            return count > 0;
        };
        Heard        plain, loud, soft, softPlain, settled;
        const bool   ran = listen( 0.0f, 0.9f, false, plain ) && listen( 1.0f, 0.9f, false, loud ) &&
                           listen( 1.0f, 0.15f, false, soft ) && listen( 0.0f, 0.15f, false, softPlain ) &&
                           listen( 1.0f, 0.9f, true, settled );
        const double rise = loud.m_centroid - plain.m_centroid, softRise = soft.m_centroid - softPlain.m_centroid;
        check(
            ran && rise > 0.3 && std::fabs( loud.m_db - plain.m_db ) < 1.5 && softRise < 0.6 * rise &&
                std::fabs( settled.m_back - plain.m_centroid ) < 0.1,
            "turbulence: a loud chord's energy climbs, a soft one's hardly",
            fmt(
                "centroid %.2f octaves higher loud (%.1f against %.1f dBFS), %.2f soft; turned off, %.2f octaves from the plain chord after 20 s",
                rise, loud.m_db, plain.m_db, softRise, settled.m_back - plain.m_centroid ) );
    }

    // 56. The filter morphs by transport (Phase 16d): a glide from a low pass at 300 Hz into a high
    //     pass at 3 kHz. Half way the filter is neither faded out nor both at half: its mass has
    //     travelled into the middle -- the partials between 600 Hz and 1.5 kHz pass more than four
    //     times as strongly as those under 100 Hz and over 8 kHz (a fade would leave them alike).
    //     The filter's gain at a partial: its amplitude against the same voice without a filter.
    if ( from <= 56 )
    {
        // Every watched partial's (Hz, amplitude) at the start, half way and at the end of the glide.
        auto take = [&]( bool filtered, std::vector<float> &start, std::vector<float> &middle, std::vector<float> &end )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.05f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::KEY_TRACK, 0.0f );
            e.setParam( ParamIdE::FILTER_ENV, 0.0f );
            e.setParam( ParamIdE::FILTER_DRIFT, 0.0f );
            e.setParam( ParamIdE::FILTER_MODEL, filtered ? 3.0f : 0.0f );   // LP 24
            e.setParam( ParamIdE::CUTOFF, 300.0f );
            e.setParam( ParamIdE::RESONANCE, 0.2f );
            if ( !openOffline( e, false, 177 ) ) return false;
            e.noteOn( 31, 0.8f, 0 );
            std::vector<float> L( 128 ), R( 128 ), dots;
            int                per = 0;
            for ( int k = 0; k < 375 * 2; ++k )
            {
                e.process( L.data(), R.data(), 128 );
                if ( e.partialSnapshot( dots, per ) ) start = dots;
            }
            e.beginMorph( 6.0 );
            e.setParam( ParamIdE::FILTER_MODEL, filtered ? 4.0f : 0.0f );   // HP 12
            e.setParam( ParamIdE::CUTOFF, 3000.0f );
            bool half = false;
            for ( int k = 0; k < 375 * 8; ++k )
            {
                e.process( L.data(), R.data(), 128 );
                if ( !e.partialSnapshot( dots, per ) ) continue;
                if ( !half && e.morphProgress() >= 0.5f )
                {
                    middle = dots;
                    half = true;
                }
                end = dots;
            }
            return half && !start.empty();
        };
        std::vector<float> s0, m0, e0, s1, m1, e1;
        const bool         ran = take( false, s0, m0, e0 ) && take( true, s1, m1, e1 );
        // The filter's gain in a band, the power mean of the partials in it.
        auto band = []( const std::vector<float> &plain, const std::vector<float> &filtered, double lo, double hi )
        {
            double sum = 0.0;
            int    n = 0;
            for ( size_t i = 0; i + 1 < plain.size() && i + 1 < filtered.size(); i += 2 )
                if ( plain[i] >= lo && plain[i] < hi && plain[i + 1] > 1.0e-9f )
                {
                    const double g = filtered[i + 1] / plain[i + 1];
                    sum += g * g;
                    ++n;
                }
            return n > 0 ? std::sqrt( sum / n ) : 0.0;
        };
        double                    lo[3] = {}, mid[3] = {}, hi[3] = {};
        const std::vector<float> *plain[3] = { &s0, &m0, &e0 };
        const std::vector<float> *filt[3] = { &s1, &m1, &e1 };
        for ( int k = 0; ran && k < 3; ++k )
        {
            lo[k] = band( *plain[k], *filt[k], 40.0, 100.0 );
            mid[k] = band( *plain[k], *filt[k], 600.0, 1500.0 );
            hi[k] = band( *plain[k], *filt[k], 8000.0, 16000.0 );
        }
        auto db = []( double g ) { return 20.0 * std::log10( std::max( g, 1.0e-9 ) ); };
        check(
            ran && lo[0] > 4.0 * hi[0] && mid[1] > 4.0 * lo[1] && mid[1] > 4.0 * hi[1] && hi[2] > 4.0 * lo[2],
            "filter: a morph moves the pass band through the middle",
            fmt(
                "low / middle / high, dB: start %.1f / %.1f / %.1f, half way %.1f / %.1f / %.1f, end %.1f / %.1f / %.1f",
                db( lo[0] ), db( mid[0] ), db( hi[0] ), db( lo[1] ), db( mid[1] ), db( hi[1] ), db( lo[2] ),
                db( mid[2] ), db( hi[2] ) ) );
    }

    // 57. MPE (Phase 16e): two notes on two member channels. A bend on one channel moves its note
    //     alone; a note let go keeps its bend when its channel is bent again for the next; the
    //     master's bend moves every played note. Pressure on Level lifts a note by up to 2^depth
    //     (3 dB at 0.5), slide on Brightness makes it brighter.
    if ( from <= 57 )
    {
        auto pitches = []( const Engine &e )
        {
            std::vector<double> hz;
            for ( int i = 0; i < 16; ++i )
                if ( e.voiceView( i ).m_hz > 0.0f && e.voiceView( i ).m_source == 0 )
                    hz.push_back( e.voiceView( i ).m_hz );
            std::sort( hz.begin(), hz.end() );
            return hz;
        };
        Engine e;
        quietEffects( e );
        e.setParam( ParamIdE::MOTION, 0.0f );
        e.setParam( ParamIdE::RELEASE, 2.0f );
        bool good = openOffline( e, false, 179 );
        e.noteOn( 60, 0.8f, 0, 1 );
        e.noteOn( 67, 0.8f, 0, 2 );
        render( e, 0.5 );
        const std::vector<double> before = pitches( e );
        e.setChannelBend( 1, 2.0f );
        render( e, 0.2 );
        const std::vector<double> bent = pitches( e );
        e.noteOff( 67, 0, 2 );
        render( e, 0.05 );
        e.setChannelBend( 2, 5.0f );   // the channel's next note would be bent; the released one stays
        render( e, 0.2 );
        const std::vector<double> released = pitches( e );
        e.setChannelBend( -1, -1.0f );   // the master
        render( e, 0.2 );
        const std::vector<double> master = pitches( e );
        auto                      cents = []( double a, double b ) { return 1200.0 * std::log2( a / b ); };
        const bool sizes = before.size() == 2 && bent.size() == 2 && released.size() == 2 && master.size() == 2;
        const bool pitch = sizes && std::fabs( cents( bent[0], before[0] ) - 200.0 ) < 2.0 &&
                           std::fabs( cents( bent[1], before[1] ) ) < 2.0 &&
                           std::fabs( cents( released[1], before[1] ) ) < 2.0 &&
                           std::fabs( cents( master[0], before[0] ) - 100.0 ) < 2.0 &&
                           std::fabs( cents( master[1], before[1] ) + 100.0 ) < 2.0;
        // Pressure on Level, slide on Brightness: a note on channel 3 pressed fully against not at all,
        // slid up against at rest.
        auto play = [&]( float press, float slide, double &db, double &centroid )
        {
            Engine x;
            quietEffects( x );
            x.setParam( ParamIdE::MOTION, 0.0f );
            x.setParam( ParamIdE::MPE_PRESS, static_cast<float>( MpeTargetE::LEVEL ) );
            x.setParam( ParamIdE::MPE_PRESS_DEPTH, 0.5f );
            x.setParam( ParamIdE::MPE_SLIDE, static_cast<float>( MpeTargetE::BRIGHTNESS ) );
            x.setParam( ParamIdE::MPE_SLIDE_DEPTH, 1.0f );
            x.setParam( ParamIdE::BRIGHTNESS, 0.4f );
            if ( !openOffline( x, false, 181 ) ) return false;
            x.setChannelPressure( 3, press );
            x.setChannelSlide( 3, slide );
            x.noteOn( 57, 0.8f, 0, 3 );
            render( x, 1.0 );
            const std::vector<float> lr = render( x, 1.0 );
            db = rmsDb( lr, 0, lr.size() / 2 );
            std::vector<float> dots;
            int                per = 0;
            centroid = 0.0;
            double w = 0.0;
            for ( int k = 0; k < 40 && !x.partialSnapshot( dots, per ); ++k ) render( x, 0.01 );
            for ( size_t i = 0; i + 1 < dots.size(); i += 2 )
                if ( dots[i] > 20.0f )
                {
                    const double pw = static_cast<double>( dots[i + 1] ) * dots[i + 1];
                    w += pw;
                    centroid += pw * std::log2( dots[i] / 20.0 );
                }
            centroid = w > 0.0 ? centroid / w : 0.0;
            return true;
        };
        double dbRest = 0.0, dbPress = 0.0, dbSlide = 0.0, cRest = 0.0, cPress = 0.0, cSlide = 0.0;
        good = good && play( 0.0f, 0.5f, dbRest, cRest ) && play( 1.0f, 0.5f, dbPress, cPress ) &&
               play( 0.0f, 1.0f, dbSlide, cSlide );
        check(
            good && pitch && std::fabs( dbPress - dbRest - 3.01 ) < 0.5 && cSlide > cRest + 0.2,
            "MPE: each note its own bend, pressure and slide",
            fmt(
                "bent %.1f ct (the other %.1f); released, its channel bent again: %.1f ct; the master -1: %.1f / %.1f ct; "
                "pressure %+.2f dB; slide: centroid %+.2f octaves",
                sizes ? cents( bent[0], before[0] ) : 0.0, sizes ? cents( bent[1], before[1] ) : 0.0,
                sizes ? cents( released[1], before[1] ) : 0.0, sizes ? cents( master[0], before[0] ) : 0.0,
                sizes ? cents( master[1], before[1] ) : 0.0, dbPress - dbRest, cSlide - cRest ) );
    }

    // 58. The growing hall (Phase 17a): a chord held for a minute with Grow on teaches the hall its
    //     pitches. Growing, the hall's tail after the chord dies more slowly than a new hall's; eroding,
    //     faster. The memory brought out of the engine and put into a new one rings as the trained hall.
    if ( from <= 58 )
    {
        struct Tail
        {
            double m_slope = 0.0, m_level = 0.0;
        };   // dB a second, dBFS
        std::vector<float> learnt;
        double             age = 0.0;
        double readBack = 1.0;   // the largest difference of the memory read back after a restore, of its value
        auto   play = [&]( float grow, double train, const std::vector<float> *memory, bool keep, Tail &t )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MODAL_LEVEL, 1.0f );
            e.setParam( ParamIdE::MODAL_DECAY, 6.0f );
            e.setParam( ParamIdE::MODAL_BREATH,
                        0.0f );   // its decays the same at every time: only the memory changes them
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::HALL_GROW, grow );
            e.setParam( ParamIdE::HALL_MEMORY, memory != nullptr ? 14400.0f : 60.0f );
            if ( !openOffline( e, false, 177 ) ) return false;
            if ( memory != nullptr ) e.setHallMemory( *memory, 60.0 );
            for ( int note : { 48, 55, 60, 64 } ) e.noteOn( note, 0.8f, 0 );
            render( e, train );
            if ( keep && !e.hallMemory( learnt, age ) ) return false;
            if ( memory != nullptr )
            {
                std::vector<float> back;
                double             backAge = 0.0;
                e.hallMemory( back, backAge );
                readBack = 0.0;
                for ( size_t b = 0; b < back.size() && b < memory->size(); ++b )
                    readBack = std::max( readBack, std::fabs( static_cast<double>( back[b] ) - ( *memory )[b] ) /
                                                       std::max( 0.1, static_cast<double>( ( *memory )[b] ) ) );
            }
            for ( int note : { 48, 55, 60, 64 } ) e.noteOff( note, 0 );
            const std::vector<float> lr = render( e, 2.5 );
            const double             a = rmsDb( lr, 12000, 36000 ),
                                     b = rmsDb( lr, 84000, 108000 );   // 0.25 .. 0.75 s, 1.75 .. 2.25 s after
            t.m_slope = ( a - b ) / 1.5;
            t.m_level = a;
            return true;
        };
        Tail       plain, grown, eroded, restored;
        const bool ran = play( 0.0f, 6.0, nullptr, false, plain ) && play( 1.0f, 60.0, nullptr, true, grown ) &&
                         play( -1.0f, 60.0, nullptr, false, eroded ) && play( 1.0f, 6.0, &learnt, false, restored );
        // The memory: the chord's root (C3, 130.8 Hz) heard far above the average, the most of the bands below it.
        const int root = static_cast<int>( std::log2( 130.81 / 20.0 ) / 9.96578428 * HallBins );
        double    below = 0.0;
        for ( int b = 0; b < root - 40; ++b )
            below = std::max( below, static_cast<double>( learnt.empty() ? 1.0f : learnt[static_cast<size_t>( b )] ) );
        const double atRoot =
            learnt.empty() ? 0.0 : *std::max_element( learnt.begin() + root - 3, learnt.begin() + root + 4 );
        check(
            ran && atRoot > 2.0 && below < 0.5 && std::fabs( age - 60.0 ) < 2.0 &&
                grown.m_slope < 0.7 * plain.m_slope && eroded.m_slope > 1.25 * plain.m_slope &&
                restored.m_slope < 0.7 * plain.m_slope && readBack < 0.02,
            "hall: it grows into what it hears, erodes, and is kept",
            fmt(
                "memory %.1f at the root, %.2f below it, %.0f s learnt; the tail falls %.1f dB/s new, %.1f grown, %.1f eroded; "
                "restored into a new engine %.1f, read back within %.2f %%",
                atRoot, below, age, plain.m_slope, grown.m_slope, eroded.m_slope, restored.m_slope,
                100.0 * readBack ) );
    }

    // 59. The partials' own time (Phase 17b): where the spectrum is dense their drift slows (Time Bend
    //     positive) or quickens (negative), at the same level. The drift as heard: each partial's move
    //     weighted by its amplitude.
    if ( from <= 59 )
    {
        double      speed[3] = { 0.0, 0.0, 0.0 }, level[3] = { 0.0, 0.0, 0.0 };
        bool        ran = true;
        const float bends[3] = { 0.0f, 1.0f, -1.0f };
        for ( int k = 0; k < 3; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::DRIFT, 30.0f );
            e.setParam( ParamIdE::DRIFT_RATE, 0.3f );
            e.setParam( ParamIdE::DOPPLER, 0.0f );
            e.setParam( ParamIdE::BREATH, 0.0f );
            e.setParam( ParamIdE::TIME_BEND, bends[k] );
            if ( !openOffline( e, false, 179 ) ) return 1;
            for ( int note : { 48, 55, 60, 64 } ) e.noteOn( note, 0.8f, 0 );
            render( e, 2.0 );
            std::vector<float> L( 256 ), R( 256 ), dots, last, lr;
            int                per = 0;
            double             moved = 0.0, weight = 0.0;
            long               count = 0;
            for ( int b = 0; b < 750; ++b )
            {   // 4 s
                e.process( L.data(), R.data(), 256 );
                for ( int i = 0; i < 256; ++i )
                {
                    lr.push_back( L[static_cast<size_t>( i )] );
                    lr.push_back( R[static_cast<size_t>( i )] );
                }
                if ( !e.partialSnapshot( dots, per ) ) continue;
                if ( last.size() == dots.size() )
                    for ( size_t i = 0; i + 1 < dots.size(); i += 2 )
                        if ( dots[i] > 20.0f && last[i] > 20.0f && dots[i + 1] > 0.0f )
                        {
                            moved += dots[i + 1] *
                                     std::fabs( 1200.0 * std::log2( static_cast<double>( dots[i] ) / last[i] ) );
                            weight += dots[i + 1];
                            ++count;
                        }
                last = dots;
            }
            ran = ran && count > 100;
            speed[k] = weight > 0.0 ? moved / weight : 0.0;
            level[k] = rmsDb( lr, 0, lr.size() / 2 );
        }
        check( ran && speed[1] < 0.75 * speed[0] && speed[2] > 1.3 * speed[0] &&
                   std::fabs( level[1] - level[0] ) < 1.0 && std::fabs( level[2] - level[0] ) < 1.0,
               "time bend: the partials' drift slower where the spectrum is dense, or faster",
               fmt( "%.2f cents a snapshot plain, %.2f bent slow, %.2f bent fast; %.1f / %.1f / %.1f dBFS", speed[0],
                    speed[1], speed[2], level[0], level[1], level[2] ) );
    }

    // 60. The Orbit field (Phase 17c): a partial circles its well -- its pitch swings about the
    //     scale's pitch, back and forth many times, within the radius -- at the level of no field.
    if ( from <= 60 )
    {
        double sd[2] = { 0.0, 0.0 }, level[2] = { 0.0, 0.0 }, mean[2] = { 0.0, 0.0 }, widest = 0.0;
        int    crossings = 0;
        bool   ran = true;
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ATTACK, 0.2f );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 0.0f );
            e.setParam( ParamIdE::DOPPLER, 0.0f );
            e.setParam( ParamIdE::BREATH, 0.0f );
            e.setParam( ParamIdE::FIELD_MODE, k == 0 ? 0.0f : static_cast<float>( FieldModeE::ORBIT ) );
            e.setParam( ParamIdE::FIELD_STRENGTH, 0.5f );
            e.setParam( ParamIdE::FIELD_RADIUS,
                        20.0f );   // orbits of about 10 cents: within the well (wider ones hop to the next pitch)
            e.setParam( ParamIdE::FIELD_SPRING, 0.2f );
            e.setParam( ParamIdE::FIELD_RATE, 0.2f );
            if ( !openOffline( e, false, 181 ) ) return 1;
            e.noteOn( 60, 0.8f, 0 );
            render( e, 2.0 );
            std::vector<float>  L( 256 ), R( 256 ), dots, lr;
            std::vector<double> cents;
            int                 per = 0;
            for ( int b = 0; b < 1875; ++b )
            {   // 10 s
                e.process( L.data(), R.data(), 256 );
                for ( int i = 0; i < 256; ++i )
                {
                    lr.push_back( L[static_cast<size_t>( i )] );
                    lr.push_back( R[static_cast<size_t>( i )] );
                }
                if ( e.partialSnapshot( dots, per ) && dots.size() >= 2 && dots[0] > 20.0f )
                    cents.push_back( 1200.0 * std::log2( dots[0] / 261.6256 ) );   // the tuning's C sets the mean
            }
            ran = ran && cents.size() > 200;
            double m = 0.0, m2 = 0.0;
            for ( double c : cents )
            {
                m += c;
                m2 += c * c;
            }
            m /= std::max<size_t>( cents.size(), 1 );
            sd[k] = std::sqrt( std::max( 0.0, m2 / std::max<size_t>( cents.size(), 1 ) - m * m ) );
            level[k] = rmsDb( lr, 0, lr.size() / 2 );
            mean[k] = m;
            if ( k == 1 )
                for ( size_t i = 1; i < cents.size(); ++i )
                {
                    if ( ( cents[i] - m ) * ( cents[i - 1] - m ) < 0.0 ) ++crossings;
                    widest = std::max( widest, std::fabs( cents[i] - mean[0] ) );
                }
        }
        check(
            ran && sd[0] < 1.0 && sd[1] > 3.0 && sd[1] < 25.0 && widest < 60.0 &&
                std::fabs( mean[1] - mean[0] ) < 10.0 && crossings >= 10 && std::fabs( level[1] - level[0] ) < 1.5,
            "orbit: the partials circle the scale's pitches",
            fmt(
                "the fundamental swings %.1f cents (%.2f without a field), at most %.0f from its pitch, mean %+.1f, %d crossings in 10 s; %.1f / %.1f dBFS",
                sd[1], sd[0], widest, mean[1] - mean[0], crossings, level[1], level[0] ) );
    }

    // 61. The ripple's bend (Phase 17b): a struck note's wave lingers where it sounds -- with Bend, 0.6 s
    //     later the string still stands far higher at the note's fundamental; without, the wave has run on.
    //     And the waves leaving the slow zone are no louder for it (the whole string holds no more).
    if ( from <= 61 )
    {
        double kept[2] = { 0.0, 0.0 }, whole[2] = { 0.0, 0.0 };
        bool   ran = true;
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MOTION, 0.0f );
            e.setParam( ParamIdE::RIPPLE_DEPTH, 1.0f );
            e.setParam( ParamIdE::RIPPLE_SPEED, 1.5f );
            e.setParam( ParamIdE::RIPPLE_DECAY, 8.0f );
            e.setParam( ParamIdE::RIPPLE_BEND, static_cast<float>( k ) );
            if ( !openOffline( e, false, 183 ) ) return 1;
            e.noteOn( 50, 0.8f, 0 );
            float           u[RippleCells];
            constexpr float perOctave = static_cast<float>( RippleCells - 1 ) / 9.96578428f;
            const float     c1 = std::log2( 146.83f / 20.0f ) * perOctave;
            auto atRoot = [&](double& energy) {   // the string's height at the fundamental, and its whole energy
                e.rippleString(u);
                double peak = 0.0;
                energy = 0.0;
                for (int i = 0; i < RippleCells; ++i) {
                    if (std::fabs(static_cast<float>(i) - c1) <= 3.0f) peak = std::max(peak, std::fabs(static_cast<double>(u[i])));
                    energy += static_cast<double>(u[i]) * u[i];
                }
                return peak;
            };
            render( e, 0.05 );   // past the latency: the strike is on the string
            double       energy0 = 0.0, energy1 = 0.0;
            const double start = atRoot( energy0 );
            render( e, 0.6 );
            const double later = atRoot( energy1 );
            ran = ran && start > 0.0 && energy0 > 0.0;
            kept[k] = start > 0.0 ? later / start : 0.0;
            whole[k] = energy0 > 0.0 ? energy1 / energy0 : 0.0;
        }
        check(
            ran && kept[1] > 2.0 * kept[0] && whole[1] < 1.0, "ripple: with Bend its waves linger at the notes",
            fmt(
                "0.6 s after the strike the string stands at %.0f %% of its height at the fundamental plain, %.0f %% bent; "
                "the whole string %.0f / %.0f %% of its energy",
                100.0 * kept[0], 100.0 * kept[1], 100.0 * whole[0], 100.0 * whole[1] ) );
    }

    // 62. Follow Light (Phase 19a): a landscape bright only between 200 and 400 Hz, read by the Aurora
    //     motion. Following the light, the conductor puts far more of its notes there than without
    //     (without, the band is already the middle of its register, which it favours; and its rules
    //     against doublings and crowding keep spreading the chord).
    if ( from <= 62 )
    {
        Landscape band;
        band.m_columns = 128;
        band.m_rows = 64;
        band.m_rowSeconds = 0.5f;
        band.m_amps.assign( static_cast<size_t>( band.m_columns ) * band.m_rows, 0.02f );
        for ( int r = 0; r < band.m_rows; ++r )
            for ( int c = 0; c < band.m_columns; ++c )
            {
                const float hz = landscapeColumnHz( c, band.m_columns );
                if ( hz >= 200.0f && hz <= 400.0f ) band.m_amps[static_cast<size_t>( r ) * band.m_columns + c] = 1.0f;
            }
        double share[2] = { 0.0, 0.0 };
        int    seen[2] = { 0, 0 };
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::MOTION, static_cast<float>( MotionKindE::AURORA ) );
            e.setParam( ParamIdE::MOTION_DEPTH, 1.0f );
            e.setParam( ParamIdE::LANDSCAPE, static_cast<float>( UserLandscape ) );
            e.setParam( ParamIdE::AUTO_MODE, 0.0f );   // Free: notes come and go on their own timers
            e.setParam( ParamIdE::BRAIN_RATE, 2.0f );
            e.setParam( ParamIdE::BRAIN_HOLD_MIN, 1.0f );
            e.setParam( ParamIdE::BRAIN_HOLD_MAX, 3.0f );
            e.setParam( ParamIdE::BRAIN_LOW, 36.0f );
            e.setParam( ParamIdE::BRAIN_HIGH, 84.0f );
            e.setParam( ParamIdE::BRAIN_LIGHT, static_cast<float>( k ) );
            e.setUserLandscape( band );
            if ( !openOffline( e, true, 191 ) ) return 1;
            std::vector<float> L( 256 ), R( 256 );
            int                inBand = 0, all = 0;
            for ( int b = 0; b < 187 * 120; ++b )
            {   // two minutes, a look every second
                e.process( L.data(), R.data(), 256 );
                if ( b % 187 != 0 || b < 187 * 10 ) continue;
                for ( int i = 0; i < MaxViewVoices; ++i )
                {
                    const VoiceView v = e.voiceView( i );
                    if ( v.m_source != 1 || v.m_hz <= 0.0f || v.m_gain <= 0.0f ) continue;
                    ++all;
                    if ( v.m_hz >= 190.0f && v.m_hz <= 420.0f ) ++inBand;
                }
            }
            seen[k] = all;
            share[k] = all > 0 ? static_cast<double>( inBand ) / all : 0.0;
        }
        check(
            seen[0] > 100 && seen[1] > 100 && share[1] > 0.6 && share[1] > share[0] + 0.2,
            "conductor: Follow Light puts the notes where the landscape is bright",
            fmt(
                "fundamentals in the bright band (200 .. 400 Hz): %.0f %% without, %.0f %% following the light (%d / %d voices seen)",
                100.0 * share[0], 100.0 * share[1], seen[0], seen[1] ) );
    }

    // The measuring chord as additiv_presets plays it (conductor off, attack at most half a second,
    // master at 0 dB, 512-sample blocks, 14 s): the description of the last eight seconds and the
    // level of the loudest second -- so that what is measured here is what the bank was measured by.
    auto measureLike = [&]( const float *values, const std::string &mod, Description &d, double &db )
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
        std::string error;
        if ( !e.prepare( c, &error ) ) return false;
        for ( int note : { 50, 57, 62, 66 } ) e.noteOn( note, 0.8f, 0 );
        const int          blocks = static_cast<int>( 14.0 * 48000.0 / 512 ),
                           keepFrom = blocks - static_cast<int>( 8.0 * 48000.0 / 512 );
        std::vector<float> L( 512 ), R( 512 ), keptL, keptR;
        double             sq = 0.0, loudest = 0.0;
        int                inWindow = 0;
        for ( int b = 0; b < blocks; ++b )
        {
            e.process( L.data(), R.data(), 512 );
            if ( b >= keepFrom )
            {
                keptL.insert( keptL.end(), L.begin(), L.end() );
                keptR.insert( keptR.end(), R.begin(), R.end() );
            }
            for ( int i = 0; i < 512; ++i )
                sq += 0.5 * ( static_cast<double>( L[static_cast<size_t>( i )] ) * L[static_cast<size_t>( i )] +
                              static_cast<double>( R[static_cast<size_t>( i )] ) * R[static_cast<size_t>( i )] );
            if ( ++inWindow == 93 )
            {   // 48000 / 512: about a second
                loudest = std::max( loudest, sq / ( 93.0 * 512.0 ) );
                sq = 0.0;
                inWindow = 0;
            }
        }
        d = describe( keptL, keptR, 48000.0 );
        db = 10.0 * std::log10( std::max( loudest, 1e-20 ) );
        return true;
    };

    // 63. Nudges (Phase 20): around a preset, Brighter raises the measured brightness and Wider the
    //     width, each by a good part of what the measured sensitivities predict, the level held; Calmer
    //     (a rule: the motion measured over eight seconds is too unsteady to steer by) slows every rate
    //     that moves something by 1.23 a quarter step and touches no knob at nothing.
    if ( from <= 63 )
    {
        const int   index = 64;   // Choirs' first
        float       values[NumParams], up[NumParams], wide[NumParams], calm[NumParams];
        std::string mod;
        presetValues( index, values, &mod );
        const float wantB = nudge( values, 0, 1.0f, 0.5f, up );
        const float wantW = nudge( values, 2, 1.0f, 0.5f, wide );
        nudge( values, 1, -1.0f, 0.25f, calm );
        Description d0, d1, d2;
        double      l0 = 0.0, l1 = 0.0, l2 = 0.0;
        const bool  ran =
            measureLike( values, mod, d0, l0 ) && measureLike( up, mod, d1, l1 ) && measureLike( wide, mod, d2, l2 );
        const double gotB = ( d1.m_brightness - d0.m_brightness ) / measureSpread( MeasureE::BRIGHTNESS );
        const double gotW = ( d2.m_width - d0.m_width ) / measureSpread( MeasureE::WIDTH );
        // The rule: rates down by 1.23 (within their ranges), nothing that was at nothing moved.
        bool rule = values[static_cast<int>( ParamIdE::MOTION_RATE )] > 0.0f;
        for ( ParamIdE id : { ParamIdE::MOTION_RATE, ParamIdE::DRIFT_RATE, ParamIdE::SHIMMER_RATE } )
        {
            const float v = values[static_cast<int>( id )], c = calm[static_cast<int>( id )];
            if ( v > 0.0f && paramHeard( id, values ) )
                rule = rule && ( std::fabs( c - std::max( v / std::exp2( 0.3f ), paramDesc( id ).m_min ) ) < 1.0e-4f );
        }
        for ( int i = 0; i < NumParams; ++i )
            if ( values[i] == 0.0f && paramDesc( static_cast<ParamIdE>( i ) ).m_min == 0.0f )
                rule = rule && calm[i] == 0.0f && up[i] == 0.0f && wide[i] == 0.0f;
        std::string movedB, movedW;   // the knobs each moved, for the report
        for ( int pass = 0; pass < 2; ++pass )
        {
            const float *to = pass == 0 ? up : wide;
            std::string &s = pass == 0 ? movedB : movedW;
            for ( int i = 0; i < NumParams; ++i )
                if ( to[i] != values[i] )
                    s += fmt( "%s %+.2f ", paramTable()[static_cast<size_t>( i )].m_key,
                              toNorm( paramTable()[static_cast<size_t>( i )], to[i] ) -
                                  toNorm( paramTable()[static_cast<size_t>( i )], values[i] ) );
        }
        check(
            ran && rule && wantB > 0.2f && gotB > 0.4 * wantB && wantW > 0.2f && gotW > 0.4 * wantW &&
                std::fabs( l1 - l0 ) < 1.5 && std::fabs( l2 - l0 ) < 1.5,
            "nudge: brighter is brighter, wider is wider, calmer is slower",
            fmt(
                "brighter: predicted %.2f, measured %.2f spreads (%+.1f dB; %s); wider: predicted %.2f, measured %.2f (%+.1f dB; %s); calmer by its rule: %s",
                wantB, gotB, l1 - l0, movedB.c_str(), wantW, gotW, l2 - l0, movedW.c_str(), rule ? "yes" : "NO" ) );
    }

    // 64. Variations: the same dice give the same variation, other dice another; a few heard shaping
    //     knobs move (four to eight), nothing else but the master that holds the level.
    if ( from <= 64 )
    {
        float values[NumParams], a[NumParams], b[NumParams], c[NumParams];
        presetValues( 64, values, nullptr );
        vary( values, 0.4f, 7, a );
        vary( values, 0.4f, 7, b );
        vary( values, 0.4f, 8, c );
        int             count = 0;
        const ParamIdE *shaping = nullptr;
        int             shapingCount = 0;
        shaping = shapingParams( shapingCount );
        bool onlyShaping = true, onlyHeard = true;
        for ( int i = 0; i < NumParams; ++i )
        {
            if ( a[i] == values[i] || static_cast<ParamIdE>( i ) == ParamIdE::MASTER_GAIN ) continue;
            ++count;
            onlyShaping = onlyShaping && std::find( shaping, shaping + shapingCount, static_cast<ParamIdE>( i ) ) !=
                                             shaping + shapingCount;
            onlyHeard = onlyHeard && paramHeard( static_cast<ParamIdE>( i ), values );
        }
        const bool same = std::equal( a, a + NumParams, b ), differ = !std::equal( a, a + NumParams, c );
        check( same && differ && count >= 1 && count <= 8 && onlyShaping && onlyHeard,
               "variations: repeatable, different, a few heard knobs",
               fmt( "%d knobs moved, %s, %s, %s", count, same ? "the same dice the same" : "NOT repeatable",
                    differ ? "other dice another" : "other dice the SAME",
                    onlyShaping && onlyHeard ? "all heard shaping knobs" : "some NOT heard or not shaping" ) );
    }

    // 65. A macro taught by example: at its places the knobs stand where they were taught, between them
    //     half way, and the macro's value glides them.
    if ( from <= 65 )
    {
        Engine e;
        quietEffects( e );
        if ( !openOffline( e, false, 199 ) ) return 1;
        MacroLesson lesson;
        float       zero[NumParams] = {}, moved[NumParams] = {};
        moved[static_cast<int>( ParamIdE::TILT )] = -0.2f;
        moved[static_cast<int>( ParamIdE::SHIMMER )] = 0.1f;
        lesson.set( 0.0f, zero );
        lesson.set( 1.0f, moved );
        e.setMacroLesson( 0, lesson );
        const float tilt0 = toNorm( paramDesc( ParamIdE::TILT ), e.param( ParamIdE::TILT ) );
        e.setParam( ParamIdE::MACRO1, 0.5f );
        render( e, 0.5 );
        const float half = e.playedNorm( ParamIdE::TILT );
        e.setParam( ParamIdE::MACRO1, 1.0f );
        render( e, 0.5 );
        const float whole = e.playedNorm( ParamIdE::TILT );
        e.setParam( ParamIdE::MACRO1, 0.0f );
        render( e, 0.5 );
        const float none = e.playedNorm( ParamIdE::TILT );
        check( std::fabs( half - ( tilt0 - 0.1f ) ) < 0.005f && std::fabs( whole - ( tilt0 - 0.2f ) ) < 0.005f &&
                   ( std::isnan( none ) || std::fabs( none - tilt0 ) < 0.005f ) && lesson.knobs() == 2,
               "macro: taught by example, it glides between its places",
               fmt( "tilt %.3f untaught, %.3f at Macro 0.5, %.3f at 1, %.3f at 0 again (%d knobs taught)", tilt0, half,
                    whole, std::isnan( none ) ? tilt0 : none, lesson.knobs() ) );
    }

    // 66. Search by example: a preset measured as the bank was measured is found, nearest first.
    if ( from <= 66 )
    {
        const int   index = 100;
        float       values[NumParams];
        std::string mod;
        presetValues( index, values, &mod );
        Description d;
        double      db = 0.0;
        int         found[5] = { -1, -1, -1, -1, -1 };
        float       dist[5] = {};
        const bool  ran = measureLike( values, mod, d, db );
        const int   n = ran ? findLike( d, found, 5, dist ) : 0;
        int         rank = -1;
        for ( int i = 0; i < n; ++i )
            if ( found[i] == index ) rank = i;
        check( ran && rank >= 0 && rank < 3 && dist[rank] < 0.05f, "find: a preset is found by its own sound",
               fmt( "%s found %s (distance %.3f); nearest: %s", preset( index ).m_name,
                    rank >= 0 ? fmt( "at %d", rank + 1 ).c_str() : "NOT", rank >= 0 ? dist[rank] : -1.0f,
                    n > 0 ? preset( found[0] ).m_name : "-" ) );
    }

    // 67. Foundation: with it the lowest sounding note is the root's, in the lowest octave of the
    //     register, nearly all the time, and every chord note stands a fifth above it or higher;
    //     without it the conductor's lowest note is only by chance the root.
    if ( from <= 67 )
    {
        double rootShare[2] = { 0.0, 0.0 };
        int    looks[2] = { 0, 0 }, under = 0;
        for ( int k = 0; k < 2; ++k )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::AUTO_MODE, 1.0f );   // Chords, as nearly every preset plays
            e.setParam( ParamIdE::AUTO_RATE, 4.0f );
            e.setParam( ParamIdE::AUTO_ROOT_MOVE, 0.6f );
            e.setParam( ParamIdE::BRAIN_DENSITY, 5.0f );
            e.setParam( ParamIdE::BRAIN_LOW, 36.0f );
            e.setParam( ParamIdE::BRAIN_HIGH, 79.0f );
            e.setParam( ParamIdE::BRAIN_FOUNDATION, k == 0 ? 0.0f : 0.7f );
            if ( !openOffline( e, true, 67 ) ) return 1;
            std::vector<float> L( 256 ), R( 256 );
            int                rootLowest = 0;
            for ( int b = 0; b < 187 * 150; ++b )
            {   // two and a half minutes, a look every second
                e.process( L.data(), R.data(), 256 );
                if ( b % 187 != 0 || b < 187 * 15 ) continue;
                int              lowest = 128, notes = 0;
                std::vector<int> all;
                for ( int i = 0; i < MaxViewVoices; ++i )
                {
                    const VoiceView v = e.voiceView( i );
                    if ( v.m_source != 1 || v.m_hz <= 0.0f || v.m_gain < 0.3f || v.m_note < 0 ) continue;
                    lowest = std::min( lowest, v.m_note );
                    all.push_back( v.m_note );
                    ++notes;
                }
                if ( notes == 0 ) continue;
                ++looks[k];
                const int root = e.brainRoot();
                if ( ( ( lowest - root ) % 12 + 12 ) % 12 == 0 && lowest < 48 ) ++rootLowest;
                if ( k == 1 )
                    for ( int n : all )
                        if ( n != lowest && n < lowest + 7 ) ++under;   // a chord note crowding the foundation
            }
            rootShare[k] = looks[k] > 0 ? static_cast<double>( rootLowest ) / looks[k] : 0.0;
        }
        check(
            looks[0] > 100 && looks[1] > 100 && rootShare[1] > 0.9 && rootShare[1] > rootShare[0] + 0.3 &&
                under < looks[1] / 10,
            "conductor: Foundation holds the root at the bottom of the register",
            fmt(
                "the lowest note the root, under C3: %.0f %% without, %.0f %% with it; chord notes less than a fifth over it: %d of %d looks",
                100.0 * rootShare[0], 100.0 * rootShare[1], under, looks[1] ) );
    }

    // Phase 26: a plain note to listen into -- no motion, drift or shimmer, the effects off -- and what the
    // engine sets every partial to, from the picture's snapshot (every eighth partial's frequency and target
    // amplitude): a cloud's members beat against each other, so its level in the sound wanders by decibels
    // from second to second; its members' targets do not.
    auto plainNote = []( Engine &e )
    {
        quietEffects( e );
        e.setParam( ParamIdE::MOTION, 0.0f );
        e.setParam( ParamIdE::DRIFT, 0.0f );
        e.setParam( ParamIdE::SHIMMER, 0.0f );
        e.setParam( ParamIdE::CLOUD_WIDTH, 4.0f );
        e.setParam( ParamIdE::FILTER_MODEL, 0.0f );
        e.setParam( ParamIdE::ATTACK, 0.05f );
        e.setParam( ParamIdE::RELEASE, 0.5f );
    };
    // The power of the first voice's members within 50 cents of a frequency (0: all of them), from a snapshot.
    auto powerAt = []( const std::vector<float> &dots, int per, double hz )
    {
        double sum = 0.0;
        for ( int i = 0; i < per; ++i )
        {
            const double f = dots[static_cast<size_t>( 2 * i )], a = dots[static_cast<size_t>( 2 * i + 1 )];
            if ( f <= 0.0 ) continue;
            if ( hz <= 0.0 || std::fabs( 1200.0 * std::log2( f / hz ) ) < 50.0 ) sum += a * a;
        }
        return sum;
    };
    // Runs an engine for a while, handing every snapshot after `from` seconds to `look`.
    auto watch = []( Engine &e, double seconds, double from,
                     const std::function<void( const std::vector<float> &, int, double )> &look )
    {
        std::vector<float> L( 128 ), R( 128 ), dots;
        int                per = 0;
        const int          blocks = static_cast<int>( seconds * 375.0 );
        for ( int k = 0; k < blocks; ++k )
        {
            e.process( L.data(), R.data(), 128 );
            const double t = k / 375.0;
            if ( e.partialSnapshot( dots, per ) && t >= from ) look( dots, per, t );
        }
    };
    auto dB = []( double ratio ) { return 10.0 * std::log10( std::max( ratio, 1.0e-30 ) ); };

    // 68. Overblow (Phase 26a): with breath the octave takes over -- the fundamental falls far under the
    //     second harmonic, the voice's power stays. And it holds: a note overblown, then given less breath,
    //     stays up where a note starting with that little breath stays down (hysteresis).
    if ( from <= 68 )
    {
        auto play = [&]( float first, float then, double &oneOverTwo, double &power )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::OVERBLOW, first );
            if ( !openOffline( e, false, 191 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            render( e, 3.0 );
            const double f0 = soundingHz( e );
            e.setParam( ParamIdE::OVERBLOW, then );
            double p1 = 0.0, p2 = 0.0, all = 0.0;
            watch( e, 4.0, 2.0,
                   [&]( const std::vector<float> &dots, int per, double )
                   {
                       p1 += powerAt( dots, per, f0 );
                       p2 += powerAt( dots, per, 2.0 * f0 );
                       all += powerAt( dots, per, 0.0 );
                   } );
            oneOverTwo = dB( p1 / std::max( p2, 1.0e-30 ) );
            power = all;
            return f0 > 0.0 && all > 0.0;
        };
        double     plain = 0.0, blown = 0.0, held = 0.0, low = 0.0, wPlain = 0.0, wBlown = 0.0, w0 = 0.0, w1 = 0.0;
        const bool ran = play( 0.0f, 0.0f, plain, wPlain ) && play( 0.5f, 0.5f, blown, wBlown ) &&
                         play( 0.5f, 0.25f, held, w0 ) && play( 0.25f, 0.25f, low, w1 );
        check( ran && plain > 5.0 && blown < -20.0 && std::fabs( dB( wBlown / wPlain ) ) < 1.0 && held < -15.0 &&
                   low > 5.0,
               "overblow: the octave takes over, the power stays, and it holds (hysteresis)",
               fmt( "fundamental over octave: %+.1f dB plain, %+.1f blown (power %+.2f dB); at the same little breath "
                    "%+.1f after blowing, %+.1f from the start",
                    plain, blown, dB( wBlown / wPlain ), held, low ) );
    }

    // 69. Seasons (Phase 26b): five families take turns -- over a 20 s year, the fifth's family (harmonic 3)
    //     and the third's (harmonic 5) each swing by more than 10 dB, loudest at different times, while the
    //     voice's power holds within a decibel.
    if ( from <= 69 )
    {
        Engine e;
        plainNote( e );
        e.setParam( ParamIdE::SEASONS, 1.0f );
        e.setParam( ParamIdE::SEASON_TIME, 20.0f );
        bool ran = openOffline( e, false, 193 );
        e.noteOn( 45, 0.8f, 0 );
        render( e, 0.5 );
        const double f0 = soundingHz( e );
        double       lo3 = 1e9, hi3 = -1e9, lo5 = 1e9, hi5 = -1e9, loW = 1e9, hiW = -1e9, at3 = 0.0, at5 = 0.0;
        watch( e, 50.0, 10.0,
               [&]( const std::vector<float> &dots, int per, double t )
               {
                   const double l3 = dB( powerAt( dots, per, 3.0 * f0 ) ), l5 = dB( powerAt( dots, per, 5.0 * f0 ) ),
                                w = dB( powerAt( dots, per, 0.0 ) );
                   if ( l3 > hi3 )
                   {
                       hi3 = l3;
                       at3 = t;
                   }
                   if ( l5 > hi5 )
                   {
                       hi5 = l5;
                       at5 = t;
                   }
                   lo3 = std::min( lo3, l3 );
                   lo5 = std::min( lo5, l5 );
                   loW = std::min( loW, w );
                   hiW = std::max( hiW, w );
               } );
        ran = ran && f0 > 0.0;
        check(
            ran && hi3 - lo3 > 10.0 && hi5 - lo5 > 10.0 && std::fabs( at3 - at5 ) > 1.0 && hiW - loW < 1.0,
            "seasons: the families take turns, the power holds",
            fmt( "harmonic 3 swings %.1f dB (loudest at %.0f s), harmonic 5 %.1f dB (at %.0f s); power within %.2f dB",
                 hi3 - lo3, at3, hi5 - lo5, at5, hiW - loW ) );
    }

    // 70. Avalanche (Phase 26c): the rice pile on a held note -- over a minute, many moments with a harmonic
    //     glinting 4 dB over its usual level, some with six or more glinting at once; without it none. The
    //     same seed renders the same bits.
    if ( from <= 70 )
    {
        auto play = [&]( float depth, int &events, int &large, std::vector<float> *keep )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::AVALANCHE, depth );
            e.setParam( ParamIdE::AVALANCHE_RATE, 1.0f );
            e.setParam( ParamIdE::AVALANCHE_GLOW, 0.4f );
            if ( !openOffline( e, false, 197 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            render( e, 0.5 );
            const double                     f0 = soundingHz( e );
            constexpr int                    Hs = 12;
            std::vector<std::vector<double>> lev( Hs );
            std::vector<float>               lr;
            std::vector<float>               L( 128 ), R( 128 ), dots;
            int                              per = 0;
            for ( int k = 0; k < 375 * 60; ++k )
            {
                e.process( L.data(), R.data(), 128 );
                if ( keep && k < 375 * 10 )
                    for ( int i = 0; i < 128; ++i )
                    {
                        lr.push_back( L[static_cast<size_t>( i )] );
                        lr.push_back( R[static_cast<size_t>( i )] );
                    }
                if ( k >= 375 * 4 && e.partialSnapshot( dots, per ) )
                    for ( int h = 0; h < Hs; ++h )
                        lev[static_cast<size_t>( h )].push_back( dB( powerAt( dots, per, ( h + 2 ) * f0 ) ) );
            }
            if ( keep ) *keep = lr;
            events = large = 0;
            const size_t        n = lev[0].size();
            std::vector<double> median( Hs );
            for ( int h = 0; h < Hs; ++h )
            {
                std::vector<double> col = lev[static_cast<size_t>( h )];
                std::nth_element( col.begin(), col.begin() + static_cast<long>( n / 2 ), col.end() );
                median[static_cast<size_t>( h )] = col[n / 2];
            }
            bool wasLit = false;
            for ( size_t w = 0; w < n; ++w )
            {
                int lit = 0;
                for ( int h = 0; h < Hs; ++h )
                    if ( lev[static_cast<size_t>( h )][w] > median[static_cast<size_t>( h )] + 4.0 ) ++lit;
                if ( lit >= 1 && !wasLit ) ++events;   // a glint begins
                if ( lit >= 6 ) ++large;
                wasLit = lit >= 1;
            }
            return f0 > 0.0 && n > 100;
        };
        int                events = 0, large = 0, quiet = 0, quietLarge = 0, e2 = 0, l2 = 0;
        std::vector<float> a, b;
        const bool         ran =
            play( 1.0f, events, large, &a ) && play( 0.0f, quiet, quietLarge, nullptr ) && play( 1.0f, e2, l2, &b );
        const bool same =
            a.size() == b.size() && !a.empty() && std::memcmp( a.data(), b.data(), sizeof( float ) * a.size() ) == 0;
        check( ran && events >= 10 && large >= 3 && quiet == 0 && same,
               "avalanche: glints of every size on the harmonics, the same each time",
               fmt( "in a minute: %d glints, %d moments with six or more harmonics at once; without it %d; %s", events,
                    large, quiet, same ? "bit for bit" : "differs" ) );
    }

    // 71. Subharmonics (Phase 26d): a loud held note grows a cloud an octave under it from nothing --
    //     absent at first, within 15 dB of the fundamental after 20 s -- and it fades again when the pump
    //     stops; without it there is none. The voice's power stays within a decibel.
    if ( from <= 71 )
    {
        auto play = [&]( float depth, double &early, double &late, double &after, double &power )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::SUBHARMONICS, depth );
            e.setParam( ParamIdE::SUB_TIME, 0.5f );
            if ( !openOffline( e, false, 199 ) ) return false;
            e.noteOn( 45, 1.0f, 0 );
            render( e, 0.2 );
            const double f0 = soundingHz( e );
            auto         ratio = [&]( double from, double to, double &sumAll )
            {
                double sub = 0.0, fund = 0.0;
                sumAll = 0.0;
                watch( e, to - from, 0.0,
                       [&]( const std::vector<float> &dots, int per, double )
                       {
                           sub += powerAt( dots, per, 0.5 * f0 );
                           fund += powerAt( dots, per, f0 );
                           sumAll += powerAt( dots, per, 0.0 );
                       } );
                return dB( sub / std::max( fund, 1.0e-30 ) );
            };
            double w = 0.0;
            render( e, 0.3 );
            early = ratio( 0.5, 1.5, w );
            render( e, 17.5 );
            late = ratio( 19.0, 21.0, power );
            e.setParam( ParamIdE::SUBHARMONICS, 0.0f );   // the pump stops: the floor closes
            render( e, 3.0 );
            after = ratio( 24.0, 25.0, w );
            return f0 > 0.0;
        };
        double     early = 0, late = 0, after = 0, w = 0, early0 = 0, late0 = 0, after0 = 0, w0 = 0;
        const bool ran = play( 1.0f, early, late, after, w ) && play( 0.0f, early0, late0, after0, w0 );
        check( ran && early < -35.0 && late > -15.0 && after < late - 10.0 && late0 < -100.0 &&
                   std::fabs( dB( w / w0 ) ) < 1.0,
               "subharmonics: an octave below grows from nothing, and fades",
               fmt( "half the fundamental against it: %+.1f dB at first, %+.1f after 20 s, %+.1f 3 s after the pump "
                    "stopped; without: %+.1f; power %+.2f dB",
                    early, late, after, late0, dB( w / w0 ) ) );
    }

    // 72. Horizon (Phase 26f): bent fully towards a horizon three octaves up, no partial of the note
    //     lies above it (and its cloud), where without it they reach past 10 kHz; the second harmonic
    //     stays within 15 cents of its place (6 bent, the rest its cloud).
    if ( from <= 72 )
    {
        auto play = [&]( float amount, double &top, double &second, double &f0 )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::HORIZON, amount );
            e.setParam( ParamIdE::HORIZON_HEIGHT, 3.0f );
            if ( !openOffline( e, false, 211 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            std::vector<float> L( 128 ), R( 128 ), dots;
            int                per = 0;
            bool               got = false;
            for ( int k = 0; k < 375 * 2; ++k )
            {
                e.process( L.data(), R.data(), 128 );
                if ( k > 375 && e.partialSnapshot( dots, per ) ) got = true;
            }
            f0 = soundingHz( e );
            top = 0.0;
            second = 0.0;
            for ( int i = 0; got && i < per; ++i )
            {
                const double hz = dots[static_cast<size_t>( 2 * i )], amp = dots[static_cast<size_t>( 2 * i + 1 )];
                if ( amp > 0.0 ) top = std::max( top, hz );
                if ( std::fabs( 1200.0 * std::log2( std::max( hz, 1.0 ) / ( 2.0 * f0 ) ) ) < 60.0 )
                    second = std::max( second, std::fabs( 1200.0 * std::log2( hz / ( 2.0 * f0 ) ) ) );
            }
            return got && f0 > 0.0;
        };
        double     topBent = 0, second = 0, topPlain = 0, second0 = 0, f0 = 0, f1 = 0;
        const bool ran = play( 1.0f, topBent, second, f0 ) && play( 0.0f, topPlain, second0, f1 );
        check( ran && topBent < 8.0 * f0 * 1.06 && topPlain > 10000.0 && second < 15.0,
               "horizon: the harmonics crowd under it and never pass it",
               fmt( "the highest partial %.0f Hz bent (the horizon at %.0f Hz), %.0f Hz plain; the second harmonic's "
                    "members within %.1f cents",
                    topBent, 8.0 * f0, topPlain, second ) );
    }

    // 73. Grains (Phase 26g): all partials in grains a tenth of their time -- in the picture most of them
    //     are silent at any moment, where without grains none is; the level stays within 1.5 dB.
    if ( from <= 73 )
    {
        auto play = [&]( float share, double &silent, double &db )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::CLOUD_WIDTH, 25.0f );
            e.setParam( ParamIdE::GRAIN, share );
            e.setParam( ParamIdE::GRAIN_LENGTH, 0.05f );
            e.setParam( ParamIdE::GRAIN_DENSITY, 0.1f );
            if ( !openOffline( e, false, 223 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            std::vector<float> L( 128 ), R( 128 ), dots, lr;
            int                per = 0, quiet = 0, all = 0;
            for ( int k = 0; k < 375 * 6; ++k )
            {
                e.process( L.data(), R.data(), 128 );
                if ( k >= 375 * 2 )
                    for ( int i = 0; i < 128; ++i )
                    {
                        lr.push_back( L[static_cast<size_t>( i )] );
                        lr.push_back( R[static_cast<size_t>( i )] );
                    }
                if ( k > 375 && e.partialSnapshot( dots, per ) )
                    for ( int i = 0; i < per; ++i )
                    {
                        if ( dots[static_cast<size_t>( 2 * i )] <= 0.0f ) continue;
                        ++all;
                        if ( dots[static_cast<size_t>( 2 * i + 1 )] < 1.0e-7f ) ++quiet;
                    }
            }
            silent = all > 0 ? static_cast<double>( quiet ) / all : 0.0;
            db = rmsDb( lr, 0, lr.size() / 2 );
            return all > 0;
        };
        double     silent = 0, db = 0, silent0 = 0, db0 = 0;
        const bool ran = play( 1.0f, silent, db ) && play( 0.0f, silent0, db0 );
        check( ran && silent > 0.7 && silent0 < 0.01 && std::fabs( db - db0 ) < 1.5,
               "grains: the partials sound in grains, the level stays",
               fmt( "%.0f %% of the partials silent in the picture with grains, %.1f %% without; level %+.2f dB",
                    100.0 * silent, 100.0 * silent0, db - db0 ) );
    }

    // 74. Breathers (Phase 26e): with the string a sine-Gordon chain a struck note's humps stay where they
    //     fell, drifting slowly -- after 8 s most of the string's (periodic) energy is still within 28 cells
    //     (about an octave) of the note's first four harmonics, where without breathers the waves have run
    //     up and down the whole string.
    if ( from <= 74 )
    {
        auto play = [&]( float breathers, double &held )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::RIPPLE_DEPTH, 1.0f );
            e.setParam( ParamIdE::RIPPLE_SPEED, 0.8f );
            e.setParam( ParamIdE::RIPPLE_DECAY, 30.0f );
            e.setParam( ParamIdE::RIPPLE_BREATHERS, breathers );
            if ( !openOffline( e, false, 227 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            render( e, 0.05 );
            const double f0 = soundingHz( e );
            render( e, 8.0 );
            float           u[RippleCells];
            constexpr float perOctave = static_cast<float>( RippleCells - 1 ) / 9.96578428f;
            e.rippleString( u );
            double inside = 0.0, total = 0.0;
            for ( int i = 0; i < RippleCells; ++i )
            {
                const double w = std::tanh( 2.0 * std::sin( 0.5 * u[i] ) ), en = w * w;
                total += en;
                for ( int k = 1; k <= 4; ++k )
                    if ( std::fabs( static_cast<float>( i ) -
                                    std::log2( static_cast<float>( f0 ) * static_cast<float>( k ) / 20.0f ) *
                                        perOctave ) <= 28.0f )
                    {
                        inside += en;
                        break;
                    }
            }
            held = total > 1.0e-9 ? inside / total : 0.0;
            return f0 > 0.0 && total > 1.0e-9;
        };
        double     withB = 0.0, without = 0.0;
        const bool ran = play( 1.0f, withB ) && play( 0.0f, without );
        check( ran && withB > 0.8 && without < 0.65, "ripple: breathers stay where the note struck",
               fmt( "after 8 s, %.0f %% of the string's energy within an octave of where the note struck with "
                    "breathers, %.0f %% without",
                    100.0 * withB, 100.0 * without ) );
    }

    // 75. Phantom (Phase 27a): the phantom layer alone, carrying its melody. Between the ears the phase is turned
    //     half way round at the pitch it carries and not at all a fifth above, while each ear alone has no peak
    //     there: the pitch is only between them.
    if ( from <= 75 )
    {
        Engine e;
        plainNote( e );
        e.setParam( ParamIdE::PHANTOM, 1.0f );
        e.setParam( ParamIdE::PHANTOM_LINE, 0.0f );
        e.setParam( ParamIdE::PHANTOM_WIDTH, 0.12f );
        bool ran = openOffline( e, false, 229 );
        render( e, 1.8 );
        float     hz[PhantomTargets] = {}, w[PhantomTargets] = {};
        const int count = e.phantomPitches( hz, w );
        double    f0 = 0.0;
        for ( int k = 0; k < count; ++k )
            if ( w[k] > 0.99f ) f0 = hz[k];
        const std::vector<float> lr = render( e, 2.0 );   // still its first note (a note holds four to eight seconds)
        const size_t             end = lr.size() / 2;
        auto ipd = [&]( double f )
        {   // the phase of the cross spectrum summed over +-1.5 Hz
            std::complex<double> acc( 0.0, 0.0 );
            for ( double d = -1.5; d <= 1.5; d += 0.5 )
                acc += toneAt( lr, 0, end, f + d, 0 ) * std::conj( toneAt( lr, 0, end, f + d, 1 ) );
            return std::arg( acc );
        };
        auto leftDb = [&]( double f )
        {   // the left ear's power over +-5 Hz
            double p = 0.0;
            for ( double d = -5.0; d <= 5.0; d += 0.5 ) p += std::norm( toneAt( lr, 0, end, f + d, 0 ) );
            return 10.0 * std::log10( std::max( p, 1.0e-30 ) );
        };
        ran = ran && f0 > 0.0;
        const double at = ran ? std::fabs( ipd( f0 ) ) : 0.0, off = ran ? std::fabs( ipd( 1.5 * f0 ) ) : 9.0;
        const double peak = ran ? leftDb( f0 ) - 0.5 * ( leftDb( 0.85 * f0 ) + leftDb( 1.18 * f0 ) ) : 99.0;
        check( ran && at > 2.5 && off < 0.6 && std::fabs( peak ) < 4.0, "phantom: a pitch heard only between the ears",
               fmt( "at its pitch %.0f Hz the ears are %.2f rad apart, a fifth above %.2f; the left ear alone %+.1f dB "
                    "there against beside it",
                    f0, at, off, peak ) );
    }

    // 76. Talbot (Phase 27b): Revival over a 20 s period -- half way the timbre stands upside down (the 24th harmonic
    //     over the fundamental), after the period it is back within two decibels, the voice's power within one.
    //     Bloch: the energy's centre climbs the harmonics half way and comes back.
    if ( from <= 76 )
    {
        auto play = [&]( float mode, double p1[3], double p24[3], double all[3], double centre[3] )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::TILT, 1.0f );
            e.setParam( ParamIdE::TALBOT, 1.0f );
            e.setParam( ParamIdE::TALBOT_MODE, mode );
            e.setParam( ParamIdE::TALBOT_TIME, 20.0f );
            if ( !openOffline( e, false, 231 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            render( e, 0.1 );
            const double f0 = soundingHz( e );
            const double when[3] = { 0.5, 10.5, 20.5 };
            bool         seen[3] = { false, false, false };
            watch( e, 21.0, 0.3,
                   [&]( const std::vector<float> &dots, int per, double t )
                   {
                       for ( int k = 0; k < 3; ++k )
                       {
                           if ( seen[k] || t + 0.1 < when[k] ) continue;
                           seen[k] = true;
                           p1[k] = powerAt( dots, per, f0 );
                           p24[k] = powerAt( dots, per, 24.0 * f0 );
                           all[k] = powerAt( dots, per, 0.0 );
                           double num = 0.0, den = 0.0;
                           for ( int h = 1; h <= 24; ++h )
                           {
                               const double ph = powerAt( dots, per, h * f0 );
                               num += h * ph;
                               den += ph;
                           }
                           centre[k] = den > 0.0 ? num / den : 0.0;
                       }
                   } );
            return f0 > 0.0 && seen[0] && seen[1] && seen[2];
        };
        double       p1[3] = {}, p24[3] = {}, all[3] = {}, c[3] = {}, b1[3] = {}, b24[3] = {}, ball[3] = {}, bc[3] = {};
        const bool   ran = play( 0.0f, p1, p24, all, c ) && play( 1.0f, b1, b24, ball, bc );
        const double before = dB( p1[0] / std::max( p24[0], 1e-30 ) ), half = dB( p1[1] / std::max( p24[1], 1e-30 ) ),
                     back = dB( p1[2] / std::max( p24[2], 1e-30 ) );
        check( ran && before > 10.0 && half < 0.0 && std::fabs( back - before ) < 2.0 &&
                   std::fabs( dB( all[2] / all[0] ) ) < 1.0 && bc[1] > bc[0] + 3.0 && std::fabs( bc[2] - bc[0] ) < 2.0,
               "talbot: the timbre stands upside down half way and comes back",
               fmt( "fundamental over the 24th: %+.1f dB at first, %+.1f half way, %+.1f after the period (power "
                    "%+.2f dB); Bloch's centre: harmonic %.1f, %.1f, %.1f",
                    before, half, back, dB( all[2] / all[0] ), bc[0], bc[1], bc[2] ) );
    }

    // 77. Chimes (Phase 27c): in a wind of 0.6 the chimes strike many times in ten seconds and are heard across the
    //     field (left and right differ); without wind none strikes and the field is silent.
    if ( from <= 77 )
    {
        auto play = [&]( float wind, long long &strikes, double &db, double &corr )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::CHIMES, 1.0f );
            e.setParam( ParamIdE::WIND, wind );
            e.setParam( ParamIdE::GUSTS, 0.5f );
            if ( !openOffline( e, false, 233 ) ) return false;
            render( e, 4.0 );
            const long long          s0 = e.chimeStrikes();
            const std::vector<float> lr = render( e, 10.0 );
            strikes = e.chimeStrikes() - s0;
            db = rmsDb( lr, 0, lr.size() / 2 );
            double ll = 0.0, rr = 0.0, l_r = 0.0;
            for ( size_t i = 0; i < lr.size() / 2; ++i )
            {
                ll += static_cast<double>( lr[2 * i] ) * lr[2 * i];
                rr += static_cast<double>( lr[2 * i + 1] ) * lr[2 * i + 1];
                l_r += static_cast<double>( lr[2 * i] ) * lr[2 * i + 1];
            }
            corr = ll > 0.0 && rr > 0.0 ? l_r / std::sqrt( ll * rr ) : 1.0;
            return true;
        };
        long long  windy = 0, calm = 0;
        double     db = 0.0, calmDb = 0.0, corr = 1.0, calmCorr = 1.0;
        const bool ran = play( 0.6f, windy, db, corr ) && play( 0.0f, calm, calmDb, calmCorr );
        check(
            ran && windy > 30 && db > -70.0 && corr < 0.95 && calm == 0 && calmDb < -100.0,
            "chimes: the wind rings them across the field, the calm leaves them still",
            fmt( "in a wind of 0.6: %lld strikes in 10 s, %.1f dBFS, left and right correlated %.2f; in a calm %lld, "
                 "%.0f dBFS",
                 windy, db, corr, calm, calmDb ) );
    }

    // 78. Dream (Phase 27d): two chords heard, then nothing played: the dream keeps them (at least two memories) and
    //     sings what it remembered -- most of its power within 40 cents of the first six harmonics of the notes that
    //     were played.
    if ( from <= 78 )
    {
        Engine e;
        plainNote( e );
        e.setParam( ParamIdE::DREAM, 1.0f );
        e.setParam( ParamIdE::DREAM_DEPTH, 0.0f );
        e.setParam( ParamIdE::DREAM_FOCUS, 0.6f );
        e.setParam( ParamIdE::DREAM_DRIFT, 0.3f );
        bool                ran = openOffline( e, false, 239 );
        const int           chordA[3] = { 48, 55, 64 }, chordB[3] = { 50, 57, 65 };
        std::vector<double> played;
        for ( const int *chord : { chordA, chordB } )
        {
            for ( int k = 0; k < 3; ++k ) e.noteOn( chord[k], 0.8f, 0 );
            render( e, 0.2 );
            for ( const double f : soundingAll( e ) ) played.push_back( f );
            render( e, 9.8 );
            for ( int k = 0; k < 3; ++k ) e.noteOff( chord[k], 0 );
            render( e, 1.0 );
        }
        render( e, 5.0 );
        const int dreamVoice = 16 + 4 + 1;   // after the voices, the ghosts and the phantom layer
        double    inside = 0.0, total = 0.0;
        watch( e, 4.0, 0.0,
               [&]( const std::vector<float> &dots, int per, double )
               {
                   for ( int i = dreamVoice * per;
                         i < ( dreamVoice + 1 ) * per && 2 * i + 1 < static_cast<int>( dots.size() ); ++i )
                   {
                       const double f = dots[static_cast<size_t>( 2 * i )], a = dots[static_cast<size_t>( 2 * i + 1 )];
                       if ( f <= 0.0 ) continue;
                       total += a * a;
                       bool close = false;
                       for ( const double p : played )
                           for ( int h = 1; h <= 6 && !close; ++h )
                               close = std::fabs( 1200.0 * std::log2( f / ( h * p ) ) ) < 40.0;
                       if ( close ) inside += a * a;
                   }
               } );
        const int memories = e.dreamMemories();
        ran = ran && played.size() >= 6;
        check( ran && memories >= 2 && total > 0.0 && inside / std::max( total, 1e-30 ) > 0.5,
               "dream: it keeps what it heard and sings it when nothing is played",
               fmt( "%d memories; %.0f %% of the dream's power inside the harmonics of the notes played", memories,
                    100.0 * inside / std::max( total, 1e-30 ) ) );
    }

    // 79. Premonition (Phase 27e): the conductor's first note heard coming -- two seconds into its premonition its
    //     cloud is spread three times as far or more as two seconds after it began, and softer.
    if ( from <= 79 )
    {
        Engine e;
        plainNote( e );
        e.setParam( ParamIdE::CLOUD_WIDTH, 10.0f );
        e.setParam( ParamIdE::PREMONITION, 1.0f );
        e.setParam( ParamIdE::PREMONITION_TIME, 4.0f );
        e.setParam( ParamIdE::ATTACK, 0.5f );
        bool ran = openOffline( e, true, 241 );
        e.requestFill();
        double before = -1.0, after = -1.0, gainBefore = 0.0, gainAfter = 0.0;
        watch( e, 30.0, 0.0,
               [&]( const std::vector<float> &dots, int per, double )
               {
                   const VoiceView v = e.voiceView( 0 );
                   if ( v.m_hz <= 0.0f ) return;
                   double sum = 0.0, sq = 0.0;
                   int    count = 0;
                   for ( int i = 0; i < per; ++i )
                   {
                       const double f = dots[static_cast<size_t>( 2 * i )], a = dots[static_cast<size_t>( 2 * i + 1 )];
                       if ( f <= 0.0 || a <= 0.0 ) continue;
                       const double c = 1200.0 * std::log2( f / v.m_hz );
                       if ( std::fabs( c ) > 300.0 ) continue;   // the fundamental's cloud only
                       sum += c;
                       sq += c * c;
                       ++count;
                   }
                   if ( count < 5 ) return;
                   const double spread = std::sqrt( std::max( sq / count - ( sum / count ) * ( sum / count ), 0.0 ) );
                   if ( before < 0.0 && v.m_time > 2.0f && v.m_time < 2.5f )
                   {
                       before = spread;
                       gainBefore = v.m_gain;
                   }
                   if ( after < 0.0 && v.m_time > 6.0f && v.m_time < 6.5f )
                   {
                       after = spread;
                       gainAfter = v.m_gain;
                   }
               } );
        ran = ran && before > 0.0 && after > 0.0;
        check( ran && before > 3.0 * after && gainBefore < gainAfter,
               "premonition: the conductor's note is heard coming, wide and soft",
               fmt( "its fundamental's cloud spread %.1f cents two seconds before it began, %.1f two seconds after; "
                    "level %.2f, then %.2f",
                    before, after, gainBefore, gainAfter ) );
    }

    // 80. Rain (Phase 28a): on a tin roof the drops ring long enough to be counted -- four times the rate, about four
    //     times the drops sounding at once -- and the rain is heard; switched off, it falls silent.
    if ( from <= 80 )
    {
        auto play = [&]( float rate, float rain, float &sounding, double &db )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::RAIN, rain );
            e.setParam( ParamIdE::RAIN_RATE, rate );
            e.setParam( ParamIdE::RAIN_SURFACE, 3.0f );
            e.setParam( ParamIdE::RAIN_SHOWERS, 0.0f );
            if ( !openOffline( e, false, 251 ) ) return false;
            render( e, 4.0 );
            const std::vector<float> lr = render( e, 4.0 );
            float                    nature[10];
            e.natureStats( nature );
            sounding = nature[0];
            db = rmsDb( lr, 0, lr.size() / 2 );
            return true;
        };
        float       few = 0.0f, many = 0.0f, none = 0.0f;
        double      fewDb = 0.0, manyDb = 0.0, noneDb = 0.0;
        const bool  ran = play( 150.0f, 1.0f, few, fewDb ) && play( 600.0f, 1.0f, many, manyDb ) &&
                          play( 600.0f, 0.0f, none, noneDb );
        const float ratio = few > 0.0f ? many / few : 0.0f;
        check( ran && few > 30.0f && ratio > 2.8f && ratio < 5.5f && manyDb > fewDb + 3.0 && fewDb > -70.0 &&
                   noneDb < -100.0,
               "rain: the drops fall at their rate and the rain is heard",
               fmt( "on a tin roof, 150 drops a second: %.0f sounding, %.1f dBFS; 600: %.0f (%.1f times), %.1f dBFS; "
                    "no rain: %.0f dBFS",
                    few, fewDb, many, ratio, manyDb, noneDb ) );
    }

    // 81. Leaves (Phase 28a): in a calm they are silent; they rustle in the wind, louder in a strong one.
    if ( from <= 81 )
    {
        auto play = [&]( float wind, double &db )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::LEAVES, 1.0f );
            e.setParam( ParamIdE::WIND, wind );
            if ( !openOffline( e, false, 253 ) ) return false;
            render( e, 6.0 );
            const std::vector<float> lr = render( e, 6.0 );
            db = rmsDb( lr, 0, lr.size() / 2 );
            return true;
        };
        double     calm = 0.0, breeze = 0.0, gale = 0.0;
        const bool ran = play( 0.0f, calm ) && play( 0.3f, breeze ) && play( 0.9f, gale );
        check( ran && calm < -100.0 && breeze > -80.0 && gale > breeze + 3.0,
               "leaves: silent in a calm, rustling in the wind, louder in a gale",
               fmt( "calm %.0f dBFS, wind 0.3 %.1f dBFS, wind 0.9 %.1f dBFS", calm, breeze, gale ) );
    }

    // 82. Surf (Phase 28a): the sea comes in sets -- its loudness over a minute and a half, second by second, rises and
    //     falls by more than 6 dB -- and a wind raises it.
    if ( from <= 82 )
    {
        auto play = [&]( float wind, double &range, double &db )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::SURF, 1.0f );
            e.setParam( ParamIdE::WIND, wind );
            e.setParam( ParamIdE::SURF_SWELL, 8.0f );
            if ( !openOffline( e, false, 257 ) ) return false;
            render( e, 10.0 );
            const std::vector<float> lr = render( e, 90.0 );
            double                   lo = 1.0e9, hi = -1.0e9;
            for ( size_t s = 0; s + 48000 <= lr.size() / 2; s += 48000 )
            {
                const double d = rmsDb( lr, s, s + 48000 );
                lo = std::min( lo, d );
                hi = std::max( hi, d );
            }
            range = hi - lo;
            db = rmsDb( lr, 0, lr.size() / 2 );
            return true;
        };
        double     range = 0.0, db = 0.0, calmRange = 0.0, calmDb = 0.0;
        const bool ran = play( 0.6f, range, db ) && play( 0.0f, calmRange, calmDb );
        check( ran && range > 6.0 && db > calmDb + 3.0, "surf: the sea comes in sets, and the wind raises it",
               fmt( "in a wind of 0.6: %.1f dBFS, second by second over %.1f dB; in a calm %.1f dBFS", db, range,
                    calmDb ) );
    }

    // 83. Bloom (Phase 28b): a chord struck and let go rings the plate; with its drive, half a second after the chord
    //     ends more of the plate's energy stands above 2 kHz than without -- it has climbed -- and the plate is not
    //     louder for it (the cascade only moves energy).
    if ( from <= 83 )
    {
        auto play = [&]( float drive, float &high, double &db )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::BLOOM, 1.0f );
            e.setParam( ParamIdE::BLOOM_DRIVE, drive );
            e.setParam( ParamIdE::BLOOM_GLIDE, 0.0f );
            e.setParam( ParamIdE::ATTACK, 0.01f );
            e.setParam( ParamIdE::RELEASE, 0.2f );
            e.setParam( ParamIdE::MASTER_GAIN, 6.0f );
            if ( !openOffline( e, false, 259 ) ) return false;
            e.noteOn( 45, 1.0f, 0 );
            e.noteOn( 52, 1.0f, 0 );
            e.noteOn( 57, 1.0f, 0 );
            render( e, 1.0 );
            e.noteOff( 45, 0 );
            e.noteOff( 52, 0 );
            e.noteOff( 57, 0 );
            const std::vector<float> lr = render( e, 0.5 );
            float                    nature[10];
            e.natureStats( nature );
            high = nature[5];
            const std::vector<float> tail = render( e, 4.0 );
            db = rmsDb( tail, 0, tail.size() / 2 );
            return true;
        };
        float      still = 0.0f, bloom = 0.0f;
        double     stillDb = 0.0, bloomDb = 0.0;
        const bool ran = play( 0.0f, still, stillDb ) && play( 1.0f, bloom, bloomDb );
        check( ran && bloom > 2.0f * still + 0.02f && bloomDb < stillDb + 0.5,
               "bloom: the plate's energy climbs into the high modes, and it is not louder for it",
               fmt( "half a second after the chord: above 2 kHz %.1f %% of the plate's energy without drive, %.1f %% "
                    "with; its tail %.1f dBFS, then %.1f dBFS",
                    100.0f * still, 100.0f * bloom, stillDb, bloomDb ) );
    }

    // 84. Swarmalators (Phase 28c): O'Keeffe's states. Like phases gathering and nothing else: a ring whose phase
    //     follows its angle (S high, R low); falling into step too: all pulse as one (R near 1); falling out of step
    //     twice as strongly as they gather: a still disc of phases (both low). (At J = 1, K = -1 the ring turns
    //     instead -- the active phase wave, S 0.55: the disc lies past K = -1.2 J.)
    if ( from <= 84 )
    {
        auto play = [&]( float j, float k, float &R, float &S )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::SWARMAL, 1.0f );
            e.setParam( ParamIdE::SWARMAL_GATHER, j );
            e.setParam( ParamIdE::SWARMAL_SYNC, k );
            e.setParam( ParamIdE::SWARMAL_SPEED, 10.0f );
            e.setParam( ParamIdE::SWARMAL_SPREAD, 0.0f );
            if ( !openOffline( e, false, 263 ) ) return false;
            e.noteOn( 57, 0.8f, 0 );
            render( e, 30.0 );   // 300 of the swarm's time units: out of step it settles last
            float nature[10];
            e.natureStats( nature );
            R = nature[6];
            S = nature[7];
            return true;
        };
        float      waveR = 0.0f, waveS = 0.0f, syncR = 0.0f, syncS = 0.0f, asyncR = 0.0f, asyncS = 0.0f;
        const bool ran =
            play( 1.0f, 0.0f, waveR, waveS ) && play( 1.0f, 1.0f, syncR, syncS ) && play( 0.5f, -1.0f, asyncR, asyncS );
        check( ran && waveS > 0.8f && waveR < 0.2f && syncR > 0.9f && asyncR < 0.2f && asyncS < 0.3f,
               "swarmalators: a phase wave, a synchronised swarm, a still disc",
               fmt( "gathering alone R %.2f S %.2f; falling into step R %.2f; out of step R %.2f S %.2f", waveR, waveS,
                    syncR, asyncR, asyncS ) );
    }

    // 85. Physarum (Phase 28d): a note's partials as crawlers with a long sensor leave the harmonics -- a fifth of
    //     their power or more stands between them after fifteen seconds (without the physarum almost none) -- and the
    //     trail they leave is veined (its spread over its mean well above an even trail's).
    if ( from <= 85 )
    {
        auto play = [&]( bool phys, double &between, float &spread )
        {
            Engine e;
            plainNote( e );
            e.setParam( ParamIdE::FIELD_MODE, phys ? 6.0f : 0.0f );
            e.setParam( ParamIdE::PHYSARUM_SHARE, 1.0f );
            e.setParam( ParamIdE::FIELD_STRENGTH, 0.6f );
            e.setParam( ParamIdE::FIELD_RADIUS, 200.0f );
            e.setParam( ParamIdE::FIELD_SPRING, 0.0f );
            if ( !openOffline( e, false, 269 ) ) return false;
            e.noteOn( 45, 0.8f, 0 );
            render( e, 0.2 );
            const double f0 = soundingHz( e );
            double       off = 0.0, all = 0.0;
            watch( e, 15.0, 14.0,
                   [&]( const std::vector<float> &dots, int per, double )
                   {
                       for ( int i = 0; i < per; ++i )
                       {
                           const double f = dots[static_cast<size_t>( 2 * i )],
                                        a = dots[static_cast<size_t>( 2 * i + 1 )];
                           if ( f <= 0.0 || f0 <= 0.0 ) continue;
                           const double h = std::max( 1.0, std::round( f / f0 ) );
                           all += a * a;
                           if ( std::fabs( 1200.0 * std::log2( f / ( h * f0 ) ) ) > 35.0 ) off += a * a;
                       }
                   } );
            float nature[10];
            e.natureStats( nature );
            spread = nature[9];
            between = all > 0.0 ? off / all : 0.0;
            return f0 > 0.0;
        };
        double     between = 0.0, home = 0.0;
        float      spread = 0.0f, unused = 0.0f;
        const bool ran = play( true, between, spread ) && play( false, home, unused );
        check(
            ran && between > 0.2 && home < 0.05 && spread > 1.0f,
            "physarum: the crawlers leave the harmonics along a veined trail",
            fmt( "after fifteen seconds %.0f %% of the power between the harmonics (without it %.1f %%); the trail's "
                 "spread %.2f",
                 100.0 * between, 100.0 * home, spread ) );
    }

    // 86. Wave room (Phase 29a): a short chord in the room, the room alone (the render with it less the render without:
    //     the engine renders the same, sample for sample) -- its sound lies below 2.5 kHz (the grid's band), and with
    //     rigid walls its tail half a second to a second and a half after the chord is far louder than with walls that
    //     take it up.
    if ( from <= 86 )
    {
        auto play = [&]( float room, float absorb, std::vector<float> &out )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::WAVE_ROOM, room );
            e.setParam( ParamIdE::ROOM_ABSORB, absorb );
            e.setParam( ParamIdE::ROOM_SIZE, 0.6f );
            e.setParam( ParamIdE::ATTACK, 0.01f );
            e.setParam( ParamIdE::RELEASE, 0.15f );
            if ( !openOffline( e, false, 271 ) ) return false;
            e.noteOn( 45, 1.0f, 0 );
            e.noteOn( 52, 1.0f, 0 );
            render( e, 0.3 );
            e.noteOff( 45, 0 );
            e.noteOff( 52, 0 );
            out = render( e, 2.0 );
            return true;
        };
        std::vector<float> dry, rigid, soft;
        const bool         ran = play( 0.0f, 0.0f, dry ) && play( 1.0f, 0.0f, rigid ) && play( 1.0f, 1.0f, soft );
        double             tailRigid = 0.0, tailSoft = 0.0, low = 0.0, all = 0.0;
        if ( ran )
        {
            std::vector<float> r( dry.size() ), s( dry.size() );
            for ( size_t i = 0; i < dry.size(); ++i )
            {
                r[i] = rigid[i] - dry[i];
                s[i] = soft[i] - dry[i];
            }
            tailRigid = rmsDb( r, 24000, 72000 );
            tailSoft = rmsDb( s, 24000, 72000 );
            // its band: a one-pole split at 2.5 kHz, the power below against all of it
            double       lp = 0.0;
            const double k = 1.0 - std::exp( -6.283185307179586 * 2500.0 / 48000.0 );
            for ( size_t i = 0; i < r.size() / 2; ++i )
            {
                const double x = 0.5 * ( r[2 * i] + r[2 * i + 1] );
                lp += k * ( x - lp );
                low += lp * lp;
                all += x * x;
            }
        }
        const double share = all > 0.0 ? low / all : 0.0;
        check(
            ran && tailRigid > -90.0 && tailRigid > tailSoft + 6.0 && share > 0.7,
            "wave room: the room rings below 2.5 kHz, longer between rigid walls",
            fmt( "the room's tail a second after the chord %.1f dBFS between rigid walls, %.1f with soft ones; %.0f %% "
                 "of it below 2.5 kHz",
                 tailRigid, tailSoft, 100.0 * share ) );
    }

    // 87. Topological halo (Phase 29b): a note struck and let go -- its halo stays on the rim and runs round, counter-
    //     clockwise or clockwise as asked, more than a quarter of the way in three seconds; made trivial, the gap holds
    //     nothing and it moves less than a third as far (what moves at all is the strike's edges, outside the gap:
    //     measured -35 degrees against +142).
    if ( from <= 87 )
    {
        auto play = [&]( float turn, float topology, float &moved, float &rim )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::HALO, 1.0f );
            e.setParam( ParamIdE::HALO_SPEED, 8.0f );
            e.setParam( ParamIdE::HALO_TURN, turn );
            e.setParam( ParamIdE::HALO_TOPOLOGY, topology );
            e.setParam( ParamIdE::ATTACK, 0.01f );
            e.setParam( ParamIdE::RELEASE, 0.1f );
            if ( !openOffline( e, false, 277 ) ) return false;
            e.noteOn( 60, 1.0f, 0 );
            render( e, 0.5 );
            e.noteOff( 60, 0 );
            render( e, 0.2 );
            float energy[HaloCarriers], angle[HaloCarriers], rims[HaloCarriers];
            e.haloStats( energy, angle, rims );
            int c = 0;
            for ( int k = 1; k < HaloCarriers; ++k )
                if ( energy[k] > energy[c] ) c = k;
            const float from = angle[c];
            render( e, 3.0 );
            e.haloStats( energy, angle, rims );
            float d = angle[c] - from;
            while ( d > 3.1415927f ) d -= 6.2831853f;
            while ( d < -3.1415927f ) d += 6.2831853f;
            moved = d * 57.29578f;
            rim = rims[c];
            return energy[c] > 0.0f;
        };
        float      ccw = 0.0f, cw = 0.0f, still = 0.0f, rimCcw = 0.0f, rimCw = 0.0f, rimStill = 0.0f;
        const bool ran =
            play( 0.0f, 0.0f, ccw, rimCcw ) && play( 1.0f, 0.0f, cw, rimCw ) && play( 0.0f, 1.0f, still, rimStill );
        check( ran && ccw > 90.0f && cw < -90.0f && rimCcw > 0.8f && rimCw > 0.8f && std::fabs( still ) < ccw / 3.0f,
               "halo: it runs round the rim one way or the other; trivial, it stays",
               fmt( "in three seconds %+.0f degrees counter-clockwise (%.0f %% on the rim), %+.0f clockwise (%.0f %%); "
                    "trivial %+.0f",
                    ccw, 100.0f * rimCcw, cw, 100.0f * rimCw, still ) );
    }

    // 88. Quantum walk (Phase 29c): the root moves on the Tonnetz, and how often to a neighbour there (a fifth or a
    //     fourth, a major or minor third or sixth) does not fall steadily as the wave spreads further between two
    //     looks: at a phase of 2 radians more of the moves go to a neighbour than at 0.5 -- interference, which a
    //     random walk cannot do (simulated: 78 % against 47 %). With it off the root does not move by it.
    if ( from <= 88 )
    {
        auto play = [&]( float spread, long long &moves, long long &toNeighbour )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::QUANTUM, 1.0f );
            e.setParam( ParamIdE::QUANTUM_TIME, 5.0f );
            e.setParam( ParamIdE::QUANTUM_SPREAD, spread );
            if ( !openOffline( e, true, 281 ) ) return false;
            e.requestFill();
            moves = toNeighbour = 0;
            for ( int k = 0; k < 90; ++k )
            {
                render( e, 5.0 );
                int             step = 0;
                const long long m = e.quantumMoves( step );
                if ( m > moves )
                {
                    moves = m;
                    if ( step == 3 || step == 4 || step == 5 || step == 7 || step == 8 || step == 9 ) ++toNeighbour;
                }
            }
            return true;
        };
        long long  m05 = 0, n05 = 0, m2 = 0, n2 = 0;
        const bool ran = play( ( 0.5f - 0.15f ) / 5.85f, m05, n05 ) && play( ( 2.0f - 0.15f ) / 5.85f, m2, n2 );
        Engine     off;
        quietEffects( off );
        bool ranOff = openOffline( off, true, 281 );
        off.requestFill();
        render( off, 60.0 );
        int             unused = 0;
        const long long offMoves = off.quantumMoves( unused );
        const double    s05 = m05 > 0 ? static_cast<double>( n05 ) / m05 : 0.0,
                        s2 = m2 > 0 ? static_cast<double>( n2 ) / m2 : 0.0;
        check( ran && ranOff && m05 >= 20 && m2 >= 20 && s2 > s05 + 0.15 && offMoves == 0,
               "quantum walk: the root moves on the Tonnetz, its steps shaped by interference",
               fmt( "a phase of 0.5 rad: %lld moves, %.0f %% to a neighbour; 2 rad: %lld, %.0f %%; off: %lld", m05,
                    100.0 * s05, m2, 100.0 * s2, offMoves ) );
    }

    // 89. Aeolian harp (Phase 30a): silent in a calm; in a wind its strings sing -- each locked onto a harmonic of its
    //     tuning --, and in a stronger wind higher harmonics (Strouhal's frequency grows with the wind's speed).
    if ( from <= 89 )
    {
        auto play = [&]( float wind, double &db, float &harmonic )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::HARP, 1.0f );
            e.setParam( ParamIdE::WIND, wind );
            e.setParam( ParamIdE::GUSTS, 0.0f );
            if ( !openOffline( e, false, 281 ) ) return false;
            render( e, 12.0 );
            const std::vector<float> lr = render( e, 6.0 );
            db = rmsDb( lr, 0, lr.size() / 2 );
            float night[6];
            e.nightStats( night );
            harmonic = night[1];
            return true;
        };
        double     calm = 0.0, breeze = 0.0, gale = 0.0;
        float      hCalm = 0.0f, hBreeze = 0.0f, hGale = 0.0f;
        const bool ran = play( 0.0f, calm, hCalm ) && play( 0.3f, breeze, hBreeze ) && play( 0.9f, gale, hGale );
        check( ran && calm < -100.0 && breeze > -50.0 && gale > -50.0 && hGale > hBreeze + 1.0,
               "harp: silent in a calm, it sings in the wind, higher harmonics in a gale",
               fmt( "calm %.0f dBFS; wind 0.3 %.1f dBFS at harmonic %.1f; wind 0.9 %.1f dBFS at harmonic %.1f", calm,
                    breeze, hBreeze, gale, hGale ) );
    }

    // 90. Thunder (Phase 30b): a strike is heard as late as it is far (three seconds a kilometre, nothing before), deep
    //     (most of its energy below 400 Hz), and from where it struck: the head turned round (the headset) swaps the
    //     ears.
    if ( from <= 90 )
    {
        auto play = [&]( float yaw, double &delay, double &distance, double &low, double &balance )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::THUNDER, 0.3f );
            e.setParam( ParamIdE::THUNDER_DISTANCE, 1.5f );
            e.setParam( ParamIdE::HEAD_TRACKING, 1.0f );
            e.setParam( ParamIdE::BASS_MONO, 40.0f );   // the rumble is low: not folded to mono at 150 Hz
            if ( !openOffline( e, false, 283 ) ) return false;
            HeadsetPose pose;
            pose.m_valid = true;
            pose.m_yaw = yaw;
            e.setHeadsetPose( pose );
            std::vector<float> all;
            double             strikeAt = -1.0, heardAt = -1.0;
            for ( int k = 0; k < 160 && heardAt < 0.0; ++k )
            {   // a tenth of a second at a time: when it struck, when it is heard
                const std::vector<float> lr = render( e, 0.1 );
                float                    night[6];
                e.nightStats( night );
                if ( strikeAt < 0.0 && night[4] >= 1.0f )
                {
                    strikeAt = 0.1 * k;
                    distance = night[5];
                }
                if ( strikeAt >= 0.0 && rmsDb( lr, 0, lr.size() / 2 ) > -90.0 ) heardAt = 0.1 * k;
            }
            if ( strikeAt < 0.0 || heardAt < 0.0 ) return false;
            delay = heardAt - strikeAt;
            const std::vector<float> lr = render( e, 4.0 );
            // its energy below 400 Hz (a one-pole split), and its balance between the ears
            double       lowE = 0.0, allE = 0.0, l2 = 0.0, r2 = 0.0, y = 0.0;
            const double a = 1.0 - std::exp( -2.0 * 3.14159265358979 * 400.0 / 48000.0 );
            for ( size_t i = 0; i + 1 < lr.size(); i += 2 )
            {
                const double x = 0.5 * ( lr[i] + lr[i + 1] );
                y += a * ( x - y );
                lowE += y * y;
                allE += x * x;
                l2 += static_cast<double>( lr[i] ) * lr[i];
                r2 += static_cast<double>( lr[i + 1] ) * lr[i + 1];
            }
            low = allE > 0.0 ? lowE / allE : 0.0;
            balance = 10.0 * std::log10( std::max( l2, 1e-30 ) / std::max( r2, 1e-30 ) );
            return true;
        };
        // The head turned round and turned to either side: a strike ahead or behind shows in the second pair, one to a
        // side in the first; the stronger contrast must swap the ears.
        double       delay = 0.0, distance = 0.0, low = 0.0, b[4] = {}, unused[3] = {};
        const bool   ran = play( 0.0f, delay, distance, low, b[0] ) &&
                           play( 3.1415927f, unused[0], unused[1], unused[2], b[1] ) &&
                           play( 1.5707963f, unused[0], unused[1], unused[2], b[2] ) &&
                           play( -1.5707963f, unused[0], unused[1], unused[2], b[3] );
        const bool   first = std::fabs( b[0] - b[1] ) >= std::fabs( b[2] - b[3] );
        const double balance = first ? b[0] : b[2], balanceR = first ? b[1] : b[3];
        const double expect = distance / 343.0;
        check( ran && std::fabs( delay - expect ) < 0.4 && low > 0.6 && balance * balanceR < 0.0 &&
                   std::fabs( balance - balanceR ) > 2.0,
               "thunder: heard as late as it is far, deep, and from where it struck",
               fmt( "struck %.0f m off, heard %.1f s later (%.1f s at the speed of sound); %.0f %% below 400 Hz; the "
                    "ears %+.1f dB, the head turned the other way %+.1f dB",
                    distance, delay, expect, 100.0 * low, balance, balanceR ) );
    }

    // 91. Singing ice (Phase 30c): its cracks come as often as the cold says, and each is a chirp that falls -- its
    //     highs arrive first, then lower and lower (flexural waves are dispersive).
    if ( from <= 91 )
    {
        auto play = [&]( float cold, float &chirps, std::vector<float> &lr )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::ICE, 1.0f );
            e.setParam( ParamIdE::ICE_COLD, cold );
            e.setParam( ParamIdE::ICE_THICKNESS, 0.1f );
            if ( !openOffline( e, false, 285 ) ) return false;
            lr = render( e, 20.0 );
            float night[6];
            e.nightStats( night );
            chirps = night[3];
            return true;
        };
        std::vector<float> few, many;
        float              cFew = 0.0f, cMany = 0.0f;
        const bool         ran = play( 0.2f, cFew, few ) && play( 0.9f, cMany, many );
        // In the sparse one, each chirp's zero crossings a second near its start against a tenth of a second later.
        auto zcr = [&]( size_t from, size_t to )
        {
            int crossings = 0;
            for ( size_t i = from + 1; i < to && i < few.size() / 2; ++i )
                if ( ( few[2 * i] + few[2 * i + 1] >= 0.0f ) !=
                     ( few[2 * ( i - 1 )] + few[2 * ( i - 1 ) + 1] >= 0.0f ) )
                    ++crossings;
            return static_cast<double>( crossings ) / ( static_cast<double>( to - from ) / 48000.0 );
        };
        int    falls = 0, chirpsSeen = 0;
        size_t quietSince = 0;
        for ( size_t f = 0; f + 480 < few.size() / 2; f += 480 )
        {
            const double db = rmsDb( few, f, f + 480 );
            if ( db < -70.0 )
            {
                ++quietSince;
                continue;
            }
            if ( quietSince >= 20 && f + 48000 / 4 < few.size() / 2 )
            {   // an onset after a fifth of a second of quiet
                ++chirpsSeen;
                if ( zcr( f, f + 1920 ) > zcr( f + 4800, f + 6720 ) ) ++falls;
            }
            quietSince = 0;
        }
        check( ran && cMany > 2.0f * cFew && cFew >= 2.0f && chirpsSeen >= 2 && falls * 10 >= chirpsSeen * 7,
               "ice: cracks as often as the cold says, each a falling chirp",
               fmt( "in 20 s at cold 0.2 %.0f chirps, at 0.9 %.0f; %d of %d onsets fall", cFew, cMany, falls,
                    chirpsSeen ) );
    }

    // 92. Streams (Phase 30d): with Fusion at nothing the odd and the even harmonics move apart (their envelopes unlike:
    //     two streams), at full they move as one (alike: one sound).
    if ( from <= 92 )
    {
        auto play = [&]( float fusion, double &corr )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::DRIFT, 0.0f );
            e.setParam( ParamIdE::CLOUD_WIDTH, 0.0f );
            e.setParam( ParamIdE::MOTION_DEPTH, 0.0f );
            e.setParam( ParamIdE::AIR, 0.0f );
            e.setParam( ParamIdE::STREAMS, 0.0f );
            e.setParam( ParamIdE::STREAM_DEPTH, 1.0f );
            e.setParam( ParamIdE::STREAM_RATE, 3.0f );
            e.setParam( ParamIdE::FUSION, fusion );
            if ( !openOffline( e, false, 287 ) ) return false;
            e.noteOn( 57, 1.0f, 0 );
            render( e, 1.0 );
            const std::vector<float> lr = render( e, 6.0 );
            const double             f0 = soundingHz( e );
            std::vector<double>      a, b;
            for ( size_t s = 0; s + 1920 <= lr.size() / 2; s += 960 )
            {
                a.push_back( std::pow( 10.0, toneDb( lr, s, s + 1920, f0 ) / 20.0 ) );
                b.push_back( std::pow( 10.0, toneDb( lr, s, s + 1920, 2.0 * f0 ) / 20.0 ) );
            }
            double ma = 0.0, mb = 0.0;
            for ( size_t k = 0; k < a.size(); ++k )
            {
                ma += a[k];
                mb += b[k];
            }
            ma /= static_cast<double>( a.size() );
            mb /= static_cast<double>( b.size() );
            double sab = 0.0, saa = 0.0, sbb = 0.0;
            for ( size_t k = 0; k < a.size(); ++k )
            {
                sab += ( a[k] - ma ) * ( b[k] - mb );
                saa += ( a[k] - ma ) * ( a[k] - ma );
                sbb += ( b[k] - mb ) * ( b[k] - mb );
            }
            corr = sab / std::sqrt( std::max( saa * sbb, 1e-30 ) );
            return f0 > 0.0;
        };
        double     apart = 0.0, fused = 0.0;
        const bool ran = play( 0.0f, apart ) && play( 1.0f, fused );
        check( ran && fused > 0.8 && apart < 0.5, "streams: apart the harmonics move each their own way, fused as one",
               fmt( "the envelopes of harmonics 1 and 2 correlate %.2f apart, %.2f fused", apart, fused ) );
    }

    // 93. Risset's rhythm (Phase 30e): the voices pulse, and over an octave's time the comb of the pulse's tempi slides
    //     down -- by 0.7 of an octave between 15 % and 85 % of it, so its strongest tempo lies 0.3 octaves (mod 1) up
    //     from where it began --, quickening the other way (0.7); it never arrives: the octave starts over.
    if ( from <= 93 )
    {
        auto play = [&]( float way, double &shift, double &depth )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::SHIMMER, 0.0f );
            e.setParam( ParamIdE::RISSET, 1.0f );
            e.setParam( ParamIdE::RISSET_RATE, 2.0f );
            e.setParam( ParamIdE::RISSET_TIME, 60.0f );
            e.setParam( ParamIdE::RISSET_WAY, way );
            if ( !openOffline( e, false, 289 ) ) return false;
            for ( int note : { 48, 55, 60, 64 } ) e.noteOn( note, 1.0f, 0 );
            const std::vector<float> lr = render( e, 60.0 );
            std::vector<double>      env;   // the level every 10 ms
            for ( size_t s = 0; s + 480 <= lr.size() / 2; s += 480 )
            {
                double sq = 0.0;
                for ( size_t i = s; i < s + 480; ++i )
                    sq += static_cast<double>( lr[2 * i] ) * lr[2 * i] +
                          static_cast<double>( lr[2 * i + 1] ) * lr[2 * i + 1];
                env.push_back( std::sqrt( sq / 960.0 ) );
            }
            double lo = 1e9, hi = 0.0;
            for ( size_t k = 500; k < env.size(); ++k )
            {
                lo = std::min( lo, env[k] );
                hi = std::max( hi, env[k] );
            }
            depth = hi > 0.0 ? 20.0 * std::log10( hi / std::max( lo, 1e-9 ) ) : 0.0;
            // the strongest tempo between 1 and 4 Hz in eight seconds about 9 s and 51 s (15 % and 85 % of the octave)
            auto tempo = [&]( size_t mid )
            {
                const size_t from = mid - 400, to = mid + 400;
                double       mean = 0.0;
                for ( size_t k = from; k < to; ++k ) mean += env[k];
                mean /= static_cast<double>( to - from );
                double best = 0.0, at = 0.0;
                for ( double f = 1.0; f <= 4.0; f += 0.005 )
                {
                    std::complex<double> acc( 0.0, 0.0 );
                    for ( size_t k = from; k < to; ++k )
                    {
                        const double w = 0.5 - 0.5 * std::cos( 6.283185307179586 * static_cast<double>( k - from ) /
                                                               static_cast<double>( to - from ) );
                        acc += w * ( env[k] - mean ) *
                               std::polar( 1.0, -6.283185307179586 * f * 0.01 * static_cast<double>( k ) );
                    }
                    if ( std::abs( acc ) > best )
                    {
                        best = std::abs( acc );
                        at = f;
                    }
                }
                return at;
            };
            const double a = tempo( 900 ), b = tempo( 5100 );
            const double oct = std::log2( b / a );
            shift = oct - std::floor( oct );
            return a > 0.0 && b > 0.0;
        };
        double     slowing = 0.0, quickening = 0.0, depth = 0.0, depthQ = 0.0;
        const bool ran = play( 0.0f, slowing, depth ) && play( 1.0f, quickening, depthQ );
        check( ran && std::fabs( slowing - 0.3 ) < 0.15 && std::fabs( quickening - 0.7 ) < 0.15 && depth > 6.0,
               "risset: the pulse's tempi slide down an octave and start over",
               fmt( "the strongest tempo moved %.2f octaves (mod 1) slowing, %.2f quickening; the pulse %.0f dB deep",
                    slowing, quickening, depth ) );
    }

    // 94. Night arc (Phase 30f): the sun where it should be -- Berlin at noon on the June solstice about 61 degrees up,
    //     at midnight on the December one about 61 below --, and at night the arc darkens the sound; by day it changes
    //     nothing.
    if ( from <= 94 )
    {
        const double june = 1782000000.0 + 11.0 * 3600.0 + 8.0 * 60.0;       // 2026-06-21 11:08 UTC, the sun due south
        const double december = 1797811200.0 + 23.0 * 3600.0 + 8.0 * 60.0;   // 2026-12-21 23:08 UTC
        const double up = sunElevation( june, 52.52, 13.40 ), down = sunElevation( december, 52.52, 13.40 );
        auto         play = [&]( double clock, float arc, double &bright )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::NIGHT_ARC, arc );
            e.setParam( ParamIdE::BRIGHTNESS, 0.8f );
            e.setLocation( 52.52, 13.40 );
            e.setRealtime( false );
            e.setParam( ParamIdE::BRAIN_ON, 0.0f );
            EngineConfig c;
            c.m_maxHostBlock = 256;
            c.m_realtimeGpuClass = false;
            c.m_seed = 291;
            c.m_sunClock = clock;
            std::string error;
            if ( !e.prepare( c, &error ) ) return false;
            for ( int note : { 48, 55, 64 } ) e.noteOn( note, 1.0f, 0 );
            render( e, 1.0 );
            const std::vector<float> lr = render( e, 2.0 );
            double                   hiE = 0.0, allE = 0.0, y = 0.0;
            const double             a = 1.0 - std::exp( -2.0 * 3.14159265358979 * 2000.0 / 48000.0 );
            for ( size_t i = 0; i + 1 < lr.size(); i += 2 )
            {   // the share of the energy above 2 kHz
                const double x = 0.5 * ( lr[i] + lr[i + 1] );
                y += a * ( x - y );
                hiE += ( x - y ) * ( x - y );
                allE += x * x;
            }
            bright = allE > 0.0 ? hiE / allE : 0.0;
            return true;
        };
        double     dayOff = 0.0, dayOn = 0.0, nightOff = 0.0, nightOn = 0.0;
        const bool ran = play( june, 0.0f, dayOff ) && play( june, 1.0f, dayOn ) && play( december, 0.0f, nightOff ) &&
                         play( december, 1.0f, nightOn );
        check( ran && std::fabs( up - 60.9 ) < 1.0 && std::fabs( down + 60.9 ) < 1.0 &&
                   std::fabs( dayOn - dayOff ) < 0.02 * dayOff && nightOn < 0.8 * nightOff,
               "night arc: the sun where it stands, and the night darkens the sound",
               fmt( "Berlin: %.1f degrees at the June noon, %.1f at the December midnight; above 2 kHz by day %.1f %% "
                    "(arc %.1f %%), at night %.1f %% (arc %.1f %%)",
                    up, down, 100.0 * dayOff, 100.0 * dayOn, 100.0 * nightOff, 100.0 * nightOn ) );
    }

    // 95. Dream journal (Phase 30g): what the dream kept comes down as a journal, and an instrument given it dreams from
    //     the same memories.
    if ( from <= 95 )
    {
        std::vector<std::uint8_t> journal;
        int                       kept = 0, restored = -1;
        std::uint32_t             count = 0;
        bool                      ran = false;
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::DREAM, 1.0f );
            if ( openOffline( e, false, 293 ) )
            {
                const int chords[4][3] = { { 48, 55, 64 }, { 50, 57, 65 }, { 45, 52, 60 }, { 53, 60, 69 } };
                for ( int k = 0; k < 16; ++k )
                {   // a chord every four seconds, the dream listening
                    for ( int n : chords[k % 4] ) e.noteOn( n, 0.9f, 0 );
                    render( e, 4.0 );
                    for ( int n : chords[k % 4] ) e.noteOff( n, 0 );
                }
                kept = e.dreamMemories();
                ran = e.dreamJournal( journal );
                if ( ran ) std::memcpy( &count, journal.data() + 4, 4 );
            }
        }
        if ( ran )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::DREAM, 1.0f );
            if ( openOffline( e, false, 295 ) )
            {
                e.setDreamJournal( journal );
                render( e, 0.2 );
                restored = e.dreamMemories();
            }
        }
        check( ran && count >= 2 && static_cast<int>( count ) <= kept && restored == static_cast<int>( count ),
               "dream journal: the memories come down and dream on in another instrument",
               fmt( "kept %d, the journal %u (%zu bytes), dreamt from in another: %d", kept, count, journal.size(),
                    restored ) );
    }

    // 96. The headset (Phase 30h): the head turned right, a voice ahead moves into the left ear (the binaural cloud
    //     stays put in the room); a hand at the height of 490 Hz feeds the slime mould there, and its partials gather.
    if ( from <= 96 )
    {
        auto turn = [&]( float yaw, double &balance )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::BINAURAL, 1.0f );
            e.setParam( ParamIdE::STEREO, 0.0f );
            e.setParam( ParamIdE::BINAURAL_HEIGHT, 0.0f );
            e.setParam( ParamIdE::HEAD_TRACKING, 1.0f );
            if ( !openOffline( e, false, 297 ) ) return false;
            HeadsetPose pose;
            pose.m_valid = true;
            pose.m_yaw = yaw;
            e.setHeadsetPose( pose );
            e.noteOn( 57, 1.0f, 0 );
            render( e, 0.5 );
            const std::vector<float> lr = render( e, 2.0 );
            double                   l2 = 0.0, r2 = 0.0;
            for ( size_t i = 0; i + 1 < lr.size(); i += 2 )
            {
                l2 += static_cast<double>( lr[i] ) * lr[i];
                r2 += static_cast<double>( lr[i + 1] ) * lr[i + 1];
            }
            balance = 10.0 * std::log10( std::max( l2, 1e-30 ) / std::max( r2, 1e-30 ) );
            return true;
        };
        auto feed = [&]( bool hand, double &db )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::FIELD_MODE, 6.0f );   // Physarum
            e.setParam( ParamIdE::PHYSARUM_SHARE, 1.0f );
            e.setParam( ParamIdE::VR_HANDS, hand ? 1.0f : 0.0f );
            if ( !openOffline( e, false, 299 ) ) return false;
            HeadsetPose pose;
            pose.m_valid = true;
            pose.m_tracked[0] = true;
            pose.m_hand[0][0] = -0.3f;
            pose.m_hand[0][1] = -0.1f;   // 490 Hz: halfway between 60 Hz and 4 kHz on the log axis
            pose.m_hand[0][2] = 0.4f;
            e.setHeadsetPose( pose );
            e.noteOn( 45, 1.0f, 0 );   // A2: harmonics at 110, 220 .. 440, 550 Hz, none at 490
            render( e, 8.0 );
            const std::vector<float> lr = render( e, 2.0 );
            db = toneDb( lr, 0, lr.size() / 2, 490.0 );
            return true;
        };
        double     ahead = 0.0, turned = 0.0, fed = 0.0, unfed = 0.0;
        const bool ran = turn( 0.0f, ahead ) && turn( 1.5707963f, turned ) && feed( true, fed ) && feed( false, unfed );
        check( ran && std::fabs( ahead ) < 2.0 && turned > ahead + 3.0 && fed > unfed + 6.0,
               "headset: the head turned, the cloud stays; a hand feeds the slime mould at its height",
               fmt( "the ears %+.1f dB with the head ahead, %+.1f turned right; at 490 Hz %.1f dB with the hand, %.1f "
                    "without",
                    ahead, turned, fed, unfed ) );
    }

    // 97. Night chorus (Phase 31): warm crickets call faster than cold ones (Dolbear's law), and callers that answer
    //     each other fall into step -- the most numerous kind's clocks gather.
    if ( from <= 97 )
    {
        auto play = [&]( float warmth, float sync, double &perSecond, double &order )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::CHORUS, 1.0f );
            e.setParam( ParamIdE::CHORUS_KIND, 0.0f );   // a meadow: crickets
            e.setParam( ParamIdE::CHORUS_WARMTH, warmth );
            e.setParam( ParamIdE::CHORUS_SYNC, sync );
            if ( !openOffline( e, false, 301 ) ) return false;
            render( e, 4.0 );
            float s[4];
            e.chorusStats( s );
            const double from = s[0];
            double       sum = 0.0;
            for ( int k = 0; k < 80; ++k )
            {
                render( e, 0.1 );
                e.chorusStats( s );
                sum += s[2];
            }
            e.chorusStats( s );
            perSecond = ( s[0] - from ) / 8.0;
            order = sum / 80.0;
            return true;
        };
        double     warm = 0.0, cold = 0.0, unused = 0.0, apart = 0.0, together = 0.0;
        const bool ran = play( 28.0f, 0.5f, warm, unused ) && play( 10.0f, 0.5f, cold, unused ) &&
                         play( 18.0f, 0.0f, unused, apart ) && play( 18.0f, 1.0f, unused, together );
        check(
            ran && warm > 2.0 * cold && cold > 0.0 && together > apart + 0.1,
            "chorus: warm crickets call faster, and answering they fall into step",
            fmt( "%.0f calls a second at 28 degrees, %.0f at 10; the kind's order %.2f not answering, %.2f answering",
                 warm, cold, apart, together ) );
    }

    // 98. Time mirror (Phase 31): a chord struck and let go in the wave room; without the mirror the room dies away,
    //     with it what the mirror heard comes back reversed in the next window, swelling towards its end.
    if ( from <= 98 )
    {
        auto play = [&]( float mirror, double &chord, double &early, double &late )
        {
            Engine e;
            quietEffects( e );
            e.setParam( ParamIdE::WAVE_ROOM, 1.0f );
            e.setParam( ParamIdE::ROOM_ABSORB, 0.6f );
            e.setParam( ParamIdE::MIRROR, mirror );
            e.setParam( ParamIdE::MIRROR_TIME, 2.0f );
            e.setParam( ParamIdE::MEMORY_LEVEL, 0.0f );   // no ghosts rising after the chord: the room alone
            if ( !openOffline( e, false, 303 ) ) return false;
            for ( int note : { 48, 55, 64 } ) e.noteOn( note, 1.0f, 0 );
            const std::vector<float> a = render( e, 0.3 );
            for ( int note : { 48, 55, 64 } ) e.noteOff( note, 0 );
            const std::vector<float> b = render( e, 3.7 );   // to 4 s: the second window
            early =
                rmsDb( b, static_cast<size_t>( 1.8 * 48000 ), static_cast<size_t>( 2.5 * 48000 ) );      // 2.1 .. 2.8 s
            late = rmsDb( b, static_cast<size_t>( 2.9 * 48000 ), static_cast<size_t>( 3.6 * 48000 ) );   // 3.2 .. 3.9 s
            chord = rmsDb( a, static_cast<size_t>( 0.1 * 48000 ), static_cast<size_t>( 0.3 * 48000 ) );
            return !a.empty();
        };
        double     chord = 0.0, early = 0.0, late = 0.0, chord0 = 0.0, early0 = 0.0, late0 = 0.0;
        const bool ran = play( 1.0f, chord, early, late ) && play( 0.0f, chord0, early0, late0 );
        check( ran && late > early + 6.0 && late > late0 + 10.0 && late < chord + 3.0,
               "time mirror: what the room heard comes back reversed, swelling",
               fmt( "the chord %.1f dBFS; with the mirror %.1f early in the next window, %.1f late; without %.1f, %.1f",
                    chord, early, late, early0, late0 ) );
    }

    // 99. The floor (Phase 32). A chord through a formant filter, which takes every voice's bass: the foundation keeps
    //     its depth through it. A loud harp over it: Floor raises the foundation until the share under 150 Hz is near
    //     its own. Pedal at 1 holds the foundation on the key's tonic while the conductor's root moves. Pressure
    //     makes the floor beat; Sub Octave puts power an octave under it.
    if ( from <= 99 )
    {
        // power under fc, a second-order low pass (RBJ, Q 0.707) over interleaved stereo, per window of w frames
        auto lowWindows = []( const std::vector<float> &lr, double fc, size_t w )
        {
            const double        w0 = 2.0 * 3.14159265358979 * fc / 48000.0, alpha = std::sin( w0 ) / ( 2.0 * 0.7071 ),
                                cw = std::cos( w0 );
            const double        a0 = 1.0 + alpha, b0 = ( 1.0 - cw ) / 2.0 / a0, b1 = ( 1.0 - cw ) / a0, b2 = b0,
                                a1 = -2.0 * cw / a0, a2 = ( 1.0 - alpha ) / a0;
            double              z[2][2] = {};
            std::vector<double> out;
            double              sq = 0.0;
            for ( size_t i = 0; i < lr.size() / 2; ++i )
            {
                for ( int c = 0; c < 2; ++c )
                {
                    const double x = lr[2 * i + static_cast<size_t>( c )], y = b0 * x + z[c][0];
                    z[c][0] = b1 * x - a1 * y + z[c][1];
                    z[c][1] = b2 * x - a2 * y;
                    sq += y * y;
                }
                if ( ( i + 1 ) % w == 0 )
                {
                    out.push_back( sq / ( 2.0 * static_cast<double>( w ) ) );
                    sq = 0.0;
                }
            }
            return out;
        };
        auto power = []( const std::vector<float> &lr )
        {
            double sq = 0.0;
            for ( float x : lr ) sq += static_cast<double>( x ) * x;
            return sq / std::max<double>( 1.0, static_cast<double>( lr.size() ) );
        };
        auto mean = []( const std::vector<double> &v )
        {
            double s = 0.0;
            for ( double x : v ) s += x;
            return v.empty() ? 0.0 : s / static_cast<double>( v.size() );
        };
        struct Floor
        {
            double share = 0.0, deep = 0.0, swing = 0.0, tonic = 0.0, moved = 0.0;
        };
        auto play = [&]( float foundation, float floor, float harp, float pedal, float pressure, float sub, Floor &f )
        {
            auto    held = std::make_unique<Engine>();   // on the heap: main's frame holds enough of them
            Engine &e = *held;
            quietEffects( e );
            e.setParam( ParamIdE::AUTO_MODE, 1.0f );
            e.setParam( ParamIdE::AUTO_RATE, 4.0f );
            e.setParam( ParamIdE::AUTO_ROOT_MOVE, 0.8f );
            e.setParam( ParamIdE::BRAIN_DENSITY, 4.0f );
            e.setParam( ParamIdE::BRAIN_LOW, 36.0f );
            e.setParam( ParamIdE::BRAIN_HIGH, 79.0f );
            e.setParam( ParamIdE::FILTER_MODEL, static_cast<float>( FilterModelE::FORMANT ) );
            e.setParam( ParamIdE::CUTOFF, 1200.0f );
            e.setParam( ParamIdE::BASS_MONO, 40.0f );
            e.setParam( ParamIdE::MOTION_DEPTH, 0.0f );   // the floor itself, not the clouds breathing through it
            e.setParam( ParamIdE::BRAIN_FOUNDATION, foundation );
            e.setParam( ParamIdE::FLOOR, floor );
            e.setParam( ParamIdE::HARP, harp );
            e.setParam( ParamIdE::PEDAL, pedal );
            e.setParam( ParamIdE::PRESSURE, pressure );
            e.setParam( ParamIdE::PRESSURE_RATE, 0.25f );
            e.setParam( ParamIdE::FLOOR_SUB, sub );
            if ( pedal > 0.0f )
            {   // the root moved by the quantum walk, every five seconds
                e.setParam( ParamIdE::QUANTUM, 1.0f );
                e.setParam( ParamIdE::QUANTUM_TIME, 5.0f );
                e.setParam( ParamIdE::QUANTUM_SPREAD, 0.8f );
            }
            if ( !openOffline( e, true, 99 ) ) return false;
            e.requestFill();
            (void)render( e, 20.0 );   // the chord within seconds, the floor followed
            std::vector<float> all;
            int                looks = 0, onTonic = 0, moved = 0;
            for ( int s = 0; s < 40; ++s )
            {   // forty seconds, a look at the foundation and the root each second
                const std::vector<float> a = render( e, 1.0 );
                all.insert( all.end(), a.begin(), a.end() );
                const int fn = e.foundationNow();
                if ( fn < 0 ) continue;
                ++looks;
                if ( fn % 12 == 2 ) ++onTonic;   // the key's tonic: D, Root's default
                if ( ( ( e.brainRoot() % 12 ) + 12 ) % 12 != 2 ) ++moved;
            }
            const std::vector<double> low = lowWindows( all, 150.0, 12000 );   // quarter seconds
            const std::vector<double> deep = lowWindows( all, 55.0, 48000 );
            f.share = mean( low ) / std::max( power( all ), 1e-20 );
            f.deep = 10.0 * std::log10( std::max( mean( deep ) / std::max( power( all ), 1e-20 ), 1e-12 ) );
            // the floor's envelope (quarter seconds): its autocorrelation at Pressure's period (4 s at 0.25 Hz) less
            // that at half of it -- a beat is alike after a period and opposite after half of one; a random swell
            // neither
            const std::vector<double> sub90 = lowWindows( all, 90.0, 12000 );   // the fundamental, not its octave
            std::vector<double>       env( sub90.size() );
            double                    m = 0.0;
            for ( size_t k = 0; k < sub90.size(); ++k ) m += env[k] = std::sqrt( sub90[k] );
            m /= std::max<size_t>( 1, env.size() );
            auto corr = [&]( size_t lag )
            {
                double xy = 0.0, xx = 0.0;
                for ( size_t k = 0; k + lag < env.size(); ++k ) xy += ( env[k] - m ) * ( env[k + lag] - m );
                for ( double x : env ) xx += ( x - m ) * ( x - m );
                return xx > 0.0 ? xy / xx * static_cast<double>( env.size() ) / static_cast<double>( env.size() - lag )
                                : 0.0;
            };
            f.swing = corr( 16 ) - corr( 8 );
            f.tonic = looks > 0 ? static_cast<double>( onTonic ) / looks : 0.0;
            f.moved = looks > 0 ? static_cast<double>( moved ) / looks : 0.0;
            return foundation <= 0.0f || looks > 20;   // without a foundation there is nothing to look at
        };
        Floor      none, kept, covered, raised, pedal, beat, sub;
        const bool ran =
            play( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, none ) && play( 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, kept ) &&
            play( 0.7f, 0.0f, 0.9f, 0.0f, 0.0f, 0.0f, covered ) && play( 0.7f, 1.0f, 0.9f, 0.0f, 0.0f, 0.0f, raised ) &&
            play( 0.7f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, pedal ) && play( 0.7f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, beat ) &&
            play( 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, sub );
        check( ran && kept.share > 0.06 && kept.share > 4.0 * none.share,
               "floor: the foundation keeps its depth through a formant filter",
               fmt( "under 150 Hz %.1f %% of the power with the foundation, %.1f %% without", 100.0 * kept.share,
                    100.0 * none.share ) );
        check( ran && raised.share > covered.share + 0.05, "floor: Floor raises the foundation while a harp covers it",
               fmt( "under 150 Hz with the harp %.1f %%, with Floor %.1f %%", 100.0 * covered.share,
                    100.0 * raised.share ) );
        check( ran && pedal.tonic > 0.95 && pedal.moved > 0.2,
               "floor: Pedal holds the foundation on the tonic while the root moves",
               fmt( "on the tonic %.0f %% of the looks, the root elsewhere %.0f %%", 100.0 * pedal.tonic,
                    100.0 * pedal.moved ) );
        check( ran && beat.swing > 0.6 && beat.swing > kept.swing + 0.4,
               "floor: Pressure makes the floor beat at its rate",
               fmt( "the floor's envelope alike after 4 s less after 2 s: %.2f with Pressure, %.2f without", beat.swing,
                    kept.swing ) );
        check( ran && sub.deep > kept.deep + 6.0, "floor: Sub Octave puts power an octave under the foundation",
               fmt( "under 55 Hz against the whole: %.1f dB with it, %.1f dB without", sub.deep, kept.deep ) );
    }

    // 100. Star (Phase 33): a sun's modes stand where asteroseismology puts them -- the radial ones at
    //      dnu (n + 1.4), dnu the root's (D2) two octaves down -- and between them is little.
    if ( from <= 100 )
    {
        auto    held = std::make_unique<Engine>();   // on the heap: main's frame holds enough of them
        Engine &e = *held;
        quietEffects( e );
        e.setParam( ParamIdE::STAR, 1.0f );
        e.setParam( ParamIdE::STAR_KIND, 0.0f );
        e.setParam( ParamIdE::STAR_PITCH, -2.0f );
        if ( !openOffline( e, false, 100 ) ) return 1;
        (void)render( e, 4.0 );
        const std::vector<float> a = render( e, 10.0 );
        const double             dnu = e.noteHz( 38 ) * 0.25;
        double                   on = 0.0, off = 0.0;
        for ( int n = 15; n <= 27; ++n )
        {
            on += std::pow( 10.0, 0.1 * toneDb( a, 0, a.size() / 2, dnu * ( n + 1.4 ) ) );
            off += std::pow( 10.0, 0.1 * toneDb( a, 0, a.size() / 2, dnu * ( n + 1.6 ) ) );
        }
        const double ratio = 10.0 * std::log10( std::max( on, 1e-30 ) / std::max( off, 1e-30 ) );
        check( ratio > 10.0 && rmsDb( a, 0, a.size() / 2 ) > -50.0, "star: a sun's radial modes on dnu (n + eps)",
               fmt( "dnu %.2f Hz: the radial modes %.1f dB over the gaps between them; the star %.1f dBFS", dnu, ratio,
                    rmsDb( a, 0, a.size() / 2 ) ) );
    }

    // 101. Black hole (Phase 33): a note held at 440 Hz; with the hole a share of its partials falls, redshifted, so
    //      power appears far under the note where there was none.
    if ( from <= 101 )
    {
        auto under = [&]( float hole )
        {
            auto    held = std::make_unique<Engine>();   // on the heap: main's frame holds enough of them
            Engine &e = *held;
            quietEffects( e );
            e.setParam( ParamIdE::BLACK_HOLE, hole );
            e.setParam( ParamIdE::HOLE_FALL, 10.0f );
            e.setParam( ParamIdE::MOTION_DEPTH, 0.0f );
            if ( !openOffline( e, false, 101 ) ) return 0.0;
            e.noteOn( 69, 1.0f, 0 );
            (void)render( e, 6.0 );
            const std::vector<float> a = render( e, 8.0 );
            double                   p = 0.0;
            for ( double hz : { 150.0, 200.0, 250.0, 300.0 } )
                p += std::pow( 10.0, 0.1 * toneDb( a, 0, a.size() / 2, hz ) );
            return 10.0 * std::log10( std::max( p, 1e-30 ) ) - toneDb( a, 0, a.size() / 2, e.noteHz( 69 ) );
        };
        const double without = under( 0.0f ), with = under( 1.0f );
        check( with > without + 10.0, "black hole: falling partials redshifted far under their note",
               fmt( "150 .. 300 Hz against the note: %.1f dB with the hole, %.1f dB without", with, without ) );
    }

    // 102. Merger (Phase 33): asked for at once with a 20 s inspiral and a 70 Hz ringdown -- the field's width swings
    //      while the wave is too slow to hear (a swing of its own on top of the note's, the two adding as powers), and
    //      the ringdown stands far over the bass before it.
    if ( from <= 102 )
    {
        double loudest = -200.0;   // dBFS: the loudest sample of a render with a merger
        auto   play = [&]( float merger, double &peak, double &before, double &swing )
        {
            auto    held = std::make_unique<Engine>();   // on the heap: main's frame holds enough of them
            Engine &e = *held;
            quietEffects( e );
            e.setParam( ParamIdE::MERGER, merger );
            e.setParam( ParamIdE::MERGER_TIME, 20.0f );
            e.setParam( ParamIdE::MERGER_MASS, 70.0f );
            e.setParam( ParamIdE::MERGER_RATIO, 1.0f );
            e.setParam( ParamIdE::MOTION_DEPTH, 0.0f );
            if ( !openOffline( e, false, 102 ) ) return false;
            e.noteOn( 72, 0.8f, 0 );
            e.beginMerger();
            const std::vector<float> a = render( e, 24.0 );
            peak = -200.0;
            for ( size_t s = 0; s + 4800 <= a.size() / 2; s += 2400 )
                peak = std::max( peak, toneDb( a, s, s + 4800, 70.0 ) );
            before = toneDb( a, 2 * 48000, 2 * 48000 + 4800, 70.0 );
            if ( merger > 0.0f )
                for ( float x : a )
                    loudest = std::max( loudest,
                                        20.0 * std::log10( std::max( static_cast<double>( std::abs( x ) ), 1e-12 ) ) );
            // the width: side over mid in tenths of a second, 4 .. 12 s, its spread
            std::vector<double> r;
            for ( size_t s = 4 * 48000; s + 4800 <= 12 * 48000; s += 4800 )
            {
                double m = 0.0, d = 0.0;
                for ( size_t i = s; i < s + 4800; ++i )
                {
                    const double l = a[2 * i], rr = a[2 * i + 1];
                    m += ( l + rr ) * ( l + rr );
                    d += ( l - rr ) * ( l - rr );
                }
                r.push_back( 10.0 * std::log10( std::max( d, 1e-30 ) / std::max( m, 1e-30 ) ) );
            }
            double mean = 0.0, sq = 0.0;
            for ( double x : r ) mean += x;
            mean /= std::max<size_t>( 1, r.size() );
            for ( double x : r ) sq += ( x - mean ) * ( x - mean );
            swing = std::sqrt( sq / std::max<size_t>( 1, r.size() ) );
            return true;
        };
        double     peak = 0.0, before = 0.0, swing = 0.0, peak0 = 0.0, before0 = 0.0, swing0 = 0.0;
        const bool ran = play( 1.0f, peak, before, swing ) && play( 0.0f, peak0, before0, swing0 );
        check( ran && peak > before + 20.0 && peak > peak0 + 20.0 && swing * swing - swing0 * swing0 > 1.5 &&
                   loudest < -6.0,
               "merger: the inspiral swings the field's width, the ringdown rings in the bass",
               fmt( "70 Hz at its height %.1f dB, before %.1f, without a merger %.1f; the width's swing %.2f dB, "
                    "without %.2f; the loudest sample %.1f dBFS",
                    peak, before, peak0, swing, swing0, loudest ) );
    }

    // 103. Redshift hall (Phase 33): a chord into the modal hall and let go; with Redshift the tail sinks -- three
    //      seconds on, its power lies lower than without.
    if ( from <= 103 )
    {
        auto tail = [&]( float red )
        {
            auto    held = std::make_unique<Engine>();   // on the heap: main's frame holds enough of them
            Engine &e = *held;
            quietEffects( e );
            e.setParam( ParamIdE::MODAL_LEVEL, 1.0f );
            e.setParam( ParamIdE::MODAL_DECAY, 12.0f );
            e.setParam( ParamIdE::REDSHIFT, red );
            e.setParam( ParamIdE::REDSHIFT_TIME, 2.0f );
            e.setParam( ParamIdE::RELEASE, 0.2f );
            if ( !openOffline( e, false, 103 ) ) return 0.0;
            for ( int note : { 60, 64, 67, 72 } ) e.noteOn( note, 1.0f, 0 );
            (void)render( e, 0.6 );
            for ( int note : { 60, 64, 67, 72 } ) e.noteOff( note, 0 );
            const std::vector<float> a = render( e, 5.0 );
            // the power-weighted mean octave of the tail, 3 .. 5 s, over a comb of tones every sixth of an octave
            double w = 0.0, sum = 0.0;
            for ( double oct = 0.0; oct <= 7.0; oct += 1.0 / 6.0 )
            {
                const double p = std::pow( 10.0, 0.1 * toneDb( a, 3 * 48000, 5 * 48000, 60.0 * std::exp2( oct ) ) );
                w += p;
                sum += p * oct;
            }
            return w > 0.0 ? sum / w : 0.0;
        };
        const double plain = tail( 0.0f ), sunk = tail( 1.0f );
        check( sunk < plain - 0.7, "redshift hall: the tail sinks as it dies",
               fmt( "the tail's centre %.2f octaves over 60 Hz with Redshift, %.2f without", sunk, plain ) );
    }

    std::printf( g_failures == 0 ? "selftest passed\n" : "selftest: %d FAILED\n", g_failures );
    return g_failures == 0 ? 0 : 1;
}
