/**
 * @file Engine.cpp
 * @brief The instrument (Engine.h): voices, conductors, tuning, the GPU cloud, far reverb, master,
 *        and the worker that renders ahead of the audio callback.
 * @author Rene Weller
 */
#include "additiv/Engine.h"

#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Presets.h"
#include "additiv/Shapes.h"
#include "additiv/Spectrum.h"
#include "additiv/Sun.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <avrt.h>
#include <xmmintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace additiv
{
namespace
{

constexpr std::uint32_t QueueSize = 4096;   ///< note events in flight from the audio thread (a power of two)
constexpr float         VoiceLevel =
    0.1f;   ///< RMS of one voice at full gain, all of its partials together (set so the autoplay sits near -24 dBFS, Noctuary's range)
constexpr double GlideSeconds = 0.3;   ///< how fast a sounding voice follows a change of tuning
constexpr float  PurityNarrowing =
    0.6f;   ///< at Purity 1 the clouds are this much narrower, so pure intervals can lock (PLAN.md)
constexpr float MetresPerUnit = 20.0f;   ///< one unit of distance, for the doppler (Noctuary: about twenty metres)
constexpr float SpeedOfSound = 343.0f;   ///< m/s

/**
 * @brief Denormals off while it lives (flush to zero, denormals are zero), the old mode back after.
 *
 * The effects' tails -- reverbs, echoes, the far hall -- decay into numbers the CPU takes a
 * hundred times longer over. JUCE's ScopedNoDenormals covers only the audio callback, and the
 * blocks are rendered on the worker (or, offline, wherever process() is called). Measured before:
 * the slowest block of a second at 2.5 ms of a block's 2.7 (and an underrun at a standalone's start).
 */
struct NoDenormals
{
    unsigned int m_csr = _mm_getcsr();                        ///< the mode it found
    NoDenormals() { _mm_setcsr( m_csr | 0x8040u ); }          ///< FTZ (bit 15) and DAZ (bit 6) on
    ~NoDenormals() { _mm_setcsr( m_csr ); }                   ///< the mode back
    NoDenormals( const NoDenormals & ) = delete;              ///< one per scope
    NoDenormals &operator=( const NoDenormals & ) = delete;   ///< one per scope
};

/**
 * @brief Raises this process's GPU scheduling class to REALTIME (PLAN.md, Phase 0: the only class
 *        that kept other processes' GPU work from delaying our blocks).
 * @return the NTSTATUS, 0 when it took
 */
long raiseGpuClass()
{
    using SetClassT = long( APIENTRY * )( HANDLE, int );
    HMODULE gdi = GetModuleHandleW( L"gdi32.dll" );
    if ( !gdi ) gdi = LoadLibraryW( L"gdi32.dll" );
    const auto fn = gdi ? reinterpret_cast<SetClassT>( reinterpret_cast<void *>(
                              GetProcAddress( gdi, "D3DKMTSetProcessSchedulingPriorityClass" ) ) )
                        : nullptr;
    return fn ? fn( GetCurrentProcess(), 5 ) : -1;
}

/** @brief Registers the calling thread with MMCSS as "Pro Audio" while it lives. */
struct ProAudioThread
{
    HANDLE m_task = nullptr;   ///< the registration, null when it failed

    /** @brief Registers the thread at high priority within Pro Audio. */
    ProAudioThread()
    {
        DWORD index = 0;
        m_task = AvSetMmThreadCharacteristicsW( L"Pro Audio", &index );
        if ( m_task ) AvSetMmThreadPriority( m_task, AVRT_PRIORITY_HIGH );
    }
    ~ProAudioThread()
    {
        if ( m_task ) AvRevertMmThreadCharacteristics( m_task );
    }
    ProAudioThread( const ProAudioThread & ) = delete;              ///< one registration per thread
    ProAudioThread &operator=( const ProAudioThread & ) = delete;   ///< one registration per thread
};

/**
 * @brief The smallest power of two at least x.
 * @param[in] x  a positive number
 * @return the power
 */
std::int64_t nextPow2( std::int64_t x )
{
    std::int64_t p = 1;
    while ( p < x ) p <<= 1;
    return p;
}

/** @brief The generated landscapes and their RMS, made once (generating one takes up to a tenth of a second). */
struct FactoryLandscapes
{
    std::vector<Landscape> m_land;   ///< NumFactoryLandscapes of them
    std::vector<float>     m_rms;    ///< and their RMS

    FactoryLandscapes()
    {
        for ( int i = 0; i < NumFactoryLandscapes; ++i )
        {
            m_land.push_back( factoryLandscape( i ) );
            m_rms.push_back( landscapeRms( m_land.back() ) );
        }
    }
};

/** @brief The one set, built on first use. @return the landscapes */
const FactoryLandscapes &factoryLandscapes()
{
    static const FactoryLandscapes f;
    return f;
}

/**
 * @brief The Revival's clock (Phase 27): the carpet passed through quarter by quarter, slow at the quarter points
 *        and fast between. Its landmarks -- whole, half and half, mirrored -- are exact only for about a
 *        3300th of the period (the 23rd mode turns 529 times as fast as the first), a flash of milliseconds on a
 *        plain clock; here each is heard for a while, and the timbre leaves it from the finest detail down.
 * @param[in] age     the note's age in seconds
 * @param[in] period  Talbot Time in seconds
 * @return the time the wave function has lived, in seconds; equal to age at every quarter of the period
 */
double talbotClock( double age, double period )
{
    constexpr double Lambda = 17.0;   // e-foldings per quarter: the first modes leave after about 15 % of it
    const double     q = period / 4.0, x = age / q, k = std::floor( x ), u = x - k;
    const double     e = std::expm1( 0.5 * Lambda );
    const double g = u <= 0.5 ? 0.5 * std::expm1( Lambda * u ) / e : 1.0 - 0.5 * std::expm1( Lambda * ( 1.0 - u ) ) / e;
    return q * ( k + g );
}

/**
 * @brief An ice sheet that has cracked for a long time (Phase 30): Olami, Feder and Christensen's slider blocks
 *        (IceN x IceN, IceShare to each neighbour, the rim's lost) driven as the GPU drives them -- every cell raised
 *        until the most stressed one slips, the avalanche run out -- 30000 times from random stress. After about 20000
 *        the cracks' sizes stop growing (measured on 32 x 32: median 1, one in a hundred past 90, the largest a few
 *        hundred), so the sheet sings cracks of every size from its first. Once a process (about 10 ms).
 * @return IceN x IceN stresses, row by row
 */
const std::vector<float> &warmIce()
{
    static const std::vector<float> sheet = []
    {
        constexpr int      N = IceN * IceN;
        std::vector<float> F( N );
        Rng                r( 0x1CE5EE7ull );
        for ( float &f : F ) f = r.uniform();
        std::vector<int> stack;
        stack.reserve( 4 * N );
        for ( int crack = 0; crack < 30000; ++crack )
        {
            const int   at = static_cast<int>( std::max_element( F.begin(), F.end() ) - F.begin() );
            const float add = 1.0f - F[static_cast<size_t>( at )];
            for ( float &f : F ) f += add;
            F[static_cast<size_t>( at )] = 1.0f;
            stack.push_back( at );
            while ( !stack.empty() )
            {
                const int c = stack.back();
                stack.pop_back();
                if ( F[static_cast<size_t>( c )] < 1.0f ) continue;
                const float give = IceShare * F[static_cast<size_t>( c )];
                F[static_cast<size_t>( c )] = 0.0f;
                const int x = c % IceN, y = c / IceN;
                auto      to = [&]( int nx, int ny )
                {
                    if ( nx < 0 || ny < 0 || nx >= IceN || ny >= IceN ) return;
                    float &g = F[static_cast<size_t>( ny * IceN + nx )];
                    g += give;
                    if ( g >= 1.0f ) stack.push_back( ny * IceN + nx );
                };
                to( x + 1, y );
                to( x - 1, y );
                to( x, y + 1 );
                to( x, y - 1 );
            }
        }
        for ( float &f : F ) f = std::min( f, 0.999f );
        return F;
    }();
    return sheet;
}

}   // namespace

Engine::Engine()
{
    resetParams();
    for ( auto &r : m_modRows ) r.store( packModRow( ModRow{} ), std::memory_order_relaxed );
    for ( auto &p : m_played ) p.store( std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed );
    for ( auto &s : m_chanSlide ) s.store( 0.5f, std::memory_order_relaxed );   // MPE: a slide at rest
    double lat = 45.0, lon = 0.0;   // the night arc's place until one is set: the time zone's city
    defaultLocation( lat, lon );
    setLocation( lat, lon );
}

void Engine::setModRow( int i, const ModRow &row )
{
    if ( i < 0 || i >= ModRows ) return;
    ModRow r = row;
    if ( r.m_target >= 0 && !isModTarget( static_cast<ParamIdE>( r.m_target ) ) ) r.m_target = -1;
    r.m_depth = std::clamp( r.m_depth, -1.0f, 1.0f );
    m_modRows[static_cast<size_t>( i )].store( packModRow( r ), std::memory_order_relaxed );
}

ModRow Engine::modRow( int i ) const
{
    if ( i < 0 || i >= ModRows ) return {};
    return unpackModRow( m_modRows[static_cast<size_t>( i )].load( std::memory_order_relaxed ) );
}

std::string Engine::modText() const
{
    ModRow rows[ModRows];
    for ( int i = 0; i < ModRows; ++i ) rows[i] = modRow( i );
    return modRowsToText( rows, ModRows );
}

void Engine::setModText( const std::string &text )
{
    ModRow rows[ModRows];
    modRowsFromText( text, rows, ModRows );
    for ( int i = 0; i < ModRows; ++i ) setModRow( i, rows[i] );
}

Engine::~Engine()
{
    release();
}

void Engine::resetParams()
{
    for ( const ParamDesc &d : paramTable() )
        m_params[static_cast<size_t>( d.m_id )].store( d.m_def, std::memory_order_relaxed );
}

void Engine::setParam( ParamIdE id, float value )
{
    const ParamDesc &d = paramDesc( id );
    m_params[static_cast<size_t>( id )].store( std::clamp( value, d.m_min, d.m_max ), std::memory_order_relaxed );
}

std::string Engine::deviceName() const
{
    return m_prepared ? m_cloud.deviceName() : std::string();
}

bool Engine::prepare( const EngineConfig &config, std::string *error )
{
    release();
    m_cfg = config;
    const int G = m_cfg.m_gpuBlock;
    if ( G < 32 || G > 4096 || ( G & ( G - 1 ) ) != 0 )
    {
        if ( error ) *error = "gpuBlock must be a power of two, 32 .. 4096";
        return false;
    }
    m_cfg.m_maxHostBlock = std::max( m_cfg.m_maxHostBlock, 1 );

    CloudConfig cc;
    cc.m_sampleRate = m_cfg.m_sampleRate;
    cc.m_blockSize = G;
    cc.m_voices = m_cfg.m_voices + std::max( 0, m_cfg.m_ghosts ) + 3;   // and the phantom layer, the dream, the halo
    cc.m_partialsPerVoice = m_cfg.m_partialsPerVoice;
    cc.m_spinWait = true;
    if ( !m_cloud.open( cc, error ) ) return false;
    m_cloud.setLayers( cc.m_voices - 3, cc.m_voices - 2 );
    m_cloud.halo().m_voice = cc.m_voices - 1;
    {   // the hall remembers across a new start (a new sample rate, a new block): its memory goes up again
        const std::lock_guard<std::mutex> lock( m_snapLock );
        if ( !m_hallPendingSet && m_snapHallValid )
        {
            m_hallPending = m_snapHall;
            m_hallPendingAge = m_snapHallAge;
            m_hallPendingSet = true;
        }
    }
    const std::vector<EnvShape> bank = factoryShapes();
    m_cloud.setShapes( bank.data(), static_cast<int>( bank.size() ) );
    factoryLandscapes();   // made now, not in the middle of a block
    m_landChoice = -1;
    if ( m_cfg.m_realtimeGpuClass ) raiseGpuClass();

    m_voices.assign( static_cast<size_t>( m_cfg.m_voices ), Voice{} );
    m_foundationNote = -1;
    for ( Voice &v : m_voices ) v.m_env.setSampleRate( m_cfg.m_sampleRate / G );   // stepped once per GPU block
    m_rng.seed( m_cfg.m_seed );
    m_side.seed( m_cfg.m_seed ^ 0xA5A5A5A55A5A5A5Aull );
    m_tide.init( m_side );
    std::fill( std::begin( m_specKey ), std::end( m_specKey ), -1.0f );
    m_scaleChoice = -1;
    m_mods.reset( m_cfg.m_seed ^ 0x6D6F647363ull );
    for ( OwnUnit &o : m_own ) o.m_live = false;
    m_soundValid = false;
    m_morphT = 1.0f;
    m_morphSeen = m_morphAsked.load( std::memory_order_acquire );
    m_abWas = false;
    m_mapWas = false;
    m_phraseStart = false;
    m_prevSounding = 0;
    readParams();
    refreshSpectrum();
    refreshScale();
    refreshLandscape();
    m_brain.reset( m_rng.fork(), 48 + m_rootPc );
    m_brain2.reset( m_side.fork(), 36 + m_rootPc );   // its own stream: switching it on never moves the first
    m_stepWas = param( ParamIdE::AUTO_STEP ) >= 0.5f;
    m_far.prepare( m_cfg.m_sampleRate );
    m_diffuser.prepare( m_cfg.m_sampleRate );
    m_unmask.prepare( m_cfg.m_sampleRate );
    m_delay.prepare( m_cfg.m_sampleRate );
    m_delay2.prepare( m_cfg.m_sampleRate );
    m_early.prepare( m_cfg.m_sampleRate );
    m_nearRev.prepare( m_cfg.m_sampleRate );
    m_midSide.prepare( m_cfg.m_sampleRate );
    for ( std::vector<float> *b : { &m_nearL,    &m_nearR,    &m_farL,   &m_farR,   &m_dryL,  &m_dryR,  &m_wetL,
                                    &m_wetR,     &m_modalL,   &m_modalR, &m_specL,  &m_specR, &m_meltL, &m_meltR,
                                    &m_phantomL, &m_phantomR, &m_bloomL, &m_bloomR, &m_roomL, &m_roomR } )
        b->assign( static_cast<size_t>( G ), 0.0f );
    m_ghosts.assign( static_cast<size_t>( std::max( 0, m_cfg.m_ghosts ) ), Ghost{} );
    m_ghostRng = Rng( m_cfg.m_seed ^ 0x47686F7374ull );
    m_windRng = Rng( m_cfg.m_seed ^ 0x57696E64ull );
    m_phantomRng = Rng( m_cfg.m_seed ^ 0x5068616Eull );
    m_melodyAt = -1;
    m_melodyNext = 0.0f;
    m_melodyHz[0] = m_melodyHz[1] = 0.0f;
    m_melodyWeight[0] = m_melodyWeight[1] = 0.0f;
    m_echoCount = 0;
    m_gust = 0.0f;
    m_chimesOn = false;
    m_rainLevel = m_leavesLevel = m_surfLevel = m_bloomLevel = m_swarmDepth = 0.0f;
    m_rainWas = m_windWas = m_swarmWas = m_physWas = false;
    for ( float &s : m_shower ) s = 0.0f;
    m_showerRng = Rng( m_cfg.m_seed ^ 0x5261696Eull );
    m_harpLevel = m_thunderLevel = m_iceLevel = 0.0f;
    m_harpWas = m_iceWas = false;
    m_harpHz.fill( 0.0f );
    m_strikeIn = 0.0f;
    m_strikes = 0;
    m_crackIn = 0.0f;
    warmIce();   // once a process, here rather than when the ice first sings
    m_sunTime = m_cfg.m_sunClock > 0.0
                    ? m_cfg.m_sunClock
                    : std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count();
    m_night = 0.0f;
    m_rissetPhase = 0.0;
    m_rissetU = 0.0f;
    m_rissetGain = 1.0f;
    m_rissetPulse = m_lightning = 0.0f;
    m_journalClock = 30.0f;
    m_pose = HeadsetPose{};
    m_chorusLevel = 0.0f;
    m_chorusWas = false;
    m_chorusCalls = 0;
    m_mirrorPos = m_mirrorBuf = 0;
    m_stormRng = Rng( m_cfg.m_seed ^ 0x53746F726Dull );
    m_haloGain = m_roomLevel = 0.0f;
    m_haloNote.fill( -1 );
    m_haloAge.fill( 0.0f );
    m_haloHeld.fill( false );
    m_roomDims[0] = m_roomDims[1] = m_roomDims[2] = 0;
    m_qwPsi.fill( 0.0 );
    m_qwNext = 0.0f;
    m_qwWas = false;
    m_qwMoves = 0;
    m_qwLastStep = 0;
    m_qwRootPc = -1;
    m_qwRng = Rng( m_cfg.m_seed ^ 0x5157616Cull );
    {   // the halo's rim: its sites in the card's order (row by row), those on the outer ring, with their angles
        int s = 0;
        m_haloRimCount = 0;
        for ( int y = -HaloRadius; y <= HaloRadius; ++y )
            for ( int x = -HaloRadius; x <= HaloRadius; ++x )
            {
                if ( x * x + y * y > HaloRadius * HaloRadius ) continue;
                if ( x * x + y * y > ( HaloRadius - 1 ) * ( HaloRadius - 1 ) && m_haloRimCount < HaloSites )
                {
                    m_haloRim[static_cast<size_t>( m_haloRimCount )] = static_cast<short>( s );
                    m_haloRimAngle[static_cast<size_t>( m_haloRimCount )] =
                        std::atan2( static_cast<float>( y ), static_cast<float>( x ) );
                    ++m_haloRimCount;
                }
                ++s;
            }
    }
    m_chimeLevel = m_phantomGain = m_dreamGain = 0.0f;
    m_dreamBlocks = 0;
    m_chimeStrikes.store( 0, std::memory_order_relaxed );
    m_noteRng = Rng( m_cfg.m_seed ^ 0x4E6F7465ull );
    m_randomNote = 0.0f;
    m_roughness = 0.0f;
    m_roughTick = 0;
    m_prevSpecMix = m_prevSpecToFar = 0.0f;
    m_specWasOn = false;
    m_prevMeltMix = m_prevMeltToFar = 0.0f;
    m_meltWasOn = false;
    m_binAngle = 0.0f;
    m_paintTime = 0.0f;
    m_prevDelayMix = m_prevDelayToFar = m_prevDelay2Mix = m_prevDelay2ToFar = m_prevModalLevel = 0.0f;
    m_modalTime = 0.0f;
    m_srcDistance = 0.3f;
    {   // The attractors' Lorenz systems: each started near the butterfly from the seed, apart from the others.
        Rng r( m_cfg.m_seed ^ 0x4C6F72656Eull );
        for ( auto &s : m_lorenz )
        {
            s[0] = 8.0 * r.bipolar();
            s[1] = 8.0 * r.bipolar();
            s[2] = 20.0 + 8.0 * r.bipolar();
        }
    }

    m_queue.assign( QueueSize, Event{} );
    m_qHead.store( 0 );
    m_qTail.store( 0 );
    m_pending.clear();
    m_pending.reserve( QueueSize );

    // The timeline runs two host blocks plus the lookahead ahead of what the callback has taken,
    // rounded up to whole GPU blocks; the ring holds that and room to spare. Two, because a driver
    // may deliver two callbacks back to back (WASAPI did, in the standalone: the ring stood at 448
    // samples for a callback of 480 while the slowest block took 2 ms), and the worker cannot
    // render between them.
    const int ahead =
        2 * m_cfg.m_maxHostBlock + static_cast<int>( std::ceil( m_cfg.m_lookaheadMs * 0.001 * m_cfg.m_sampleRate ) );
    m_latency = ( ahead + G - 1 ) / G * G;
    // The start reserve: as much again rendered before the first callback and not replaced while it
    // is used up, so the first callbacks may come in a burst (a device starting fills its buffer at
    // once: the standalone's ring fell to 224 of 1536 in its first second, with an underrun) while
    // the first GPU blocks are still slow. The latency the host is told stays; an event in the
    // reserve's first stretch comes up to the reserve later (stampFor), never late.
    m_startReserve = m_realtime ? m_latency : 0;   // offline, process() renders what it takes: nothing runs ahead
    const std::int64_t frames =
        nextPow2( static_cast<std::int64_t>( m_latency ) + m_startReserve + m_cfg.m_maxHostBlock + 2 * G );
    m_ring.assign( static_cast<size_t>( frames ) * 2, 0.0f );
    m_ringMask = frames - 1;
    m_renderPos.store( 0 );
    m_outPos.store( 0 );
    m_underruns.store( 0 );
    m_late.store( 0 );
    m_sounding.store( 0 );
    m_prepared = true;
    if ( m_realtime )
    {
        startWorker();
        // The first callback must find the ring full: the first blocks are the slowest (the GPU's
        // module loads on its first launch), and the host may call at once.
        for ( int waited = 0;
              waited < 500 && m_renderPos.load( std::memory_order_acquire ) + G <= m_latency + m_startReserve;
              ++waited )
            Sleep( 1 );
    }
    return true;
}

void Engine::release()
{
    stopWorker();
    if ( m_prepared ) m_cloud.close();
    m_prepared = false;
}

void Engine::setRealtime( bool realtime )
{
    if ( realtime == m_realtime ) return;
    if ( !realtime ) stopWorker();
    m_realtime = realtime;
    if ( m_realtime && m_prepared ) startWorker();
}

void Engine::startWorker()
{
    if ( m_worker.joinable() ) return;
    m_running.store( true );
    m_worker = std::thread( [this] { workerLoop(); } );
}

void Engine::stopWorker()
{
    if ( !m_worker.joinable() ) return;
    m_running.store( false );
    m_worker.join();
}

void Engine::workerLoop()
{
    ProAudioThread pro;
    // A high-resolution timer for the naps: a plain Sleep(1) can last a whole 15.6 ms timer tick,
    // more than the lookahead.
    HANDLE timer = CreateWaitableTimerExW( nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS );
    const int G = m_cfg.m_gpuBlock;
    while ( m_running.load( std::memory_order_relaxed ) )
    {
        const std::int64_t out = m_outPos.load( std::memory_order_acquire );
        std::int64_t       r = m_renderPos.load( std::memory_order_relaxed );
        if ( r < out )
        {
            r = out;
            m_renderPos.store( r, std::memory_order_release );
        }   // after an underrun: catch up
        // Ahead by the latency, and at the start by the reserve not used up yet (out + its rest
        // stays at its end until the callbacks have taken it).
        const std::int64_t reserve = std::max<std::int64_t>( 0, m_startReserve - out );
        if ( r + G <= out + m_latency + reserve )
        {
            renderBlock();
            continue;
        }
        if ( timer )
        {
            LARGE_INTEGER due;
            due.QuadPart = -5000;   // 0.5 ms
            if ( SetWaitableTimerEx( timer, &due, 0, nullptr, nullptr, nullptr, 0 ) )
            {
                WaitForSingleObject( timer, 5 );
                continue;
            }
        }
        Sleep( 1 );
    }
    if ( timer ) CloseHandle( timer );
}

void Engine::noteOn( int note, float velocity, int offset, int channel )
{
    const std::uint32_t head = m_qHead.load( std::memory_order_relaxed );
    if ( !m_prepared || head - m_qTail.load( std::memory_order_acquire ) >= QueueSize ) return;
    m_queue[head & ( QueueSize - 1 )] = { 0, note, std::clamp( velocity, 0.0f, 1.0f ), stampFor( offset ),
                                          std::clamp( channel, -1, 15 ) };
    m_qHead.store( head + 1, std::memory_order_release );
}

void Engine::noteOff( int note, int offset, int channel )
{
    const std::uint32_t head = m_qHead.load( std::memory_order_relaxed );
    if ( !m_prepared || head - m_qTail.load( std::memory_order_acquire ) >= QueueSize ) return;
    m_queue[head & ( QueueSize - 1 )] = { 1, note, 0.0f, stampFor( offset ), std::clamp( channel, -1, 15 ) };
    m_qHead.store( head + 1, std::memory_order_release );
}

void Engine::allNotesOff()
{
    const std::uint32_t head = m_qHead.load( std::memory_order_relaxed );
    if ( !m_prepared || head - m_qTail.load( std::memory_order_acquire ) >= QueueSize ) return;
    m_queue[head & ( QueueSize - 1 )] = { 2, 0, 0.0f, stampFor( 0 ) };
    m_qHead.store( head + 1, std::memory_order_release );
}

void Engine::process( float *L, float *R, int n )
{
    if ( !m_prepared )
    {
        std::fill( L, L + n, 0.0f );
        std::fill( R, R + n, 0.0f );
        return;
    }
    for ( int done = 0; done < n; )
    {
        const int          len = std::min( n - done, m_cfg.m_maxHostBlock );
        const std::int64_t out = m_outPos.load( std::memory_order_relaxed );
        if ( !m_realtime )
            while ( m_renderPos.load( std::memory_order_relaxed ) < out + len ) renderBlock();
        const std::int64_t have = m_renderPos.load( std::memory_order_acquire ) - out;
        const int          take = static_cast<int>( std::clamp<std::int64_t>( have, 0, len ) );
        const int          fill = static_cast<int>( std::clamp<std::int64_t>( have, 0, 1 << 30 ) );
        int                seen = m_minFill.load( std::memory_order_relaxed );
        while ( ( seen < 0 || fill < seen ) &&
                !m_minFill.compare_exchange_weak( seen, fill, std::memory_order_relaxed ) )
        {
        }
        int big = m_maxCallback.load( std::memory_order_relaxed );
        while ( n > big && !m_maxCallback.compare_exchange_weak( big, n, std::memory_order_relaxed ) )
        {
        }
        for ( int i = 0; i < take; ++i )
        {
            const size_t at = static_cast<size_t>( ( out + i ) & m_ringMask ) * 2;
            L[done + i] = m_ring[at];
            R[done + i] = m_ring[at + 1];
        }
        if ( take < len )
        {
            std::fill( L + done + take, L + done + len, 0.0f );
            std::fill( R + done + take, R + done + len, 0.0f );
            m_underruns.fetch_add( 1, std::memory_order_relaxed );
        }
        m_outPos.store( out + len, std::memory_order_release );
        done += len;
    }
}

void Engine::drainEvents()
{
    std::uint32_t       tail = m_qTail.load( std::memory_order_relaxed );
    const std::uint32_t head = m_qHead.load( std::memory_order_acquire );
    while ( tail != head && m_pending.size() < QueueSize ) m_pending.push_back( m_queue[tail++ & ( QueueSize - 1 )] );
    m_qTail.store( tail, std::memory_order_release );
}

void Engine::beginMorph( double seconds )
{
    m_morphSeconds.store( static_cast<float>( std::max( seconds, 0.05 ) ), std::memory_order_relaxed );
    m_morphAsked.fetch_add( 1, std::memory_order_release );
}

void Engine::setMorphSlot( int slot, const float *values, const std::string &modText )
{
    ModRow    rows[ModRows];
    const int n = modRowsFromText( modText, rows, ModRows );
    {
        const std::lock_guard<std::mutex> lock( m_slotLock );
        m_pendingSlots[slot & 1].set( values, rows, n );
    }
    m_slotSet[static_cast<size_t>( slot & 1 )].store( true, std::memory_order_relaxed );
    m_abVersion.fetch_add( 1, std::memory_order_release );
    m_slotVersion.fetch_add( 1, std::memory_order_release );
}

void Engine::setMapTarget( const MorphState &s )
{
    {
        const std::lock_guard<std::mutex> lock( m_slotLock );
        m_pendingSlots[2] = s;
    }
    m_slotSet[2].store( true, std::memory_order_relaxed );
    m_slotVersion.fetch_add( 1, std::memory_order_release );
}

void Engine::loadPreset( int index, double glideSeconds )
{
    if ( index < 0 || index >= numPresets() ) return;
    float       values[NumParams];
    std::string mod;
    presetValues( index, values, &mod );
    if ( glideSeconds > 0.0 ) beginMorph( glideSeconds );
    for ( int i = 0; i < NumParams; ++i )
        if ( !keptByPresets( static_cast<ParamIdE>( i ) ) ) setParam( static_cast<ParamIdE>( i ), values[i] );
    setModText( mod );
}

void Engine::updateSound()
{
    const double dt = m_cfg.m_gpuBlock / m_cfg.m_sampleRate;
    // The slots, when the editor chose one (taken only if the lock is free: the next block tries again).
    bool      slotsChanged = false;
    const int version = m_slotVersion.load( std::memory_order_acquire );
    if ( version != m_slotSeen && m_slotLock.try_lock() )
    {
        const int abVersion = m_abVersion.load( std::memory_order_acquire );
        if ( abVersion != m_abSeen )
        {
            m_slots[0] = m_pendingSlots[0];
            m_slots[1] = m_pendingSlots[1];
            m_abSeen = abVersion;
            slotsChanged = true;
        }
        m_slots[2] = m_pendingSlots[2];
        m_slotLock.unlock();
        m_slotSeen = version;
    }
    // Where the sound goes: the map's blend round its cursor, the A/B blend, or the knobs and the matrix.
    const bool map = param( ParamIdE::MAP_ACTIVE ) >= 0.5f && mapTargetSet();
    const bool ab = !map && param( ParamIdE::MORPH_ACTIVE ) >= 0.5f && morphSlotSet( 0 ) && morphSlotSet( 1 );
    if ( map )
    {
        m_target = m_slots[2];
    }
    else if ( ab )
    {
        blendMorph( m_slots[0], m_slots[1], std::clamp( param( ParamIdE::MORPH_POS ), 0.0f, 1.0f ), m_target );
    }
    else
    {
        float live[NumParams];
        for ( int i = 0; i < NumParams; ++i ) live[i] = param( static_cast<ParamIdE>( i ) );
        ModRow rows[ModRows];
        for ( int r = 0; r < ModRows; ++r )
            rows[r] = unpackModRow( m_modRows[static_cast<size_t>( r )].load( std::memory_order_relaxed ) );
        m_target.set( live, rows, ModRows );
    }
    for ( int i = 0; i < NumParams; ++i )   // the listener's and the player's own stay as they are set
        if ( keptByPresets( static_cast<ParamIdE>( i ) ) || static_cast<ParamIdE>( i ) == ParamIdE::AUTO_STEP )
            m_target.m_values[i] = param( static_cast<ParamIdE>( i ) );
    // On the map the conductor plays or rests as the player set it: a preset made for the hands among
    // the neighbours would otherwise stop it where its weight passes the middle.
    if ( map ) m_target.m_values[static_cast<size_t>( ParamIdE::BRAIN_ON )] = param( ParamIdE::BRAIN_ON );
    // A glide asked for, or the A/B morph switched or re-chosen: from what played the last block.
    float     len = -1.0f;
    const int asked = m_morphAsked.load( std::memory_order_acquire );
    if ( asked != m_morphSeen )
    {
        m_morphSeen = asked;
        len = m_morphSeconds.load( std::memory_order_relaxed );
    }
    if ( ab != m_abWas || ( ab && slotsChanged ) )
    {
        if ( len < 0.0f ) len = std::max( param( ParamIdE::PRESET_GLIDE ), 1.0f );
        m_abWas = ab;
    }
    if ( map != m_mapWas )
    {   // the map taking over or letting go glides too
        if ( len < 0.0f ) len = std::max( param( ParamIdE::MAP_GLIDE ), 1.0f );
        m_mapWas = map;
    }
    if ( len > 0.0f && m_soundValid )
    {
        m_from = m_sound;
        m_morphT = 0.0f;
        m_morphLen = len;
    }
    if ( m_morphT < 1.0f )
    {
        m_morphT = std::min( 1.0f, m_morphT + static_cast<float>( dt ) / m_morphLen );
        if ( m_morphT < 1.0f )
            blendMorph( m_from, m_target, morphCurve( m_morphT ), m_sound );
        else
            m_sound = m_target;
    }
    else if ( map )
    {
        // The map's cursor moves (a hand, a route, automation): the sound follows it, about 95 %
        // of the way in Map Glide.
        const float coef =
            1.0f - std::exp( -static_cast<float>( dt ) / std::max( param( ParamIdE::MAP_GLIDE ) / 3.0f, 0.01f ) );
        approachMorph( m_sound, m_target, coef, m_from );   // m_from is free outside a glide
        m_sound = m_from;
    }
    else
    {
        m_sound = m_target;
    }
    m_soundValid = true;
    m_morphProgress.store( m_morphT, std::memory_order_relaxed );
    for ( int i = 0; i < NumParams; ++i )
        m_soundLive[static_cast<size_t>( i )].store( m_sound.m_values[i], std::memory_order_relaxed );
    m_soundMix.store( m_sound.m_filterMix, std::memory_order_relaxed );
}

void Engine::runOwnModulators( float dt )
{
    bool  read[MorphSetups] = {};
    float weight[MorphSetups] = {};
    for ( int r = 0; r < m_sound.m_rowCount; ++r )
    {
        const ModRow &row = m_sound.m_rows[r];
        if ( ( isOwnSource( row.m_source ) || isOwnSource( row.m_via ) ) && row.m_setup >= 0 &&
             row.m_setup < m_sound.m_setupCount )
        {
            read[row.m_setup] = true;
            weight[row.m_setup] += std::fabs( row.m_depth );
        }
    }
    bool  wanted[OwnUnits] = {};
    int   unmatched[MorphSetups];
    int   pending = 0, strongest = -1;
    float strongestWeight = -1.0f;
    for ( int k = 0; k < m_sound.m_setupCount; ++k )
    {
        m_ownOf[k] = -1;
        if ( !read[k] ) continue;
        for ( int u = 0; u < OwnUnits; ++u )
            if ( m_own[static_cast<size_t>( u )].m_live && !wanted[u] &&
                 m_own[static_cast<size_t>( u )].m_setup == m_sound.m_setups[k] )
            {
                m_ownOf[k] = u;
                wanted[u] = true;
                break;
            }
        if ( m_ownOf[k] < 0 )
            unmatched[pending++] = k;
        else if ( weight[k] > strongestWeight )
        {
            strongestWeight = weight[k];
            strongest = m_ownOf[k];
        }
    }
    for ( int j = 0; j < pending; ++j )
    {
        const int k = unmatched[j];
        int       u = -1;
        for ( int v = 0; v < OwnUnits && u < 0; ++v )   // one no row reads any more runs on with the new settings
            if ( m_own[static_cast<size_t>( v )].m_live && !wanted[v] ) u = v;
        if ( u < 0 )
        {
            for ( int v = 0; v < OwnUnits && u < 0; ++v )
                if ( !m_own[static_cast<size_t>( v )].m_live ) u = v;
            if ( u < 0 ) continue;   // all running: its rows read the panel's (rare: a blend of blends of many sounds)
            m_own[static_cast<size_t>( u )].m_mods =
                strongest >= 0 ? m_own[static_cast<size_t>( strongest )].m_mods : m_mods.own();
            m_own[static_cast<size_t>( u )].m_live = true;
        }
        m_own[static_cast<size_t>( u )].m_setup = m_sound.m_setups[k];
        m_ownOf[k] = u;
        wanted[u] = true;
    }
    for ( int u = 0; u < OwnUnits; ++u )
    {
        OwnUnit &o = m_own[static_cast<size_t>( u )];
        if ( !wanted[u] )
        {
            o.m_live = false;
            continue;
        }
        o.m_mods.advance( dt, m_mods.envTime(), o.m_setup );
    }
}

void Engine::readParams()
{
    // What plays before the matrix: the knobs, the A/B morph, or a glide between (Morph.h).
    updateSound();
    // The modulation: the sources advance by a block, then every row adds its depth times its
    // source to its target in the target's normalised domain (Modulation.h).
    {
        ModInputs in;
        for ( int i = 0; i < 4; ++i )
        {
            in.m_lfoShape[i] = static_cast<int>(
                std::lround( base( static_cast<ParamIdE>( static_cast<int>( ParamIdE::LFO1_SHAPE ) + 2 * i ) ) ) );
            in.m_lfoRate[i] = base( static_cast<ParamIdE>( static_cast<int>( ParamIdE::LFO1_RATE ) + 2 * i ) );
        }
        for ( int e = 0; e < 2; ++e )
        {
            in.m_envShape[e] = static_cast<int>(
                std::lround( base( static_cast<ParamIdE>( static_cast<int>( ParamIdE::ENV1_SHAPE ) + 2 * e ) ) ) );
            in.m_envRate[e] = base( static_cast<ParamIdE>( static_cast<int>( ParamIdE::ENV1_RATE ) + 2 * e ) );
        }
        for ( int m = 0; m < 4; ++m )
            in.m_macro[m] = base( static_cast<ParamIdE>( static_cast<int>( ParamIdE::MACRO1 ) + m ) );
        in.m_wheel = m_wheel.load( std::memory_order_relaxed );
        in.m_pressure = m_pressure.load( std::memory_order_relaxed );
        in.m_velocity = m_lastVelocity;
        in.m_phraseStart = m_phraseStart;
        in.m_coherence = base( ParamIdE::COHERENCE );
        in.m_coherenceRate = base( ParamIdE::COHERENCE_RATE );
        in.m_leniaRate = base( ParamIdE::LENIA_RATE );
        in.m_leniaGrowth = base( ParamIdE::LENIA_GROWTH );
        in.m_chaosPeriod = base( ParamIdE::CHAOS_PERIOD );
        in.m_excitation = static_cast<float>( m_brain.excitation() );
        in.m_rootAge = static_cast<float>( m_brain.rootAgeSeconds() );
        in.m_randomNote = m_randomNote;
        {
            // The Beat (Noctuary's): the two lowest distinct pitches, the simplest ratio near their
            // interval, and the difference of the harmonics that would meet if it were just.
            double f1 = 0.0, f2 = 0.0;
            for ( const Voice &v : m_voices )
                if ( v.m_note >= 0 && v.m_lastGain > 0.0f && ( f1 <= 0.0 || v.m_hz < f1 ) ) f1 = v.m_hz;
            if ( f1 > 0.0 )
                for ( const Voice &v : m_voices )
                    if ( v.m_note >= 0 && v.m_lastGain > 0.0f && v.m_hz / f1 > 1.003 && ( f2 <= 0.0 || v.m_hz < f2 ) )
                        f2 = v.m_hz;
            float want = 0.0f;
            if ( f1 > 0.0 && f2 > 0.0 )
            {
                static const int Ratios[][2] = { { 1, 1 }, { 6, 5 }, { 5, 4 }, { 4, 3 }, { 3, 2 }, { 8, 5 },
                                                 { 5, 3 }, { 7, 4 }, { 2, 1 }, { 5, 2 }, { 3, 1 }, { 4, 1 } };
                const double     r = f2 / f1;
                double           bestErr = 1e30;
                int              bp = 1, bq = 1;
                for ( const auto &pq : Ratios )
                {
                    const double err = std::fabs( std::log( r / ( static_cast<double>( pq[0] ) / pq[1] ) ) );
                    if ( err < bestErr )
                    {
                        bestErr = err;
                        bp = pq[0];
                        bq = pq[1];
                    }
                }
                want =
                    static_cast<float>( std::fabs( static_cast<double>( bq ) * f2 - static_cast<double>( bp ) * f1 ) );
                if ( !( want == want ) || want > 30.0f ) want = 30.0f;
            }
            in.m_beatHz = want;
            // Note and Distance: the loudest voice's, as Noctuary reads its per-voice sources for the matrix.
            const Voice *loud = nullptr;
            for ( const Voice &v : m_voices )
                if ( v.m_note >= 0 && ( loud == nullptr || v.m_lastGain > loud->m_lastGain ) ) loud = &v;
            in.m_note = loud ? ( static_cast<float>( loud->m_note ) - 24.0f ) / 84.0f : 0.5f;
            in.m_distance = loud ? loud->m_distance : 0.5f;
            // What was heard in the last block (the GPU's), and the chord's roughness: Sethares'
            // dissonance curve (after Plomp and Levelt) over the first eight harmonics of every pair of
            // sounding voices, weighted by their loudness and 1/h, over its own weight -- every eighth
            // block (up to 15,000 exponentials; the source is smoothed over a second anyway).
            if ( m_prepared ) m_cloud.heard( in.m_heardDb, in.m_heardCentroid, in.m_heardFlux );
            double     rough = 0.0, weight = 0.0;
            const bool roughNow = --m_roughTick <= 0;
            if ( roughNow ) m_roughTick = 8;
            for ( size_t a = 0; roughNow && a < m_voices.size(); ++a )
            {
                const Voice &va = m_voices[a];
                if ( va.m_note < 0 || va.m_lastGain <= 0.0f ) continue;
                for ( size_t b = a + 1; b < m_voices.size(); ++b )
                {
                    const Voice &vb = m_voices[b];
                    if ( vb.m_note < 0 || vb.m_lastGain <= 0.0f ) continue;
                    for ( int ha = 1; ha <= 8; ++ha )
                        for ( int hb = 1; hb <= 8; ++hb )
                        {
                            const double fa = va.m_hz * partialRatio( ha, m_blk.m_inharmonicB, m_blk.m_stretch,
                                                                      m_blk.m_horizon, m_blk.m_horizonRatio ) +
                                              m_blk.m_shift;
                            const double fb = vb.m_hz * partialRatio( hb, m_blk.m_inharmonicB, m_blk.m_stretch,
                                                                      m_blk.m_horizon, m_blk.m_horizonRatio ) +
                                              m_blk.m_shift;
                            const double s = 0.24 / ( 0.021 * std::min( fa, fb ) + 19.0 ), d = std::fabs( fb - fa ) * s;
                            const double w = static_cast<double>( va.m_lastGain ) * vb.m_lastGain / ( ha * hb );
                            rough += w * ( std::exp( -3.5 * d ) - std::exp( -5.75 * d ) );
                            weight += w;
                        }
                }
            }
            // Scaled so that a minor second, the roughest common interval, reads about 0.55 and a pure
            // fifth under 0.1 (most of the 64 pairs lie far apart and add nothing to the mean).
            if ( roughNow ) m_roughness = weight > 1.0e-12 ? static_cast<float>( 16.0 * rough / weight ) : 0.0f;
            in.m_roughness = m_roughness;
            if ( m_prepared ) m_cloud.reservoirOut( in.m_reservoir );
            in.m_night = m_night;   // Phase 30: the night, Risset's pulse, the lightning
            in.m_risset = m_rissetPulse;
            in.m_lightning = m_lightning;
        }
        m_phraseStart = false;
        m_mods.advance( static_cast<float>( m_cfg.m_gpuBlock / m_cfg.m_sampleRate ), in );
        runOwnModulators( static_cast<float>( m_cfg.m_gpuBlock / m_cfg.m_sampleRate ) );
        for ( int s = 1; s < NumModSources; ++s )
            m_sourceLive[static_cast<size_t>( s )].store( m_mods.value( static_cast<ModSourceE>( s ) ),
                                                          std::memory_order_relaxed );
        // A source as a row reads it: an LFO or envelope from the modulators of the sound the row came with.
        auto source = [this]( ModSourceE s, int setup )
        {
            if ( isOwnSource( s ) && setup >= 0 && setup < m_sound.m_setupCount && m_ownOf[setup] >= 0 )
                return m_own[static_cast<size_t>( m_ownOf[setup] )].m_mods.value( s );
            return m_mods.value( s );
        };
        float offset[NumParams] = {};
        bool  modded[NumParams] = {};
        for ( int r = 0; r < m_sound.m_rowCount; ++r )
        {   // the matrix where the sound stands (a glide weighs two)
            const ModRow &row = m_sound.m_rows[r];
            if ( row.m_source == ModSourceE::OFF || row.m_target < 0 || row.m_target >= NumParams ||
                 row.m_depth == 0.0f )
                continue;
            const float amount =
                row.m_via == ModSourceE::OFF ? 1.0f : viaAmount( row.m_via, source( row.m_via, row.m_setup ) );
            offset[row.m_target] += row.m_depth * amount * source( row.m_source, row.m_setup );
            modded[row.m_target] = true;
        }
        // The macros taught by example (Focus.h), each at its macro's value -- like rows, before the gates.
        {
            std::unique_lock<std::mutex> lock( m_lessonLock, std::try_to_lock );
            if ( lock.owns_lock() )
                for ( size_t m = 0; m < 4; ++m )
                    if ( m_lessonPendingSet[m] )
                    {
                        m_lessons[m] = m_lessonsPending[m];
                        m_lessonPendingSet[m] = false;
                    }
        }
        for ( size_t m = 0; m < 4; ++m )
            if ( m_lessons[m].m_count >= 2 ) m_lessons[m].apply( in.m_macro[m], offset, modded );
        // The night arc (Phase 30): the real night where the instrument is darkens, slows, thins and widens the sound
        // -- a push on five knobs as the matrix's would be, by Night Arc and how deep the night is.
        if ( const float arc = std::clamp( base( ParamIdE::NIGHT_ARC ), 0.0f, 1.0f ) * m_night; arc > 0.0f )
        {
            const std::pair<ParamIdE, float> push[] = { { ParamIdE::BRIGHTNESS, -0.35f },
                                                        { ParamIdE::TILT, 0.15f },
                                                        { ParamIdE::BRAIN_RATE, -0.3f },
                                                        { ParamIdE::BRAIN_DENSITY, -0.2f },
                                                        { ParamIdE::FAR_LEVEL, 0.15f } };
            for ( const auto &[id, by] : push )
            {
                offset[static_cast<int>( id )] += by * arc;
                modded[static_cast<int>( id )] = true;
            }
        }
        // A gated amount is heard by its gate, the matrix's push too: nothing where its kind changes.
        for ( int g = 0; g < MorphGates; ++g ) offset[static_cast<int>( GatedAmounts[g] )] *= m_sound.m_gate[g];
        m_cutoffPush = offset[static_cast<int>( ParamIdE::CUTOFF )];
        m_resonancePush = offset[static_cast<int>( ParamIdE::RESONANCE )];
        for ( int i = 0; i < NumParams; ++i )
        {
            const ParamDesc &d = paramTable()[static_cast<size_t>( i )];
            const float      base = this->base( d.m_id );
            if ( !modded[i] )
            {
                // Whole numbers glide unrounded while the map's cursor is followed: rounded here.
                m_cur[static_cast<size_t>( i )] = d.m_kind == ParamKindE::INT ? std::round( base ) : base;
                m_played[static_cast<size_t>( i )].store( std::numeric_limits<float>::quiet_NaN(),
                                                          std::memory_order_relaxed );
                continue;
            }
            const float n = std::clamp( toNorm( d, base ) + offset[i], 0.0f, 1.0f );
            float       v = fromNorm( d, n );
            if ( d.m_kind == ParamKindE::INT ) v = std::round( v );
            m_cur[static_cast<size_t>( i )] = v;
            m_played[static_cast<size_t>( i )].store( n, std::memory_order_relaxed );
        }
    }
    // The family jam's energy: a follower opens and closes with the leader's section, by up to an
    // octave of cutoff and 40 % of brightness (at 0.5, nothing).
    {
        const float e = std::clamp( m_jamEnergy.load( std::memory_order_relaxed ), 0.0f, 1.0f );
        m_jamCutoff = e != 0.5f ? std::exp2( 2.0f * ( e - 0.5f ) ) : 1.0f;
        if ( e != 0.5f )
        {
            m_cur[static_cast<size_t>( ParamIdE::BRIGHTNESS )] =
                std::clamp( m_cur[static_cast<size_t>( ParamIdE::BRIGHTNESS )] * ( 0.6f + 0.8f * e ), 0.0f, 1.0f );
            m_cur[static_cast<size_t>( ParamIdE::CUTOFF )] = std::clamp(
                m_cur[static_cast<size_t>( ParamIdE::CUTOFF )] * std::exp2( 2.0f * ( e - 0.5f ) ), 40.0f, 18000.0f );
        }
    }
    auto g = [this]( ParamIdE id ) { return cur( id ); };
    m_blk.m_master = dbToGain( g( ParamIdE::MASTER_GAIN ) );
    m_blk.m_partials = static_cast<int>( std::lround( g( ParamIdE::PARTIALS ) ) );
    m_blk.m_tilt = g( ParamIdE::TILT );
    m_blk.m_brightness = g( ParamIdE::BRIGHTNESS );
    m_blk.m_oddEven = g( ParamIdE::ODD_EVEN );
    m_blk.m_inharmonicB = inharmonicB( g( ParamIdE::INHARMONIC ) );
    m_blk.m_stretch = g( ParamIdE::STRETCH ) / 1200.0f;
    m_blk.m_shift = g( ParamIdE::SHIFT );
    m_blk.m_purity = g( ParamIdE::PURITY );
    m_blk.m_width = g( ParamIdE::CLOUD_WIDTH ) * ( 1.0f - PurityNarrowing * m_blk.m_purity );
    m_blk.m_widthSlope = g( ParamIdE::WIDTH_SLOPE );
    m_blk.m_drift = g( ParamIdE::DRIFT );
    m_blk.m_driftRate = g( ParamIdE::DRIFT_RATE );
    m_blk.m_shimmer = g( ParamIdE::SHIMMER );
    m_blk.m_shimmerRate = g( ParamIdE::SHIMMER_RATE );
    m_blk.m_streams = std::clamp( static_cast<int>( std::lround( g( ParamIdE::STREAMS ) ) ), 0, NumStreams - 1 );
    m_blk.m_streamDepth = g( ParamIdE::STREAM_DEPTH );
    m_blk.m_streamRate = g( ParamIdE::STREAM_RATE );
    m_blk.m_fusion = g( ParamIdE::FUSION );
    m_blk.m_stereo = g( ParamIdE::STEREO );
    m_blk.m_air = g( ParamIdE::AIR );
    m_blk.m_airCenter = g( ParamIdE::AIR_CENTER );
    m_blk.m_airWidth = g( ParamIdE::AIR_WIDTH );
    m_blk.m_airJitter = g( ParamIdE::AIR_JITTER );
    m_blk.m_filterModel = static_cast<int>( std::lround( g( ParamIdE::FILTER_MODEL ) ) );
    m_blk.m_cutoff = g( ParamIdE::CUTOFF );
    m_blk.m_resonance = g( ParamIdE::RESONANCE );
    m_blk.m_filterEnv = g( ParamIdE::FILTER_ENV );
    m_blk.m_keyTrack = g( ParamIdE::KEY_TRACK );
    m_blk.m_filterDrift = g( ParamIdE::FILTER_DRIFT );
    m_blk.m_motion = static_cast<int>( std::lround( g( ParamIdE::MOTION ) ) );
    m_blk.m_motionRate = g( ParamIdE::MOTION_RATE );
    m_blk.m_motionDepth = g( ParamIdE::MOTION_DEPTH );
    m_blk.m_attack = g( ParamIdE::ATTACK );
    m_blk.m_decay = g( ParamIdE::DECAY );
    m_blk.m_sustain = g( ParamIdE::SUSTAIN );
    m_blk.m_release = g( ParamIdE::RELEASE );
    m_blk.m_refPitch = g( ParamIdE::REF_PITCH );
    m_blk.m_tide = g( ParamIdE::TIDE );
    m_blk.m_tidePeriod = g( ParamIdE::TIDE_PERIOD );
    m_blk.m_depth = g( ParamIdE::DEPTH );
    m_blk.m_keysDepth = g( ParamIdE::KEYS_DEPTH );
    m_blk.m_farLevel = g( ParamIdE::FAR_LEVEL );
    m_blk.m_presence = g( ParamIdE::PRESENCE );
    m_blk.m_padLowCut = g( ParamIdE::PAD_LOW_CUT );
    m_blk.m_breath = g( ParamIdE::BREATH );
    m_blk.m_breathRate = g( ParamIdE::BREATH_RATE );
    m_blk.m_doppler = g( ParamIdE::DOPPLER );
    m_blk.m_brain2 = g( ParamIdE::BRAIN2_ON ) >= 0.5f;
    m_blk.m_brain2Interval = static_cast<int>( std::lround( g( ParamIdE::BRAIN2_INTERVAL ) ) );
    m_blk.m_brain2Depth = g( ParamIdE::BRAIN2_DEPTH );
    m_blk.m_overblow = g( ParamIdE::OVERBLOW );
    m_blk.m_overblowTime = g( ParamIdE::OVERBLOW_TIME );
    m_blk.m_seasons = g( ParamIdE::SEASONS );
    m_blk.m_seasonTime = g( ParamIdE::SEASON_TIME );
    m_blk.m_avalanche = g( ParamIdE::AVALANCHE );
    m_blk.m_avalancheRate = g( ParamIdE::AVALANCHE_RATE );
    m_blk.m_avalancheGlow = g( ParamIdE::AVALANCHE_GLOW );
    m_blk.m_subharmonics = g( ParamIdE::SUBHARMONICS );
    m_blk.m_subTime = g( ParamIdE::SUB_TIME );
    m_blk.m_horizon = g( ParamIdE::HORIZON );
    m_blk.m_horizonRatio = std::exp2( g( ParamIdE::HORIZON_HEIGHT ) );
    m_blk.m_grain = g( ParamIdE::GRAIN );
    m_blk.m_grainLength = g( ParamIdE::GRAIN_LENGTH );
    m_blk.m_grainDuty = g( ParamIdE::GRAIN_DENSITY );
    m_blk.m_talbot = g( ParamIdE::TALBOT );
    m_blk.m_talbotMode = std::round( g( ParamIdE::TALBOT_MODE ) );
    m_blk.m_talbotTime = g( ParamIdE::TALBOT_TIME );
    m_blk.m_premonition = g( ParamIdE::PREMONITION );
    m_blk.m_premonitionTime = g( ParamIdE::PREMONITION_TIME );

    m_bp.m_on = g( ParamIdE::BRAIN_ON ) >= 0.5f;
    m_bp.m_mode = static_cast<BrainModeE>(
        std::clamp( static_cast<int>( std::lround( g( ParamIdE::AUTO_MODE ) ) ), 0, NumBrainModes - 1 ) );
    m_bp.m_density = static_cast<int>( std::lround( g( ParamIdE::BRAIN_DENSITY ) ) );
    m_bp.m_rateSeconds = m_bp.m_mode == BrainModeE::CHORDS ? g( ParamIdE::AUTO_RATE ) : g( ParamIdE::BRAIN_RATE );
    m_bp.m_holdMin = g( ParamIdE::BRAIN_HOLD_MIN );
    m_bp.m_cascade = g( ParamIdE::BRAIN_CASCADE );
    m_bp.m_holdMax = g( ParamIdE::BRAIN_HOLD_MAX );
    m_bp.m_low = static_cast<int>( std::lround( g( ParamIdE::BRAIN_LOW ) ) );
    m_bp.m_high = static_cast<int>( std::lround( g( ParamIdE::BRAIN_HIGH ) ) );
    m_bp.m_consonance = g( ParamIdE::BRAIN_CONSONANCE );
    m_bp.m_wander = g( ParamIdE::BRAIN_WANDER );
    m_bp.m_voiceLead = g( ParamIdE::AUTO_LEAD );
    m_bp.m_chordTension = g( ParamIdE::AUTO_TENSION );
    m_bp.m_rootMove = g( ParamIdE::AUTO_ROOT_MOVE );
    m_bp.m_timbre = g( ParamIdE::BRAIN_TIMBRE );
    m_bp.m_spectrum = &m_brainSpec;

    // The second conductor: the first one's knobs, its own pace, density and register (its root
    // follows the first one's plus the interval), always in Free -- a slow background.
    m_bp2 = m_bp;
    m_bp2.m_on = m_blk.m_brain2;
    m_bp2.m_mode = BrainModeE::FREE;
    m_bp2.m_density = static_cast<int>( std::lround( g( ParamIdE::BRAIN2_DENSITY ) ) );
    m_bp2.m_rateSeconds = g( ParamIdE::BRAIN2_RATE );
    m_bp2.m_holdMin = 2.0f * m_bp.m_holdMin;
    m_bp2.m_holdMax = 2.0f * m_bp.m_holdMax;
    m_bp2.m_low = std::clamp( m_bp.m_low + m_blk.m_brain2Interval, 12, 115 );
    m_bp2.m_high = std::clamp( m_bp.m_high + m_blk.m_brain2Interval, m_bp2.m_low, 120 );

    // The foundation: a voice on the root in the octave from C2 up (from Lowest when the register
    // begins lower); the chord is drawn from a fifth above it (an octave of room at the least), so
    // nothing crowds or doubles it there.
    m_foundation = std::clamp( g( ParamIdE::BRAIN_FOUNDATION ), 0.0f, 1.0f );
    m_foundationLow = std::min( { m_bp.m_low, m_bp.m_high, 36 } );
    if ( m_foundation > 0.0f && m_bp.m_on )
    {
        const int hi = std::max( m_bp.m_low, m_bp.m_high );
        const int f = foundationNote( m_brain.root(), m_foundationLow );
        m_bp.m_low = std::max( std::min( m_bp.m_low, hi ), std::min( f + 7, hi - 12 ) );
        m_bp.m_high = hi;
    }
}

void Engine::updateFoundation()
{
    const int want = m_bp.m_on && m_foundation > 0.0f ? foundationNote( m_brain.root(), m_foundationLow ) : -1;
    if ( m_foundationNote >= 0 && want == m_foundationNote )
    {
        // Still sounding? Its voice may have been taken when every voice was busy.
        bool held = false;
        for ( const Voice &v : m_voices )
            if ( v.m_source == FoundationSource && v.m_note == m_foundationNote && !v.m_released ) held = true;
        if ( held ) return;
        for ( const Voice &v : m_voices )
            if ( v.m_stealing && v.m_nextSource == FoundationSource && v.m_next.m_note == m_foundationNote )
                held = true;
        if ( held ) return;
        m_foundationNote = -1;
    }
    if ( want == m_foundationNote ) return;
    if ( m_foundationNote >= 0 ) stopNote( m_foundationNote, FoundationSource );
    if ( want >= 0 )
        startNote( want, m_foundation, m_blk.m_depth * 0.15f, FoundationSource,
                   0 );   // close, as the conductor's near notes: the floor is under you
    m_foundationNote = want;
}

void Engine::refreshSpectrum()
{
    const float key[7] = { m_blk.m_tilt,    m_blk.m_oddEven, m_blk.m_brightness,  m_blk.m_inharmonicB,
                           m_blk.m_stretch, m_blk.m_horizon, m_blk.m_horizonRatio };
    if ( std::equal( std::begin( key ), std::end( key ), std::begin( m_specKey ) ) ) return;
    std::copy( std::begin( key ), std::end( key ), std::begin( m_specKey ) );
    // The conductors judge timbre against the voice's own harmonic spectrum, both stretches included
    // (so the Timbre scale of a stretched spectrum is the stretched scale that makes it consonant).
    m_brainSpec.m_count = BrainSpectrum::Max;
    for ( int h = 1; h <= BrainSpectrum::Max; ++h )
    {
        m_brainSpec.m_ratio[h - 1] = static_cast<double>(
            partialRatio( h, m_blk.m_inharmonicB, m_blk.m_stretch, m_blk.m_horizon, m_blk.m_horizonRatio ) );
        m_brainSpec.m_amp[h - 1] = harmonicShape( h, m_blk.m_tilt, m_blk.m_oddEven, m_blk.m_brightness );
    }
    if ( m_scaleChoice == TimbreScaleIndex ) m_scaleChoice = -1;   // the Timbre scale follows the spectrum
}

void Engine::refreshScale()
{
    const int choice = static_cast<int>( std::lround( base( ParamIdE::SCALE ) ) );
    if ( choice != m_scaleChoice )
    {
        FixedScale s;
        // The Timbre scale (Sethares) is the one a timbre asks for: here the cloud's own spectrum.
        const bool built =
            choice == TimbreScaleIndex ? makeTimbreScale( m_brainSpec, s ) : makeBuiltinScale( choice, s );
        if ( built )
            m_scale = s;
        else if ( m_scaleChoice < 0 )
            makeBuiltinScale( 0, m_scale );   // the Scala slot comes with the editor: 12-TET until then
        m_scaleChoice = choice;
    }
    m_rootPc = std::clamp( static_cast<int>( std::lround( base( ParamIdE::ROOT_NOTE ) ) ), 0, 11 );
    m_snapKeys = std::lround( base( ParamIdE::KEY_MAP ) ) == 0;
}

void Engine::setUserLandscape( const Landscape &l )
{
    if ( l.m_columns < 2 || l.m_columns > LandMaxColumns || l.m_rows < 1 ||
         l.m_amps.size() < static_cast<size_t>( l.m_columns ) * l.m_rows )
        return;
    Landscape fit = l;
    if ( fit.m_rows > LandMaxRows )
    {   // a long recording: rows averaged in groups, the loop kept
        const int          k = ( fit.m_rows + LandMaxRows - 1 ) / LandMaxRows, rows = fit.m_rows / k;
        std::vector<float> a( static_cast<size_t>( rows ) * fit.m_columns, 0.0f );
        for ( int r = 0; r < rows; ++r )
            for ( int j = 0; j < k; ++j )
                for ( int c = 0; c < fit.m_columns; ++c )
                    a[static_cast<size_t>( r ) * fit.m_columns + c] +=
                        fit.m_amps[static_cast<size_t>( r * k + j ) * fit.m_columns + c] / k;
        fit.m_amps.swap( a );
        fit.m_rowSeconds *= static_cast<float>( k );
        fit.m_rows = rows;
    }
    const float rms = landscapeRms( fit );
    {
        const std::lock_guard<std::mutex> lock( m_userLandLock );
        m_userLand = std::move( fit );
        m_userLandRms = rms;
    }
    m_userLandVersion.fetch_add( 1, std::memory_order_release );
}

void Engine::refreshLandscape()
{
    int choice =
        std::clamp( static_cast<int>( std::lround( base( ParamIdE::LANDSCAPE ) ) ), 0, NumLandscapeChoices - 1 );
    const int version = m_userLandVersion.load( std::memory_order_acquire );
    if ( choice == UserLandscape && version == 0 ) choice = 0;   // none of the player's yet: Aurora
    if ( choice == UserLandscape )
    {
        if ( m_landChoice == choice && version == m_userLandSeen ) return;
        if ( !m_userLandLock.try_lock() ) return;   // the editor is handing one over: the next block
        m_cloud.setLandscape( m_userLand.m_amps.data(), m_userLand.m_columns, m_userLand.m_rows,
                              m_userLand.m_rowSeconds );
        m_landRms = m_userLandRms;
        m_landLoop = static_cast<float>( m_userLand.m_rows ) * m_userLand.m_rowSeconds;
        m_userLandLock.unlock();
        m_landChoice = choice;
        m_userLandSeen = version;
        return;
    }
    if ( choice == m_landChoice ) return;
    const FactoryLandscapes &f = factoryLandscapes();
    const Landscape         &l = f.m_land[static_cast<size_t>( choice )];
    m_cloud.setLandscape( l.m_amps.data(), l.m_columns, l.m_rows, l.m_rowSeconds );
    m_landRms = f.m_rms[static_cast<size_t>( choice )];
    m_landLoop = static_cast<float>( l.m_rows ) * l.m_rowSeconds;
    m_landChoice = choice;
}

void Engine::updateLight()
{
    const float amount = cur( ParamIdE::BRAIN_LIGHT );
    if ( amount <= 0.0f )
    {
        m_lightOn = false;
        return;
    }
    if ( m_lightOn && --m_lightCountdown > 0 ) return;
    m_lightCountdown = 8;
    // The light over the spectrum's axis (20 Hz .. 20 kHz, log, as the partials read it), 128 points.
    constexpr int Points = 128;
    float         profile[Points];
    bool          have = false;
    const int     lifeMode =
        std::clamp( static_cast<int>( std::lround( cur( ParamIdE::LIFE_MODE ) ) ), 0, NumLifeModes - 1 );
    const float lifeDepth = cur( ParamIdE::LIFE_DEPTH );
    if ( lifeMode != 0 && lifeDepth > 0.0f )
    {
        if ( const float *cells = m_cloud.lifePicture() )
        {   // the living field: its columns, averaged along
            float column[LifeW];
            for ( int x = 0; x < LifeW; ++x )
            {
                float s = 0.0f;
                for ( int y = 0; y < LifeH; ++y ) s += cells[static_cast<size_t>( y ) * LifeW + x];
                column[x] = s / static_cast<float>( LifeH );
            }
            const float zoom = cur( ParamIdE::LIFE_ZOOM );
            for ( int i = 0; i < Points; ++i )
            {
                float x = ( static_cast<float>( i ) + 0.5f ) / Points * zoom * static_cast<float>( LifeW );
                x -= std::floor( x / static_cast<float>( LifeW ) ) * static_cast<float>( LifeW );
                profile[i] = 1.0f - lifeDepth + lifeDepth * column[std::min( static_cast<int>( x ), LifeW - 1 )];
            }
            have = true;
        }
    }
    const float motionDepth = cur( ParamIdE::MOTION_DEPTH );
    if ( !have && motionUsesLandscape( static_cast<MotionKindE>( std::lround( cur( ParamIdE::MOTION ) ) ) ) &&
         motionDepth > 0.0f && m_landChoice >= 0 )
    {
        // The landscape as a new note reads it: its first eight seconds at the motion's rate.
        auto readLand = [&]( const Landscape &l )
        {
            if ( l.m_columns < 2 || l.m_rows < 1 ) return;
            const int rows = std::clamp( static_cast<int>( std::lround( 8.0f * cur( ParamIdE::MOTION_RATE ) /
                                                                        std::max( l.m_rowSeconds, 0.001f ) ) ),
                                         1, l.m_rows );
            for ( int i = 0; i < Points; ++i )
            {
                const int c = std::min(
                    static_cast<int>( ( static_cast<float>( i ) + 0.5f ) / Points * static_cast<float>( l.m_columns ) ),
                    l.m_columns - 1 );
                float s = 0.0f;
                for ( int r = 0; r < rows; ++r ) s += l.m_amps[static_cast<size_t>( r ) * l.m_columns + c];
                profile[i] = 1.0f - motionDepth + motionDepth * s / static_cast<float>( rows );
            }
            have = true;
        };
        if ( m_landChoice == UserLandscape )
        {
            if ( m_userLandLock.try_lock() )
            {   // the editor handing one over: the light as it was
                readLand( m_userLand );
                m_userLandLock.unlock();
            }
            else if ( m_lightOn )
            {
                return;
            }
        }
        else
        {
            readLand( factoryLandscapes().m_land[static_cast<size_t>( m_landChoice )] );
        }
    }
    if ( !have )
    {
        m_lightOn = false;
        return;
    }
    // Per note: its harmonics' weight in the light (the voice's own spectrum), against the register's mean.
    float  score[128];
    double mean = 0.0;
    int    counted = 0;
    for ( int n = 0; n < 128; ++n )
    {
        const double f0 = frequencyOf( n );
        double       s = 0.0, w = 0.0;
        for ( int h = 0; h < std::max( m_brainSpec.m_count, 1 ) && h < BrainSpectrum::Max; ++h )
        {
            const double fh = f0 * ( m_brainSpec.m_count > 0 ? m_brainSpec.m_ratio[h] : static_cast<double>( h + 1 ) );
            if ( fh < LandLowHz || fh > 20000.0 ) continue;
            const double a = m_brainSpec.m_count > 0 ? m_brainSpec.m_amp[h] : 1.0 / ( h + 1 );
            const int    i =
                std::clamp( static_cast<int>( std::log2( fh / LandLowHz ) / 9.96578428 * Points ), 0, Points - 1 );
            s += a * profile[i];
            w += a;
        }
        score[n] = w > 0.0 ? static_cast<float>( s / w ) : 0.0f;
        if ( n >= m_bp.m_low && n <= m_bp.m_high )
        {
            mean += score[n];
            ++counted;
        }
    }
    mean = counted > 0 ? mean / counted : 0.5;
    // The factor: the light's share against the register's, cubed at full Follow Light -- twice the
    // light, eight times as likely.
    for ( int n = 0; n < 128; ++n )
        m_light[n] = std::clamp(
            static_cast<float>( std::pow( std::max( score[n], 0.01f ) / std::max( mean, 0.01 ), 3.0 * amount ) ), 0.02f,
            50.0f );
    m_lightOn = true;
}

double Engine::frequencyOf( int note ) const
{
    // Purity blends 12-TET (0) with the scale (1) in the log domain, as in Noctuary.
    const double ref = m_blk.m_refPitch;
    const double pure = scaleFrequency( m_scale, note, 60 + m_rootPc, ref, m_snapKeys );
    const double et = ref * std::pow( 2.0, ( note - 69 ) / 12.0 );
    const double p = std::clamp( static_cast<double>( m_blk.m_purity ), 0.0, 1.0 );
    return std::exp( std::log( et ) + ( std::log( pure ) - std::log( et ) ) * p );
}

float Engine::brainDistance()
{
    // Noctuary's spatial model: the conductor's notes are bimodal, 40 % close, 60 % deep.
    const float u = m_rng.uniform();
    return m_rng.uniform() < 0.4f ? m_blk.m_depth * 0.15f * u : m_blk.m_depth * ( 0.55f + 0.45f * u );
}

void Engine::beginOn( int i, int note, float velocity, float distance, int source, int onset, int channel )
{
    Voice &v = m_voices[static_cast<size_t>( i )];
    m_lastVelocity = velocity;
    v.m_note = note;
    v.m_source = source;
    v.m_channel = source == 0 ? channel : -1;
    v.m_bend =
        v.m_channel >= 0 ? m_chanBend[static_cast<size_t>( v.m_channel )].load( std::memory_order_relaxed ) : 0.0f;
    v.m_press =
        v.m_channel >= 0 ? m_chanPress[static_cast<size_t>( v.m_channel )].load( std::memory_order_relaxed ) : 0.0f;
    v.m_slide =
        v.m_channel >= 0 ? m_chanSlide[static_cast<size_t>( v.m_channel )].load( std::memory_order_relaxed ) : 0.5f;
    v.m_velocity = velocity;
    v.m_place = std::clamp( distance, 0.0f, 1.0f );
    v.m_distance = v.m_place;
    v.m_prevDistance = v.m_place;
    v.m_breath.init( m_side );
    v.m_cutoffDrift.init( m_side );
    v.m_env.kill();
    // Premonition (Phase 27): a conductor's note is heard coming -- its swell first, its envelope after.
    v.m_pre = source != 0 && m_blk.m_premonition > 0.0f ? std::max( m_blk.m_premonitionTime, 0.5f ) : 0.0f;
    v.m_preTotal = v.m_pre;
    v.m_suck = 0.0f;
    if ( v.m_pre <= 0.0f ) v.m_env.noteOn();
    v.m_seed = static_cast<std::uint32_t>( m_rng.next() ) | 1u;
    v.m_restart = true;
    v.m_onset = onset;
    v.m_released = false;
    v.m_tailBlocks = 1;
    v.m_tailArmed = false;
    v.m_tuneCents = 0.0f;
    v.m_lastGain = 0.0f;
    m_randomNote = m_noteRng.bipolar();
    v.m_age = 0.0;
    v.m_stealing = false;
    v.m_stealBlocks = 0;
    // Phase 26: the harmonics' lives begin at rest; the subharmonics' clouds are kept for the note if it
    // starts while they are wanted (the partials are shared out once, so they never jump).
    v.m_share = { 1.0f, 0.0f, 0.0f, 0.0f };
    v.m_blowWander.init( m_blowRng );
    v.m_pileLive = false;
    v.m_pileDepth = 0.0f;
    v.m_glow.fill( 0.0f );
    v.m_glowTarget.fill( 0.0f );
    v.m_subOn = m_blk.m_subharmonics > 0.0f;
    v.m_sub.fill( 0.0f );
    m_cloud.voice( i ).m_releaseAt = -1.0f;
}

void Engine::startNote( int note, float velocity, float distance, int source, int onset, int channel )
{
    const int count = static_cast<int>( m_voices.size() );
    for ( int i = 0; i < count; ++i )
        if ( m_voices[static_cast<size_t>( i )].m_note < 0 && !m_voices[static_cast<size_t>( i )].m_stealing )
        {
            beginOn( i, note, velocity, distance, source, onset, channel );
            return;
        }
    // Every voice sounds: take the quietest released one, else the oldest. It fades to silence
    // over one block (a restart from full level would click) and the new note begins after it.
    int   best = -1;
    float bestLevel = 2.0f;
    for ( int i = 0; i < count; ++i )
    {
        const Voice &v = m_voices[static_cast<size_t>( i )];
        if ( !v.m_stealing && v.m_released && v.m_env.level() < bestLevel )
        {
            bestLevel = v.m_env.level();
            best = i;
        }
    }
    if ( best < 0 )
    {   // the oldest -- not the foundation, which is the oldest by design
        double oldest = -1.0;
        for ( int i = 0; i < count; ++i )
        {
            const Voice &v = m_voices[static_cast<size_t>( i )];
            if ( !v.m_stealing && v.m_source != FoundationSource && v.m_age > oldest )
            {
                oldest = v.m_age;
                best = i;
            }
        }
    }
    if ( best < 0 )
    {
        double oldest = -1.0;
        for ( int i = 0; i < count; ++i )
        {
            const Voice &v = m_voices[static_cast<size_t>( i )];
            if ( !v.m_stealing && v.m_age > oldest )
            {
                oldest = v.m_age;
                best = i;
            }
        }
    }
    if ( best < 0 ) return;
    Voice &v = m_voices[static_cast<size_t>( best )];
    v.m_stealing = true;
    v.m_stealBlocks = 1;
    v.m_next = { 0, note, velocity, 0, channel };
    v.m_nextSource = source;
    v.m_nextDistance = distance;
}

void Engine::stopNote( int note, int source, int channel )
{
    for ( size_t i = 0; i < m_voices.size(); ++i )
    {
        Voice     &v = m_voices[i];
        const bool ours = channel < 0 || v.m_channel < 0 || v.m_channel == channel;
        if ( v.m_stealing && v.m_next.m_type == 0 && v.m_next.m_note == note && v.m_nextSource == source &&
             ( channel < 0 || v.m_next.m_channel == channel ) )
        {
            v.m_next.m_type = -1;
            return;
        }
        if ( v.m_note == note && v.m_source == source && ours && !v.m_released && !v.m_stealing )
        {
            v.m_released = true;
            if ( v.m_pre > 0.0f )
            {   // let go before it sounded: its swell is drawn away
                v.m_pre = 0.0f;
                v.m_suck = 1.0f;
            }
            v.m_env.noteOff();
            CloudVoice &cv = m_cloud.voice( static_cast<int>( i ) );
            cv.m_releaseAt = cv.m_time;   // the motion's release tails start here
            if ( cur( ParamIdE::MEMORY_LEVEL ) > 0.0f && !m_ghosts.empty() ) captureGhost( static_cast<int>( i ) );
            return;
        }
    }
}

void Engine::lifeSnapshot( std::vector<float> &cells )
{
    const std::lock_guard<std::mutex> lock( m_snapLock );
    cells = m_snapLife;
}

void Engine::setMacroLesson( int macro, const MacroLesson &lesson )
{
    if ( macro < 0 || macro >= 4 ) return;
    const std::lock_guard<std::mutex> lock( m_lessonLock );
    m_lessonsPending[static_cast<size_t>( macro )] = lesson;
    m_lessonPendingSet[static_cast<size_t>( macro )] = true;
}

void Engine::rippleSnapshot( std::vector<float> &cells )
{
    const std::lock_guard<std::mutex> lock( m_snapLock );
    cells = m_snapRipple;
}

bool Engine::partialSnapshot( std::vector<float> &dots, int &perVoice )
{
    const std::lock_guard<std::mutex> lock( m_snapLock );
    if ( m_snapSeq == m_snapSeen || m_snap.empty() ) return false;
    m_snapSeen = m_snapSeq;
    dots = m_snap;
    perVoice = m_snapPer;
    return true;
}

bool Engine::hallMemory( std::vector<float> &out, double &age )
{
    const std::lock_guard<std::mutex> lock( m_snapLock );
    out.resize( static_cast<size_t>( HallBins ) );
    if ( m_hallPendingSet )
    {
        std::copy( m_hallPending.begin(), m_hallPending.end(), out.begin() );
        age = m_hallPendingAge;
    }
    else if ( m_snapHallValid )
    {
        std::copy( m_snapHall.begin(), m_snapHall.end(), out.begin() );
        age = m_snapHallAge;
    }
    else
    {
        std::fill( out.begin(), out.end(), 1.0f );
        age = 0.0;
    }
    return age > 0.0;
}

void Engine::setHallMemory( const std::vector<float> &in, double age )
{
    const std::lock_guard<std::mutex> lock( m_snapLock );
    const bool                        whole = in.size() == static_cast<size_t>( HallBins );
    for ( int b = 0; b < HallBins; ++b )
    {
        const float m = whole ? in[static_cast<size_t>( b )] : 1.0f;
        m_hallPending[static_cast<size_t>( b )] = std::isfinite( m ) ? std::clamp( m, 0.0f, 1024.0f ) : 1.0f;
    }
    m_hallPendingAge = whole ? std::max( 0.0, age ) : 0.0;
    m_hallPendingSet = true;
}

void Engine::stepRipple( float dt )
{
    Ripple &R = m_cloud.ripple();
    R.m_depth = cur( ParamIdE::RIPPLE_DEPTH );
    // Breathers (Phase 26): the string as a sine-Gordon chain, u_tt = (c^2 u_x)_x - g u_t - m^2 sin u, its
    // mass up to 1 Hz. A strike shaped like a breather (a bell about as wide as c / m) stays where it fell
    // and pulses there below the mass's frequency -- it cannot run away as a wave, there is no wave that
    // slow -- instead of parting into two halves; the string is read as the angle it is (2 sin(u/2)), so
    // a kink that turned it a whole round leaves nothing behind.
    const float breathers = std::clamp( cur( ParamIdE::RIPPLE_BREATHERS ), 0.0f, 1.0f );
    const float bell = std::min( 1.0f, 2.0f * breathers );
    R.m_periodic = bell;
    const int strikes = m_rippleStrikeCount;
    m_rippleStrikeCount = 0;
    const int sounding = m_rippleVoices;
    m_rippleVoices = 0;
    if ( R.m_depth <= 0.0f )
    {   // at rest: the string is let go, so it starts quiet when it is turned up
        std::fill( std::begin( m_rippleNow ), std::end( m_rippleNow ), 0.0f );
        std::fill( std::begin( m_ripplePrev ), std::end( m_ripplePrev ), 0.0f );
        std::fill( std::begin( R.m_u ), std::end( R.m_u ), 0.0f );
        R.m_makeup = 1.0f;
        return;
    }
    constexpr float CellsPerOctave = static_cast<float>( RippleCells - 1 ) / 9.96578428f;
    // Strikes: every note that started this block, at its first eight harmonics, the stronger the
    // lower the harmonic -- a pebble into the spectrum where the note lands. The shape is a Ricker
    // wavelet (a "Mexican hat", zero in sum) about a quarter octave wide, so a passing wave makes a
    // partial louder and then softer rather than louder on the whole; and it goes into both the now
    // and the step before, a displacement at rest, which parts into two halves running up and down
    // (a struck velocity would leave a standing plateau behind in one dimension: measured, 3 dB
    // louder where the notes are and no motion).
    // The damped wave, u_tt = (c^2 u_x)_x - g u_t, by leapfrog; the ends reflect.
    const float c = std::max( cur( ParamIdE::RIPPLE_SPEED ), 0.01f ) * CellsPerOctave;   // cells a second
    const float g = 2.0f / std::max( cur( ParamIdE::RIPPLE_DECAY ), 0.1f );
    const int   steps = std::max( 1, static_cast<int>( std::ceil( c * dt / 0.9f ) ) );
    const float h = dt / static_cast<float>( steps ), r2 = ( c * h ) * ( c * h ), damp = g * h * 0.5f;
    const float mass2 = breathers * 39.478418f;   // (2 pi 1 Hz)^2 at full
    for ( int s = 0; s < strikes; ++s )
    {
        for ( int k = 1; k <= 8; ++k )
        {
            const float pos = std::log2( std::max( m_rippleStrikes[s] * k, 20.0f ) / 20.0f ) * CellsPerOctave;
            // With breathers the first four harmonics are struck as bells, each drifting a little up or
            // down (a moving breather: the step before is the bell a step behind).
            const float drift = bell > 0.0f && k <= 4 ? c * ( 0.05f + 0.1f * m_breatherRng.uniform() ) *
                                                            ( m_breatherRng.uniform() < 0.5f ? -1.0f : 1.0f )
                                                      : 0.0f;
            for ( int cc = std::max( 0, static_cast<int>( pos ) - 24 );
                  cc <= std::min( RippleCells - 1, static_cast<int>( pos ) + 24 ); ++cc )
            {
                const float d = ( static_cast<float>( cc ) - pos ) / 5.0f;
                // 6/h: a passing half-wave of the fundamental's strike takes the loudness from near silence
                // to twice (tanh saturates); at 2/h it moved a harmonic by a dB or two, too tame to hear.
                const float hat = ( 6.0f / static_cast<float>( k ) ) * ( 1.0f - d * d ) * std::exp( -0.5f * d * d );
                const float db = ( static_cast<float>( cc ) - pos + drift * h ) / 5.0f;
                const float bellNow = k <= 4 ? 4.0f * std::exp( -0.5f * d * d ) : 0.0f;
                const float bellPrev = k <= 4 ? 4.0f * std::exp( -0.5f * db * db ) : 0.0f;
                m_rippleNow[cc] += ( 1.0f - bell ) * hat + bell * bellNow;
                m_ripplePrev[cc] += ( 1.0f - bell ) * hat + bell * bellPrev;
            }
        }
    }
    // Bend: the wave slower where the spectrum is loud -- the sounding voices' first sixteen harmonics,
    // their amplitudes against the spectrum's mean, each over a sixth of an octave (as wide as a strike:
    // a narrower slow zone lets the wave leak out at once) -- up to a ninth of its speed, so a wave
    // lingers at the notes, gathers there and is partly thrown back, as light in a lens -- a fifth of its
    // speed where a lone note's fundamental is, up to a seventeenth. (A slower cell is as stable.)
    float r2c[RippleCells];
    std::fill( std::begin( r2c ), std::end( r2c ), r2 );
    const float bend = cur( ParamIdE::RIPPLE_BEND );
    if ( bend > 0.0f && sounding > 0 )
    {
        float power[RippleCells] = {};
        float shape[16];
        for ( int k = 1; k <= 16; ++k )
            shape[k - 1] = harmonicShape( k, m_blk.m_tilt, m_blk.m_oddEven, m_blk.m_brightness );
        double total = 0.0;
        for ( int v = 0; v < sounding; ++v )
            for ( int k = 1; k <= 16; ++k )
            {
                const float fk = m_rippleHz[v] * partialRatio( k, m_blk.m_inharmonicB, m_blk.m_stretch, m_blk.m_horizon,
                                                               m_blk.m_horizonRatio ) +
                                 m_blk.m_shift;
                if ( fk < 20.0f || fk > 20000.0f ) continue;
                const float w = m_rippleAmp[v] * shape[k - 1];
                const float pos = std::log2( fk / 20.0f ) * CellsPerOctave;
                for ( int cc = std::max( 0, static_cast<int>( pos ) - 12 );
                      cc <= std::min( RippleCells - 1, static_cast<int>( pos ) + 13 ); ++cc )
                {
                    const float d = ( static_cast<float>( cc ) - pos ) / 4.0f;
                    power[cc] += w * std::exp( -0.5f * d * d );
                }
                total += w;
            }
        if ( total > 1.0e-20 )
        {
            const float mean = static_cast<float>( total * 10.0265 / RippleCells );   // a bump's sum, 4 sqrt(2 pi)
            for ( int i = 0; i < RippleCells; ++i )
            {
                const float dense = std::clamp( std::log2( 1.0f + power[i] / mean ) - 1.0f, 0.0f, 4.0f );
                const float ci = 1.0f / ( 1.0f + 4.0f * bend * dense );
                r2c[i] = r2 * ci * ci;
            }
        }
    }
    // In the conservative form, u_tt = (c^2 u_x)_x: a wave entering a slow zone piles up there, as a
    // wave on a shelving beach, and leaves it no louder than it came (in c^2 u_xx it would leave louder).
    float next[RippleCells];
    for ( int s = 0; s < steps; ++s )
    {
        for ( int i = 0; i < RippleCells; ++i )
        {
            const int   il = i > 0 ? i - 1 : 1, ir = i < RippleCells - 1 ? i + 1 : RippleCells - 2;
            const float kl = 0.5f * ( r2c[i] + r2c[il] ), kr = 0.5f * ( r2c[i] + r2c[ir] );
            const float flow = kr * ( m_rippleNow[ir] - m_rippleNow[i] ) - kl * ( m_rippleNow[i] - m_rippleNow[il] );
            const float pull = mass2 > 0.0f ? h * h * mass2 * std::sin( m_rippleNow[i] ) : 0.0f;
            next[i] = ( 2.0f * m_rippleNow[i] - ( 1.0f - damp ) * m_ripplePrev[i] + flow - pull ) / ( 1.0f + damp );
        }
        std::copy( std::begin( m_rippleNow ), std::end( m_rippleNow ), std::begin( m_ripplePrev ) );
        std::copy( std::begin( next ), std::end( next ), std::begin( m_rippleNow ) );
    }
    double mean2 = 0.0;
    for ( int i = 0; i < RippleCells; ++i )
    {
        if ( !std::isfinite( m_rippleNow[i] ) ) m_rippleNow[i] = 0.0f;
        R.m_u[i] = m_rippleNow[i];
        const float  u = m_rippleNow[i] + R.m_periodic * ( 2.0f * std::sin( 0.5f * m_rippleNow[i] ) - m_rippleNow[i] );
        const double t = std::tanh( static_cast<double>( u ) );
        mean2 += t * t;
    }
    R.m_makeup = static_cast<float>(
        1.0 / std::sqrt( 1.0 + static_cast<double>( R.m_depth ) * R.m_depth * mean2 / RippleCells ) );
}

void Engine::stepSeasons( float dt )
{
    const float want = std::clamp( m_blk.m_seasons, 0.0f, 1.0f );
    m_seasonDepth += ( want - m_seasonDepth ) * ( 1.0f - std::exp( -dt / 0.5f ) );
    if ( want <= 0.0f && m_seasonDepth < 1.0e-4f )
    {   // at rest: the families keep their shares, the cycle goes on from there when it is turned up
        m_seasonDepth = 0.0f;
        return;
    }
    // A family wins 1 against each of the next two and loses 2 against each of the two before; the
    // mutation, a hundredth, keeps every family alive (none falls below a few thousandths), so a round
    // takes 12.9 of the dynamics' time units (measured, Python): it is scaled to Year.
    constexpr float Win = 1.0f, Loss = 2.0f, Mutation = 0.01f, Round = 12.9f;
    const float     rate = Round / std::max( m_blk.m_seasonTime, 1.0f );
    float          *x = m_seasonShare.data();
    float           f[SeasonFamilies];
    float           mean = 0.0f;
    for ( int i = 0; i < SeasonFamilies; ++i )
    {
        f[i] =
            Win * ( x[( i + 1 ) % SeasonFamilies] + x[( i + 2 ) % SeasonFamilies] ) -
            Loss * ( x[( i + SeasonFamilies - 1 ) % SeasonFamilies] + x[( i + SeasonFamilies - 2 ) % SeasonFamilies] );
        mean += x[i] * f[i];
    }
    float sum = 0.0f;
    for ( int i = 0; i < SeasonFamilies; ++i )
    {
        x[i] = std::max( 0.0f, x[i] + dt * rate * ( x[i] * ( f[i] - mean ) + Mutation * ( 0.2f - x[i] ) ) );
        sum += x[i];
    }
    for ( int i = 0; i < SeasonFamilies; ++i ) x[i] = sum > 0.0f ? x[i] / sum : 0.2f;
}

void Engine::stepHarmonics( Voice &v, CloudVoice &cv, float dt, float gain, float breath )
{
    int life = 0;

    // Overblow: the registers (the note's 1st, 2nd, 3rd, 4th harmonic as its fundamental) as shares, in
    // replicator dynamics -- each prefers a breath (0, 0.45, 0.72, 0.95), the one sounding holds by its
    // share (so a register jumps up later than it falls back: hysteresis), a little mutation lets a
    // register return. The breath: the knob, by the note's velocity and MPE's Breath, with a slow
    // unrest of its own, so a note near a threshold flickers between two registers.
    if ( m_blk.m_overblow > 0.0f || v.m_share[0] < 0.999f )
    {
        constexpr float Best[4] = { 0.0f, 0.45f, 0.72f, 0.95f };
        constexpr float Width = 0.5f, Hold = 0.12f;
        const bool      on = m_blk.m_overblow > 0.0f;
        const float     P = on ? m_blk.m_overblow * ( 0.55f + 0.45f * v.m_velocity + 0.5f * breath ) +
                                     0.04f * m_blk.m_overblow * v.m_blowWander.update( dt, 0.15f, m_blowRng )
                               : 0.0f;
        const float     eps = on ? 0.0005f : 0.0f;
        const float     tau = std::max( 0.1f * m_blk.m_overblowTime, 0.01f );
        const int       sub = std::max( 1, static_cast<int>( std::ceil( dt / ( 0.2f * tau ) ) ) );
        const float     h = dt / static_cast<float>( sub );
        float          *s = v.m_share.data();
        for ( int step = 0; step < sub; ++step )
        {
            float F[4], mean = 0.0f;
            for ( int k = 0; k < 4; ++k )
            {
                F[k] = -std::fabs( P - Best[k] ) / Width + Hold * s[k];
                mean += s[k] * F[k];
            }
            float sum = 0.0f;
            for ( int k = 0; k < 4; ++k )
            {
                s[k] = std::max( 0.0f, s[k] + h * ( s[k] * ( F[k] - mean ) + eps * ( 0.25f - s[k] ) ) / tau );
                sum += s[k];
            }
            for ( int k = 0; k < 4; ++k ) s[k] = sum > 0.0f ? s[k] / sum : ( k == 0 ? 1.0f : 0.0f );
        }
        if ( !on && s[0] >= 0.999f ) v.m_share = { 1.0f, 0.0f, 0.0f, 0.0f };
        for ( int k = 0; k < 4; ++k ) cv.m_register[k] = s[k];
        life |= 1;
    }

    // Seasons: the family whose turn it is louder, the others softer (stepSeasons).
    if ( m_seasonDepth > 0.0f )
    {
        for ( int i = 0; i < SeasonFamilies; ++i )
            cv.m_season[i] = 1.0f - m_seasonDepth +
                             m_seasonDepth * std::sqrt( static_cast<float>( SeasonFamilies ) * m_seasonShare[i] );
        life |= 2;
    }

    // Avalanche: the Oslo rice pile (Christensen et al. 1996) on harmonics 2 .. 49 -- the slope at each
    // harmonic, a threshold of 1 or 2 drawn anew after every toppling, a grain falling on harmonic 2 now
    // and then. A site over its threshold topples: it gives a grain to either neighbour (the first only
    // upwards, the last lets one go). One generation of topplings every 35 ms, so an avalanche runs up
    // and down the harmonics where it can be heard; every toppling lights its harmonic. Even in one
    // dimension this pile is critical: avalanches of every size, mostly small, rarely through the whole.
    constexpr float PileStep = 0.035f;
    const bool      piling = m_blk.m_avalanche > 0.0f;
    if ( piling && !v.m_pileLive )
    {   // laid out near its critical slope at once (a pile that had to grow first would take minutes)
        v.m_pileDice.seed( 0xA5A5A5A5ull ^ v.m_seed );
        for ( int i = 0; i < AvalancheSites; ++i )
        {
            const std::uint8_t th = v.m_pileDice.uniform() < 0.5f ? 1 : 2;
            v.m_threshold[static_cast<size_t>( i )] = th;
            v.m_slope[static_cast<size_t>( i )] =
                static_cast<std::uint8_t>( v.m_pileDice.uniform() < 0.73f ? th : th - 1 );
        }
        v.m_grainDue =
            -std::log( std::max( 1.0f - v.m_pileDice.uniform(), 1.0e-6f ) ) / std::max( m_blk.m_avalancheRate, 0.01f );
        v.m_pileClock = 0.0f;
        v.m_pileLive = true;
    }
    if ( v.m_pileLive )
    {
        v.m_pileDepth += ( m_blk.m_avalanche - v.m_pileDepth ) * ( 1.0f - std::exp( -dt / 0.1f ) );
        if ( piling )
        {
            // Grains fall while the note sounds, as much as it sounds.
            v.m_grainDue -= dt * std::min( 1.0f, 2.0f * gain );
            while ( v.m_grainDue <= 0.0f )
            {
                ++v.m_slope[0];
                v.m_grainDue += -std::log( std::max( 1.0f - v.m_pileDice.uniform(), 1.0e-6f ) ) /
                                std::max( m_blk.m_avalancheRate, 0.01f );
            }
            v.m_pileClock += dt;
            while ( v.m_pileClock >= PileStep )
            {
                v.m_pileClock -= PileStep;
                std::array<std::uint8_t, AvalancheSites> over{};
                int                                      count = 0;
                for ( int i = 0; i < AvalancheSites; ++i )
                    if ( v.m_slope[static_cast<size_t>( i )] > v.m_threshold[static_cast<size_t>( i )] )
                    {
                        over[static_cast<size_t>( i )] = 1;
                        ++count;
                    }
                if ( count == 0 ) break;
                for ( int i = 0; i < AvalancheSites; ++i )
                {
                    if ( !over[static_cast<size_t>( i )] ) continue;
                    const size_t at = static_cast<size_t>( i );
                    if ( i == AvalancheSites - 1 )
                    {   // the top lets the grain go
                        v.m_slope[at] = static_cast<std::uint8_t>( v.m_slope[at] - 1 );
                        ++v.m_slope[at - 1];
                    }
                    else
                    {
                        v.m_slope[at] = static_cast<std::uint8_t>( v.m_slope[at] - 2 );
                        ++v.m_slope[at + 1];
                        if ( i > 0 ) ++v.m_slope[at - 1];
                    }
                    v.m_threshold[at] = v.m_pileDice.uniform() < 0.5f ? 1 : 2;
                    v.m_glowTarget[at] = 1.0f;
                }
            }
        }
        // The glow: lit at once, fading over Glow; heard softened over 40 ms (no click).
        const float fade = std::exp( -dt / std::max( m_blk.m_avalancheGlow, 0.05f ) );
        const float soft = 1.0f - std::exp( -dt / 0.04f );
        float       most = 0.0f;
        for ( int i = 0; i < AvalancheSites; ++i )
        {
            const size_t at = static_cast<size_t>( i );
            v.m_glowTarget[at] *= fade;
            v.m_glow[at] += ( v.m_glowTarget[at] - v.m_glow[at] ) * soft;
            most = std::max( most, v.m_glow[at] );
            // up to three times the harmonic's amplitude (+9.5 dB) at full Avalanche
            cv.m_glow[i] = 2.0f * v.m_pileDepth * v.m_glow[at];
        }
        if ( !piling && ( most < 1.0e-4f || v.m_pileDepth < 1.0e-4f ) )
            v.m_pileLive = false;
        else
            life |= 4;
    }

    // Talbot (Phase 27): the first TalbotHarmonics amplitudes as a wave function, evolved in the eigen system
    // (talbotSystem) from the note's start, mixed with the timbre at rest by power (Depth).
    if ( m_blk.m_talbot > 0.0f )
    {
        talbotSystem();
        constexpr int M = TalbotHarmonics;
        float         s[M], c[M];
        for ( int h = 0; h < M; ++h ) s[h] = harmonicShape( h + 1, cv.m_tilt, cv.m_oddEven, cv.m_brightness );
        for ( int n = 0; n < M; ++n )
        {
            float acc = 0.0f;
            for ( int h = 0; h < M; ++h ) acc += m_talbotVec[static_cast<size_t>( n * M + h )] * s[h];
            c[n] = acc;
        }
        float        re[M] = {}, im[M] = {};
        const double age = m_blk.m_talbotMode < 0.5f ? talbotClock( v.m_age, std::max( m_blk.m_talbotTime, 1.0f ) )
                                                     : static_cast<double>( v.m_age );
        for ( int n = 0; n < M; ++n )
        {
            const double ph = std::fmod( m_talbotVal[static_cast<size_t>( n )] * age, 6.283185307179586 );
            const float  cs = static_cast<float>( std::cos( ph ) ) * c[n],
                         sn = static_cast<float>( std::sin( ph ) ) * c[n];
            for ( int h = 0; h < M; ++h )
            {
                re[h] += cs * m_talbotVec[static_cast<size_t>( n * M + h )];
                im[h] -= sn * m_talbotVec[static_cast<size_t>( n * M + h )];
            }
        }
        const float d = std::clamp( m_blk.m_talbot, 0.0f, 1.0f );
        for ( int h = 0; h < M; ++h )
        {
            const float amp = std::sqrt( ( 1.0f - d ) * s[h] * s[h] + d * ( re[h] * re[h] + im[h] * im[h] ) );
            cv.m_talbot[h] = s[h] > 1.0e-12f ? std::min( amp / s[h], 1.0e4f ) : 1.0f;
        }
        life |= 8;
    }

    // The make-up: the voice's power over its first harmonics kept as it was without these lives (the
    // harmonics above are too soft to matter), so the registers, the seasons and the glints move the
    // power between harmonics and do not make the voice louder or softer.
    const float hz = std::max( cv.m_hz, 1.0f );
    const int   Hm = std::clamp( static_cast<int>( 0.9f * 20000.0f / hz ), 1, 256 );
    float       shape[8] = {};   // harmonics 1 .. 7 at rest, for the pumps below
    float       makeup = 1.0f;
    if ( life != 0 || v.m_subOn )
    {
        double before = 0.0, after = 0.0;
        for ( int k = 1; k <= Hm; ++k )
        {
            const float a = harmonicShape( k, cv.m_tilt, cv.m_oddEven, cv.m_brightness );
            float       f = 1.0f;
            if ( life & 1 ) f *= registerGain( k, cv.m_register );
            if ( ( life & 2 ) && k > 1 ) f *= cv.m_season[seasonFamily( k )];
            if ( ( life & 4 ) && k >= 2 && k < 2 + AvalancheSites ) f *= 1.0f + cv.m_glow[k - 2];
            if ( ( life & 8 ) && k <= TalbotHarmonics ) f *= cv.m_talbot[k - 1];
            before += static_cast<double>( a ) * a;
            after += static_cast<double>( a * f ) * ( a * f );
            if ( k <= 7 ) shape[k] = a * f;
        }
        makeup = after > 1.0e-30 ? static_cast<float>( std::sqrt( before / after ) ) : 1.0f;
        for ( int k = 1; k <= 7; ++k ) shape[k] *= makeup;
    }
    cv.m_harmonicLife = life;
    cv.m_lifeMakeup = makeup;

    // Subharmonics: parametric down-conversion, as in a Faraday experiment or a parametric amplifier.
    // Sub cloud m/2 (m = 1, 3, 5, 7) is pumped by harmonic m, sub m/4 by sub m/2: above a threshold the
    // pump makes it grow from a seed, the more the further above (slowly near it: it seems to come from
    // nothing), saturating; the power it gets the pump gives up. The pump's strength: the knob, the
    // note's strength and envelope, MPE's Breath. Below 20 .. 40 Hz a sub cloud is not pumped at all
    // (nothing would be heard of the power taken from the pump).
    cv.m_subClouds = v.m_subOn ? 1 : 0;
    if ( v.m_subOn )
    {
        constexpr float Threshold = 0.25f, Saturation = 1.5f, Seed = 1.0e-3f;
        const float     strength = 0.4f + 0.6f * v.m_velocity;
        const float     D = m_blk.m_subharmonics * 1.3f * strength * strength * gain * ( 1.0f + breath );
        const float     tau = std::max( m_blk.m_subTime, 0.05f );
        auto            grow = [&]( float a, float pump, float subHz )
        {
            const float heard = std::clamp( ( subHz - 20.0f ) / 20.0f, 0.0f, 1.0f );
            if ( pump <= 1.0e-9f ) return 0.0f;
            const float x = a / pump;
            const float r = ( D * heard * pump - Threshold ) - Saturation * x * x;
            return std::clamp( a + dt * a * r / tau, Seed * pump, 0.999f * pump );
        };
        float sub[SubClouds];
        for ( int j = 0; j < 4; ++j )
        {
            const int   m = 2 * j + 1;
            const float pump =
                shape[m] / std::max( harmonicShape( 1, cv.m_tilt, cv.m_oddEven, cv.m_brightness ), 1.0e-9f );
            v.m_sub[static_cast<size_t>( j )] = grow( v.m_sub[static_cast<size_t>( j )], pump, hz * subRatio( j ) );
            v.m_sub[static_cast<size_t>( j + 4 )] = grow( v.m_sub[static_cast<size_t>( j + 4 )],
                                                          v.m_sub[static_cast<size_t>( j )], hz * subRatio( j + 4 ) );
            const float a1 = v.m_sub[static_cast<size_t>( j )], a2 = v.m_sub[static_cast<size_t>( j + 4 )];
            sub[j] = std::sqrt( std::max( a1 * a1 - a2 * a2, 0.0f ) );   // what it kept of what it got
            sub[j + 4] = a2;
            cv.m_pumpKeep[j] =
                pump > 1.0e-9f ? std::sqrt( std::max( 1.0f - ( a1 * a1 ) / ( pump * pump ), 0.0f ) ) : 1.0f;
        }
        for ( int j = 0; j < SubClouds; ++j ) cv.m_sub[j] = sub[j];
    }
}

void Engine::talbotSystem()
{
    const float mode = m_blk.m_talbotMode, period = std::max( m_blk.m_talbotTime, 1.0f );
    if ( mode == m_talbotKey[0] && period == m_talbotKey[1] ) return;
    m_talbotKey[0] = mode;
    m_talbotKey[1] = period;
    constexpr int M = TalbotHarmonics;
    const double  w = 6.283185307179586 / period;
    if ( mode < 0.5f )
    {   // Revival: the cosine basis, mode n at n^2 w -- every phase whole again after the period
        for ( int n = 0; n < M; ++n )
        {
            const double norm = std::sqrt( ( n == 0 ? 1.0 : 2.0 ) / M );
            for ( int h = 0; h < M; ++h )
                m_talbotVec[static_cast<size_t>( n * M + h )] =
                    static_cast<float>( norm * std::cos( 3.141592653589793 * n * ( h + 0.5 ) / M ) );
            m_talbotVal[static_cast<size_t>( n )] = w * n * n;
        }
        return;
    }
    // Bloch: hopping J between neighbouring harmonics, a tilt w from one to the next; J = w M / 8, so the swing
    // spans about half the chain -- upwards from the fundamental, the timbre at rest being the packet at rest. Its eigen system by Jacobi's rotations (cyclic, to round-off).
    double       a[M][M] = {}, v[M][M] = {};
    const double J = w * M / 8.0;
    for ( int h = 0; h < M; ++h )
    {
        a[h][h] = w * h;
        v[h][h] = 1.0;
        if ( h + 1 < M ) a[h][h + 1] = a[h + 1][h] = J;   // + J: a packet at rest climbs (its band's top)
    }
    for ( int sweep = 0; sweep < 50; ++sweep )
    {
        double off = 0.0;
        for ( int p = 0; p < M; ++p )
            for ( int q = p + 1; q < M; ++q ) off += a[p][q] * a[p][q];
        if ( off < 1.0e-24 ) break;
        for ( int p = 0; p < M; ++p )
            for ( int q = p + 1; q < M; ++q )
            {
                if ( std::fabs( a[p][q] ) < 1.0e-30 ) continue;
                const double theta = 0.5 * ( a[q][q] - a[p][p] ) / a[p][q];
                const double t =
                    ( theta >= 0.0 ? 1.0 : -1.0 ) / ( std::fabs( theta ) + std::sqrt( theta * theta + 1.0 ) );
                const double c = 1.0 / std::sqrt( t * t + 1.0 ), s = t * c;
                for ( int k = 0; k < M; ++k )
                {   // the columns p and q
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for ( int k = 0; k < M; ++k )
                {   // and the rows
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for ( int k = 0; k < M; ++k )
                {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
    }
    // The ladder made exact: the eigenvalues in order, set w apart -- in the middle of the chain they are already, at
    // its ends nearly -- so the swing comes back whole after the period.
    int order[M];
    for ( int n = 0; n < M; ++n ) order[n] = n;
    std::sort( order, order + M, [&a]( int x, int y ) { return a[x][x] < a[y][y]; } );
    for ( int r = 0; r < M; ++r )
    {
        const int n = order[r];
        for ( int h = 0; h < M; ++h ) m_talbotVec[static_cast<size_t>( r * M + h )] = static_cast<float>( v[h][n] );
        m_talbotVal[static_cast<size_t>( r )] = w * r;
    }
}

void Engine::stepLayers( float dt )
{
    const int first = static_cast<int>( m_voices.size() + m_ghosts.size() );

    // The phantom layer: the pitches it carries -- a melody of its own, the notes gone lately, or the chord -- folded
    // into 220 .. 880 Hz, where a turn of phase between the ears is heard best.
    {
        CloudVoice &cv = m_cloud.voice( first );
        const float want = cur( ParamIdE::PHANTOM );
        m_phantomGain += ( want - m_phantomGain ) * ( 1.0f - std::exp( -dt / 0.3f ) );
        if ( want <= 0.0f && m_phantomGain < 1.0e-4f )
        {
            m_phantomGain = 0.0f;
            cv.m_active = 0;
        }
        else
        {
            struct Pitch
            {
                double hz;
                float  weight;
            };
            std::array<Pitch, 64> sounding{};   // fixed: nothing is allocated a block
            int                   soundingCount = 0;
            auto                  sound = [&]( double hz, float weight )
            {
                if ( soundingCount < static_cast<int>( sounding.size() ) )
                    sounding[static_cast<size_t>( soundingCount++ )] = { hz, weight };
            };
            const int line = std::clamp( static_cast<int>( std::lround( cur( ParamIdE::PHANTOM_LINE ) ) ), 0, 2 );
            for ( int k = 0; k < m_echoCount; ++k ) m_echoAge[static_cast<size_t>( k )] += dt;
            while ( m_echoCount > 0 && m_echoAge[static_cast<size_t>( m_echoCount - 1 )] > 30.0f ) --m_echoCount;
            if ( line == 0 )
            {   // Melody: a line of its own among the scale's pitches in 220 .. 880 Hz, a step of one or two every
                // four to eight seconds, each note fading in over a second and a half as the last fades out
                std::array<double, 128> scale{};
                int                     n = 0;
                for ( int k = 0; k < 128; ++k )
                {
                    const double f = frequencyOf( k );
                    if ( f >= 220.0 && f < 880.0 && ( n == 0 || f > scale[static_cast<size_t>( n - 1 )] * 1.01 ) )
                        scale[static_cast<size_t>( n++ )] = f;
                }
                m_melodyNext -= dt;
                if ( n > 0 && ( m_melodyNext <= 0.0f || m_melodyAt < 0 ) )
                {
                    int       at = m_melodyAt < 0 ? n / 2 : m_melodyAt;
                    const int step = 1 + static_cast<int>( m_phantomRng.uniform() * 2.0f );
                    at += m_phantomRng.uniform() < 0.5f ? -step : step;
                    if ( at < 0 || at >= n )
                        at = std::clamp( 2 * ( m_melodyAt < 0 ? n / 2 : m_melodyAt ) - at, 0, n - 1 );
                    m_melodyHz[1] = m_melodyHz[0];
                    m_melodyWeight[1] = m_melodyWeight[0];
                    m_melodyHz[0] = static_cast<float>( scale[static_cast<size_t>( at )] );
                    m_melodyWeight[0] = 0.0f;
                    m_melodyAt = at;
                    m_melodyNext = 4.0f + 4.0f * m_phantomRng.uniform();
                }
                const float fade = dt / 1.5f;
                m_melodyWeight[0] = std::min( 1.0f, m_melodyWeight[0] + fade );
                m_melodyWeight[1] = std::max( 0.0f, m_melodyWeight[1] - fade );
                for ( int k = 0; k < 2; ++k )
                    if ( m_melodyHz[k] > 0.0f && m_melodyWeight[k] > 0.0f ) sound( m_melodyHz[k], m_melodyWeight[k] );
            }
            else if ( line == 1 )
            {   // Echo: the notes that ended lately, fading over half a minute
                for ( int k = 0; k < m_echoCount; ++k )
                    sound( m_echoHz[static_cast<size_t>( k )], 1.0f - m_echoAge[static_cast<size_t>( k )] / 30.0f );
            }
            else
            {   // Chord: the notes sounding now
                for ( const Voice &v : m_voices )
                    if ( v.m_note >= 0 && !v.m_stealing && v.m_lastGain > 0.02f && v.m_pre <= 0.0f )
                        sound( v.m_hz, std::min( 1.0f, 1.5f * v.m_lastGain ) );
            }
            cv.m_phantomCount = 0;
            for ( int s = 0; s < soundingCount; ++s )
            {
                const Pitch &p = sounding[static_cast<size_t>( s )];
                if ( cv.m_phantomCount >= PhantomTargets ) break;
                double f = std::max( p.hz, 1.0 );
                while ( f < 220.0 ) f *= 2.0;
                while ( f >= 880.0 ) f *= 0.5;
                cv.m_phantomHz[cv.m_phantomCount] = static_cast<float>( f );
                cv.m_phantomWeight[cv.m_phantomCount] = p.weight;
                ++cv.m_phantomCount;
            }
            cv.m_phantomWidth = cur( ParamIdE::PHANTOM_WIDTH );
            cv.m_restart = cv.m_active ? 0 : 1;
            cv.m_active = 1;
            cv.m_kind = VoiceKindE::PHANTOM;
            cv.m_partials = 8192;
            cv.m_level = VoiceLevel * 0.6f;
            cv.m_gain = m_phantomGain;
            cv.m_distance = 0.0f;
            cv.m_stereo = 0.0f;
            cv.m_airAbsorb = 1.0f;
            cv.m_seed = 0x50484E54u;
            cv.m_onset = 0;
            cv.m_harmonicLife = 0;
            cv.m_subClouds = 0;
            cv.m_grain = 0.0f;
        }
    }

    // The dream: it keeps what was heard every three seconds if it is new, and wanders among its memories; its voice
    // fades in once there are two of them to wander between.
    {
        Dream      &D = m_cloud.dream();
        CloudVoice &cv = m_cloud.voice( first + 1 );
        const float want = cur( ParamIdE::DREAM );
        const int   memories = m_cloud.dreamMemories();
        const float target = memories >= 2 ? want : 0.0f;
        m_dreamGain += ( target - m_dreamGain ) * ( 1.0f - std::exp( -dt / 2.0f ) );
        D.m_on = want > 0.0f || m_dreamGain > 1.0e-4f ? 1 : 0;
        if ( D.m_on )
        {
            const float focus = cur( ParamIdE::DREAM_FOCUS ), drift = cur( ParamIdE::DREAM_DRIFT );
            D.m_focus = 4.0f + 96.0f * focus * focus;
            D.m_noise = 0.05f + 0.6f * drift;
            D.m_ease = ( 0.05f + 0.6f * drift * drift ) * dt;   // a step of 1/20 .. 2/3 a second towards the recall
            D.m_novelty = 0.985f;
            if ( ++m_dreamBlocks >= static_cast<int>( 3.0f / dt ) )
            {
                D.m_store = want > 0.0f ? 1 : 0;   // nothing new is kept while it fades out
                m_dreamBlocks = 0;
            }
        }
        if ( !D.m_on )
        {
            cv.m_active = 0;
        }
        else
        {
            cv.m_restart = cv.m_active ? 0 : 1;
            cv.m_active = 1;
            cv.m_kind = VoiceKindE::DREAM;
            cv.m_partials = 16384;
            cv.m_level = VoiceLevel * 0.8f;
            cv.m_gain = m_dreamGain;
            cv.m_distance = cur( ParamIdE::DREAM_DEPTH );
            cv.m_stereo = 1.0f;
            cv.m_drift = 8.0f;
            cv.m_driftRate = 0.05f;
            cv.m_shimmer = 0.3f;
            cv.m_shimmerRate = 0.1f;
            cv.m_airAbsorb = 1.0f;
            cv.m_seed = 0x44524D53u;
            cv.m_onset = 0;
            cv.m_harmonicLife = 0;
            cv.m_subClouds = 0;
            cv.m_grain = 0.0f;
        }
    }

    // The chime field: its level gliding, the wind's push by the knob (measured: a strike or none a second at 0.3,
    // a few at half, dozens at 0.7, more than a hundred at the top) and
    // its gust (a slow random of three seconds' memory), the chimes tuned to the scale's pitches in their kind's
    // register, moved by Chime Pitch.
    {
        ChimeField &C = m_cloud.chimes();
        const float want = cur( ParamIdE::CHIMES );
        m_chimeLevel += ( want - m_chimeLevel ) * ( 1.0f - std::exp( -dt / 0.3f ) );
        const bool on = want > 0.0f || m_chimeLevel > 1.0e-4f;
        C.m_on = on ? 1 : 0;
        if ( on && !m_chimesOn ) C.m_reset = 1;   // a field begins from calm air and still clappers
        m_chimesOn = on;
        if ( on )
        {
            C.m_kick = 16.0f;
            C.m_level = 0.25f * m_chimeLevel * VoiceLevel;
            C.m_decay = cur( ParamIdE::CHIME_DECAY );
            C.m_kind =
                std::clamp( static_cast<int>( std::lround( cur( ParamIdE::CHIME_KIND ) ) ), 0, NumChimeKinds - 1 );
            static constexpr float Low[NumChimeKinds] = { 400.0f, 600.0f, 220.0f, 300.0f };
            const float            lo = Low[C.m_kind] * std::exp2( cur( ParamIdE::CHIME_PITCH ) ), hi = 4.0f * lo;
            int                    count = 0;
            for ( int k = 0; k < 128 && count < ChimePitches; ++k )
            {
                const double f = frequencyOf( k );
                if ( f >= lo && f < hi ) C.m_pitch[count++] = static_cast<float>( f );
            }
            C.m_pitchCount = count;
        }
    }
}

void Engine::stepNature( float dt )
{
    auto glide = [dt]( float &v, float want, float tau )
    {
        v += ( want - v ) * ( 1.0f - std::exp( -dt / tau ) );
        if ( want <= 0.0f && v < 1.0e-4f ) v = 0.0f;
    };
    Weather    &W = m_cloud.weather();
    ChimeField &C = m_cloud.chimes();
    const float rain = cur( ParamIdE::RAIN ), leaves = cur( ParamIdE::LEAVES ), surf = cur( ParamIdE::SURF );
    glide( m_rainLevel, rain, 0.3f );
    glide( m_leavesLevel, leaves, 0.3f );
    glide( m_surfLevel, surf, 0.5f );
    W.m_rainOn = ( rain > 0.0f || m_rainLevel > 0.0f ) ? 1 : 0;
    W.m_leavesOn = ( leaves > 0.0f || m_leavesLevel > 0.0f ) ? 1 : 0;
    W.m_surfOn = ( surf > 0.0f || m_surfLevel > 0.0f ) ? 1 : 0;
    if ( W.m_rainOn && !m_rainWas ) W.m_rainFresh = 1;   // the showers begin even
    m_rainWas = W.m_rainOn != 0;
    const bool weather = W.m_rainOn || W.m_leavesOn || W.m_surfOn;
    W.m_wind = weather ? 1 : 0;   // the chimes' own need is added at launch

    // The wind, for the chimes or the weather: its gust a slow random of three seconds' memory, its push by the knob
    // (measured with the chimes: a strike or none a second at 0.3, a few at half, dozens at 0.7, more than a hundred
    // at the top). It begins from calm air when it begins to blow.
    const bool windOn =
        weather || C.m_on || cur( ParamIdE::HARP ) > 0.0f || m_harpLevel > 0.0f;   // and the harp (Phase 30)
    if ( windOn && !m_windWas ) W.m_windFresh = 1;
    m_windWas = windOn;
    const float wind = cur( ParamIdE::WIND );
    if ( windOn )
    {
        m_gust += -m_gust * dt / 3.0f + std::sqrt( 2.0f * dt / 3.0f ) * m_windRng.bipolar() * 1.7320508f;
        const float gust = std::max( 0.0f, 1.0f + cur( ParamIdE::GUSTS ) * m_gust );
        C.m_force = 2.0e-5f * std::pow( 2.0f * wind, 1.2f ) * gust;
    }

    // The rain: its showers' windward feed (four slow randoms of ten seconds' memory, near to far: a shower about half as
    // wide as the field; with six seconds the left and the right differed by a decibel), what it falls on,
    // the pitches tuned drops sing (bubbles 1 .. 8 kHz, a roof's ring 300 Hz .. 2.4 kHz).
    if ( W.m_rainOn )
    {
        for ( int k = 0; k < 4; ++k )
        {
            m_shower[k] +=
                -m_shower[k] * dt / 10.0f + std::sqrt( 2.0f * dt / 10.0f ) * m_showerRng.bipolar() * 1.7320508f;
            W.m_shower[k] = std::clamp( 0.8f * m_shower[k], -1.0f, 1.0f );
        }
        W.m_rainLevel = 4.0f * m_rainLevel * VoiceLevel;
        W.m_rainRate = rain > 0.0f ? cur( ParamIdE::RAIN_RATE ) : 0.0f;   // fading out: no new drops
        W.m_surface =
            std::clamp( static_cast<int>( std::lround( cur( ParamIdE::RAIN_SURFACE ) ) ), 0, NumRainSurfaces - 1 );
        W.m_showers = cur( ParamIdE::RAIN_SHOWERS );
        W.m_tune = cur( ParamIdE::RAIN_TUNE );
        W.m_advect = 0.15f;
        const float lo = W.m_surface == 3 ? 300.0f : 1000.0f, hi = W.m_surface == 3 ? 2400.0f : 8000.0f;
        int         count = 0;
        for ( int k = 0; k < 128 && count < RainPitches; ++k )
        {
            const double f = frequencyOf( k );
            if ( f >= lo && f < hi ) W.m_pitch[count++] = static_cast<float>( f );
        }
        W.m_pitchCount = count;
    }
    // The leaves: they rustle fully at about the wind's mean speed at the top of the knob (measured: 0.026, 0.050,
    // 0.069 lattice units at 0.2, 0.5, 0.8), their gusting part counted twice.
    if ( W.m_leavesOn )
    {
        W.m_leavesLevel = 2.5f * m_leavesLevel * VoiceLevel;
        W.m_foliage = std::clamp( static_cast<int>( std::lround( cur( ParamIdE::FOLIAGE ) ) ), 0, NumFoliages - 1 );
        W.m_windRef = 0.08f;
    }
    // The sea: its height by the wind (a calm swell at none), its clock.
    if ( W.m_surfOn )
    {
        W.m_surfLevel = 3.0f * m_surfLevel * VoiceLevel;
        W.m_surfHeight = 0.7f + 1.3f * wind * wind;
        W.m_surfPeriod = cur( ParamIdE::SURF_SWELL );
        W.m_shore = std::clamp( static_cast<int>( std::lround( cur( ParamIdE::SHORE ) ) ), 0, NumShores - 1 );
        W.m_surfDistance = cur( ParamIdE::SURF_DISTANCE );
        W.m_time += dt;
        if ( W.m_time > 86400.0f ) W.m_time -= 86400.0f;
    }

    // The bloom: a plate of the knob's side (bronze, 1.5 mm: a mode every 3.24 / area Hz, the plate 1.37 times as
    // long as wide), its ring at 200 Hz, its drive and glide.
    {
        Bloom      &B = m_cloud.bloom();
        const float want = cur( ParamIdE::BLOOM );
        glide( m_bloomLevel, want, 0.3f );
        B.m_on = ( want > 0.0f || m_bloomLevel > 0.0f ) ? 1 : 0;
        const float side = cur( ParamIdE::BLOOM_SIZE );
        B.m_level = m_bloomLevel;
        B.m_spacing = 3.24f / ( 1.37f * side * side );
        B.m_decay = cur( ParamIdE::BLOOM_DECAY );
        B.m_drive = cur( ParamIdE::BLOOM_DRIVE );
        B.m_glide = cur( ParamIdE::BLOOM_GLIDE );
        B.m_width = 1.0f;
    }

    // The swarmalators: scattered afresh when they begin.
    {
        SwarmSpace &S = m_cloud.swarm();
        const float want = cur( ParamIdE::SWARMAL );
        glide( m_swarmDepth, want, 0.5f );
        S.m_on = ( want > 0.0f || m_swarmDepth > 0.0f ) ? 1 : 0;
        if ( S.m_on && !m_swarmWas ) S.m_reset = 1;
        m_swarmWas = S.m_on != 0;
        S.m_depth = m_swarmDepth;
        S.m_j = cur( ParamIdE::SWARMAL_GATHER );
        S.m_k = cur( ParamIdE::SWARMAL_SYNC );
        S.m_rate = cur( ParamIdE::SWARMAL_RATE );
        S.m_speed = cur( ParamIdE::SWARMAL_SPEED );
        S.m_spread = cur( ParamIdE::SWARMAL_SPREAD );
    }
}

void Engine::setLocation( double latitude, double longitude )
{
    m_latitude.store( std::clamp( latitude, -90.0, 90.0 ), std::memory_order_relaxed );
    m_longitude.store( std::clamp( longitude, -180.0, 180.0 ), std::memory_order_relaxed );
}

void Engine::location( double &latitude, double &longitude ) const
{
    latitude = m_latitude.load( std::memory_order_relaxed );
    longitude = m_longitude.load( std::memory_order_relaxed );
}

void Engine::setHeadsetPose( const HeadsetPose &p )
{
    const std::lock_guard<std::mutex> lock( m_poseLock );
    m_posePending = p;
}

bool Engine::dreamJournal( std::vector<std::uint8_t> &out )
{
    const std::lock_guard<std::mutex> lock( m_snapLock );
    out = m_journalPendingSet ? m_journalPending : m_journal;
    return out.size() > 12;
}

void Engine::setDreamJournal( const std::vector<std::uint8_t> &in )
{
    if ( in.size() < 12 || std::memcmp( in.data(), "DRJ1", 4 ) != 0 ) return;
    const std::lock_guard<std::mutex> lock( m_snapLock );
    m_journalPending = in;
    m_journalPendingSet = true;
}

void Engine::stepJournal( float dt )
{
    // A journal set (a state restored): onto the card before this block, its memories the dream's.
    {
        std::unique_lock<std::mutex> lock( m_snapLock, std::try_to_lock );
        if ( lock.owns_lock() && m_journalPendingSet )
        {
            const std::vector<std::uint8_t> &j = m_journalPending;
            std::uint32_t                    count = 0, bands = 0;
            std::memcpy( &count, j.data() + 4, 4 );
            std::memcpy( &bands, j.data() + 8, 4 );
            const size_t per = 4 + static_cast<size_t>( bands );
            if ( bands == static_cast<std::uint32_t>( DreamBands ) && count <= DreamJournalMost &&
                 j.size() >= 12 + per * count )
            {
                m_journalScratch.assign( static_cast<size_t>( count ) * DreamBands, 0.0f );
                for ( std::uint32_t m = 0; m < count; ++m )
                {
                    const std::uint8_t *at = j.data() + 12 + per * m;
                    float               scale = 0.0f;
                    std::memcpy( &scale, at, 4 );
                    if ( !std::isfinite( scale ) ) scale = 0.0f;
                    for ( int b = 0; b < DreamBands; ++b )
                        m_journalScratch[static_cast<size_t>( m ) * DreamBands + b] =
                            scale * static_cast<float>( at[4 + b] ) * ( 1.0f / 255.0f );
                }
                m_cloud.dreamWrite( m_journalScratch.data(), static_cast<int>( count ) );
                m_journal = j;
            }
            m_journalPendingSet = false;
            m_journalPending.clear();
        }
    }
    // While the dream dreams, every half minute: its newest memories brought down, a byte a band (each memory's own
    // scale), for the state to keep.
    m_journalClock -= dt;
    if ( m_journalClock > 0.0f ) return;
    m_journalClock = 30.0f;
    if ( m_dreamGain <= 0.0f || cur( ParamIdE::DREAM_JOURNAL ) < 0.5f ) return;
    const int count = m_cloud.dreamRead( m_journalScratch, DreamJournalMost );
    if ( count <= 0 ) return;
    std::vector<std::uint8_t> j( 12 + static_cast<size_t>( count ) * ( 4 + DreamBands ) );
    const std::uint32_t head[2] = { static_cast<std::uint32_t>( count ), static_cast<std::uint32_t>( DreamBands ) };
    std::memcpy( j.data(), "DRJ1", 4 );
    std::memcpy( j.data() + 4, head, 8 );
    for ( int m = 0; m < count; ++m )
    {
        const float *v = m_journalScratch.data() + static_cast<size_t>( m ) * DreamBands;
        float        scale = 0.0f;
        for ( int b = 0; b < DreamBands; ++b ) scale = std::max( scale, v[b] );
        std::uint8_t *at = j.data() + 12 + static_cast<size_t>( m ) * ( 4 + DreamBands );
        std::memcpy( at, &scale, 4 );
        for ( int b = 0; b < DreamBands; ++b )
            at[4 + b] = static_cast<std::uint8_t>(
                scale > 0.0f ? std::lround( std::clamp( v[b] / scale, 0.0f, 1.0f ) * 255.0f ) : 0 );
    }
    const std::lock_guard<std::mutex> lock( m_snapLock );
    m_journal.swap( j );
}

void Engine::stepHeadset( float dt )
{
    (void)dt;
    {
        std::unique_lock<std::mutex> lock( m_poseLock, std::try_to_lock );
        if ( lock.owns_lock() ) m_pose = m_posePending;
    }
    const HeadsetPose &P = m_pose;
    const float        hands = P.m_valid ? std::clamp( cur( ParamIdE::VR_HANDS ), 0.0f, 1.0f ) : 0.0f;
    // The head's turn: the binaural cloud and the thunder held where they are in the room as the head turns.
    m_headYaw = P.m_valid && cur( ParamIdE::HEAD_TRACKING ) >= 0.5f ? P.m_yaw : 0.0f;
    m_cloud.thunder().m_yaw = m_headYaw;
    // The hands, each where it is (an arm's reach of 0.8 m across the field, ahead of the listener):
    //   - in the wind a post that moves, its wake shedding vortices into the chimes and the leaves;
    //   - in the halo a wall of potential (on the rim the sound goes round it: the rim cannot be stopped);
    //   - in the wave room an absorber where the hand is open, the voices' source where it pinches;
    //   - in the physarum's trail a line of food at the hand's height as a pitch (low at the hip, high above the head).
    Weather  &W = m_cloud.weather();
    Halo     &H = m_cloud.halo();
    WaveRoom &R = m_cloud.room();
    for ( int h = 0; h < 2; ++h )
    {
        const bool   on = hands > 0.0f && P.m_tracked[h];
        const float *x = P.m_hand[h];
        const float  across = std::clamp( x[0] / 0.8f, -1.0f, 1.0f ), ahead = std::clamp( x[2] / 0.8f, 0.0f, 1.0f );
        W.m_handR[h] = on ? 2 + static_cast<int>( std::lround( 4.0f * hands ) ) : 0;
        W.m_handX[h] =
            std::clamp( static_cast<int>( std::lround( ( 0.5f + 0.5f * across ) * ( WindW - 1 ) ) ), 0, WindW - 1 );
        W.m_handY[h] = std::clamp( static_cast<int>( std::lround( ahead * ( WindH - 1 ) ) ), 0, WindH - 1 );
        H.m_defectX[h] = on ? static_cast<int>( std::lround( across * HaloRadius ) ) : 99;
        H.m_defectY[h] = on ? static_cast<int>( std::lround( ahead * HaloRadius ) ) : 99;
        R.m_absorb[h] = -1;
        if ( on && R.m_on && R.m_nx > 2 && R.m_ny > 2 && R.m_nz > 2 )
        {
            constexpr float Cell = 0.1143f;
            const int       ex = ( R.m_ear[0] % RoomNX + R.m_ear[1] % RoomNX ) / 2;
            const int       ey = ( R.m_ear[0] / RoomNX ) % RoomNY, ez = R.m_ear[0] / ( RoomNX * RoomNY );
            const int       cx = std::clamp( ex + static_cast<int>( std::lround( x[0] / Cell ) ), 1, R.m_nx - 2 );
            const int       cy = std::clamp( ey - static_cast<int>( std::lround( x[2] / Cell ) ), 1, R.m_ny - 2 );
            const int       cz = std::clamp( ez + static_cast<int>( std::lround( x[1] / Cell ) ), 1, R.m_nz - 2 );
            const int       cell = cx + RoomNX * ( cy + RoomNY * cz );
            if ( P.m_pinch[h] > 0.5f )
            {   // the voices sound from the hand
                R.m_srcZ = cz;
                for ( int s = 0; s < R.m_count; ++s ) R.m_source[s] = cell;
            }
            else
            {
                R.m_absorb[h] = cell;
            }
        }
        m_handFood[h] = -1.0f;
        if ( on )
        {   // food at the hand's height: 60 Hz at the hip (0.6 m below the head) .. 4 kHz above the head (0.4 m)
            const float  u = std::clamp( ( x[1] + 0.6f ) / 1.0f, 0.0f, 1.0f );
            const double f = 60.0 * std::pow( 4000.0 / 60.0, static_cast<double>( u ) );
            m_handFood[h] = static_cast<float>( std::log2( f / 20.0 ) * PhysW / 9.96578428 );
        }
    }
    m_handFoodWeight = 1.0f + 3.0f * hands;
    H.m_defectV = 4.0f * hands;
    H.m_defectR2 = 4.0f;
    R.m_absorbR2 = 4.0f;
    R.m_absorbK = 0.08f * hands;
}

void Engine::stepNight( float dt )
{
    auto glide = [dt]( float &v, float want, float tau )
    {
        v += ( want - v ) * ( 1.0f - std::exp( -dt / tau ) );
        if ( want <= 0.0f && v < 1.0e-4f ) v = 0.0f;
    };

    // The night arc's sun (Phase 30): where it stands for the place and the clock, the clock running on with the time
    // rendered.
    {
        m_sunTime += dt;
        const double e = sunElevation( m_sunTime, m_latitude.load( std::memory_order_relaxed ),
                                       m_longitude.load( std::memory_order_relaxed ) );
        m_night = nightOf( e );
        m_nightLive.store( m_night, std::memory_order_relaxed );
        m_sunLive.store( static_cast<float>( e ), std::memory_order_relaxed );
    }

    // Risset's rhythm (Phase 30): five layers of a pulse an octave of tempo apart, all slowing (or quickening) an octave
    // in Time, each weighted by a bell over its place in the octaves -- the slowest fading out below as a new fastest
    // fades in above, the beats of every layer on the slowest's -- so the pulse slows for ever and never arrives (Risset's
    // eternal ritardando). The voices' gain follows it; the pulse is a source of the matrix too.
    {
        constexpr int K = 5;
        const float   depth = std::clamp( cur( ParamIdE::RISSET ), 0.0f, 1.0f );
        const float   T = std::max( cur( ParamIdE::RISSET_TIME ), 1.0f );
        const bool    quicker = cur( ParamIdE::RISSET_WAY ) >= 0.5f;
        m_rissetU += ( quicker ? -dt : dt ) / T;
        if ( m_rissetU >= 1.0f )
        {   // an octave slower: every layer is now the one below it was
            m_rissetU -= 1.0f;
            m_rissetPhase = std::fmod( 2.0 * m_rissetPhase, static_cast<double>( 1 << K ) );
        }
        else if ( m_rissetU < 0.0f )
        {   // an octave quicker: every layer is now the one above it was
            m_rissetU += 1.0f;
            m_rissetPhase *= 0.5;
        }
        const double slowest =
            cur( ParamIdE::RISSET_RATE ) * std::exp2( 1.0 - 0.5 * K );   // the middle's pulse is Rate
        m_rissetPhase += dt * slowest * std::exp2( -static_cast<double>( m_rissetU ) );
        if ( m_rissetPhase >= static_cast<double>( 1 << K ) ) m_rissetPhase -= static_cast<double>( 1 << K );
        float sum = 0.0f, weight = 0.0f;
        for ( int k = 0; k < K; ++k )
        {
            const float  pos = static_cast<float>( k ) + 1.0f - m_rissetU;
            const float  s = std::sin( 3.1415927f * pos / static_cast<float>( K ) );
            const float  w = s * s;
            const double ph = std::fmod( std::ldexp( m_rissetPhase, k ), 1.0 );
            const float  bump = 0.5f - 0.5f * static_cast<float>( std::cos( 6.283185307179586 * ph ) );
            const float  b2 = bump * bump;
            sum += w * b2 * b2;   // sin^8: a pulse about a quarter of its period wide
            weight += w;
        }
        m_rissetPulse = weight > 0.0f ? sum / weight : 0.0f;
        m_rissetGain = depth > 0.0f ? std::clamp( 1.0f - depth + depth * 2.2f * m_rissetPulse, 0.0f, 2.0f ) : 1.0f;
    }
    m_lightning *= std::exp( -dt / 0.6f );   // the last strike's flash fading

    // The night chorus (Phase 31): scattered anew as it begins; its warmth the knob's, a few degrees cooler as the night
    // arc's night deepens (and the insects slower); its carriers pulled to the scale's pitches 200 Hz .. 12 kHz.
    {
        NightChorus &C = m_cloud.chorus();
        const float  want = cur( ParamIdE::CHORUS );
        glide( m_chorusLevel, want, 0.5f );
        C.m_on = ( want > 0.0f || m_chorusLevel > 0.0f ) ? 1 : 0;
        if ( C.m_on && !m_chorusWas ) C.m_reset = 1;
        m_chorusWas = C.m_on != 0;
        if ( C.m_on )
        {
            C.m_level = 0.6f * m_chorusLevel * VoiceLevel;
            C.m_kind =
                std::clamp( static_cast<int>( std::lround( cur( ParamIdE::CHORUS_KIND ) ) ), 0, NumChorusKinds - 1 );
            C.m_warmth =
                cur( ParamIdE::CHORUS_WARMTH ) - 3.0f * m_night * std::clamp( cur( ParamIdE::NIGHT_ARC ), 0.0f, 1.0f );
            C.m_sync = cur( ParamIdE::CHORUS_SYNC );
            C.m_tune = cur( ParamIdE::CHORUS_TUNE );
            int count = 0;
            for ( int k = 0; k < 128 && count < ChorusPitches; ++k )
            {
                const double f = frequencyOf( k );
                if ( f >= 200.0 && f < 12000.0 ) C.m_pitch[count++] = static_cast<float>( f );
            }
            C.m_pitchCount = count;
        }
    }

    // The Aeolian harp: twelve strings to the chord's pitch classes (the scale's root and fifth when nothing sounds),
    // each in the octave above 55 Hz moved by Harp Pitch, the tuning gliding (3 s) as the chord moves; their gauges
    // from thick to thin about Gauge's, so each sings another harmonic; spread across the front, near.
    {
        AeolianHarp &H = m_cloud.harp();
        const float  want = cur( ParamIdE::HARP );
        glide( m_harpLevel, want, 0.5f );
        H.m_on = ( want > 0.0f || m_harpLevel > 0.0f ) ? 1 : 0;
        if ( H.m_on && !m_harpWas )
        {
            H.m_clear = 1;
            m_harpHz.fill( 0.0f );
        }
        m_harpWas = H.m_on != 0;
        if ( H.m_on )
        {
            std::array<int, 12> pcs{};
            int                 count = 0;
            bool                seen[12] = {};
            for ( const Voice &v : m_voices )
            {
                if ( v.m_note < 0 || v.m_stealing || v.m_lastGain < 0.02f ) continue;
                seen[( ( v.m_note % 12 ) + 12 ) % 12] = true;
            }
            for ( int pc = 0; pc < 12; ++pc )
                if ( seen[pc] ) pcs[static_cast<size_t>( count++ )] = pc;
            if ( count == 0 )
            {
                pcs[0] = m_rootPc;
                pcs[1] = ( m_rootPc + 7 ) % 12;
                count = 2;
            }
            const float octave = std::exp2( cur( ParamIdE::HARP_PITCH ) );
            const float gauge = 0.4e-3f + 1.4e-3f * cur( ParamIdE::HARP_GAUGE );
            const float a = 1.0f - std::exp( -dt / 3.0f );
            for ( int k = 0; k < HarpStrings; ++k )
            {
                const int   pc = pcs[static_cast<size_t>( k % count )];
                const int   note = 33 + ( ( pc - 9 ) % 12 + 12 ) % 12;   // A1 = 55 Hz .. G#2
                const float hz = std::max( 25.0f, static_cast<float>( frequencyOf( note ) ) * octave );
                float      &g = m_harpHz[static_cast<size_t>( k )];
                g = g <= 0.0f ? hz : std::exp( std::log( g ) + a * ( std::log( hz ) - std::log( g ) ) );
                H.m_hz[k] = g;
                H.m_diameter[k] = gauge * ( 0.7f + 0.6f * static_cast<float>( k ) / ( HarpStrings - 1 ) );
                H.m_place[k] = -0.55f + 1.1f * static_cast<float>( k ) / ( HarpStrings - 1 );
            }
            H.m_level = 35.0f * m_harpLevel * VoiceLevel;   // measured: the strings at Wind 0.5 about -20 dBFS
            H.m_speed = 120.0f;   // the lattice's mean speed at Wind 0.5, about 0.05, as 6 m/s
            H.m_x = WindW / 2;
            H.m_y = WindH / 3;
        }
    }

    // Thunder: strikes as a Poisson stream (from one every two minutes to eight a minute), the first within two seconds
    // of its beginning (heard as late as it is far: three seconds a kilometre); each strike's foot at the storm's distance (a lognormal spread) in any direction, the cloud's
    // base 1.5 to 4 km up, its energy lognormal, its channel as tortuous and branched as Rumble says.
    {
        Thunder    &T = m_cloud.thunder();
        const float want = cur( ParamIdE::THUNDER );
        glide( m_thunderLevel, want, 0.5f );
        const bool wasOn = T.m_on != 0;
        T.m_on = ( want > 0.0f || m_thunderLevel > 0.0f ) ? 1 : 0;
        if ( T.m_on && !wasOn ) m_strikeIn = 0.5f + 1.5f * m_stormRng.uniform();
        if ( T.m_on )
        {
            T.m_level = 30.0f * m_thunderLevel * VoiceLevel;   // measured: a strike at 3 km peaks about -12 dBFS
            T.m_yaw = 0.0f;
            m_strikeIn -= dt;
            if ( want > 0.0f && m_strikeIn <= 0.0f )
            {
                const float perMinute = 0.5f + 7.5f * want * want;
                m_strikeIn = -std::log( std::max( 1.0e-6f, 1.0f - m_stormRng.uniform() ) ) * 60.0f / perMinute;
                auto gauss = [this]
                {
                    const float u1 = std::max( 1.0e-6f, m_stormRng.uniform() ), u2 = m_stormRng.uniform();
                    return std::sqrt( -2.0f * std::log( u1 ) ) * std::cos( 6.2831853f * u2 );
                };
                const float rumble = cur( ParamIdE::THUNDER_RUMBLE );
                T.m_strike = 1;
                T.m_seed = static_cast<int>( m_stormRng.next() & 0x7FFFFFFF );
                T.m_distance = std::clamp( 1000.0f * cur( ParamIdE::THUNDER_DISTANCE ) * std::exp( 0.4f * gauss() ),
                                           300.0f, 20000.0f );
                T.m_azimuth = 6.2831853f * m_stormRng.uniform() - 3.1415927f;
                T.m_height = 1500.0f + 2500.0f * m_stormRng.uniform();
                T.m_energy = std::exp( 0.5f * gauss() );
                T.m_tortuosity = 0.12f + 0.3f * rumble;
                T.m_branches = rumble;
                T.m_near = std::clamp( 1.0f - T.m_distance / 2500.0f, 0.05f, 0.5f );
                ++m_strikes;
                m_lastStrikeM = T.m_distance;
                m_lightning = 1.0f;   // the flash, seen at once (the thunder comes three seconds a kilometre later)
            }
        }
    }

    // The singing ice: a sheet warmed into its critical state as it begins, the cold's cracks as often as Cold says,
    // its flexural constant from its thickness, a lake of 400 m.
    {
        SingingIce &I = m_cloud.ice();
        const float want = cur( ParamIdE::ICE );
        glide( m_iceLevel, want, 0.5f );
        I.m_on = ( want > 0.0f || m_iceLevel > 0.0f ) ? 1 : 0;
        if ( I.m_on && !m_iceWas )
        {   // a sheet that has cracked for a long time
            I.m_reset = 1;
            m_cloud.setIceStress( warmIce().data() );
            m_crackIn = 0.3f * m_stormRng.uniform();
        }
        m_iceWas = I.m_on != 0;
        if ( I.m_on )
        {
            I.m_level = 1.2f * m_iceLevel * VoiceLevel;
            I.m_beta = 958.0f * cur( ParamIdE::ICE_THICKNESS );
            I.m_size = 400.0f;
            m_crackIn -= dt;
            if ( want > 0.0f && m_crackIn <= 0.0f )
            {   // the cold's cracks as a Poisson stream: 0.15 .. 6 a second
                const float perSecond = 0.15f * std::pow( 40.0f, cur( ParamIdE::ICE_COLD ) );
                m_crackIn = -std::log( std::max( 1.0e-6f, 1.0f - m_stormRng.uniform() ) ) / perSecond;
                I.m_crack = 1;
            }
        }
    }
}

/** @brief The time mirror's cells' share of what they heard, sent back (Phase 31, measured: selftest 98). */
constexpr float MirrorShare = 0.08f;

void Engine::stepRooms( float dt )
{
    auto glide = [dt]( float &v, float want, float tau )
    {
        v += ( want - v ) * ( 1.0f - std::exp( -dt / tau ) );
        if ( want <= 0.0f && v < 1.0e-4f ) v = 0.0f;
    };
    const int first = static_cast<int>( m_voices.size() + m_ghosts.size() );

    // The halo: its carriers given to the sounding notes, each driven at the rim from its own direction (the golden
    // angle by its note, so a chord comes in from round the circle) -- strongly for its first 0.4 s, gently while it
    // sounds -- and let ring once its note is gone, until it has died away and the carrier is free again.
    {
        Halo       &H = m_cloud.halo();
        CloudVoice &cv = m_cloud.voice( first + 2 );
        const float want = cur( ParamIdE::HALO );
        glide( m_haloGain, want, 0.5f );
        H.m_on = ( want > 0.0f || m_haloGain > 0.0f ) ? 1 : 0;
        if ( !H.m_on )
        {
            cv.m_active = 0;
            m_haloNote.fill( -1 );
        }
        else
        {
            float energy[HaloCarriers], angle[HaloCarriers], rim[HaloCarriers];
            m_cloud.haloStats( energy, angle, rim );
            float peak = 1.0e-12f;
            for ( float e : energy ) peak = std::max( peak, e );
            m_haloHeld.fill( false );
            for ( const Voice &v : m_voices )
            {
                if ( v.m_note < 0 || v.m_stealing || v.m_lastGain < 0.02f || v.m_pre > 0.0f ) continue;
                int slot = -1;
                for ( int c = 0; c < HaloCarriers && slot < 0; ++c )
                    if ( m_haloNote[static_cast<size_t>( c )] == v.m_note ) slot = c;
                if ( slot < 0 )
                {   // a free carrier, or the faintest one not held
                    float least = 1.0e30f;
                    for ( int c = 0; c < HaloCarriers; ++c )
                    {
                        const float e = m_haloNote[static_cast<size_t>( c )] < 0 ? -1.0f : energy[c];
                        if ( e < least && !m_haloHeld[static_cast<size_t>( c )] )
                        {
                            least = e;
                            slot = c;
                        }
                    }
                    if ( slot < 0 ) continue;
                    m_haloNote[static_cast<size_t>( slot )] = v.m_note;
                    m_haloAge[static_cast<size_t>( slot )] = 0.0f;
                    H.m_clear |= 1 << slot;
                    cv.m_haloHz[slot] = static_cast<float>( v.m_hz );
                    // where it is struck: the rim site nearest its direction
                    const float a = std::fmod( static_cast<float>( v.m_note ) * 2.3999632f, 6.2831853f ) - 3.1415927f;
                    int         best = 0;
                    float       bestD = 1.0e9f;
                    for ( int k = 0; k < m_haloRimCount; ++k )
                    {
                        float d = std::fabs( m_haloRimAngle[static_cast<size_t>( k )] - a );
                        d = std::min( d, 6.2831853f - d );
                        if ( d < bestD )
                        {
                            bestD = d;
                            best = m_haloRim[static_cast<size_t>( k )];
                        }
                    }
                    H.m_site[slot] = best;
                }
                m_haloHeld[static_cast<size_t>( slot )] = true;
                cv.m_haloHz[slot] = static_cast<float>( v.m_hz );   // it follows its note's glides
                H.m_drive[slot] = ( m_haloAge[static_cast<size_t>( slot )] < 0.4f ? 1.0f : 0.25f ) *
                                  std::min( 1.0f, 1.5f * v.m_lastGain );
            }
            for ( int c = 0; c < HaloCarriers; ++c )
            {
                m_haloAge[static_cast<size_t>( c )] += dt;
                if ( !m_haloHeld[static_cast<size_t>( c )] )
                {
                    H.m_drive[c] = 0.0f;
                    if ( m_haloNote[static_cast<size_t>( c )] >= 0 && m_haloAge[static_cast<size_t>( c )] > 1.0f &&
                         energy[c] < 1.0e-6f * peak )
                    {   // died away: free, silent
                        m_haloNote[static_cast<size_t>( c )] = -1;
                        cv.m_haloHz[c] = 0.0f;
                    }
                }
            }
            const float round = cur( ParamIdE::HALO_SPEED );
            const float sign = std::lround( cur( ParamIdE::HALO_TURN ) ) == 1 ? -1.0f : 1.0f;
            H.m_omega = 125.0f / std::max( round, 1.0f );   // measured: omega 10 runs round a radius of 12 in 12.5 s
            H.m_mass = sign * ( 1.0f + 1.5f * cur( ParamIdE::HALO_TOPOLOGY ) );
            H.m_disorder = 2.0f * cur( ParamIdE::HALO_DISORDER );
            H.m_damp = 1.0f / ( 1.5f * std::max( round, 1.0f ) );
            cv.m_restart = cv.m_active ? 0 : 1;
            cv.m_active = 1;
            cv.m_kind = VoiceKindE::HALO;
            cv.m_partials = HaloCarriers * HaloSites;
            cv.m_level = VoiceLevel * 3.5f;
            cv.m_gain = m_haloGain;
            cv.m_distance = 0.0f;
            cv.m_stereo = 0.0f;
            cv.m_seed = 0x48414C4Fu;
            cv.m_onset = 0;
            cv.m_harmonicLife = 0;
            cv.m_subClouds = 0;
            cv.m_grain = 0.0f;
        }
    }

    // The wave room: its size (a change clears the air), its walls, where the listener stands, the voices in it --
    // each where it stands, round in front of the listener by a direction of its own and as far as its distance.
    {
        WaveRoom   &W = m_cloud.room();
        const float want = cur( ParamIdE::WAVE_ROOM );
        glide( m_roomLevel, want, 0.5f );
        constexpr float Cell = 0.1143f;   // m: the grid at 48 kHz, a Courant number of a half
        const float     lambda = 343.0f * RoomDecim / ( static_cast<float>( m_cfg.m_sampleRate ) * Cell );
        W.m_on = ( ( want > 0.0f || m_roomLevel > 0.0f ) && lambda < 0.57f ) ? 1 : 0;
        if ( W.m_on )
        {
            const float s = cur( ParamIdE::ROOM_SIZE );
            const int   nx = std::clamp( static_cast<int>( std::lround( RoomNX * s ) ), 8, RoomNX );
            const int   ny = std::clamp( static_cast<int>( std::lround( RoomNY * s ) ), 8, RoomNY );
            const int   nz = std::clamp( static_cast<int>( std::lround( RoomNZ * ( 0.5f + 0.5f * s ) ) ), 8, RoomNZ );
            if ( nx != m_roomDims[0] || ny != m_roomDims[1] || nz != m_roomDims[2] ) W.m_clear = 1;
            m_roomDims[0] = W.m_nx = nx;
            m_roomDims[1] = W.m_ny = ny;
            m_roomDims[2] = W.m_nz = nz;
            W.m_lambda2 = lambda * lambda;
            W.m_beta = 0.03f + 0.7f * cur( ParamIdE::ROOM_ABSORB );
            W.m_level = m_roomLevel * 40.0f;
            auto      cell = [&]( int x, int y, int z ) { return x + RoomNX * ( y + RoomNY * z ); };
            const int xl = nx / 2;
            const int yl = std::clamp( static_cast<int>( std::lround( ( 0.25f + 0.5f * cur( ParamIdE::ROOM_WALK ) ) *
                                                                      static_cast<float>( ny - 1 ) ) ),
                                       1, ny - 2 );
            const int zl = std::min( nz - 2, static_cast<int>( std::lround( 1.6f / Cell ) ) );
            W.m_ear[0] = cell( xl - 1, yl, zl );
            W.m_ear[1] = cell( xl + 1, yl, zl );
            W.m_srcZ = std::min( nz - 2, static_cast<int>( std::lround( 1.4f / Cell ) ) );
            // The time mirror (Phase 31): RoomMirrors cells on a ring round the listener on the sources' plane, its
            // window the knob's; it listens for a window and sends it back reversed in the next, each cell by
            // MirrorShare of what it heard (sixty-four of them, the swell measured about as loud as the sound itself).
            {
                const float mirror = cur( ParamIdE::MIRROR );
                W.m_mirrorOn = mirror > 0.0f ? 1 : 0;
                W.m_mirrorGain = MirrorShare * mirror;
                W.m_mirrorLen = std::clamp( static_cast<int>( cur( ParamIdE::MIRROR_TIME ) *
                                                              static_cast<float>( m_cfg.m_sampleRate ) / RoomDecim ),
                                            64, MirrorMax );
                if ( m_mirrorPos >= W.m_mirrorLen ) m_mirrorPos = 0;
                W.m_mirrorPos = m_mirrorPos;
                W.m_mirrorBuf = m_mirrorBuf;
                m_mirrorPos += m_cfg.m_gpuBlock / RoomDecim;
                if ( m_mirrorPos >= W.m_mirrorLen )
                {
                    m_mirrorPos -= W.m_mirrorLen;
                    m_mirrorBuf ^= 1;
                }
                const float radius = 0.35f * static_cast<float>( std::min( nx, ny ) );
                for ( int k = 0; k < RoomMirrors; ++k )
                {
                    const float a = 6.2831853f * static_cast<float>( k ) / RoomMirrors;
                    const int   x =
                        std::clamp( static_cast<int>( std::lround( xl + radius * std::cos( a ) ) ), 1, nx - 2 );
                    const int y =
                        std::clamp( static_cast<int>( std::lround( yl + radius * std::sin( a ) ) ), 1, ny - 2 );
                    W.m_mirror[k] = cell( x, y, W.m_srcZ );
                }
            }
            int count = 0;
            for ( int i = 0; i < first && count < RoomSources; ++i )
            {
                const CloudVoice &v = m_cloud.voice( i );
                if ( !v.m_active ) continue;
                const float side = 0.8f * ( 2.0f * std::fmod( 0.618034f * static_cast<float>( i + 1 ), 1.0f ) - 1.0f );
                const float theta = side * 1.0471976f, r = ( 1.5f + 4.0f * v.m_distance ) / Cell;
                const int   x = std::clamp( static_cast<int>( std::lround( xl + r * std::sin( theta ) ) ), 1, nx - 2 );
                const int   y = std::clamp( static_cast<int>( std::lround( yl - r * std::cos( theta ) ) ), 1, ny - 2 );
                W.m_source[count] = cell( x, y, W.m_srcZ );
                W.m_voiceOf[count] = i;
                W.m_gain[count] = 1.0f;
                ++count;
            }
            W.m_count = count;
        }
    }

    // The quantum walk on the Tonnetz: twelve amplitudes on the pitch classes, each joined to its fifths and its
    // thirds (major and minor, up and down); the walk turns as e^(-i A w t), w such that between two looks the
    // phase is 0.15 .. 6 radians (Spread). Its share of moves to a neighbour does not fall steadily with it -- 79 %
    // at 0.15, 47 % at 0.5, 78 % at 2, 36 % at 4 (simulated, 4000 looks): interference, which a random walk has not. Every Look Every seconds it is looked at: the root moves to a pitch class
    // drawn by |psi|^2 -- the nearest note of it to where the root is -- and the walk is partly collapsed there (70 %),
    // so it keeps a memory of where it might have gone. A family jam's leader keeps the root: then it waits.
    {
        const float amount = cur( ParamIdE::QUANTUM );
        const bool  on = amount > 0.0f && m_bp.m_on && m_jamRoot.load( std::memory_order_relaxed ) < 0;
        const int   rootPc = ( ( m_brain.root() % 12 ) + 12 ) % 12;
        if ( on && ( !m_qwWas || rootPc != m_qwRootPc ) )
        {   // begun, or the conductor moved the root itself: the walk sees it there and goes on from it
            m_qwPsi.fill( 0.0 );
            m_qwPsi[static_cast<size_t>( 2 * rootPc )] = 1.0;
            if ( !m_qwWas ) m_qwNext = cur( ParamIdE::QUANTUM_TIME );
            m_qwRootPc = rootPc;
        }
        m_qwWas = on;
        if ( on )
        {
            const double T = std::max( 1.0f, cur( ParamIdE::QUANTUM_TIME ) );
            const double w = ( 0.15 + 5.85 * cur( ParamIdE::QUANTUM_SPREAD ) ) / T;
            // A step by the circulant's own basis: psi_hat_k *= e^(-i lambda_k w dt), lambda_k = 2 sum cos(2 pi k s / 12).
            std::array<double, 24> hat{};
            for ( int k = 0; k < 12; ++k )
                for ( int j = 0; j < 12; ++j )
                {
                    const double a = -6.283185307179586 * k * j / 12.0;
                    hat[2 * k] += m_qwPsi[2 * j] * std::cos( a ) - m_qwPsi[2 * j + 1] * std::sin( a );
                    hat[2 * k + 1] += m_qwPsi[2 * j] * std::sin( a ) + m_qwPsi[2 * j + 1] * std::cos( a );
                }
            for ( int k = 0; k < 12; ++k )
            {
                double l = 0.0;
                for ( int s : { 3, 4, 7 } ) l += 2.0 * std::cos( 6.283185307179586 * k * s / 12.0 );
                const double ph = -l * w * dt, c = std::cos( ph ), sn = std::sin( ph );
                const double re = hat[2 * k] * c - hat[2 * k + 1] * sn, im = hat[2 * k] * sn + hat[2 * k + 1] * c;
                hat[2 * k] = re;
                hat[2 * k + 1] = im;
            }
            for ( int j = 0; j < 12; ++j )
            {
                double re = 0.0, im = 0.0;
                for ( int k = 0; k < 12; ++k )
                {
                    const double a = 6.283185307179586 * k * j / 12.0;
                    re += hat[2 * k] * std::cos( a ) - hat[2 * k + 1] * std::sin( a );
                    im += hat[2 * k] * std::sin( a ) + hat[2 * k + 1] * std::cos( a );
                }
                m_qwPsi[2 * j] = re / 12.0;
                m_qwPsi[2 * j + 1] = im / 12.0;
            }
            m_qwNext -= dt;
            if ( m_qwNext <= 0.0f )
            {
                m_qwNext = static_cast<float>( T );
                double norm = 0.0;
                for ( double x : m_qwPsi ) norm += x * x;
                if ( norm > 1.0e-12 && m_qwRng.uniform() < amount )
                {
                    double u = m_qwRng.uniform() * norm;
                    int    pc = 11;
                    for ( int j = 0; j < 12; ++j )
                    {
                        u -= m_qwPsi[static_cast<size_t>( 2 * j )] * m_qwPsi[static_cast<size_t>( 2 * j )] +
                             m_qwPsi[static_cast<size_t>( 2 * j + 1 )] * m_qwPsi[static_cast<size_t>( 2 * j + 1 )];
                        if ( u <= 0.0 )
                        {
                            pc = j;
                            break;
                        }
                    }
                    const double amp = std::sqrt(
                        m_qwPsi[static_cast<size_t>( 2 * pc )] * m_qwPsi[static_cast<size_t>( 2 * pc )] +
                        m_qwPsi[static_cast<size_t>( 2 * pc + 1 )] * m_qwPsi[static_cast<size_t>( 2 * pc + 1 )] );
                    for ( double &x : m_qwPsi ) x *= std::sqrt( 0.3 / norm );
                    if ( amp > 0.0 )
                    {   // partly collapsed onto what was found, its phase kept
                        m_qwPsi[static_cast<size_t>( 2 * pc )] += std::sqrt( 0.7 ) *
                                                                  m_qwPsi[static_cast<size_t>( 2 * pc )] /
                                                                  ( amp * std::sqrt( 0.3 / norm ) );
                        m_qwPsi[static_cast<size_t>( 2 * pc + 1 )] += std::sqrt( 0.7 ) *
                                                                      m_qwPsi[static_cast<size_t>( 2 * pc + 1 )] /
                                                                      ( amp * std::sqrt( 0.3 / norm ) );
                    }
                    double n2 = 0.0;
                    for ( double x : m_qwPsi ) n2 += x * x;
                    for ( double &x : m_qwPsi ) x /= std::sqrt( std::max( n2, 1.0e-30 ) );
                    if ( pc != rootPc )
                    {
                        const int r = m_brain.root();
                        int       best = r, bestDist = 99;
                        for ( int d = -6; d <= 6; ++d )
                            if ( ( ( ( r + d ) % 12 ) + 12 ) % 12 == pc && std::abs( d ) < bestDist )
                            {
                                bestDist = std::abs( d );
                                best = r + d;
                            }
                        m_brain.setRoot( std::clamp( best, 0, 127 ) );
                        m_qwLastStep = ( ( pc - rootPc ) % 12 + 12 ) % 12;
                        m_qwRootPc = pc;
                        ++m_qwMoves;
                    }
                }
            }
        }
    }
}

void Engine::captureGhost( int i )
{
    size_t slot = 0;
    for ( size_t g = 0; g < m_ghosts.size(); ++g )
    {
        if ( !m_ghosts[g].m_on )
        {
            slot = g;
            break;
        }
        if ( m_ghosts[g].m_env < m_ghosts[slot].m_env ) slot = g;
    }
    Ghost &g = m_ghosts[slot];
    g.m_base = m_cloud.voice( i );
    g.m_base.m_releaseAt = -1.0f;   // a ghost is held: its motion loops on
    g.m_on = true;
    g.m_restart = true;
    g.m_seed = static_cast<std::uint32_t>( m_ghostRng.next() ) | 1u;
    g.m_age = 0.0f;
    g.m_life = std::max( 1.0f, cur( ParamIdE::MEMORY_TIME ) );
    g.m_bloom = std::max( 0.05f, cur( ParamIdE::MEMORY_BLOOM ) );
    g.m_env = 0.0f;
}

void Engine::renderBlock()
{
    const NoDenormals noDenormals;
    LARGE_INTEGER     qpcStart;
    QueryPerformanceCounter( &qpcStart );
    const int          G = m_cfg.m_gpuBlock;
    const double       dt = G / m_cfg.m_sampleRate;
    const float        dtf = static_cast<float>( dt );
    const std::int64_t pos = m_renderPos.load( std::memory_order_relaxed );
    readParams();
    refreshSpectrum();
    refreshScale();
    refreshLandscape();
    {   // the scale for the panel: four octaves of keys folded into one over the root, the same pitch once
        const double ref = m_blk.m_refPitch;
        const int    rootNote = 60 + m_rootPc;
        const double f0 = ref * std::pow( 2.0, ( rootNote - 69 ) / 12.0 );
        float        cents[48];
        int          n = 0;
        for ( int k = 0; k < 48 && f0 > 0.0; ++k )
        {
            const double hz = scaleFrequency( m_scale, rootNote - 12 + k, rootNote, ref, m_snapKeys );
            if ( !( hz > 0.0 ) || !std::isfinite( hz ) )
                continue;   // a scale that has no pitch there: nothing to draw (and no NaN for the sort)
            double c = std::fmod( 1200.0 * std::log2( hz / f0 ), 1200.0 );
            if ( !std::isfinite( c ) ) continue;
            if ( c < 0.0 ) c += 1200.0;
            if ( c > 1199.5 ) c = 0.0;
            bool seen = false;
            for ( int j = 0; j < n && !seen; ++j ) seen = std::fabs( cents[j] - c ) < 1.0;
            if ( !seen ) cents[n++] = static_cast<float>( c );
        }
        std::sort( cents, cents + n );
        for ( int i = 0; i < n; ++i )
            m_viewScale[static_cast<size_t>( i )].store( cents[i], std::memory_order_relaxed );
        m_viewScaleCount.store( n, std::memory_order_relaxed );
        m_viewRootHz.store( static_cast<float>( f0 ), std::memory_order_relaxed );
        // Every key's pitch, anew only when what tunes it changed.
        const double key[5] = { static_cast<double>( m_scaleChoice ), static_cast<double>( m_rootPc ), ref,
                                static_cast<double>( m_blk.m_purity ), m_snapKeys ? 1.0 : 0.0 };
        if ( !std::equal( std::begin( key ), std::end( key ), std::begin( m_noteHzKey ) ) )
        {
            std::copy( std::begin( key ), std::end( key ), std::begin( m_noteHzKey ) );
            for ( int k = 0; k < 128; ++k )
                m_viewNoteHz[static_cast<size_t>( k )].store( static_cast<float>( frequencyOf( k ) ),
                                                              std::memory_order_relaxed );
        }
    }

    // 1. The played notes that are due, in the order they came, each on its own sample.
    drainEvents();
    for ( size_t i = 0; i < m_pending.size(); )
    {
        const Event e = m_pending[i];
        if ( e.m_stamp >= pos + G )
        {
            ++i;
            continue;
        }
        m_pending.erase( m_pending.begin() + static_cast<std::ptrdiff_t>( i ) );
        if ( e.m_stamp < pos ) m_late.fetch_add( 1, std::memory_order_relaxed );
        const int onset = static_cast<int>( std::clamp<std::int64_t>( e.m_stamp - pos, 0, G - 1 ) );
        if ( e.m_type == 0 )
            startNote( e.m_note, e.m_velocity, m_blk.m_keysDepth, 0, onset, e.m_channel );
        else if ( e.m_type == 1 )
            stopNote( e.m_note, 0, e.m_channel );
        else
            for ( const Voice &v : m_voices )
                if ( v.m_note >= 0 && v.m_source == 0 && !v.m_released ) stopNote( v.m_note, 0 );
    }

    // 2. The conductors. Step fires on the rising edge of its switch, or from autoplayStep().
    const bool step = cur( ParamIdE::AUTO_STEP ) >= 0.5f;
    if ( ( step && !m_stepWas ) | m_stepAsked.exchange( false, std::memory_order_relaxed ) ) m_brain.requestStep();
    m_stepWas = step;
    if ( m_fillAsked.exchange( false, std::memory_order_relaxed ) )
    {
        if ( m_bp.m_on ) m_brain.requestFill();
        if ( m_blk.m_brain2 ) m_brain2.requestFill();
    }
    updateLight();
    m_bp.m_light = m_lightOn ? m_light : nullptr;
    m_bp2.m_light = m_bp.m_light;
    auto freqOf = [this]( int note ) { return frequencyOf( note ); };
    m_brain.update( dt, m_bp, -1, freqOf,
                    [this]( const BrainEvent &e )
                    {
                        if ( e.m_type == BrainEvent::TypeE::NOTE_ON )
                            startNote( e.m_note, e.m_velocity, brainDistance(), 1, 0 );
                        else
                            stopNote( e.m_note, 1 );
                    } );
    // The family jam: a follower's conductor keeps its root on the leader's pitch class, the nearest
    // note of it (Noctuary's way: its tuning stays its own).
    if ( const int jr = m_jamRoot.load( std::memory_order_relaxed );
         jr >= 0 && ( ( m_brain.root() % 12 ) + 12 ) % 12 != jr )
    {
        const int r = m_brain.root();
        int       best = r, bestDist = 99;
        for ( int d = -6; d <= 6; ++d )
            if ( ( ( ( r + d ) % 12 ) + 12 ) % 12 == jr && std::abs( d ) < bestDist )
            {
                bestDist = std::abs( d );
                best = r + d;
            }
        m_brain.setRoot( std::clamp( best, 0, 127 ) );
    }
    updateFoundation();
    m_brain2.setRoot( std::clamp( m_brain.root() + m_blk.m_brain2Interval, 0, 127 ) );
    m_brain2.update( dt, m_bp2, -1, freqOf,
                     [this]( const BrainEvent &e )
                     {
                         if ( e.m_type == BrainEvent::TypeE::NOTE_ON )
                             startNote( e.m_note, e.m_velocity,
                                        m_blk.m_brain2Depth * ( 0.85f + 0.15f * m_rng.uniform() ), 2, 0 );
                         else
                             stopNote( e.m_note, 2 );
                     } );
    m_brainRoot.store( m_brain.root(), std::memory_order_relaxed );
    {   // the conductor's countdown and the foundation, for the panel
        const double next = m_brain.untilNext();
        if ( next > m_conductorWas + 0.5 )
            m_conductorSpan.store( static_cast<float>( next ), std::memory_order_relaxed );
        m_conductorWas = next;
        m_conductorNext.store( static_cast<float>( next ), std::memory_order_relaxed );
        m_foundationView.store( m_foundationNote, std::memory_order_relaxed );
    }

    // 3. The voices: envelope, place, frequency, colour, filter, air, motion.
    // While the sound glides, the tuning follows slowly: a new scale is reached by the partials
    // sliding into it over seconds (a tenth of the glide, at most six).
    const double glideSeconds = m_morphT < 1.0f ? std::clamp( 0.1 * m_morphLen, GlideSeconds, 6.0 ) : GlideSeconds;
    const double follow = 1.0 - std::exp( -dt / glideSeconds );
    const double tideMul =
        std::pow( 2.0, m_blk.m_tide * m_tide.update( dtf, 1.0f / ( m_blk.m_tidePeriod * 60.0f ), m_side ) / 1200.0 );
    // Self-tuning: where a harmonic of one voice nearly meets a harmonic of another (within Reach),
    // the two pull together -- weighted by their loudness, by a bell over the distance and by how low
    // the harmonics are (the beats of low ones are what is heard). Every voice moves half the way
    // its weighted mismatch asks for, at the speed Self-Tune sets (the other moves the other half),
    // and the anchor draws it back to its pitch in the scale. A chord in equal temperament settles
    // towards just intervals and gives way again at the next change; a note starts on the scale.
    {
        const float grav = cur( ParamIdE::TUNE_GRAVITY ), reach = std::max( 1.0f, cur( ParamIdE::TUNE_REACH ) ),
                    anchor = cur( ParamIdE::TUNE_ANCHOR );
        const int   nv = static_cast<int>( m_voices.size() );
        if ( grav > 0.0f )
        {
            float kLogH[9] = { 0.0f };   // log2 of where harmonics 1 .. 8 lie (both stretches)
            for ( int h = 1; h <= 8; ++h )
                kLogH[h] = std::log2(
                    partialRatio( h, m_blk.m_inharmonicB, m_blk.m_stretch, m_blk.m_horizon, m_blk.m_horizonRatio ) );
            std::array<float, 64> lb{}, moved{};
            for ( int a = 0; a < nv && a < 64; ++a )
            {
                const Voice &v = m_voices[static_cast<size_t>( a )];
                lb[static_cast<size_t>( a )] =
                    v.m_note >= 0
                        ? static_cast<float>( std::log2( frequencyOf( v.m_note ) * tideMul ) ) + v.m_tuneCents / 1200.0f
                        : 0.0f;
            }
            const float rate = 1.0f - std::exp( -2.0f * grav * dtf );
            for ( int a = 0; a < nv && a < 64; ++a )
            {
                const Voice &va = m_voices[static_cast<size_t>( a )];
                if ( va.m_note < 0 || va.m_stealing || va.m_lastGain <= 0.0f ) continue;
                double force = 0.0, wsum = 0.0;
                for ( int b = 0; b < nv && b < 64; ++b )
                {
                    const Voice &vb = m_voices[static_cast<size_t>( b )];
                    if ( b == a || vb.m_note < 0 || vb.m_stealing || vb.m_lastGain <= 0.0f ) continue;
                    for ( int ha = 1; ha <= 8; ++ha )
                        for ( int hb = 1; hb <= 8; ++hb )
                        {
                            const float d = 1200.0f * ( lb[static_cast<size_t>( b )] + kLogH[hb] -
                                                        lb[static_cast<size_t>( a )] - kLogH[ha] );
                            if ( std::fabs( d ) >= reach ) continue;
                            const double w = static_cast<double>( va.m_lastGain ) * vb.m_lastGain *
                                             std::exp( -( d / reach ) * ( d / reach ) ) / ( ha * hb );
                            force += w * d;
                            wsum += w;
                        }
                }
                if ( wsum > 1.0e-12 ) moved[static_cast<size_t>( a )] = static_cast<float>( 0.5 * force / wsum ) * rate;
            }
            const float back = std::exp( -2.0f * anchor * dtf );
            for ( int a = 0; a < nv && a < 64; ++a )
            {
                Voice &v = m_voices[static_cast<size_t>( a )];
                v.m_tuneCents = std::clamp( ( v.m_tuneCents + moved[static_cast<size_t>( a )] ) * back, -50.0f, 50.0f );
            }
        }
        else
        {
            for ( Voice &v : m_voices ) v.m_tuneCents *= std::exp( -2.0f * dtf );   // switched off: back to the scale
        }
    }
    const MotionKindE motion = static_cast<MotionKindE>( std::clamp( m_blk.m_motion, 0, NumMotions - 1 ) );
    // A voice that reads the landscape is as much quieter as the landscape is dark: made up here
    // (at most 18 dB), so a dim landscape changes the motion, not the level.
    float landMakeup = 1.0f;
    if ( motionUsesLandscape( motion ) )
        landMakeup = 1.0f / std::max( 0.125f, 1.0f - m_blk.m_motionDepth + m_blk.m_motionDepth * m_landRms );
    stepSeasons( dtf );
    stepLayers( dtf );
    stepNature( dtf );
    stepRooms( dtf );
    stepNight( dtf );
    stepHeadset( dtf );   // after the rooms and the binaural cloud: the hands and the head on top
    stepJournal( dtf );
    int   sounding = 0;
    float chordGain = 0.0f;   // the conductor's voices' envelopes (not the foundation's), for the foundation's level
    int   chordVoices = 0;
    int   by[3] = { 0, 0, 0 };
    float srcW = 0.0f, srcD = 0.0f;   // the near room's source: the voices' distance, weighted by loudness
    for ( int i = 0; i < static_cast<int>( m_voices.size() ); ++i )
    {
        Voice      &v = m_voices[static_cast<size_t>( i )];
        CloudVoice &cv = m_cloud.voice( i );
        if ( v.m_stealing )
        {
            if ( v.m_stealBlocks > 0 )
            {
                --v.m_stealBlocks;
                cv.m_gain = 0.0f;
                v.m_env.kill();
                ++sounding;
                continue;
            }
            v.m_stealing = false;
            if ( v.m_next.m_type == 0 )
                beginOn( i, v.m_next.m_note, v.m_next.m_velocity, v.m_nextDistance, v.m_nextSource, 0,
                         v.m_next.m_channel );
            else
                v.m_note = -1;
        }
        if ( v.m_note < 0 )
        {
            cv.m_active = 0;
            continue;
        }
        v.m_env.setTimes( m_blk.m_attack, m_blk.m_decay, m_blk.m_sustain, m_blk.m_release );
        // Premonition: before the note its swell -- rising as a reverb does backwards, a quarter of the way an e-fold
        // -- the cloud wide and far; at its start the swell is drawn into the note over 0.15 s.
        float preWide = 0.0f;
        if ( v.m_pre > 0.0f )
        {
            v.m_pre -= dtf;
            if ( v.m_pre <= 0.0f )
            {
                v.m_pre = 0.0f;
                v.m_suck = 1.0f;
                if ( !v.m_released ) v.m_env.noteOn();
            }
        }
        float gain;
        if ( v.m_pre > 0.0f )
        {
            gain = 0.5f * m_blk.m_premonition * std::exp( -4.0f * v.m_pre / std::max( v.m_preTotal, 0.5f ) );
            preWide = 1.0f;
        }
        else
        {
            gain = v.m_env.process();
            if ( v.m_suck > 0.0f )
            {
                gain = std::max( gain, 0.5f * m_blk.m_premonition * v.m_suck );
                preWide = v.m_suck;
                v.m_suck = std::max( 0.0f, v.m_suck - dtf / 0.15f );
            }
        }
        v.m_lastGain = gain;
        if ( v.m_source == 1 && !v.m_released )
        {
            chordGain += gain * gain;
            ++chordVoices;
        }   // the foundation matches these
        if ( !v.m_env.isActive() && v.m_pre <= 0.0f && v.m_suck <= 0.0f )
        {
            // With dispersion the late partials sound on after the envelope: the voice stays until
            // the latest of them has heard the end of it.
            if ( !v.m_tailArmed )
            {
                v.m_tailArmed = true;
                v.m_tailBlocks +=
                    static_cast<int>( std::ceil( std::fabs( cur( ParamIdE::DISPERSION ) ) * m_cfg.m_sampleRate / G ) );
            }
            if ( v.m_tailBlocks > 0 )
            {
                --v.m_tailBlocks;
                cv.m_gain = 0.0f;
                ++sounding;
                continue;
            }
            v.m_note = -1;
            cv.m_active = 0;
            if ( v.m_source != FoundationSource )
            {   // the phantom echo remembers it: the newest first, the oldest dropped
                const int n = std::min( m_echoCount, PhantomTargets - 1 );
                for ( int k = n; k > 0; --k )
                {
                    m_echoHz[static_cast<size_t>( k )] = m_echoHz[static_cast<size_t>( k - 1 )];
                    m_echoAge[static_cast<size_t>( k )] = m_echoAge[static_cast<size_t>( k - 1 )];
                }
                m_echoHz[0] = static_cast<float>( v.m_hz );
                m_echoAge[0] = 0.0f;
                m_echoCount = n + 1;
            }
            continue;
        }
        // Breath: the voice's plane wanders (Noctuary's "the room breathes"); Doppler: coming closer
        // raises it, receding lowers it.
        v.m_distance = std::clamp(
            v.m_place + 0.35f * m_blk.m_breath * v.m_breath.update( dtf, m_blk.m_breathRate, m_side ), 0.0f, 1.0f );
        const float  velocity = ( v.m_distance - v.m_prevDistance ) / dtf * MetresPerUnit;
        const double doppler =
            v.m_restart ? 1.0 : 1.0 - static_cast<double>( m_blk.m_doppler * velocity / SpeedOfSound );
        v.m_prevDistance = v.m_distance;
        const double target =
            frequencyOf( v.m_note ) * tideMul * std::exp2( static_cast<double>( v.m_tuneCents ) / 1200.0 );
        v.m_hz = v.m_restart ? target : v.m_hz * std::pow( target / v.m_hz, follow );
        // MPE: a played note's own bend, pressure and slide while it is held (kept as they were once it
        // is let go, so a channel used again moves only its new note), and the master's bend. Each of
        // pressure and slide moves its target: pressure from nothing up, slide either way about its rest.
        float mpeBend = 0.0f;
        float mpe[NumMpeTargets] = {};
        if ( v.m_source == 0 )
        {
            if ( !v.m_released && v.m_channel >= 0 )
            {
                v.m_bend = m_chanBend[static_cast<size_t>( v.m_channel )].load( std::memory_order_relaxed );
                v.m_press = m_chanPress[static_cast<size_t>( v.m_channel )].load( std::memory_order_relaxed );
                v.m_slide = m_chanSlide[static_cast<size_t>( v.m_channel )].load( std::memory_order_relaxed );
            }
            mpeBend = m_chanBend[16].load( std::memory_order_relaxed ) + v.m_bend;
            if ( v.m_channel >= 0 )
            {
                mpe[std::clamp( static_cast<int>( std::lround( cur( ParamIdE::MPE_PRESS ) ) ), 0,
                                NumMpeTargets - 1 )] += cur( ParamIdE::MPE_PRESS_DEPTH ) * v.m_press;
                mpe[std::clamp( static_cast<int>( std::lround( cur( ParamIdE::MPE_SLIDE ) ) ), 0,
                                NumMpeTargets - 1 )] += cur( ParamIdE::MPE_SLIDE_DEPTH ) * ( 2.0f * v.m_slide - 1.0f );
            }
        }
        auto expr = [&mpe]( MpeTargetE t ) { return mpe[static_cast<int>( t )]; };
        // The filter's cutoff: key tracking, the envelope, its own drift of up to an octave, and MPE's up to three.
        const float oct = m_blk.m_keyTrack * static_cast<float>( v.m_note - 60 ) / 12.0f + m_blk.m_filterEnv * gain +
                          m_blk.m_filterDrift * v.m_cutoffDrift.update( dtf, 0.05f, m_side ) +
                          3.0f * expr( MpeTargetE::FILTER );
        const float fc = std::clamp( m_blk.m_cutoff * std::exp2( oct ), 20.0f, 20000.0f );
        const FilterModelE model =
            static_cast<FilterModelE>( std::clamp( m_blk.m_filterModel, 0, NumFilterModels - 1 ) );
        const float q =
            filterQ( static_cast<int>( model ), m_blk.m_resonance );   // Spectrum.h: the picture's curve reads the same

        // The foundation: as loud as the chord is (m_foundationMatch) -- a held voice beside voices that
        // spend half their lives fading in, or under a chord still arriving, would stand far over it --
        // and its knob on top.
        float level = VoiceLevel * ( v.m_source == FoundationSource ? ( 0.6f + 0.4f * m_foundation ) * m_foundationMatch
                                     : v.m_source != 0              ? 0.6f + 0.4f * v.m_velocity
                                                                    : 0.4f + 0.6f * v.m_velocity );
        cv.m_hz = static_cast<float>( v.m_hz * doppler * std::exp2( static_cast<double>( mpeBend ) / 12.0 ) );
        cv.m_level = level * landMakeup * std::exp2( expr( MpeTargetE::LEVEL ) );   // up to 6 dB
        cv.m_gain = gain * m_rissetGain;   // Risset's pulse (Phase 30), 1 when it is off
        // MPE's Brightness: the tilt flattened (or steepened) by up to 0.6 and the window opened by up
        // to a half -- the window alone moves little under a falling spectrum.
        cv.m_tilt = std::max( 0.0f, m_blk.m_tilt - 0.6f * expr( MpeTargetE::BRIGHTNESS ) );
        cv.m_brightness = std::clamp( m_blk.m_brightness + 0.5f * expr( MpeTargetE::BRIGHTNESS ), 0.0f, 1.0f );
        cv.m_oddEven = m_blk.m_oddEven;
        cv.m_inharmonic = m_blk.m_inharmonicB;
        cv.m_stretch = m_blk.m_stretch;
        cv.m_shift = m_blk.m_shift;
        // Turbulence: the cascade's drive, by the cube of the note's strength (with how loud the
        // instrument is, below): a soft chord stays pure.
        const float strength = 0.4f + 0.6f * v.m_velocity;
        cv.m_turbulence = m_turbDrive * strength * strength * strength;
        // Phase 26: the horizon, the grains, the harmonics' own lives.
        cv.m_horizon = m_blk.m_horizon;
        cv.m_horizonRatio = m_blk.m_horizonRatio;
        cv.m_grain = m_blk.m_grain;
        cv.m_grainLength = m_blk.m_grainLength;
        cv.m_grainDuty = m_blk.m_grainDuty;
        // a grain partial sounds sin^4 of its window a share Density of its time: mean square 3/8 Density
        cv.m_grainNorm = 1.0f / std::sqrt( 0.375f * std::clamp( m_blk.m_grainDuty, 0.01f, 1.0f ) );
        stepHarmonics( v, cv, dtf, gain, std::clamp( expr( MpeTargetE::BREATH ), 0.0f, 1.0f ) );
        cv.m_presenceDb = m_blk.m_presence;
        cv.m_lowCut = m_blk.m_padLowCut;
        cv.m_filterModel = static_cast<int>( model );
        cv.m_filterMix = m_sound.m_filterMix;   // for the panel; the card reads the filter list below
        cv.m_cutoff = fc;
        cv.m_filterQ = q;
        cv.m_vowel = 4.0f * std::clamp( std::log( fc / 40.0f ) / std::log( 18000.0f / 40.0f ), 0.0f, 1.0f );
        // During a morph between different filters (Morph.h): the heaviest four of the sound's, each
        // with this voice's key tracking, envelope and drift and the matrix's push, transported into
        // each other on the card. One filter alone is the filter above.
        cv.m_filterCount = 0;
        if ( m_sound.m_filterCount > 1 )
        {
            int       order[MorphFilters];
            const int n = m_sound.m_filterCount;
            for ( int k = 0; k < n; ++k ) order[k] = k;
            std::stable_sort( order, order + n, [this]( int a, int b )
                              { return m_sound.m_filters[a].m_weight > m_sound.m_filters[b].m_weight; } );
            const int count = std::min( n, VoiceFilters );
            float     sum = 0.0f;
            for ( int k = 0; k < count; ++k ) sum += m_sound.m_filters[order[k]].m_weight;
            const ParamDesc &cd = paramDesc( ParamIdE::CUTOFF );
            const ParamDesc &rd = paramDesc( ParamIdE::RESONANCE );
            for ( int k = 0; k < count; ++k )
            {
                const MorphFilter &mf = m_sound.m_filters[order[k]];
                const FilterModelE m = static_cast<FilterModelE>( std::clamp( mf.m_model, 0, NumFilterModels - 1 ) );
                const float        base =
                    fromNorm( cd, std::clamp( toNorm( cd, mf.m_cutoff ) + m_cutoffPush, 0.0f, 1.0f ) ) * m_jamCutoff;
                const float fck = std::clamp( base * std::exp2( oct ), 20.0f, 20000.0f );
                const float res =
                    fromNorm( rd, std::clamp( toNorm( rd, mf.m_resonance ) + m_resonancePush, 0.0f, 1.0f ) );
                const float qk = filterQ( static_cast<int>( m ), res );
                cv.m_filterWeight[k] = mf.m_weight / std::max( sum, 1.0e-9f );
                cv.m_filterModels[k] = static_cast<float>( m );
                cv.m_filterCutoffs[k] = fck;
                cv.m_filterQs[k] = qk;
                cv.m_filterVowels[k] =
                    4.0f * std::clamp( std::log( fck / 40.0f ) / std::log( 18000.0f / 40.0f ), 0.0f, 1.0f );
            }
            cv.m_filterCount = count;
        }
        cv.m_air = m_blk.m_air;
        cv.m_airCenter = m_blk.m_airCenter;
        cv.m_airWidth = m_blk.m_airWidth;
        cv.m_airJitter = m_blk.m_airJitter;
        cv.m_width = m_blk.m_width * std::exp2( 1.5f * expr( MpeTargetE::WIDTH ) );
        cv.m_widthSlope = m_blk.m_widthSlope;
        cv.m_drift = m_blk.m_drift;
        cv.m_driftRate = m_blk.m_driftRate;
        cv.m_shimmer = std::clamp( m_blk.m_shimmer + 0.5f * expr( MpeTargetE::SHIMMER ), 0.0f, 1.0f );
        cv.m_shimmerRate = m_blk.m_shimmerRate;
        cv.m_streams = m_blk.m_streams;
        cv.m_streamDepth = m_blk.m_streamDepth;
        cv.m_streamRate = m_blk.m_streamRate;
        cv.m_fusion = m_blk.m_fusion;
        cv.m_stereo = m_blk.m_stereo;
        cv.m_partials = m_blk.m_partials;
        cv.m_distance = std::clamp( v.m_distance - 0.5f * expr( MpeTargetE::NEARER ), 0.0f, 1.0f );
        if ( preWide > 0.0f )
        {   // the premonition: wide, far and a little dark, as a reverb's tail
            const float p = m_blk.m_premonition * preWide;
            cv.m_width *= 1.0f + 4.0f * p;
            cv.m_distance += ( std::max( cv.m_distance, 0.9f ) - cv.m_distance ) * p;
            cv.m_brightness *= 1.0f - 0.3f * p;
        }
        srcW += gain * level;
        srcD += gain * level * v.m_distance;
        if ( m_rippleVoices < 64 )
        {
            m_rippleHz[m_rippleVoices] = static_cast<float>( v.m_hz );
            m_rippleAmp[m_rippleVoices++] = gain * level;
        }
        cv.m_airAbsorb = 1.0f;
        cv.m_seed = v.m_seed;
        cv.m_active = 1;
        cv.m_restart = v.m_restart ? 1 : 0;
        if ( v.m_restart && m_rippleStrikeCount < 64 )
            m_rippleStrikes[m_rippleStrikeCount++] = static_cast<float>( v.m_hz );
        cv.m_onset = v.m_restart ? v.m_onset : 0;
        applyMotion( motion, cv, m_blk.m_motionRate,
                     std::clamp( m_blk.m_motionDepth * ( 1.0f + expr( MpeTargetE::MOTION ) ), 0.0f, 1.0f ) );
        v.m_restart = false;
        v.m_onset = 0;
        v.m_age += dt;
        ++sounding;
        ++by[v.m_source == FoundationSource ? 1
                                            : std::clamp( v.m_source, 0, 2 )];   // the foundation is the conductor's
    }
    m_sounding.store( sounding, std::memory_order_relaxed );
    {   // the chord's mean envelope, followed over a few seconds (the foundation's own sustain divided out)
        // As loud as half the chord's voices together (their power), seen over the foundation's own
        // sustain: half a voice under a lone note, a little over one voice under five.
        const float target = chordVoices > 0 ? 0.5f * std::sqrt( chordGain ) / std::max( m_blk.m_sustain, 0.2f ) : 0.0f;
        m_foundationMatch += ( std::min( target, 1.5f ) - m_foundationMatch ) * ( 1.0f - std::exp( -dtf / 4.0f ) );
    }
    if ( srcW > 1.0e-6f ) m_srcDistance = srcD / srcW;
    if ( m_prevSounding == 0 && sounding > 0 )
        m_phraseStart = true;   // a note into silence: the envelopes' phrase begins
    m_prevSounding = sounding;
    for ( int s = 0; s < 3; ++s ) m_soundingBy[s].store( by[s], std::memory_order_relaxed );
    // The memory's ghosts: each a released voice sounding on from where it was let go, rising over
    // Bloom, fading by 60 dB over Memory Time, and as it fades widening, drifting, darkening and
    // receding (Blur, Sink). New phases of its own, so it never adds to the voice it came from.
    {
        const float level = cur( ParamIdE::MEMORY_LEVEL ), blur = cur( ParamIdE::MEMORY_BLUR ),
                    sink = cur( ParamIdE::MEMORY_SINK );
        const int   first = static_cast<int>( m_voices.size() );
        for ( size_t gi = 0; gi < m_ghosts.size(); ++gi )
        {
            Ghost      &g = m_ghosts[gi];
            CloudVoice &cv = m_cloud.voice( first + static_cast<int>( gi ) );
            if ( !g.m_on )
            {
                cv.m_active = 0;
                continue;
            }
            g.m_age += dtf;
            g.m_env = std::min( 1.0f, g.m_age / g.m_bloom ) *
                      std::exp( -6.9077553f * std::max( 0.0f, g.m_age - g.m_bloom ) / g.m_life );
            if ( g.m_age > g.m_bloom && g.m_env < 1.0e-4f )
            {
                g.m_on = false;
                cv.m_active = 0;
                continue;
            }
            const float x = std::min( 1.0f, g.m_age / g.m_life );
            const float keepTime = cv.m_time;
            cv = g.m_base;
            cv.m_time = g.m_restart ? 0.0f : keepTime;
            cv.m_restart = g.m_restart ? 1 : 0;
            cv.m_onset = 0;
            cv.m_seed = g.m_seed;
            cv.m_active = 1;
            cv.m_gain = g.m_env * level;
            cv.m_width = g.m_base.m_width * ( 1.0f + 3.0f * blur * x );
            cv.m_drift = g.m_base.m_drift + 12.0f * blur * x;
            cv.m_brightness = g.m_base.m_brightness * ( 1.0f - 0.7f * blur * x );
            cv.m_distance = g.m_base.m_distance + ( 1.0f - g.m_base.m_distance ) * sink * std::min( 1.0f, 2.0f * x );
            g.m_restart = false;
        }
    }

    for ( int i = 0; i < static_cast<int>( m_voices.size() ) && i < MaxViewVoices; ++i )
    {
        const Voice      &v = m_voices[static_cast<size_t>( i )];
        const CloudVoice &cv = m_cloud.voice( i );
        const bool        on = v.m_note >= 0 && cv.m_active;
        m_viewHz[static_cast<size_t>( i )].store( on ? cv.m_hz : 0.0f, std::memory_order_relaxed );
        m_viewGain[static_cast<size_t>( i )].store( on ? cv.m_gain : 0.0f, std::memory_order_relaxed );
        m_viewDistance[static_cast<size_t>( i )].store( v.m_distance, std::memory_order_relaxed );
        m_viewTime[static_cast<size_t>( i )].store( cv.m_time, std::memory_order_relaxed );
        m_viewSource[static_cast<size_t>( i )].store( v.m_source == FoundationSource ? 1 : v.m_source,
                                                      std::memory_order_relaxed );
        m_viewNote[static_cast<size_t>( i )].store( on ? v.m_note : -1, std::memory_order_relaxed );
        m_viewPress[static_cast<size_t>( i )].store( v.m_press, std::memory_order_relaxed );
        m_viewSlide[static_cast<size_t>( i )].store( v.m_slide, std::memory_order_relaxed );
        m_viewBend[static_cast<size_t>( i )].store( v.m_bend, std::memory_order_relaxed );
        m_viewTune[static_cast<size_t>( i )].store( v.m_tuneCents, std::memory_order_relaxed );
    }
    for ( size_t gi = 0; gi < m_ghosts.size(); ++gi )
    {   // the ghosts after the voices, as source 3
        const size_t i = m_voices.size() + gi;
        if ( i >= static_cast<size_t>( MaxViewVoices ) ) break;
        const CloudVoice &cv = m_cloud.voice( static_cast<int>( i ) );
        const bool        on = m_ghosts[gi].m_on && cv.m_active;
        m_viewHz[i].store( on ? cv.m_hz : 0.0f, std::memory_order_relaxed );
        m_viewGain[i].store( on ? cv.m_gain : 0.0f, std::memory_order_relaxed );
        m_viewDistance[i].store( cv.m_distance, std::memory_order_relaxed );
        m_viewTime[i].store( cv.m_time, std::memory_order_relaxed );
        m_viewSource[i].store( 3, std::memory_order_relaxed );
        m_viewNote[i].store( -1, std::memory_order_relaxed );
    }
    for ( size_t k = 0; k < 3; ++k )
    {   // the phantom layer, the dream and the halo, drawn as the memory is, with no key of their own
        const size_t i = m_voices.size() + m_ghosts.size() + k;
        if ( i >= static_cast<size_t>( MaxViewVoices ) ) break;
        const CloudVoice &cv = m_cloud.voice( static_cast<int>( i ) );
        m_viewHz[i].store( 0.0f, std::memory_order_relaxed );
        m_viewGain[i].store( cv.m_active ? cv.m_gain : 0.0f, std::memory_order_relaxed );
        m_viewDistance[i].store( cv.m_distance, std::memory_order_relaxed );
        m_viewTime[i].store( cv.m_time, std::memory_order_relaxed );
        m_viewSource[i].store( 3, std::memory_order_relaxed );
        m_viewNote[i].store( -1, std::memory_order_relaxed );
    }

    // 4. The modal hall's settings: it runs on the GPU in the same graph, hearing the voices' block
    //    there. The level is applied here, so the hall rests (on = 0) only once the level has been
    //    at 0 for a whole block; its pitches are the scale's, from its lowest mode to 16 kHz.
    {
        ModalHall  &mh = m_cloud.modal();
        const float level = cur( ParamIdE::MODAL_LEVEL );
        mh.m_on = ( level > 0.0f || m_prevModalLevel > 0.0f ) ? 1 : 0;
        mh.m_freeze = cur( ParamIdE::MODAL_FREEZE ) >= 0.5f ? 1 : 0;
        mh.m_decay = cur( ParamIdE::MODAL_DECAY );
        mh.m_damp = cur( ParamIdE::MODAL_DAMP );
        mh.m_tune = cur( ParamIdE::MODAL_TUNE );
        mh.m_spread = cur( ParamIdE::MODAL_SPREAD );
        mh.m_drift = cur( ParamIdE::MODAL_DRIFT );
        mh.m_width = cur( ParamIdE::MODAL_WIDTH );
        mh.m_low = cur( ParamIdE::MODAL_LOW );
        mh.m_time = m_modalTime;
        mh.m_breath = cur( ParamIdE::MODAL_BREATH );
        // The painted hall reads the landscape at its own clock, wrapped at the landscape's loop.
        mh.m_paint = cur( ParamIdE::MODAL_PAINT );
        mh.m_paintNorm = 1.0f / std::max( 0.125f, m_landRms );
        m_paintTime += dtf;
        if ( m_landLoop > 0.0f && m_paintTime >= m_landLoop ) m_paintTime -= m_landLoop;
        mh.m_paintTime = m_paintTime;
        int count = 0;
        if ( mh.m_on )
            for ( int note = 0; note < 128 && count < ModalTuned; ++note )
            {
                const double f = frequencyOf( note ) * tideMul;
                if ( f >= mh.m_low && f <= 16000.0 ) mh.m_tuned[count++] = static_cast<float>( f );
            }
        mh.m_tunedCount = count;
        // Its memory (Grow): it listens while Grow or the partials' time asks for it -- the partials'
        // amplitudes gathered over the octaves on the card, the spectrum's density followed over a third of
        // a second, and the memory a step of Memory towards it while anything sounds. Grow at nothing:
        // it neither learns nor changes. It learns with the hall's level at nothing too.
        mh.m_grow = cur( ParamIdE::HALL_GROW );
        const bool bent = cur( ParamIdE::TIME_BEND ) != 0.0f;
        mh.m_listen = ( mh.m_grow != 0.0f || bent || cur( ParamIdE::DREAM ) > 0.0f ) ? 1 : 0;
        mh.m_memoryCoef =
            mh.m_grow != 0.0f ? 1.0f - std::exp( -dtf / std::max( cur( ParamIdE::HALL_MEMORY ), 1.0f ) ) : 0.0f;
        mh.m_densityCoef = mh.m_listen ? 1.0f - std::exp( -dtf / 0.3f ) : 0.0f;
        if ( mh.m_memoryCoef > 0.0f && m_rippleVoices > 0 ) m_hallAge += dt;
        m_modalTime += dtf;
        if ( m_modalTime >= 1000.0f ) m_modalTime -= 1000.0f;
    }

    // The field on the partials and the dispersion of their clocks. Scale: the scale's pitches,
    // ascending; Swarm and Scatter: eight Lorenz systems always wander (so the count can change
    // without a jump), the first `count` are the attractors -- x across seven octaves from 60 Hz,
    // z the weight, negative for Scatter.
    {
        SpectralField &F = m_cloud.field();
        const int      mode =
            std::clamp( static_cast<int>( std::lround( cur( ParamIdE::FIELD_MODE ) ) ), 0, NumFieldModes - 1 );
        F.m_mode = static_cast<FieldModeE>( mode );
        const float pull = cur( ParamIdE::FIELD_STRENGTH ), spring = cur( ParamIdE::FIELD_SPRING );
        F.m_strength = 600.0f * pull * pull;
        F.m_radius = cur( ParamIdE::FIELD_RADIUS );
        F.m_spring = 4.0f * spring * spring;
        F.m_dispersion = cur( ParamIdE::DISPERSION );
        F.m_flockFresh = ( F.m_mode == FieldModeE::FLOCK && !m_flockWas ) ? 1 : 0;
        m_flockWas = F.m_mode == FieldModeE::FLOCK;
        F.m_orbitFresh = ( F.m_mode == FieldModeE::ORBIT && !m_orbitWas ) ? 1 : 0;
        m_orbitWas = F.m_mode == FieldModeE::ORBIT;
        // The Physarum (Phase 28): Pull its agents' speed (20 .. 600 cents a second), Radius how far ahead they sense,
        // Spring how they are drawn back home, Trail how long what they leave lasts; its food the sounding notes'
        // first eight harmonics, each line as strong as its note and weaker up the series.
        {
            Physarum &P = m_cloud.physarum();
            P.m_on = F.m_mode == FieldModeE::PHYSARUM ? 1 : 0;
            P.m_fresh = ( P.m_on && !m_physWas ) ? 1 : 0;
            m_physWas = P.m_on != 0;
            if ( P.m_on )
            {
                const float cellCents = 1200.0f * 9.96578428f / static_cast<float>( PhysW );
                const float dtb = static_cast<float>( dt );
                P.m_share = cur( ParamIdE::PHYSARUM_SHARE );
                P.m_speed = ( 20.0f + 580.0f * pull * pull ) * dtb / cellCents;
                P.m_sensor = std::max( F.m_radius / cellCents, 1.0f );
                P.m_spring = F.m_spring;
                P.m_keep = std::exp( -dtb / std::max( cur( ParamIdE::PHYSARUM_TRAIL ), 0.1f ) );
                P.m_spread = 0.15f;
                int count = 0;
                for ( const Voice &v : m_voices )
                {
                    if ( v.m_note < 0 || v.m_lastGain < 0.01f ) continue;
                    for ( int h = 1; h <= 8 && count < PhysFood; ++h )
                    {
                        const double f = static_cast<double>( v.m_hz ) * h;
                        if ( f < 20.0 || f > 20000.0 ) break;
                        P.m_food[count] = static_cast<float>( std::log2( f / 20.0 ) * PhysW / 9.96578428 );
                        P.m_foodWeight[count] =
                            std::min( 1.0f, 1.5f * v.m_lastGain ) / std::sqrt( static_cast<float>( h ) );
                        ++count;
                    }
                }
                for ( int h = 0; h < 2 && count < PhysFood; ++h )
                    if ( m_handFood[h] >= 0.0f )
                    {   // the headset's hands (Phase 30, stepHeadset)
                        P.m_food[count] = m_handFood[h];
                        P.m_foodWeight[count] = m_handFoodWeight;
                        ++count;
                    }
                P.m_count = count;
            }
        }
        // The Orbit: Pull the wells' depth (0.5 .. 8.5 rad/s), Wander the magnetic field (the circling,
        // 0.5 .. 30 rad/s), Spring the damping -- light, the circles keep their course; heavy, they
        // wander; Radius the cents to a full side, and the orbits' size about half of it.
        F.m_orbitWell = 0.5f + 8.0f * pull;
        F.m_orbitField = 0.5f + 30.0f * cur( ParamIdE::FIELD_RATE );
        F.m_orbitDamp = 0.05f + 1.5f * spring * spring;
        F.m_timeBend = cur( ParamIdE::TIME_BEND );
        const double span = static_cast<double>( cur( ParamIdE::FIELD_RATE ) ) * dt;
        const int    steps = std::max( 1, static_cast<int>( std::ceil( span / 0.002 ) ) );
        const double h = span / steps;
        for ( auto &s : m_lorenz )
            for ( int k = 0; k < steps; ++k )
            {   // the midpoint rule, small steps
                auto d = []( const double *p, double *o )
                {
                    o[0] = 10.0 * ( p[1] - p[0] );
                    o[1] = p[0] * ( 28.0 - p[2] ) - p[1];
                    o[2] = p[0] * p[1] - ( 8.0 / 3.0 ) * p[2];
                };
                double k1[3], mid[3], k2[3];
                d( s, k1 );
                for ( int j = 0; j < 3; ++j ) mid[j] = s[j] + 0.5 * h * k1[j];
                d( mid, k2 );
                for ( int j = 0; j < 3; ++j ) s[j] += h * k2[j];
            }
        if ( F.m_mode == FieldModeE::SCALE || F.m_mode == FieldModeE::ORBIT )
        {
            int count = 0;
            for ( int note = 0; note < 128 && count < FieldPoints; ++note )
            {
                const float lf = static_cast<float>( std::log2( frequencyOf( note ) * tideMul ) );
                if ( count == 0 || lf > F.m_pos[count - 1] + 1.0e-4f ) F.m_pos[count++] = lf;
            }
            F.m_count = count;
        }
        else if ( F.m_mode == FieldModeE::FLOCK )
        {
            // The flock's winds: each cloud blown by one of the eight Lorenz systems' x (cloud h by h mod 8).
            for ( int k = 0; k < FieldAttractors; ++k )
                F.m_pos[k] = std::clamp( static_cast<float>( m_lorenz[k][0] / 22.0 ), -1.0f, 1.0f );
            F.m_count = FieldAttractors;
        }
        else if ( F.m_mode != FieldModeE::OFF )
        {
            const int count =
                std::clamp( static_cast<int>( std::lround( cur( ParamIdE::FIELD_COUNT ) ) ), 1, FieldAttractors );
            const float lo = std::log2( 60.0f ), octaves = std::log2( 8000.0f / 60.0f );
            for ( int k = 0; k < count; ++k )
            {
                const float u = std::clamp( static_cast<float>( ( m_lorenz[k][0] + 22.0 ) / 44.0 ), 0.0f, 1.0f );
                const float w = 0.4f + 0.8f * std::clamp( static_cast<float>( m_lorenz[k][2] / 50.0 ), 0.0f, 1.0f );
                F.m_pos[k] = lo + u * octaves;
                F.m_mass[k] = F.m_mode == FieldModeE::SCATTER ? -w : w;
            }
            F.m_count = count;
        }
        else
        {
            F.m_count = 0;
        }
    }

    // The living landscape: its mode (a change seeds it), its steps (up to two a block: Lenia 2 to 40
    // a second, the Turing field 20 to 750, by Life Rate), its character (Lenia's growth; the
    // Turing field along spots, mitosis, coral, worms, mazes in Pearson's map), the make-up from
    // the field's measured power, and a new seeding when it has died or filled up.
    {
        LifeField &L = m_cloud.life();
        const int  mode =
            std::clamp( static_cast<int>( std::lround( cur( ParamIdE::LIFE_MODE ) ) ), 0, NumLifeModes - 1 );
        L.m_mode = static_cast<LifeModeE>( mode );
        m_lifeQuiet += dtf;
        if ( mode != m_lifeModeWas && mode != 0 )
        {
            L.m_reseed = 1;
            ++L.m_salt;
            m_lifeAcc = 0.0f;
            m_lifeQuiet = 0.0f;
            m_lifeMakeup = 1.0f;
        }
        m_lifeModeWas = mode;
        if ( mode != 0 )
        {
            const float rate = cur( ParamIdE::LIFE_RATE );
            if ( lifeIsPicture( L.m_mode ) )
            {
                // The animated pictures: a step every block; the flow at Life Rate (squared: slow at
                // the bottom), the turn at Turn, Rain's drops by Life Rate too.
                L.m_steps = 1;
                m_pictureTime += ( 0.01 + 0.5 * static_cast<double>( rate ) * rate ) * dt;
                m_pictureAngle +=
                    2.0 * 3.14159265358979 * static_cast<double>( cur( ParamIdE::LIFE_TURN ) ) / 60.0 * dt;
                m_pictureSeconds += dt;
                m_pictureAngle =
                    std::remainder( m_pictureAngle, 2.0 * 3.14159265358979 * 60.0 );   // whole turns away, kept small
                L.m_time = static_cast<float>( std::fmod( m_pictureTime, 4096.0 ) );
                L.m_angle = static_cast<float>( m_pictureAngle );
                L.m_seconds = static_cast<float>( std::fmod( m_pictureSeconds, 600000.0 ) );
                L.m_sides = std::round( cur( ParamIdE::LIFE_SIDES ) );
                L.m_character = std::clamp( cur( ParamIdE::LIFE_CHARACTER ), 0.0f, 1.0f );
                L.m_drops = static_cast<float>( ( 0.5 + 30.0 * static_cast<double>( rate ) * rate ) * dt /
                                                ( LifeW * LifeH ) );   // drops a second, over every cell
                L.m_turnStep = static_cast<float>( 2.0 * 3.14159265358979 *
                                                   static_cast<double>( cur( ParamIdE::LIFE_TURN ) ) / 60.0 * dt );
            }
            else if ( L.m_mode == LifeModeE::ISLANDS )
            {
                // The separation: the field drawn every block, Life Rate's substeps of it (squared: slow
                // at the bottom), drifting across.
                L.m_steps = 1;
                L.m_substeps = std::clamp( 1 + static_cast<int>( std::lround( 15.0f * rate * rate ) ), 1, 16 );
                m_islandDrift = std::fmod( m_islandDrift + ( 0.05 + 0.6 * static_cast<double>( rate ) ) * dt,
                                           static_cast<double>( IsleW ) );
                L.m_drift = static_cast<float>( m_islandDrift );
            }
            else
            {
                // Lenia 2 to 40 steps a second, the Fluid 30 to 430, the Turing field 20 to 750.
                const float perSecond =
                    L.m_mode == LifeModeE::LENIA
                        ? 2.0f + 38.0f * rate
                        : ( L.m_mode == LifeModeE::FLUID ? 30.0f + 400.0f * rate : 20.0f + 730.0f * rate );
                m_lifeAcc += perSecond * dtf;
                const int steps = std::min( 2, static_cast<int>( m_lifeAcc ) );
                m_lifeAcc = std::min( m_lifeAcc - static_cast<float>( steps ), 2.0f );
                L.m_steps = steps;
                L.m_fade = std::exp( -1.0f / ( 25.0f * perSecond ) );   // the Fluid's ink lasts about 25 s
            }
            m_lifeSeconds += dt;
            const float ch = std::clamp( cur( ParamIdE::LIFE_CHARACTER ), 0.0f, 1.0f );
            L.m_lenMu = 0.12f + 0.08f * ch;
            L.m_lenSigma = 0.012f + 0.012f * ch;
            static const float F[5] = { 0.035f, 0.0367f, 0.0545f, 0.078f, 0.029f },
                               K[5] = { 0.065f, 0.0649f, 0.062f, 0.061f, 0.057f };
            const float        at = ch * 4.0f;
            const int          seg = std::min( 3, static_cast<int>( at ) );
            const float        fr = at - static_cast<float>( seg );
            L.m_feed = F[seg] + ( F[seg + 1] - F[seg] ) * fr;
            L.m_kill = K[seg] + ( K[seg + 1] - K[seg] ) * fr;
            // The Fluid: Character from syrup to turbulence. The Islands: from a fine labyrinth to a
            // few large islands of the minority (the mean moved off the middle, Oono and Puri's term
            // weakened so the domains keep growing).
            L.m_tau = 0.56f - 0.055f * ch;
            L.m_mean = -0.4f * ch;
            L.m_eps = 0.0015f * ( 1.0f - ch ) + 0.00005f;
            // The Fluid's jets: the eight loudest notes, each blowing at its frequency and at its
            // octave, leaning slowly to and fro.
            std::fill( std::begin( L.m_jetAmt ), std::end( L.m_jetAmt ), 0.0f );
            if ( L.m_mode == LifeModeE::FLUID )
            {
                std::array<int, 64> loud{};
                int                 blowing = 0;
                for ( int i = 0; i < static_cast<int>( m_voices.size() ) && blowing < static_cast<int>( loud.size() );
                      ++i )
                {
                    const Voice &v = m_voices[static_cast<size_t>( i )];
                    if ( v.m_note >= 0 && v.m_lastGain > 0.001f && v.m_hz > 0.0 )
                        loud[static_cast<size_t>( blowing++ )] = i;
                }
                std::stable_sort( loud.begin(), loud.begin() + blowing,
                                  [this]( int a, int b )
                                  {
                                      return m_voices[static_cast<size_t>( a )].m_lastGain >
                                             m_voices[static_cast<size_t>( b )].m_lastGain;
                                  } );
                const int count = std::min( blowing, LifeJets / 2 );
                for ( int j = 0; j < count; ++j )
                {
                    const Voice &v = m_voices[static_cast<size_t>( loud[j] )];
                    const float  across = L.m_zoom * static_cast<float>( LifeW ) / std::log2( 1000.0f );
                    const float  x = std::log2( std::max( static_cast<float>( v.m_hz ), 20.0f ) / 20.0f ) * across;
                    const float  amt = std::min( 1.0f, 1.5f * v.m_lastGain );
                    const float  sway = 0.35f * static_cast<float>( std::sin( 0.13 * m_lifeSeconds + 1.7 * v.m_note ) );
                    L.m_jetX[2 * j] = x;
                    L.m_jetAmt[2 * j] = amt;
                    L.m_jetSway[2 * j] = sway;
                    L.m_jetX[2 * j + 1] = x + across;   // the octave
                    L.m_jetAmt[2 * j + 1] = 0.6f * amt;
                    L.m_jetSway[2 * j + 1] = -sway;
                }
            }
            L.m_depth = cur( ParamIdE::LIFE_DEPTH );
            L.m_zoom = cur( ParamIdE::LIFE_ZOOM );
            float mean = 0.0f, meanSq = 0.0f;
            m_cloud.lifeStats( mean, meanSq );
            // The make-up m that keeps the partials' power: E[(1 - d + d v m)^2] = 1, a quadratic in m
            // (one over the RMS is right only at full depth; below it the steady share is mixed in).
            {
                const double d = std::clamp( static_cast<double>( L.m_depth ), 0.0, 1.0 ), e = mean,
                             q = std::max( static_cast<double>( meanSq ), 1.0 / 4096.0 );
                const double A = d * d * q, B = 2.0 * ( 1.0 - d ) * d * e, C = ( 1.0 - d ) * ( 1.0 - d ) - 1.0;
                const double m =
                    A > 1.0e-12 ? ( -B + std::sqrt( std::max( B * B - 4.0 * A * C, 0.0 ) ) ) / ( 2.0 * A ) : 1.0;
                m_lifeMakeup += ( 1.0f - std::exp( -dtf / 0.5f ) ) *
                                ( static_cast<float>( std::clamp( m, 0.0, 8.0 ) ) - m_lifeMakeup );
                L.m_makeup = m_lifeMakeup;
            }
            const bool dead = !lifeIsPicture( L.m_mode ) && mean < 0.002f,
                       full = L.m_mode == LifeModeE::LENIA && mean > 0.85f;
            if ( ( dead || full ) && m_lifeQuiet > 2.0f && L.m_reseed == 0 )
            {
                L.m_reseed = 1;
                ++L.m_salt;
                m_lifeQuiet = 0.0f;
            }
        }
        else
        {
            L.m_steps = 0;
        }
    }

    stepRipple( dtf );

    // Turbulence: drive 7 T^2 a second at full strength, by the cube of how loud the instrument sounds; the
    // cascade keeps running for twice its settling time after it is turned off, so the harmonics
    // come back to rest before it stops.
    {
        const float T = cur( ParamIdE::TURBULENCE );
        Turbulence &tb = m_cloud.turbulence();
        tb.m_settle = cur( ParamIdE::TURBULENCE_TIME );
        const float loud = 0.25f + 0.75f * m_mods.value( ModSourceE::LOUDNESS );
        m_turbDrive = 7.0f * T * T * loud * loud * loud;
        m_turbAfter = T > 0.0f ? 2.0f * tb.m_settle : std::max( 0.0f, m_turbAfter - dtf );
        tb.m_on = ( T > 0.0f || m_turbAfter > 0.0f ) ? 1.0f : 0.0f;
    }

    // The spectral comb: its phase runs on at the sweep (kept within a notch), and the make-up is
    // exact -- the raised cosine's mean square is (1 - d/2)^2 + d^2/8 over the notches.
    {
        SpectralComb &K = m_cloud.comb();
        K.m_depth = cur( ParamIdE::COMB_DEPTH );
        K.m_density = cur( ParamIdE::COMB_DENSITY );
        K.m_stereo = cur( ParamIdE::COMB_STEREO );
        m_combPhase += static_cast<double>( cur( ParamIdE::COMB_SWEEP ) ) * dt;
        m_combPhase -= std::floor( m_combPhase );
        K.m_phase = static_cast<float>( m_combPhase );
        m_combPhaseView.store( K.m_phase, std::memory_order_relaxed );
        const float d = K.m_depth;
        K.m_makeup =
            1.0f / std::sqrt( std::max( ( 1.0f - 0.5f * d ) * ( 1.0f - 0.5f * d ) + 0.125f * d * d, 1.0e-6f ) );
    }

    // The reservoir: stepped only while a row reads it; fed with what the instrument hears and
    // plays (the sources as the last block left them), each standardised against its last half
    // minute -- the reservoir hears how the music departs from its recent self.
    {
        Reservoir &R = m_cloud.reservoir();
        R.m_on = 0;
        for ( int r = 0; r < ModRows; ++r )
        {
            const ModRow row = unpackModRow( m_modRows[static_cast<size_t>( r )].load( std::memory_order_relaxed ) );
            auto isRes = []( ModSourceE s ) { return s >= ModSourceE::RESERVOIR1 && s <= ModSourceE::RESERVOIR4; };
            if ( row.m_target >= 0 && row.m_depth != 0.0f && ( isRes( row.m_source ) || isRes( row.m_via ) ) )
                R.m_on = 1;
        }
        if ( m_sourcesWatched.load( std::memory_order_relaxed ) ) R.m_on = 1;
        R.m_chaos = cur( ParamIdE::RESERVOIR_CHAOS );
        R.m_leak = 1.0f - std::exp( -dtf / std::max( cur( ParamIdE::RESERVOIR_MEMORY ), 0.05f ) );
        R.m_input = cur( ParamIdE::RESERVOIR_INPUT );
        const ModSourceE feed[ReservoirInputs] = { ModSourceE::LOUDNESS,  ModSourceE::CENTROID, ModSourceE::FLUX,
                                                   ModSourceE::ROUGHNESS, ModSourceE::BEAT,     ModSourceE::NOTE,
                                                   ModSourceE::CASCADE,   ModSourceE::LENIA1 };
        const float      a = 1.0f - std::exp( -dtf / 30.0f );
        for ( int k = 0; k < ReservoirInputs; ++k )
        {
            const float v = m_mods.value( feed[k] );
            if ( !m_resStarted )
            {
                m_resMean[k] = v;
                m_resVar[k] = 0.0f;
            }
            const float d = v - m_resMean[k];
            m_resMean[k] += a * d;
            m_resVar[k] = ( 1.0f - a ) * ( m_resVar[k] + a * d * d );
            R.m_u[k] = std::clamp( 0.5f * d / ( std::sqrt( m_resVar[k] ) + 0.05f ), -2.0f, 2.0f );
        }
        m_resStarted = true;
    }

    // Partial sync: the coupling in units of the oscillators' own spread of rates -- 2.5 times it at
    // full Coupling, well past the critical strength (for a uniform spread about 1.3 times it).
    {
        PartialSync &S = m_cloud.sync();
        S.m_depth = cur( ParamIdE::SYNC_DEPTH );
        S.m_rate = cur( ParamIdE::SYNC_RATE );
        S.m_spread = cur( ParamIdE::SYNC_SPREAD );
        S.m_mode = static_cast<SyncModeE>(
            std::clamp( static_cast<int>( std::lround( cur( ParamIdE::SYNC_MODE ) ) ), 0, NumSyncModes - 1 ) );
        S.m_coupling = cur( ParamIdE::SYNC_COUPLING ) * 2.5f * TwoPi * S.m_rate * ( S.m_spread + 0.05f );
        // Chimera and Swarm live on their own time, a couple of the rate's turns: in units of it.
        if ( S.m_mode != SyncModeE::GLOBAL )
            S.m_coupling = cur( ParamIdE::SYNC_COUPLING ) * 2.0f * TwoPi * std::max( S.m_rate, 0.05f );
        S.m_swarm = cur( ParamIdE::SYNC_SWARM ) * 2.0f * TwoPi * std::max( S.m_rate, 0.05f );
        S.m_reach = cur( ParamIdE::SYNC_REACH );
        S.m_lag = 1.25f + 0.3f * std::clamp( cur( ParamIdE::SYNC_LAG ), 0.0f,
                                             1.0f );   // 1.39 (Abrams and Strogatz) at about 0.47
    }

    // The binaural cloud (its orbit turns the whole of it) and the spectral echo (its lines
    // cleared when it is switched on; its delay less the 1024 - G samples its window adds).
    {
        Binaural &b = m_cloud.binaural();
        b.m_on = cur( ParamIdE::BINAURAL ) >= 0.5f ? 1 : 0;
        b.m_span = cur( ParamIdE::BINAURAL_SPAN );
        b.m_height = cur( ParamIdE::BINAURAL_HEIGHT );
        m_binAngle += 6.2831853f * cur( ParamIdE::BINAURAL_ORBIT ) * dtf;
        if ( m_binAngle > 3.14159265f ) m_binAngle -= 6.2831853f;
        if ( m_binAngle < -3.14159265f ) m_binAngle += 6.2831853f;
        b.m_angle = m_binAngle - m_headYaw;   // the head's turn held off (Phase 30, the headset)
        SpectralEcho &se = m_cloud.spectralEcho();
        const bool    on = cur( ParamIdE::SPEC_MIX ) > 0.0f ||
                           cur( ParamIdE::SPEC_TO_FAR ) * cur( ParamIdE::SPEC_MIX ) > 0.0f || m_prevSpecMix > 0.0f;
        if ( on && !m_specWasOn ) se.m_clear = 1;
        m_specWasOn = on;
        se.m_on = on ? 1 : 0;
        se.m_frames = std::max( 1.0f, ( cur( ParamIdE::SPEC_TIME ) * static_cast<float>( m_cfg.m_sampleRate ) -
                                        static_cast<float>( 1024 - G ) ) /
                                          static_cast<float>( G ) );
        se.m_tilt = cur( ParamIdE::SPEC_TILT );
        se.m_scatter = cur( ParamIdE::SPEC_SCATTER );
        se.m_feedback = cur( ParamIdE::SPEC_FEEDBACK );
        se.m_damp = cur( ParamIdE::SPEC_DAMP );

        // The melt: its time to -60 dB as the held energy's share kept a frame, the spread as a
        // diffusion (after a second the standard deviation is Spread octaves), the bands' pull to
        // their centres and their detuning growing with it.
        Melt      &me = m_cloud.melt();
        const bool meltOn = cur( ParamIdE::MELT_MIX ) > 0.0f || m_prevMeltMix > 0.0f;
        if ( meltOn && !m_meltWasOn ) me.m_clear = 1;
        m_meltWasOn = meltOn;
        me.m_on = meltOn ? 1 : 0;
        const float hopS = static_cast<float>( MeltHop ) / static_cast<float>( m_cfg.m_sampleRate );
        const float spread = cur( ParamIdE::MELT_SPREAD );
        me.m_keep = std::pow( 10.0f, -6.0f * hopS / std::max( cur( ParamIdE::MELT_TIME ), 0.1f ) );
        me.m_widen = spread * 0.693147f * std::sqrt( hopS );
        me.m_pull = 1.0f - std::exp( -2.0f * spread * hopS );
        me.m_detune = ( 0.1f + 0.4f * std::min( spread, 1.0f ) ) * 6.2831853f / static_cast<float>( MeltN );
    }

    // 5. The GPU: near and far, left and right, the modal hall and the spectral echo.
    // The picture's snapshot about thirty times a second, brought down with the block.
    if ( --m_snapCountdown <= 0 )
    {
        m_snapCountdown = std::max( 1, static_cast<int>( std::lround( 0.033 * m_cfg.m_sampleRate / G ) ) );
        m_cloud.requestSnapshot();
    }
    {   // a hall loaded (or a new one): up with this block, when the message thread is not just writing it
        std::unique_lock<std::mutex> lock( m_snapLock, std::try_to_lock );
        if ( lock.owns_lock() && m_hallPendingSet )
        {
            m_cloud.setHallMemory( m_hallPending.data() );
            m_hallAge = m_hallPendingAge;
            m_snapHall = m_hallPending;
            m_snapHallAge = m_hallAge;
            m_snapHallValid = true;
            m_hallPendingSet = false;
        }
    }
    LARGE_INTEGER qpcLaunch, qpcBack;
    QueryPerformanceCounter( &qpcLaunch );
    m_cloud.launch();
    const float *q = m_cloud.wait();
    m_chimeStrikes.fetch_add( m_cloud.chimeStrikes(), std::memory_order_relaxed );
    if ( m_chorusWas )
    {   // the night chorus's calls, counted (Phase 31)
        float chorus[4];
        m_cloud.chorusStats( chorus );
        m_chorusCalls += static_cast<long long>( chorus[0] );
    }
    QueryPerformanceCounter( &qpcBack );
    {
        const float *dots = nullptr;
        int          per = 0;
        if ( m_cloud.takeSnapshot( dots, per ) )
        {
            std::unique_lock<std::mutex> lock( m_snapLock,
                                               std::try_to_lock );   // the picture is reading: this one is skipped
            if ( lock.owns_lock() )
            {
                const size_t count = static_cast<size_t>( m_cloud.config().m_voices ) * static_cast<size_t>( per ) * 2;
                if ( m_snap.size() != count ) m_snap.resize( count );   // once: the size stays
                std::memcpy( m_snap.data(), dots, count * sizeof( float ) );
                m_snapPer = per;
                if ( const float *cells = m_cloud.lifePicture() )
                    m_snapLife.assign( cells, cells + LifeW * LifeH );
                else
                    m_snapLife.clear();
                if ( m_cloud.ripple().m_depth > 0.0f )
                {   // the ripple's string (once sized: no allocation after)
                    m_snapRipple.resize( static_cast<size_t>( RippleCells ) );
                    std::copy( std::begin( m_rippleNow ), std::end( m_rippleNow ), m_snapRipple.begin() );
                }
                else
                {
                    m_snapRipple.clear();
                }
                if ( m_cloud.hallMemory( m_snapHall.data() ) )
                {   // and the hall's memory
                    m_snapHallValid = true;
                    m_snapHallAge = m_hallAge;
                }
                ++m_snapSeq;
            }
        }
    }
    m_gpuMs.store( m_cloud.lastGpuMs(), std::memory_order_relaxed );
    for ( int s = 0; s < G; ++s )
    {
        const float *f = q + static_cast<size_t>( CloudOutChannels ) * s;
        const size_t k = static_cast<size_t>( s );
        m_nearL[k] = f[0];
        m_nearR[k] = f[1];
        m_farL[k] = f[2] * m_blk.m_farLevel;
        m_farR[k] = f[3] * m_blk.m_farLevel;
        m_modalL[k] = f[4];
        m_modalR[k] = f[5];
        m_specL[k] = f[6];
        m_specR[k] = f[7];
        m_meltL[k] = f[8];
        m_meltR[k] = f[9];
        m_nearL[k] += f[10] + f[16] + f[24];   // the chime field, the weather and the night, near and far, as the
        m_nearR[k] += f[11] + f[17] + f[25];   // voices' planes
        m_farL[k] += ( f[12] + f[18] + f[26] ) * m_blk.m_farLevel;
        m_farR[k] += ( f[13] + f[19] + f[27] ) * m_blk.m_farLevel;
        m_bloomL[k] = f[20];   // the bloom (Phase 28), its level applied on the card
        m_bloomR[k] = f[21];
        m_roomL[k] = f[22];   // the wave room (Phase 29), likewise
        m_roomR[k] = f[23];
        m_phantomL[k] = f[14];   // the phantom layer, heard dry at the end
        m_phantomR[k] = f[15];
    }
    // A level from the last block's to this one's, across the block (no zipper when it is modulated).
    auto ramp = [G]( float from, float to, int s )
    { return from + ( to - from ) * static_cast<float>( s + 1 ) / static_cast<float>( G ); };

    // 6. The near plane, as Noctuary's: two echoes, the second in series after the first, receding
    //    partly into the far plane; a small reverb; the early reflections of a room, from the dry voices.
    std::copy( m_nearL.begin(), m_nearL.end(), m_dryL.begin() );
    std::copy( m_nearR.begin(), m_nearR.end(), m_dryR.begin() );
    m_delay.set( cur( ParamIdE::DELAY_TIME_L ), cur( ParamIdE::DELAY_TIME_R ), cur( ParamIdE::DELAY_FEEDBACK ),
                 cur( ParamIdE::DELAY_CROSS ), cur( ParamIdE::DELAY_DAMP ), cur( ParamIdE::DELAY_ABSORB ) );
    m_delay.setDuck( cur( ParamIdE::DELAY_DUCK ) );
    m_delay.process( m_nearL.data(), m_nearR.data(), m_wetL.data(), m_wetR.data(), G );
    {
        const float m1 = cur( ParamIdE::DELAY_MIX ), t1 = cur( ParamIdE::DELAY_TO_FAR );
        for ( int s = 0; s < G; ++s )
        {
            const size_t k = static_cast<size_t>( s );
            const float  m = ramp( m_prevDelayMix, m1, s ), tf = ramp( m_prevDelayToFar, t1, s );
            m_nearL[k] += m_wetL[k] * m;
            m_nearR[k] += m_wetR[k] * m;
            m_farL[k] += m_wetL[k] * tf;
            m_farR[k] += m_wetR[k] * tf;
        }
        m_prevDelayMix = m1;
        m_prevDelayToFar = t1;
    }
    m_delay2.set( cur( ParamIdE::DELAY2_TIME_L ), cur( ParamIdE::DELAY2_TIME_R ), cur( ParamIdE::DELAY2_FEEDBACK ),
                  cur( ParamIdE::DELAY_CROSS ), cur( ParamIdE::DELAY2_DAMP ), cur( ParamIdE::DELAY_ABSORB ) );
    m_delay2.setDuck( cur( ParamIdE::DELAY_DUCK ) );
    m_delay2.process( m_nearL.data(), m_nearR.data(), m_wetL.data(), m_wetR.data(), G );
    {
        const float m2 = cur( ParamIdE::DELAY2_MIX ), t2 = cur( ParamIdE::DELAY2_TO_FAR );
        for ( int s = 0; s < G; ++s )
        {
            const size_t k = static_cast<size_t>( s );
            const float  m = ramp( m_prevDelay2Mix, m2, s ), tf = ramp( m_prevDelay2ToFar, t2, s );
            m_nearL[k] += m_wetL[k] * m;
            m_nearR[k] += m_wetR[k] * m;
            m_farL[k] += m_wetL[k] * tf;
            m_farR[k] += m_wetR[k] * tf;
        }
        m_prevDelay2Mix = m2;
        m_prevDelay2ToFar = t2;
    }
    {   // the spectral echo, from the GPU: onto the near plane and partly into the far
        const float m3 = cur( ParamIdE::SPEC_MIX ), t3 = cur( ParamIdE::SPEC_TO_FAR ) * m3;
        for ( int s = 0; s < G; ++s )
        {
            const size_t k = static_cast<size_t>( s );
            const float  m = ramp( m_prevSpecMix, m3, s ), tf = ramp( m_prevSpecToFar, t3, s );
            m_nearL[k] += m_specL[k] * m;
            m_nearR[k] += m_specR[k] * m;
            m_farL[k] += m_specL[k] * tf;
            m_farR[k] += m_specR[k] * tf;
        }
        m_prevSpecMix = m3;
        m_prevSpecToFar = t3;
    }
    {   // the melt, from the GPU: onto the near plane and partly into the far
        const float m4 = cur( ParamIdE::MELT_MIX ), t4 = cur( ParamIdE::MELT_TO_FAR ) * m4;
        for ( int s = 0; s < G; ++s )
        {
            const size_t k = static_cast<size_t>( s );
            const float  m = ramp( m_prevMeltMix, m4, s ), tf = ramp( m_prevMeltToFar, t4, s );
            m_nearL[k] += m_meltL[k] * m;
            m_nearR[k] += m_meltR[k] * m;
            m_farL[k] += m_meltL[k] * tf;
            m_farR[k] += m_meltR[k] * tf;
        }
        m_prevMeltMix = m4;
        m_prevMeltToFar = t4;
    }
    m_nearRev.setSpace( 0.3f, 20000.0f, 80.0f );
    m_nearRev.set( 0.6f, cur( ParamIdE::NEAR_DECAY ), cur( ParamIdE::NEAR_DAMP ), 5.0f, false,
                   cur( ParamIdE::NEAR_MIX ) );
    m_nearRev.process( m_nearL.data(), m_nearR.data(), G );
    m_early.setRoom( cur( ParamIdE::EARLY_SIZE ), cur( ParamIdE::EARLY_ABSORB ), 1.0f );
    m_early.setSource( 0.0f, m_srcDistance );
    m_early.setLevel( cur( ParamIdE::EARLY_LEVEL ) );
    if ( m_early.active() ) m_early.process( m_dryL.data(), m_dryR.data(), m_nearL.data(), m_nearR.data(), G );

    // 7. The far plane is heard only through the far reverb: the swell of the diffusion first, the
    //    hall stepping aside for the near plane band by band after (Unmask).
    m_diffuser.set( cur( ParamIdE::FAR_DIFFUSE ) );
    m_diffuser.process( m_farL.data(), m_farR.data(), G );
    m_far.setMode( std::clamp( static_cast<int>( std::lround( cur( ParamIdE::FAR_MODE ) ) ), 0, NumFarModes - 1 ) );
    m_far.set( cur( ParamIdE::FAR_SIZE ), cur( ParamIdE::FAR_DECAY ), cur( ParamIdE::FAR_DAMP ),
               cur( ParamIdE::FAR_PRE_DELAY ), cur( ParamIdE::FAR_FREEZE ) >= 0.5f, 1.0f );
    m_far.setSpace( 0.5f, 9000.0f, 60.0f );
    m_far.process( m_farL.data(), m_farR.data(), G );
    m_unmask.set( cur( ParamIdE::FAR_UNMASK ), 0.3f, 0.5f );
    m_unmask.process( m_nearL.data(), m_nearR.data(), m_farL.data(), m_farR.data(), G );

    // 8. Together: the near plane, the far hall and the modal hall.
    {
        const float ml = cur( ParamIdE::MODAL_LEVEL );
        for ( int s = 0; s < G; ++s )
        {
            const size_t k = static_cast<size_t>( s );
            const float  m = ramp( m_prevModalLevel, ml, s );
            m_nearL[k] += m_farL[k] + m_modalL[k] * m + m_bloomL[k] + m_roomL[k];
            m_nearR[k] += m_farR[k] + m_modalR[k] * m + m_bloomR[k] + m_roomR[k];
        }
        m_prevModalLevel = ml;
    }
    // Binaural: the master leaves the two ears alone (no mono bass, no width): both would undo the head.
    if ( cur( ParamIdE::BINAURAL ) >= 0.5f )
        m_midSide.set( 20.0f, 0.0f, 1.0f );
    else
        m_midSide.set( cur( ParamIdE::BASS_MONO ), 0.0f, cur( ParamIdE::STEREO_WIDTH ) );
    m_midSide.process( m_nearL.data(), m_nearR.data(), G );
    // The phantom layer after everything (Phase 27): the turn of phase between the ears is all there is to hear,
    // and the echoes, the rooms and the width would blur it.
    for ( int s = 0; s < G; ++s )
    {
        m_nearL[static_cast<size_t>( s )] += m_phantomL[static_cast<size_t>( s )];
        m_nearR[static_cast<size_t>( s )] += m_phantomR[static_cast<size_t>( s )];
    }
    for ( int s = 0; s < G; ++s )
    {
        const size_t at = static_cast<size_t>( ( pos + s ) & m_ringMask ) * 2;
        m_ring[at] = softClip( m_nearL[static_cast<size_t>( s )] * m_blk.m_master );
        m_ring[at + 1] = softClip( m_nearR[static_cast<size_t>( s )] * m_blk.m_master );
    }
    m_renderPos.store( pos + G, std::memory_order_release );
    LARGE_INTEGER qpcEnd, qpcFreq;
    QueryPerformanceCounter( &qpcEnd );
    QueryPerformanceFrequency( &qpcFreq );
    auto keepWorst = [&]( std::atomic<int> &at, LONGLONG from, LONGLONG to )
    {
        const int us = static_cast<int>( ( to - from ) * 1000000 / qpcFreq.QuadPart );
        int       worst = at.load( std::memory_order_relaxed );
        while ( us > worst && !at.compare_exchange_weak( worst, us, std::memory_order_relaxed ) )
        {
        }
    };
    keepWorst( m_worstBlockUs, qpcStart.QuadPart, qpcEnd.QuadPart );
    keepWorst( m_worstPrepUs, qpcStart.QuadPart, qpcLaunch.QuadPart );
    keepWorst( m_worstWaitUs, qpcLaunch.QuadPart, qpcBack.QuadPart );
    keepWorst( m_worstPostUs, qpcBack.QuadPart, qpcEnd.QuadPart );
}

}   // namespace additiv
