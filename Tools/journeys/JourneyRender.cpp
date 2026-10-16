/**
 * @file JourneyRender.cpp
 * @brief The journeys listed, their lengths planned, and one rendered: the engine offline (as fast as the card allows),
 *        a step's preset glided into as the instrument does it (Engine::loadPreset, or the player's own preset by
 *        Engine::beginMorph and its knobs), the conductor on, the end faded; into a JUCE writer (WAV, FLAC, Ogg Vorbis,
 *        on a background thread) or the MP3 writer.
 * @author Rene Weller
 */
#include "JourneyRender.h"

#include "Mp3Writer.h"
#include "additiv/Engine.h"
#include "additiv/Params.h"
#include "additiv/Presets.h"
#include "additiv/Sun.h"

#include <juce_data_structures/juce_data_structures.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <map>
#include <memory>

namespace journeys
{

namespace
{

constexpr int Block = 4096;   ///< samples a call of the engine
/** @brief Samples a GPU block when not exact: four of the instrument's. Every partial is set once a block and ramped
 *         across it, so the block is the rate of every modulation; a launch and a wait every 128 samples is
 *         what bounds the speed offline (measured 16.10.2026 with the card busy: 128 1.6 .. 2.6x real time, 512
 *         5.1 .. 8.3x, 1024 8.2 .. 11.6x, 2048 12 .. 15.6x). Six journeys at 512 against 128, the same seed: level
 *         within 0.2 dB, third octaves within 0.1 .. 0.3 dB RMS (another seed: 2.5 .. 8.7), the envelope's
 *         fluctuation within 0.5 dB -- only Scintillation's quickest twinkle (in the gaps between partials, -3 dB
 *         there) is smoothed; at 1024 and beyond that grew to -5 .. -8 dB and took sub-bass sidebands. */
constexpr int    FastGpuBlock = 512;
constexpr double SampleRate = 48000.0;   ///< the files' rate

/** @brief One of the player's own presets: its knobs and its matrix. */
struct OwnPreset
{
    std::vector<float> m_values;   ///< every knob (NumParams)
    std::string        m_mod;      ///< its matrix
};

/**
 * @brief The player's Empyrean folder in Documents (Empyrean/...).
 * @return the folder
 */
juce::File empyreanDocuments()
{
    return juce::File::getSpecialLocation( juce::File::userDocumentsDirectory ).getChildFile( "Empyrean" );
}

/**
 * @brief The player's own presets by name, read as the instrument reads them (Documents/Empyrean/Presets and its
 *        subfolders, knobs not named at their defaults, each clamped to its range).
 * @return them
 */
std::map<std::string, OwnPreset> ownPresets()
{
    std::map<std::string, OwnPreset> out;
    for ( const juce::File &f :
          empyreanDocuments().getChildFile( "Presets" ).findChildFiles( juce::File::findFiles, true, "*.empyrean" ) )
    {
        const std::unique_ptr<juce::XmlElement> xml = juce::XmlDocument::parse( f );
        if ( !xml || !xml->hasTagName( "EmpyreanPreset" ) ) continue;
        OwnPreset p;
        p.m_values.resize( static_cast<size_t>( additiv::NumParams ) );
        for ( const additiv::ParamDesc &d : additiv::paramTable() ) p.m_values[static_cast<size_t>( d.m_id )] = d.m_def;
        for ( const juce::String &kv : juce::StringArray::fromTokens( xml->getStringAttribute( "values" ), " ", "" ) )
            if ( const additiv::ParamDesc *d =
                     additiv::findParam( kv.upToFirstOccurrenceOf( "=", false, false ).toRawUTF8() ) )
                p.m_values[static_cast<size_t>( d->m_id )] =
                    std::clamp( kv.fromFirstOccurrenceOf( "=", false, false ).getFloatValue(), d->m_min, d->m_max );
        p.m_mod = xml->getStringAttribute( "mod" ).toStdString();
        out[f.getFileNameWithoutExtension().toStdString()] = std::move( p );
    }
    return out;
}

/**
 * @brief A journey's step applied as the instrument applies it: the bank's preset, or else the player's own by that
 *        name, glided into (at once, the chord filled, when the glide is nothing); the conductor plays on. A name
 *        neither knows changes nothing.
 * @param[in,out] e      the engine
 * @param[in]     name   the step's preset
 * @param[in]     glide  seconds of the glide (0: at once)
 * @param[in]     own    the player's presets
 * @return true when a preset was applied
 */
bool applyStep( additiv::Engine &e, const std::string &name, double glide, const std::map<std::string, OwnPreset> &own )
{
    const int index = additiv::findPreset( name.c_str() );
    if ( index >= 0 )
    {
        e.loadPreset( index, glide );
    }
    else
    {
        const auto it = own.find( name );
        if ( it == own.end() ) return false;
        if ( glide > 0.0 ) e.beginMorph( glide );
        for ( int k = 0; k < additiv::NumParams; ++k )
            if ( !additiv::keptByPresets( static_cast<additiv::ParamIdE>( k ) ) )
                e.setParam( static_cast<additiv::ParamIdE>( k ), it->second.m_values[static_cast<size_t>( k )] );
        e.setModText( it->second.m_mod );
    }
    if ( glide <= 0.0 ) e.requestFill();   // the arriving preset's chord within seconds
    e.setParam( additiv::ParamIdE::BRAIN_ON, 1.0f );
    return true;
}

/** @brief Where the samples go: a JUCE writer behind a background thread, or the MP3 writer. */
class Sink
{
public:
    /**
     * @brief Opens the file.
     * @param[in]  format  the format
     * @param[in]  file    the file
     * @param[out] why     receives why it failed
     * @return true when it is open
     */
    bool open( FormatE format, const juce::File &file, juce::String &why )
    {
        file.deleteFile();
        if ( format == FormatE::MP3 )
        {
            std::string w;
            m_mp3 = std::make_unique<Mp3Writer>();
            if ( !m_mp3->open( file.getFullPathName().toWideCharPointer(), static_cast<int>( SampleRate ), 320, w ) )
            {
                why = w;
                m_mp3.reset();
                return false;
            }
            return true;
        }
        std::unique_ptr<juce::AudioFormat> fmt;
        int                                bits = 24, quality = 0;
        if ( format == FormatE::WAV ) fmt = std::make_unique<juce::WavAudioFormat>();
        if ( format == FormatE::FLAC )
        {
            fmt = std::make_unique<juce::FlacAudioFormat>();
            quality = 5;
        }
        if ( format == FormatE::OGG )
        {
            fmt = std::make_unique<juce::OggVorbisAudioFormat>();
            bits = 16;
            quality = std::max( 0, fmt->getQualityOptions().indexOf( "320 kbps" ) );
        }
        std::unique_ptr<juce::OutputStream> stream = std::make_unique<juce::FileOutputStream>( file );
        if ( !static_cast<juce::FileOutputStream *>( stream.get() )->openedOk() )
        {
            why = "cannot write " + file.getFullPathName();
            return false;
        }
        const auto                               options = juce::AudioFormatWriterOptions{}
                                                               .withSampleRate( SampleRate )
                                                               .withNumChannels( 2 )
                                                               .withBitsPerSample( bits )
                                                               .withQualityOptionIndex( quality );
        std::unique_ptr<juce::AudioFormatWriter> writer = fmt->createWriterFor( stream, options );
        if ( !writer )
        {
            why = juce::String( "no " ) + formatName( format ) + " writer for this file";
            return false;
        }
        m_thread.startThread();
        // encoding beside the rendering: eight seconds of room before write() has to wait
        m_writer = std::make_unique<juce::AudioFormatWriter::ThreadedWriter>( writer.release(), m_thread,
                                                                              static_cast<int>( 8 * SampleRate ) );
        return true;
    }
    /**
     * @brief Appends samples.
     * @param[in] L  left
     * @param[in] R  right
     * @param[in] n  samples
     * @return false when the file refused them
     */
    bool write( const float *L, const float *R, int n )
    {
        if ( m_mp3 ) return m_mp3->write( L, R, n );
        const float *channels[2] = { L, R };
        for ( int tries = 0; !m_writer->write( channels, n ); ++tries )
        {   // the encoder behind: give it a moment
            if ( tries > 20000 ) return false;
            juce::Thread::sleep( 1 );
        }
        return true;
    }
    /**
     * @brief Finishes the file.
     * @return true when it was finished cleanly
     */
    bool close()
    {
        bool ok = true;
        if ( m_mp3 ) ok = m_mp3->close();
        m_mp3.reset();
        m_writer.reset();   // flushes what is left and closes the file
        m_thread.stopThread( 10000 );
        return ok;
    }

private:
    std::unique_ptr<Mp3Writer>                               m_mp3;   ///< the MP3 writer, or null
    juce::TimeSliceThread                                    m_thread{ "Empyrean Journeys writer" };   ///< encodes
    std::unique_ptr<juce::AudioFormatWriter::ThreadedWriter> m_writer;   ///< the JUCE writer behind it, or null
};

/**
 * @brief The night arc's clock for a rendering: 23:00 today, local time, when asked; else now.
 * @param[in] night  as at night
 * @return Unix seconds (0: the engine takes the computer's clock)
 */
double sunClock( bool night )
{
    if ( !night ) return 0.0;
    std::time_t now = std::time( nullptr );
    std::tm     local{};
    localtime_s( &local, &now );
    local.tm_hour = 23;
    local.tm_min = 0;
    local.tm_sec = 0;
    return static_cast<double>( std::mktime( &local ) );
}

/**
 * @brief Where the night arc's sun stands: the place the instrument was given (its settings), else the time zone's.
 * @param[out] latitude   receives degrees north
 * @param[out] longitude  receives degrees east
 */
void place( double &latitude, double &longitude )
{
    juce::PropertiesFile::Options o;
    o.applicationName = "Empyrean";
    o.filenameSuffix = ".frame";
    o.folderName = "Empyrean";
    juce::PropertiesFile settings( o );
    const juce::String   lat = settings.getValue( "night_lat" ), lon = settings.getValue( "night_lon" );
    if ( lat.isNotEmpty() && lon.isNotEmpty() )
    {
        latitude = lat.getDoubleValue();
        longitude = lon.getDoubleValue();
        return;
    }
    additiv::defaultLocation( latitude, longitude );
}

}   // namespace

const char *formatName( FormatE f )
{
    switch ( f )
    {
    case FormatE::WAV: return "WAV (24 bit)";
    case FormatE::FLAC: return "FLAC (24 bit, lossless)";
    case FormatE::OGG: return "Ogg Vorbis (320 kbit/s)";
    case FormatE::MP3: return "MP3 (320 kbit/s)";
    default: return "?";
    }
}

const char *formatExtension( FormatE f )
{
    switch ( f )
    {
    case FormatE::WAV: return ".wav";
    case FormatE::FLAC: return ".flac";
    case FormatE::OGG: return ".ogg";
    case FormatE::MP3: return ".mp3";
    default: return ".wav";
    }
}

std::vector<JourneyItem> listJourneys()
{
    std::vector<JourneyItem> out;
    auto                     add = [&out]( additiv::Journey &&j, bool own, const juce::String &fallback )
    {
        JourneyItem item;
        item.m_name = j.m_name.empty() ? fallback : juce::String( j.m_name );
        item.m_own = own;
        item.m_kind = own                                       ? "Yours"
                      : item.m_name.startsWith( "Whole Night" ) ? "Whole night"
                      : item.m_name.endsWith( " - Night" )      ? "Night"
                                                                : "Journey";
        for ( const additiv::JourneyStep &st : j.m_steps )
        {
            item.m_shortest += st.m_dwellLo;
            item.m_longest += st.m_dwellHi;
        }
        item.m_mean = j.meanLength();
        item.m_journey = std::move( j );
        out.push_back( std::move( item ) );
    };
    for ( int i = 0; i < additiv::numBuiltinJourneys(); ++i )
    {
        additiv::Journey j;
        if ( j.parse( additiv::builtinJourneyText( i ) ) && !j.m_steps.empty() ) add( std::move( j ), false, {} );
    }
    juce::Array<juce::File> files =
        empyreanDocuments().getChildFile( "Journeys" ).findChildFiles( juce::File::findFiles, false, "*.journey" );
    files.sort();
    for ( const juce::File &f : files )
    {
        additiv::Journey j;
        if ( j.load( f.getFullPathName().toRawUTF8() ) && !j.m_steps.empty() )
            add( std::move( j ), true, f.getFileNameWithoutExtension() );
    }
    return out;
}

double planLength( const additiv::Journey &journey, std::uint64_t seed, int passes )
{
    const int steps = static_cast<int>( journey.m_steps.size() ) * ( journey.m_cyclic ? std::max( passes, 1 ) : 1 );
    additiv::JourneyPlayer p;
    p.start( journey, seed );
    additiv::JourneyStep st;
    double               fade = 0.0, total = 0.0;
    if ( !p.advance( 0.0, st, fade ) ) return 0.0;
    for ( int k = 0; k < steps; ++k )
    {   // each step's dwell as the player draws it (its glide first), the next begun when it is over
        total += p.remaining();
        if ( k + 1 < steps && !p.advance( p.remaining(), st, fade ) ) break;
    }
    return total;
}

juce::String clockText( double seconds )
{
    const long long s = static_cast<long long>( std::llround( std::max( seconds, 0.0 ) ) );
    const long long h = s / 3600, m = ( s / 60 ) % 60, r = s % 60;
    return h > 0 ? juce::String::formatted( "%lld:%02lld:%02lld", h, m, r )
                 : juce::String::formatted( "%lld:%02lld", m, r );
}

bool Renderer::render( const RenderJob &job, const std::function<bool()> &cancelled, juce::String &why )
{
    const additiv::Journey &journey = job.m_item.m_journey;
    const double            planned = planLength( journey, job.m_seed, job.m_passes );
    const double            total = job.m_limit > 0.0 ? std::min( job.m_limit, planned ) : planned;
    m_done.store( 0.0 );
    m_total.store( total );
    m_speed.store( 0.0 );
    m_step.store( -1 );
    if ( total <= 0.0 )
    {
        why = "the journey has no steps";
        return false;
    }
    const std::map<std::string, OwnPreset> own = ownPresets();
    // the engine on the heap: it is large, and this runs on a thread with a small stack
    auto e = std::make_unique<additiv::Engine>();
    e->setRealtime( false );
    additiv::JourneyPlayer player;
    player.start( journey, job.m_seed );
    additiv::JourneyStep st;
    double               glide = 0.0;
    if ( player.advance( 0.0, st, glide ) ) applyStep( *e, st.m_preset, 0.0, own );
    m_step.store( player.step() );
    additiv::EngineConfig c;
    c.m_sampleRate = SampleRate;
    c.m_maxHostBlock = Block;
    c.m_realtimeGpuClass = false;
    c.m_gpuBlock = job.m_exact ? 128 : FastGpuBlock;
    c.m_seed = job.m_seed;
    c.m_sunClock = sunClock( job.m_night );
    std::string error;
    if ( !e->prepare( c, &error ) )
    {
        why = error;
        return false;
    }
    double latitude = 52.52, longitude = 13.40;
    place( latitude, longitude );
    e->setLocation( latitude, longitude );
    e->requestFill();
    Sink sink;
    if ( !sink.open( job.m_format, job.m_file, why ) ) return false;
    const long long frames = static_cast<long long>( std::llround( total * SampleRate ) );
    const long long fadeFrames =
        static_cast<long long>( std::max( 0.0, std::min( job.m_fade, total * 0.5 ) ) * SampleRate );
    std::vector<float> L( Block ), R( Block );
    const auto         t0 = std::chrono::steady_clock::now();
    long long          done = 0;
    bool               ok = true;
    while ( done < frames )
    {
        if ( cancelled() )
        {
            why = "cancelled";
            ok = false;
            break;
        }
        const int n = static_cast<int>( std::min<long long>( Block, frames - done ) );
        if ( player.advance( n / SampleRate, st, glide ) && done + n < frames )
        {   // a step begins: its preset glided into, as the instrument does on a journey (not the one after the end)
            applyStep( *e, st.m_preset, std::max( glide, 0.5 ), own );
            m_step.store( player.step() );
        }
        e->process( L.data(), R.data(), n );
        if ( done + n > frames - fadeFrames )
            for ( int i = 0; i < n; ++i )
            {   // the end faded out on a raised cosine
                const double u = std::clamp(
                    static_cast<double>( frames - ( done + i ) ) / std::max<long long>( fadeFrames, 1 ), 0.0, 1.0 );
                const float g = static_cast<float>( 0.5 - 0.5 * std::cos( 3.14159265358979 * u ) );
                L[static_cast<size_t>( i )] *= g;
                R[static_cast<size_t>( i )] *= g;
            }
        if ( !sink.write( L.data(), R.data(), n ) )
        {
            why = "the file could not be written (is the disk full?)";
            ok = false;
            break;
        }
        done += n;
        const double elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
        m_done.store( done / SampleRate );
        m_speed.store( elapsed > 0.0 ? done / SampleRate / elapsed : 0.0 );
    }
    if ( !sink.close() && ok )
    {
        why = "the file could not be finished";
        ok = false;
    }
    return ok;
}

int selfTest( double seconds )
{
    std::vector<JourneyItem> items = listJourneys();
    if ( items.empty() )
    {
        std::printf( "journeys: no journey\n" );
        return 1;
    }
    const juce::File dir =
        juce::File::getSpecialLocation( juce::File::tempDirectory ).getChildFile( "EmpyreanJourneysTest" );
    dir.deleteRecursively();
    dir.createDirectory();
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    int failures = 0;
    std::printf( "journeys: %zu to choose; %s, %s .. %s a pass\n", items.size(), items[0].m_name.toRawUTF8(),
                 clockText( items[0].m_shortest ).toRawUTF8(), clockText( items[0].m_longest ).toRawUTF8() );
    for ( int f = 0; f < static_cast<int>( FormatE::COUNT ); ++f )
    {
        RenderJob job;
        job.m_item = items[0];
        job.m_format = static_cast<FormatE>( f );
        job.m_limit = seconds;
        job.m_fade = 1.0;
        job.m_file = dir.getChildFile( juce::String( "test" ) + formatExtension( job.m_format ) );
        Renderer     r;
        juce::String why;
        const bool   rendered = r.render( job, [] { return false; }, why );
        double       length = 0.0, db = -200.0;
        if ( rendered )
            if ( std::unique_ptr<juce::AudioFormatReader> reader{ formats.createReaderFor( job.m_file ) } )
            {
                length = static_cast<double>( reader->lengthInSamples ) / reader->sampleRate;
                juce::AudioBuffer<float> buf( 2, static_cast<int>( reader->lengthInSamples ) );
                reader->read( &buf, 0, buf.getNumSamples(), 0, true, true );
                const double rms = 0.5 * ( buf.getRMSLevel( 0, 0, buf.getNumSamples() ) +
                                           buf.getRMSLevel( 1, 0, buf.getNumSamples() ) );
                db = 20.0 * std::log10( std::max( rms, 1e-10 ) );
            }
        const bool ok = rendered && std::abs( length - seconds ) < 0.5 && db > -60.0;
        if ( !ok ) ++failures;
        std::printf( "  %s  %-26s %5.2f s, %6.1f dBFS, %.1fx real time%s%s\n", ok ? "ok  " : "FAIL",
                     formatName( job.m_format ), length, db, r.m_speed.load(), why.isEmpty() ? "" : ": ",
                     why.toRawUTF8() );
    }
    dir.deleteRecursively();
    std::printf( failures == 0 ? "journeys passed\n" : "journeys: %d FAILED\n", failures );
    return failures == 0 ? 0 : 1;
}

}   // namespace journeys
