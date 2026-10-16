/**
 * @file Source.cpp
 * @brief Empyrean Journeys: the render app's source for Empyrean -- its journeys (the instrument's and the player's
 *        files in Documents/Empyrean/Journeys), each rendered as the instrument plays it: every step's preset glided
 *        into (Engine::loadPreset, or the player's own preset by Engine::beginMorph and its knobs), the conductor on,
 *        the engine offline as fast as the card allows.
 *
 * The journeys draw their dwells and glides from ranges; a seed fixes them, so the length of a rendering is known
 * before it begins (lengthOf simulates the JourneyPlayer) and the file holds exactly that. Its presets are the bank's
 * and the player's own (.empyrean files in Documents/Empyrean/Presets, its subfolders too).
 * @author Rene Weller
 */
#include "RenderKit.h"
#include "additiv/Engine.h"
#include "additiv/Journey.h"
#include "additiv/Params.h"
#include "additiv/Presets.h"
#include "additiv/Sun.h"

#include <juce_data_structures/juce_data_structures.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>

namespace
{

constexpr double SampleRate = 48000.0;   ///< the files' rate
constexpr int    Block = 4096;           ///< samples a call of the engine at most (the kit's block)
/** @brief Samples a GPU block when not exact: four of the instrument's. Every partial is set once a block and ramped
 *         across it, so the block is the rate of every modulation; a launch and a wait every 128 samples is
 *         what bounds the speed offline (measured 16.10.2026 with the card busy: 128 1.6 .. 2.6x real time, 512
 *         5.1 .. 8.3x, 1024 8.2 .. 11.6x, 2048 12 .. 15.6x). Six journeys at 512 against 128, the same seed: level
 *         within 0.2 dB, third octaves within 0.1 .. 0.3 dB RMS (another seed: 2.5 .. 8.7), the envelope's
 *         fluctuation within 0.5 dB -- only Scintillation's quickest twinkle (in the gaps between partials, -3 dB
 *         there) is smoothed; at 1024 and beyond that grew to -5 .. -8 dB and took sub-bass sidebands. */
constexpr int FastGpuBlock = 512;

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
#if defined( _WIN32 )
    localtime_s( &local, &now );
#else
    localtime_r( &now, &local );
#endif
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

/**
 * @brief How long a rendering of a journey lasts: its dwells as the seed draws them, for so many passes (one for a
 *        journey that does not cycle) -- exactly what the rendering then writes.
 * @param[in] journey  the journey
 * @param[in] seed     the dice (JourneyPlayer's)
 * @param[in] passes   passes of a cyclic journey (at least one)
 * @return seconds
 */
double lengthOf( const additiv::Journey &journey, std::uint64_t seed, int passes )
{
    const int steps = static_cast<int>( journey.m_steps.size() ) * ( journey.m_cyclic ? std::max( passes, 1 ) : 1 );
    additiv::JourneyPlayer p;
    p.start( journey, seed );
    additiv::JourneyStep st;
    double               fade = 0.0, total = 0.0;
    if ( !p.advance( 0.0, st, fade ) ) return 0.0;
    for ( int k = 0; k < steps; ++k )
    {   // each step's dwell as the player draws it (its glide in it), the next begun when it is over
        total += p.remaining();
        if ( k + 1 < steps && !p.advance( p.remaining(), st, fade ) ) break;
    }
    return total;
}

/** @brief A journey being rendered: the engine offline, the player handing it the steps. */
class JourneyRendering : public renderkit::Rendering
{
public:
    /**
     * @brief Sets the engine up on the journey's first step.
     * @param[in]  journey  the journey
     * @param[in]  r        the request (passes, fade, night, exact)
     * @param[in]  seed     the dice for this journey (renderkit::seedFor)
     * @param[out] why      receives why the engine would not start
     */
    JourneyRendering( const additiv::Journey &journey, const renderkit::Request &r, std::uint64_t seed,
                      juce::String &why )
        : m_journey( journey ), m_own( ownPresets() )
    {
        m_seconds = lengthOf( journey, seed, r.option( "passes" ) + 1 );
        m_fade = r.option( "fade", 1 ) ? 20.0 : 0.0;
        m_engine = std::make_unique<additiv::Engine>();   // on the heap: it is large
        m_engine->setRealtime( false );
        m_player.start( journey, seed );
        additiv::JourneyStep st;
        double               glide = 0.0;
        if ( m_player.advance( 0.0, st, glide ) ) applyStep( *m_engine, st.m_preset, 0.0, m_own );
        m_step = st.m_preset;
        additiv::EngineConfig c;
        c.m_sampleRate = SampleRate;
        c.m_maxHostBlock = Block;
        c.m_realtimeGpuClass = false;
        c.m_gpuBlock = r.option( "exact" ) ? 128 : FastGpuBlock;
        c.m_seed = seed;
        c.m_sunClock = sunClock( r.option( "night" ) != 0 );
        std::string error;
        if ( !m_engine->prepare( c, &error ) )
        {
            why = error;
            m_engine.reset();
            return;
        }
        double latitude = 52.52, longitude = 13.40;
        place( latitude, longitude );
        m_engine->setLocation( latitude, longitude );
        m_engine->requestFill();
    }
    /** @brief Whether the engine started. @return true when it can render */
    bool ok() const { return m_engine != nullptr; }
    /** @brief The length. @return seconds */
    double seconds() const override { return m_seconds; }
    /** @brief The fade over the end. @return seconds */
    double fadeSeconds() const override { return m_fade; }
    /**
     * @brief The next samples: a step begun where the journey says, its preset glided into.
     * @param[out] left    receives the left channel
     * @param[out] right   receives the right channel
     * @param[in]  frames  how many
     */
    void render( float *left, float *right, int frames ) override
    {
        additiv::JourneyStep st;
        double               glide = 0.0;
        const double         dt = frames / SampleRate;
        if ( m_player.advance( dt, st, glide ) && m_time + dt < m_seconds )
        {   // a step begins, as the instrument does it on a journey (not the one after the end)
            applyStep( *m_engine, st.m_preset, std::max( glide, 0.5 ), m_own );
            const std::lock_guard<std::mutex> hold( m_lock );
            m_step = st.m_preset;
        }
        m_engine->process( left, right, frames );
        m_time += dt;
    }
    /** @brief The step that plays. @return "step 3 of 24: Velvet Tide" */
    juce::String now() const override
    {
        const std::lock_guard<std::mutex> hold( m_lock );
        return "step " + juce::String( m_player.step() + 1 ) + " of " +
               juce::String( static_cast<int>( m_journey.m_steps.size() ) ) + ": " + juce::String( m_step );
    }

private:
    additiv::Journey                 m_journey;         ///< the journey
    std::map<std::string, OwnPreset> m_own;             ///< the player's presets
    std::unique_ptr<additiv::Engine> m_engine;          ///< the instrument offline
    additiv::JourneyPlayer           m_player;          ///< the journey's clock
    double                           m_seconds = 0.0;   ///< the length
    double                           m_fade = 0.0;      ///< the fade over the end, s
    double                           m_time = 0.0;      ///< s rendered
    mutable std::mutex               m_lock;            ///< guards m_step
    std::string                      m_step;            ///< the step's preset
};

/** @brief Empyrean's journeys for the render app. */
class EmpyreanSource : public renderkit::Source
{
public:
    juce::String product() const override { return "Empyrean"; }
    juce::String title() const override { return "Empyrean Journeys"; }
    juce::String subtitle() const override { return "Render journeys into files, faster than real time"; }
    juce::String noun() const override { return "journey"; }
    std::vector<renderkit::Column> columns() const override
    {
        return { { "Steps", 56, true }, { "Shortest .. longest", 150, true }, { "Kind", 100, false } };
    }
    std::vector<renderkit::Item> list() override
    {
        std::vector<renderkit::Item> out;
        auto                         add = [&out, this]( additiv::Journey &&j, bool own, const juce::String &fallback )
        {
            renderkit::Item item;
            item.m_name = j.m_name.empty() ? fallback : juce::String( j.m_name );
            item.m_kind = own                                       ? "Yours"
                          : item.m_name.startsWith( "Whole Night" ) ? "Whole night"
                          : item.m_name.endsWith( " - Night" )      ? "Night"
                                                                    : "Journey";
            double shortest = 0.0, longest = 0.0;
            for ( const additiv::JourneyStep &st : j.m_steps )
            {
                shortest += st.m_dwellLo;
                longest += st.m_dwellHi;
                item.m_detail << st.m_preset << " ";
            }
            item.m_cells.add( juce::String( static_cast<int>( j.m_steps.size() ) ) );
            item.m_cells.add( renderkit::clockText( shortest ) + " .. " + renderkit::clockText( longest ) );
            item.m_cells.add( item.m_kind );
            item.m_keys = { static_cast<double>( j.m_steps.size() ), shortest, std::nan( "" ) };
            const std::lock_guard<std::mutex> hold( m_lock );
            item.m_index = static_cast<int>( m_store.size() );
            m_store.push_back( std::make_shared<const additiv::Journey>( std::move( j ) ) );
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
    std::vector<renderkit::Option> options() const override
    {
        renderkit::Option passes{ "passes",
                                  "Passes",
                                  "How many times a cyclic journey goes round (one that does not cycle "
                                  "plays once).",
                                  {},
                                  0,
                                  true };
        for ( int p = 1; p <= 8; ++p ) passes.m_choices.add( juce::String( p ) );
        return {
            passes,
            { "fade",
              "Fade out over the last 20 seconds",
              "The end faded out, so the file does not stop dead.",
              {},
              1,
              false },
            { "night",
              "As at night (the night arc from 23:00)",
              "The presets with a night arc darken, slow and widen as the real night does; rendered as from 23:00 "
              "where the instrument stands (its settings), else as the clock says now.",
              {},
              0,
              false },
            { "exact",
              "Exactly as the instrument plays (slower)",
              "The instrument sets every partial 375 times a second; a rendering does it 94 times, about three "
              "times faster -- the same sound, only Scintillation's quickest twinkle a little smoother. On: as the "
              "instrument plays, sample for sample.",
              {},
              0,
              false }
        };
    }
    bool applies( const renderkit::Item &item, const juce::String &key ) const override
    {
        return key != "passes" || journey( item )->m_cyclic;
    }
    double length( const renderkit::Item &item, const renderkit::Request &r ) override
    {
        return lengthOf( *journey( item ), renderkit::seedFor( item, r ), r.option( "passes" ) + 1 );
    }
    juce::String describe( const renderkit::Item &item, const renderkit::Request &r ) override
    {
        const std::shared_ptr<const additiv::Journey> j = journey( item );
        double                                        shortest = 0.0, longest = 0.0;
        for ( const additiv::JourneyStep &st : j->m_steps )
        {
            shortest += st.m_dwellLo;
            longest += st.m_dwellHi;
        }
        juce::String s;
        s << juce::String( static_cast<int>( j->m_steps.size() ) ) << " steps"
          << ( j->m_cyclic ? ", cyclic" : ", once through" ) << "\n"
          << "A pass: " << renderkit::clockText( shortest ) << " .. " << renderkit::clockText( longest ) << " (about "
          << renderkit::clockText( j->meanLength() ) << ")";
        const int passes = r.option( "passes" ) + 1;
        if ( j->m_cyclic && passes > 1 ) s << "\n" << juce::String( passes ) << " passes";
        return s;
    }
    std::unique_ptr<renderkit::Rendering> open( const renderkit::Item &item, const renderkit::Request &r,
                                                juce::String &why ) override
    {
        auto rendering = std::make_unique<JourneyRendering>( *journey( item ), r, renderkit::seedFor( item, r ), why );
        if ( !rendering->ok() ) return nullptr;
        return rendering;
    }

private:
    /**
     * @brief A row's journey.
     * @param[in] item  the row
     * @return it (shared: a rendering may outlive a Refresh)
     */
    std::shared_ptr<const additiv::Journey> journey( const renderkit::Item &item ) const
    {
        const std::lock_guard<std::mutex> hold( m_lock );
        return m_store[static_cast<size_t>( item.m_index )];
    }
    mutable std::mutex                                   m_lock;    ///< guards m_store
    std::vector<std::shared_ptr<const additiv::Journey>> m_store;   ///< every journey listed (only ever added to)
};

}   // namespace

std::unique_ptr<renderkit::Source> renderkit::makeSource()
{
    return std::make_unique<EmpyreanSource>();
}
