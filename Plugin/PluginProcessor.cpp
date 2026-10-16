/**
 * @file PluginProcessor.cpp
 * @brief The JUCE shell around the engine (PluginProcessor.h).
 * @author Rene Weller
 *
 * @par Compile Flags:
 *   @arg @c ADDITIV_VERSION - the project's version (the top CMakeLists.txt), written into the saved state
 */
#include "PluginProcessor.h"

#include "PluginEditor.h"

#include "additiv/Motion.h"
#include "additiv/Find.h"
#include "additiv/Focus.h"
#include "additiv/Presets.h"
#include "additiv/Sun.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>

using additiv::ParamDesc;
using additiv::ParamIdE;
using additiv::ParamKindE;

namespace
{

/**
 * @brief The host's parameter for one row of the table.
 * @param[in] d  the row
 * @return a float, int, switch or choice parameter with the row's range, default and unit
 */
std::unique_ptr<juce::RangedAudioParameter> makeParameter( const ParamDesc &d )
{
    const juce::ParameterID id{ d.m_key, 1 };
    switch ( d.m_kind )
    {
    case ParamKindE::INT:
        return std::make_unique<juce::AudioParameterInt>( id, d.m_name, static_cast<int>( d.m_min ),
                                                          static_cast<int>( d.m_max ), static_cast<int>( d.m_def ),
                                                          juce::AudioParameterIntAttributes().withLabel( d.m_unit ) );
    case ParamKindE::BOOL: return std::make_unique<juce::AudioParameterBool>( id, d.m_name, d.m_def >= 0.5f );
    case ParamKindE::CHOICE:
    {
        juce::StringArray names;
        for ( int i = 0; i < d.m_numChoices; ++i ) names.add( d.m_choices[i] );
        return std::make_unique<juce::AudioParameterChoice>( id, d.m_name, names, static_cast<int>( d.m_def ) );
    }
    case ParamKindE::FLOAT:
    default:
        // Shown with the precision its size deserves: 440 Hz, 25.0 ct, 1.20, 0.150.
        return std::make_unique<juce::AudioParameterFloat>(
            id, d.m_name, juce::NormalisableRange<float>( d.m_min, d.m_max, 0.0f, d.m_skew ), d.m_def,
            juce::AudioParameterFloatAttributes()
                .withLabel( d.m_unit )
                .withStringFromValueFunction(
                    []( float v, int )
                    {
                        const float a = std::fabs( v );
                        return juce::String( v, a >= 100.0f ? 0 : a >= 10.0f ? 1 : a >= 1.0f ? 2 : 3 );
                    } ) );
    }
}

}   // namespace

AdditivProcessor::AdditivProcessor()
    : AudioProcessor( BusesProperties().withOutput( "Output", juce::AudioChannelSet::stereo(), true ) )
{
    refreshUserPresets();   // the player's own presets, for the bank and the journeys
    // The table's sections become the hosts' groups, in the order they first appear.
    m_params.assign( static_cast<size_t>( additiv::NumParams ), nullptr );
    std::vector<std::unique_ptr<juce::AudioProcessorParameterGroup>> groups;
    std::map<std::string, juce::AudioProcessorParameterGroup *>      bySection;
    for ( const ParamDesc &d : additiv::paramTable() )
    {
        auto it = bySection.find( d.m_section );
        if ( it == bySection.end() )
        {
            groups.push_back( std::make_unique<juce::AudioProcessorParameterGroup>( d.m_section, d.m_section, "|" ) );
            it = bySection.emplace( d.m_section, groups.back().get() ).first;
        }
        auto p = makeParameter( d );
        m_params[static_cast<size_t>( d.m_id )] = p.get();
        it->second->addChild( std::move( p ) );
    }
    for ( auto &g : groups ) addParameterGroup( std::move( g ) );
    m_pushed.assign( m_params.size(), -1.0e30f );
    for ( auto &c : m_ccMap ) c.store( -1, std::memory_order_relaxed );
    {   // the night arc's place, when the settings keep one (else the engine's, from the time zone)
        const frame::Settings &st = frame::Settings::of( App );
        const juce::String     lat = st.value( "night_lat" ), lon = st.value( "night_lon" );
        if ( lat.isNotEmpty() && lon.isNotEmpty() ) m_engine.setLocation( lat.getDoubleValue(), lon.getDoubleValue() );
    }
    startTimerHz( 30 );
}

void AdditivProcessor::setPlace( double latitude, double longitude )
{
    latitude = juce::jlimit( -90.0, 90.0, latitude );
    longitude = juce::jlimit( -180.0, 180.0, longitude );
    frame::Settings &st = frame::Settings::of( App );
    st.setValue( "night_lat", juce::String( latitude, 4 ) );
    st.setValue( "night_lon", juce::String( longitude, 4 ) );
    m_engine.setLocation( latitude, longitude );
}

void AdditivProcessor::placeFromTimeZone()
{
    double lat = 45.0, lon = 0.0;
    additiv::defaultLocation( lat, lon );
    frame::Settings &st = frame::Settings::of( App );
    st.setValue( "night_lat", {} );
    st.setValue( "night_lon", {} );
    m_engine.setLocation( lat, lon );
}

void AdditivProcessor::timerCallback()
{
    m_jam.setRole(
        frame::Settings::of( App ).jamRole() );   // the family jam (Jam.h): Off, Lead or Follow, from the settings
    pollHeadset();
    journeyTick();
    mapTick();
}

void AdditivProcessor::pollHeadset()
{
    frame::Settings &st = frame::Settings::of( App );
    m_headset.listen( st.headset() == frame::Settings::HeadsetModeE::OFF ? 0 : st.headsetPort() );
    m_pose.listen( st.headset() == frame::Settings::HeadsetModeE::OFF ? 0 : st.headsetPort() + 100 );
    m_engine.setHeadsetPose( m_pose.pose() );   // Phase 30: the head and the hands in the cloud
    const frame::HeadsetEvents e = m_headset.poll();
    // The family's grammar on what every preset means the same by: the left hand the brightness
    // (Macro 2), the right the motion (Macro 1); pinches play and stop the conductor, step it,
    // bring Brain 2 in and out, and both hands move to the next preset.
    if ( e.m_playStop )
        setValue( static_cast<int>( ParamIdE::BRAIN_ON ),
                  value( static_cast<int>( ParamIdE::BRAIN_ON ) ) >= 0.5f ? 0.0f : 1.0f );
    if ( e.m_action ) m_engine.autoplayStep();
    if ( e.m_hold )
        setValue( static_cast<int>( ParamIdE::BRAIN2_ON ),
                  value( static_cast<int>( ParamIdE::BRAIN2_ON ) ) >= 0.5f ? 0.0f : 1.0f );
    if ( e.m_filterMoved ) setValue( static_cast<int>( ParamIdE::MACRO2 ), 0.5f * ( e.m_filter + 1.0f ) );
    if ( e.m_throwMoved ) setValue( static_cast<int>( ParamIdE::MACRO1 ), e.m_throwAmount );
    if ( e.m_next ) choosePreset( ( currentPreset() + 1 ) % additiv::numPresets() );
}

AdditivProcessor::~AdditivProcessor()
{
    m_engine.release();
}

void AdditivProcessor::prepareToPlay( double sampleRate, int samplesPerBlock )
{
    additiv::EngineConfig c;
    c.m_sampleRate = sampleRate;
    c.m_maxHostBlock = std::max( samplesPerBlock, 256 );
    // The REALTIME GPU class is a property of the whole process (PLAN.md): the standalone takes it,
    // a plugin leaves its host alone until that is decided.
    c.m_realtimeGpuClass = wrapperType == wrapperType_Standalone;
    m_engine.setRealtime( !isNonRealtime() );
    std::string error;
    if ( !m_engine.prepare( c, &error ) )
    {
        DBG( "Empyrean: the engine did not open: " << error );
    }
    {
        const std::lock_guard<std::mutex> lock( m_errorLock );
        m_engineError = m_engine.prepared() ? std::string() : error;
    }
    std::fill( m_pushed.begin(), m_pushed.end(), -1.0e30f );
    setLatencySamples( m_engine.latencySamples() );
}

void AdditivProcessor::releaseResources()
{
    m_engine.release();
}

bool AdditivProcessor::isBusesLayoutSupported( const BusesLayout &layouts ) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo() &&
           layouts.getMainInputChannelSet().isDisabled();
}

float AdditivProcessor::value( int id ) const
{
    const auto *p = parameter( id );
    return p != nullptr ? p->convertFrom0to1( p->getValue() ) : 0.0f;
}

void AdditivProcessor::setValue( int id, float v )
{
    auto *p = parameter( id );
    if ( p == nullptr ) return;
    const ParamDesc &d = additiv::paramDesc( static_cast<ParamIdE>( id ) );
    p->setValueNotifyingHost( p->convertTo0to1( std::clamp( v, d.m_min, d.m_max ) ) );
}

int AdditivProcessor::controllerFor( int id ) const
{
    for ( int cc = 0; cc < 128; ++cc )
        if ( m_ccMap[static_cast<size_t>( cc )].load( std::memory_order_relaxed ) == id ) return cc;
    return -1;
}

void AdditivProcessor::forget( int id )
{
    for ( auto &c : m_ccMap )
        if ( c.load( std::memory_order_relaxed ) == id ) c.store( -1, std::memory_order_relaxed );
}

void AdditivProcessor::resetToDefault( int id )
{
    if ( id < 0 || id >= additiv::NumParams ) return;
    setValue( id, additiv::paramDesc( static_cast<ParamIdE>( id ) ).m_def );
}

int AdditivProcessor::getNumPrograms()
{
    return additiv::numPresets();
}

const juce::String AdditivProcessor::getProgramName( int index )
{
    const additiv::Preset &p = additiv::preset( index );
    return juce::String( additiv::PresetGroupNames[p.m_group] ) + ": " + p.m_name;
}

void AdditivProcessor::applyPreset( int index, double glide )
{
    if ( index < 0 || index >= additiv::numPresets() ) return;
    float       values[additiv::NumParams];
    std::string mod;
    additiv::presetValues( index, values, &mod );
    // The glide is asked for first: the engine starts it from what it played the block before,
    // whatever of the new preset has reached it by then (Engine::beginMorph).
    const double g = glide >= 0.0 ? glide : static_cast<double>( value( static_cast<int>( ParamIdE::PRESET_GLIDE ) ) );
    // With the map on, a preset chosen is a place: the cursor goes there and the sound follows it.
    if ( value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f && !m_route.running() )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( index, x, y );
        setValue( static_cast<int>( ParamIdE::MAP_X ), x );
        setValue( static_cast<int>( ParamIdE::MAP_Y ), y );
    }
    else if ( g > 0.0 )
    {
        m_engine.beginMorph( g );
    }
    for ( int i = 0; i < additiv::NumParams; ++i )
        if ( !additiv::keptByPresets( static_cast<additiv::ParamIdE>( i ) ) ) setValue( i, values[i] );
    m_engine.setModText( mod );
    if ( g <= 0.0 )
        m_engine
            .requestFill();   // the arriving preset's chord within seconds (Noctuary's way); a glide keeps the chord
    m_current.store( index, std::memory_order_relaxed );
    m_currentUser.clear();
    setReference();
}

// ---- the player's own presets

/**
 * @brief A matrix's text as the engine would write it back: each row through its packing (what the
 *        audio side holds), so that "changed" compares like with like.
 * @param[in] text  the text form
 * @return the engine's spelling of it
 */
static std::string modTextAsEngine( const std::string &text )
{
    additiv::ModRow rows[additiv::ModRows];
    additiv::modRowsFromText( text, rows, additiv::ModRows );
    for ( auto &r : rows ) r = additiv::unpackModRow( additiv::packModRow( r ) );
    return additiv::modRowsToText( rows, additiv::ModRows );
}

juce::File AdditivProcessor::presetFolder()
{
    return juce::File::getSpecialLocation( juce::File::userDocumentsDirectory )
        .getChildFile( "Empyrean" )
        .getChildFile( "Presets" );
}

void AdditivProcessor::refreshUserPresets()
{
    m_userPresets.clear();
    for ( const juce::File &f : presetFolder().findChildFiles( juce::File::findFiles, true, "*.empyrean" ) )
    {
        const std::unique_ptr<juce::XmlElement> xml = juce::XmlDocument::parse( f );
        if ( !xml || !xml->hasTagName( "EmpyreanPreset" ) ) continue;
        UserPreset u;
        u.m_name = f.getFileNameWithoutExtension();
        u.m_file = f;
        u.m_values.resize( static_cast<size_t>( additiv::NumParams ) );
        for ( const ParamDesc &d : additiv::paramTable() ) u.m_values[static_cast<size_t>( d.m_id )] = d.m_def;
        for ( const juce::String &kv : juce::StringArray::fromTokens( xml->getStringAttribute( "values" ), " ", "" ) )
            if ( const ParamDesc *d = additiv::findParam( kv.upToFirstOccurrenceOf( "=", false, false ).toRawUTF8() ) )
                u.m_values[static_cast<size_t>( d->m_id )] =
                    std::clamp( kv.fromFirstOccurrenceOf( "=", false, false ).getFloatValue(), d->m_min, d->m_max );
        u.m_mod = xml->getStringAttribute( "mod" ).toStdString();
        m_userPresets.push_back( std::move( u ) );
    }
    std::sort( m_userPresets.begin(), m_userPresets.end(),
               []( const UserPreset &a, const UserPreset &b ) { return a.m_name.compareNatural( b.m_name ) < 0; } );
}

int AdditivProcessor::findUserPreset( const juce::String &name ) const
{
    for ( size_t i = 0; i < m_userPresets.size(); ++i )
        if ( m_userPresets[i].m_name == name ) return static_cast<int>( i );
    return -1;
}

bool AdditivProcessor::saveUserPreset( const juce::String &name, juce::String &why )
{
    const juce::String legal = juce::File::createLegalFileName( name.trim() );
    if ( legal.isEmpty() )
    {
        why = "a preset needs a name";
        return false;
    }
    if ( additiv::findPreset( legal.toRawUTF8() ) >= 0 )
    {
        why = "a factory preset has that name (journeys find presets by name)";
        return false;
    }
    juce::String values;
    for ( const ParamDesc &d : additiv::paramTable() )
        if ( !additiv::keptByPresets( d.m_id ) )
            values << d.m_key << "=" << juce::String( value( static_cast<int>( d.m_id ) ), 6 ) << " ";
    juce::XmlElement xml( "EmpyreanPreset" );
    xml.setAttribute( "version", ADDITIV_VERSION );
    xml.setAttribute( "values", values.trim() );
    xml.setAttribute( "mod", juce::String( m_engine.modText() ) );
    presetFolder().createDirectory();
    if ( !xml.writeTo( presetFolder().getChildFile( legal + ".empyrean" ) ) )
    {
        why = "it could not be written into " + presetFolder().getFullPathName();
        return false;
    }
    refreshUserPresets();
    m_current.store( -1, std::memory_order_relaxed );
    m_currentUser = legal;
    setReference();
    return true;
}

void AdditivProcessor::applyUserPreset( int i, double glide )
{
    if ( i < 0 || i >= static_cast<int>( m_userPresets.size() ) ) return;
    const UserPreset &u = m_userPresets[static_cast<size_t>( i )];
    const double g = glide >= 0.0 ? glide : static_cast<double>( value( static_cast<int>( ParamIdE::PRESET_GLIDE ) ) );
    if ( g > 0.0 ) m_engine.beginMorph( g );
    for ( int k = 0; k < additiv::NumParams; ++k )
        if ( !additiv::keptByPresets( static_cast<ParamIdE>( k ) ) )
            setValue( k, u.m_values[static_cast<size_t>( k )] );
    m_engine.setModText( u.m_mod );
    if ( g <= 0.0 ) m_engine.requestFill();
    m_current.store( -1, std::memory_order_relaxed );
    m_currentUser = u.m_name;
    setReference();
}

void AdditivProcessor::chooseUserPreset( int i )
{
    if ( i < 0 || i >= static_cast<int>( m_userPresets.size() ) ) return;
    m_browsing = false;
    noteRecent( m_userPresets[static_cast<size_t>( i )].m_name );
    m_journeyPlayer.stop();
    setValue( static_cast<int>( ParamIdE::MORPH_ACTIVE ), 0.0f );
    leaveMap( false );
    applyUserPreset( i, -1.0 );
}

void AdditivProcessor::takeSnapshot( const juce::Image &picture )
{
    Snapshot s;
    s.m_values = knobValues();
    s.m_mod = m_engine.modText();
    s.m_picture = picture;
    m_snapshots.push_back( std::move( s ) );
    if ( m_snapshots.size() > 8 ) m_snapshots.erase( m_snapshots.begin() );
}

void AdditivProcessor::recallSnapshot( int i )
{
    if ( i < 0 || i >= static_cast<int>( m_snapshots.size() ) ) return;
    stepOutOfBlends();
    applyValues( m_snapshots[static_cast<size_t>( i )].m_values, 1.0 );
    m_engine.setModText( m_snapshots[static_cast<size_t>( i )].m_mod );
}

void AdditivProcessor::forgetSnapshot( int i )
{
    if ( i >= 0 && i < static_cast<int>( m_snapshots.size() ) ) m_snapshots.erase( m_snapshots.begin() + i );
}

void AdditivProcessor::snapshotToSlot( int i, int slot )
{
    if ( i < 0 || i >= static_cast<int>( m_snapshots.size() ) ) return;
    slot &= 1;
    m_slotValues[slot] = m_snapshots[static_cast<size_t>( i )].m_values;
    m_slotMods[slot] = m_snapshots[static_cast<size_t>( i )].m_mod;
    m_engine.setMorphSlot( slot, m_slotValues[slot].data(), m_slotMods[slot] );
    m_slotNames[slot] = "snapshot " + juce::String( i + 1 );
}

juce::PropertiesFile &AdditivProcessor::userSettings()
{
    if ( m_userSettings == nullptr )
    {
        juce::PropertiesFile::Options o;
        o.applicationName = "player";
        o.folderName = "Empyrean";
        o.filenameSuffix = ".settings";
        o.osxLibrarySubFolder = "Application Support";
        m_userSettings = std::make_unique<juce::PropertiesFile>( o );
    }
    return *m_userSettings;
}

bool AdditivProcessor::favorite( const juce::String &name ) const
{
    auto *self = const_cast<AdditivProcessor *>( this );   // the settings file is read on first use
    return juce::StringArray::fromLines( self->userSettings().getValue( "favorites" ) ).contains( name );
}

void AdditivProcessor::setFavorite( const juce::String &name, bool on )
{
    juce::StringArray f = juce::StringArray::fromLines( userSettings().getValue( "favorites" ) );
    f.removeEmptyStrings();
    f.removeString( name );
    if ( on ) f.add( name );
    userSettings().setValue( "favorites", f.joinIntoString( "\n" ) );
    userSettings().saveIfNeeded();
}

juce::StringArray AdditivProcessor::recentPresets() const
{
    auto             *self = const_cast<AdditivProcessor *>( this );
    juce::StringArray r = juce::StringArray::fromLines( self->userSettings().getValue( "recent" ) );
    r.removeEmptyStrings();
    return r;
}

void AdditivProcessor::noteRecent( const juce::String &name )
{
    juce::StringArray r = recentPresets();
    r.removeString( name );
    r.insert( 0, name );
    while ( r.size() > 40 ) r.remove( r.size() - 1 );
    userSettings().setValue( "recent", r.joinIntoString( "\n" ) );
    userSettings().saveIfNeeded();
}

void AdditivProcessor::previewValues( const std::vector<float> &v, const std::string &mod )
{
    if ( !m_browsing )
    {
        stepOutOfBlends();
        m_browseFrom = knobValues();
        m_browseFromMod = m_engine.modText();
        m_browsing = true;
    }
    m_engine.beginMorph( 0.4 );
    for ( int k = 0; k < additiv::NumParams; ++k )
        if ( !additiv::keptByPresets( static_cast<ParamIdE>( k ) ) ) setValue( k, v[static_cast<size_t>( k )] );
    m_engine.setModText( mod );
}

void AdditivProcessor::previewPreset( int index )
{
    if ( index < 0 || index >= additiv::numPresets() ) return;
    float       values[additiv::NumParams];
    std::string mod;
    additiv::presetValues( index, values, &mod );
    previewValues( std::vector<float>( values, values + additiv::NumParams ), mod );
}

void AdditivProcessor::previewUserPreset( int i )
{
    if ( i >= 0 && i < static_cast<int>( m_userPresets.size() ) )
        previewValues( m_userPresets[static_cast<size_t>( i )].m_values,
                       m_userPresets[static_cast<size_t>( i )].m_mod );
}

void AdditivProcessor::endPreview()
{
    if ( !m_browsing ) return;
    m_browsing = false;
    m_engine.beginMorph( 0.4 );
    for ( int k = 0; k < additiv::NumParams; ++k )
        if ( !additiv::keptByPresets( static_cast<ParamIdE>( k ) ) )
            setValue( k, m_browseFrom[static_cast<size_t>( k )] );
    m_engine.setModText( m_browseFromMod );
}

void AdditivProcessor::setReference()
{
    m_reference = knobValues();
    m_referenceMod = m_engine.modText();
}

bool AdditivProcessor::modified() const
{
    if ( m_reference.size() != static_cast<size_t>( additiv::NumParams ) ) return false;
    for ( const ParamDesc &d : additiv::paramTable() )
    {
        if ( additiv::keptByPresets( d.m_id ) ) continue;
        if ( std::fabs( value( static_cast<int>( d.m_id ) ) - m_reference[static_cast<size_t>( d.m_id )] ) >
             1.0e-5f * std::max( 1.0f, d.m_max - d.m_min ) )
            return true;
    }
    return m_engine.modText() != m_referenceMod;
}

void AdditivProcessor::revert()
{
    if ( m_reference.size() != static_cast<size_t>( additiv::NumParams ) ) return;
    m_engine.beginMorph( 1.0 );
    for ( int k = 0; k < additiv::NumParams; ++k )
        if ( !additiv::keptByPresets( static_cast<ParamIdE>( k ) ) )
            setValue( k, m_reference[static_cast<size_t>( k )] );
    m_engine.setModText( m_referenceMod );
}

void AdditivProcessor::revertKnob( int id )
{
    if ( id >= 0 && id < additiv::NumParams && m_reference.size() == static_cast<size_t>( additiv::NumParams ) )
        setValue( id, m_reference[static_cast<size_t>( id )] );
}

void AdditivProcessor::choosePreset( int index, bool recent )
{
    if ( index < 0 || index >= additiv::numPresets() ) return;
    m_browsing = false;   // chosen while listened to: it stays
    if ( recent ) noteRecent( additiv::preset( index ).m_name );
    m_journeyPlayer.stop();
    setValue( static_cast<int>( ParamIdE::MORPH_ACTIVE ), 0.0f );
    if ( value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f )
    {
        leaveMap( false );   // the route stops with it; the engine glides from the map's blend (Map Glide)
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( index, x, y );
        setValue( static_cast<int>( ParamIdE::MAP_X ), x );
        setValue( static_cast<int>( ParamIdE::MAP_Y ), y );
    }
    else
    {
        m_route.stop();
    }
    applyPreset( index );
}

void AdditivProcessor::setSlotFromSound( int slot )
{
    slot &= 1;
    m_slotValues[slot].assign( static_cast<size_t>( additiv::NumParams ), 0.0f );
    for ( int i = 0; i < additiv::NumParams; ++i ) m_slotValues[slot][static_cast<size_t>( i )] = value( i );
    m_slotMods[slot] = m_engine.modText();
    m_engine.setMorphSlot( slot, m_slotValues[slot].data(), m_slotMods[slot] );
    const int cp = currentPreset();
    m_slotNames[slot] =
        cp >= 0 ? juce::String( "the sound of " ) + additiv::preset( cp ).m_name : juce::String( "the sound" );
}

void AdditivProcessor::setSlotFromPreset( int slot, int index )
{
    if ( index < 0 || index >= additiv::numPresets() ) return;
    slot &= 1;
    m_slotValues[slot].assign( static_cast<size_t>( additiv::NumParams ), 0.0f );
    additiv::presetValues( index, m_slotValues[slot].data(), &m_slotMods[slot] );
    m_engine.setMorphSlot( slot, m_slotValues[slot].data(), m_slotMods[slot] );
    m_slotNames[slot] = juce::String( additiv::PresetGroupNames[additiv::preset( index ).m_group] ) + ": " +
                        additiv::preset( index ).m_name;
}

bool AdditivProcessor::loadLandscape( const juce::File &file, bool letItSound, juce::String *error )
{
    additiv::Landscape l;
    std::string        why;
    const juce::String ext = file.getFileExtension().toLowerCase();
    const bool         audio = ext == ".wav" || ext == ".flac";
    // A picture lasts a minute from left to right (Motion Rate speeds it); a recording keeps its own
    // time, its columns whitened so it carries its motion rather than its colour (the probe's settings).
    const bool read =
        audio ? additiv::landscapeFromAudio( file.getFullPathName().toStdString(), 40.0f, 0.7f, 0.1f, 0.15f, l, &why )
              : additiv::landscapeFromImage( file.getFullPathName().toStdString(), 60.0f, 40.0f, l, &why );
    if ( !read )
    {
        if ( error != nullptr ) *error = juce::String( why.empty() ? std::string( "it could not be read" ) : why );
        return false;
    }
    m_engine.setUserLandscape( l );
    m_userLand = std::move( l );
    m_userLandFile = file;
    ++m_userLandVersion;
    setValue( static_cast<int>( ParamIdE::LANDSCAPE ), static_cast<float>( additiv::UserLandscape ) );
    // Heard: the Aurora motion reads the landscape, the living landscape's picture modes too.
    const bool read2 = additiv::motionUsesLandscape(
        static_cast<additiv::MotionKindE>( juce::roundToInt( value( static_cast<int>( ParamIdE::MOTION ) ) ) ) );
    const bool pictured = additiv::lifeIsPicture(
        static_cast<additiv::LifeModeE>( juce::roundToInt( value( static_cast<int>( ParamIdE::LIFE_MODE ) ) ) ) );
    if ( letItSound && !read2 && !pictured )
        setValue( static_cast<int>( ParamIdE::MOTION ), static_cast<float>( additiv::MotionKindE::AURORA ) );
    return true;
}

void AdditivProcessor::moveCursor( float x, float y )
{
    m_route.stop();
    setValue( static_cast<int>( ParamIdE::MAP_ACTIVE ), 1.0f );
    setValue( static_cast<int>( ParamIdE::MAP_X ), juce::jlimit( 0.0f, 1.0f, x ) );
    setValue( static_cast<int>( ParamIdE::MAP_Y ), juce::jlimit( 0.0f, 1.0f, y ) );
}

int AdditivProcessor::nearestPreset( float x, float y ) const
{
    const additiv::MapNeighbours b = additiv::mapNeighbours( x, y, 0.01f );
    return b.m_count > 0 ? b.m_index[0] : 0;
}

void AdditivProcessor::leaveMap( bool keepSound )
{
    m_route.stop();
    if ( value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) < 0.5f )
    {
        m_mapOnWas = false;
        return;
    }
    setValue( static_cast<int>( ParamIdE::MAP_ACTIVE ), 0.0f );
    m_mapOnWas = false;
    if ( keepSound )
        applyPreset(
            nearestPreset( value( static_cast<int>( ParamIdE::MAP_X ) ), value( static_cast<int>( ParamIdE::MAP_Y ) ) ),
            std::max( 1.0, static_cast<double>( value( static_cast<int>( ParamIdE::MAP_GLIDE ) ) ) ) );
}

void AdditivProcessor::mapTick()
{
    const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
    const double dt = m_mapLastTick > 0.0 ? juce::jlimit( 0.0, 1.0, now - m_mapLastTick ) : 0.0;
    m_mapLastTick = now;
    // The route walks the cursor (its times divided by Route Speed); the map plays while it does.
    if ( m_route.running() )
    {
        float x = 0.0f, y = 0.0f, r = 0.0f;
        m_route.update( static_cast<float>( dt ) * value( static_cast<int>( ParamIdE::ROUTE_SPEED ) ), x, y, r );
        if ( value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) < 0.5f )
            setValue( static_cast<int>( ParamIdE::MAP_ACTIVE ), 1.0f );
        setValue( static_cast<int>( ParamIdE::MAP_X ), x );
        setValue( static_cast<int>( ParamIdE::MAP_Y ), y );
        setValue( static_cast<int>( ParamIdE::MAP_RADIUS ), r );
    }
    const bool on = value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f;
    if ( !on )
    {
        if ( m_mapOnWas ) leaveMap( true );   // switched off by hand: the sound stays near where it was
        return;
    }
    m_mapOnWas = true;
    const float x = value( static_cast<int>( ParamIdE::MAP_X ) ), y = value( static_cast<int>( ParamIdE::MAP_Y ) ),
                r = value( static_cast<int>( ParamIdE::MAP_RADIUS ) );
    if ( x == m_mapX && y == m_mapY && r == m_mapR && m_engine.mapTargetSet() ) return;
    m_mapX = x;
    m_mapY = y;
    m_mapR = r;
    additiv::MorphState s;
    m_blender.blend( x, y, r, s );
    m_engine.setMapTarget( s );
}

/**
 * @brief A hall's memory as text: its HallBins floats, base64.
 * @param[in] m  the memory
 * @return the text
 */
static juce::String hallText( const std::vector<float> &m )
{
    const juce::MemoryBlock b( m.data(), m.size() * sizeof( float ) );
    return b.toBase64Encoding();
}

/**
 * @brief A hall's memory from its text.
 * @param[in]  s  the text
 * @param[out] m  receives the memory
 * @return false when it is no memory
 */
static bool hallFromText( const juce::String &s, std::vector<float> &m )
{
    juce::MemoryBlock b;
    if ( !b.fromBase64Encoding( s ) || b.getSize() != sizeof( float ) * static_cast<size_t>( additiv::HallBins ) )
        return false;
    m.resize( static_cast<size_t>( additiv::HallBins ) );
    std::memcpy( m.data(), b.getData(), b.getSize() );
    return true;
}

// ---- the Focus page

std::vector<float> AdditivProcessor::knobValues() const
{
    std::vector<float> v( static_cast<size_t>( additiv::NumParams ) );
    for ( int i = 0; i < additiv::NumParams; ++i ) v[static_cast<size_t>( i )] = value( i );
    return v;
}

void AdditivProcessor::applyValues( const std::vector<float> &v, double glide )
{
    if ( v.size() != static_cast<size_t>( additiv::NumParams ) ) return;
    if ( glide > 0.0 ) m_engine.beginMorph( glide );
    for ( int i = 0; i < additiv::NumParams; ++i )
        if ( !additiv::keptByPresets( static_cast<ParamIdE>( i ) ) && v[static_cast<size_t>( i )] != value( i ) )
            setValue( i, v[static_cast<size_t>( i )] );
}

void AdditivProcessor::stepOutOfBlends()
{
    if ( value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f ) leaveMap( true );
    if ( value( static_cast<int>( ParamIdE::MORPH_ACTIVE ) ) >= 0.5f )
        setValue( static_cast<int>( ParamIdE::MORPH_ACTIVE ), 0.0f );
}

float AdditivProcessor::nudgeSound( int axis, float sign )
{
    stepOutOfBlends();
    const std::vector<float> now = knobValues();
    std::vector<float>       to( now.size() );
    const float              got = additiv::nudge( now.data(), axis, sign, 0.25f, to.data() );
    applyValues( to, 0.6 );
    return got;
}

bool AdditivProcessor::variationsFree() const
{
    return value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) < 0.5f &&
           value( static_cast<int>( ParamIdE::MORPH_ACTIVE ) ) < 0.5f && !m_browsing && !m_comparing &&
           m_previewing < 0;
}

void AdditivProcessor::compareWithPreset( bool on )
{
    if ( on == m_comparing || m_reference.size() != static_cast<size_t>( additiv::NumParams ) ) return;
    m_comparing = on;
    if ( on )
    {
        m_compareFrom = knobValues();
        m_compareFromMod = m_engine.modText();
        applyValues( m_reference, 0.3 );
        m_engine.setModText( m_referenceMod );
    }
    else
    {
        applyValues( m_compareFrom, 0.3 );
        m_engine.setModText( m_compareFromMod );
    }
}

void AdditivProcessor::makeVariations( float far )
{
    ++m_variationGeneration;
    stepOutOfBlends();
    m_varFar = std::clamp( far, 0.05f, 1.0f );
    m_parent = knobValues();
    m_variations.assign( 6, std::vector<float>( m_parent.size() ) );
    for ( size_t k = 0; k < m_variations.size(); ++k )
        additiv::vary( m_parent.data(), m_varFar, m_varSeed++, m_variations[k].data() );
}

void AdditivProcessor::previewVariation( int k )
{
    if ( k >= 0 && k < variationCount() )
    {
        if ( m_previewing < 0 ) m_previewFrom = knobValues();
        m_previewing = k;
        applyValues( m_variations[static_cast<size_t>( k )], 0.4 );
    }
    else if ( m_previewing >= 0 )
    {
        m_previewing = -1;
        applyValues( m_previewFrom, 0.4 );
    }
}

void AdditivProcessor::takeVariation( int k )
{
    if ( k < 0 || k >= variationCount() ) return;
    if ( m_previewing >= 0 )
    {   // the sound before the listening is where it is kept from
        m_previewing = -1;
        applyValues( m_previewFrom, 0.0 );
    }
    m_parents.push_back( m_parent );
    if ( m_parents.size() > 32 ) m_parents.erase( m_parents.begin() );
    applyValues( m_variations[static_cast<size_t>( k )], 0.5 );
    makeVariations( m_varFar );
}

bool AdditivProcessor::variationBack()
{
    if ( m_parents.empty() ) return false;
    const std::vector<float> back = m_parents.back();
    m_parents.pop_back();
    applyValues( back, 0.5 );
    makeVariations( m_varFar );
    return true;
}

juce::String AdditivProcessor::variationSummary( int k ) const
{
    if ( k < 0 || k >= variationCount() || m_parent.size() != static_cast<size_t>( additiv::NumParams ) ) return {};
    std::vector<std::pair<float, int>> moved;
    for ( int i = 0; i < additiv::NumParams; ++i )
    {
        const additiv::ParamDesc &d = additiv::paramTable()[static_cast<size_t>( i )];
        if ( d.m_id == ParamIdE::MASTER_GAIN ) continue;
        const float dn = additiv::toNorm( d, m_variations[static_cast<size_t>( k )][static_cast<size_t>( i )] ) -
                         additiv::toNorm( d, m_parent[static_cast<size_t>( i )] );
        if ( std::fabs( dn ) > 0.01f ) moved.emplace_back( dn, i );
    }
    std::sort( moved.begin(), moved.end(),
               []( const auto &a, const auto &b ) { return std::fabs( a.first ) > std::fabs( b.first ); } );
    juce::String s;
    for ( size_t j = 0; j < moved.size() && j < 3; ++j )
        s << ( j > 0 ? ", " : "" ) << additiv::paramTable()[static_cast<size_t>( moved[j].second )].m_name
          << ( moved[j].first > 0.0f ? " up" : " down" );
    if ( moved.size() > 3 ) s << " ...";
    return s;
}

void AdditivProcessor::teachBegin( int m )
{
    if ( m < 0 || m >= 4 ) return;
    stepOutOfBlends();
    m_teachBase[static_cast<size_t>( m )] = knobValues();
    additiv::MacroLesson                 &l = m_lessons[static_cast<size_t>( m )];
    std::array<float, additiv::NumParams> zero{};
    if ( l.m_count < 2 )
    {   // a new lesson: its first place is the sound now
        l = additiv::MacroLesson{};
        l.set( value( static_cast<int>( ParamIdE::MACRO1 ) + m ), zero.data() );
    }
    m_engine.setMacroLesson( m, l );
}

bool AdditivProcessor::teachHere( int m )
{
    if ( !teaching( m ) ) return false;
    const std::vector<float> &from = m_teachBase[static_cast<size_t>( m )];
    additiv::MacroLesson     &l = m_lessons[static_cast<size_t>( m )];
    const float               at = value( static_cast<int>( ParamIdE::MACRO1 ) + m );
    // What the macro already carries here, and what the knobs moved since: the new place.
    std::array<float, additiv::NumParams> carried{}, delta{};
    l.apply( at, carried.data(), nullptr );
    for ( int i = 0; i < additiv::NumParams; ++i )
    {
        const additiv::ParamDesc &d = additiv::paramTable()[static_cast<size_t>( i )];
        if ( !additiv::teachable( d.m_id ) ) continue;
        const float dn = additiv::toNorm( d, value( i ) ) - additiv::toNorm( d, from[static_cast<size_t>( i )] ) +
                         carried[static_cast<size_t>( i )];
        delta[static_cast<size_t>( i )] = std::fabs( dn ) > 0.002f ? dn : 0.0f;
    }
    l.set( at, delta.data() );
    // The macro carries the change, the knobs go back where the teaching began -- in the same moment, no
    // glide: a glide of the knobs under a lesson already in force would count the change twice for its length.
    m_engine.setMacroLesson( m, l );
    for ( int i = 0; i < additiv::NumParams; ++i )
        if ( additiv::teachable( static_cast<ParamIdE>( i ) ) && value( i ) != from[static_cast<size_t>( i )] )
            setValue( i, from[static_cast<size_t>( i )] );
    return true;
}

void AdditivProcessor::teachCancel( int m )
{
    if ( !teaching( m ) ) return;
    const std::vector<float> from = m_teachBase[static_cast<size_t>( m )];
    m_teachBase[static_cast<size_t>( m )].clear();
    if ( m_lessons[static_cast<size_t>( m )].m_count < 2 )
    {   // a new lesson had only its first place: none
        m_lessons[static_cast<size_t>( m )] = additiv::MacroLesson{};
        m_engine.setMacroLesson( m, m_lessons[static_cast<size_t>( m )] );
    }
    for ( int i = 0; i < additiv::NumParams; ++i )
        if ( additiv::teachable( static_cast<ParamIdE>( i ) ) && value( i ) != from[static_cast<size_t>( i )] )
            setValue( i, from[static_cast<size_t>( i )] );
}

void AdditivProcessor::teachForget( int m )
{
    if ( m < 0 || m >= 4 ) return;
    m_lessons[static_cast<size_t>( m )] = additiv::MacroLesson{};
    m_teachBase[static_cast<size_t>( m )].clear();
    m_engine.setMacroLesson( m, m_lessons[static_cast<size_t>( m )] );
}

juce::String AdditivProcessor::macroName( int m ) const
{
    if ( m < 0 || m >= 4 ) return {};
    std::vector<std::pair<float, int>> moved;   // per knob how far the macro moves it, the larger first
    const additiv::MacroLesson        &l = m_lessons[static_cast<size_t>( m )];
    if ( l.m_count >= 2 )
        for ( int p = 0; p < additiv::NumParams; ++p )
        {
            float most = 0.0f;
            for ( int i = 0; i < l.m_count; ++i )
                most = std::max( most, std::fabs( l.m_delta[i][static_cast<size_t>( p )] ) );
            if ( most > 0.0f ) moved.emplace_back( most, p );
        }
    const auto source = static_cast<additiv::ModSourceE>( static_cast<int>( additiv::ModSourceE::MACRO1 ) + m );
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const additiv::ModRow r = m_engine.modRow( i );
        if ( r.m_source == source && r.m_target >= 0 && r.m_target < additiv::NumParams && r.m_depth != 0.0f )
            moved.emplace_back( std::fabs( r.m_depth ), r.m_target );
    }
    if ( moved.empty() ) return "Macro " + juce::String( m + 1 );
    std::sort( moved.begin(), moved.end(), []( const auto &a, const auto &b ) { return a.first > b.first; } );
    std::vector<int> knobs;
    for ( const auto &x : moved )
        if ( std::find( knobs.begin(), knobs.end(), x.second ) == knobs.end() ) knobs.push_back( x.second );
    // A name two knobs share ("Depth": the motion's and the space's) carries its section.
    const ParamDesc &first = additiv::paramTable()[static_cast<size_t>( knobs[0] )];
    bool             shared = false;
    for ( const ParamDesc &d : additiv::paramTable() )
        shared = shared || ( d.m_id != first.m_id && juce::String( d.m_name ) == first.m_name );
    juce::String s = shared ? juce::String( first.m_section ) + " " + first.m_name : juce::String( first.m_name );
    if ( knobs.size() > 1 ) s << " & " << static_cast<int>( knobs.size() - 1 ) << " more";
    return s;
}

juce::String AdditivProcessor::teachStatus( int m ) const
{
    if ( m < 0 || m >= 4 ) return {};
    const additiv::MacroLesson &l = m_lessons[static_cast<size_t>( m )];
    juce::String                s;
    if ( l.m_count >= 2 )
    {
        s << l.m_count << " places (";
        for ( int i = 0; i < l.m_count; ++i ) s << ( i > 0 ? ", " : "" ) << juce::String( l.m_pos[i], 2 );
        s << "), " << l.knobs() << " knobs";
    }
    else
    {
        s << "nothing taught";
    }
    if ( teaching( m ) ) s << " -- teaching: move the knobs, set the macro, press Here";
    return s;
}

bool AdditivProcessor::findLike( const juce::File &file, juce::String &why )
{
    additiv::Description d;
    std::string          error;
    if ( !additiv::describeFile( file.getFullPathName().toStdString(), d, nullptr, &error ) )
    {
        why = juce::String( error );
        return false;
    }
    int       idx[12];
    float     dist[12];
    const int n = additiv::findLike( d, idx, 12, dist );
    m_found.assign( idx, idx + n );
    m_foundDistance.assign( dist, dist + n );
    return true;
}

juce::File AdditivProcessor::hallFolder()
{
    return juce::File::getSpecialLocation( juce::File::userDocumentsDirectory )
        .getChildFile( "Empyrean" )
        .getChildFile( "Halls" );
}

bool AdditivProcessor::saveHall( const juce::File &file )
{
    std::vector<float> m;
    double             age = 0.0;
    m_engine.hallMemory( m, age );
    juce::XmlElement xml( "EmpyreanHall" );
    xml.setAttribute( "version", ADDITIV_VERSION );
    xml.setAttribute( "age", age );
    xml.setAttribute( "memory", hallText( m ) );
    if ( !xml.writeTo( file ) ) return false;
    m_hallName = file.getFileNameWithoutExtension();
    return true;
}

bool AdditivProcessor::loadHall( const juce::File &file )
{
    const std::unique_ptr<juce::XmlElement> xml = juce::XmlDocument::parse( file );
    std::vector<float>                      m;
    if ( !xml || !xml->hasTagName( "EmpyreanHall" ) || !hallFromText( xml->getStringAttribute( "memory" ), m ) )
        return false;
    m_engine.setHallMemory( m, xml->getDoubleAttribute( "age" ) );
    m_hallName = file.getFileNameWithoutExtension();
    return true;
}

void AdditivProcessor::newHall()
{
    m_engine.setHallMemory( {}, 0.0 );
    m_hallName.clear();
}

juce::File AdditivProcessor::routeFolder()
{
    return juce::File::getSpecialLocation( juce::File::userDocumentsDirectory )
        .getChildFile( "Empyrean" )
        .getChildFile( "Routes" );
}

std::vector<AdditivProcessor::RouteEntry> AdditivProcessor::routes() const
{
    std::vector<RouteEntry> out;
    for ( int i = 0; i < additiv::numRoutePresets(); ++i )
    {
        RouteEntry e;
        e.m_name = additiv::routePreset( i ).m_name;
        e.m_builtin = i;
        out.push_back( e );
    }
    juce::Array<juce::File> files = routeFolder().findChildFiles( juce::File::findFiles, false, "*.route" );
    files.sort();
    for ( const juce::File &f : files )
    {
        RouteEntry e;
        e.m_name = f.getFileNameWithoutExtension();
        e.m_file = f;
        out.push_back( e );
    }
    return out;
}

bool AdditivProcessor::loadRoute( const RouteEntry &entry )
{
    const juce::String text = entry.m_builtin >= 0 ? juce::String( additiv::routePreset( entry.m_builtin ).m_points )
                                                   : entry.m_file.loadFileAsString();
    additiv::Route     r;
    if ( !r.parse( text.toRawUTF8() ) ) return false;
    m_route = r;
    m_routeName = entry.m_name;
    return true;
}

bool AdditivProcessor::startRoute()
{
    if ( m_route.count() == 0 ) return false;
    m_journeyPlayer.stop();
    setValue( static_cast<int>( ParamIdE::MORPH_ACTIVE ), 0.0f );
    setValue( static_cast<int>( ParamIdE::MAP_ACTIVE ), 1.0f );
    setValue( static_cast<int>( ParamIdE::BRAIN_ON ),
              1.0f );   // a route is listened to: the conductor plays (the map keeps it so)
    m_route.start( value( static_cast<int>( ParamIdE::MAP_X ) ), value( static_cast<int>( ParamIdE::MAP_Y ) ),
                   value( static_cast<int>( ParamIdE::MAP_RADIUS ) ), m_routeLoop );
    return true;
}

void AdditivProcessor::addWaypoint( float x, float y, int preset )
{
    additiv::Waypoint w;
    w.m_x = juce::jlimit( 0.0f, 1.0f, x );
    w.m_y = juce::jlimit( 0.0f, 1.0f, y );
    w.m_preset = preset;
    if ( preset >= 0 ) additiv::presetMapPosition( preset, w.m_x, w.m_y );
    w.m_radius = value( static_cast<int>( ParamIdE::MAP_RADIUS ) );
    w.m_travel = m_route.count() == 0 ? 20.0f : 120.0f;
    w.m_hold = 60.0f;
    m_route.stop();
    m_route.add( w );
}

bool AdditivProcessor::saveRoute( const juce::File &file )
{
    if ( m_route.count() == 0 ) return false;
    std::vector<char> buf( static_cast<size_t>( m_route.count() ) * 200 + 64 );
    if ( m_route.write( buf.data(), buf.size() ) <= 0 ) return false;
    // One waypoint a line, under a line that says what the fields are.
    const juce::String text = juce::String( "# Empyrean route: <preset or x,y> | travel s | hold s | radius\n" ) +
                              juce::String( buf.data() ).replace( ";", "\n" ) + "\n";
    file.getParentDirectory().createDirectory();
    if ( !file.replaceWithText( text ) ) return false;
    m_routeName = file.getFileNameWithoutExtension();
    return true;
}

juce::String AdditivProcessor::routeStatus() const
{
    if ( !m_route.running() ) return {};
    const int                i = m_route.segment();
    const additiv::Waypoint &w = m_route.point( i );
    juce::String             s = m_routeName.isNotEmpty() ? m_routeName + "   " : juce::String();
    s << ( i + 1 ) << "/" << m_route.count() << "   "
      << ( w.m_preset >= 0 ? juce::String( additiv::preset( w.m_preset ).m_name ) : juce::String( "a place" ) ) << "   "
      << juce::roundToInt( 100.0f * m_route.progress() ) << " %";
    return s;
}

juce::File AdditivProcessor::journeyFolder()
{
    return juce::File::getSpecialLocation( juce::File::userDocumentsDirectory )
        .getChildFile( "Empyrean" )
        .getChildFile( "Journeys" );
}

std::vector<AdditivProcessor::JourneyEntry> AdditivProcessor::journeys() const
{
    std::vector<JourneyEntry> out;
    for ( int i = 0; i < additiv::numBuiltinJourneys(); ++i )
    {
        additiv::Journey j;
        if ( !j.parse( additiv::builtinJourneyText( i ) ) ) continue;
        JourneyEntry e;
        e.m_name = j.m_name;
        e.m_builtin = i;
        out.push_back( e );
    }
    juce::Array<juce::File> files = journeyFolder().findChildFiles( juce::File::findFiles, false, "*.journey" );
    files.sort();
    for ( const juce::File &f : files )
    {
        JourneyEntry e;
        e.m_name = f.getFileNameWithoutExtension();
        e.m_file = f;
        out.push_back( e );
    }
    return out;
}

bool AdditivProcessor::startJourney( const JourneyEntry &entry )
{
    additiv::Journey j;
    const bool       read = entry.m_builtin >= 0 ? j.parse( additiv::builtinJourneyText( entry.m_builtin ) )
                                                 : j.load( entry.m_file.getFullPathName().toRawUTF8() );
    if ( !read || j.m_steps.empty() ) return false;
    bool known = false;
    for ( const additiv::JourneyStep &st : j.m_steps )
        known = known || additiv::findPreset( st.m_preset.c_str() ) >= 0 ||
                findUserPreset( juce::String( st.m_preset ) ) >= 0;
    if ( !known ) return false;
    if ( j.m_name.empty() ) j.m_name = entry.m_name.toStdString();
    m_journey = j;
    // The A/B morph holds the sound between its slots, the map plays the blend at its cursor: a
    // journey that loads presets under either would seem to do nothing (Noctuary's lesson), so both let go.
    setValue( static_cast<int>( ParamIdE::MORPH_ACTIVE ), 0.0f );
    leaveMap( false );
    // Seeded from the clock: the same journey runs differently every night.
    m_journeyPlayer.start( m_journey, static_cast<std::uint64_t>( juce::Time::currentTimeMillis() ) | 1ull, 0 );
    m_journeyLastTick = 0.0;
    journeyTick();   // the first step begins now
    return true;
}

void AdditivProcessor::journeyTick()
{
    const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
    if ( m_journeyLastTick <= 0.0 ) m_journeyLastTick = now;
    const double dt = juce::jlimit( 0.0, 5.0, now - m_journeyLastTick );
    m_journeyLastTick = now;
    if ( !m_journeyPlayer.running() ) return;
    additiv::JourneyStep st;
    double               glide = 60.0;
    if ( !m_journeyPlayer.advance( dt, st, glide ) ) return;
    const int index = additiv::findPreset( st.m_preset.c_str() );
    const int own =
        index < 0 ? findUserPreset( juce::String( st.m_preset ) ) : -1;   // one of the player's presets by that name
    if ( index >= 0 )
        applyPreset( index,
                     std::max( glide, 0.5 ) );   // a name the bank no longer has: nothing changes until the next step
    else if ( own >= 0 )
        applyUserPreset( own, std::max( glide, 0.5 ) );
    // A journey is listened to, not played: the conductor plays on through a preset made for the
    // hands (it would stop for the whole of that step).
    if ( index >= 0 || own >= 0 ) setValue( static_cast<int>( ParamIdE::BRAIN_ON ), 1.0f );
}

juce::String AdditivProcessor::journeyStatus() const
{
    if ( !m_journeyPlayer.running() ) return {};
    const int    n = static_cast<int>( m_journey.m_steps.size() );
    const int    step = m_journeyPlayer.step();
    juce::String s = juce::String( m_journey.m_name ) + "   " + juce::String( step + 1 ) + "/" + juce::String( n );
    if ( step >= 0 && step < n ) s << "   " << m_journey.m_steps[static_cast<size_t>( step )].m_preset;
    const float p = m_engine.morphProgress();
    if ( p < 1.0f ) s << "   gliding " << juce::roundToInt( 100.0f * p ) << " %";
    s << "   next in " << additiv::Journey::timeText( std::max( 0.0, m_journeyPlayer.remaining() ) );
    return s;
}

void AdditivProcessor::skipJourneyTo( int step )
{
    if ( !m_journeyPlayer.running() || m_journey.m_steps.empty() ) return;
    m_journeyPlayer.start( m_journey, static_cast<std::uint64_t>( juce::Time::currentTimeMillis() ) | 1ull, step );
    m_journeyLastTick = 0.0;
    journeyTick();   // the step begins now
}

void AdditivProcessor::journeyAddCurrent()
{
    const int cp = currentPreset();
    if ( cp < 0 && m_currentUser.isEmpty() ) return;
    additiv::JourneyStep st;
    st.m_preset = cp >= 0 ? std::string( additiv::preset( cp ).m_name ) : m_currentUser.toStdString();
    st.m_dwellLo = 480.0;
    st.m_dwellHi = 900.0;
    st.m_fadeLo = 90.0;
    st.m_fadeHi = 180.0;
    m_draft.m_steps.push_back( st );
}

bool AdditivProcessor::saveJourneyDraft( const juce::File &file )
{
    if ( m_draft.m_steps.empty() ) return false;
    additiv::Journey j = m_draft;
    j.m_name = file.getFileNameWithoutExtension().toStdString();
    file.getParentDirectory().createDirectory();
    if ( !j.save( file.getFullPathName().toRawUTF8() ) ) return false;
    m_draft = additiv::Journey();
    return true;
}

void AdditivProcessor::handleAsyncUpdate()
{
    const int index = m_pendingProgram.exchange( -1, std::memory_order_relaxed );
    if ( index >= 0 ) choosePreset( index );
}

bool AdditivProcessor::mpeController( int ch, int cc, int value )
{
    if ( cc == 99 || cc == 98 )
    {
        m_rpn[ch] = 0x3FFF;
        return false;
    }   // an NRPN: what follows is not ours
    if ( cc == 101 )
    {
        m_rpn[ch] = ( m_rpn[ch] & 0x7F ) | ( value << 7 );
        return true;
    }
    if ( cc == 100 )
    {
        m_rpn[ch] = ( m_rpn[ch] & 0x3F80 ) | value;
        return true;
    }
    if ( cc != 6 || m_rpn[ch] == 0x3FFF ) return false;
    if ( m_rpn[ch] == 0 )
    {   // the bend range
        if ( ch == 0 )
            m_masterBend = static_cast<float>( value );
        else
            setValue( static_cast<int>( ParamIdE::MPE_BEND ), static_cast<float>( std::max( value, 1 ) ) );
    }
    else if ( m_rpn[ch] == 6 && ch == 0 )
    {   // MPE's configuration message: the lower zone with `value` member channels
        setValue( static_cast<int>( ParamIdE::MPE_ON ), value > 0 ? 1.0f : 0.0f );
    }
    return true;
}

void AdditivProcessor::processBlock( juce::AudioBuffer<float> &buffer, juce::MidiBuffer &midi )
{
    juce::ScopedNoDenormals noDenormals;
    m_taps.processNextMidiBuffer( midi, 0, buffer.getNumSamples(), true );   // the keys the panel plays
    // MPE (the lower zone): with it on, a note on channels 2 .. 16 has that channel's bend, pressure
    // and slide (CC 74) for its own; channel 1 is the master, its bend moves every played note.
    const bool  mpe = value( static_cast<int>( ParamIdE::MPE_ON ) ) >= 0.5f;
    const float memberBend = value( static_cast<int>( ParamIdE::MPE_BEND ) );
    for ( const auto meta : midi )
    {
        const juce::MidiMessage m = meta.getMessage();
        const int               pos = meta.samplePosition;
        const int               ch = std::clamp( m.getChannel() - 1, 0, 15 );
        const bool              member = mpe && ch >= 1;
        if ( m.isNoteOn() )
        {
            m_held.reset( static_cast<size_t>( m.getNoteNumber() ) );
            m_engine.noteOn( m.getNoteNumber(), m.getFloatVelocity(), pos, member ? ch : -1 );
        }
        else if ( m.isNoteOff() )
        {
            // The hold pedal keeps a note sounding until it comes up (the family's CC 64).
            if ( m_sustain )
                m_held.set( static_cast<size_t>( m.getNoteNumber() ) );
            else
                m_engine.noteOff( m.getNoteNumber(), pos, member ? ch : -1 );
        }
        else if ( m.isAllNotesOff() || m.isAllSoundOff() )
        {
            m_held.reset();
            m_engine.allNotesOff();
        }
        else if ( m.isPitchWheel() )
        {
            const float x = static_cast<float>( m.getPitchWheelValue() - 8192 ) / 8192.0f;
            if ( member )
                m_engine.setChannelBend( ch, x * memberBend );
            else
                m_engine.setChannelBend( -1, x * m_masterBend );
        }
        else if ( m.isController() && mpeController( ch, m.getControllerNumber(), m.getControllerValue() ) )
        {
            // an RPN: taken
        }
        else if ( m.isController() && member && m.getControllerNumber() == 74 )
        {
            m_engine.setChannelSlide( ch, static_cast<float>( m.getControllerValue() ) /
                                              127.0f );   // a note's slide (never learned)
        }
        else if ( m.isController() )
        {
            const int   cc = m.getControllerNumber();
            const float x = static_cast<float>( m.getControllerValue() ) / 127.0f;
            const int   waiting = m_learning.exchange( -1, std::memory_order_relaxed );
            if ( waiting >= 0 && cc != 64 )
            {
                forget( waiting );
                m_ccMap[static_cast<size_t>( cc )].store( waiting, std::memory_order_relaxed );
            }
            else if ( waiting >= 0 )
            {
                m_learning.store( waiting, std::memory_order_relaxed );   // the pedal is never learned
            }
            const int target = m_ccMap[static_cast<size_t>( cc )].load( std::memory_order_relaxed );
            if ( cc == 0 && target < 0 )
            {
                m_bankGroup = m.getControllerValue() % additiv::NumPresetGroups;   // bank select: a group of the bank
            }
            else if ( target >= 0 )
            {
                if ( auto *p = parameter( target ) ) p->setValueNotifyingHost( x );
            }
            else if ( cc == 64 )
            {
                const bool down = m.getControllerValue() >= 64;
                if ( m_sustain && !down )
                    for ( int n = 0; n < 128; ++n )
                        if ( m_held.test( static_cast<size_t>( n ) ) ) m_engine.noteOff( n, pos );
                if ( !down ) m_held.reset();
                m_sustain = down;
            }
            else if ( cc == 74 )
            {
                if ( auto *p = parameter( static_cast<int>( ParamIdE::CUTOFF ) ) )
                    p->setValueNotifyingHost( x );   // the family's filter controller
            }
            else if ( cc == 1 )
            {
                m_engine.setModWheel( x );   // a source of the matrix (the family leaves the wheel free)
            }
        }
        else if ( m.isProgramChange() )
        {
            // Bank select (CC 0) chooses the group, the program its preset: applied on the message thread.
            const int index = additiv::presetAt( m_bankGroup, m.getProgramChangeNumber() );
            if ( index >= 0 )
            {
                m_pendingProgram.store( index, std::memory_order_relaxed );
                triggerAsyncUpdate();
            }
        }
        else if ( m.isChannelPressure() && member )
        {
            m_engine.setChannelPressure( ch, static_cast<float>( m.getChannelPressureValue() ) /
                                                 127.0f );   // a note's pressure
        }
        else if ( m.isChannelPressure() )
        {
            m_engine.setPressure( static_cast<float>( m.getChannelPressureValue() ) / 127.0f );
        }
        else if ( m.isAftertouch() )
        {
            m_engine.setPressure( static_cast<float>( m.getAfterTouchValue() ) / 127.0f );
        }
    }
    for ( size_t i = 0; i < m_params.size(); ++i )
    {
        const float v = m_params[i]->convertFrom0to1( m_params[i]->getValue() );
        if ( v != m_pushed[i] )
        {
            m_engine.setParam( static_cast<ParamIdE>( i ), v );
            m_pushed[i] = v;
        }
    }
    // The family jam: following, the leader's root and energy; leading, this conductor's root and scale.
    if ( m_jam.role() == frame::Settings::JamRoleE::FOLLOW )
    {
        const frame::JamState js = m_jam.stateAt( -1.0 );
        m_engine.setJamRoot( js.m_root );
        m_engine.setJamEnergy( js.m_root >= 0 ? js.m_energy : 0.5f );
    }
    else
    {
        m_engine.setJamRoot( -1 );
        m_engine.setJamEnergy( 0.5f );
        if ( m_jam.role() == frame::Settings::JamRoleE::LEAD )
        {
            const int root = ( ( m_engine.brainRoot() % 12 ) + 12 ) % 12;
            const int scale = juce::roundToInt( value( static_cast<int>( ParamIdE::SCALE ) ) );
            if ( root != m_jamPostedRoot || scale != m_jamPostedScale )
            {
                m_jam.postKey( root, additiv::ScaleNames[std::clamp( scale, 0, additiv::NumScaleChoices - 1 )] );
                m_jamPostedRoot = root;
                m_jamPostedScale = scale;
            }
        }
    }
    const int n = buffer.getNumSamples();
    float    *L = buffer.getWritePointer( 0 );
    float    *R = nullptr;
    if ( buffer.getNumChannels() > 1 )
        R = buffer.getWritePointer( 1 );
    else
    {
        m_spare.setSize( 1, n, false, false, true );
        R = m_spare.getWritePointer( 0 );
    }
    m_engine.process( L, R, n );
    {   // the meter's peaks, the larger kept until the panel reads them
        float pl = 0.0f, pr = 0.0f;
        for ( int i = 0; i < n; ++i )
        {
            pl = std::max( pl, std::fabs( L[i] ) );
            pr = std::max( pr, std::fabs( R[i] ) );
        }
        if ( pl > m_peak[0].load( std::memory_order_relaxed ) ) m_peak[0].store( pl, std::memory_order_relaxed );
        if ( pr > m_peak[1].load( std::memory_order_relaxed ) ) m_peak[1].store( pr, std::memory_order_relaxed );
    }
    for ( int c = 2; c < buffer.getNumChannels(); ++c ) buffer.clear( c, 0, n );
    if ( m_muted )
        buffer.clear();   // ADDITIV_MUTE: the engine runs and is seen, nothing is heard (the manual's pictures)
}

juce::AudioProcessorEditor *AdditivProcessor::createEditor()
{
    return new AdditivEditor( *this );
}

void AdditivProcessor::getStateInformation( juce::MemoryBlock &dest )
{
    juce::XmlElement xml( "AdditivState" );
    xml.setAttribute( "version", ADDITIV_VERSION );
    for ( const ParamDesc &d : additiv::paramTable() )
    {
        const auto *p = m_params[static_cast<size_t>( d.m_id )];
        xml.setAttribute( d.m_key, static_cast<double>( p->convertFrom0to1( p->getValue() ) ) );
    }
    xml.createNewChildElement( "Mod" )->setAttribute( "rows", juce::String( m_engine.modText() ) );
    xml.setAttribute( "preset", m_current.load( std::memory_order_relaxed ) );
    if ( m_currentUser.isNotEmpty() ) xml.setAttribute( "userPreset", m_currentUser );
    if ( m_userLandFile != juce::File() ) xml.setAttribute( "landscapeFile", m_userLandFile.getFullPathName() );
    if ( m_route.count() > 0 )
    {   // the route being played or written (not whether it runs)
        std::vector<char> buf( static_cast<size_t>( m_route.count() ) * 200 + 64 );
        if ( m_route.write( buf.data(), buf.size() ) > 0 )
        {
            auto *r = xml.createNewChildElement( "Route" );
            r->setAttribute( "points", juce::String( buf.data() ) );
            r->setAttribute( "name", m_routeName );
            r->setAttribute( "loop", m_routeLoop );
        }
    }
    for ( int k = 0; k < 2; ++k )
    {   // the morph's slots, as their values and their matrix
        if ( m_slotValues[k].size() != static_cast<size_t>( additiv::NumParams ) ) continue;
        auto *s = xml.createNewChildElement( "Slot" );
        s->setAttribute( "index", k );
        s->setAttribute( "name", m_slotNames[k] );
        s->setAttribute( "rows", juce::String( m_slotMods[k] ) );
        for ( const ParamDesc &d : additiv::paramTable() )
            s->setAttribute( juce::String( "p_" ) + d.m_key,
                             static_cast<double>( m_slotValues[k][static_cast<size_t>( d.m_id )] ) );
    }
    for ( const Snapshot &s : m_snapshots )
    {   // the snapshots: their knobs and matrices (not their pictures)
        juce::String knobs;
        for ( const ParamDesc &d : additiv::paramTable() )
            knobs << d.m_key << "=" << juce::String( s.m_values[static_cast<size_t>( d.m_id )], 6 ) << " ";
        auto *e = xml.createNewChildElement( "Snapshot" );
        e->setAttribute( "values", knobs.trim() );
        e->setAttribute( "mod", juce::String( s.m_mod ) );
    }
    for ( int m = 0; m < 4; ++m )
    {   // the macros taught by example: per place its value and the knobs it moves
        const additiv::MacroLesson &l = m_lessons[static_cast<size_t>( m )];
        if ( l.m_count < 2 ) continue;
        auto *e = xml.createNewChildElement( "Lesson" );
        e->setAttribute( "macro", m );
        for ( int i = 0; i < l.m_count; ++i )
        {
            juce::String knobs;
            for ( int p = 0; p < additiv::NumParams; ++p )
                if ( l.m_delta[i][static_cast<size_t>( p )] != 0.0f )
                    knobs << additiv::paramTable()[static_cast<size_t>( p )].m_key << "="
                          << juce::String( l.m_delta[i][static_cast<size_t>( p )], 5 ) << ";";
            auto *place = e->createNewChildElement( "Place" );
            place->setAttribute( "at", static_cast<double>( l.m_pos[i] ) );
            place->setAttribute( "knobs", knobs );
        }
    }
    {   // the hall's memory, once it has heard anything (the instrument's, not a preset's)
        std::vector<float> m;
        double             age = 0.0;
        if ( m_engine.hallMemory( m, age ) )
        {
            auto *h = xml.createNewChildElement( "Hall" );
            h->setAttribute( "age", age );
            h->setAttribute( "name", m_hallName );
            h->setAttribute( "memory", hallText( m ) );
        }
    }
    {   // the dream journal (Phase 30): the dream's newest memories, packed, when it keeps one
        std::vector<std::uint8_t> j;
        if ( value( static_cast<int>( ParamIdE::DREAM_JOURNAL ) ) >= 0.5f && m_engine.dreamJournal( j ) )
        {
            juce::MemoryOutputStream packed;
            {
                juce::GZIPCompressorOutputStream z( packed, 9 );
                z.write( j.data(), j.size() );
            }
            xml.createNewChildElement( "Dream" )->setAttribute( "journal", packed.getMemoryBlock().toBase64Encoding() );
        }
    }
    for ( int cc = 0; cc < 128; ++cc )
    {
        const int id = m_ccMap[static_cast<size_t>( cc )].load( std::memory_order_relaxed );
        if ( id < 0 ) continue;
        auto *e = xml.createNewChildElement( "Learn" );
        e->setAttribute( "cc", cc );
        e->setAttribute( "key", additiv::paramDesc( static_cast<ParamIdE>( id ) ).m_key );
    }
    copyXmlToBinary( xml, dest );
}

void AdditivProcessor::setStateInformation( const void *data, int size )
{
    const std::unique_ptr<juce::XmlElement> xml = getXmlFromBinary( data, size );
    if ( !xml || !xml->hasTagName( "AdditivState" ) ) return;
    // A parameter the state does not name goes to its default, never stays where it was: otherwise
    // a state would sound different depending on what was loaded before it (the family's lesson).
    for ( const ParamDesc &d : additiv::paramTable() )
    {
        auto       *p = m_params[static_cast<size_t>( d.m_id )];
        const float v =
            xml->hasAttribute( d.m_key ) ? static_cast<float>( xml->getDoubleAttribute( d.m_key ) ) : d.m_def;
        p->setValueNotifyingHost( p->convertTo0to1( std::clamp( v, d.m_min, d.m_max ) ) );
    }
    m_current.store( xml->getIntAttribute( "preset", -1 ), std::memory_order_relaxed );
    m_currentUser = xml->getStringAttribute( "userPreset" );
    {   // the player's own landscape, read again from its file (a moved file: Aurora plays instead)
        const juce::File f( xml->getStringAttribute( "landscapeFile" ) );
        const float      choice = value( static_cast<int>( ParamIdE::LANDSCAPE ) );
        if ( f.existsAsFile() && f != m_userLandFile )
        {
            juce::String why;
            loadLandscape( f, false, &why );
            setValue( static_cast<int>( ParamIdE::LANDSCAPE ), choice );   // as the state chose, not forced to Your Own
        }
    }
    const juce::XmlElement *mod = xml->getChildByName( "Mod" );
    m_engine.setModText( mod != nullptr ? mod->getStringAttribute( "rows" ).toStdString()
                                        : std::string() );   // no rows named: an empty matrix
    if ( const juce::XmlElement *r = xml->getChildByName( "Route" ) )
    {
        additiv::Route rt;
        if ( rt.parse( r->getStringAttribute( "points" ).toRawUTF8() ) )
        {
            m_route = rt;
            m_routeName = r->getStringAttribute( "name" );
            m_routeLoop = r->getBoolAttribute( "loop", true );
        }
    }
    m_mapX = -1.0f;   // the cursor is blended anew
    for ( auto *s : xml->getChildWithTagNameIterator( "Slot" ) )
    {
        const int k = s->getIntAttribute( "index", -1 );
        if ( k < 0 || k > 1 ) continue;
        m_slotValues[k].assign( static_cast<size_t>( additiv::NumParams ), 0.0f );
        for ( const ParamDesc &d : additiv::paramTable() )
            m_slotValues[k][static_cast<size_t>( d.m_id )] =
                std::clamp( static_cast<float>( s->getDoubleAttribute( juce::String( "p_" ) + d.m_key, d.m_def ) ),
                            d.m_min, d.m_max );
        m_slotMods[k] = s->getStringAttribute( "rows" ).toStdString();
        m_slotNames[k] = s->getStringAttribute( "name" );
        m_engine.setMorphSlot( k, m_slotValues[k].data(), m_slotMods[k] );
    }
    m_snapshots.clear();   // the snapshots as the state kept them
    for ( auto *e : xml->getChildWithTagNameIterator( "Snapshot" ) )
    {
        Snapshot s;
        s.m_values.resize( static_cast<size_t>( additiv::NumParams ) );
        for ( const ParamDesc &d : additiv::paramTable() )
            s.m_values[static_cast<size_t>( d.m_id )] = value( static_cast<int>( d.m_id ) );
        for ( const juce::String &kv : juce::StringArray::fromTokens( e->getStringAttribute( "values" ), " ", "" ) )
            if ( const ParamDesc *d = additiv::findParam( kv.upToFirstOccurrenceOf( "=", false, false ).toRawUTF8() ) )
                s.m_values[static_cast<size_t>( d->m_id )] =
                    std::clamp( kv.fromFirstOccurrenceOf( "=", false, false ).getFloatValue(), d->m_min, d->m_max );
        s.m_mod = e->getStringAttribute( "mod" ).toStdString();
        if ( m_snapshots.size() < 8 ) m_snapshots.push_back( std::move( s ) );
    }
    for ( int m = 0; m < 4; ++m )
    {   // the taught macros as the state kept them; none named: none
        m_lessons[static_cast<size_t>( m )] = additiv::MacroLesson{};
        m_teachBase[static_cast<size_t>( m )].clear();
    }
    for ( auto *e : xml->getChildWithTagNameIterator( "Lesson" ) )
    {
        const int m = e->getIntAttribute( "macro", -1 );
        if ( m < 0 || m >= 4 ) continue;
        additiv::MacroLesson &l = m_lessons[static_cast<size_t>( m )];
        for ( auto *place : e->getChildWithTagNameIterator( "Place" ) )
        {
            std::array<float, additiv::NumParams> delta{};
            for ( const juce::String &kv :
                  juce::StringArray::fromTokens( place->getStringAttribute( "knobs" ), ";", "" ) )
                if ( const additiv::ParamDesc *d =
                         additiv::findParam( kv.upToFirstOccurrenceOf( "=", false, false ).toRawUTF8() ) )
                    if ( additiv::teachable( d->m_id ) )
                        delta[static_cast<size_t>( d->m_id )] =
                            std::clamp( kv.fromFirstOccurrenceOf( "=", false, false ).getFloatValue(), -1.0f, 1.0f );
            l.set( std::clamp( static_cast<float>( place->getDoubleAttribute( "at" ) ), 0.0f, 1.0f ), delta.data() );
        }
    }
    for ( int m = 0; m < 4; ++m ) m_engine.setMacroLesson( m, m_lessons[static_cast<size_t>( m )] );
    if ( const juce::XmlElement *d = xml->getChildByName( "Dream" ) )
    {   // the dream journal (Phase 30): the memories dreamt on (at most DreamJournalMost, a byte a band)
        juce::MemoryBlock packed;
        if ( packed.fromBase64Encoding( d->getStringAttribute( "journal" ) ) )
        {
            juce::MemoryInputStream           in( packed, false );
            juce::GZIPDecompressorInputStream z( in );
            juce::MemoryBlock                 raw;
            constexpr size_t                  most = 12 + static_cast<size_t>( additiv::DreamJournalMost ) *
                                                              ( 4 + static_cast<size_t>( additiv::DreamBands ) );
            z.readIntoMemoryBlock( raw, static_cast<juce::ssize_t>( most ) );
            const auto *b = static_cast<const std::uint8_t *>( raw.getData() );
            m_engine.setDreamJournal( std::vector<std::uint8_t>( b, b + raw.getSize() ) );
        }
    }
    {   // the hall's memory as the state kept it; a state without one: a new hall
        std::vector<float>      m;
        const juce::XmlElement *h = xml->getChildByName( "Hall" );
        if ( h != nullptr && hallFromText( h->getStringAttribute( "memory" ), m ) )
        {
            m_engine.setHallMemory( m, h->getDoubleAttribute( "age" ) );
            m_hallName = h->getStringAttribute( "name" );
        }
        else
        {
            newHall();
        }
    }
    for ( auto &c : m_ccMap ) c.store( -1, std::memory_order_relaxed );
    for ( auto *e : xml->getChildWithTagNameIterator( "Learn" ) )
    {
        const int        cc = e->getIntAttribute( "cc", -1 );
        const ParamDesc *d = additiv::findParam( e->getStringAttribute( "key" ).toRawUTF8() );
        if ( cc >= 0 && cc < 128 && d != nullptr )
            m_ccMap[static_cast<size_t>( cc )].store( static_cast<int>( d->m_id ), std::memory_order_relaxed );
    }
    // What "changed" is measured against: the preset the state's sound came from, when it is known.
    setReference();
    const int cp = m_current.load( std::memory_order_relaxed );
    if ( cp >= 0 && cp < additiv::numPresets() )
    {
        float       values[additiv::NumParams];
        std::string rows;
        additiv::presetValues( cp, values, &rows );
        m_reference.assign( values, values + additiv::NumParams );
        for ( int k = 0; k < additiv::NumParams; ++k )   // the listener's own are as the state has them
            if ( additiv::keptByPresets( static_cast<ParamIdE>( k ) ) )
                m_reference[static_cast<size_t>( k )] = value( k );
        m_referenceMod = modTextAsEngine( rows );
    }
    else if ( const int own = findUserPreset( m_currentUser ); own >= 0 )
    {
        m_reference = m_userPresets[static_cast<size_t>( own )].m_values;
        for ( int k = 0; k < additiv::NumParams; ++k )
            if ( additiv::keptByPresets( static_cast<ParamIdE>( k ) ) )
                m_reference[static_cast<size_t>( k )] = value( k );
        m_referenceMod = modTextAsEngine( m_userPresets[static_cast<size_t>( own )].m_mod );
    }
}

/** @brief JUCE's factory for the plugin and the standalone. @return a new processor */
juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter()
{
    return new AdditivProcessor();
}
