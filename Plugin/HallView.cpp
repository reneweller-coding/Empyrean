/**
 * @file HallView.cpp
 * @brief The Hall page (see HallView.h).
 * @author Rene Weller
 */
#include "HallView.h"
#include "Theme.h"

#include <algorithm>
#include <cmath>

using additiv::ParamIdE;

/**
 * @brief A band's effect on its modes' decay, octaves at Grow 1: the card's own curve (modalKernel).
 * @param[in] m  the memory, 1 the average
 * @return -0.5 .. 1.5
 */
static float grownOctaves( float m )
{
    return std::clamp( std::log2( 1.0f + std::max( m, 0.0f ) ) - 1.0f, -0.5f, 1.5f );
}

/**
 * @brief Seconds as "3 h 12 min", "12 min", "40 s".
 * @param[in] s  seconds
 * @return the text
 */
static juce::String ageText( double s )
{
    if ( s < 60.0 ) return juce::String( static_cast<int>( s ) ) + " s";
    const int minutes = static_cast<int>( s / 60.0 );
    if ( minutes < 60 ) return juce::String( minutes ) + " min";
    return juce::String( minutes / 60 ) + " h " + juce::String( minutes % 60 ) + " min";
}

HallView::HallView( AdditivProcessor &proc ) : m_proc( proc )
{
    m_memory.assign( static_cast<size_t>( additiv::HallBins ), 1.0f );
    m_save.onClick = [this] { save(); };
    m_load.onClick = [this] { load(); };
    m_new.setTooltip( "A new hall: everything it has heard forgotten, every band at the average" );
    m_new.onClick = [this] { m_proc.newHall(); };
    m_status.setColour( juce::Label::textColourId, additivui::skin().m_dim );
    m_status.setFont( juce::FontOptions( 12.0f ) );
    for ( juce::Component *c :
          { static_cast<juce::Component *>( &m_save ), static_cast<juce::Component *>( &m_load ),
            static_cast<juce::Component *>( &m_new ), static_cast<juce::Component *>( &m_status ) } )
        addAndMakeVisible( c );
    timerCallback();
    startTimerHz( 4 );
}

void HallView::timerCallback()
{
    // Not over a stroke, nor before the engine has taken it in (a block or two).
    if ( m_painting || juce::Time::getMillisecondCounterHiRes() - m_paintedAt < 400.0 ) return;
    m_proc.engine().hallMemory( m_memory, m_age );
    const juce::String s = statusText();
    if ( m_status.getText() != s ) m_status.setText( s, juce::dontSendNotification );
    repaint( plot().getSmallestIntegerContainer().expanded( 2 ) );
}

juce::String HallView::statusText() const
{
    const float  grow = m_proc.value( static_cast<int>( ParamIdE::HALL_GROW ) );
    juce::String s = m_proc.hallName().isNotEmpty() ? m_proc.hallName() : juce::String( "This hall" );
    s << ": " << ( m_age > 0.0 ? "heard for " + ageText( m_age ) : juce::String( "new, nothing heard yet" ) ) << ". ";
    if ( grow > 0.0f )
        s << "Growth: where it heard much, its modes ring up to " << juce::String( std::exp2( 1.5f * grow ), 1 )
          << " times longer.";
    else if ( grow < 0.0f )
        s << "Erosion: where it heard much, its modes ring up to " << juce::String( std::exp2( -1.5f * grow ), 1 )
          << " times shorter.";
    else
        s << "Grow at nothing: it neither learns nor changes.";
    return s;
}

juce::Rectangle<float> HallView::plot() const
{
    return juce::Rectangle<float>( 16.0f, 34.0f, static_cast<float>( getWidth() ) - 32.0f,
                                   static_cast<float>( getHeight() ) - 34.0f - 66.0f );
}

void HallView::resized()
{
    const int w = getWidth(), h = getHeight();
    m_status.setBounds( 12, h - 58, w - 24, 20 );
    m_save.setBounds( 16, h - 34, 28, 26 );
    m_load.setBounds( 48, h - 34, 28, 26 );
    m_new.setBounds( 84, h - 34, 90, 26 );
}

void HallView::mouseDown( const juce::MouseEvent &e )
{
    if ( !plot().contains( e.position ) || m_memory.size() != static_cast<size_t>( additiv::HallBins ) ) return;
    m_painting = true;
    // The memory acts only while the hall grows from it: Grow switched on (a little) for the painting to be heard.
    if ( m_proc.value( static_cast<int>( ParamIdE::HALL_GROW ) ) == 0.0f )
    {
        m_proc.setValue( static_cast<int>( ParamIdE::HALL_GROW ), 0.4f );
        m_note = "Grow set to 0.4 so that the painted memory is heard";
    }
    else
    {
        m_note.clear();
    }
    if ( m_proc.value( static_cast<int>( ParamIdE::MODAL_LEVEL ) ) <= 0.0f )
        m_note = "The modal hall is off (Modal Hall at 0): turn it up to hear what is painted";
    m_last = e.position;
    paintStroke( e.position, e.position );
}

void HallView::mouseDrag( const juce::MouseEvent &e )
{
    if ( !m_painting ) return;
    paintStroke( m_last, e.position );
    m_last = e.position;
}

void HallView::mouseUp( const juce::MouseEvent & )
{
    m_painting = false;
    m_paintedAt = juce::Time::getMillisecondCounterHiRes();
}

void HallView::paintStroke( juce::Point<float> from, juce::Point<float> to )
{
    const juce::Rectangle<float> r = plot();
    // Height to the memory a band needs for that many octaves of decay at Grow 1 (the curve's own scale, inverted).
    auto memoryAt = [&r]( float y )
    {
        const float oct = std::clamp( ( r.getBottom() - y ) / r.getHeight() * 2.0f - 0.5f, -0.5f, 1.5f );
        return std::exp2( oct + 1.0f ) - 1.0f;
    };
    auto binAt = [&r]( float x ) { return ( x - r.getX() ) / r.getWidth() * static_cast<float>( additiv::HallBins ); };
    const float     b0 = binAt( from.x ), b1 = binAt( to.x );
    const float     lo = std::min( b0, b1 ), hi = std::max( b0, b1 );
    constexpr float Brush = 10.0f;   // bands either side: about two semitones
    for ( int b = std::max( 0, static_cast<int>( lo - 3.0f * Brush ) );
          b < std::min( additiv::HallBins, static_cast<int>( hi + 3.0f * Brush ) + 1 ); ++b )
    {
        const float fb = static_cast<float>( b ) + 0.5f;
        const float t = hi > lo ? std::clamp( ( fb - b0 ) / ( b1 - b0 ), 0.0f, 1.0f ) : 0.0f;
        const float want = memoryAt( from.y + ( to.y - from.y ) * t );
        const float d = fb < lo ? lo - fb : ( fb > hi ? fb - hi : 0.0f );
        const float w = std::exp( -0.5f * ( d / Brush ) * ( d / Brush ) );
        float      &m = m_memory[static_cast<size_t>( b )];
        m += ( want - m ) * w;
    }
    m_proc.engine().setHallMemory( m_memory, m_age );
    repaint();
}

void HallView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( getLocalBounds().toFloat(), skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( getLocalBounds().toFloat().reduced( 0.5f ), skin.m_radius, 1.0f );
    const juce::Colour colour = skin.family( frame::FamilyE::MOTION );
    g.setColour( colour );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.drawText( "Memory: what the hall has heard, band by band (up: its modes ring longer as it grows)", 16, 4,
                getWidth() - 24, 20, juce::Justification::centredLeft );

    const juce::Rectangle<float> r = plot();
    if ( r.getWidth() < 20.0f || r.getHeight() < 20.0f ) return;
    // -0.5 .. 1.5 octaves of decay at Grow 1: the average (0) a quarter up from the bottom.
    auto yOf = [&r]( float oct ) { return r.getBottom() - ( oct + 0.5f ) / 2.0f * r.getHeight(); };
    auto xOf = [&r]( float hz ) { return r.getX() + std::log2( hz / 20.0f ) / 9.96578428f * r.getWidth(); };
    g.setFont( juce::FontOptions( 11.0f ) );
    for ( const float hz : { 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f } )
    {
        const float x = xOf( hz );
        g.setColour( skin.m_faint.withAlpha( 0.5f ) );
        g.drawVerticalLine( static_cast<int>( x ), r.getY(), r.getBottom() );
        g.setColour( skin.m_dim );
        g.drawText( hz >= 1000.0f ? juce::String( static_cast<int>( hz / 1000.0f ) ) + "k"
                                  : juce::String( static_cast<int>( hz ) ),
                    static_cast<int>( x ) - 20, static_cast<int>( r.getBottom() ) + 1, 40, 14,
                    juce::Justification::centred );
    }
    g.setColour( skin.m_faint );
    g.drawHorizontalLine( static_cast<int>( yOf( 0.0f ) ), r.getX(), r.getRight() );
    if ( m_memory.size() != static_cast<size_t>( additiv::HallBins ) ) return;
    juce::Path  curve, fill;
    const float base = yOf( 0.0f );
    fill.startNewSubPath( r.getX(), base );
    for ( int b = 0; b < additiv::HallBins; ++b )
    {
        const float x =
            r.getX() + ( static_cast<float>( b ) + 0.5f ) / static_cast<float>( additiv::HallBins ) * r.getWidth();
        const float y = yOf( grownOctaves( m_memory[static_cast<size_t>( b )] ) );
        if ( b == 0 )
            curve.startNewSubPath( x, y );
        else
            curve.lineTo( x, y );
        fill.lineTo( x, y );
    }
    fill.lineTo( r.getRight(), base );
    fill.closeSubPath();
    const bool learning = m_proc.value( static_cast<int>( ParamIdE::HALL_GROW ) ) != 0.0f;
    g.setColour( colour.withAlpha( learning ? 0.25f : 0.12f ) );
    g.fillPath( fill );
    g.setColour( learning ? colour : colour.withAlpha( 0.5f ) );
    g.strokePath( curve, juce::PathStrokeType( 1.5f ) );
    g.setColour( skin.m_dim );
    g.setFont( juce::FontOptions( 11.0f ) );
    g.drawText( m_note.isNotEmpty()
                    ? m_note
                    : juce::String( "Paint the memory with the mouse: up, the modes there ring longer; down, shorter" ),
                r.withHeight( 14.0f ).translated( 0.0f, 2.0f ), juce::Justification::centredRight );
}

void HallView::save()
{
    const juce::File dir = AdditivProcessor::hallFolder();
    dir.createDirectory();
    const juce::String name = m_proc.hallName().isNotEmpty() ? m_proc.hallName() : juce::String( "My Hall" );
    m_chooser = std::make_unique<juce::FileChooser>( "Save the hall", dir.getChildFile( name + ".hall" ), "*.hall" );
    m_chooser->launchAsync( juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting,
                            [this]( const juce::FileChooser &fc )
                            {
                                const juce::File f = fc.getResult();
                                if ( f == juce::File() ) return;
                                if ( !m_proc.saveHall( f.withFileExtension( "hall" ) ) )
                                    juce::AlertWindow::showMessageBoxAsync( juce::MessageBoxIconType::WarningIcon,
                                                                            "Hall",
                                                                            "The hall could not be written there." );
                            } );
}

void HallView::load()
{
    const juce::File dir = AdditivProcessor::hallFolder();
    dir.createDirectory();
    m_chooser = std::make_unique<juce::FileChooser>( "Load a hall", dir, "*.hall" );
    m_chooser->launchAsync( juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                            [this]( const juce::FileChooser &fc )
                            {
                                const juce::File f = fc.getResult();
                                if ( f == juce::File() ) return;
                                if ( !m_proc.loadHall( f ) )
                                    juce::AlertWindow::showMessageBoxAsync( juce::MessageBoxIconType::WarningIcon,
                                                                            "Hall", "That file holds no hall." );
                            } );
}

juce::String HallView::describe() const
{
    juce::String s( "Hall memory: " );
    s << statusText() << "\n";
    // The memory in words: its brightest and darkest regions.
    if ( m_memory.size() == static_cast<size_t>( additiv::HallBins ) && m_age > 0.0 )
    {
        int hi = 0, lo = 0;
        for ( int b = 1; b < additiv::HallBins; ++b )
        {
            if ( m_memory[static_cast<size_t>( b )] > m_memory[static_cast<size_t>( hi )] ) hi = b;
            if ( m_memory[static_cast<size_t>( b )] < m_memory[static_cast<size_t>( lo )] ) lo = b;
        }
        auto hz = []( int b ) { return 20.0 * std::exp2( ( b + 0.5 ) / additiv::HallBins * 9.96578428 ); };
        s << "    most heard near " << juce::String( hz( hi ), 0 ) << " Hz ("
          << juce::String( m_memory[static_cast<size_t>( hi )], 1 ) << " times the average), least near "
          << juce::String( hz( lo ), 0 ) << " Hz\n";
    }
    return s;
}

// ---- the page

HallPage::HallPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live )
    : m_params( proc, actions,
                { { "Modal Hall",
                    frame::FamilyE::MOTION,
                    { ParamIdE::MODAL_LEVEL, ParamIdE::MODAL_DECAY, ParamIdE::MODAL_DAMP, ParamIdE::MODAL_TUNE,
                      ParamIdE::MODAL_SPREAD, ParamIdE::MODAL_DRIFT, ParamIdE::MODAL_WIDTH, ParamIdE::MODAL_LOW,
                      ParamIdE::MODAL_BREATH, ParamIdE::MODAL_PAINT, ParamIdE::MODAL_FREEZE },
                    { ParamIdE::MODAL_LEVEL } },
                  { "Growth",
                    frame::FamilyE::ENVELOPE,
                    { ParamIdE::HALL_GROW, ParamIdE::HALL_MEMORY },
                    { ParamIdE::HALL_GROW } } },
                live ),
      m_view( proc )
{
    addAndMakeVisible( m_params );
    addAndMakeVisible( m_view );
}

/** @brief The memory's box: its height. */
constexpr int HallViewHeight = 220;

int HallPage::heightFor( int w ) const
{
    return m_params.heightFor( w ) + HallViewHeight + 10;
}

void HallPage::resized()
{
    const int top = m_params.heightFor( getWidth() );
    m_params.setBounds( 0, 0, getWidth(), top );
    m_view.setBounds( 10, top, getWidth() - 20, std::max( HallViewHeight, getHeight() - 10 - top ) );
}

juce::String HallPage::describe() const
{
    return m_params.describe() + m_view.describe() +
           "\nThe modal hall is a room of thousands of resonances the voices excite. With Grow it remembers what it hears: "
           "band by band over the octaves, a step of Memory (a minute to four hours) towards the spectrum while anything sounds. "
           "Growth lets the bands it heard much ring longer -- a room that grows into the music played in it; erosion lets them "
           "ring shorter, so it wears away where it was used and the rest rings on. The memory stays through every preset, is "
           "kept with the session, and a hall can be saved, loaded, or begun anew. The curve can be painted with the mouse: above the line the memory rises, below it falls (with Grow at nothing, painting sets it to 0.4, so that it is heard).\n";
}
