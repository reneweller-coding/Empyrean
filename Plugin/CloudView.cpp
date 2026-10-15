/**
 * @file CloudView.cpp
 * @brief The cloud picture (CloudView.h).
 * @author Rene Weller
 */
#include "CloudView.h"

#include "Theme.h"

#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Spectrum.h"

#include <cmath>

using additiv::ParamIdE;

namespace
{

constexpr float Handle = 9.0f;   ///< how near a handle a click takes it, pixels

/** @brief The axis' span: 20 Hz to 20 kHz, in octaves. */
const float Span = std::log2( 1000.0f );

}   // namespace

CloudView::CloudView( AdditivProcessor &proc ) : m_proc( proc )
{
    static const char *const MacroNames[4] = { "Macro 1", "Macro 2", "Macro 3", "Macro 4" };
    for ( int i = 0; i < 4; ++i )
    {
        juce::Slider &s = m_macros[static_cast<size_t>( i )];
        s.setSliderStyle( juce::Slider::RotaryHorizontalVerticalDrag );
        s.setTextBoxStyle( juce::Slider::NoTextBox, true, 0, 0 );
        s.setColour( juce::Slider::rotarySliderFillColourId, additivui::skin().family( frame::FamilyE::ENVELOPE ) );
        s.setTooltip( juce::String( MacroNames[i] ) + ": whatever the matrix lets it move" );
        const int id = static_cast<int>( ParamIdE::MACRO1 ) + i;
        if ( auto *p = m_proc.parameter( id ) )
            m_macroLinks.push_back( std::make_unique<juce::SliderParameterAttachment>( *p, s ) );
        juce::Label &l = m_macroNames[static_cast<size_t>( i )];
        l.setText( MacroNames[i], juce::dontSendNotification );
        l.setColour( juce::Label::textColourId, additivui::skin().m_dim );
        l.setFont( juce::FontOptions( 12.0f ) );
        l.setJustificationType( juce::Justification::centredLeft );
        addChildComponent( s );
        addChildComponent( l );
    }
}

CloudView::~CloudView()
{
    if ( m_playing >= 0 ) m_proc.tapNote( m_playing, 0.0f );
}

float CloudView::xOf( float hz, juce::Rectangle<float> area )
{
    const float u = std::log2( std::max( hz, 20.0f ) / 20.0f ) / Span;
    return area.getX() + juce::jlimit( 0.0f, 1.0f, u ) * area.getWidth();
}

float CloudView::hzAt( float x, juce::Rectangle<float> area )
{
    const float u = juce::jlimit( 0.0f, 1.0f, ( x - area.getX() ) / std::max( area.getWidth(), 1.0f ) );
    return 20.0f * std::exp2( u * Span );
}

int CloudView::keyAt( float x ) const
{
    const float hz = hzAt( x, m_spec );
    return juce::jlimit( 21, 108, juce::roundToInt( 69.0f + 12.0f * std::log2( hz / 440.0f ) ) );
}

juce::Image CloudView::thumbnail( int w, int h ) const
{
    juce::Image    img( juce::Image::RGB, std::max( w, 1 ), std::max( h, 1 ), true );
    juce::Graphics g( img );
    g.fillAll( additivui::skin().m_bg );
    if ( m_dots.isValid() ) g.drawImage( m_dots, img.getBounds().toFloat(), juce::RectanglePlacement::stretchToFit );
    return img;
}

void CloudView::setCanvas( bool on )
{
    m_canvas = on;
    for ( juce::Slider &s : m_macros ) s.setVisible( on );
    for ( juce::Label &l : m_macroNames ) l.setVisible( on );
    if ( !on && m_playing >= 0 )
    {
        m_proc.tapNote( m_playing, 0.0f );
        m_playing = -1;
    }
    resized();
    repaint();
}

void CloudView::resized()
{
    // The macros under the keys, in the canvas: four knobs side by side on the left, each with its name beside it.
    const int size = 54, y = getHeight() - size - 8, w = 190;
    for ( int i = 0; i < 4; ++i )
    {
        m_macros[static_cast<size_t>( i )].setBounds( 20 + i * w, y, size, size );
        m_macroNames[static_cast<size_t>( i )].setBounds( 20 + i * w + size + 4, y + size / 2 - 9, w - size - 10, 18 );
    }
}

void CloudView::paintKeys( juce::Graphics &g, juce::Rectangle<float> keys, juce::Rectangle<float> spec )
{
    const frame::Skin &skin = additivui::skin();
    const juce::Colour bySource[3] = { skin.m_accent, additivui::colour::Violet,
                                       skin.family( frame::FamilyE::SPACE ) };   // played, conductor, Brain 2
    // Each sounding key once: its loudest voice, near and loud brightest.
    float level[128] = {};
    int   source[128] = {};
    for ( const additiv::VoiceView &v : m_views )
    {
        if ( v.m_note < 0 || v.m_note > 127 || v.m_hz <= 0.0f || v.m_gain <= 0.0f || v.m_source > 2 ) continue;
        const float l = juce::jlimit(
            0.0f, 1.0f, std::sqrt( v.m_gain ) * ( 1.0f - 0.6f * juce::jlimit( 0.0f, 1.0f, v.m_distance ) ) );
        if ( l > level[v.m_note] )
        {
            level[v.m_note] = std::max( l, 0.05f );
            source[v.m_note] = juce::jlimit( 0, 2, v.m_source );
        }
    }
    constexpr int First = 21, Last = 108;   // A0 .. C8
    auto          edge = [&]( float n ) { return xOf( 440.0f * std::exp2( ( n - 69.0f ) / 12.0f ), spec ); };
    // As a keyboard: the white keys one strip, a line where two white keys meet (B-C, E-F), the
    // black keys short and dark over it; a sounding key lit in its part.
    const float left = edge( static_cast<float>( First ) - 0.5f ), right = edge( static_cast<float>( Last ) + 0.5f );
    const float blackH = std::round( keys.getHeight() * 0.62f );
    g.setColour( skin.m_dim.interpolatedWith( skin.m_faint, 0.45f ) );
    g.fillRect( left, keys.getY(), right - left, keys.getHeight() );
    for ( int n = First; n <= Last; ++n )
    {
        const float x0 = edge( static_cast<float>( n ) - 0.5f ), x1 = edge( static_cast<float>( n ) + 0.5f );
        const bool  black = juce::MidiMessage::isMidiNoteBlack( n );
        if ( black )
        {
            g.setColour( level[n] > 0.0f ? skin.m_bg.interpolatedWith( bySource[source[n]], 0.55f + 0.45f * level[n] )
                                         : skin.m_bg );
            g.fillRect( x0, keys.getY(), x1 - x0, blackH );
            continue;
        }
        if ( level[n] > 0.0f )
        {
            // A white key reaches under its black neighbours' halves; lit, it is its own semitone.
            g.setColour( skin.m_faint.interpolatedWith( bySource[source[n]], 0.55f + 0.45f * level[n] ) );
            g.fillRect( x0, keys.getY(), x1 - x0, keys.getHeight() );
        }
        if ( n > First && !juce::MidiMessage::isMidiNoteBlack( n - 1 ) )
        {   // B|C, E|F
            g.setColour( skin.m_bg.withAlpha( 0.9f ) );
            g.fillRect( x0 - 0.5f, keys.getY(), 1.0f, keys.getHeight() );
        }
    }
    // MPE at the keys: a played note's pressure as a bar above its key, its slide as a dot in it, its bend
    // as a line from the key to where it sounds.
    for ( const additiv::VoiceView &v : m_views )
    {
        if ( v.m_source != 0 || v.m_note < First || v.m_note > Last || v.m_hz <= 0.0f ) continue;
        const float x0 = edge( static_cast<float>( v.m_note ) - 0.5f ),
                    x1 = edge( static_cast<float>( v.m_note ) + 0.5f ), xc = 0.5f * ( x0 + x1 );
        const float top = keys.getY() - 2.0f;
        if ( v.m_press > 0.01f )
        {
            const float h = 16.0f * juce::jlimit( 0.0f, 1.0f, v.m_press );
            g.setColour( skin.m_accent.withAlpha( 0.75f ) );
            g.fillRect( juce::Rectangle<float>::leftTopRightBottom( x0, top - h, std::max( x1, x0 + 3.0f ), top ) );
        }
        if ( std::abs( v.m_slide - 0.5f ) > 0.02f )
        {
            g.setColour( additivui::colour::Root );
            g.fillEllipse( xc - 2.5f, top - 2.0f - 16.0f * juce::jlimit( 0.0f, 1.0f, v.m_slide ) - 2.5f, 5.0f, 5.0f );
        }
        if ( std::abs( v.m_bend ) > 0.05f )
        {
            const float xb = edge( static_cast<float>( v.m_note ) + v.m_bend );
            g.setColour( skin.m_accent );
            g.drawLine( xc, top + 1.0f, xb, top + 1.0f, 1.5f );
            g.fillEllipse( xb - 2.0f, top - 1.0f, 4.0f, 4.0f );
        }
    }
    // The conductor's root, while it conducts.
    const int root = m_proc.engine().brainRoot();
    if ( m_proc.value( static_cast<int>( ParamIdE::BRAIN_ON ) ) >= 0.5f && root >= First && root <= Last )
    {
        const float x =
            0.5f * ( edge( static_cast<float>( root ) - 0.5f ) + edge( static_cast<float>( root ) + 0.5f ) );
        juce::Path tri;
        tri.addTriangle( x - 3.5f, keys.getY() - 1.0f, x + 3.5f, keys.getY() - 1.0f, x, keys.getY() + 4.0f );
        g.setColour( additivui::colour::Root );
        g.fillPath( tri );
    }
    // The foundation (Foundation): its key underlined in the root's colour, the floor the chord stands on.
    const int floorNote = m_proc.engine().foundationNow();
    if ( floorNote >= First && floorNote <= Last )
    {
        const float x0 = edge( static_cast<float>( floorNote ) - 0.5f ),
                    x1 = edge( static_cast<float>( floorNote ) + 0.5f );
        g.setColour( additivui::colour::Root );
        g.fillRoundedRectangle( x0 - 1.0f, keys.getBottom() + 1.0f, x1 - x0 + 2.0f, 2.5f, 1.0f );
    }
    // The names, low to high, each in its colour.
    const juce::Font font( juce::FontOptions( 11.0f ) );
    g.setFont( font );
    float x = spec.getX() + 4.0f;
    for ( int n = 0; n < 128; ++n )
    {
        if ( level[n] <= 0.0f ) continue;
        const juce::String name = juce::MidiMessage::getMidiNoteName( n, true, true, 4 );
        const float        w = juce::GlyphArrangement::getStringWidth( font, name ) + 8.0f;
        if ( x + w > spec.getRight() ) break;
        g.setColour( bySource[source[n]].withAlpha( 0.45f + 0.55f * level[n] ) );
        g.drawText( name, juce::Rectangle<float>( x, spec.getY() + 2.0f, w, 14.0f ), juce::Justification::centredLeft );
        x += w;
    }
}

void CloudView::refreshLandscape()
{
    int        choice = juce::jlimit( 0, additiv::NumLandscapeChoices - 1,
                                      juce::roundToInt( m_proc.value( static_cast<int>( ParamIdE::LANDSCAPE ) ) ) );
    const bool own = choice == additiv::UserLandscape && !m_proc.userLandscape().m_amps.empty();
    if ( choice == additiv::UserLandscape && !own ) choice = 0;   // none loaded: the engine plays Aurora
    if ( choice == m_landChoice && ( !own || m_landVersion == m_proc.userLandscapeVersion() ) ) return;
    m_landChoice = choice;
    m_landVersion = m_proc.userLandscapeVersion();
    const additiv::Landscape l = own ? m_proc.userLandscape() : additiv::factoryLandscape( choice );
    m_landLoop = l.m_rows * l.m_rowSeconds;
    // About 256 x 128 points are enough for a picture this size.
    const int sx = std::max( 1, l.m_rows / 256 ), sy = std::max( 1, l.m_columns / 128 );
    const int w = std::max( 1, l.m_rows / sx ), h = std::max( 1, l.m_columns / sy );
    m_land = juce::Image( juce::Image::RGB, w, h, false );
    const juce::Colour lo = additivui::colour::Bg, hi = additivui::colour::Accent;
    for ( int x = 0; x < w; ++x )
        for ( int y = 0; y < h; ++y )
        {
            const float a =
                l.m_amps[static_cast<size_t>( x * sx ) * l.m_columns + static_cast<size_t>( ( h - 1 - y ) * sy )];
            m_land.setPixelAt( x, y, lo.interpolatedWith( hi, additiv::ampToBrightness( a, 40.0f ) ) );
        }
}

std::vector<ParamIdE> CloudView::paramsOf( HandleE h )
{
    switch ( h )
    {
    case HandleE::FILTER: return { ParamIdE::CUTOFF, ParamIdE::RESONANCE };
    case HandleE::AIR_CENTRE: return { ParamIdE::AIR_CENTER };
    case HandleE::AIR_EDGE: return { ParamIdE::AIR_WIDTH };
    case HandleE::BRIGHTNESS: return { ParamIdE::BRIGHTNESS };
    case HandleE::REGISTER_LOW: return { ParamIdE::BRAIN_LOW };
    case HandleE::REGISTER_HIGH: return { ParamIdE::BRAIN_HIGH };
    case HandleE::HALL_LOWEST: return { ParamIdE::MODAL_LOW };
    default: return {};
    }
}

CloudView::HandleE CloudView::handleAt( juce::Point<float> p ) const
{
    if ( m_canvas ) return HandleE::NONE;
    HandleE best = HandleE::NONE;
    float   nearest = Handle;
    for ( int i = 1; i < static_cast<int>( HandleE::COUNT ); ++i )
    {
        if ( !m_shown[static_cast<size_t>( i )] ) continue;
        const float d = p.getDistanceFrom( m_handles[static_cast<size_t>( i )] );
        if ( d < nearest )
        {
            nearest = d;
            best = static_cast<HandleE>( i );
        }
    }
    return best;
}

float CloudView::referenceHz() const
{
    float lowest = 0.0f;
    for ( const additiv::VoiceView &v : m_views )
        if ( v.m_hz > 0.0f && v.m_gain > 0.0f && v.m_source <= 2 && ( lowest <= 0.0f || v.m_hz < lowest ) )
            lowest = v.m_hz;
    return lowest > 0.0f ? lowest : 130.81f;
}

void CloudView::mouseDown( const juce::MouseEvent &e )
{
    m_held = handleAt( e.position );
    if ( m_held != HandleE::NONE )
    {
        for ( ParamIdE id : paramsOf( m_held ) )
            if ( auto *p = m_proc.parameter( static_cast<int>( id ) ) ) p->beginChangeGesture();
        m_heldResonance = value( ParamIdE::RESONANCE );
        m_heldY = e.position.y;
        mouseDrag( e );
        return;
    }
    if ( m_canvas && m_spec.contains( e.position ) && e.mods.isLeftButtonDown() )
    {   // the canvas plays the key under the mouse
        m_playing = keyAt( e.position.x );
        m_proc.tapNote(
            m_playing, juce::jlimit( 0.2f, 1.0f,
                                     1.0f - ( e.position.y - m_spec.getY() ) / std::max( m_spec.getHeight(), 1.0f ) ) );
    }
}

void CloudView::mouseDrag( const juce::MouseEvent &e )
{
    const float hz = hzAt( e.position.x, m_spec );
    switch ( m_held )
    {
    case HandleE::FILTER:
        set( ParamIdE::CUTOFF, hz );
        set( ParamIdE::RESONANCE, m_heldResonance + ( m_heldY - e.position.y ) / 160.0f );
        break;
    case HandleE::AIR_CENTRE: set( ParamIdE::AIR_CENTER, hz / referenceHz() ); break;
    case HandleE::AIR_EDGE:
    {
        const float centre = referenceHz() * value( ParamIdE::AIR_CENTER );
        set( ParamIdE::AIR_WIDTH, 1200.0f * std::log2( std::max( hz, centre * 1.01f ) / centre ) );
        break;
    }
    case HandleE::BRIGHTNESS:
        set( ParamIdE::BRIGHTNESS, std::sqrt( std::max( 0.0f, hz / referenceHz() - 1.0f ) / 200.0f ) );
        break;
    case HandleE::REGISTER_LOW:
        set( ParamIdE::BRAIN_LOW,
             static_cast<float>(
                 std::min( keyAt( e.position.x ), juce::roundToInt( value( ParamIdE::BRAIN_HIGH ) ) - 1 ) ) );
        break;
    case HandleE::REGISTER_HIGH:
        set( ParamIdE::BRAIN_HIGH,
             static_cast<float>(
                 std::max( keyAt( e.position.x ), juce::roundToInt( value( ParamIdE::BRAIN_LOW ) ) + 1 ) ) );
        break;
    case HandleE::HALL_LOWEST: set( ParamIdE::MODAL_LOW, hz ); break;
    default:
        if ( m_playing >= 0 )
        {   // the canvas: on to the next key, legato
            const int key = keyAt( e.position.x );
            if ( key != m_playing )
            {
                m_proc.tapNote( m_playing, 0.0f );
                m_playing = key;
                m_proc.tapNote( m_playing, juce::jlimit( 0.2f, 1.0f,
                                                         1.0f - ( e.position.y - m_spec.getY() ) /
                                                                    std::max( m_spec.getHeight(), 1.0f ) ) );
            }
        }
        return;
    }
    repaint();
}

void CloudView::mouseUp( const juce::MouseEvent &e )
{
    if ( m_held != HandleE::NONE )
    {
        for ( ParamIdE id : paramsOf( m_held ) )
            if ( auto *p = m_proc.parameter( static_cast<int>( id ) ) ) p->endChangeGesture();
        m_held = HandleE::NONE;
        repaint();
        return;
    }
    if ( m_playing >= 0 )
    {
        m_proc.tapNote( m_playing, 0.0f );
        m_playing = -1;
        return;
    }
    if ( m_canvas || !m_landArea.contains( e.position ) || e.getNumberOfClicks() > 1 ) return;
    m_chooser = std::make_unique<juce::FileChooser>(
        "Your own landscape: a picture (time left to right, low at the bottom) or a recording",
        m_proc.userLandscapeFile().existsAsFile() ? m_proc.userLandscapeFile()
                                                  : juce::File::getSpecialLocation( juce::File::userPicturesDirectory ),
        "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.gif;*.wav;*.flac" );
    m_chooser->launchAsync( juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                            [this]( const juce::FileChooser &fc )
                            {
                                const juce::File f = fc.getResult();
                                if ( f == juce::File() ) return;
                                juce::String why;
                                if ( !m_proc.loadLandscape( f, true, &why ) )
                                    juce::AlertWindow::showMessageBoxAsync( juce::MessageBoxIconType::WarningIcon,
                                                                            "Landscape",
                                                                            "This file could not be read: " + why );
                            } );
}

void CloudView::mouseMove( const juce::MouseEvent &e )
{
    const HandleE h = handleAt( e.position );
    if ( h != m_hover )
    {
        m_hover = h;
        setMouseCursor( h != HandleE::NONE ? juce::MouseCursor::DraggingHandCursor : juce::MouseCursor::NormalCursor );
        repaint();
    }
}

void CloudView::mouseDoubleClick( const juce::MouseEvent &e )
{
    if ( m_spec.contains( e.position ) && handleAt( e.position ) == HandleE::NONE && m_onDoubleClick )
        m_onDoubleClick();
}

void CloudView::refresh()
{
    for ( int i = 0; i < additiv::MaxViewVoices; ++i )
        m_views[static_cast<size_t>( i )] = m_proc.engine().voiceView( i );
    refreshLandscape();
    if ( m_proc.engine().partialSnapshot( m_snap, m_snapPer ) )
    {
        m_proc.engine().lifeSnapshot( m_lifeCells );
        if ( m_lifeCells.size() == static_cast<size_t>( additiv::LifeW * additiv::LifeH ) )
        {
            // The living field as a picture: across the spectrum (left low), along a cloud's members (up).
            if ( !m_life.isValid() ) m_life = juce::Image( juce::Image::RGB, additiv::LifeW, additiv::LifeH, false );
            const juce::Colour            lo = additivui::colour::Bg, hi = additivui::colour::Violet;
            const juce::Image::BitmapData px( m_life, juce::Image::BitmapData::writeOnly );
            for ( int y = 0; y < additiv::LifeH; ++y )
                for ( int x = 0; x < additiv::LifeW; ++x )
                    px.setPixelColour(
                        x, additiv::LifeH - 1 - y,
                        lo.interpolatedWith(
                            hi, std::sqrt( m_lifeCells[static_cast<size_t>( y ) * additiv::LifeW + x] ) ) );
        }
        if ( m_page == "Motion" ) m_proc.engine().rippleSnapshot( m_ripple );
    }
    if ( m_page == "Hall" ) m_proc.engine().hallMemory( m_hall, m_hallAge );
    if ( m_canvas )   // the macros' names follow what they move
        for ( int i = 0; i < 4; ++i )
        {
            const juce::String name = m_proc.macroName( i );
            if ( m_macroNames[static_cast<size_t>( i )].getText() != name )
                m_macroNames[static_cast<size_t>( i )].setText( name, juce::dontSendNotification );
        }
    if ( !m_dotsArea.isEmpty() ) plotDots( m_dotsArea.getWidth(), m_dotsArea.getHeight() );
    repaint();
}

void CloudView::plotDots( int w, int h )
{
    if ( w <= 0 || h <= 0 ) return;
    if ( m_dots.getWidth() != w || m_dots.getHeight() != h )
    {
        m_dots = juce::Image( juce::Image::ARGB, w, h, true );
        m_glow.assign( static_cast<size_t>( w ) * h * 3, 0.0f );
    }
    // The trails fade by a fifth each frame (about half a second to vanish), then the points come in:
    // each partial at its frequency and loudness (-100 .. -35 dB of a partial's amplitude), its
    // colour by whose note it is, as bright as it is loud. The canvas lets them linger longer.
    const float fade = m_canvas ? 0.9f : 0.8f;
    for ( float &x : m_glow ) x *= fade;
    const frame::Skin &skin = additivui::skin();
    const juce::Colour bySource[4] = {
        skin.m_accent, additivui::colour::Violet, skin.family( frame::FamilyE::SPACE ), skin.m_dim.brighter( 0.4f )
    };   // played, conductor, Brain 2, the memory's ghosts (pale grey)
    const int voices = m_snapPer > 0 ? static_cast<int>( m_snap.size() / 2 ) / m_snapPer : 0;
    for ( int v = 0; v < voices; ++v )
    {
        const int src =
            v < additiv::MaxViewVoices ? juce::jlimit( 0, 3, m_views[static_cast<size_t>( v )].m_source ) : 3;
        const juce::Colour c = bySource[src];
        const float        cr = c.getFloatRed(), cg = c.getFloatGreen(), cb = c.getFloatBlue();
        const float       *d = m_snap.data() + static_cast<size_t>( v ) * m_snapPer * 2;
        for ( int i = 0; i < m_snapPer; ++i )
        {
            const float hz = d[2 * i], a = d[2 * i + 1];
            if ( hz <= 20.0f || a <= 1.0e-6f ) continue;
            const float db = 20.0f * std::log10( a );
            const float u = std::log2( hz / 20.0f ) / Span, l = ( db + 100.0f ) / 65.0f;
            if ( u < 0.0f || u >= 1.0f || l <= 0.0f ) continue;
            const int x = static_cast<int>( u * static_cast<float>( w ) );
            const int y = juce::jlimit(
                0, h - 1, static_cast<int>( ( 1.0f - std::min( l, 1.0f ) ) * static_cast<float>( h - 1 ) ) );
            const float k = 0.25f + 0.75f * std::min( l, 1.0f );
            float      *p = m_glow.data() + ( static_cast<size_t>( y ) * w + x ) * 3;   // two pixels wide
            p[0] += cr * k;
            p[1] += cg * k;
            p[2] += cb * k;
            if ( x + 1 < w )
            {
                p[3] += 0.6f * cr * k;
                p[4] += 0.6f * cg * k;
                p[5] += 0.6f * cb * k;
            }
            if ( m_canvas )
            {   // the canvas is larger: a point three pixels across
                for ( const int dy : { -1, 1 } )
                    if ( y + dy >= 0 && y + dy < h )
                    {
                        float *q = m_glow.data() + ( static_cast<size_t>( y + dy ) * w + x ) * 3;
                        q[0] += 0.5f * cr * k;
                        q[1] += 0.5f * cg * k;
                        q[2] += 0.5f * cb * k;
                    }
                if ( x > 0 )
                {
                    p[-3] += 0.5f * cr * k;
                    p[-2] += 0.5f * cg * k;
                    p[-1] += 0.5f * cb * k;
                }
            }
        }
    }
    const juce::Image::BitmapData px( m_dots, juce::Image::BitmapData::writeOnly );
    for ( int y = 0; y < h; ++y )
        for ( int x = 0; x < w; ++x )
        {
            const float *p = m_glow.data() + ( static_cast<size_t>( y ) * w + x ) * 3;
            const float  m = std::max( { p[0], p[1], p[2] } );
            const float  s = m > 1.0f ? 1.0f / m : 1.0f;
            px.setPixelColour( x, y, juce::Colour::fromFloatRGBA( p[0] * s, p[1] * s, p[2] * s, std::min( 1.0f, m ) ) );
        }
}

void CloudView::paintHandle( juce::Graphics &g, HandleE h, juce::Point<float> at, juce::Colour c )
{
    const size_t i = static_cast<size_t>( h );
    m_handles[i] = at;
    m_shown[i] = true;
    const bool lit = m_held == h || m_hover == h;
    g.setColour( c.withAlpha( lit ? 0.35f : 0.18f ) );
    g.fillEllipse( at.x - 7.0f, at.y - 7.0f, 14.0f, 14.0f );
    g.setColour( lit ? c.brighter( 0.5f ) : c );
    g.drawEllipse( at.x - 5.0f, at.y - 5.0f, 10.0f, 10.0f, 1.6f );
}

void CloudView::paintOverlay( juce::Graphics &g, juce::Rectangle<float> spec, juce::Rectangle<float> keys )
{
    m_shown.fill( false );
    if ( m_canvas ) return;
    const frame::Skin &skin = additivui::skin();
    g.setFont( juce::FontOptions( 11.0f ) );
    auto note = [&](juce::Point<float> at, const juce::String& text, juce::Colour c) {   // a handle's value while it is held or hovered
        g.setColour(c);
        g.drawText(text, juce::Rectangle<float>(at.x + 10.0f, at.y - 18.0f, 220.0f, 14.0f), juce::Justification::centredLeft);
    };
    if ( m_page == "Cloud & Filter" )
    {
        const int          model = juce::roundToInt( value( ParamIdE::FILTER_MODEL ) );
        const juce::Colour c = skin.family( frame::FamilyE::FILTER );
        if ( model > 0 )
        {
            // The filter's response over the axis: +18 dB at the top, -48 at the bottom.
            const float cutoff = value( ParamIdE::CUTOFF ), res = value( ParamIdE::RESONANCE );
            const float q = additiv::filterQ( model, res ), vowel = additiv::vowelOf( cutoff );
            auto        yOf = [&]( float gain )
            {
                const float db = juce::jlimit( -48.0f, 18.0f, 20.0f * std::log10( std::max( gain, 1.0e-6f ) ) );
                return spec.getY() + ( 18.0f - db ) / 66.0f * spec.getHeight();
            };
            juce::Path curve, fill;
            for ( float x = spec.getX(); x <= spec.getRight(); x += 2.0f )
            {
                const float y = yOf( additiv::filterResponse( model, hzAt( x, spec ), cutoff, q, vowel ) );
                if ( x == spec.getX() )
                {
                    curve.startNewSubPath( x, y );
                    fill.startNewSubPath( x, spec.getBottom() );
                }
                else
                    curve.lineTo( x, y );
                fill.lineTo( x, y );
            }
            fill.lineTo( spec.getRight(), spec.getBottom() );
            fill.closeSubPath();
            g.setColour( c.withAlpha( 0.07f ) );
            g.fillPath( fill );
            g.setColour( c.withAlpha( 0.85f ) );
            g.strokePath( curve, juce::PathStrokeType( 1.5f ) );
            g.setColour( c.withAlpha( 0.35f ) );
            g.drawHorizontalLine( juce::roundToInt( yOf( 1.0f ) ), spec.getX(), spec.getRight() );
            const juce::Point<float> at( xOf( cutoff, spec ),
                                         yOf( additiv::filterResponse( model, cutoff, cutoff, q, vowel ) ) );
            paintHandle( g, HandleE::FILTER, at, c );
            if ( m_held == HandleE::FILTER || m_hover == HandleE::FILTER )
                note( at,
                      "Cutoff " +
                          juce::String( cutoff >= 1000.0f ? juce::String( cutoff / 1000.0f, 2 ) + " kHz"
                                                          : juce::String( cutoff, 0 ) + " Hz" ) +
                          "   Resonance " + juce::String( res, 2 ) + "  (drag: across, up)",
                      c );
        }
        else
        {
            g.setColour( c.withAlpha( 0.6f ) );
            g.drawText( "Filter off: choose a model to draw and take its curve",
                        spec.withTrimmedTop( spec.getHeight() - 16.0f ), juce::Justification::centredRight );
        }
        const float depth = value( ParamIdE::COMB_DEPTH );
        if ( depth > 0.0f )
        {   // the comb's notches where they stand now: log2(f) density + phase a whole number
            const float density = value( ParamIdE::COMB_DENSITY ), phase = m_proc.engine().combPhase();
            g.setColour( c.withAlpha( 0.15f + 0.35f * depth ) );
            for ( int k = static_cast<int>( std::floor( std::log2( 20.0f ) * density ) );; ++k )
            {
                const float hz = std::exp2( ( static_cast<float>( k ) - phase ) / density );
                if ( hz > 20000.0f ) break;
                if ( hz >= 20.0f )
                    g.drawVerticalLine( juce::roundToInt( xOf( hz, spec ) ), spec.getY(), spec.getBottom() );
            }
        }
    }
    if ( m_page == "Cloud & Filter" )
    {
        const float        f0 = referenceHz();
        const juce::Colour c = skin.family( frame::FamilyE::SOURCE );
        // The air's band: its partials gather about Centre times the fundamental, Width wide.
        const float air = value( ParamIdE::AIR ), centre = f0 * value( ParamIdE::AIR_CENTER ),
                    width = value( ParamIdE::AIR_WIDTH );
        const float lo = centre * std::exp2( -width / 1200.0f ), hi = centre * std::exp2( width / 1200.0f );
        const float alpha = air > 0.0f ? 1.0f : 0.45f;
        g.setColour( c.withAlpha( 0.08f * alpha ) );
        g.fillRect( juce::Rectangle<float>::leftTopRightBottom( xOf( lo, spec ), spec.getY(), xOf( hi, spec ),
                                                                spec.getBottom() ) );
        g.setColour( c.withAlpha( 0.5f * alpha ) );
        g.drawVerticalLine( juce::roundToInt( xOf( centre, spec ) ), spec.getY(), spec.getBottom() );
        const juce::Point<float> at( xOf( centre, spec ), spec.getY() + spec.getHeight() * 0.22f );
        paintHandle( g, HandleE::AIR_CENTRE, at, c.withAlpha( alpha ) );
        paintHandle( g, HandleE::AIR_EDGE, { xOf( hi, spec ), at.y }, c.withAlpha( alpha ) );
        g.setColour( c.withAlpha( 0.7f * alpha ) );
        g.drawText( air > 0.0f ? "Air" : "Air (silent at Air 0)",
                    juce::Rectangle<float>( at.x + 9.0f, at.y + 6.0f, 140.0f, 14.0f ),
                    juce::Justification::centredLeft );
        if ( m_hover == HandleE::AIR_CENTRE || m_held == HandleE::AIR_CENTRE )
            note( at, "Centre " + juce::String( value( ParamIdE::AIR_CENTER ), 2 ) + " x the fundamental", c );
        if ( m_hover == HandleE::AIR_EDGE || m_held == HandleE::AIR_EDGE )
            note( { xOf( hi, spec ), at.y }, "Width " + juce::String( width, 0 ) + " ct", c );
        // Where the brightness lets the harmonics fade: above 1 + b^2 200 times the fundamental.
        const float b = value( ParamIdE::BRIGHTNESS );
        const float fade = f0 * ( 1.0f + b * b * 200.0f );
        if ( fade < 20000.0f )
        {
            const float x = xOf( fade, spec );
            g.setColour( c.withAlpha( 0.45f ) );
            for ( float y = spec.getY(); y < spec.getBottom(); y += 6.0f )
                g.drawVerticalLine( juce::roundToInt( x ), y, std::min( y + 3.0f, spec.getBottom() ) );
            const juce::Point<float> bt( x, spec.getY() + spec.getHeight() * 0.5f );
            paintHandle( g, HandleE::BRIGHTNESS, bt, c );
            g.setColour( c.withAlpha( 0.7f ) );
            g.drawText( "Brightness", juce::Rectangle<float>( bt.x + 9.0f, bt.y + 6.0f, 90.0f, 14.0f ),
                        juce::Justification::centredLeft );
            if ( m_hover == HandleE::BRIGHTNESS || m_held == HandleE::BRIGHTNESS )
                note( bt, "Brightness " + juce::String( b, 3 ), c );
        }
    }
    else if ( m_page == "Conductor" )
    {
        // The register as a bracket over the keys, its ends to drag.
        const juce::Colour c = additivui::colour::Violet;
        const int          low = juce::roundToInt( value( ParamIdE::BRAIN_LOW ) ),
                           high = juce::roundToInt( value( ParamIdE::BRAIN_HIGH ) );
        auto               edge = [&]( float n ) { return xOf( 440.0f * std::exp2( ( n - 69.0f ) / 12.0f ), spec ); };
        const float x0 = edge( static_cast<float>( low ) - 0.5f ), x1 = edge( static_cast<float>( high ) + 0.5f ),
                    y = keys.getY() - 5.0f;
        g.setColour( c.withAlpha( 0.05f ) );
        g.fillRect( juce::Rectangle<float>::leftTopRightBottom( x0, spec.getY(), x1, spec.getBottom() ) );
        g.setColour( c.withAlpha( 0.8f ) );
        g.drawLine( x0, y, x1, y, 1.5f );
        g.drawLine( x0, y, x0, keys.getBottom(), 1.5f );
        g.drawLine( x1, y, x1, keys.getBottom(), 1.5f );
        paintHandle( g, HandleE::REGISTER_LOW, { x0, y }, c );
        paintHandle( g, HandleE::REGISTER_HIGH, { x1, y }, c );
        g.drawText( "the conductor's register", juce::Rectangle<float>( x0 + 10.0f, y - 16.0f, 200.0f, 14.0f ),
                    juce::Justification::centredLeft );
        if ( m_hover == HandleE::REGISTER_LOW || m_held == HandleE::REGISTER_LOW )
            note( { x0, y - 12.0f }, "Lowest " + juce::MidiMessage::getMidiNoteName( low, true, true, 4 ), c );
        if ( m_hover == HandleE::REGISTER_HIGH || m_held == HandleE::REGISTER_HIGH )
            note( { x1, y - 12.0f }, "Highest " + juce::MidiMessage::getMidiNoteName( high, true, true, 4 ), c );
    }
    else if ( m_page == "Motion" )
    {
        // The ripple's string along the spectrum, as the partials ride it.
        const float depth = value( ParamIdE::RIPPLE_DEPTH );
        if ( depth > 0.0f && m_ripple.size() == static_cast<size_t>( additiv::RippleCells ) )
        {
            const juce::Colour c = skin.family( frame::FamilyE::SPACE );
            const float        mid = spec.getCentreY(), amp = spec.getHeight() * 0.3f;
            juce::Path         p;
            for ( int i = 0; i < additiv::RippleCells; ++i )
            {
                const float x = spec.getX() + static_cast<float>( i ) / static_cast<float>( additiv::RippleCells - 1 ) *
                                                  spec.getWidth();
                const float y = mid - amp * depth * std::tanh( m_ripple[static_cast<size_t>( i )] );
                if ( i == 0 )
                    p.startNewSubPath( x, y );
                else
                    p.lineTo( x, y );
            }
            g.setColour( c.withAlpha( 0.75f ) );
            g.strokePath( p, juce::PathStrokeType( 1.4f ) );
            g.drawText( "Ripple", juce::Rectangle<float>( spec.getX() + 4.0f, mid - 16.0f, 80.0f, 14.0f ),
                        juce::Justification::centredLeft );
        }
    }
    else if ( m_page == "Hall" )
    {
        const juce::Colour c = skin.family( frame::FamilyE::MOTION );
        // The hall's memory over the axis (its decay's change at Grow 1, the lower half of the picture).
        if ( m_hall.size() == static_cast<size_t>( additiv::HallBins ) && m_hallAge > 0.0 )
        {
            const float base = spec.getBottom() - spec.getHeight() * 0.125f, span = spec.getHeight() * 0.25f;
            juce::Path  p;
            for ( int b = 0; b < additiv::HallBins; b += 2 )
            {
                const float x = spec.getX() + ( static_cast<float>( b ) + 0.5f ) /
                                                  static_cast<float>( additiv::HallBins ) * spec.getWidth();
                const float oct = juce::jlimit(
                    -0.5f, 1.5f, std::log2( 1.0f + std::max( m_hall[static_cast<size_t>( b )], 0.0f ) ) - 1.0f );
                const float y = base - oct * span;
                if ( b == 0 )
                    p.startNewSubPath( x, y );
                else
                    p.lineTo( x, y );
            }
            g.setColour( c.withAlpha( 0.7f ) );
            g.strokePath( p, juce::PathStrokeType( 1.4f ) );
            g.drawText( "the hall's memory",
                        juce::Rectangle<float>( spec.getX() + 4.0f, base - 1.5f * span - 14.0f, 160.0f, 14.0f ),
                        juce::Justification::centredLeft );
        }
        // Its lowest mode, to drag.
        const float lowest = value( ParamIdE::MODAL_LOW ), x = xOf( lowest, spec );
        g.setColour( c.withAlpha( 0.15f ) );
        g.fillRect( juce::Rectangle<float>::leftTopRightBottom( spec.getX(), spec.getY(), x, spec.getBottom() ) );
        g.setColour( c.withAlpha( 0.7f ) );
        g.drawVerticalLine( juce::roundToInt( x ), spec.getY(), spec.getBottom() );
        const juce::Point<float> at( x, spec.getY() + spec.getHeight() * 0.3f );
        paintHandle( g, HandleE::HALL_LOWEST, at, c );
        g.drawText( "Lowest", juce::Rectangle<float>( at.x + 9.0f, at.y + 6.0f, 80.0f, 14.0f ),
                    juce::Justification::centredLeft );
        if ( m_hover == HandleE::HALL_LOWEST || m_held == HandleE::HALL_LOWEST )
            note( at, "the hall's lowest mode " + juce::String( lowest, 0 ) + " Hz", c );
    }
}

void CloudView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    const auto         all = getLocalBounds().toFloat();
    g.setColour( m_canvas ? skin.m_bg : skin.m_group.withAlpha( 0.9f ) );
    g.fillRoundedRectangle( all, skin.m_radius );
    if ( !m_canvas )
    {
        g.setColour( skin.m_edge );
        g.drawRoundedRectangle( all.reduced( 0.5f ), skin.m_radius, 1.0f );
    }

    auto inner = all.reduced( m_canvas ? 20.0f : 10.0f );
    if ( m_canvas ) inner.removeFromBottom( 62.0f );   // the macros
    const auto landArea = m_canvas ? inner : inner.removeFromRight( inner.getWidth() * 0.28f ).withTrimmedLeft( 10.0f );
    m_landArea = m_canvas ? juce::Rectangle<float>() : landArea;
    const float keyH = m_canvas ? 14.0f : 10.0f;   // the keys' strip under the spectrum
    const auto  spec = inner.withTrimmedBottom( 14.0f + keyH + 3.0f );
    m_spec = spec;
    const juce::Rectangle<float> keys( spec.getX(), spec.getBottom() + 2.0f, spec.getWidth(), keyH );

    const bool reading = additiv::motionUsesLandscape(
        static_cast<additiv::MotionKindE>( juce::roundToInt( m_proc.engine().soundValue( ParamIdE::MOTION ) ) ) );
    const bool living = !m_lifeCells.empty() && m_life.isValid();
    if ( m_canvas )
    {   // the landscape faintly behind the partials, the whole picture
        if ( living )
        {
            g.setOpacity( 0.45f );
            g.drawImage( m_life, spec, juce::RectanglePlacement::stretchToFit );
        }
        else if ( m_land.isValid() )
        {
            g.setOpacity( reading ? 0.3f : 0.12f );
            g.drawImage( m_land, spec, juce::RectanglePlacement::stretchToFit );
        }
        g.setOpacity( 1.0f );
    }

    // The axis: 100 Hz, 1 kHz, 10 kHz.
    g.setFont( juce::FontOptions( 11.0f ) );
    for ( float f : { 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f } )
    {
        const float x = xOf( f, spec );
        g.setColour( skin.m_faint.withAlpha( m_canvas ? 0.18f : 0.5f ) );
        g.drawVerticalLine( juce::roundToInt( x ), spec.getY(), spec.getBottom() );
        g.setColour( m_canvas ? skin.m_dim.withAlpha( 0.5f ) : skin.m_dim );
        g.drawText( f >= 1000.0f ? juce::String( f / 1000.0f, 0 ) + "k" : juce::String( f, 0 ),
                    juce::Rectangle<float>( x - 20.0f, keys.getBottom(), 40.0f, 14.0f ), juce::Justification::centred );
    }
    paintKeys( g, keys, spec );

    // The partials themselves: every eighth one a point with its trail (plotDots, from the GPU's snapshot).
    m_dotsArea = spec.toNearestInt();
    if ( m_dots.isValid() ) g.drawImage( m_dots, spec, juce::RectanglePlacement::stretchToFit );
    paintOverlay( g, spec, keys );
    if ( m_canvas )
    {
        g.setColour( skin.m_dim.withAlpha( 0.6f ) );
        g.drawText( "click a key in the spectrum to play it (higher: louder)    double-click or Esc: back",
                    juce::Rectangle<float>( all.getRight() - 520.0f, all.getBottom() - 34.0f, 500.0f, 16.0f ),
                    juce::Justification::centredRight );
        return;
    }
    float firstTime = -1.0f;
    for ( const additiv::VoiceView &v : m_views )
        if ( v.m_hz > 0.0f && v.m_gain > 0.0f )
        {
            firstTime = v.m_time;
            break;
        }

    // The landscape, and where the voices are in its loop (when the Aurora motion reads it).
    if ( living )
    {   // the living landscape takes the panel while it lives
        g.drawImage( m_life, landArea, juce::RectanglePlacement::stretchToFit );
    }
    else if ( m_land.isValid() )
    {
        g.setOpacity( reading ? 1.0f : 0.35f );
        g.drawImage( m_land, landArea, juce::RectanglePlacement::stretchToFit );
        g.setOpacity( 1.0f );
    }
    g.setColour( skin.m_edge );
    g.drawRect( landArea, 1.0f );
    g.setColour( skin.m_dim );
    const int lifeMode = juce::roundToInt( m_proc.engine().soundValue( ParamIdE::LIFE_MODE ) );
    if ( living )
        g.drawText( juce::String( "Living landscape: " ) +
                        additiv::LifeModeNames[juce::jlimit( 0, additiv::NumLifeModes - 1, lifeMode )],
                    landArea.withHeight( 14.0f ).translated( 0.0f, -1.0f ), juce::Justification::centredLeft );
    else
    {
        const bool         own = m_landChoice == additiv::UserLandscape;
        const juce::String name =
            own ? juce::String( "Your own: " ) + m_proc.userLandscapeFile().getFileNameWithoutExtension()
                : juce::String( additiv::LandscapeChoiceNames[std::max( m_landChoice, 0 )] );
        g.drawText( name + ( reading ? "" : "  (Motion: Aurora reads it)" ),
                    landArea.withHeight( 14.0f ).translated( 0.0f, -1.0f ), juce::Justification::centredLeft );
    }
    if ( isMouseOver( true ) )
    {   // what a click does
        g.setColour( skin.m_dim );
        g.drawText( "click: your own picture or recording", landArea.withTrimmedTop( landArea.getHeight() - 14.0f ),
                    juce::Justification::centredRight );
    }
    if ( reading && !living && firstTime >= 0.0f )
    {
        const float rate = m_proc.engine().soundValue( ParamIdE::MOTION_RATE );
        float       u = firstTime * rate / std::max( m_landLoop, 0.1f );
        u -= std::floor( u );
        g.setColour( skin.m_onset );
        g.drawVerticalLine( juce::roundToInt( landArea.getX() + u * landArea.getWidth() ), landArea.getY(),
                            landArea.getBottom() );
    }
}
