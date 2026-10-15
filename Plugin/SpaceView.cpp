/**
 * @file SpaceView.cpp
 * @brief The Space page (SpaceView.h).
 * @author Rene Weller
 */
#include "SpaceView.h"

#include "Theme.h"

#include <cmath>

using additiv::ParamIdE;

namespace
{

constexpr float Head = 0.1f;   ///< the head room round you, of the radius: the nearest plane stands outside it

/**
 * @brief A voice's place across: its pitch, C1 at the left to C8 at the right.
 * @param[in] hz  its frequency
 * @return 0 .. 1
 */
float across( float hz )
{
    return juce::jlimit( 0.0f, 1.0f, ( 12.0f * std::log2( std::max( hz, 1.0f ) / 32.70f ) ) / 84.0f );
}

}   // namespace

SpaceView::SpaceView( AdditivProcessor &proc ) : m_proc( proc )
{
    m_seen.fill( 0 );
    startTimerHz( 24 );
}

juce::Point<float> SpaceView::ear() const
{
    return { getWidth() * 0.5f, getHeight() - 30.0f };
}

float SpaceView::radius() const
{
    return std::max( 20.0f, std::min( getWidth() * 0.5f - 30.0f, getHeight() - 70.0f ) );
}

float SpaceView::planeRadius( float depth ) const
{
    return radius() * ( Head + ( 1.0f - Head ) * juce::jlimit( 0.0f, 1.0f, depth ) );
}

float SpaceView::depthAt( float r ) const
{
    return juce::jlimit( 0.0f, 1.0f, ( r / radius() - Head ) / ( 1.0f - Head ) );
}

void SpaceView::timerCallback()
{
    if ( !isShowing() ) return;
    // The places every eighth of a second into the trails (three seconds of Breath's wandering).
    for ( int i = 0; i < additiv::MaxViewVoices; ++i )
    {
        const additiv::VoiceView v = m_proc.engine().voiceView( i );
        additiv::VoiceView      &was = m_views[static_cast<size_t>( i )];
        if ( v.m_hz <= 0.0f || v.m_gain <= 0.0f || v.m_note != was.m_note )
            m_seen[static_cast<size_t>( i )] = 0;   // a new note: a new trail
        was = v;
    }
    if ( ++m_tick % 3 == 0 )
    {
        m_head = ( m_head + 1 ) % Trail;
        for ( int i = 0; i < additiv::MaxViewVoices; ++i )
        {
            const additiv::VoiceView &v = m_views[static_cast<size_t>( i )];
            if ( v.m_hz <= 0.0f || v.m_gain <= 0.0f ) continue;
            m_trails[static_cast<size_t>( i )][static_cast<size_t>( m_head )] = { v.m_distance, across( v.m_hz ) };
            m_seen[static_cast<size_t>( i )] = std::min( m_seen[static_cast<size_t>( i )] + 1, Trail );
        }
    }
    repaint();
}

void SpaceView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( getLocalBounds().toFloat(), skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( getLocalBounds().toFloat().reduced( 0.5f ), skin.m_radius, 1.0f );
    const juce::Colour space = skin.family( frame::FamilyE::SPACE );
    g.setColour( space );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.drawText(
        "The space from above: you at the bottom, near close to you, far at the rim; low notes left, high right", 16, 4,
        getWidth() - 24, 20, juce::Justification::centredLeft );

    const auto  e = ear();
    const float R = radius();
    // The far plane's glow at the rim, as loud as the far reverb.
    const float far = m_proc.value( static_cast<int>( ParamIdE::FAR_LEVEL ) );
    {
        juce::Path rim;
        rim.addCentredArc( e.x, e.y, R, R, 0.0f, -juce::MathConstants<float>::halfPi,
                           juce::MathConstants<float>::halfPi, true );
        g.setColour( space.withAlpha( 0.08f + 0.25f * far ) );
        g.strokePath( rim, juce::PathStrokeType( 6.0f ) );
    }
    // The planes: rings at a quarter, a half, three quarters, the rim.
    g.setFont( juce::FontOptions( 11.0f ) );
    for ( float d : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f } )
    {
        const float r = planeRadius( d );
        juce::Path  ring;
        ring.addCentredArc( e.x, e.y, r, r, 0.0f, -juce::MathConstants<float>::halfPi,
                            juce::MathConstants<float>::halfPi, true );
        g.setColour( skin.m_faint.withAlpha( 0.6f ) );
        g.strokePath( ring, juce::PathStrokeType( 1.0f ) );
        g.setColour( skin.m_dim );
        const juce::String name = d == 0.0f ? "near" : d == 1.0f ? "far" : juce::String( d, 2 );
        g.drawText( name, juce::Rectangle<float>( e.x + r - 30.0f, e.y + 2.0f, 60.0f, 14.0f ),
                    juce::Justification::centred );
    }
    // You.
    g.setColour( skin.m_ink );
    g.fillEllipse( e.x - 5.0f, e.y - 5.0f, 10.0f, 10.0f );
    g.setColour( skin.m_dim );
    g.drawText( "you", juce::Rectangle<float>( e.x - 30.0f, e.y + 8.0f, 60.0f, 14.0f ), juce::Justification::centred );

    // The two depths: where the conductor puts its notes, where the played notes stand -- rings to drag.
    const float        depth = m_proc.value( static_cast<int>( ParamIdE::DEPTH ) ),
                       keys = m_proc.value( static_cast<int>( ParamIdE::KEYS_DEPTH ) );
    const juce::Colour conductor = additivui::colour::Violet, played = skin.m_accent;
    for ( int k = 0; k < 2; ++k )
    {
        const float r = planeRadius( k == 0 ? depth : keys );
        juce::Path  ring;
        ring.addCentredArc( e.x, e.y, r, r, 0.0f, -juce::MathConstants<float>::halfPi * 0.95f,
                            juce::MathConstants<float>::halfPi * 0.95f, true );
        const juce::Colour c = k == 0 ? conductor : played;
        g.setColour( c.withAlpha( m_held == k ? 0.9f : 0.45f ) );
        const float dash[2] = { 5.0f, 5.0f };
        juce::Path  dashed;
        juce::PathStrokeType( 1.4f ).createDashedStroke( dashed, ring, dash, 2 );
        g.fillPath( dashed );
        const float hx = e.x + ( k == 0 ? -0.72f : 0.72f ) * r,
                    hy = e.y - 0.694f * r;   // its handle, on the ring left (Depth) or right (Keys Depth)
        g.drawEllipse( hx - 5.0f, hy - 5.0f, 10.0f, 10.0f, 1.6f );
        g.drawText( k == 0 ? "Depth" : "Keys Depth",
                    juce::Rectangle<float>( hx + ( k == 0 ? -84.0f : 8.0f ), hy - 7.0f, 76.0f, 14.0f ),
                    k == 0 ? juce::Justification::centredRight : juce::Justification::centredLeft );
    }

    // The voices: a light at its plane and its pitch, its trail fading behind it.
    const juce::Colour bySource[4] = { played, conductor, space, skin.m_dim.brighter( 0.4f ) };
    auto place = [&](juce::Point<float> dp) {   // (depth, across) to pixels
        const float a = (dp.y - 0.5f) * juce::MathConstants<float>::pi * 0.9f;
        const float r = planeRadius(dp.x);
        return juce::Point<float>(e.x + std::sin(a) * r, e.y - std::cos(a) * r);
    };
    for ( int i = 0; i < additiv::MaxViewVoices; ++i )
    {
        const additiv::VoiceView &v = m_views[static_cast<size_t>( i )];
        if ( v.m_hz <= 0.0f || v.m_gain <= 0.0f ) continue;
        const juce::Colour c = bySource[juce::jlimit( 0, 3, v.m_source )];
        const int          n = m_seen[static_cast<size_t>( i )];
        for ( int k = 1; k < n; ++k )
        {
            const auto              &tr = m_trails[static_cast<size_t>( i )];
            const juce::Point<float> a = place( tr[static_cast<size_t>( ( m_head - k + Trail ) % Trail )] );
            const juce::Point<float> b = place( tr[static_cast<size_t>( ( m_head - k + 1 + Trail ) % Trail )] );
            g.setColour( c.withAlpha( 0.35f * ( 1.0f - static_cast<float>( k ) / Trail ) ) );
            g.drawLine( a.x, a.y, b.x, b.y, 1.2f );
        }
        const juce::Point<float> at = place( { v.m_distance, across( v.m_hz ) } );
        const float              s =
            3.0f + 7.0f * std::sqrt( juce::jlimit( 0.0f, 1.0f, v.m_gain ) ) * ( 1.0f - 0.5f * v.m_distance );
        g.setColour( c.withAlpha( 0.25f ) );
        g.fillEllipse( at.x - 2.0f * s, at.y - 2.0f * s, 4.0f * s, 4.0f * s );
        g.setColour( c );
        g.fillEllipse( at.x - s, at.y - s, 2.0f * s, 2.0f * s );
        if ( v.m_note >= 0 )
        {
            g.setColour( c.withAlpha( 0.8f ) );
            g.drawText( juce::MidiMessage::getMidiNoteName( v.m_note, true, true, 4 ),
                        juce::Rectangle<float>( at.x - 20.0f, at.y - s - 15.0f, 40.0f, 13.0f ),
                        juce::Justification::centred );
        }
    }
}

void SpaceView::mouseDown( const juce::MouseEvent &e )
{
    // The ring nearer the click, when the click is near one.
    const float r = e.position.getDistanceFrom( ear() );
    const float rd = std::abs( r - planeRadius( m_proc.value( static_cast<int>( ParamIdE::DEPTH ) ) ) );
    const float rk = std::abs( r - planeRadius( m_proc.value( static_cast<int>( ParamIdE::KEYS_DEPTH ) ) ) );
    m_held = std::min( rd, rk ) > 10.0f ? -1 : ( rd <= rk ? 0 : 1 );
    if ( m_held < 0 ) return;
    if ( auto *p = m_proc.parameter( static_cast<int>( m_held == 0 ? ParamIdE::DEPTH : ParamIdE::KEYS_DEPTH ) ) )
        p->beginChangeGesture();
    mouseDrag( e );
}

void SpaceView::mouseDrag( const juce::MouseEvent &e )
{
    if ( m_held < 0 ) return;
    m_proc.setValue( static_cast<int>( m_held == 0 ? ParamIdE::DEPTH : ParamIdE::KEYS_DEPTH ),
                     depthAt( e.position.getDistanceFrom( ear() ) ) );
    repaint();
}

void SpaceView::mouseUp( const juce::MouseEvent & )
{
    if ( m_held >= 0 )
        if ( auto *p = m_proc.parameter( static_cast<int>( m_held == 0 ? ParamIdE::DEPTH : ParamIdE::KEYS_DEPTH ) ) )
            p->endChangeGesture();
    m_held = -1;
    repaint();
}

juce::String SpaceView::describe() const
{
    juce::String s( "The space: " );
    int          n = 0;
    for ( const additiv::VoiceView &v : m_views )
    {
        if ( v.m_hz <= 0.0f || v.m_gain <= 0.0f ) continue;
        s << ( n++ > 0 ? ", " : "" )
          << ( v.m_note >= 0 ? juce::MidiMessage::getMidiNoteName( v.m_note, true, true, 4 )
                             : juce::String( "a ghost" ) )
          << " at " << juce::String( v.m_distance, 2 );
    }
    if ( n == 0 ) s << "nothing sounds";
    return s + "\n";
}

// ---- the page

SpacePage::SpacePage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live )
    : m_params(
          proc, actions,
          { { "Planes",
              frame::FamilyE::SPACE,
              { ParamIdE::DEPTH, ParamIdE::KEYS_DEPTH, ParamIdE::BREATH, ParamIdE::BREATH_RATE, ParamIdE::DOPPLER } },
            { "Near", frame::FamilyE::SPACE, { ParamIdE::PRESENCE, ParamIdE::PAD_LOW_CUT } },
            { "Far Reverb",
              frame::FamilyE::SPACE,
              { ParamIdE::FAR_LEVEL, ParamIdE::FAR_SIZE, ParamIdE::FAR_DECAY, ParamIdE::FAR_DAMP, ParamIdE::FAR_DIFFUSE,
                ParamIdE::FAR_PRE_DELAY, ParamIdE::FAR_UNMASK, ParamIdE::FAR_MODE, ParamIdE::FAR_FREEZE },
              { ParamIdE::FAR_LEVEL } },
            { "Master", frame::FamilyE::SPACE, { ParamIdE::MASTER_GAIN, ParamIdE::BASS_MONO, ParamIdE::STEREO_WIDTH } },
            { "Binaural",
              frame::FamilyE::SPACE,
              { ParamIdE::BINAURAL, ParamIdE::BINAURAL_SPAN, ParamIdE::BINAURAL_HEIGHT, ParamIdE::BINAURAL_ORBIT },
              { ParamIdE::BINAURAL } } },
          live ),
      m_view( proc )
{
    addAndMakeVisible( m_params );
    addAndMakeVisible( m_view );
}

/** @brief The space's box: its least height. */
constexpr int SpaceViewHeight = 200;

int SpacePage::heightFor( int w ) const
{
    return m_params.heightFor( w ) + SpaceViewHeight + 10;
}

void SpacePage::resized()
{
    const int top = m_params.heightFor( getWidth() );
    m_params.setBounds( 0, 0, getWidth(), top );
    m_view.setBounds( 10, top, getWidth() - 20, std::max( SpaceViewHeight, getHeight() - 10 - top ) );
}

juce::String SpacePage::describe() const
{
    return m_params.describe() + m_view.describe() +
           "\nBelow the knobs the space is seen from above: you at the bottom, the near plane close to you and the far plane at the rim. "
           "Every voice is a light at its plane -- Breath lets it wander, and its trail shows where it was -- across from its pitch, as large "
           "as it is loud. Drag the dashed rings to set where the conductor puts its notes (Depth) and where yours stand (Keys Depth).\n";
}
