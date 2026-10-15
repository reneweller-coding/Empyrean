/**
 * @file MapView.cpp
 * @brief The Map page (MapView.h).
 * @author Rene Weller
 */
#include "MapView.h"

#include "Theme.h"

#include "additiv/Journey.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>

using additiv::ParamIdE;

namespace
{

/**
 * @brief A section's colour: the sounds, the living systems, the sleep concerts, the visions, the swarms and currents,
 *        the halls and orbits.
 * @param[in] group  the preset's group
 * @return the colour
 */
juce::Colour sectionColour( int group )
{
    const frame::Skin &skin = additivui::skin();
    switch ( group / 16 )
    {
    case 0: return skin.family( frame::FamilyE::SOURCE );
    case 1: return additivui::colour::Violet;
    case 2: return skin.family( frame::FamilyE::SPACE );
    case 3: return skin.family( frame::FamilyE::ENVELOPE );
    case 4: return skin.family( frame::FamilyE::FILTER );
    default: return additivui::colour::Root;
    }
}

}   // namespace

// ---- the map

namespace
{

constexpr float  MostZoom = 32.0f;   ///< the closest view: a thirty-second of the map across
constexpr size_t MostSeen = 400;   ///< the presets' names stand at their points when no more than this many are in view
constexpr size_t MostNames = 60;   ///< and no more names than this, the middle's first (the rest by the mouse)

}   // namespace

MapView::MapView( AdditivProcessor &proc ) : m_proc( proc )
{
    m_whole.setTooltip( "Shows the whole map again" );
    m_whole.onClick = [this] { setView( { 0.0f, 0.0f, 1.0f, 1.0f } ); };
    addChildComponent( m_whole );
    startTimerHz( 15 );
}

void MapView::timerCallback()
{
    // The cursor (or the preset playing) walking out of the view by itself -- a route, a preset chosen: the view glides after it.
    const auto f = focus();
    const bool moved = f != m_lastFocus;
    m_lastFocus = f;
    if ( zoomed() && moved && m_focusInside && !inView( f.x, f.y ) && !m_panning && m_dragPoint < 0 )
    {
        const float w = m_view.getWidth();
        m_target = clampView( { f.x - 0.5f * w, f.y - 0.5f * w, w, w } );
        m_gliding = true;
    }
    if ( m_gliding )
    {   // a third of the way each tick
        const float dx = m_target.getX() - m_view.getX(), dy = m_target.getY() - m_view.getY();
        const bool  there = std::abs( dx ) + std::abs( dy ) < 0.004f * m_view.getWidth();
        setView( there ? m_target : m_view.translated( 0.35f * dx, 0.35f * dy ), false );
        m_gliding = !there;
    }
    m_focusInside = inView( f.x, f.y );
    // The names written once the view has come to rest.
    if ( !m_named && !m_panning && !m_gliding && juce::Time::getMillisecondCounter() - m_viewChanged > 180u )
    {
        drawPoints( true );
        m_named = true;
    }
    repaint();
}

juce::Rectangle<float> MapView::plane() const
{
    const auto  b = getLocalBounds().toFloat().reduced( 26.0f, 22.0f );
    const float s = std::min( b.getWidth(), b.getHeight() );
    return { b.getX(), b.getY(), s, s };
}

juce::Rectangle<float> MapView::mini() const
{
    const auto  p = plane();
    const float s = juce::jlimit( 60.0f, 130.0f, p.getWidth() * 0.24f );
    return { p.getRight() - s - 8.0f, p.getY() + 8.0f, s, s };
}

bool MapView::zoomed() const
{
    return m_view.getWidth() < 0.999f;
}

juce::Point<float> MapView::focus() const
{
    // With the map on (or no preset chosen), the cursor; else the preset playing.
    const int cp = m_proc.currentPreset();
    if ( m_proc.value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f || cp < 0 )
        return { m_proc.value( static_cast<int>( ParamIdE::MAP_X ) ),
                 m_proc.value( static_cast<int>( ParamIdE::MAP_Y ) ) };
    float x = 0.0f, y = 0.0f;
    additiv::presetMapPosition( cp, x, y );
    return { x, y };
}

bool MapView::inView( float x, float y ) const
{
    return x >= m_view.getX() && x <= m_view.getRight() && y >= m_view.getY() && y <= m_view.getBottom();
}

juce::Rectangle<float> MapView::clampView( juce::Rectangle<float> v )
{
    const float w = juce::jlimit( 1.0f / MostZoom, 1.0f, v.getWidth() );
    return { juce::jlimit( 0.0f, 1.0f - w, v.getX() ), juce::jlimit( 0.0f, 1.0f - w, v.getY() ), w, w };
}

void MapView::setView( juce::Rectangle<float> v, bool stopGlide )
{
    if ( stopGlide ) m_gliding = false;
    v = clampView( v );
    if ( v == m_view ) return;
    m_view = v;
    m_viewChanged = juce::Time::getMillisecondCounter();
    m_named = false;
    m_whole.setVisible( zoomed() );
    drawPoints( false );   // the names follow when the view rests (timerCallback)
    repaint();
}

void MapView::zoomAt( juce::Point<float> s, float factor )
{
    const auto  p = plane();
    const float w = juce::jlimit( 1.0f / MostZoom, 1.0f, m_view.getWidth() / factor );
    const float u = ( s.x - p.getX() ) / p.getWidth(), v = ( p.getBottom() - s.y ) / p.getHeight();
    const float mx = m_view.getX() + u * m_view.getWidth(), my = m_view.getY() + v * m_view.getHeight();
    setView( { mx - u * w, my - v * w, w, w } );   // the place under the mouse stays under it
}

void MapView::centreOn( float x, float y )
{
    const float w = m_view.getWidth();
    setView( { x - 0.5f * w, y - 0.5f * w, w, w } );
}

juce::Point<float> MapView::toScreen( float x, float y ) const
{
    const auto  p = plane();
    const float u = ( x - m_view.getX() ) / m_view.getWidth(), v = ( y - m_view.getY() ) / m_view.getHeight();
    return { p.getX() + u * p.getWidth(), p.getBottom() - v * p.getHeight() };
}

juce::Point<float> MapView::toMap( juce::Point<float> s ) const
{
    const auto  p = plane();
    const float u = juce::jlimit( 0.0f, 1.0f, ( s.x - p.getX() ) / p.getWidth() ),
                v = juce::jlimit( 0.0f, 1.0f, ( p.getBottom() - s.y ) / p.getHeight() );
    return { m_view.getX() + u * m_view.getWidth(), m_view.getY() + v * m_view.getHeight() };
}

int MapView::presetAt( juce::Point<float> s, float reach ) const
{
    const auto m = toMap( s );
    const int  i = m_proc.nearestPreset( m.x, m.y );
    float      x = 0.0f, y = 0.0f;
    additiv::presetMapPosition( i, x, y );
    return toScreen( x, y ).getDistanceFrom( s ) <= reach ? i : -1;
}

void MapView::resized()
{
    const auto m = mini();
    m_whole.setBounds(
        juce::Rectangle<float>( m.getX() - 3.0f, m.getBottom() + 8.0f, m.getWidth() + 6.0f, 22.0f ).toNearestInt() );
    drawOverview();
    drawPoints( true );
    m_named = true;
}

void MapView::drawPoints( bool names )
{
    const auto p = plane();
    if ( p.getWidth() < 10.0f || getWidth() <= 0 || getHeight() <= 0 )
    {
        m_points = {};
        return;
    }
    if ( !m_points.isValid() || m_points.getWidth() != getWidth() || m_points.getHeight() != getHeight() )
        m_points = juce::Image( juce::Image::ARGB, getWidth(), getHeight(), true );
    else
        m_points.clear( m_points.getBounds() );
    juce::Graphics g( m_points );
    g.reduceClipRegion( p.getSmallestIntegerContainer() );
    const float zoom = 1.0f / m_view.getWidth();
    const float r = std::min( 4.0f, 1.6f * std::pow( zoom, 0.35f ) );   // nearer, a little larger
    const auto  seen = m_view.expanded( 8.0f * m_view.getWidth() / p.getWidth() );
    const auto  mid = p.getCentre();
    std::vector<std::pair<float, int>> shown;   // (distance from the middle, preset), for the names
    for ( int i = 0; i < additiv::numPresets(); ++i )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( i, x, y );
        if ( !seen.contains( x, y ) ) continue;
        const auto s = toScreen( x, y );
        g.setColour( sectionColour( additiv::preset( i ).m_group ).withAlpha( 0.75f ) );
        g.fillEllipse( s.x - r, s.y - r, 2.0f * r, 2.0f * r );
        if ( names && p.contains( s ) ) shown.emplace_back( s.getDistanceSquaredFrom( mid ), i );
    }
    if ( shown.empty() || shown.size() > MostSeen ) return;   // too many for names: they would hide the points
    // The names where there is room for them, the middle's first.
    std::sort( shown.begin(), shown.end() );
    const juce::Font font( juce::FontOptions( 10.5f ) );
    g.setFont( font );
    g.setColour( additivui::skin().m_dim );
    std::vector<juce::Rectangle<float>> taken;
    for ( const auto &[distance, i] : shown )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( i, x, y );
        const auto             s = toScreen( x, y );
        const juce::String     name( additiv::preset( i ).m_name );
        const float            w = static_cast<float>( juce::GlyphArrangement::getStringWidthInt( font, name ) ) + 4.0f;
        juce::Rectangle<float> box( s.x + r + 3.0f, s.y - 7.0f, w, 14.0f );
        if ( box.getRight() > p.getRight() ) box.setX( s.x - r - 3.0f - w );
        if ( std::any_of( taken.begin(), taken.end(), [&box]( const juce::Rectangle<float> &t )
                          { return t.intersects( box.expanded( 3.0f, 2.0f ) ); } ) )
            continue;
        taken.push_back( box );
        g.drawText( name, box, juce::Justification::centredLeft );
        if ( taken.size() >= MostNames ) break;
    }
}

void MapView::drawOverview()
{
    const auto m = mini();
    m_overview = {};
    if ( m.getWidth() < 20.0f ) return;
    const int n = juce::roundToInt( m.getWidth() );
    m_overview = juce::Image( juce::Image::ARGB, n, n, true );
    juce::Graphics g( m_overview );
    for ( int i = 0; i < additiv::numPresets(); ++i )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( i, x, y );
        g.setColour( sectionColour( additiv::preset( i ).m_group ).withAlpha( 0.55f ) );
        g.fillRect( x * static_cast<float>( n - 1 ), ( 1.0f - y ) * static_cast<float>( n - 1 ), 1.0f, 1.0f );
    }
}

void MapView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( getLocalBounds().toFloat(), skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( getLocalBounds().toFloat().reduced( 0.5f ), skin.m_radius, 1.0f );
    const auto p = plane();
    g.setColour( skin.m_bg.withAlpha( 0.6f ) );
    g.fillRect( p );
    if ( m_points.isValid() ) g.drawImageAt( m_points, 0, 0 );
    g.setColour( skin.m_dim );
    g.setFont( juce::FontOptions( 11.0f ) );
    g.drawText( "darker", juce::Rectangle<float>( p.getX(), p.getBottom() + 3.0f, 80.0f, 14.0f ),
                juce::Justification::centredLeft );
    g.drawText( "brighter", juce::Rectangle<float>( p.getRight() - 80.0f, p.getBottom() + 3.0f, 80.0f, 14.0f ),
                juce::Justification::centredRight );
    // How near the view is, and how to move it.
    g.drawText( zoomed() ? juce::String( 1.0f / m_view.getWidth(), 1 ) + "x  -  right-drag moves"
                         : juce::String( "wheel zooms, right-drag moves" ),
                juce::Rectangle<float>( p.getX() + 70.0f, p.getBottom() + 3.0f, p.getWidth() - 140.0f, 14.0f ),
                juce::Justification::centred );
    g.saveState();
    g.addTransform(
        juce::AffineTransform::rotation( -juce::MathConstants<float>::halfPi, p.getX() - 8.0f, p.getBottom() ) );
    g.drawText( "still", juce::Rectangle<float>( p.getX() - 8.0f, p.getBottom() - 14.0f, 60.0f, 14.0f ),
                juce::Justification::centredLeft );
    g.drawText( "moving",
                juce::Rectangle<float>( p.getX() - 8.0f + p.getHeight() - 60.0f, p.getBottom() - 14.0f, 60.0f, 14.0f ),
                juce::Justification::centredRight );
    g.restoreState();

    const bool  on = m_proc.value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f;
    const float cx = m_proc.value( static_cast<int>( ParamIdE::MAP_X ) ),
                cy = m_proc.value( static_cast<int>( ParamIdE::MAP_Y ) );
    const float cr = m_proc.value( static_cast<int>( ParamIdE::MAP_RADIUS ) );
    const auto  c = toScreen( cx, cy );

    g.saveState();
    g.reduceClipRegion( p.getSmallestIntegerContainer() );   // what lies beyond the view is not drawn

    // The presets the last search by example found (Focus page), ringed, numbered nearest first.
    {
        const auto &found = m_proc.found();
        g.setFont( juce::FontOptions( 10.0f, juce::Font::bold ) );
        for ( size_t i = 0; i < found.size(); ++i )
        {
            float x = 0.0f, y = 0.0f;
            additiv::presetMapPosition( found[i], x, y );
            const auto s = toScreen( x, y );
            g.setColour( skin.m_accent.withAlpha( i == 0 ? 1.0f : 0.7f ) );
            g.drawEllipse( s.x - 6.0f, s.y - 6.0f, 12.0f, 12.0f, i == 0 ? 2.0f : 1.2f );
            g.drawText( juce::String( static_cast<int>( i ) + 1 ),
                        juce::Rectangle<float>( s.x + 6.0f, s.y - 14.0f, 14.0f, 12.0f ),
                        juce::Justification::centredLeft );
        }
    }

    // The route: its line, its numbered points, the one under way.
    const additiv::Route &r = m_proc.route();
    if ( r.count() > 0 )
    {
        juce::Path line;
        for ( int i = 0; i < r.count(); ++i )
        {
            const auto s = toScreen( r.point( i ).m_x, r.point( i ).m_y );
            if ( i == 0 )
                line.startNewSubPath( s );
            else
                line.lineTo( s );
        }
        if ( m_proc.routeLoop() && r.count() > 2 ) line.closeSubPath();
        g.setColour( skin.family( frame::FamilyE::MOTION ).withAlpha( 0.7f ) );
        g.strokePath( line, juce::PathStrokeType( 1.5f ) );
        g.setFont( juce::FontOptions( 11.0f, juce::Font::bold ) );
        for ( int i = 0; i < r.count(); ++i )
        {
            const auto s = toScreen( r.point( i ).m_x, r.point( i ).m_y );
            const bool now = r.running() && r.segment() == i;
            g.setColour( now ? skin.m_ink : skin.family( frame::FamilyE::MOTION ) );
            g.fillEllipse( s.x - 7.0f, s.y - 7.0f, 14.0f, 14.0f );
            g.setColour( skin.m_bg );
            g.drawText( juce::String( i + 1 ), juce::Rectangle<float>( s.x - 7.0f, s.y - 7.0f, 14.0f, 14.0f ),
                        juce::Justification::centred );
        }
    }

    // The cursor, its radius and the presets it blends.
    if ( on )
    {
        const additiv::MapNeighbours b = additiv::mapNeighbours( cx, cy, cr );
        for ( int k = 0; k < b.m_count; ++k )
        {
            if ( b.m_weight[k] < 0.02f ) continue;
            float x = 0.0f, y = 0.0f;
            additiv::presetMapPosition( b.m_index[k], x, y );
            const auto s = toScreen( x, y );
            g.setColour( skin.m_ink.withAlpha( 0.25f + 0.75f * b.m_weight[k] ) );
            const float rad = 3.0f + 5.0f * b.m_weight[k];
            g.drawEllipse( s.x - rad, s.y - rad, 2.0f * rad, 2.0f * rad, 1.2f );
        }
        const float rr = std::max( b.m_sigma * p.getWidth() / m_view.getWidth(),
                                   6.0f );   // the blend's width there (the radius in the local spacing)
        g.setColour( additivui::colour::Accent.withAlpha( 0.12f ) );
        g.fillEllipse( c.x - rr, c.y - rr, 2.0f * rr, 2.0f * rr );
        g.setColour( additivui::colour::Accent );
        g.drawEllipse( c.x - rr, c.y - rr, 2.0f * rr, 2.0f * rr, 1.0f );
        g.fillEllipse( c.x - 4.0f, c.y - 4.0f, 8.0f, 8.0f );
    }
    else
    {
        g.setColour( skin.m_dim.withAlpha( 0.6f ) );
        g.drawEllipse( c.x - 5.0f, c.y - 5.0f, 10.0f, 10.0f, 1.0f );
    }

    // The preset playing (without the map).
    const int cp = m_proc.currentPreset();
    if ( !on && cp >= 0 )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( cp, x, y );
        const auto s = toScreen( x, y );
        g.setColour( skin.m_ink );
        g.drawEllipse( s.x - 5.0f, s.y - 5.0f, 10.0f, 10.0f, 1.5f );
    }
    g.restoreState();

    if ( zoomed() )
    {
        // The cursor (or the preset playing) beyond the view: a ring at the edge, a line pointing to it.
        const auto f = focus();
        if ( !inView( f.x, f.y ) )
        {
            const auto               in = p.reduced( 7.0f );
            const auto               fs = toScreen( f.x, f.y );
            const juce::Point<float> e( juce::jlimit( in.getX(), in.getRight(), fs.x ),
                                        juce::jlimit( in.getY(), in.getBottom(), fs.y ) );
            const auto               towards = fs - e;
            const float              len = towards.getDistanceFromOrigin();
            g.setColour( on ? additivui::colour::Accent : skin.m_ink );
            g.drawEllipse( e.x - 5.0f, e.y - 5.0f, 10.0f, 10.0f, 1.5f );
            if ( len > 0.0f ) g.drawLine( juce::Line<float>( e, e + towards * ( 11.0f / len ) ), 1.5f );
        }
        // The overview: the whole map small, the view on it as a frame, the cursor.
        const auto m = mini();
        g.setColour( skin.m_bg.withAlpha( 0.92f ) );
        g.fillRoundedRectangle( m.expanded( 3.0f ), 4.0f );
        g.setColour( skin.m_edge );
        g.drawRoundedRectangle( m.expanded( 3.0f ), 4.0f, 1.0f );
        if ( m_overview.isValid() )
            g.drawImageAt( m_overview, juce::roundToInt( m.getX() ), juce::roundToInt( m.getY() ) );
        const juce::Rectangle<float> v( m.getX() + m_view.getX() * m.getWidth(),
                                        m.getBottom() - m_view.getBottom() * m.getHeight(),
                                        m_view.getWidth() * m.getWidth(), m_view.getHeight() * m.getHeight() );
        g.setColour( skin.m_ink );
        g.drawRect( v.withSizeKeepingCentre( std::max( v.getWidth(), 4.0f ), std::max( v.getHeight(), 4.0f ) ), 1.0f );
        g.setColour( on ? additivui::colour::Accent : skin.m_ink );
        g.fillEllipse( m.getX() + f.x * m.getWidth() - 2.0f, m.getBottom() - f.y * m.getHeight() - 2.0f, 4.0f, 4.0f );
    }

    // The preset under the mouse, with its name.
    if ( m_hover >= 0 )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( m_hover, x, y );
        const auto s = toScreen( x, y );
        g.setColour( skin.m_ink );
        g.drawEllipse( s.x - 5.0f, s.y - 5.0f, 10.0f, 10.0f, 1.2f );
        const juce::String name = juce::String( additiv::PresetGroupNames[additiv::preset( m_hover ).m_group] ) + ": " +
                                  additiv::preset( m_hover ).m_name;
        g.setFont( juce::FontOptions( 12.0f ) );
        const float w =
            static_cast<float>( juce::GlyphArrangement::getStringWidthInt( g.getCurrentFont(), name ) ) + 12.0f;
        juce::Rectangle<float> box( s.x + 8.0f, s.y - 22.0f, w, 18.0f );
        if ( box.getRight() > p.getRight() ) box.setX( s.x - 8.0f - w );
        g.setColour( skin.m_raised.withAlpha( 0.95f ) );
        g.fillRoundedRectangle( box, 4.0f );
        g.setColour( skin.m_ink );
        g.drawText( name, box, juce::Justification::centred );
    }
}

int MapView::waypointAt( juce::Point<float> p ) const
{
    const additiv::Route &r = m_proc.route();
    for ( int i = r.count() - 1; i >= 0; --i )
        if ( toScreen( r.point( i ).m_x, r.point( i ).m_y ).getDistanceFrom( p ) <= 9.0f ) return i;
    return -1;
}

void MapView::mouseDown( const juce::MouseEvent &e )
{
    m_gliding = false;
    if ( zoomed() && mini().expanded( 3.0f ).contains( e.position ) )
    {   // the overview: the view goes where it is clicked
        m_overviewDrag = true;
        mouseDrag( e );
        return;
    }
    if ( !plane().expanded( 4.0f ).contains( e.position ) ) return;
    const int w = waypointAt( e.position );
    // A waypoint's right-click: removed.
    if ( w >= 0 && e.mods.isPopupMenu() )
    {
        juce::PopupMenu m;
        m.addItem( "Remove waypoint " + juce::String( w + 1 ),
                   [this, w]
                   {
                       m_proc.removeWaypoint( w );
                       repaint();
                   } );
        m.showMenuAsync( juce::PopupMenu::Options().withTargetComponent( this ).withTargetScreenArea(
            juce::Rectangle<int>( e.getScreenX(), e.getScreenY(), 1, 1 ) ) );
        return;
    }
    // The right or the middle button moves the view.
    if ( e.mods.isPopupMenu() || e.mods.isMiddleButtonDown() )
    {
        m_panning = true;
        m_panFrom = m_view;
        return;
    }
    // A waypoint: dragged where it should be.
    if ( w >= 0 && !e.mods.isShiftDown() )
    {
        m_dragPoint = w;
        return;
    }
    const int  hit = presetAt( e.position, 7.0f );
    const auto m = toMap( e.position );
    if ( e.mods.isShiftDown() )
    {   // a waypoint: the preset under the mouse, or this place
        m_proc.addWaypoint( m.x, m.y, hit );
        return;
    }
    if ( hit >= 0 )
    {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition( hit, x, y );
        m_proc.moveCursor( x, y );
    }
    else
    {
        m_proc.moveCursor( m.x, m.y );
    }
}

void MapView::mouseDrag( const juce::MouseEvent &e )
{
    if ( m_overviewDrag )
    {   // the view centred where the mouse is on the overview
        const auto m = mini();
        centreOn( juce::jlimit( 0.0f, 1.0f, ( e.position.x - m.getX() ) / m.getWidth() ),
                  juce::jlimit( 0.0f, 1.0f, ( m.getBottom() - e.position.y ) / m.getHeight() ) );
        return;
    }
    if ( m_panning )
    {   // the map held under the mouse
        const auto p = plane();
        const auto d = e.position - e.mouseDownPosition;
        setView( m_panFrom.translated( -d.x / p.getWidth() * m_panFrom.getWidth(),
                                       d.y / p.getHeight() * m_panFrom.getHeight() ) );
        return;
    }
    if ( m_dragPoint >= 0 && m_dragPoint < m_proc.route().count() )
    {   // the waypoint follows; over a preset it snaps to it
        additiv::Waypoint w = m_proc.route().point( m_dragPoint );
        const int         hit = presetAt( e.position, 7.0f );
        float             x = 0.0f, y = 0.0f;
        if ( hit >= 0 )
            additiv::presetMapPosition( hit, x, y );
        else
        {
            const auto m = toMap( e.position );
            x = m.x;
            y = m.y;
        }
        w.m_x = std::clamp( x, 0.0f, 1.0f );
        w.m_y = std::clamp( y, 0.0f, 1.0f );
        w.m_preset = hit;
        m_proc.setWaypoint( m_dragPoint, w );
        repaint();
        return;
    }
    if ( e.mods.isShiftDown() ) return;
    const auto m = toMap( e.position );
    m_proc.moveCursor( m.x, m.y );
}

void MapView::mouseUp( const juce::MouseEvent & )
{
    m_dragPoint = -1;
    m_panning = false;
    m_overviewDrag = false;
}

void MapView::mouseMove( const juce::MouseEvent &e )
{
    if ( zoomed() && mini().expanded( 3.0f ).contains( e.position ) )
        m_hover = -1;
    else
        m_hover = plane().expanded( 4.0f ).contains( e.position ) ? presetAt( e.position, 7.0f ) : -1;
}

void MapView::mouseExit( const juce::MouseEvent & )
{
    m_hover = -1;
}

void MapView::mouseWheelMove( const juce::MouseEvent &e, const juce::MouseWheelDetails &wheel )
{
    if ( !plane().contains( e.position ) )
    {
        juce::Component::mouseWheelMove( e, wheel );   // the page's, should it scroll
        return;
    }
    const float sign = wheel.isReversed ? -1.0f : 1.0f;
    if ( wheel.deltaY != 0.0f )
        zoomAt( e.position,
                std::exp2( 2.0f * juce::jlimit( -1.0f, 1.0f, sign * wheel.deltaY ) ) );   // a notch: about 1.4 times
    else if ( wheel.deltaX != 0.0f )
        setView( m_view.translated( -0.5f * sign * wheel.deltaX * m_view.getWidth(), 0.0f ) );
}

void MapView::mouseMagnify( const juce::MouseEvent &e, float scaleFactor )
{
    if ( plane().contains( e.position ) && scaleFactor > 0.0f ) zoomAt( e.position, scaleFactor );
}

// ---- the routes

RouteView::RouteView( AdditivProcessor &proc ) : m_proc( proc )
{
    m_routes.setTextWhenNothingSelected( "Choose a route" );
    m_routes.setTooltip(
        "Routes over the map: the ones that come with the instrument, then yours (Documents/Empyrean/Routes)" );
    m_routes.onChange = [this]
    {
        const int id = m_routes.getSelectedId();
        if ( id > 0 && id <= static_cast<int>( m_entries.size() ) )
            m_proc.loadRoute( m_entries[static_cast<size_t>( id - 1 )] );
    };
    m_play.onClick = [this] { m_proc.startRoute(); };
    m_stop.onClick = [this] { m_proc.stopRoute(); };
    m_loop.setToggleState( m_proc.routeLoop(), juce::dontSendNotification );
    m_loop.onClick = [this] { m_proc.setRouteLoop( m_loop.getToggleState() ); };
    m_addCursor.setTooltip( "Appends the cursor's place as a waypoint (Shift-click on the map does it too)" );
    m_addCursor.onClick = [this]
    {
        const float x = m_proc.value( static_cast<int>( ParamIdE::MAP_X ) ),
                    y = m_proc.value( static_cast<int>( ParamIdE::MAP_Y ) );
        m_proc.addWaypoint( x, y, -1 );
    };
    m_addPreset.setTooltip( "Appends the preset playing as a waypoint" );
    m_addPreset.onClick = [this]
    {
        if ( m_proc.currentPreset() >= 0 ) m_proc.addWaypoint( 0.0f, 0.0f, m_proc.currentPreset() );
    };
    m_clear.onClick = [this] { m_proc.clearRoute(); };
    m_save.onClick = [this] { save(); };
    m_folder.onClick = []
    {
        const juce::File dir = AdditivProcessor::routeFolder();
        dir.createDirectory();
        dir.startAsProcess();
    };
    for ( juce::Label *l : { &m_status, &m_heads } )
    {
        l->setColour( juce::Label::textColourId, additivui::skin().m_dim );
        l->setFont( juce::FontOptions( 12.0f ) );
        addAndMakeVisible( l );
    }
    m_heads.setText( "waypoint                                   travel     hold     radius",
                     juce::dontSendNotification );
    for ( juce::Component *c :
          { static_cast<juce::Component *>( &m_routes ), static_cast<juce::Component *>( &m_play ),
            static_cast<juce::Component *>( &m_stop ), static_cast<juce::Component *>( &m_loop ),
            static_cast<juce::Component *>( &m_addCursor ), static_cast<juce::Component *>( &m_addPreset ),
            static_cast<juce::Component *>( &m_clear ), static_cast<juce::Component *>( &m_save ),
            static_cast<juce::Component *>( &m_folder ) } )
        addAndMakeVisible( c );
    refreshRoutes();
    rebuild();
    startTimerHz( 4 );
}

void RouteView::refreshRoutes()
{
    m_entries = m_proc.routes();
    m_routes.clear( juce::dontSendNotification );
    bool yours = false;
    m_routes.addSectionHeading( "Routes" );
    for ( size_t i = 0; i < m_entries.size(); ++i )
    {
        if ( m_entries[i].m_builtin < 0 && !yours )
        {
            m_routes.addSectionHeading( "Yours" );
            yours = true;
        }
        m_routes.addItem( m_entries[i].m_name, static_cast<int>( i ) + 1 );
    }
}

void RouteView::rebuild()
{
    m_rows.clear();
    const additiv::Route &r = m_proc.route();
    for ( int i = 0; i < r.count(); ++i )
    {
        auto                     row = std::make_unique<Row>();
        const additiv::Waypoint &w = r.point( i );
        row->m_name.setText( juce::String( i + 1 ) + "  " +
                                 ( w.m_preset >= 0 ? juce::String( additiv::preset( w.m_preset ).m_name )
                                                   : juce::String( w.m_x, 2 ) + ", " + juce::String( w.m_y, 2 ) ),
                             juce::dontSendNotification );
        row->m_name.setColour( juce::Label::textColourId, additivui::skin().m_ink );
        row->m_travel.setText( additiv::Journey::timeText( w.m_travel ), false );
        row->m_hold.setText( additiv::Journey::timeText( w.m_hold ), false );
        row->m_radius.setText( juce::String( w.m_radius, 3 ), false );
        row->m_travel.setTooltip( "How long the sound takes to glide here from the point before (m:ss)" );
        row->m_hold.setTooltip( "How long it stays here (m:ss)" );
        row->m_radius.setTooltip(
            "How far the blend reaches here, in the spacing of the presets round it (0.1 .. 4; under 0.5 a preset's own point plays it alone)" );
        for ( juce::TextEditor *t : { &row->m_travel, &row->m_hold, &row->m_radius } )
        {
            t->setJustification( juce::Justification::centred );
            t->onReturnKey = [this, i] { commit( i ); };
            t->onFocusLost = [this, i] { commit( i ); };
            addAndMakeVisible( t );
        }
        row->m_remove.onClick = [this, i] { m_proc.removeWaypoint( i ); };
        addAndMakeVisible( row->m_name );
        addAndMakeVisible( row->m_remove );
        m_rows.push_back( std::move( row ) );
    }
    std::vector<char> buf( static_cast<size_t>( r.count() ) * 200 + 64, 0 );
    m_shown = r.count() > 0 && r.write( buf.data(), buf.size() ) > 0 ? juce::String( buf.data() ) : juce::String();
    resized();
    if ( auto *p = getParentComponent() ) p->resized();   // the page grows with the rows
}

void RouteView::commit( int i )
{
    if ( i < 0 || i >= static_cast<int>( m_rows.size() ) || i >= m_proc.route().count() ) return;
    const Row        &row = *m_rows[static_cast<size_t>( i )];
    additiv::Waypoint w = m_proc.route().point( i );
    double            t = 0.0;
    if ( additiv::Journey::parseTime( row.m_travel.getText().trim().toStdString(), t ) )
        w.m_travel = static_cast<float>( std::max( 0.0, t ) );
    if ( additiv::Journey::parseTime( row.m_hold.getText().trim().toStdString(), t ) )
        w.m_hold = static_cast<float>( std::max( 0.0, t ) );
    w.m_radius = juce::jlimit( 0.1f, 4.0f,
                               row.m_radius.getText().getFloatValue() > 0.0f ? row.m_radius.getText().getFloatValue()
                                                                             : w.m_radius );
    m_proc.setWaypoint( i, w );
}

void RouteView::timerCallback()
{
    // The box shows the route that is loaded (chosen here, from the state, or by a shot).
    for ( size_t i = 0; i < m_entries.size(); ++i )
        if ( m_entries[i].m_name == m_proc.routeName() && m_routes.getSelectedId() != static_cast<int>( i ) + 1 )
        {
            m_routes.setSelectedId( static_cast<int>( i ) + 1, juce::dontSendNotification );
            break;
        }
    const juce::String st = m_proc.routeStatus();
    m_status.setText( st.isEmpty() ? ( m_proc.route().count() > 0
                                           ? juce::String( "Not walking." )
                                           : juce::String( "No route: choose one, or Shift-click the map." ) )
                                   : st,
                      juce::dontSendNotification );
    m_stop.setEnabled( m_proc.routeRunning() );
    m_play.setEnabled( m_proc.route().count() > 0 );
    // The rows anew when the route changed (a waypoint added, removed, a route chosen) and no field is being typed in.
    const additiv::Route &r = m_proc.route();
    std::vector<char>     buf( static_cast<size_t>( r.count() ) * 200 + 64, 0 );
    const juce::String    now =
        r.count() > 0 && r.write( buf.data(), buf.size() ) > 0 ? juce::String( buf.data() ) : juce::String();
    bool typing = false;
    for ( const auto &row : m_rows )
        typing = typing || row->m_travel.hasKeyboardFocus( true ) || row->m_hold.hasKeyboardFocus( true ) ||
                 row->m_radius.hasKeyboardFocus( true );
    if ( now != m_shown && !typing ) rebuild();
    if ( ++m_tick % 20 == 0 )
    {   // every five seconds: a route written by hand in the folder appears
        size_t files = 0;
        for ( const auto &en : m_entries )
            if ( en.m_builtin < 0 ) ++files;
        if ( AdditivProcessor::routeFolder().findChildFiles( juce::File::findFiles, false, "*.route" ).size() !=
             static_cast<int>( files ) )
            refreshRoutes();
    }
}

void RouteView::save()
{
    const juce::File dir = AdditivProcessor::routeFolder();
    dir.createDirectory();
    m_chooser =
        std::make_unique<juce::FileChooser>( "Save the route", dir.getChildFile( "My Route.route" ), "*.route" );
    m_chooser->launchAsync( juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting,
                            [this]( const juce::FileChooser &fc )
                            {
                                const juce::File f = fc.getResult();
                                if ( f == juce::File() ) return;
                                if ( m_proc.saveRoute( f.withFileExtension( "route" ) ) )
                                    refreshRoutes();
                                else
                                    juce::AlertWindow::showMessageBoxAsync( juce::MessageBoxIconType::WarningIcon,
                                                                            "Route",
                                                                            "The route could not be written there." );
                            } );
}

int RouteView::idealHeight() const
{
    return 28 + 30 + 22 + static_cast<int>( m_rows.size() ) * 28 + 40 + 10;
}

void RouteView::resized()
{
    const int w = getWidth();
    m_routes.setBounds( 12, 28, std::max( 120, w - 12 - 140 ), 26 );
    m_play.setBounds( w - 124, 28, 28, 26 );
    m_stop.setBounds( w - 92, 28, 28, 26 );
    m_loop.setBounds( w - 60, 28, 56, 26 );
    m_status.setBounds( 12, 58, w - 24, 20 );
    m_heads.setBounds( 12, 80, w - 24, 18 );
    int       y = 100;
    const int nameW = std::max( 80, w - 12 - 3 * 64 - 30 - 20 );
    for ( const auto &row : m_rows )
    {
        row->m_name.setBounds( 12, y, nameW, 24 );
        row->m_travel.setBounds( 12 + nameW + 4, y, 60, 24 );
        row->m_hold.setBounds( 12 + nameW + 68, y, 60, 24 );
        row->m_radius.setBounds( 12 + nameW + 132, y, 60, 24 );
        row->m_remove.setBounds( 12 + nameW + 196, y, 26, 24 );
        y += 28;
    }
    y += 6;
    m_addCursor.setBounds( 12, y, 90, 26 );
    m_addPreset.setBounds( 106, y, 90, 26 );
    m_clear.setBounds( 200, y, 64, 26 );
    m_save.setBounds( 270, y, 28, 26 );
    m_folder.setBounds( 302, y, 28, 26 );
}

void RouteView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( getLocalBounds().toFloat(), skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( getLocalBounds().toFloat().reduced( 0.5f ), skin.m_radius, 1.0f );
    g.setColour( skin.family( frame::FamilyE::MOTION ) );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.drawText( "Route: waypoints, the time to glide to each and to stay", 16, 4, getWidth() - 24, 20,
                juce::Justification::centredLeft );
}

juce::String RouteView::describe() const
{
    juce::String s( "Route: " );
    s << ( m_proc.routeName().isNotEmpty() ? m_proc.routeName() : juce::String( "(unnamed)" ) ) << ", "
      << m_proc.route().count() << " waypoints" << ( m_proc.routeLoop() ? ", looping" : "" ) << "\n";
    for ( int i = 0; i < m_proc.route().count(); ++i )
    {
        const additiv::Waypoint &w = m_proc.route().point( i );
        s << "    " << ( i + 1 ) << ". "
          << ( w.m_preset >= 0 ? juce::String( additiv::preset( w.m_preset ).m_name )
                               : juce::String( w.m_x, 2 ) + ", " + juce::String( w.m_y, 2 ) )
          << "  travel " << additiv::Journey::timeText( w.m_travel ) << "  hold "
          << additiv::Journey::timeText( w.m_hold ) << "  radius " << juce::String( w.m_radius, 3 ) << "\n";
    }
    return s;
}

// ---- the page

MapPage::MapPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live )
    : m_params( proc, actions,
                { { "Map",
                    frame::FamilyE::MOTION,
                    { ParamIdE::MAP_ACTIVE, ParamIdE::MAP_RADIUS, ParamIdE::MAP_GLIDE },
                    { ParamIdE::MAP_ACTIVE } },
                  { "Route", frame::FamilyE::SPACE, { ParamIdE::ROUTE_SPEED } } },
                live ),
      m_map( proc ), m_route( proc )
{
    addAndMakeVisible( m_params );
    addAndMakeVisible( m_map );
    addAndMakeVisible( m_route );
}

/** @brief The smallest side the map is given before the page scrolls. */
constexpr int MapMin = 360;

/**
 * @brief The map's largest side at a page width: a square, at most 600 pixels.
 * @param[in] w  the page's width
 * @return pixels
 */
static int mapSide( int w )
{
    return std::min( 600, ( w - 30 ) * 55 / 100 );
}

/**
 * @brief The right column's width beside a map.
 * @param[in] w     the page's width
 * @param[in] side  the map's side
 * @return pixels
 */
static int rightWidth( int w, int side )
{
    return w - 20 - side;
}

int MapPage::heightFor( int w ) const
{
    // Only the least it needs: the viewport makes the page as high as the window shows, and the
    // map takes that height (resized).
    const int right = m_params.heightFor( rightWidth( w, mapSide( w ) ) ) + m_route.idealHeight();
    return std::max( 10 + std::min( MapMin, mapSide( w ) ), right ) + 10;
}

void MapPage::resized()
{
    // The map from the top and as large as the page's height allows, so that the whole square is
    // seen without scrolling; the knobs and the routes beside it.
    const int side = juce::jlimit( std::min( MapMin, mapSide( getWidth() ) ), mapSide( getWidth() ), getHeight() - 20 );
    m_map.setBounds( 10, 10, side, side );
    const int x = 10 + side, w = rightWidth( getWidth(), side );
    const int top = m_params.heightFor( w );
    m_params.setBounds( x, 0, w, top );
    m_route.setBounds( x + 10, top, w - 20, std::max( m_route.idealHeight(), getHeight() - 10 - top ) );
}

juce::String MapPage::describe() const
{
    return m_params.describe() + m_route.describe() +
           "\nThe map lays every preset on a square: across from darker to brighter, up from still to moving. With Map on, the "
           "sound is the blend of the presets round the cursor (Radius: how far, in the spacing of the presets there), and it follows the cursor over Map Glide. Click "
           "a preset to go there, drag to wander; Shift-click adds a waypoint, a waypoint dragged moves (over a preset it snaps to it), "
           "a right-click removes it. The wheel zooms round the mouse (up to 32 times), a right- or middle-drag moves the view; "
           "zoomed in, the names stand at the points, the overview in the corner shows where the view is (click it to go there), "
           "Whole map goes back, and the view follows the cursor (with the map off: the preset playing) when a route or a preset takes it out. A route walks the cursor from waypoint to waypoint: "
           "each reached over its travel time -- the time the sound takes to turn into the next -- and held for its hold time, "
           "Route Speed faster or slower, round and round with Loop.\n";
}
