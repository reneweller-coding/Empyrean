/**
 * @file PageStrip.cpp
 * @brief The pages' strip (PageStrip.h).
 * @author Rene Weller
 */
#include "PageStrip.h"

#include "Theme.h"

namespace
{

constexpr int Head = 12;     ///< the headings' row
constexpr int GapIn = 2;     ///< between the pages of a section
constexpr int GapOut = 22;   ///< between the sections
constexpr int DotSize = 7;   ///< a dot's size
constexpr int DotStep = 9;   ///< a dot's place

/** @brief The pages' names' font. @return it */
juce::Font nameFont()
{
    return juce::Font( juce::FontOptions( 14.0f ) );
}

}   // namespace

void PageStrip::setPages( juce::StringArray names, std::vector<Section> sections )
{
    m_names = std::move( names );
    m_sections = std::move( sections );
    m_status.assign( static_cast<size_t>( m_names.size() ), Status{} );
    m_bounds.assign( static_cast<size_t>( m_names.size() ), {} );
    resized();
}

void PageStrip::setCurrent( int page )
{
    if ( page == m_current ) return;
    m_current = page;
    repaint();
}

void PageStrip::refresh()
{
    if ( !m_statusOf ) return;
    bool changed = false;
    for ( int p = 0; p < m_names.size(); ++p )
    {
        Status  s = m_statusOf( p );
        Status &was = m_status[static_cast<size_t>( p )];
        bool    same = s.m_badge == was.m_badge && s.m_quiet == was.m_quiet && s.m_dots.size() == was.m_dots.size();
        for ( size_t k = 0; same && k < s.m_dots.size(); ++k ) same = s.m_dots[k].m_on == was.m_dots[k].m_on;
        if ( !same ) changed = true;
        was = std::move( s );
    }
    if ( changed )
    {
        resized();
        repaint();
    }
}

int PageStrip::widthOf( int page ) const
{
    const Status &s = m_status[static_cast<size_t>( page )];
    int           w = juce::GlyphArrangement::getStringWidthInt( nameFont(), m_names[page] ) + 20;
    if ( !s.m_dots.empty() ) w += 4 + static_cast<int>( s.m_dots.size() ) * DotStep;
    if ( s.m_badge.isNotEmpty() )
        w += 6 +
             juce::GlyphArrangement::getStringWidthInt( juce::Font( juce::FontOptions( 10.5f, juce::Font::bold ) ),
                                                        s.m_badge ) +
             8;
    return w;
}

int PageStrip::idealWidth() const
{
    int w = 0;
    for ( const Section &sec : m_sections )
    {
        for ( int p : sec.m_pages ) w += widthOf( p ) + GapIn;
        w += GapOut;
    }
    return w;
}

void PageStrip::resized()
{
    m_heads.clear();
    int x = 0;
    for ( const Section &sec : m_sections )
    {
        const int x0 = x;
        for ( int p : sec.m_pages )
        {
            const int w = widthOf( p );
            m_bounds[static_cast<size_t>( p )] = { x, Head, w, getHeight() - Head };
            x += w + GapIn;
        }
        m_heads.emplace_back( x0, 0, x - GapIn - x0, Head );
        x += GapOut - GapIn;
    }
}

void PageStrip::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_edge );
    g.fillRect( 0, getHeight() - 1, getWidth(), 1 );
    // The headings: small capitals over their pages, a hairline from the name to the section's end.
    for ( size_t i = 0; i < m_sections.size() && i < m_heads.size(); ++i )
    {
        if ( m_sections[i].m_name.isEmpty() ) continue;
        const auto         h = m_heads[i];
        const juce::Font   f( juce::FontOptions( 10.0f, juce::Font::bold ) );
        const juce::String text = m_sections[i].m_name.toUpperCase();
        const int          tw = juce::GlyphArrangement::getStringWidthInt( f, text );
        g.setColour( skin.m_dim.withAlpha( 0.8f ) );
        g.setFont( f );
        g.drawText( text, h.getX() + 8, h.getY(), tw + 2, h.getHeight(), juce::Justification::centredLeft );
        g.setColour( skin.m_edge );
        g.fillRect( h.getX() + 8 + tw + 6, h.getCentreY(), std::max( 0, h.getRight() - 8 - ( h.getX() + 8 + tw + 6 ) ),
                    1 );
    }
    for ( int p = 0; p < m_names.size(); ++p )
    {
        const auto    r = m_bounds[static_cast<size_t>( p )];
        const Status &s = m_status[static_cast<size_t>( p )];
        const bool    front = p == m_current, hover = p == m_hover;
        if ( front )
        {
            g.setColour( skin.m_panel );
            g.fillRect( r );
            g.setColour( skin.m_accent );
            g.fillRect( static_cast<float>( r.getX() ) + 6.0f, static_cast<float>( r.getBottom() ) - 2.5f,
                        static_cast<float>( r.getWidth() ) - 12.0f, 2.5f );
        }
        else if ( hover )
        {
            g.setColour( skin.m_group.withAlpha( 0.8f ) );
            g.fillRect( r );
        }
        juce::Colour text = front ? skin.m_accent : ( hover ? skin.m_ink : skin.m_dim );
        if ( s.m_quiet && !front ) text = text.withAlpha( 0.5f );
        g.setColour( text );
        g.setFont( nameFont() );
        const int nameW = juce::GlyphArrangement::getStringWidthInt( nameFont(), m_names[p] );
        int       x = r.getX() + 10;
        g.drawText( m_names[p], x, r.getY(), nameW + 2, r.getHeight() - 3, juce::Justification::centredLeft );
        x += nameW + 4;
        // The groups: lit while awake, a faint ring while resting.
        const float cy = static_cast<float>( r.getY() ) + static_cast<float>( r.getHeight() - 3 ) * 0.5f;
        for ( const Dot &d : s.m_dots )
        {
            const juce::Rectangle<float> dot( static_cast<float>( x ) + 1.0f, cy - DotSize * 0.5f, DotSize, DotSize );
            if ( d.m_on )
            {
                g.setColour( d.m_colour );
                g.fillEllipse( dot );
            }
            else
            {
                g.setColour( skin.m_faint );
                g.drawEllipse( dot.reduced( 0.5f ), 1.0f );
            }
            x += DotStep;
        }
        if ( s.m_badge.isNotEmpty() )
        {
            const juce::Font             f( juce::FontOptions( 10.5f, juce::Font::bold ) );
            const int                    bw = juce::GlyphArrangement::getStringWidthInt( f, s.m_badge ) + 8;
            const juce::Rectangle<float> b( static_cast<float>( x ) + 4.0f, cy - 7.0f, static_cast<float>( bw ),
                                            14.0f );
            g.setColour( skin.m_accent.withAlpha( 0.25f ) );
            g.fillRoundedRectangle( b, 7.0f );
            g.setColour( skin.m_ink );
            g.setFont( f );
            g.drawText( s.m_badge, b, juce::Justification::centred );
        }
    }
}

int PageStrip::pageAt( juce::Point<int> p ) const
{
    for ( size_t i = 0; i < m_bounds.size(); ++i )
        if ( m_bounds[i].contains( p ) ) return static_cast<int>( i );
    return -1;
}

void PageStrip::mouseDown( const juce::MouseEvent &e )
{
    const int p = pageAt( e.getPosition() );
    if ( p >= 0 && m_onSelect ) m_onSelect( p );
}

void PageStrip::mouseMove( const juce::MouseEvent &e )
{
    const int p = pageAt( e.getPosition() );
    if ( p != m_hover )
    {
        m_hover = p;
        repaint();
    }
}

void PageStrip::mouseExit( const juce::MouseEvent & )
{
    m_hover = -1;
    repaint();
}

juce::String PageStrip::getTooltip()
{
    if ( m_hover < 0 || static_cast<size_t>( m_hover ) >= m_status.size() ) return {};
    const Status &s = m_status[static_cast<size_t>( m_hover )];
    return m_names[m_hover] + ( s.m_tip.isNotEmpty() ? ": " + s.m_tip : juce::String() );
}
