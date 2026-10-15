/**
 * @file HeaderParts.cpp
 * @brief The header's own glyph buttons and the meter (HeaderParts.h).
 * @author Rene Weller
 */
#include "HeaderParts.h"

#include "Theme.h"

#include <algorithm>
#include <cmath>

ViewButton::ViewButton( GlyphE glyph, const juce::String &tooltip ) : juce::Button( tooltip ), m_glyph( glyph )
{
    setTooltip( tooltip );
    setWantsKeyboardFocus( false );
}

void ViewButton::paintButton( juce::Graphics &g, bool highlighted, bool down )
{
    getLookAndFeel().drawButtonBackground(
        g, *this,
        findColour( getToggleState() ? juce::TextButton::buttonOnColourId : juce::TextButton::buttonColourId ),
        highlighted && isEnabled(), down );
    const auto         b = getLocalBounds().toFloat();
    const float        s = std::min( std::min( b.getWidth(), b.getHeight() ) * 0.56f, 20.0f );
    const auto         r = b.withSizeKeepingCentre( s, s );
    const juce::Colour ink =
        ( getToggleState() ? additivui::skin().m_accent : findColour( juce::TextButton::textColourOffId ) )
            .withMultipliedAlpha( isEnabled() ? 1.0f : 0.35f );
    g.setColour( ink );
    const float                w = std::max( 1.4f, s * 0.11f );
    const juce::PathStrokeType stroke( w, juce::PathStrokeType::curved, juce::PathStrokeType::rounded );
    switch ( m_glyph )
    {
    case GlyphE::NIGHT:
    {   // a crescent: a disc with a disc taken out of it (clipped away, not painted over)
        juce::Path moon, outsideBite;
        moon.addEllipse( r.reduced( s * 0.08f ) );
        outsideBite.addRectangle( b );
        outsideBite.addEllipse( r.reduced( s * 0.08f ).translated( s * 0.32f, -s * 0.22f ) );
        outsideBite.setUsingNonZeroWinding( false );
        g.saveState();
        g.reduceClipRegion( outsideBite );
        g.fillPath( moon );
        g.restoreState();
        break;
    }
    case GlyphE::SEARCH:
    {   // a lens and its handle
        const float rad = s * 0.3f;
        const float cx = r.getX() + s * 0.42f, cy = r.getY() + s * 0.42f;
        juce::Path  lens;
        lens.addEllipse( cx - rad, cy - rad, 2.0f * rad, 2.0f * rad );
        g.strokePath( lens, stroke );
        g.drawLine( cx + rad * 0.72f, cy + rad * 0.72f, r.getRight() - s * 0.06f, r.getBottom() - s * 0.06f, w * 1.3f );
        break;
    }
    case GlyphE::CANVAS:
    {   // four corners of a frame
        const float k = s * 0.32f;
        juce::Path  p;
        p.startNewSubPath( r.getX(), r.getY() + k );
        p.lineTo( r.getX(), r.getY() );
        p.lineTo( r.getX() + k, r.getY() );
        p.startNewSubPath( r.getRight() - k, r.getY() );
        p.lineTo( r.getRight(), r.getY() );
        p.lineTo( r.getRight(), r.getY() + k );
        p.startNewSubPath( r.getRight(), r.getBottom() - k );
        p.lineTo( r.getRight(), r.getBottom() );
        p.lineTo( r.getRight() - k, r.getBottom() );
        p.startNewSubPath( r.getX() + k, r.getBottom() );
        p.lineTo( r.getX(), r.getBottom() );
        p.lineTo( r.getX(), r.getBottom() - k );
        g.strokePath( p, stroke );
        break;
    }
    case GlyphE::COMPARE:
    {   // A | B
        g.setFont( juce::FontOptions( s * 0.62f, juce::Font::bold ) );
        g.drawText( "A", r.withWidth( s * 0.45f ), juce::Justification::centred );
        g.drawText( "B", r.withTrimmedLeft( s * 0.55f ), juce::Justification::centred );
        g.drawLine( r.getCentreX(), r.getY() + s * 0.1f, r.getCentreX(), r.getBottom() - s * 0.1f, w * 0.8f );
        break;
    }
    }
}

// ============================================================================================================ meter

OutputMeter::OutputMeter()
{
    setTooltip(
        "The output: left above, right below, from -60 to 0 dBFS; red at the end once it reached full scale (a click puts it out)" );
}

void OutputMeter::push( float left, float right )
{
    const float in[2] = { left, right };
    for ( int c = 0; c < 2; ++c )
    {
        m_level[c] = std::max( in[c], m_level[c] * 0.82f );   // falls about 20 dB a second at 15 looks a second
        if ( in[c] >= m_hold[c] || m_holdTicks[c] <= 0 )
        {
            m_hold[c] = in[c];
            m_holdTicks[c] = 22;   // a second and a half
        }
        else
        {
            --m_holdTicks[c];
        }
        if ( in[c] >= 0.999f ) m_clipped = true;
    }
    repaint();
}

void OutputMeter::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    const auto         b = getLocalBounds().toFloat();
    const float        clipW = 6.0f;
    const auto         bars = b.withTrimmedRight( clipW + 3.0f );
    const float        h = ( bars.getHeight() - 2.0f ) / 2.0f;
    auto               xOf = [&]( float amp )
    {
        const float db = 20.0f * std::log10( std::max( amp, 1.0e-6f ) );
        return bars.getX() + bars.getWidth() * std::clamp( ( db + 60.0f ) / 60.0f, 0.0f, 1.0f );
    };
    for ( int c = 0; c < 2; ++c )
    {
        const juce::Rectangle<float> bar( bars.getX(), bars.getY() + static_cast<float>( c ) * ( h + 2.0f ),
                                          bars.getWidth(), h );
        g.setColour( skin.m_raised );
        g.fillRoundedRectangle( bar, 1.5f );
        const float x = xOf( m_level[c] );
        // Teal to -12 dBFS, amber to -3, red above.
        const float x12 = xOf( std::pow( 10.0f, -12.0f / 20.0f ) ), x3 = xOf( std::pow( 10.0f, -3.0f / 20.0f ) );
        g.setColour( skin.m_accent.withAlpha( 0.85f ) );
        g.fillRect(
            juce::Rectangle<float>::leftTopRightBottom( bar.getX(), bar.getY(), std::min( x, x12 ), bar.getBottom() ) );
        if ( x > x12 )
        {
            g.setColour( juce::Colour( 0xffe0b050 ) );
            g.fillRect(
                juce::Rectangle<float>::leftTopRightBottom( x12, bar.getY(), std::min( x, x3 ), bar.getBottom() ) );
        }
        if ( x > x3 )
        {
            g.setColour( juce::Colour( 0xffe05a4a ) );
            g.fillRect( juce::Rectangle<float>::leftTopRightBottom( x3, bar.getY(), x, bar.getBottom() ) );
        }
        if ( m_hold[c] > 1.0e-4f )
        {
            g.setColour( skin.m_ink.withAlpha( 0.8f ) );
            g.fillRect( xOf( m_hold[c] ) - 1.0f, bar.getY(), 1.5f, bar.getHeight() );
        }
    }
    const juce::Rectangle<float> clip( b.getRight() - clipW, b.getY(), clipW, b.getHeight() );
    g.setColour( m_clipped ? juce::Colour( 0xffe05a4a ) : skin.m_raised );
    g.fillRoundedRectangle( clip, 1.5f );
}

void OutputMeter::mouseDown( const juce::MouseEvent & )
{
    m_clipped = false;
    repaint();
}
