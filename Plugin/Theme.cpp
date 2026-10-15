/**
 * @file Theme.cpp
 * @brief The instrument's skin (Theme.h).
 * @author Rene Weller
 */
#include "Theme.h"
#include "AdditivData.h"

#include <cmath>

namespace additivui
{

void drawLogo( juce::Graphics &g, juce::Rectangle<float> r )
{
    const float                  side = std::min( r.getWidth(), r.getHeight() );
    const juce::Rectangle<float> box = r.withSizeKeepingCentre( side, side );
    const float                  base = box.getY() + 0.86f * side;
    // The overtone ladder: ten partials of one tone as lights on stalks, their heights falling as the harmonic series
    // does, mint turning violet, each light in a faint glow. It fills the square as the siblings' logos do (the icon,
    // Deploy/make_icon.py, draws it smaller inside its rounded square).
    for ( int i = 0; i < 10; ++i )
    {
        const float              t = static_cast<float>( i ) / 9.0f;
        const float              x = box.getX() + ( 0.07f + 0.76f * t ) * side;
        const float              top = base - 0.68f * side / std::pow( static_cast<float>( i + 1 ), 0.55f );
        const float              dot = ( 0.13f - 0.057f * t ) * side;
        const juce::Colour       c = colour::Accent.interpolatedWith( colour::Violet, t );
        const juce::Point<float> p( x, top );
        g.setColour( c.withAlpha( 0.6f ) );
        g.drawLine( x, base, x, top, std::max( 1.0f, 0.024f * side ) );
        g.setColour( c.withAlpha( 0.2f ) );
        g.fillEllipse( juce::Rectangle<float>( 2.2f * dot, 2.2f * dot ).withCentre( p ) );
        g.setColour( c );
        g.fillEllipse( juce::Rectangle<float>( dot, dot ).withCentre( p ) );
    }
}

const frame::Skin &skin()
{
    static const frame::Skin s = []
    {
        frame::Skin k;
        k.m_name = "Empyrean";
        k.m_bg = colour::Bg;
        k.m_panel = colour::Panel;
        k.m_group = colour::GroupColour;
        k.m_raised = colour::Raised;
        k.m_edge = colour::Edge;
        k.m_ink = colour::Ink;
        k.m_dim = colour::Dim;
        k.m_faint = colour::Faint;
        k.m_accent = colour::Accent;
        k.m_onset = colour::Onset;
        k.m_good = colour::Good;
        k.m_bad = colour::Bad;
        k.m_families[static_cast<int>( frame::FamilyE::SOURCE )] = colour::Accent;
        k.m_families[static_cast<int>( frame::FamilyE::FILTER )] = juce::Colour( 0xffe0b36a );
        k.m_families[static_cast<int>( frame::FamilyE::ENVELOPE )] = colour::Violet;
        k.m_families[static_cast<int>( frame::FamilyE::MOTION )] = juce::Colour( 0xff6fb4ff );
        k.m_families[static_cast<int>( frame::FamilyE::SPACE )] = juce::Colour( 0xff9fd0e6 );
        for ( int d = 0; d < 3; ++d ) k.m_decks[d] = colour::Accent;
        k.m_radius = 6.0f;   // soft, as the light is
        k.m_tracking = 0.22f;
        k.m_typeface = "Segoe UI Light";
        k.m_titleBold = false;
        k.m_glow = true;
        k.m_backdropData = AdditivData::backdrop_jpg;
        k.m_backdropSize = AdditivData::backdrop_jpgSize;
        k.m_backdropTop = 0.8f;
        k.m_backdropPage = 0.45f;
        k.m_logo = []( juce::Graphics &g, juce::Rectangle<float> r ) { drawLogo( g, r ); };
        return k;
    }();
    return s;
}

}   // namespace additivui
