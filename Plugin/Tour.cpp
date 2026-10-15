/**
 * @file Tour.cpp
 * @brief The first steps (Tour.h).
 * @author Rene Weller
 */
#include "Tour.h"

#include "Theme.h"

Tour::Tour()
{
    m_next.onClick = [this] { next(); };
    m_skip.onClick = [this] { finish(); };
    addAndMakeVisible( m_next );
    addAndMakeVisible( m_skip );
    setInterceptsMouseClicks( true, true );
}

void Tour::start( std::vector<Step> steps )
{
    m_steps = std::move( steps );
    m_at = 0;
    if ( m_steps.empty() ) return;
    setVisible( true );
    toFront( true );
    resized();
    repaint();
}

void Tour::next()
{
    if ( ++m_at >= m_steps.size() )
    {
        finish();
        return;
    }
    resized();
    repaint();
}

void Tour::finish()
{
    setVisible( false );
    m_steps.clear();
    if ( m_onDone ) m_onDone();
}

juce::Rectangle<int> Tour::hole() const
{
    if ( m_at >= m_steps.size() || m_steps[m_at].m_target == nullptr ) return {};
    return getLocalArea( m_steps[m_at].m_target.getComponent(), m_steps[m_at].m_target->getLocalBounds() )
        .expanded( 6 );
}

juce::Rectangle<int> Tour::bubble() const
{
    const auto h = hole();
    const int  w = 420, ht = 150;
    int        x = juce::jlimit( 12, std::max( 12, getWidth() - w - 12 ), h.getCentreX() - w / 2 );
    int        y = h.getBottom() + 14;
    if ( y + ht > getHeight() - 12 ) y = std::max( 12, h.getY() - ht - 14 );   // above it when there is no room below
    return { x, y, w, ht };
}

void Tour::resized()
{
    const auto b = bubble();
    m_next.setButtonText( m_at + 1 >= m_steps.size() ? "Done" : "Next" );
    m_next.setBounds( b.getRight() - 92, b.getBottom() - 36, 80, 26 );
    m_skip.setBounds( b.getRight() - 180, b.getBottom() - 36, 80, 26 );
    m_skip.setVisible( m_at + 1 < m_steps.size() );
}

void Tour::paint( juce::Graphics &g )
{
    if ( m_at >= m_steps.size() ) return;
    const frame::Skin &skin = additivui::skin();
    const auto         h = hole();
    // The window dimmed, the control left bright.
    juce::Path dim;
    dim.addRectangle( getLocalBounds().toFloat() );
    if ( !h.isEmpty() ) dim.addRoundedRectangle( h.toFloat(), 8.0f );
    dim.setUsingNonZeroWinding( false );
    g.setColour( skin.m_bg.withAlpha( 0.72f ) );
    g.fillPath( dim );
    if ( !h.isEmpty() )
    {
        g.setColour( skin.m_accent );
        g.drawRoundedRectangle( h.toFloat(), 8.0f, 2.0f );
    }
    const auto b = bubble().toFloat();
    g.setColour( skin.m_group );
    g.fillRoundedRectangle( b, 8.0f );
    g.setColour( skin.m_accent.withAlpha( 0.7f ) );
    g.drawRoundedRectangle( b.reduced( 0.5f ), 8.0f, 1.2f );
    const Step &s = m_steps[m_at];
    g.setColour( skin.m_accent );
    g.setFont( juce::FontOptions( 15.0f, juce::Font::bold ) );
    g.drawText( s.m_title, b.reduced( 14.0f, 10.0f ).withHeight( 22.0f ), juce::Justification::centredLeft );
    g.setColour( skin.m_dim );
    g.setFont( juce::FontOptions( 11.0f ) );
    g.drawText( juce::String( static_cast<int>( m_at ) + 1 ) + " / " +
                    juce::String( static_cast<int>( m_steps.size() ) ),
                b.reduced( 14.0f, 10.0f ).withHeight( 22.0f ), juce::Justification::centredRight );
    g.setColour( skin.m_ink );
    g.setFont( juce::FontOptions( 13.0f ) );
    g.drawFittedText( s.m_text,
                      b.reduced( 14.0f, 10.0f ).withTrimmedTop( 26.0f ).withTrimmedBottom( 36.0f ).toNearestInt(),
                      juce::Justification::topLeft, 4 );
}
