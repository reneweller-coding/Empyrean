/**
 * @file TuningView.cpp
 * @brief The Tuning page and its picture (TuningView.h).
 * @author Rene Weller
 */
#include "TuningView.h"

#include "Theme.h"

#include <algorithm>
#include <cmath>

using additiv::ParamIdE;

namespace
{

constexpr int     ViewMin = 200;   ///< the picture's least height
const char *const Names[12] = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};   ///< the steps' names

}   // namespace

TuningView::TuningView( AdditivProcessor &proc ) : m_proc( proc )
{
    startTimerHz( 15 );
}

void TuningView::timerCallback()
{
    if ( isShowing() )
    {
        repaint();
        return;
    }
    // The page left (its tab hides the page, not this view): nothing it plays may hang.
    if ( m_playing >= 0 ) m_proc.tapNote( m_playing, 0.0f );
    m_playing = -1;
    if ( !m_held.empty() ) releaseAll();
}

void TuningView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    const auto         b = getLocalBounds().toFloat();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( b, skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( b.reduced( 0.5f ), skin.m_radius, 1.0f );
    g.setColour( skin.family( frame::FamilyE::MOTION ) );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.drawText(
        "The octave over the root: equal temperament's steps, the scale's pitches (cents off the step), the voices where they sound",
        b.reduced( 18.0f, 4.0f ).withHeight( 20.0f ), juce::Justification::centredLeft );

    additiv::Engine &e = m_proc.engine();
    const int        rootPc =
        juce::jlimit( 0, 11, juce::roundToInt( m_proc.value( static_cast<int>( ParamIdE::ROOT_NOTE ) ) ) );
    const float purity = m_proc.value( static_cast<int>( ParamIdE::PURITY ) );
    const auto  area = this->area();
    auto        xOf = [&]( float cents ) { return area.getX() + area.getWidth() * cents / 1200.0f; };
    const float ruler = area.getBottom();

    // Equal temperament: a faint line every hundred cents, the step's name under it.
    g.setFont( juce::FontOptions( 11.0f ) );
    for ( int k = 0; k <= 12; ++k )
    {
        const float x = xOf( 100.0f * static_cast<float>( k ) );
        g.setColour( skin.m_faint.withAlpha( k % 12 == 0 ? 0.9f : 0.5f ) );
        g.drawVerticalLine( juce::roundToInt( x ), area.getY(), ruler );
        g.setColour( k % 12 == 0 ? skin.m_ink : skin.m_dim );
        g.drawText( Names[( rootPc + k ) % 12], juce::Rectangle<float>( x - 20.0f, ruler + 3.0f, 40.0f, 14.0f ),
                    juce::Justification::centred );
    }
    g.setColour( skin.m_edge );
    g.drawHorizontalLine( juce::roundToInt( ruler ), area.getX(), area.getRight() );

    // The scale: a line at each pitch, how far it stands from the nearest step above it.
    float              cents[48];
    const int          n = e.scaleCents( cents, 48 );
    const juce::Colour scaleColour = skin.family( frame::FamilyE::MOTION );
    // The pitches the mouse plays or Shift holds: their lines lit.
    std::vector<float> sounding;
    for ( int note : m_held )
        sounding.push_back( static_cast<float>(
            std::fmod( 1200.0 * std::log2( std::max( e.noteHz( note ), 1.0f ) / rootHz0() ) + 12000.0, 1200.0 ) ) );
    if ( m_playing >= 0 )
        sounding.push_back( static_cast<float>( std::fmod(
            1200.0 * std::log2( std::max( e.noteHz( m_playing ), 1.0f ) / rootHz0() ) + 12000.0, 1200.0 ) ) );
    for ( int i = 0; i < n; ++i )
    {
        const float x = xOf( cents[i] );
        bool        lit = false;
        for ( float c : sounding ) lit = lit || std::fabs( c - cents[i] ) < 3.0f;
        g.setColour( lit ? skin.m_ink : scaleColour.withAlpha( 0.75f ) );
        g.fillRect( x - ( lit ? 1.5f : 1.0f ), area.getY() + 14.0f, lit ? 3.0f : 2.0f, ruler - area.getY() - 14.0f );
        const float off = cents[i] - 100.0f * std::round( cents[i] / 100.0f );
        if ( std::fabs( off ) >= 1.0f )
        {
            g.setColour( scaleColour );
            g.drawText( ( off > 0.0f ? "+" : "" ) + juce::String( juce::roundToInt( off ) ),
                        juce::Rectangle<float>( x - 18.0f, area.getY(), 36.0f, 13.0f ), juce::Justification::centred );
        }
    }

    // The voices: across where they sound in the octave, up by their octave, a tail as long as Self-Tune pulls.
    const float        rootHz = std::max( e.rootHz(), 1.0f );
    const juce::Colour bySource[4] = { skin.m_accent, additivui::colour::Violet, skin.family( frame::FamilyE::SPACE ),
                                       skin.m_dim };
    const float        lane = ( ruler - area.getY() - 22.0f ) / 7.0f;   // seven octaves, from two below the root's
    for ( int i = 0; i < additiv::MaxViewVoices; ++i )
    {
        const additiv::VoiceView v = e.voiceView( i );
        if ( v.m_hz <= 0.0f || v.m_gain < 0.03f ) continue;
        const double c = 1200.0 * std::log2( static_cast<double>( v.m_hz ) / rootHz );
        const double octave = std::floor( c / 1200.0 );
        const float  within = static_cast<float>( c - 1200.0 * octave );
        const float  y =
            ruler - 8.0f - lane * static_cast<float>( juce::jlimit( 0.0, 6.0, octave + 2.0 ) ) - lane * 0.5f;
        const float        x = xOf( within ), r = 3.0f + 4.0f * v.m_gain;
        const juce::Colour col = bySource[juce::jlimit( 0, 3, v.m_source )];
        if ( std::fabs( v.m_tune ) > 0.3f )
        {
            g.setColour( col.withAlpha( 0.5f ) );
            g.drawLine( xOf( within - v.m_tune ), y, x, y, 2.0f );
        }
        g.setColour( col.withAlpha( 0.35f + 0.65f * v.m_gain ) );
        g.fillEllipse( x - r, y - r, 2.0f * r, 2.0f * r );
    }

    g.setColour( skin.m_dim );
    g.setFont( juce::FontOptions( 11.5f ) );
    g.drawText( "Click a line to hear it (Shift holds it, with others: an interval locking)   Purity " +
                    juce::String( purity, 2 ) + ": the voices stand " +
                    juce::String( juce::roundToInt( 100.0f * purity ) ) +
                    " % of the way from equal temperament to the scale",
                b.reduced( 18.0f, 4.0f ).removeFromBottom( 16.0f ), juce::Justification::centredLeft );
}

juce::Rectangle<float> TuningView::area() const
{
    return getLocalBounds().toFloat().reduced( 30.0f, 0.0f ).withTrimmedTop( 46.0f ).withTrimmedBottom( 42.0f );
}

float TuningView::rootHz0() const
{
    return std::max( m_proc.engine().rootHz(), 1.0f );
}

void TuningView::mouseDown( const juce::MouseEvent &e )
{
    additiv::Engine &eng = m_proc.engine();
    const auto       a = area();
    if ( !a.expanded( 0.0f, 10.0f ).contains( e.position ) ) return;
    // The nearest line of the scale (within ten pixels), in the octave the height says.
    float     cents[48];
    const int n = eng.scaleCents( cents, 48 );
    int       best = -1;
    float     bestD = 10.0f;
    for ( int i = 0; i < n; ++i )
    {
        const float d = std::fabs( a.getX() + a.getWidth() * cents[i] / 1200.0f - e.position.x );
        if ( d < bestD )
        {
            bestD = d;
            best = i;
        }
    }
    if ( best < 0 ) return;
    const float lane = ( a.getBottom() - a.getY() - 22.0f ) / 7.0f;
    const int   octave =
        juce::jlimit( 0, 6, static_cast<int>( ( a.getBottom() - 8.0f - e.position.y ) / std::max( lane, 1.0f ) ) ) - 2;
    const double want = rootHz0() * std::exp2( octave + cents[best] / 1200.0 );
    int          note = -1;
    double       nearest = 1.0e9;
    for ( int k = 12; k < 120; ++k )
    {
        const double hz = eng.noteHz( k );
        if ( hz <= 0.0 ) continue;
        const double d = std::fabs( std::log2( hz / want ) );
        if ( d < nearest )
        {
            nearest = d;
            note = k;
        }
    }
    if ( note < 0 || nearest > 0.05 ) return;   // no key sounds that pitch (a scale with fewer steps than keys)
    if ( e.mods.isShiftDown() )
    {   // held: added, or let go when it is held already
        if ( auto it = std::find( m_held.begin(), m_held.end(), note ); it != m_held.end() )
        {
            m_proc.tapNote( note, 0.0f );
            m_held.erase( it );
        }
        else
        {
            m_proc.tapNote( note, 0.7f );
            m_held.push_back( note );
        }
        repaint();
        return;
    }
    releaseAll();
    m_playing = note;
    m_proc.tapNote( note, 0.7f );
    repaint();
}

void TuningView::mouseUp( const juce::MouseEvent & )
{
    if ( m_playing >= 0 ) m_proc.tapNote( m_playing, 0.0f );
    m_playing = -1;
    repaint();
}

void TuningView::visibilityChanged()
{
    if ( !isShowing() ) releaseAll();
}

void TuningView::releaseAll()
{
    for ( int note : m_held ) m_proc.tapNote( note, 0.0f );
    m_held.clear();
}

juce::String TuningView::describe() const
{
    return "\nBelow the knobs, the octave over the root as a ruler: equal temperament's twelve steps as faint lines with their names, the "
           "scale's pitches as lines with how far each stands from its step (in cents), and every voice as a light where it sounds in the "
           "octave -- higher up for a higher octave, green for yours, violet for the conductor's -- with a tail as long as Self-Tune is "
           "pulling it towards a pure interval. A click on a line plays it, in the octave the height says; Shift holds it (and the others "
           "Shift held), so an interval can be heard locking as Purity and Self-Tune move.\n";
}

// ============================================================================================================== page

TuningPage::TuningPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live )
    : m_params(
          proc, actions,
          { { "Tuning",
              frame::FamilyE::MOTION,
              { ParamIdE::SCALE, ParamIdE::ROOT_NOTE, ParamIdE::KEY_MAP, ParamIdE::REF_PITCH, ParamIdE::PURITY } },
            { "Tide", frame::FamilyE::MOTION, { ParamIdE::TIDE, ParamIdE::TIDE_PERIOD }, { ParamIdE::TIDE } },
            { "Self-Tune",
              frame::FamilyE::MOTION,
              { ParamIdE::TUNE_GRAVITY, ParamIdE::TUNE_REACH, ParamIdE::TUNE_ANCHOR },
              { ParamIdE::TUNE_GRAVITY } } },
          live ),
      m_view( proc )
{
    addAndMakeVisible( m_params );
    addAndMakeVisible( m_view );
}

int TuningPage::heightFor( int w ) const
{
    return m_params.heightFor( w ) + ViewMin + 10;
}

void TuningPage::resized()
{
    const int top = m_params.heightFor( getWidth() );
    m_params.setBounds( 0, 0, getWidth(), top );
    m_view.setBounds( 10, top, getWidth() - 20, std::max( ViewMin, getHeight() - 10 - top ) );
}

juce::String TuningPage::describe() const
{
    return m_params.describe() + m_view.describe();
}
