/**
 * @file ModDock.cpp
 * @brief The modulation dock and the source inspector (ModDock.h).
 * @author Rene Weller
 */
#include "ModDock.h"

#include "KnobMod.h"
#include "Theme.h"

#include <algorithm>
#include <cmath>

using additiv::ModRow;
using additiv::ModSourceE;
using additiv::ParamIdE;
using knobmod::sourceColour;

namespace
{

/** @brief The sources that always have a chip: the classic ones and the hands'. */
constexpr ModSourceE Classic[] = { ModSourceE::LFO1,     ModSourceE::LFO2,      ModSourceE::LFO3,
                                   ModSourceE::LFO4,     ModSourceE::ENV1,      ModSourceE::ENV2,
                                   ModSourceE::MACRO1,   ModSourceE::MACRO2,    ModSourceE::MACRO3,
                                   ModSourceE::MACRO4,   ModSourceE::MOD_WHEEL, ModSourceE::PRESSURE,
                                   ModSourceE::VELOCITY, ModSourceE::RANDOM };

/**
 * @brief Whether a source always has a chip.
 * @param[in] s  the source
 * @return true then
 */
bool classic( ModSourceE s )
{
    return std::find( std::begin( Classic ), std::end( Classic ), s ) != std::end( Classic );
}

constexpr int ToggleW = 112;   ///< the dock's button
constexpr int PlusW = 30;      ///< the "+" chip
constexpr int ChipMin = 58;    ///< a chip's least width
constexpr int ChipMax = 86;    ///< its greatest

/**
 * @brief Draws a trace: bipolar about the middle, unipolar from the floor.
 * @param[in,out] g      where
 * @param[in]     r      the area
 * @param[in]     dock   whose traces
 * @param[in]     s      the source
 * @param[in]     width  the line's width
 */
void drawTrace( juce::Graphics &g, juce::Rectangle<float> r, const ModDock &dock, ModSourceE s, float width )
{
    const bool uni = additiv::isUnipolarSource( s );
    auto       yOf = [&]( float v )
    {
        const float u = uni ? std::clamp( v, 0.0f, 1.0f ) : 0.5f + 0.5f * std::clamp( v, -1.0f, 1.0f );
        return r.getBottom() - u * r.getHeight();
    };
    g.setColour( additivui::skin().m_faint );
    g.drawHorizontalLine( juce::roundToInt( yOf( 0.0f ) ), r.getX(), r.getRight() );
    juce::Path p;
    for ( int k = 0; k < ModDock::History; ++k )
    {
        const float x = r.getX() + r.getWidth() * static_cast<float>( k ) / static_cast<float>( ModDock::History - 1 );
        const float y = yOf( dock.traceAt( s, k ) );
        if ( k == 0 )
            p.startNewSubPath( x, y );
        else
            p.lineTo( x, y );
    }
    g.setColour( sourceColour( s ) );
    g.strokePath( p, juce::PathStrokeType( width ) );
}

}   // namespace

// ============================================================================================================ dock

ModDock::ModDock( AdditivProcessor &proc ) : m_proc( proc )
{
    for ( auto &h : m_history ) h.fill( 0.0f );
    m_toggle.setButtonText( "Inspector" );
    m_toggle.setTooltip( "The sources' inspector: a source's trace, its knobs and where it goes" );
    m_toggle.onClick = [this]
    {
        if ( m_onToggle ) m_onToggle();
    };
    addAndMakeVisible( m_toggle );
    startTimerHz( 30 );
}

void ModDock::setChosen( ModSourceE s )
{
    m_chosen = s;
    repaint();
}

void ModDock::setOpen( bool open )
{
    m_toggle.setToggleState( open, juce::dontSendNotification );
}

float ModDock::traceAt( ModSourceE s, int k ) const
{
    const int i = static_cast<int>( s );
    if ( i <= 0 || i >= additiv::NumModSources ) return 0.0f;
    return m_history[static_cast<size_t>( i )][static_cast<size_t>( ( m_head + k ) % History )];
}

void ModDock::timerCallback()
{
    for ( int s = 1; s < additiv::NumModSources; ++s )
        m_history[static_cast<size_t>( s )][static_cast<size_t>( m_head )] =
            m_proc.engine().sourceValue( static_cast<ModSourceE>( s ) );
    m_head = ( m_head + 1 ) % History;
    if ( isShowing() ) repaint();
}

int ModDock::rowsOf( ModSourceE s ) const
{
    int n = 0;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow r = m_proc.engine().modRow( i );
        if ( r.m_source == s && r.m_target >= 0 ) ++n;
    }
    return n;
}

std::vector<ModSourceE> ModDock::wanted() const
{
    std::vector<ModSourceE> out( std::begin( Classic ), std::end( Classic ) );
    for ( int k = 1; k < additiv::NumModSources; ++k )
    {   // the others the matrix reads, in order
        const auto s = static_cast<ModSourceE>( k );
        if ( classic( s ) ) continue;
        bool read = false;
        for ( int i = 0; i < additiv::ModRows && !read; ++i )
        {
            const ModRow r = m_proc.engine().modRow( i );
            read = ( r.m_source == s || r.m_via == s ) && r.m_target >= 0;
        }
        if ( read ) out.push_back( s );
    }
    if ( m_pinned != ModSourceE::OFF && std::find( out.begin(), out.end(), m_pinned ) == out.end() )
        out.push_back( m_pinned );
    if ( m_chosen != ModSourceE::OFF && std::find( out.begin(), out.end(), m_chosen ) == out.end() )
        out.push_back( m_chosen );
    return out;
}

juce::Rectangle<int> ModDock::chipArea() const
{
    return getLocalBounds().reduced( 0, 3 ).withTrimmedLeft( ToggleW + 8 ).withTrimmedRight( PlusW + 6 );
}

juce::Rectangle<int> ModDock::chipBounds( size_t k, size_t n ) const
{
    const auto a = chipArea();
    const int  w = std::clamp( a.getWidth() / std::max<int>( 1, static_cast<int>( n ) ), ChipMin, ChipMax );
    return { a.getX() + static_cast<int>( k ) * w, a.getY(), w - 4, a.getHeight() };
}

juce::Rectangle<int> ModDock::plusBounds() const
{
    const size_t n = m_shown.size();
    const auto   last = n > 0 ? chipBounds( n - 1, n ) : chipArea().withWidth( 0 );
    return { last.getRight() + 4, last.getY(), PlusW, chipArea().getHeight() };
}

int ModDock::chipAt( juce::Point<int> p ) const
{
    for ( size_t k = 0; k < m_shown.size(); ++k )
        if ( chipBounds( k, m_shown.size() ).contains( p ) ) return static_cast<int>( k );
    return plusBounds().contains( p ) ? -2 : -1;
}

void ModDock::resized()
{
    m_toggle.setBounds( getLocalBounds().removeFromLeft( ToggleW ).reduced( 0, 4 ) );
}

void ModDock::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    // As many chips as fit at their least width.
    std::vector<ModSourceE> want = wanted();
    const size_t            fit = static_cast<size_t>( std::max( 1, chipArea().getWidth() / ChipMin ) );
    if ( want.size() > fit ) want.resize( fit );
    m_shown = want;
    for ( size_t k = 0; k < m_shown.size(); ++k )
    {
        const ModSourceE   s = m_shown[k];
        const auto         r = chipBounds( k, m_shown.size() ).toFloat();
        const juce::Colour c = sourceColour( s );
        const bool         chosen = s == m_chosen, hover = static_cast<int>( k ) == m_hover;
        g.setColour( skin.m_raised.withAlpha( hover ? 1.0f : 0.8f ) );
        g.fillRoundedRectangle( r, 5.0f );
        g.setColour( chosen ? skin.m_accent : skin.m_edge );
        g.drawRoundedRectangle( r.reduced( 0.5f ), 5.0f, chosen ? 1.6f : 1.0f );
        g.setColour( c );
        g.fillRoundedRectangle( r.getX() + 3.0f, r.getY() + 4.0f, 2.5f, r.getHeight() - 8.0f, 1.2f );
        g.setFont( juce::FontOptions( 10.5f, rowsOf( s ) > 0 ? juce::Font::bold : juce::Font::plain ) );
        g.setColour( rowsOf( s ) > 0 ? skin.m_ink : skin.m_dim );
        g.drawFittedText( additiv::ModSourceNames[static_cast<int>( s )],
                          r.reduced( 9.0f, 2.0f ).withHeight( 13.0f ).toNearestInt(), juce::Justification::centredLeft,
                          1, 0.8f );
        drawTrace( g, r.reduced( 9.0f, 3.0f ).withTrimmedTop( 13.0f ), *this, s, 1.1f );
    }
    const auto p = plusBounds().toFloat();
    g.setColour( skin.m_raised.withAlpha( m_hover == -2 ? 1.0f : 0.8f ) );
    g.fillRoundedRectangle( p, 5.0f );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( p.reduced( 0.5f ), 5.0f, 1.0f );
    g.setColour( skin.m_ink );
    g.setFont( juce::FontOptions( 16.0f ) );
    g.drawText( "+", p, juce::Justification::centred );
}

void ModDock::mouseDown( const juce::MouseEvent &e )
{
    m_pressed = chipAt( e.getPosition() );
    m_dragging = false;
}

void ModDock::mouseDrag( const juce::MouseEvent &e )
{
    if ( m_dragging || m_pressed < 0 || static_cast<size_t>( m_pressed ) >= m_shown.size() ||
         e.getDistanceFromDragStart() < 5 )
        return;
    auto *container = juce::DragAndDropContainer::findParentDragContainerFor( this );
    if ( container == nullptr ) return;
    m_dragging = true;
    const auto r = chipBounds( static_cast<size_t>( m_pressed ), m_shown.size() );
    container->startDragging( knobmod::dragOf( m_shown[static_cast<size_t>( m_pressed )] ), this,
                              juce::ScaledImage( createComponentSnapshot( r ) ), false );
}

void ModDock::mouseUp( const juce::MouseEvent &e )
{
    const int at = chipAt( e.getPosition() );
    if ( !m_dragging && at == m_pressed )
    {
        if ( at == -2 )
            showMore();
        else if ( at >= 0 && static_cast<size_t>( at ) < m_shown.size() && m_onChoose )
            m_onChoose( m_shown[static_cast<size_t>( at )] );
    }
    m_pressed = -1;
    m_dragging = false;
}

void ModDock::mouseMove( const juce::MouseEvent &e )
{
    const int at = chipAt( e.getPosition() );
    if ( at != m_hover )
    {
        m_hover = at;
        repaint();
    }
}

void ModDock::mouseExit( const juce::MouseEvent & )
{
    m_hover = -1;
    repaint();
}

juce::String ModDock::getTooltip()
{
    if ( m_hover == -2 ) return "Every source: the one chosen gets a chip here";
    if ( m_hover < 0 || static_cast<size_t>( m_hover ) >= m_shown.size() ) return {};
    const ModSourceE s = m_shown[static_cast<size_t>( m_hover )];
    const int        n = rowsOf( s );
    return juce::String( additiv::ModSourceNames[static_cast<int>( s )] ) + ", now " +
           juce::String( m_proc.engine().sourceValue( s ), 2 ) +
           ( n > 0 ? ", moves " + juce::String( n ) + ( n == 1 ? " knob" : " knobs" )
                   : juce::String( ", moves nothing yet" ) ) +
           ". Drag it onto a knob to modulate it; click: its knobs and where it goes";
}

void ModDock::showMore()
{
    struct Section
    {
        const char *m_name;    ///< its heading
        ModSourceE  m_first;   ///< its first source
        ModSourceE  m_last;    ///< its last
    };
    static constexpr Section Sections[] = { { "Fields and chaos", ModSourceE::LENIA1, ModSourceE::KURA4 },
                                            { "The piece", ModSourceE::BEAT, ModSourceE::RANDOM_NOTE },
                                            { "What it hears", ModSourceE::LOUDNESS, ModSourceE::ROUGHNESS },
                                            { "The reservoir", ModSourceE::RESERVOIR1, ModSourceE::RESERVOIR4 },
                                            { "The night", ModSourceE::NIGHT, ModSourceE::LIGHTNING } };
    juce::PopupMenu          m;
    for ( const Section &sec : Sections )
    {
        m.addSectionHeader( sec.m_name );
        for ( int k = static_cast<int>( sec.m_first ); k <= static_cast<int>( sec.m_last ); ++k )
            m.addItem( additiv::ModSourceNames[k],
                       [this, k]
                       {
                           m_pinned = static_cast<ModSourceE>( k );
                           if ( m_onChoose ) m_onChoose( m_pinned );
                           repaint();
                       } );
    }
    m.showMenuAsync( juce::PopupMenu::Options().withTargetComponent( this ).withTargetScreenArea(
        localAreaToGlobal( plusBounds() ) ) );
}

// ======================================================================================================= inspector

SourceInspector::SourceInspector( AdditivProcessor &proc, const frame::ControlActions &actions, const ModDock &dock )
    : m_proc( proc ), m_actions( actions ), m_dock( dock )
{
    m_hint.setText( "No knob listens to it yet: drag its chip onto a knob, or right-click a knob, Modulate with.",
                    juce::dontSendNotification );
    m_hint.setColour( juce::Label::textColourId, additivui::skin().m_dim );
    m_hint.setFont( juce::FontOptions( 12.0f ) );
    m_hint.setJustificationType( juce::Justification::centredLeft );
    addChildComponent( m_hint );
    m_close.setTooltip( "Shut the inspector" );
    m_close.onClick = [this]
    {
        if ( m_onClose ) m_onClose();
    };
    addChildComponent( m_close );
    m_matrix.setTooltip( "The matrix page: every row" );
    m_matrix.onClick = [this]
    {
        if ( m_onMatrix ) m_onMatrix();
    };
    addAndMakeVisible( m_matrix );
    show( ModSourceE::LFO1 );
    startTimerHz( 30 );
}

void SourceInspector::setOverlay( bool on )
{
    m_overlay = on;
    m_close.setVisible( on );
    resized();
}

std::vector<ParamIdE> SourceInspector::ownParams( ModSourceE s )
{
    switch ( s )
    {
    case ModSourceE::LFO1: return { ParamIdE::LFO1_SHAPE, ParamIdE::LFO1_RATE };
    case ModSourceE::LFO2: return { ParamIdE::LFO2_SHAPE, ParamIdE::LFO2_RATE };
    case ModSourceE::LFO3: return { ParamIdE::LFO3_SHAPE, ParamIdE::LFO3_RATE };
    case ModSourceE::LFO4: return { ParamIdE::LFO4_SHAPE, ParamIdE::LFO4_RATE };
    case ModSourceE::ENV1: return { ParamIdE::ENV1_SHAPE, ParamIdE::ENV1_RATE };
    case ModSourceE::ENV2: return { ParamIdE::ENV2_SHAPE, ParamIdE::ENV2_RATE };
    case ModSourceE::MACRO1: return { ParamIdE::MACRO1 };
    case ModSourceE::MACRO2: return { ParamIdE::MACRO2 };
    case ModSourceE::MACRO3: return { ParamIdE::MACRO3 };
    case ModSourceE::MACRO4: return { ParamIdE::MACRO4 };
    case ModSourceE::LENIA1:
    case ModSourceE::LENIA2:
    case ModSourceE::LENIA3:
    case ModSourceE::LENIA4: return { ParamIdE::LENIA_RATE, ParamIdE::LENIA_GROWTH };
    case ModSourceE::LORENZ_X:
    case ModSourceE::LORENZ_Y:
    case ModSourceE::LORENZ_Z:
    case ModSourceE::ROSSLER_X:
    case ModSourceE::ROSSLER_Y:
    case ModSourceE::ROSSLER_Z: return { ParamIdE::CHAOS_PERIOD };
    case ModSourceE::KURA1:
    case ModSourceE::KURA2:
    case ModSourceE::KURA3:
    case ModSourceE::KURA4: return { ParamIdE::COHERENCE, ParamIdE::COHERENCE_RATE };
    case ModSourceE::RESERVOIR1:
    case ModSourceE::RESERVOIR2:
    case ModSourceE::RESERVOIR3:
    case ModSourceE::RESERVOIR4:
        return { ParamIdE::RESERVOIR_CHAOS, ParamIdE::RESERVOIR_MEMORY, ParamIdE::RESERVOIR_INPUT };
    case ModSourceE::NIGHT: return { ParamIdE::NIGHT_ARC };
    case ModSourceE::RISSET:
        return { ParamIdE::RISSET, ParamIdE::RISSET_RATE, ParamIdE::RISSET_TIME, ParamIdE::RISSET_WAY };
    case ModSourceE::LIGHTNING: return { ParamIdE::THUNDER, ParamIdE::THUNDER_DISTANCE };
    default: return {};
    }
}

ModSourceE SourceInspector::sourceOf( int id )
{
    for ( int k = 1; k < additiv::NumModSources; ++k )
        for ( ParamIdE p : ownParams( static_cast<ModSourceE>( k ) ) )
            if ( static_cast<int>( p ) == id ) return static_cast<ModSourceE>( k );
    return ModSourceE::OFF;
}

void SourceInspector::show( ModSourceE s )
{
    if ( s == ModSourceE::OFF ) return;
    m_source = s;
    rebuildControls();
    rebuildRows();
    repaint();
}

void SourceInspector::rebuildControls()
{
    m_sliders.clear();
    m_combos.clear();
    m_controls.clear();
    m_labels.clear();
    const frame::Skin &skin = additivui::skin();
    for ( ParamIdE pid : ownParams( m_source ) )
    {
        const int                   id = static_cast<int>( pid );
        juce::RangedAudioParameter *param = m_proc.parameter( id );
        if ( param == nullptr ) continue;
        const additiv::ParamDesc &d = additiv::paramDesc( pid );
        auto                      label = std::make_unique<juce::Label>( juce::String(), d.m_name );
        label->setJustificationType( juce::Justification::centred );
        label->setColour( juce::Label::textColourId, skin.m_dim );
        label->setFont( juce::FontOptions( 11.5f ) );
        label->setMinimumHorizontalScale( 0.7f );
        addAndMakeVisible( *label );
        m_labels.push_back( std::move( label ) );
        if ( d.m_kind == additiv::ParamKindE::CHOICE )
        {
            auto c = std::make_unique<frame::Choice>( &m_actions, id );
            for ( int i = 0; i < d.m_numChoices; ++i ) c->addItem( d.m_choices[i], i + 1 );
            c->setColour( juce::ComboBox::arrowColourId, sourceColour( m_source ) );
            m_combos.push_back( std::make_unique<juce::ComboBoxParameterAttachment>( *param, *c ) );
            addAndMakeVisible( *c );
            c->sendLookAndFeelChange();   // made before it had the frame's look
            m_controls.push_back( std::move( c ) );
        }
        else
        {
            auto k = std::make_unique<frame::Knob>( juce::Slider::RotaryHorizontalVerticalDrag,
                                                    juce::Slider::TextBoxBelow, &m_actions, id );
            k->setTextBoxStyle( juce::Slider::TextBoxBelow, false, 70, 15 );
            k->setColour( juce::Slider::rotarySliderFillColourId, sourceColour( m_source ) );
            k->setTextValueSuffix( d.m_unit[0] != 0 ? juce::String( " " ) + d.m_unit : juce::String() );
            k->setDoubleClickReturnValue( true, d.m_def );
            m_sliders.push_back( std::make_unique<juce::SliderParameterAttachment>( *param, *k ) );
            addAndMakeVisible( *k );
            k->sendLookAndFeelChange();   // its value box made under the frame's look
            m_controls.push_back( std::move( k ) );
        }
    }
    resized();
}

juce::String SourceInspector::rowsKey() const
{
    juce::String k;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow r = m_proc.engine().modRow( i );
        if ( r.m_source == m_source && r.m_target >= 0 )
            k << i << ":" << r.m_target << ":" << static_cast<int>( r.m_via ) << ";";
    }
    return k;
}

void SourceInspector::rebuildRows()
{
    m_rows.clear();
    m_key = rowsKey();
    const juce::Colour c = sourceColour( m_source );
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow r = m_proc.engine().modRow( i );
        if ( r.m_source != m_source || r.m_target < 0 ) continue;
        auto ui = std::make_unique<RowUi>();
        ui->m_row = i;
        juce::String name( additiv::paramDesc( static_cast<ParamIdE>( r.m_target ) ).m_name );
        if ( r.m_via != ModSourceE::OFF ) name << " (x " << additiv::ModSourceNames[static_cast<int>( r.m_via )] << ")";
        ui->m_target.setButtonText( name );
        ui->m_target.setTooltip( "Row " + juce::String( i + 1 ) + ": click to see the knob it moves" );
        ui->m_target.onClick = [this, t = r.m_target]
        {
            if ( m_onReveal ) m_onReveal( t );
        };
        ui->m_depth.setSliderStyle( juce::Slider::LinearHorizontal );
        ui->m_depth.setTextBoxStyle( juce::Slider::TextBoxRight, false, 44, 18 );
        ui->m_depth.setRange( -1.0, 1.0, 0.001 );
        ui->m_depth.setValue( r.m_depth, juce::dontSendNotification );
        ui->m_depth.setDoubleClickReturnValue( true, 0.0 );
        ui->m_depth.setColour( juce::Slider::trackColourId, c );
        ui->m_depth.setTooltip( "How far it moves the knob, of the knob's whole range (negative: the other way)" );
        ui->m_depth.onValueChange = [this, i, s = &ui->m_depth]
        {
            ModRow now = m_proc.engine().modRow( i );
            now.m_depth = static_cast<float>( s->getValue() );
            m_proc.engine().setModRow( i, now );
        };
        ui->m_remove.setTooltip( "Removes this row from the matrix" );
        ui->m_remove.onClick = [this, i]
        {
            m_proc.engine().setModRow( i, ModRow{} );
            juce::MessageManager::callAsync(
                [safe = juce::Component::SafePointer<SourceInspector>( this )]
                {
                    if ( safe != nullptr ) safe->rebuildRows();
                } );
        };
        addAndMakeVisible( ui->m_target );
        addAndMakeVisible( ui->m_depth );
        addAndMakeVisible( ui->m_remove );
        ui->m_depth.sendLookAndFeelChange();
        m_rows.push_back( std::move( ui ) );
    }
    m_hint.setVisible( m_rows.empty() );
    resized();
}

juce::Rectangle<int> SourceInspector::traceArea() const
{
    return { 14, 30, std::min( 260, getWidth() / 4 ), getHeight() - 40 };
}

void SourceInspector::resized()
{
    const auto trace = traceArea();
    m_close.setBounds( getWidth() - 30, 4, 24, 20 );
    m_matrix.setBounds( ( m_overlay ? m_close.getX() : getWidth() ) - 92, 4, 84, 20 );
    // The source's own knobs and choosers, after the trace.
    int       x = trace.getRight() + 16;
    const int top = 26;
    for ( size_t i = 0; i < m_controls.size(); ++i )
    {
        const bool choice = dynamic_cast<juce::ComboBox *>( m_controls[i].get() ) != nullptr;
        const int  w = choice ? 124 : 76;
        m_labels[i]->setBounds( x, top, w, 14 );
        if ( choice )
            m_controls[i]->setBounds( x + 4, top + 30, w - 8, 24 );
        else
            m_controls[i]->setBounds( x, top + 14, w, getHeight() - top - 18 );
        x += w;
    }
    // Its rows, in columns of three.
    x += m_controls.empty() ? 0 : 16;
    const int colW = 330, rowH = 26;
    m_hint.setBounds( x, top + 10, std::max( 0, getWidth() - x - 12 ), 40 );
    for ( size_t i = 0; i < m_rows.size(); ++i )
    {
        const int  cx = x + static_cast<int>( i / 3 ) * ( colW + 10 ), cy = top + 4 + static_cast<int>( i % 3 ) * rowH;
        RowUi     &r = *m_rows[i];
        const bool fits = cx + colW <= getWidth() - 8;
        r.m_target.setVisible( fits );
        r.m_depth.setVisible( fits );
        r.m_remove.setVisible( fits );
        r.m_target.setBounds( cx, cy, 140, rowH - 4 );
        r.m_depth.setBounds( cx + 144, cy, colW - 144 - 28, rowH - 4 );
        r.m_remove.setBounds( cx + colW - 24, cy, 22, rowH - 4 );
    }
}

void SourceInspector::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    const auto         b = getLocalBounds().toFloat();
    g.setColour( m_overlay ? skin.m_group : skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( b, skin.m_radius );
    g.setColour( m_overlay ? skin.m_accent.withAlpha( 0.6f ) : skin.m_edge );
    g.drawRoundedRectangle( b.reduced( 0.5f ), skin.m_radius, 1.0f );
    const juce::Colour c = sourceColour( m_source );
    g.setColour( c );
    g.fillRoundedRectangle( 8.0f, 9.0f, 3.0f, 10.0f, 1.5f );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    juce::String title( additiv::ModSourceNames[static_cast<int>( m_source )] );
    title << "   " << juce::String( m_proc.engine().sourceValue( m_source ), 2 );
    const int n = static_cast<int>( m_rows.size() );
    title << "   "
          << ( n == 0 ? juce::String( "moves nothing" )
                      : "moves " + juce::String( n ) + ( n == 1 ? " knob" : " knobs" ) );
    g.drawText( title, 17, 4, getWidth() - 200, 20, juce::Justification::centredLeft );
    const auto t = traceArea().toFloat();
    g.setColour( skin.m_raised.withAlpha( 0.55f ) );
    g.fillRoundedRectangle( t, 4.0f );
    drawTrace( g, t.reduced( 6.0f, 6.0f ), m_dock, m_source, 1.6f );
}

void SourceInspector::timerCallback()
{
    if ( !isShowing() ) return;
    if ( rowsKey() != m_key ) rebuildRows();
    for ( auto &r : m_rows )   // depths changed elsewhere (the matrix page, a knob's menu)
        if ( !r->m_depth.isMouseButtonDown() )
            r->m_depth.setValue( m_proc.engine().modRow( r->m_row ).m_depth, juce::dontSendNotification );
    repaint( getLocalBounds().withHeight( 26 ) );
    repaint( traceArea() );
}
