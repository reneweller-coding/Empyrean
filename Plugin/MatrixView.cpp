/**
 * @file MatrixView.cpp
 * @brief The modulation matrix on the panel (MatrixView.h).
 * @author Rene Weller
 */
#include "MatrixView.h"

#include "Theme.h"

#include "additiv/Modulation.h"

using additiv::ModRow;
using additiv::ModSourceE;
using additiv::ParamIdE;

MatrixView::MatrixView( AdditivProcessor &proc ) : m_proc( proc )
{
    for ( int i = 0; i < additiv::NumParams; ++i )
        if ( additiv::isModTarget( static_cast<ParamIdE>( i ) ) ) m_targets.push_back( i );
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        auto row = std::make_unique<Row>();
        for ( int s = 0; s < additiv::NumModSources; ++s )
        {
            if ( s == static_cast<int>( ModSourceE::LENIA1 ) )
            {
                row->m_source.addSectionHeading( "Fields and chaos" );
                row->m_via.addSectionHeading( "Fields and chaos" );
            }
            if ( s == static_cast<int>( ModSourceE::BEAT ) )
            {
                row->m_source.addSectionHeading( "The piece" );
                row->m_via.addSectionHeading( "The piece" );
            }
            if ( s == static_cast<int>( ModSourceE::LOUDNESS ) )
            {
                row->m_source.addSectionHeading( "What it hears" );
                row->m_via.addSectionHeading( "What it hears" );
            }
            if ( s == static_cast<int>( ModSourceE::RESERVOIR1 ) )
            {
                row->m_source.addSectionHeading( "Reservoir" );
                row->m_via.addSectionHeading( "Reservoir" );
            }
            if ( s == static_cast<int>( ModSourceE::NIGHT ) )
            {
                row->m_source.addSectionHeading( "Night and space" );
                row->m_via.addSectionHeading( "Night and space" );
            }
            row->m_source.addItem( additiv::ModSourceNames[s], s + 1 );
            row->m_via.addItem( s == 0 ? juce::String( "-" ) : juce::String( additiv::ModSourceNames[s] ), s + 1 );
        }
        row->m_target.addItem( "-", 1 );
        juce::String section;
        for ( size_t k = 0; k < m_targets.size(); ++k )
        {
            const additiv::ParamDesc &d = additiv::paramDesc( static_cast<ParamIdE>( m_targets[k] ) );
            if ( section != d.m_section )
            {
                section = d.m_section;
                row->m_target.addSectionHeading( section );
            }
            row->m_target.addItem( d.m_name, static_cast<int>( k ) + 2 );
        }
        row->m_depth.setSliderStyle( juce::Slider::LinearHorizontal );
        row->m_depth.setTextBoxStyle( juce::Slider::TextBoxRight, false, 52, 20 );
        row->m_depth.setRange( -1.0, 1.0, 0.01 );
        row->m_depth.setDoubleClickReturnValue( true, 0.0 );
        row->m_depth.setColour( juce::Slider::trackColourId, additivui::skin().family( frame::FamilyE::MOTION ) );
        row->m_source.onChange = [this, i] { push( i ); };
        row->m_target.onChange = [this, i] { push( i ); };
        row->m_via.onChange = [this, i] { push( i ); };
        row->m_via.setTooltip( "Via: a second source whose amount (0 .. 1) scales the depth" );
        row->m_depth.onValueChange = [this, i] { push( i ); };
        addAndMakeVisible( row->m_source );
        addAndMakeVisible( row->m_target );
        addAndMakeVisible( row->m_via );
        addAndMakeVisible( row->m_depth );
        m_rows.push_back( std::move( row ) );
    }
    timerCallback();
    startTimerHz( 4 );
}

void MatrixView::push( int i )
{
    if ( m_updating ) return;
    const Row &r = *m_rows[static_cast<size_t>( i )];
    ModRow     m;
    m.m_source = static_cast<ModSourceE>( std::max( r.m_source.getSelectedId() - 1, 0 ) );
    const int t = r.m_target.getSelectedId() - 2;
    m.m_target = t >= 0 && t < static_cast<int>( m_targets.size() ) ? m_targets[static_cast<size_t>( t )] : -1;
    m.m_depth = static_cast<float>( r.m_depth.getValue() );
    m.m_via = static_cast<ModSourceE>( std::max( r.m_via.getSelectedId() - 1, 0 ) );
    m_proc.engine().setModRow( i, m );
}

void MatrixView::timerCallback()
{
    m_updating = true;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow m = m_proc.engine().modRow( i );
        Row         &r = *m_rows[static_cast<size_t>( i )];
        r.m_source.setSelectedId( static_cast<int>( m.m_source ) + 1, juce::dontSendNotification );
        int item = 1;
        for ( size_t k = 0; k < m_targets.size(); ++k )
            if ( m_targets[k] == m.m_target ) item = static_cast<int>( k ) + 2;
        r.m_target.setSelectedId( item, juce::dontSendNotification );
        r.m_via.setSelectedId( static_cast<int>( m.m_via ) + 1, juce::dontSendNotification );
        if ( !r.m_depth.isMouseButtonDown() ) r.m_depth.setValue( m.m_depth, juce::dontSendNotification );
    }
    m_updating = false;
}

void MatrixView::resized()
{
    const int cols = columns(), per = additiv::ModRows / cols, pitch = cols == 1 ? 28 : 30;
    const int colW = ( getWidth() - 20 - ( cols - 1 ) * 10 ) / cols;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        Row      &r = *m_rows[static_cast<size_t>( i )];
        const int x = 10 + ( i / per ) * ( colW + 10 ) + 26, y = 28 + ( i % per ) * pitch;
        const int w = colW - 26;
        r.m_source.setBounds( x, y, w * 22 / 100, 24 );
        r.m_target.setBounds( x + w * 23 / 100, y, w * 31 / 100, 24 );
        r.m_via.setBounds( x + w * 55 / 100, y, w * 17 / 100, 24 );
        r.m_depth.setBounds( x + w * 73 / 100, y, w * 27 / 100, 24 );
    }
}

void MatrixView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( getLocalBounds().toFloat(), skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( getLocalBounds().toFloat().reduced( 0.5f ), skin.m_radius, 1.0f );
    g.setColour( skin.family( frame::FamilyE::MOTION ) );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.drawFittedText( "Matrix: source -> target x depth (of the target's range), scaled by a via source", 18, 4,
                      getWidth() - 30, 20, juce::Justification::centredLeft, 1, 0.85f );
    g.setColour( skin.m_dim );
    g.setFont( juce::FontOptions( 11.0f ) );
    const int cols = columns(), per = additiv::ModRows / cols, pitch = cols == 1 ? 28 : 30;
    const int colW = ( getWidth() - 20 - ( cols - 1 ) * 10 ) / cols;
    for ( int i = 0; i < additiv::ModRows; ++i )
        g.drawText( juce::String( i + 1 ), 10 + ( i / per ) * ( colW + 10 ), 28 + ( i % per ) * pitch, 22, 24,
                    juce::Justification::centredRight );
}

juce::String MatrixView::describe() const
{
    juce::String s( "Matrix rows:\n" );
    int          n = 0;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow m = m_proc.engine().modRow( i );
        if ( m.m_source == ModSourceE::OFF || m.m_target < 0 ) continue;
        s << "    " << additiv::ModSourceNames[static_cast<int>( m.m_source )] << " -> "
          << additiv::paramDesc( static_cast<ParamIdE>( m.m_target ) ).m_name << " x " << juce::String( m.m_depth, 2 );
        if ( m.m_via != ModSourceE::OFF ) s << " via " << additiv::ModSourceNames[static_cast<int>( m.m_via )];
        s << "\n";
        ++n;
    }
    if ( n == 0 ) s << "    (none)\n";
    s << "\nA source moves its target by depth times its value, in the target's own range: LFOs and the random curve swing "
         "both ways, envelopes follow their shapes, macros, the mod wheel, pressure and velocity go from 0 to 1. The envelopes "
         "start over when a note arrives into silence. A via source scales the depth by its amount (0 .. 1). Lenia, Lorenz, "
         "Roessler and the Kuramoto bank are Noctuary's: a living field, two strange attractors and four oscillators falling "
         "into step; the Beat runs at the rate the lowest interval beats, Cascade with the conductor's excitement, Root Age "
         "with how long the root has stood.\n";
    return s;
}
