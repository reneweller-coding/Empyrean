/**
 * @file KnobMod.cpp
 * @brief Modulation at the knob (KnobMod.h).
 * @author Rene Weller
 */
#include "KnobMod.h"

#include "Theme.h"

#include <cmath>

namespace knobmod
{

using additiv::ModRow;
using additiv::ModSourceE;

juce::Colour sourceColour( ModSourceE s )
{
    const frame::Skin &skin = additivui::skin();
    if ( s >= ModSourceE::NIGHT ) return skin.family( frame::FamilyE::FILTER );
    if ( s >= ModSourceE::RESERVOIR1 ) return additivui::colour::Violet;
    if ( s >= ModSourceE::LOUDNESS ) return skin.family( frame::FamilyE::SPACE );
    if ( s >= ModSourceE::BEAT ) return skin.family( frame::FamilyE::ENVELOPE );
    if ( s >= ModSourceE::LENIA1 ) return skin.family( frame::FamilyE::MOTION );
    return skin.family( frame::FamilyE::SOURCE );
}

namespace
{

/** @brief A section of the sources, as the matrix's menu has them. */
struct Section
{
    const char *m_name;    ///< its heading
    ModSourceE  m_first;   ///< its first source
    ModSourceE  m_last;    ///< its last
};

/** @brief The sources by kind. */
constexpr Section Sections[] = {
    { "LFOs and envelopes", ModSourceE::LFO1, ModSourceE::ENV2 },
    { "Macros and the hands", ModSourceE::MACRO1, ModSourceE::RANDOM },
    { "Fields and chaos", ModSourceE::LENIA1, ModSourceE::KURA4 },
    { "The piece", ModSourceE::BEAT, ModSourceE::RANDOM_NOTE },
    { "What it hears", ModSourceE::LOUDNESS, ModSourceE::ROUGHNESS },
    { "The reservoir", ModSourceE::RESERVOIR1, ModSourceE::RESERVOIR4 },
    { "The night", ModSourceE::NIGHT, ModSourceE::LIGHTNING },
};

/**
 * @brief A row's name: its source, and its via source when it has one.
 * @param[in] r  the row
 * @return the name
 */
juce::String rowName( const ModRow &r )
{
    juce::String s( additiv::ModSourceNames[static_cast<int>( r.m_source )] );
    if ( r.m_via != ModSourceE::OFF ) s << " x " << additiv::ModSourceNames[static_cast<int>( r.m_via )];
    return s;
}

/** @brief One row on the knob in its menu: its source, its depth to drag, a button that removes it. */
class RowItem final : public juce::PopupMenu::CustomComponent
{
public:
    /**
     * @brief Row @p row of @p proc's matrix.
     * @param[in,out] proc  the processor
     * @param[in]     row   0 .. ModRows-1
     */
    RowItem( AdditivProcessor &proc, int row ) : juce::PopupMenu::CustomComponent( false ), m_proc( proc ), m_row( row )
    {
        const ModRow r = m_proc.engine().modRow( m_row );
        m_name.setText( rowName( r ), juce::dontSendNotification );
        m_name.setColour( juce::Label::textColourId, sourceColour( r.m_source ) );
        m_name.setFont( juce::FontOptions( 13.0f ) );
        m_name.setInterceptsMouseClicks( false, false );
        m_depth.setSliderStyle( juce::Slider::LinearHorizontal );
        m_depth.setTextBoxStyle( juce::Slider::TextBoxRight, false, 52, 18 );
        m_depth.setRange( -1.0, 1.0, 0.001 );
        m_depth.setValue( r.m_depth, juce::dontSendNotification );
        m_depth.setDoubleClickReturnValue( true, 0.0 );
        m_depth.setColour( juce::Slider::trackColourId, sourceColour( r.m_source ) );
        m_depth.setTooltip( "How far it moves the knob, of the knob's whole range (negative: the other way)" );
        m_depth.onValueChange = [this]
        {
            ModRow now = m_proc.engine().modRow( m_row );
            now.m_depth = static_cast<float>( m_depth.getValue() );
            m_proc.engine().setModRow( m_row, now );
        };
        m_remove.setTooltip( "Removes this row from the matrix" );
        m_remove.onClick = [this]
        {
            m_proc.engine().setModRow( m_row, ModRow{} );
            triggerMenuItem();   // the menu closes: the row is gone
        };
        for ( juce::Component *c :
              { static_cast<juce::Component *>( &m_name ), static_cast<juce::Component *>( &m_depth ),
                static_cast<juce::Component *>( &m_remove ) } )
            addAndMakeVisible( c );
    }
    /**
     * @brief A row's size in the menu.
     * @param[in,out] w  width
     * @param[in,out] h  height
     */
    void getIdealSize( int &w, int &h ) override
    {
        w = 340;
        h = 30;
    }
    /** @brief Name left, depth, the button right. */
    void resized() override
    {
        auto r = getLocalBounds().reduced( 6, 3 );
        m_remove.setBounds( r.removeFromRight( 24 ) );
        m_name.setBounds( r.removeFromLeft( 120 ) );
        m_depth.setBounds( r.reduced( 4, 0 ) );
    }

private:
    AdditivProcessor &m_proc;            ///< whose matrix
    int               m_row;             ///< which row
    juce::Label       m_name;            ///< its source
    juce::Slider      m_depth;           ///< its depth
    juce::TextButton  m_remove{ "x" };   ///< removes it
};

}   // namespace

bool canTarget( int id )
{
    return id >= 0 && id < additiv::NumParams && additiv::isModTarget( static_cast<additiv::ParamIdE>( id ) );
}

int addRow( AdditivProcessor &proc, ModSourceE source, int id, float depth )
{
    if ( !canTarget( id ) || source == ModSourceE::OFF ) return -1;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {   // already there: that row, not a second one
        const ModRow r = proc.engine().modRow( i );
        if ( r.m_source == source && r.m_target == id && r.m_via == ModSourceE::OFF ) return i;
    }
    for ( int i = 0; i < additiv::ModRows; ++i )
        if ( proc.engine().modRow( i ).m_source == ModSourceE::OFF )
        {
            ModRow r;
            r.m_source = source;
            r.m_target = id;
            r.m_depth = depth;
            proc.engine().setModRow( i, r );
            return i;
        }
    return -1;
}

juce::var dragOf( ModSourceE s )
{
    return juce::var( "empyrean-source:" + juce::String( static_cast<int>( s ) ) );
}

ModSourceE sourceOfDrag( const juce::var &description )
{
    const juce::String s = description.toString();
    if ( !s.startsWith( "empyrean-source:" ) ) return ModSourceE::OFF;
    const int k = s.fromFirstOccurrenceOf( ":", false, false ).getIntValue();
    return k > 0 && k < additiv::NumModSources ? static_cast<ModSourceE>( k ) : ModSourceE::OFF;
}

void addMenuItems( AdditivProcessor &proc, juce::PopupMenu &m, int id )
{
    if ( !canTarget( id ) ) return;
    m.addSeparator();
    m.addSectionHeader( "Modulation" );
    int free = -1;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow r = proc.engine().modRow( i );
        if ( r.m_source == ModSourceE::OFF )
        {
            if ( free < 0 ) free = i;
            continue;
        }
        if ( r.m_target == id )
            m.addCustomItem( 1000 + i, std::make_unique<RowItem>( proc, i ), nullptr, rowName( r ) );
    }
    juce::PopupMenu with;
    for ( const Section &s : Sections )
    {
        with.addSectionHeader( s.m_name );
        for ( int k = static_cast<int>( s.m_first ); k <= static_cast<int>( s.m_last ); ++k )
        {
            const auto src = static_cast<ModSourceE>( k );
            with.addItem( additiv::ModSourceNames[k], free >= 0, false, [&proc, id, src] { addRow( proc, src, id ); } );
        }
    }
    m.addSubMenu( free >= 0 ? juce::String( "Modulate with" ) : juce::String( "Modulate with (all 16 rows taken)" ),
                  with, free >= 0 );
}

bool paintArcs( juce::Graphics &g, AdditivProcessor &proc, juce::Slider &knob, int id )
{
    ModRow rows[additiv::ModRows];
    int    n = 0;
    for ( int i = 0; i < additiv::ModRows; ++i )
    {
        const ModRow r = proc.engine().modRow( i );
        if ( r.m_source != ModSourceE::OFF && r.m_target == id ) rows[n++] = r;
    }
    if ( n == 0 ) return false;
    // The knob's own geometry (Frame's drawRotarySlider): its ring, and outside it the live ring.
    const auto area =
        knob.getLookAndFeel().getSliderLayout( knob ).sliderBounds.toFloat() + knob.getPosition().toFloat();
    const float size = std::min( area.getWidth(), area.getHeight() );
    const auto  c = area.getCentre();
    const float ring = size * 0.5f - 3.0f;
    const auto  rp = knob.getRotaryParameters();
    const float a0 = rp.startAngleRadians, a1 = rp.endAngleRadians;
    const float pos = static_cast<float>( knob.valueToProportionOfLength( knob.getValue() ) );
    const float step = n > 3 ? 1.8f : 2.6f;
    for ( int k = 0; k < n; ++k )
    {
        const ModRow &r = rows[k];
        const float   d = r.m_depth;
        const bool    uni = additiv::isUnipolarSource( r.m_source );
        const float   lo = juce::jlimit( 0.0f, 1.0f, uni ? std::min( pos, pos + d ) : pos - std::abs( d ) );
        const float   hi = juce::jlimit( 0.0f, 1.0f, uni ? std::max( pos, pos + d ) : pos + std::abs( d ) );
        const float   via =
            r.m_via != ModSourceE::OFF ? additiv::viaAmount( r.m_via, proc.engine().sourceValue( r.m_via ) ) : 1.0f;
        const float        now = juce::jlimit( 0.0f, 1.0f, pos + d * proc.engine().sourceValue( r.m_source ) * via );
        const float        rad = ring + 4.5f + step * static_cast<float>( k );
        const juce::Colour col = sourceColour( r.m_source );
        juce::Path         span;
        span.addCentredArc( c.x, c.y, rad, rad, 0.0f, a0 + lo * ( a1 - a0 ),
                            a0 + std::max( hi, lo + 0.002f ) * ( a1 - a0 ), true );
        g.setColour( col.withAlpha( 0.45f ) );
        g.strokePath( span, juce::PathStrokeType( 1.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded ) );
        const float an = a0 + now * ( a1 - a0 );
        g.setColour( col );
        g.fillEllipse( c.x + std::sin( an ) * rad - 2.0f, c.y - std::cos( an ) * rad - 2.0f, 4.0f, 4.0f );
    }
    return true;
}

}   // namespace knobmod
