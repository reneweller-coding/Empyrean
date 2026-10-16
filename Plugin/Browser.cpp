/**
 * @file Browser.cpp
 * @brief The preset browser (Browser.h).
 * @author Rene Weller
 */
#include "Browser.h"

#include "Theme.h"

#include "additiv/Find.h"
#include "additiv/Presets.h"

#include <cmath>

namespace
{

/** @brief The bank's sections, as the menu sections them. */
const char *const Sections[10] = { "Sounds",          "Living Systems",    "Sleep Concerts",
                                   "Visions",         "Swarms & Currents", "Halls & Orbits",
                                   "Harmonic Lives",  "Dreams & Phantoms", "Weather & Growth",
                                   "Rooms & Topology" };
/** @brief The traits' ends. */
const char *const TraitLow[4] = { "darker", "stiller", "narrower", "purer" };
const char *const TraitHigh[4] = { "brighter", "more moving", "wider", "airier" };   ///< and their other ends

}   // namespace

/** @brief A row of the list (see PresetBrowser). */
class PresetBrowser::Row final : public juce::Component
{
public:
    /**
     * @brief A row of @p owner.
     * @param[in,out] owner  the browser
     */
    explicit Row( PresetBrowser &owner ) : m_owner( owner ) {}
    /**
     * @brief Shows entry @p e at @p row.
     * @param[in] row       the list's row
     * @param[in] e         the entry, -1 none
     * @param[in] selected  chosen
     */
    void show( int row, int e, bool selected )
    {
        m_row = row;
        m_entry = e;
        m_selected = selected;
        repaint();
    }
    /**
     * @brief The play mark, the star, the name, the group, the traits.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override
    {
        if ( m_entry < 0 ) return;
        const frame::Skin &skin = additivui::skin();
        const Entry       &e = m_owner.m_all[static_cast<size_t>( m_entry )];
        if ( m_selected )
        {
            g.setColour( skin.m_accent.withAlpha( 0.16f ) );
            g.fillRoundedRectangle( getLocalBounds().toFloat().reduced( 1.0f ), skin.m_radius );
        }
        const bool listening = m_listening;
        // The play mark: a triangle, lit while held.
        juce::Path tri;
        tri.addTriangle( 10.0f, 8.0f, 10.0f, getHeight() - 8.0f, 22.0f, getHeight() * 0.5f );
        g.setColour( listening ? skin.m_accent : skin.m_dim );
        g.fillPath( tri );
        // The star.
        const bool fav = m_owner.m_proc.favorite( e.m_name );
        juce::Path star;
        star.addStar( { 40.0f, getHeight() * 0.5f }, 5, 3.0f, 7.0f );
        g.setColour( fav ? additivui::colour::Root : skin.m_faint );
        if ( fav )
            g.fillPath( star );
        else
            g.strokePath( star, juce::PathStrokeType( 1.2f ) );
        g.setColour( e.m_user >= 0 ? additivui::colour::Root : skin.m_ink );
        g.setFont( juce::FontOptions( 14.0f ) );
        g.drawText( e.m_name, 56, 0, getWidth() / 2 - 56, getHeight(), juce::Justification::centredLeft );
        g.setColour( skin.m_dim );
        g.setFont( juce::FontOptions( 12.0f ) );
        g.drawText( e.m_group + ( e.m_hands ? "   (for the hands)" : "" ), getWidth() / 2, 0, getWidth() / 2 - 150,
                    getHeight(), juce::Justification::centredLeft );
        // The traits as four small bars about their middles.
        if ( e.m_measured )
            for ( int k = 0; k < 4; ++k )
            {
                const float x = static_cast<float>( getWidth() - 140 + k * 34 ), mid = getHeight() * 0.5f;
                const float v = juce::jlimit( -2.5f, 2.5f, e.m_traits[k] ) / 2.5f * ( getHeight() * 0.38f );
                g.setColour( skin.m_faint );
                g.drawHorizontalLine( static_cast<int>( mid ), x, x + 26.0f );
                g.setColour( skin.family( frame::FamilyE::SOURCE ).withAlpha( 0.8f ) );
                g.fillRect( juce::Rectangle<float>( x + 8.0f, std::min( mid, mid - v ), 10.0f,
                                                    std::max( 1.0f, std::abs( v ) ) ) );
            }
    }
    /**
     * @brief Listens (the play mark), stars, or chooses the row.
     * @param[in] e  the press
     */
    void mouseDown( const juce::MouseEvent &e ) override
    {
        if ( m_entry < 0 ) return;
        m_owner.m_list.selectRow( m_row );
        const Entry &en = m_owner.m_all[static_cast<size_t>( m_entry )];
        if ( e.x < 30 )
        {
            m_listening = true;
            if ( en.m_factory >= 0 )
                m_owner.m_proc.previewPreset( en.m_factory );
            else
                m_owner.m_proc.previewUserPreset( en.m_user );
            repaint();
        }
        else if ( e.x < 52 )
        {
            m_owner.m_proc.setFavorite( en.m_name, !m_owner.m_proc.favorite( en.m_name ) );
            repaint();
            if ( m_owner.m_categories[static_cast<size_t>( m_owner.m_category )].m_kind == 1 ) m_owner.filter();
        }
    }
    /**
     * @brief Back to the sound before the listening.
     */
    void mouseUp( const juce::MouseEvent & ) override
    {
        if ( !m_listening ) return;
        m_listening = false;
        m_owner.m_proc.endPreview();
        repaint();
    }
    /**
     * @brief Takes it.
     * @param[in] e  the double click
     */
    void mouseDoubleClick( const juce::MouseEvent &e ) override
    {
        if ( e.x >= 52 ) m_owner.choose( m_row );
    }

private:
    PresetBrowser &m_owner;               ///< whose
    int            m_row = -1;            ///< the list's row
    int            m_entry = -1;          ///< the entry shown
    bool           m_selected = false;    ///< chosen
    bool           m_listening = false;   ///< its play mark is held
};

/** @brief A range over a trait (see PresetBrowser): -3 .. 3 spreads, its two ends dragged, a double click opens it wide. */
class PresetBrowser::RangeBar final : public juce::Component, public juce::SettableTooltipClient
{
public:
    std::function<void()> m_onChange;     ///< called while an end moves
    float                 m_lo = -3.0f;   ///< the range's lower end
    float                 m_hi = 3.0f;    ///< and its upper end
    /** @brief Whether it lets everything through. @return true when wide open */
    bool open() const { return m_lo <= -3.0f && m_hi >= 3.0f; }
    /**
     * @brief The track, the range lit, the two ends.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override
    {
        const frame::Skin &skin = additivui::skin();
        const auto         r = getLocalBounds().toFloat().reduced( 6.0f, 0.0f );
        const float        y = r.getCentreY();
        g.setColour( skin.m_faint );
        g.fillRoundedRectangle( r.getX(), y - 2.0f, r.getWidth(), 4.0f, 2.0f );
        const float x0 = xOf( m_lo ), x1 = xOf( m_hi );
        g.setColour( open() ? skin.m_dim : skin.m_accent );
        g.fillRoundedRectangle( x0, y - 2.0f, x1 - x0, 4.0f, 2.0f );
        for ( float x : { x0, x1 } ) g.fillEllipse( x - 5.0f, y - 5.0f, 10.0f, 10.0f );
    }
    /**
     * @brief Takes the nearer end.
     * @param[in] e  the press
     */
    void mouseDown( const juce::MouseEvent &e ) override
    {
        m_held = std::abs( e.position.x - xOf( m_lo ) ) <= std::abs( e.position.x - xOf( m_hi ) ) ? 0 : 1;
        mouseDrag( e );
    }
    /**
     * @brief Moves it.
     * @param[in] e  the drag
     */
    void mouseDrag( const juce::MouseEvent &e ) override
    {
        const float v = valueAt( e.position.x );
        if ( m_held == 0 )
            m_lo = std::min( v, m_hi - 0.1f );
        else
            m_hi = std::max( v, m_lo + 0.1f );
        repaint();
        if ( m_onChange ) m_onChange();
    }
    /**
     * @brief Wide open again.
     */
    void mouseDoubleClick( const juce::MouseEvent & ) override
    {
        m_lo = -3.0f;
        m_hi = 3.0f;
        repaint();
        if ( m_onChange ) m_onChange();
    }

private:
    /**
     * @brief A value's place.
     * @param[in] v  spreads
     * @return x
     */
    float xOf( float v ) const { return 6.0f + ( v + 3.0f ) / 6.0f * ( static_cast<float>( getWidth() ) - 12.0f ); }
    /**
     * @brief The value at a place.
     * @param[in] x  pixels
     * @return spreads
     */
    float valueAt( float x ) const
    {
        return juce::jlimit( -3.0f, 3.0f,
                             ( x - 6.0f ) / std::max( 1.0f, static_cast<float>( getWidth() ) - 12.0f ) * 6.0f - 3.0f );
    }
    int m_held = 0;   ///< the end being dragged
};

/** @brief The categories' list (see PresetBrowser). */
class PresetBrowser::Categories final : public juce::ListBoxModel
{
public:
    /**
     * @brief The categories of @p owner.
     * @param[in,out] owner  the browser
     */
    explicit Categories( PresetBrowser &owner ) : m_owner( owner ) {}
    /** @brief How many. @return rows */
    int getNumRows() override { return static_cast<int>( m_owner.m_categories.size() ); }
    /**
     * @brief One: a heading for a section, indented for a group.
     * @param[in]     row       the row
     * @param[in,out] g         the graphics context
     * @param[in]     width     the width in pixels
     * @param[in]     height    the height in pixels
     * @param[in]     selected  true for the chosen row
     */
    void paintListBoxItem( int row, juce::Graphics &g, int width, int height, bool selected ) override
    {
        if ( row < 0 || row >= getNumRows() ) return;
        const frame::Skin &skin = additivui::skin();
        const Category    &c = m_owner.m_categories[static_cast<size_t>( row )];
        if ( selected )
        {
            g.setColour( skin.m_accent.withAlpha( 0.18f ) );
            g.fillRoundedRectangle(
                juce::Rectangle<float>( 0.0f, 0.0f, static_cast<float>( width ), static_cast<float>( height ) ),
                skin.m_radius );
        }
        const bool heading = c.m_kind == 4 || c.m_kind <= 3;
        g.setColour( heading ? skin.m_ink : skin.m_dim );
        g.setFont( juce::FontOptions( heading ? 13.0f : 12.0f, heading ? juce::Font::bold : juce::Font::plain ) );
        g.drawText( c.m_title, c.m_kind == 5 ? 18 : 6, 0, width - 20, height, juce::Justification::centredLeft );
    }
    /**
     * @brief Chooses it.
     * @param[in] row  the row chosen
     */
    void selectedRowsChanged( int row ) override
    {
        if ( row < 0 ) return;
        m_owner.m_category = row;
        m_owner.filter();
    }

private:
    PresetBrowser &m_owner;   ///< whose
};

PresetBrowser::~PresetBrowser() = default;

PresetBrowser::PresetBrowser( AdditivProcessor &proc ) : m_proc( proc ), m_list( {}, this )
{
    const frame::Skin &skin = additivui::skin();
    m_search.setFont( juce::FontOptions( 16.0f ) );
    m_search.setTextToShowWhenEmpty( "A name or a group ...", skin.m_dim );
    m_search.setColour( juce::TextEditor::backgroundColourId, skin.m_raised );
    m_search.setColour( juce::TextEditor::outlineColourId, skin.m_edge );
    m_search.addListener( this );
    addAndMakeVisible( m_search );
    m_categoryModel = std::make_unique<Categories>( *this );
    m_categoryList = std::make_unique<juce::ListBox>( "categories", m_categoryModel.get() );
    m_categoryList->setRowHeight( 22 );
    m_categoryList->setColour( juce::ListBox::backgroundColourId, juce::Colours::transparentBlack );
    addAndMakeVisible( *m_categoryList );
    for ( int k = 0; k < 4; ++k )
    {
        m_traits[static_cast<size_t>( k )] = std::make_unique<RangeBar>();
        RangeBar &s = *m_traits[static_cast<size_t>( k )];
        s.setTooltip( juce::String( "Only the presets measured between these ends: " ) + TraitLow[k] + " to " +
                      TraitHigh[k] + " (in spreads of the bank; double-click: all)" );
        s.m_onChange = [this] { filter(); };
        addAndMakeVisible( s );
        juce::Label &l = m_traitNames[static_cast<size_t>( k )];
        l.setText( juce::String( TraitLow[k] ) + " ... " + TraitHigh[k], juce::dontSendNotification );
        l.setColour( juce::Label::textColourId, skin.m_dim );
        l.setFont( juce::FontOptions( 12.0f ) );
        addAndMakeVisible( l );
    }
    m_who.addItem( "Anyone plays", 1 );
    m_who.addItem( "The conductor plays", 2 );
    m_who.addItem( "For the hands", 3 );
    m_who.setSelectedId( 1, juce::dontSendNotification );
    m_who.onChange = [this] { filter(); };
    addAndMakeVisible( m_who );
    m_take.onClick = [this] { choose( m_list.getSelectedRow() ); };
    m_close.onClick = [this] { close(); };
    addAndMakeVisible( m_take );
    addAndMakeVisible( m_close );
    m_count.setColour( juce::Label::textColourId, skin.m_dim );
    m_count.setFont( juce::FontOptions( 12.0f ) );
    addAndMakeVisible( m_count );
    m_list.setRowHeight( 28 );
    m_list.setColour( juce::ListBox::backgroundColourId, juce::Colours::transparentBlack );
    addAndMakeVisible( m_list );
    setWantsKeyboardFocus( true );
}

void PresetBrowser::open()
{
    // Every preset: the player's, then the bank's with their measured traits.
    m_all.clear();
    m_proc.refreshUserPresets();
    for ( size_t i = 0; i < m_proc.userPresets().size(); ++i )
    {
        Entry e;
        e.m_user = static_cast<int>( i );
        e.m_name = m_proc.userPresets()[i].m_name;
        e.m_group = "Yours";
        e.m_section = 6;
        e.m_hands = m_proc.userPresets()[i].m_values[static_cast<size_t>( additiv::ParamIdE::BRAIN_ON )] < 0.5f;
        m_all.push_back( e );
    }
    for ( int i = 0; i < additiv::numPresets(); ++i )
    {
        Entry e;
        e.m_factory = i;
        e.m_name = additiv::preset( i ).m_name;
        e.m_group = additiv::PresetGroupNames[additiv::preset( i ).m_group];
        e.m_section = additiv::preset( i ).m_group / 16;
        e.m_measured = additiv::presetFeatures( i, e.m_traits );
        e.m_hands = juce::String( additiv::preset( i ).m_params ).contains( "brain_on=0" );
        m_all.push_back( e );
    }
    m_categories.clear();
    m_categories.push_back( { "All", 0, 0 } );
    m_categories.push_back( { "Favourites", 1, 0 } );
    m_categories.push_back( { "Recently played", 2, 0 } );
    m_categories.push_back( { "Yours", 3, 0 } );
    for ( int s = 0; s < 10; ++s )
    {
        m_categories.push_back( { Sections[s], 4, s } );
        for ( int g = s * 16; g < ( s + 1 ) * 16 && g < additiv::NumPresetGroups; ++g )
            m_categories.push_back( { additiv::PresetGroupNames[g], 5, g } );
    }
    m_categoryList->updateContent();
    if ( m_category >= static_cast<int>( m_categories.size() ) ) m_category = 0;
    m_categoryList->selectRow( m_category, true, true );
    filter();
    setVisible( true );
    toFront( true );
    m_search.grabKeyboardFocus();
}

void PresetBrowser::close()
{
    m_proc.endPreview();
    setVisible( false );
    if ( auto *p = getParentComponent() ) p->grabKeyboardFocus();
}

void PresetBrowser::filter()
{
    const juce::String      q = m_search.getText().trim().toLowerCase();
    const Category         &c = m_categories.empty() ? Category{} : m_categories[static_cast<size_t>( m_category )];
    const juce::StringArray recent = c.m_kind == 2 ? m_proc.recentPresets() : juce::StringArray();
    m_shown.clear();
    std::vector<int> order;
    for ( size_t i = 0; i < m_all.size(); ++i )
    {
        const Entry &e = m_all[i];
        if ( c.m_kind == 1 && !m_proc.favorite( e.m_name ) ) continue;
        if ( c.m_kind == 2 && !recent.contains( e.m_name ) ) continue;
        if ( c.m_kind == 3 && e.m_user < 0 ) continue;
        if ( c.m_kind == 4 && ( e.m_factory < 0 || e.m_section != c.m_which ) ) continue;
        if ( c.m_kind == 5 && ( e.m_factory < 0 || additiv::preset( e.m_factory ).m_group != c.m_which ) ) continue;
        if ( q.isNotEmpty() && !e.m_name.toLowerCase().contains( q ) && !e.m_group.toLowerCase().contains( q ) )
            continue;
        const int who = m_who.getSelectedId();
        if ( ( who == 2 && e.m_hands ) || ( who == 3 && !e.m_hands ) ) continue;
        bool inside = true;
        for ( int k = 0; k < 4 && inside; ++k )
        {
            const RangeBar &s = *m_traits[static_cast<size_t>( k )];
            if ( !s.open() && ( !e.m_measured || e.m_traits[k] < s.m_lo || e.m_traits[k] > s.m_hi ) ) inside = false;
        }
        if ( inside ) m_shown.push_back( static_cast<int>( i ) );
    }
    if ( c.m_kind == 2 )   // the recent ones newest first
        std::stable_sort( m_shown.begin(), m_shown.end(),
                          [&]( int a, int b )
                          {
                              return recent.indexOf( m_all[static_cast<size_t>( a )].m_name ) <
                                     recent.indexOf( m_all[static_cast<size_t>( b )].m_name );
                          } );
    m_list.updateContent();
    m_list.selectRow( m_shown.empty() ? -1 : 0 );
    m_list.repaint();
    m_count.setText( juce::String( static_cast<int>( m_shown.size() ) ) + " presets", juce::dontSendNotification );
}

void PresetBrowser::choose( int row )
{
    if ( row < 0 || row >= static_cast<int>( m_shown.size() ) ) return;
    const Entry e = m_all[static_cast<size_t>( m_shown[static_cast<size_t>( row )] )];
    if ( m_onChoose ) m_onChoose( e.m_factory, e.m_user );
    close();
}

juce::Rectangle<int> PresetBrowser::panel() const
{
    const int w = std::min( 1000, getWidth() - 40 ), h = std::min( 640, getHeight() - 80 );
    return { ( getWidth() - w ) / 2, 50, w, h };
}

void PresetBrowser::resized()
{
    auto r = panel().reduced( 12 );
    auto top = r.removeFromTop( 34 );
    m_close.setBounds( top.removeFromRight( 80 ).reduced( 2 ) );
    m_take.setBounds( top.removeFromRight( 80 ).reduced( 2 ) );
    top.removeFromRight( 8 );
    m_search.setBounds( top );
    r.removeFromTop( 8 );
    m_categoryList->setBounds( r.removeFromLeft( 210 ) );
    r.removeFromLeft( 10 );
    auto      filters = r.removeFromTop( 64 );
    const int w = filters.getWidth() / 5;
    for ( int k = 0; k < 4; ++k )
    {
        auto cell = filters.withX( filters.getX() + k * w ).withWidth( w ).reduced( 4, 0 );
        m_traitNames[static_cast<size_t>( k )].setBounds( cell.removeFromTop( 18 ) );
        m_traits[static_cast<size_t>( k )]->setBounds( cell.removeFromTop( 24 ) );
    }
    auto last = filters.withX( filters.getX() + 4 * w ).withWidth( w ).reduced( 4, 0 );
    m_count.setBounds( last.removeFromTop( 18 ) );
    m_who.setBounds( last.removeFromTop( 24 ) );
    r.removeFromTop( 6 );
    m_list.setBounds( r );
}

void PresetBrowser::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.fillAll( skin.m_bg.withAlpha( 0.55f ) );
    const auto p = panel().toFloat();
    g.setColour( skin.m_group );
    g.fillRoundedRectangle( p, skin.m_radius * 2.0f );
    g.setColour( skin.m_accent.withAlpha( 0.5f ) );
    g.drawRoundedRectangle( p.reduced( 0.5f ), skin.m_radius * 2.0f, 1.2f );
}

void PresetBrowser::mouseDown( const juce::MouseEvent &e )
{
    if ( !panel().contains( e.getPosition() ) ) close();
}

bool PresetBrowser::keyPressed( const juce::KeyPress &key )
{
    if ( key.isKeyCode( juce::KeyPress::escapeKey ) )
    {
        close();
        return true;
    }
    if ( key.isKeyCode( juce::KeyPress::returnKey ) )
    {
        choose( m_list.getSelectedRow() );
        return true;
    }
    return false;
}

int PresetBrowser::getNumRows()
{
    return static_cast<int>( m_shown.size() );
}

juce::Component *PresetBrowser::refreshComponentForRow( int row, bool selected, juce::Component *existing )
{
    auto *r = dynamic_cast<Row *>( existing );
    if ( r == nullptr )
    {
        delete existing;
        r = new Row( *this );
    }
    r->show( row, row >= 0 && row < static_cast<int>( m_shown.size() ) ? m_shown[static_cast<size_t>( row )] : -1,
             selected );
    return r;
}
