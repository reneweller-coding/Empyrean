/**
 * @file JourneysComponent.cpp
 * @brief Empyrean Journeys' window and its queue of renderings.
 * @author Rene Weller
 */
#include "JourneysComponent.h"

#include <algorithm>

namespace journeys
{

namespace
{

/** @brief The window's colours: the instrument's (Plugin/Theme.h). */
namespace Colours
{
const juce::Colour Back{ 0xff07090f };     ///< the background: the night
const juce::Colour Panel{ 0xff111725 };    ///< the panel and the table (a group box)
const juce::Colour Row{ 0xff0d121d };      ///< every other row
const juce::Colour Raised{ 0xff1a2132 };   ///< buttons and menus
const juce::Colour Edge{ 0xff283149 };     ///< hairlines
const juce::Colour Chosen{ 0xff24405a };   ///< a chosen row
const juce::Colour Text{ 0xffe2ecf2 };     ///< text
const juce::Colour Dim{ 0xff8794a8 };      ///< secondary text
const juce::Colour Accent{ 0xff7fe0c4 };   ///< the aurora's green
}   // namespace Colours

/**
 * @brief A seed that differs every time it is drawn.
 * @return the seed
 */
std::uint64_t freshSeed()
{
    return static_cast<std::uint64_t>( juce::Time::currentTimeMillis() ) * 2654435761ull + 1ull;
}

/**
 * @brief A folder as the panel shows it: from "Documents" on when it lies there, else its whole path.
 * @param[in] f  the folder
 * @return the text
 */
juce::String folderText( const juce::File &f )
{
    const juce::File docs = juce::File::getSpecialLocation( juce::File::userDocumentsDirectory );
    return f.isAChildOf( docs ) ? docs.getFileName() + juce::File::getSeparatorString() + f.getRelativePathFrom( docs )
                                : f.getFullPathName();
}

}   // namespace

// ---- the queue

RenderQueue::~RenderQueue()
{
    cancelAll();
    signalThreadShouldExit();
    m_wake.signal();
    stopThread( 30000 );
}

void RenderQueue::add( const std::vector<RenderJob> &jobs )
{
    {
        const std::lock_guard<std::mutex> hold( m_lock );
        for ( const RenderJob &j : jobs ) m_jobs.push_back( j );
    }
    if ( !isThreadRunning() ) startThread();   // once: it then waits for work until the window closes
    m_wake.signal();
}

void RenderQueue::cancelAll()
{
    const std::lock_guard<std::mutex> hold( m_lock );
    m_jobs.clear();
    m_cancel.store( true );   // the one under way (a job taken later clears it, under the same lock)
}

bool RenderQueue::status( juce::String &name, int &waiting, int &finished )
{
    const std::lock_guard<std::mutex> hold( m_lock );
    name = m_current;
    waiting = static_cast<int>( m_jobs.size() );
    finished = m_finished;
    return m_current.isNotEmpty();
}

juce::String RenderQueue::takeMessage()
{
    const std::lock_guard<std::mutex> hold( m_lock );
    juce::String                      m = m_message;
    m_message.clear();
    return m;
}

void RenderQueue::run()
{
    while ( !threadShouldExit() )
    {
        RenderJob job;
        bool      taken = false;
        {
            const std::lock_guard<std::mutex> hold( m_lock );
            if ( m_jobs.empty() )
                m_current.clear();
            else
            {
                job = m_jobs.front();
                m_jobs.pop_front();
                m_current = job.m_item.m_name;
                m_cancel.store( false );
                taken = true;
            }
        }
        if ( !taken )
        {   // nothing to do: wait for add() (or the window closing)
            m_wake.wait( 500 );
            continue;
        }
        juce::String why;
        const bool   ok = m_renderer.render( job, [this] { return m_cancel.load() || threadShouldExit(); }, why );
        const std::lock_guard<std::mutex> hold( m_lock );
        if ( ok )
        {
            ++m_finished;
            m_message = "Written: " + job.m_file.getFileName() + " (" + clockText( m_renderer.m_total.load() ) + ", " +
                        juce::String( m_renderer.m_speed.load(), 1 ) + "x real time)";
        }
        else
        {
            m_message = job.m_item.m_name + ": " + why;
            if ( why == "cancelled" ) job.m_file.deleteFile();   // a part of a journey is no rendering of it
        }
        m_current.clear();
    }
}

// ---- the window

JourneysComponent::JourneysComponent()
{
    m_look.setColourScheme( { Colours::Back, Colours::Raised, Colours::Panel, Colours::Edge, Colours::Text,
                              Colours::Raised, Colours::Back, Colours::Accent, Colours::Text } );
    m_look.setColour( juce::TableHeaderComponent::backgroundColourId, Colours::Raised );
    m_look.setColour( juce::TableHeaderComponent::textColourId, Colours::Text );
    m_look.setColour( juce::TableHeaderComponent::outlineColourId, Colours::Edge );
    m_look.setColour( juce::TextEditor::backgroundColourId, Colours::Panel );
    m_look.setColour( juce::TextEditor::outlineColourId, Colours::Edge );
    setLookAndFeel( &m_look );
    juce::PropertiesFile::Options o;
    o.applicationName = "journeys";
    o.filenameSuffix = ".settings";
    o.folderName = "Empyrean";
    m_settings = std::make_unique<juce::PropertiesFile>( o );
    m_folder =
        juce::File( m_settings->getValue( "folder", juce::File::getSpecialLocation( juce::File::userDocumentsDirectory )
                                                        .getChildFile( "Empyrean" )
                                                        .getChildFile( "Renders" )
                                                        .getFullPathName() ) );
    m_seed = freshSeed();

    m_title.setText( "Empyrean Journeys", juce::dontSendNotification );
    m_title.setFont( juce::FontOptions( 24.0f, juce::Font::bold ) );
    m_title.setColour( juce::Label::textColourId, Colours::Text );
    addAndMakeVisible( m_title );
    m_search.setText( "Search", juce::dontSendNotification );
    m_search.setColour( juce::Label::textColourId, Colours::Dim );
    addAndMakeVisible( m_search );
    m_filter.setTextToShowWhenEmpty( "a name, a word: night, storm, choir ...", Colours::Dim );
    m_filter.onTextChange = [this] { refilter(); };
    addAndMakeVisible( m_filter );

    m_table.setModel( this );
    m_table.setColour( juce::ListBox::backgroundColourId, Colours::Panel );
    m_table.setMultipleSelectionEnabled( true );
    m_table.setRowHeight( 24 );
    auto &h = m_table.getHeader();
    h.addColumn( "Journey", NAME, 330, 120, 800 );
    h.addColumn( "Steps", STEPS, 56, 40, 90 );
    h.addColumn( "Length", LENGTH, 80, 60, 140 );
    h.addColumn( "Shortest .. longest", RANGE, 150, 90, 220 );
    h.addColumn( "Kind", KIND, 104, 60, 160 );
    h.setSortColumnId( NAME, true );
    h.setStretchToFitActive( true );   // the columns share the table's width: no sideways scrolling
    addAndMakeVisible( m_table );

    m_chosenName.setFont( juce::FontOptions( 18.0f, juce::Font::bold ) );
    m_chosenName.setColour( juce::Label::textColourId, Colours::Accent );
    addAndMakeVisible( m_chosenName );
    m_chosenInfo.setColour( juce::Label::textColourId, Colours::Text );
    m_chosenInfo.setJustificationType( juce::Justification::topLeft );
    addAndMakeVisible( m_chosenInfo );

    m_formatLabel.setText( "Format", juce::dontSendNotification );
    m_passesLabel.setText( "Passes", juce::dontSendNotification );
    m_folderLabel.setText( "Folder", juce::dontSendNotification );
    for ( juce::Label *l : { &m_formatLabel, &m_passesLabel, &m_folderLabel } )
    {
        l->setColour( juce::Label::textColourId, Colours::Dim );
        addAndMakeVisible( *l );
    }
    for ( int f = 0; f < static_cast<int>( FormatE::COUNT ); ++f )
        m_format.addItem( formatName( static_cast<FormatE>( f ) ), f + 1 );
    m_format.setSelectedId( m_settings->getIntValue( "format", 1 ), juce::dontSendNotification );
    addAndMakeVisible( m_format );
    for ( int p = 1; p <= 8; ++p ) m_passes.addItem( juce::String( p ), p );
    m_passes.setSelectedId( m_settings->getIntValue( "passes", 1 ), juce::dontSendNotification );
    m_passes.setTooltip( "How many times a cyclic journey goes round (a journey that does not cycle plays once)." );
    m_passes.onChange = [this] { updateChosen(); };
    addAndMakeVisible( m_passes );
    m_dice.setTooltip(
        "A journey draws every dwell and glide from a range: new dice, a new night (and a new length)." );
    m_dice.onClick = [this]
    {
        m_seed = freshSeed();
        updateChosen();
    };
    addAndMakeVisible( m_dice );
    m_fade.setToggleState( m_settings->getBoolValue( "fade", true ), juce::dontSendNotification );
    addAndMakeVisible( m_fade );
    m_night.setToggleState( m_settings->getBoolValue( "night", false ), juce::dontSendNotification );
    m_night.setTooltip( "The presets with a night arc darken, slow and widen as the real night does; rendered as from "
                        "23:00 where the instrument stands (its settings), else as the clock says now." );
    addAndMakeVisible( m_night );
    m_exact.setToggleState( m_settings->getBoolValue( "exact", false ), juce::dontSendNotification );
    m_exact.setTooltip(
        "The instrument sets every partial 375 times a second; a rendering does it 94 times, about "
        "three times faster -- the same sound, only Scintillation's quickest twinkle a little smoother. "
        "On: as the instrument plays, sample for sample." );
    addAndMakeVisible( m_exact );
    for ( juce::ToggleButton *b : { &m_fade, &m_night, &m_exact } )
        b->setColour( juce::ToggleButton::textColourId, Colours::Text );
    m_folderText.setText( folderText( m_folder ), juce::dontSendNotification );
    m_folderText.setTooltip( m_folder.getFullPathName() );
    m_folderText.setColour( juce::Label::textColourId, Colours::Text );
    m_folderText.setMinimumHorizontalScale( 0.6f );
    addAndMakeVisible( m_folderText );
    m_chooseFolder.onClick = [this]
    {
        m_chooser = std::make_unique<juce::FileChooser>( "Where the renderings go", m_folder );
        m_chooser->launchAsync( juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
                                [this]( const juce::FileChooser &fc )
                                {
                                    if ( fc.getResult() == juce::File() ) return;
                                    m_folder = fc.getResult();
                                    m_folderText.setText( folderText( m_folder ), juce::dontSendNotification );
                                    m_folderText.setTooltip( m_folder.getFullPathName() );
                                    m_settings->setValue( "folder", m_folder.getFullPathName() );
                                } );
    };
    addAndMakeVisible( m_chooseFolder );
    m_openFolder.onClick = [this]
    {
        m_folder.createDirectory();
        m_folder.startAsProcess();
    };
    addAndMakeVisible( m_openFolder );
    m_render.setColour( juce::TextButton::buttonColourId, Colours::Chosen );
    m_render.onClick = [this] { renderChosen(); };
    addAndMakeVisible( m_render );
    m_cancel.onClick = [this] { m_queue.cancelAll(); };
    addAndMakeVisible( m_cancel );
    m_bar.setColour( juce::ProgressBar::foregroundColourId, Colours::Accent );
    addAndMakeVisible( m_bar );
    m_status.setColour( juce::Label::textColourId, Colours::Text );
    m_status.setJustificationType( juce::Justification::topLeft );
    m_status.setText( "Choose a journey (several with Ctrl or Shift), then Render. Double click renders one at once.",
                      juce::dontSendNotification );
    addAndMakeVisible( m_status );

    m_items = listJourneys();
    refilter();
    setSize( 1100, 680 );
    startTimerHz( 10 );
}

JourneysComponent::~JourneysComponent()
{
    stopTimer();
    m_settings->setValue( "format", m_format.getSelectedId() );
    m_settings->setValue( "passes", m_passes.getSelectedId() );
    m_settings->setValue( "fade", m_fade.getToggleState() );
    m_settings->setValue( "night", m_night.getToggleState() );
    m_settings->setValue( "exact", m_exact.getToggleState() );
    m_settings->saveIfNeeded();
    m_table.setModel( nullptr );
    setLookAndFeel( nullptr );
}

bool JourneysComponent::choose( const juce::String &name )
{
    for ( size_t r = 0; r < m_rows.size(); ++r )
        if ( m_items[static_cast<size_t>( m_rows[r] )].m_name == name )
        {
            m_table.selectRow( static_cast<int>( r ) );
            m_table.scrollToEnsureRowIsOnscreen( static_cast<int>( r ) );
            updateChosen();
            return true;
        }
    return false;
}

void JourneysComponent::paint( juce::Graphics &g )
{
    g.fillAll( Colours::Back );
    const juce::Rectangle<int> panel = getLocalBounds().removeFromRight( 400 ).reduced( 12 ).withTrimmedTop( 52 );
    g.setColour( Colours::Panel );
    g.fillRoundedRectangle( panel.toFloat(), 8.0f );
    g.setColour( Colours::Dim );
    g.setFont( juce::FontOptions( 13.0f ) );
    g.drawText( "Render journeys into files, faster than real time", 252, 22, 420, 22,
                juce::Justification::centredLeft );
}

void JourneysComponent::resized()
{
    juce::Rectangle<int> r = getLocalBounds().reduced( 12 );
    m_title.setBounds( r.removeFromTop( 40 ).removeFromLeft( 220 ) );
    juce::Rectangle<int> side = r.removeFromRight( 388 ).reduced( 14, 12 );
    r.removeFromRight( 12 );
    juce::Rectangle<int> search = r.removeFromTop( 30 );
    m_search.setBounds( search.removeFromLeft( 60 ) );
    m_filter.setBounds( search );
    r.removeFromTop( 8 );
    m_table.setBounds( r );

    m_chosenName.setBounds( side.removeFromTop( 28 ) );
    m_chosenInfo.setBounds( side.removeFromTop( 92 ) );
    auto row = [&side]( int h )
    {
        juce::Rectangle<int> x = side.removeFromTop( h );
        side.removeFromTop( 8 );
        return x;
    };
    juce::Rectangle<int> f = row( 28 );
    m_formatLabel.setBounds( f.removeFromLeft( 70 ) );
    m_format.setBounds( f );
    juce::Rectangle<int> p = row( 28 );
    m_passesLabel.setBounds( p.removeFromLeft( 70 ) );
    m_passes.setBounds( p.removeFromLeft( 80 ) );
    p.removeFromLeft( 12 );
    m_dice.setBounds( p );
    m_fade.setBounds( row( 24 ) );
    m_night.setBounds( row( 24 ) );
    m_exact.setBounds( row( 24 ) );
    juce::Rectangle<int> d = row( 24 );
    m_folderLabel.setBounds( d.removeFromLeft( 70 ) );
    m_folderText.setBounds( d );
    juce::Rectangle<int> b = row( 28 );
    m_chooseFolder.setBounds( b.removeFromLeft( b.getWidth() / 2 - 4 ) );
    b.removeFromLeft( 8 );
    m_openFolder.setBounds( b );
    side.removeFromTop( 10 );
    juce::Rectangle<int> go = row( 36 );
    m_render.setBounds( go.removeFromLeft( go.getWidth() * 2 / 3 - 4 ) );
    go.removeFromLeft( 8 );
    m_cancel.setBounds( go );
    m_bar.setBounds( row( 22 ) );
    m_status.setBounds( side );
}

void JourneysComponent::refilter()
{
    const juce::StringArray words = juce::StringArray::fromTokens( m_filter.getText().toLowerCase(), " ", "" );
    m_rows.clear();
    for ( size_t i = 0; i < m_items.size(); ++i )
    {
        const juce::String hay = ( m_items[i].m_name + " " + m_items[i].m_kind ).toLowerCase();
        bool               all = true;
        for ( const juce::String &w : words ) all = all && ( w.isEmpty() || hay.contains( w ) );
        if ( all ) m_rows.push_back( static_cast<int>( i ) );
    }
    const auto key = [this]( int a, int b )
    {
        const JourneyItem &x = m_items[static_cast<size_t>( a )], &y = m_items[static_cast<size_t>( b )];
        switch ( m_sortColumn )
        {
        case STEPS: return x.m_journey.m_steps.size() < y.m_journey.m_steps.size();
        case LENGTH:
        case RANGE: return x.m_mean < y.m_mean;
        case KIND: return x.m_kind.compareNatural( y.m_kind ) < 0;
        default: return x.m_name.compareNatural( y.m_name ) < 0;
        }
    };
    std::stable_sort( m_rows.begin(), m_rows.end(),
                      [&]( int a, int b ) { return m_sortForward ? key( a, b ) : key( b, a ); } );
    m_table.updateContent();
    m_table.repaint();
    updateChosen();
}

std::vector<const JourneyItem *> JourneysComponent::chosen() const
{
    std::vector<const JourneyItem *> out;
    const juce::SparseSet<int>       rows = m_table.getSelectedRows();
    for ( int k = 0; k < rows.size(); ++k )
    {
        const int row = rows[k];
        if ( row >= 0 && row < static_cast<int>( m_rows.size() ) )
            out.push_back( &m_items[static_cast<size_t>( m_rows[static_cast<size_t>( row )] )] );
    }
    return out;
}

void JourneysComponent::updateChosen()
{
    const std::vector<const JourneyItem *> c = chosen();
    if ( c.empty() )
    {
        m_chosenName.setText( "No journey chosen", juce::dontSendNotification );
        m_chosenInfo.setText( juce::String( static_cast<int>( m_items.size() ) ) + " journeys to choose from.",
                              juce::dontSendNotification );
        return;
    }
    const JourneyItem &j = *c.front();
    const int          passes = j.m_journey.m_cyclic ? m_passes.getSelectedId() : 1;
    const double       len = planLength( j.m_journey, m_seed, passes );
    double             all = 0.0;
    for ( const JourneyItem *x : c ) all += planLength( x->m_journey, m_seed, x->m_journey.m_cyclic ? passes : 1 );
    m_chosenName.setText( c.size() == 1 ? j.m_name : juce::String( static_cast<int>( c.size() ) ) + " journeys chosen",
                          juce::dontSendNotification );
    juce::String info;
    info << juce::String( static_cast<int>( j.m_journey.m_steps.size() ) ) << " steps"
         << ( j.m_journey.m_cyclic ? ", cyclic" : ", once through" ) << "\n"
         << "A pass: " << clockText( j.m_shortest ) << " .. " << clockText( j.m_longest ) << " (about "
         << clockText( j.m_mean ) << ")\n"
         << "This rendering: " << clockText( len ) << ( passes > 1 ? " (" + juce::String( passes ) + " passes)" : "" );
    if ( c.size() > 1 ) info << "\nAll " << juce::String( static_cast<int>( c.size() ) ) << ": " << clockText( all );
    m_chosenInfo.setText( info, juce::dontSendNotification );
}

juce::File JourneysComponent::fileFor( const JourneyItem &item, double len ) const
{
    const FormatE      f = static_cast<FormatE>( m_format.getSelectedId() - 1 );
    const juce::String base =
        juce::File::createLegalFileName( item.m_name + " (" + clockText( len ).replace( ":", "-" ) + ")" );
    return m_folder.getChildFile( base + formatExtension( f ) ).getNonexistentSibling( false );
}

void JourneysComponent::renderChosen()
{
    const std::vector<const JourneyItem *> c = chosen();
    if ( c.empty() )
    {
        m_status.setText( "Choose a journey first.", juce::dontSendNotification );
        return;
    }
    if ( !m_folder.createDirectory() )
    {
        m_status.setText( "Cannot write into " + m_folder.getFullPathName(), juce::dontSendNotification );
        return;
    }
    std::vector<RenderJob> jobs;
    for ( const JourneyItem *item : c )
    {
        RenderJob job;
        job.m_item = *item;
        job.m_seed = m_seed;
        job.m_passes = item->m_journey.m_cyclic ? m_passes.getSelectedId() : 1;
        job.m_format = static_cast<FormatE>( m_format.getSelectedId() - 1 );
        job.m_fade = m_fade.getToggleState() ? 20.0 : 0.0;
        job.m_night = m_night.getToggleState();
        job.m_exact = m_exact.getToggleState();
        job.m_file = fileFor( *item, planLength( item->m_journey, m_seed, job.m_passes ) );
        jobs.push_back( std::move( job ) );
    }
    m_queue.add( jobs );
}

void JourneysComponent::timerCallback()
{
    juce::String       name;
    int                waiting = 0, finished = 0;
    const bool         busy = m_queue.status( name, waiting, finished );
    const juce::String message = m_queue.takeMessage();
    m_cancel.setEnabled( busy || waiting > 0 );
    if ( busy )
    {
        const double done = m_queue.m_renderer.m_done.load(), total = m_queue.m_renderer.m_total.load();
        const double speed = m_queue.m_renderer.m_speed.load();
        m_progress = total > 0.0 ? done / total : 0.0;
        juce::String s;
        s << name << "\n"
          << clockText( done ) << " of " << clockText( total ) << "   " << juce::String( speed, 1 ) << "x real time";
        if ( speed > 0.0 ) s << "\nabout " << clockText( ( total - done ) / speed ) << " left";
        const int step = m_queue.m_renderer.m_step.load();
        s << "   (step " << juce::String( step + 1 ) << ")";
        if ( waiting > 0 ) s << "\n" << juce::String( waiting ) << " more waiting";
        if ( finished > 0 ) s << "\n" << juce::String( finished ) << " written";
        m_status.setText( s, juce::dontSendNotification );
    }
    else if ( message.isNotEmpty() )
    {
        m_progress = message.startsWith( "Written" ) ? 1.0 : 0.0;
        m_status.setText( message, juce::dontSendNotification );
    }
}

int JourneysComponent::getNumRows()
{
    return static_cast<int>( m_rows.size() );
}

void JourneysComponent::paintRowBackground( juce::Graphics &g, int row, int, int, bool selected )
{
    g.fillAll( selected ? Colours::Chosen : ( row % 2 ? Colours::Row : Colours::Panel ) );
}

void JourneysComponent::paintCell( juce::Graphics &g, int row, int column, int w, int h, bool )
{
    if ( row < 0 || row >= static_cast<int>( m_rows.size() ) ) return;
    const JourneyItem &j = m_items[static_cast<size_t>( m_rows[static_cast<size_t>( row )] )];
    juce::String       text;
    switch ( column )
    {
    case NAME: text = j.m_name; break;
    case STEPS: text = juce::String( static_cast<int>( j.m_journey.m_steps.size() ) ); break;
    case LENGTH: text = clockText( j.m_mean ); break;
    case RANGE: text = clockText( j.m_shortest ) + " .. " + clockText( j.m_longest ); break;
    case KIND: text = j.m_kind; break;
    default: break;
    }
    g.setColour( column == NAME ? Colours::Text : Colours::Dim );
    g.setFont( juce::FontOptions( 14.0f ) );
    g.drawText( text, 6, 0, w - 10, h,
                column == NAME || column == KIND ? juce::Justification::centredLeft : juce::Justification::centredRight,
                true );
}

void JourneysComponent::sortOrderChanged( int column, bool forward )
{
    m_sortColumn = column;
    m_sortForward = forward;
    refilter();
}

void JourneysComponent::selectedRowsChanged( int )
{
    updateChosen();
}

void JourneysComponent::cellDoubleClicked( int, int, const juce::MouseEvent & )
{
    renderChosen();
}

}   // namespace journeys
