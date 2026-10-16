/**
 * @file JourneyView.cpp
 * @brief The Journey page (JourneyView.h).
 * @author Rene Weller
 */
#include "JourneyView.h"

#include "Theme.h"

#include "additiv/Presets.h"

using additiv::ParamIdE;

JourneyView::JourneyView( AdditivProcessor &proc ) : m_proc( proc )
{
    for ( int k = 0; k < 2; ++k )
    {
        SlotRow &s = m_slots[k];
        s.m_name.setColour( juce::Label::textColourId, additivui::skin().m_ink );
        s.m_name.setFont( juce::FontOptions( 13.0f ) );
        s.m_fromSound.setTooltip( juce::String( "Slot " ) + ( k == 0 ? "A" : "B" ) +
                                  " takes the sound as it plays now: every knob and the matrix" );
        s.m_fromPreset.setTooltip( juce::String( "Slot " ) + ( k == 0 ? "A" : "B" ) +
                                   " takes a preset from the bank, or one of your snapshots" );
        s.m_fromSound.onClick = [this, k] { m_proc.setSlotFromSound( k ); };
        s.m_fromPreset.onClick = [this, k] { choosePreset( k ); };
        addAndMakeVisible( s.m_name );
        addAndMakeVisible( s.m_fromSound );
        addAndMakeVisible( s.m_fromPreset );
    }
    m_journeys.setTextWhenNothingSelected( "Choose a journey" );
    m_journeys.setTooltip(
        "The journeys: Whole Night plays about six and a half hours before it comes round again; a group's Night is its "
        "twelve stillest presets, its Journey twelve of its different faces. Yours are the files in Documents/Empyrean/Journeys." );
    addAndMakeVisible( m_journeys );
    m_start.onClick = [this] { startChosen(); };
    m_stop.onClick = [this] { m_proc.stopJourney(); };
    m_add.setTooltip(
        "Appends the preset playing to the journey being written (held 8-15 minutes, glided into over 1:30-3:00; edit the file for other times)" );
    m_add.onClick = [this] { m_proc.journeyAddCurrent(); };
    m_save.onClick = [this] { saveDraft(); };
    m_folder.onClick = []
    {
        const juce::File dir = AdditivProcessor::journeyFolder();
        dir.createDirectory();
        dir.startAsProcess();
    };
    for ( juce::Label *l : { &m_status, &m_draft } )
    {
        l->setColour( juce::Label::textColourId, additivui::skin().m_dim );
        l->setFont( juce::FontOptions( 12.5f ) );
        addAndMakeVisible( l );
    }
    for ( juce::Component *c : { static_cast<juce::Component *>( &m_start ), static_cast<juce::Component *>( &m_stop ),
                                 static_cast<juce::Component *>( &m_add ), static_cast<juce::Component *>( &m_save ),
                                 static_cast<juce::Component *>( &m_folder ) } )
        addAndMakeVisible( c );
    refreshJourneys();
    timerCallback();
    startTimerHz( 2 );
}

void JourneyView::refreshJourneys()
{
    m_entries = m_proc.journeys();
    const int keep = m_journeys.getSelectedId();
    m_journeys.clear( juce::dontSendNotification );
    const char *const Heads[] = { "Whole nights", "Nights", "Journeys", "Yours" };
    for ( int kind = 0; kind < 4; ++kind )
    {
        bool headed = false;
        for ( size_t i = 0; i < m_entries.size(); ++i )
        {
            const AdditivProcessor::JourneyEntry &j = m_entries[i];
            const int                             k = j.m_builtin < 0                        ? 3
                                                      : j.m_name.startsWith( "Whole Night" ) ? 0
                                                      : j.m_name.endsWith( " - Night" )      ? 1
                                                                                             : 2;
            if ( k != kind ) continue;
            if ( !headed )
            {
                m_journeys.addSectionHeading( Heads[kind] );
                headed = true;
            }
            m_journeys.addItem( j.m_name, static_cast<int>( i ) + 1 );
        }
    }
    if ( keep > 0 && keep <= static_cast<int>( m_entries.size() ) )
        m_journeys.setSelectedId( keep, juce::dontSendNotification );
}

void JourneyView::timerCallback()
{
    for ( int k = 0; k < 2; ++k )
    {
        const juce::String n = juce::String( k == 0 ? "A:  " : "B:  " ) + m_proc.slotName( k );
        if ( m_slots[k].m_name.getText() != n ) m_slots[k].m_name.setText( n, juce::dontSendNotification );
    }
    const juce::String st = m_proc.journeyStatus();
    m_status.setText( st.isEmpty() ? juce::String( "No journey under way." ) : st, juce::dontSendNotification );
    m_stop.setEnabled( st.isNotEmpty() );
    const int n = m_proc.journeyDraftSteps();
    m_draft.setText( n == 0 ? juce::String( "Writing a journey: + adds the preset playing, the disk saves it." )
                            : juce::String( "The journey being written: " ) + juce::String( n ) +
                                  ( n == 1 ? " preset" : " presets" ),
                     juce::dontSendNotification );
    m_save.setEnabled( n > 0 );
    if ( ++m_tick % 10 == 0 )
    {   // every five seconds: a journey written by hand in the folder appears
        size_t files = 0;
        for ( const auto &j : m_entries )
            if ( j.m_builtin < 0 ) ++files;
        if ( AdditivProcessor::journeyFolder().findChildFiles( juce::File::findFiles, false, "*.journey" ).size() !=
             static_cast<int>( files ) )
            refreshJourneys();
    }
}

void JourneyView::choosePreset( int slot )
{
    juce::PopupMenu bank;
    const auto     &shots = m_proc.snapshots();   // the Focus page's snapshots first: a morph between two of one's own
    if ( !shots.empty() )
    {
        bank.addSectionHeader( "Snapshots (Focus page)" );
        for ( size_t k = 0; k < shots.size(); ++k )
            bank.addItem( 100000 + static_cast<int>( k ), "Snapshot " + juce::String( static_cast<int>( k ) + 1 ) );
        bank.addSeparator();
    }
    for ( int g = 0; g < additiv::NumPresetGroups; ++g )
    {
        if ( g == 0 ) bank.addSectionHeader( "Sounds" );
        if ( g == additiv::NumSoundGroups ) bank.addSectionHeader( "Living Systems" );
        if ( g == additiv::FirstSleepGroup ) bank.addSectionHeader( "Sleep Concerts" );
        if ( g == additiv::FirstVisionGroup ) bank.addSectionHeader( "Visions" );
        if ( g == additiv::FirstSwarmGroup ) bank.addSectionHeader( "Swarms & Currents" );
        if ( g == additiv::FirstHallGroup ) bank.addSectionHeader( "Halls & Orbits" );
        if ( g == additiv::FirstLivesGroup ) bank.addSectionHeader( "Harmonic Lives" );
        if ( g == additiv::FirstDreamGroup ) bank.addSectionHeader( "Dreams & Phantoms" );
        if ( g == additiv::FirstWeatherGroup ) bank.addSectionHeader( "Weather & Growth" );
        if ( g == additiv::FirstRoomsGroup ) bank.addSectionHeader( "Rooms & Topology" );
        if ( g == additiv::FirstNightGroup ) bank.addSectionHeader( "Night & Illusions" );
        if ( g == additiv::FirstChorusGroup ) bank.addSectionHeader( "Chorus & Mirrors" );
        if ( g == additiv::FirstFloorGroup ) bank.addSectionHeader( "Deep Floors" );
        juce::PopupMenu group;
        for ( int i = 0; i < 64; ++i )
        {
            const int index = additiv::presetAt( g, i );
            if ( index < 0 ) break;
            group.addItem( index + 1, additiv::preset( index ).m_name );
        }
        bank.addSubMenu( additiv::PresetGroupNames[g], group );
    }
    juce::Component *target = &m_slots[slot & 1].m_fromPreset;
    bank.showMenuAsync( juce::PopupMenu::Options().withTargetComponent( target ),
                        [this, slot]( int id )
                        {
                            if ( id >= 100000 )
                                m_proc.snapshotToSlot( id - 100000, slot );
                            else if ( id > 0 )
                                m_proc.setSlotFromPreset( slot, id - 1 );
                        } );
}

void JourneyView::startChosen()
{
    const int id = m_journeys.getSelectedId();
    if ( id <= 0 || id > static_cast<int>( m_entries.size() ) )
    {
        m_journeys.showPopup();
        return;
    }
    if ( !m_proc.startJourney( m_entries[static_cast<size_t>( id - 1 )] ) )
        juce::AlertWindow::showMessageBoxAsync(
            juce::MessageBoxIconType::WarningIcon, "Journey",
            "This journey could not be read: a malformed line, no steps, or no preset the bank knows." );
}

void JourneyView::saveDraft()
{
    const juce::File dir = AdditivProcessor::journeyFolder();
    dir.createDirectory();
    m_chooser = std::make_unique<juce::FileChooser>( "Save the journey", dir.getChildFile( "My Journey.journey" ),
                                                     "*.journey" );
    m_chooser->launchAsync( juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting,
                            [this]( const juce::FileChooser &fc )
                            {
                                const juce::File f = fc.getResult();
                                if ( f == juce::File() ) return;
                                if ( m_proc.saveJourneyDraft( f.withFileExtension( "journey" ) ) )
                                    refreshJourneys();
                                else
                                    juce::AlertWindow::showMessageBoxAsync( juce::MessageBoxIconType::WarningIcon,
                                                                            "Journey",
                                                                            "The journey could not be written there." );
                            } );
}

void JourneyView::resized()
{
    const int w = getWidth();
    for ( int k = 0; k < 2; ++k )
    {
        const int y = 28 + k * 34;
        m_slots[k].m_name.setBounds( 14, y, w - 14 - 250 - 20, 26 );
        m_slots[k].m_fromSound.setBounds( w - 250 - 10, y, 120, 26 );
        m_slots[k].m_fromPreset.setBounds( w - 125 - 10, y, 120, 26 );
    }
    const int y1 = 28 + 2 * 34 + 30, y2 = y1 + 34;
    const int boxW = std::min( 360, w / 3 );
    m_journeys.setBounds( 14, y1, boxW, 26 );
    m_start.setBounds( 14 + boxW + 8, y1, 28, 26 );
    m_stop.setBounds( 14 + boxW + 40, y1, 28, 26 );
    m_status.setBounds( 14 + boxW + 78, y1, w - ( 14 + boxW + 78 ) - 10, 26 );
    m_add.setBounds( 14, y2, 130, 26 );
    m_save.setBounds( 152, y2, 28, 26 );
    m_folder.setBounds( 184, y2, 28, 26 );
    m_draft.setBounds( 220, y2, w - 230, 26 );
}

void JourneyView::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( getLocalBounds().toFloat(), skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( getLocalBounds().toFloat().reduced( 0.5f ), skin.m_radius, 1.0f );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.setColour( skin.family( frame::FamilyE::MOTION ) );
    g.drawText( "Morph slots: what A - B blends between, the matrix included", 18, 4, getWidth() - 30, 20,
                juce::Justification::centredLeft );
    g.setColour( skin.family( frame::FamilyE::SPACE ) );
    g.drawText( "Journeys: presets in a row, each held a while and glided into the next -- for a whole night", 18,
                28 + 2 * 34 + 6, getWidth() - 30, 20, juce::Justification::centredLeft );
}

juce::String JourneyView::describe() const
{
    juce::String s;
    s << "Morph slots:\n    A: " << m_proc.slotName( 0 ) << "\n    B: " << m_proc.slotName( 1 ) << "\n";
    const juce::String st = m_proc.journeyStatus();
    s << "Journey: " << ( st.isEmpty() ? juce::String( "none under way" ) : st ) << "\n";
    s << "\nA preset chosen glides from the sound before it over Glide (0: at once). With Morph on, the sound is the blend of the two "
         "slots at A - B: every number on its way, the filter fading from one into the other, a motion or a living landscape fading "
         "out where it changes and in again. A journey holds each of its presets for a time drawn from its range and glides into the "
         "next; the journeys that come with the instrument are made from the bank (Whole Night: about six and a half hours round). The timeline shows the journey chosen: a click on a step goes there. Below it, the one being written: drag a step's edge (how long it stays), its ramp (how long it glides in), the step itself (its place); right-click removes it; Edit a copy takes the journey above to change. A morph slot takes a snapshot of the Focus page too (Preset ..., first in the menu).\n";
    return s;
}

JourneyPage::JourneyPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live )
    : m_params( proc, actions,
                { { "Glide", frame::FamilyE::SPACE, { ParamIdE::PRESET_GLIDE } },
                  { "Morph",
                    frame::FamilyE::MOTION,
                    { ParamIdE::MORPH_ACTIVE, ParamIdE::MORPH_POS },
                    { ParamIdE::MORPH_ACTIVE } } },
                live ),
      m_view( proc ), m_timeline( proc, m_view )
{
    addAndMakeVisible( m_params );
    addAndMakeVisible( m_view );
    addAndMakeVisible( m_timeline );
}

void JourneyPage::resized()
{
    const int top = m_params.heightFor( getWidth() );
    m_params.setBounds( 0, 0, getWidth(), top );
    m_view.setBounds( 10, top, getWidth() - 20, JourneyView::idealHeight() );
    m_timeline.setBounds( 10, m_view.getBottom() + 10, getWidth() - 20,
                          std::max( TimelineMin, getHeight() - m_view.getBottom() - 20 ) );
}

// ========================================================================================================= timeline

JourneyTimeline::JourneyTimeline( AdditivProcessor &proc, const JourneyView &view ) : m_proc( proc ), m_view( view )
{
    m_copy.setTooltip(
        "Takes the journey above into the one being written, to edit here and save with the disk above" );
    m_copy.onClick = [this]
    {
        additiv::Journey j;
        if ( !upper( j ) ) return;
        j.m_name = j.m_name.empty() ? std::string( "My Journey" ) : j.m_name + " (mine)";
        m_proc.setJourneyDraft( j );
        repaint();
    };
    addAndMakeVisible( m_copy );
    startTimerHz( 2 );
}

void JourneyTimeline::timerCallback()
{
    if ( isShowing() ) repaint();
}

bool JourneyTimeline::upper( additiv::Journey &out ) const
{
    if ( m_proc.journeyPlayer().running() )
    {
        out = m_proc.journeyPlayer().journey();
        return true;
    }
    return m_view.chosen( out );
}

juce::Rectangle<float> JourneyTimeline::band( int which ) const
{
    auto r = getLocalBounds().toFloat().reduced( 18.0f, 0.0f ).withTrimmedTop( 28.0f ).withTrimmedBottom( 6.0f );
    if ( m_proc.journeyDraftSteps() == 0 ) return which == 0 ? r.withTrimmedBottom( 20.0f ) : juce::Rectangle<float>();
    const float h =
        ( r.getHeight() - 20.0f - 22.0f ) / 2.0f;   // two bands, the hours under the first, a title over the second
    return which == 0 ? r.withHeight( h ) : r.withTrimmedTop( h + 20.0f + 22.0f );
}

std::vector<JourneyTimeline::Block> JourneyTimeline::blocks( const additiv::Journey &j, juce::Rectangle<float> r )
{
    std::vector<Block> out;
    double             total = 0.0;
    for ( const additiv::JourneyStep &s : j.m_steps )
        total += 0.5 * ( s.m_fadeLo + s.m_fadeHi ) + 0.5 * ( s.m_dwellLo + s.m_dwellHi );
    if ( total <= 0.0 ) return out;
    double t = 0.0;
    for ( const additiv::JourneyStep &s : j.m_steps )
    {
        Block b;
        b.m_x0 = r.getX() + r.getWidth() * static_cast<float>( t / total );
        t += 0.5 * ( s.m_fadeLo + s.m_fadeHi );
        b.m_xg = r.getX() + r.getWidth() * static_cast<float>( t / total );
        t += 0.5 * ( s.m_dwellLo + s.m_dwellHi );
        b.m_x1 = r.getX() + r.getWidth() * static_cast<float>( t / total );
        out.push_back( b );
    }
    return out;
}

void JourneyTimeline::drawBand( juce::Graphics &g, const additiv::Journey &j, juce::Rectangle<float> r, int lit,
                                double done, bool faint ) const
{
    const frame::Skin &skin = additivui::skin();
    const juce::Colour hue = skin.family( frame::FamilyE::MOTION );
    const auto         bs = blocks( j, r );
    g.setFont( juce::FontOptions( 11.0f ) );
    for ( size_t i = 0; i < bs.size(); ++i )
    {
        const Block       &b = bs[i];
        const bool         now = static_cast<int>( i ) == lit;
        const juce::Colour c = now ? hue : ( faint ? skin.m_raised : skin.m_raised.brighter( 0.15f ) );
        juce::Path         ramp;   // the glide: rising into the step
        ramp.startNewSubPath( b.m_x0, r.getBottom() );
        ramp.lineTo( b.m_xg, r.getY() );
        ramp.lineTo( b.m_xg, r.getBottom() );
        ramp.closeSubPath();
        g.setColour( c.withAlpha( now ? 0.35f : 0.6f ) );
        g.fillPath( ramp );
        g.setColour( c.withAlpha( now ? 0.35f : 0.8f ) );
        g.fillRect( juce::Rectangle<float>::leftTopRightBottom( b.m_xg, r.getY(), b.m_x1 - 1.0f, r.getBottom() ) );
        if ( now )
        {
            g.setColour( hue.withAlpha( 0.85f ) );
            g.fillRect( juce::Rectangle<float>::leftTopRightBottom(
                b.m_xg, r.getY(), b.m_xg + ( b.m_x1 - 1.0f - b.m_xg ) * static_cast<float>( done ), r.getBottom() ) );
        }
        if ( static_cast<int>( i ) == m_dropAt && r == band( 1 ) )
        {   // where a moved block would land
            g.setColour( skin.m_accent );
            g.fillRect( b.m_x0 - 1.5f, r.getY() - 2.0f, 3.0f, r.getHeight() + 4.0f );
        }
        g.setColour( now ? skin.m_bg : skin.m_ink.withAlpha( faint ? 0.55f : 0.85f ) );
        g.drawFittedText( j.m_steps[i].m_preset,
                          juce::Rectangle<float>( b.m_xg + 4.0f, r.getY() + 2.0f,
                                                  std::max( 0.0f, b.m_x1 - b.m_xg - 8.0f ), r.getHeight() - 4.0f )
                              .toNearestInt(),
                          juce::Justification::centredLeft, 2, 0.7f );
    }
}

void JourneyTimeline::paint( juce::Graphics &g )
{
    const frame::Skin &skin = additivui::skin();
    const auto         b = getLocalBounds().toFloat();
    g.setColour( skin.m_group.withAlpha( 0.92f ) );
    g.fillRoundedRectangle( b, skin.m_radius );
    g.setColour( skin.m_edge );
    g.drawRoundedRectangle( b.reduced( 0.5f ), skin.m_radius, 1.0f );
    const juce::Colour            hue = skin.family( frame::FamilyE::MOTION );
    const additiv::JourneyPlayer &player = m_proc.journeyPlayer();
    const bool                    running = player.running();
    additiv::Journey              j;
    const bool                    any = upper( j );
    double                        total = 0.0;
    for ( const additiv::JourneyStep &s : j.m_steps )
        total += 0.5 * ( s.m_fadeLo + s.m_fadeHi ) + 0.5 * ( s.m_dwellLo + s.m_dwellHi );
    g.setFont( juce::FontOptions( 13.0f, juce::Font::bold ) );
    g.setColour( running ? hue : skin.m_dim );
    if ( !any )
    {
        g.drawText(
            "Choose a journey in the box above: its steps are drawn here over time, the one under way lit while it runs.",
            b.reduced( 18.0f, 4.0f ).withHeight( 20.0f ), juce::Justification::centredLeft );
    }
    else
    {
        juce::String head = juce::String( j.m_name ) + "   " + juce::String( static_cast<int>( j.m_steps.size() ) ) +
                            " steps, about " + juce::String( additiv::Journey::timeText( total ) ) +
                            ( j.m_cyclic ? ", then again" : "" );
        if ( running )
            head << "   -- step " << ( player.step() + 1 ) << ", "
                 << juce::String( additiv::Journey::timeText( player.remaining() ) )
                 << " left in it (click a step: there)";
        else
            head << "   (not under way: a click on a step starts it there)";
        g.drawText( head, b.reduced( 18.0f, 4.0f ).withHeight( 20.0f ).withTrimmedRight( 120.0f ),
                    juce::Justification::centredLeft );
        const auto r0 = band( 0 );
        double     done = 0.0;
        if ( running && player.step() >= 0 && player.step() < static_cast<int>( j.m_steps.size() ) )
        {
            const additiv::JourneyStep &s = j.m_steps[static_cast<size_t>( player.step() )];
            done = juce::jlimit( 0.0, 1.0,
                                 1.0 - player.remaining() / std::max( 0.5 * ( s.m_dwellLo + s.m_dwellHi ), 1.0 ) );
        }
        drawBand( g, j, r0, running ? player.step() : -1, done, !running );
        g.setColour( skin.m_dim );   // the hours under it
        g.setFont( juce::FontOptions( 11.0f ) );
        const double every = total > 4.0 * 3600.0 ? 3600.0 : ( total > 3600.0 ? 1800.0 : 600.0 );
        for ( double h = 0.0; total > 0.0 && h <= total; h += every )
        {
            const float x = r0.getX() + r0.getWidth() * static_cast<float>( h / total );
            g.drawVerticalLine( juce::roundToInt( x ), r0.getBottom(), r0.getBottom() + 4.0f );
            g.drawText( juce::String( additiv::Journey::timeText( h ) ),
                        juce::Rectangle<float>( x - 30.0f, r0.getBottom() + 4.0f, 60.0f, 14.0f ),
                        juce::Justification::centred );
        }
    }
    // The one being written, to edit.
    const additiv::Journey &d = m_proc.journeyDraft();
    if ( !d.m_steps.empty() )
    {
        const auto r1 = band( 1 );
        double     dt = 0.0;
        for ( const additiv::JourneyStep &s : d.m_steps )
            dt += 0.5 * ( s.m_fadeLo + s.m_fadeHi ) + 0.5 * ( s.m_dwellLo + s.m_dwellHi );
        g.setColour( skin.m_accent );
        g.setFont( juce::FontOptions( 12.5f, juce::Font::bold ) );
        g.drawText(
            "Being written: " + juce::String( d.m_name.empty() ? "your journey" : d.m_name ) + ", " +
                juce::String( static_cast<int>( d.m_steps.size() ) ) + " steps, about " +
                juce::String( additiv::Journey::timeText( dt ) ) +
                "   -- drag an edge: how long it stays; the ramp: how long it glides; the block: where it goes; right-click: remove",
            juce::Rectangle<float>( r1.getX(), r1.getY() - 21.0f, r1.getWidth(), 18.0f ),
            juce::Justification::centredLeft );
        drawBand( g, d, r1, -1, 0.0, false );
    }
}

void JourneyTimeline::resized()
{
    m_copy.setBounds( getWidth() - 118, 5, 104, 20 );
}

int JourneyTimeline::draftAt( juce::Point<float> p, DragE &drag ) const
{
    drag = DragE::NONE;
    const auto r = band( 1 );
    if ( r.isEmpty() || !r.expanded( 4.0f ).contains( p ) ) return -1;
    const auto bs = blocks( m_proc.journeyDraft(), r );
    for ( size_t i = 0; i < bs.size(); ++i )
    {
        const Block &b = bs[i];
        if ( p.x < b.m_x0 || p.x > b.m_x1 + 4.0f ) continue;
        if ( p.x >= b.m_x1 - 6.0f )
            drag = DragE::DWELL;
        else if ( p.x < b.m_xg )
            drag = DragE::GLIDE;
        else
            drag = DragE::MOVE;
        return static_cast<int>( i );
    }
    return -1;
}

void JourneyTimeline::mouseMove( const juce::MouseEvent &e )
{
    DragE d = DragE::NONE;
    draftAt( e.position, d );
    setMouseCursor( d == DragE::DWELL || d == DragE::GLIDE ? juce::MouseCursor::LeftRightResizeCursor
                    : d == DragE::MOVE                     ? juce::MouseCursor::DraggingHandCursor
                                                           : juce::MouseCursor::NormalCursor );
}

void JourneyTimeline::mouseDown( const juce::MouseEvent &e )
{
    // The journey above: a click on a step goes there (starting the one chosen there when none runs).
    const auto       r0 = band( 0 );
    additiv::Journey j;
    if ( r0.contains( e.position ) && upper( j ) )
    {
        const auto bs = blocks( j, r0 );
        for ( size_t i = 0; i < bs.size(); ++i )
            if ( e.position.x >= bs[i].m_x0 && e.position.x < bs[i].m_x1 )
            {
                if ( !m_proc.journeyPlayer().running() )
                {
                    for ( const auto &entry : m_proc.journeys() )
                        if ( entry.m_name.toStdString() == j.m_name ||
                             ( entry.m_builtin >= 0 && juce::String( j.m_name ) == entry.m_name ) )
                        {
                            m_proc.startJourney( entry );
                            break;
                        }
                }
                m_proc.skipJourneyTo( static_cast<int>( i ) );
                repaint();
                return;
            }
        return;
    }
    // The one being written.
    DragE     d = DragE::NONE;
    const int i = draftAt( e.position, d );
    if ( i < 0 ) return;
    if ( e.mods.isPopupMenu() )
    {
        juce::PopupMenu m;
        m.addItem( "Remove " + juce::String( m_proc.journeyDraft().m_steps[static_cast<size_t>( i )].m_preset ),
                   [this, i]
                   {
                       additiv::Journey dj = m_proc.journeyDraft();
                       dj.m_steps.erase( dj.m_steps.begin() + i );
                       m_proc.setJourneyDraft( dj );
                       repaint();
                   } );
        m.showMenuAsync( juce::PopupMenu::Options().withTargetComponent( this ).withTargetScreenArea(
            juce::Rectangle<int>( e.getScreenX(), e.getScreenY(), 1, 1 ) ) );
        return;
    }
    m_drag = d;
    m_dragStep = i;
    m_dragX = e.position.x;
    m_before = m_proc.journeyDraft().m_steps[static_cast<size_t>( i )];
    double total = 0.0;
    for ( const additiv::JourneyStep &s : m_proc.journeyDraft().m_steps )
        total += 0.5 * ( s.m_fadeLo + s.m_fadeHi ) + 0.5 * ( s.m_dwellLo + s.m_dwellHi );
    m_perPixel = total / std::max( 1.0f, band( 1 ).getWidth() );
    startTimerHz( 10 );
}

void JourneyTimeline::mouseDrag( const juce::MouseEvent &e )
{
    if ( m_drag == DragE::NONE || m_dragStep < 0 || m_dragStep >= m_proc.journeyDraftSteps() ) return;
    additiv::Journey      dj = m_proc.journeyDraft();
    additiv::JourneyStep &s = dj.m_steps[static_cast<size_t>( m_dragStep )];
    const double moved = ( e.position.x - m_dragX ) * m_perPixel;   // seconds, on the scale the drag began with
    if ( m_drag == DragE::DWELL || m_drag == DragE::GLIDE )
    {
        const double lo = m_drag == DragE::DWELL ? m_before.m_dwellLo : m_before.m_fadeLo,
                     hi = m_drag == DragE::DWELL ? m_before.m_dwellHi : m_before.m_fadeHi;
        const double mean = 0.5 * ( lo + hi ), least = m_drag == DragE::DWELL ? 30.0 : 0.0;
        const double now = std::max(
            least, mean + ( m_drag == DragE::GLIDE ? -moved : moved ) );   // the ramp: dragged left, a longer glide
        const double k = mean > 0.0 ? now / mean : 1.0;                    // the span kept in proportion
        const double newLo = mean > 0.0 ? lo * k : now, newHi = mean > 0.0 ? hi * k : now;
        if ( m_drag == DragE::DWELL )
        {
            s.m_dwellLo = newLo;
            s.m_dwellHi = newHi;
        }
        else
        {
            s.m_fadeLo = newLo;
            s.m_fadeHi = newHi;
        }
        m_proc.setJourneyDraft( dj );
    }
    else
    {   // the block: where it would land
        DragE ignored = DragE::NONE;
        m_dropAt = draftAt( { e.position.x, band( 1 ).getCentreY() }, ignored );
    }
    repaint();
}

void JourneyTimeline::mouseUp( const juce::MouseEvent & )
{
    if ( m_drag == DragE::MOVE && m_dropAt >= 0 && m_dropAt != m_dragStep && m_dragStep < m_proc.journeyDraftSteps() )
    {
        additiv::Journey           dj = m_proc.journeyDraft();
        const additiv::JourneyStep s = dj.m_steps[static_cast<size_t>( m_dragStep )];
        dj.m_steps.erase( dj.m_steps.begin() + m_dragStep );
        dj.m_steps.insert( dj.m_steps.begin() + std::min( m_dropAt, static_cast<int>( dj.m_steps.size() ) ), s );
        m_proc.setJourneyDraft( dj );
    }
    m_drag = DragE::NONE;
    m_dragStep = -1;
    m_dropAt = -1;
    startTimerHz( 2 );
    repaint();
}
