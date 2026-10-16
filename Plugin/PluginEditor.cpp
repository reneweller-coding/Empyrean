/**
 * @file PluginEditor.cpp
 * @brief The panel (PluginEditor.h).
 * @author Rene Weller
 */
#include "PluginEditor.h"

#include "AdditivData.h"
#include "JourneyView.h"
#include "FocusView.h"
#include "HallView.h"
#include "KnobMod.h"
#include "MapView.h"
#include "SourcesView.h"
#include "SpaceView.h"
#include "TuningView.h"

#include "additiv/Find.h"
#include "additiv/Landscape.h"
#include "additiv/Presets.h"

using additiv::ParamIdE;

namespace
{

constexpr const char *App = "Empyrean";   ///< the settings' name (application data/\<name\>)

/** @brief A page in a viewport: it scrolls only where the window is smaller than the page. */
class ScrollPage final : public juce::Viewport
{
public:
    /**
     * @brief Holds @p page, which needs @p heightFor at a width.
     * @param[in,out] page       the page (owned)
     * @param[in]     heightFor  its height at a width
     */
    ScrollPage( juce::Component *page, std::function<int( int )> heightFor )
        : m_page( page ), m_heightFor( std::move( heightFor ) )
    {
        setViewedComponent( page, true );
        setScrollBarsShown( true, false );
    }
    /** @brief The page as wide as the viewport and as high as it needs. */
    void resized() override
    {
        juce::Viewport::resized();
        const int w = getMaximumVisibleWidth();
        m_page->setSize( w, std::max( m_heightFor( w ), getMaximumVisibleHeight() ) );
    }
    /** @brief The height the page needs at the viewport's width. @return pixels */
    int pageHeight() const { return m_heightFor( getMaximumVisibleWidth() ); }
    /**
     * @brief The height the page needs at width @p w.
     * @param[in] w  pixels
     * @return pixels
     */
    int heightAt( int w ) const { return m_heightFor( w ); }

private:
    juce::Component          *m_page;        ///< the page (owned by the viewport)
    std::function<int( int )> m_heightFor;   ///< its height at a width
};

/**
 * @brief The Modulation page: the sixteen rows in a column, every source live beside them (below two columns of rows
 *        where the page is narrow). A click on a source opens it in the inspector -- its own knobs (an LFO's shape and
 *        rate, a macro, the fields' and the reservoir's) live there.
 */
class MatrixPage final : public juce::Component
{
public:
    /**
     * @brief Builds the page.
     * @param[in,out] proc    the processor
     * @param[in]     choose  a source clicked
     */
    MatrixPage( AdditivProcessor &proc, std::function<void( additiv::ModSourceE )> choose )
        : m_proc( proc ), m_sources( proc ), m_matrix( proc )
    {
        m_sources.m_onChoose = std::move( choose );
        addAndMakeVisible( m_matrix );
        addAndMakeVisible( m_sources );
    }
    /**
     * @brief The source the inspector shows, marked among the traces.
     * @param[in] s  the source
     */
    void setChosen( additiv::ModSourceE s ) { m_sources.setChosen( s ); }
    /**
     * @brief The height the page needs at a width.
     * @param[in] w  pixels
     * @return pixels
     */
    int heightFor( int w ) const
    {
        if ( beside( w ) )
            return 10 + std::max( MatrixView::heightFor( 1 ), SourcesView::heightFor( w - 30 - MatrixW ) ) + 10;
        return 10 + MatrixView::heightFor( 2 ) + 10 + SourcesView::heightFor( w - 20 ) + 10;
    }
    /** @brief The rows, and the sources beside or below them. */
    void resized() override
    {
        const int w = getWidth(), h = getHeight();
        if ( beside( w ) )
        {
            m_matrix.setBounds( 10, 10, MatrixW, h - 20 );
            m_sources.setBounds( 20 + MatrixW, 10, w - 30 - MatrixW, h - 20 );
        }
        else
        {
            m_matrix.setBounds( 10, 10, w - 20, MatrixView::heightFor( 2 ) );
            m_sources.setBounds( 10, m_matrix.getBottom() + 10, w - 20,
                                 std::max( SourcesView::heightFor( w - 20 ), h - m_matrix.getBottom() - 20 ) );
        }
    }
    /** @brief The page in words: the rows, and the modulators' knobs. @return the text */
    juce::String describe() const
    {
        juce::String s = m_matrix.describe() + "\nThe modulators (their knobs are in the inspector: click a source):\n";
        for ( const additiv::ParamDesc &d : additiv::paramTable() )
        {
            if ( juce::String( d.m_section ) != "Modulation" ) continue;
            s << "    " << d.m_name << " (" << d.m_key << "): now "
              << juce::String( m_proc.value( static_cast<int>( d.m_id ) ), 3 ) << "\n";
        }
        return s;
    }

private:
    static constexpr int MatrixW = 600;   ///< the rows' column, beside the sources
    AdditivProcessor    &m_proc;          ///< whose
    /**
     * @brief The sources fit beside the rows' column.
     * @param[in] w  the page's width
     * @return true then
     */
    static bool beside( int w ) { return w - 30 - MatrixW >= 440; }
    SourcesView m_sources;   ///< every source, live
    MatrixView  m_matrix;    ///< the rows
};

/** @brief A short way to list ParamIds. */
using IdsT = std::vector<ParamIdE>;

}   // namespace

static void findParamPages( juce::Component *c, std::vector<ParamPage *> &out );

AdditivEditor::AdditivEditor( AdditivProcessor &proc ) : juce::AudioProcessorEditor( proc ), m_proc( proc )
{
    setLookAndFeel( &m_lnf );
    m_actions.m_learn = [this]( int id ) { m_proc.learn( id ); };
    m_actions.m_controllerFor = [this]( int id ) { return m_proc.controllerFor( id ); };
    m_actions.m_forget = [this]( int id ) { m_proc.forget( id ); };
    m_actions.m_reset = [this]( int id )
    {
        m_history.begin( "Default", values(), matrixText(), true );
        m_proc.resetToDefault( id );
        endStep();
    };
    m_actions.m_more = [this]( juce::PopupMenu &m, int id )
    {
        // Back to where the preset had it, when it has moved.
        const auto &ref = m_proc.reference();
        if ( id >= 0 && id < additiv::NumParams && ref.size() == static_cast<size_t>( additiv::NumParams ) &&
             !additiv::keptByPresets( static_cast<ParamIdE>( id ) ) &&
             m_proc.value( id ) != ref[static_cast<size_t>( id )] )
        {
            const additiv::ParamDesc &d = additiv::paramDesc( static_cast<ParamIdE>( id ) );
            m.addItem( "Back to the preset's value (" + juce::String( ref[static_cast<size_t>( id )], 3 ) +
                           ( d.m_unit[0] != 0 ? juce::String( " " ) + d.m_unit : juce::String() ) + ")",
                       [this, id]
                       {
                           m_history.begin( "Back to the preset", values(), matrixText(), true );
                           m_proc.revertKnob( id );
                           endStep();
                       } );
        }
        knobmod::addMenuItems( m_proc, m, id );
    };
    // Alt-drag on a knob: the source the inspector shows modulates it, its depth set by the drag (up: more).
    m_actions.m_altDrag = [this]( int id, const juce::MouseEvent &e, int phase )
    {
        const additiv::ModSourceE s = m_inspector != nullptr ? m_inspector->source() : additiv::ModSourceE::OFF;
        if ( phase == 0 )
        {
            if ( s == additiv::ModSourceE::OFF || !knobmod::canTarget( id ) ) return false;
            m_altRow = knobmod::addRow( m_proc, s, id, 0.0f );
            if ( m_altRow < 0 ) return false;
            m_altDepth = m_proc.engine().modRow( m_altRow ).m_depth;
            return true;
        }
        if ( m_altRow < 0 ) return false;
        additiv::ModRow r = m_proc.engine().modRow( m_altRow );
        r.m_depth =
            juce::jlimit( -1.0f, 1.0f, m_altDepth - static_cast<float>( e.getDistanceFromDragStartY() ) / 200.0f );
        m_proc.engine().setModRow( m_altRow, r );
        if ( m_info != nullptr )
            m_info->showText( juce::String( additiv::ModSourceNames[static_cast<int>( s )] ) + " -> " +
                              additiv::paramDesc( static_cast<ParamIdE>( id ) ).m_name + "   depth " +
                              ( r.m_depth >= 0.0f ? "+" : "" ) + juce::String( r.m_depth, 2 ) +
                              "   (Alt-drag: up more, down less or the other way)" );
        if ( phase == 2 )
        {
            if ( std::fabs( r.m_depth ) < 0.005f )
                m_proc.engine().setModRow( m_altRow, additiv::ModRow{} );   // dragged back to nothing: no row
            chooseSource( s, false );
            m_altRow = -1;
        }
        return true;
    };
    m_actions.m_describe = []( int id )
    {
        const additiv::ParamDesc &d = additiv::paramDesc( static_cast<ParamIdE>( id ) );
        return juce::String( d.m_name ) + "  (" + d.m_key + ")";
    };

    addAndMakeVisible( m_title );
    addAndMakeVisible( m_play );
    m_play.onClick = [this]
    {
        m_history.begin( "Autoplay", values(), matrixText(), true );
        m_proc.setValue( static_cast<int>( ParamIdE::BRAIN_ON ),
                         m_proc.value( static_cast<int>( ParamIdE::BRAIN_ON ) ) >= 0.5f ? 0.0f : 1.0f );
        endStep();
    };
    for ( auto *b : { &m_prevPreset, &m_presetName, &m_nextPreset } ) addAndMakeVisible( b );
    m_prevPreset.setTooltip( "The preset before" );
    m_nextPreset.setTooltip( "The preset after" );
    m_presetName.setTooltip(
        "The preset playing: click to browse the bank (96 groups of 64: the sounds, the living systems, the sleep concerts, the visions, the swarms and currents, the halls and orbits)" );
    m_prevPreset.onClick = [this]
    { loadPreset( ( std::max( m_proc.currentPreset(), 0 ) + additiv::numPresets() - 1 ) % additiv::numPresets() ); };
    m_nextPreset.onClick = [this] { loadPreset( ( m_proc.currentPreset() + 1 ) % additiv::numPresets() ); };
    m_presetName.onClick = [this] { m_browser.open(); };
    addAndMakeVisible( m_step );
    m_step.setTooltip( "Autoplay: exchange one voice of the chord now" );
    m_step.onClick = [this] { m_proc.engine().autoplayStep(); };
    bindChoice( m_mode, ParamIdE::AUTO_MODE );
    bindChoice( m_scale, ParamIdE::SCALE );
    bindChoice( m_root, ParamIdE::ROOT_NOTE );
    bindChoice( m_motion, ParamIdE::MOTION );
    bindChoice( m_landscape, ParamIdE::LANDSCAPE );
    m_mode.setTooltip(
        "Autoplay: Free lets notes come and go; Chords keeps the chord full and exchanges one voice at a time" );
    m_motion.setTooltip( "How every partial's envelope moves" );
    m_landscape.setTooltip( "The landscape the Aurora motion reads" );
    m_status.setColour( juce::Label::textColourId, additivui::skin().m_dim );
    m_status.setFont( juce::FontOptions( 12.0f ) );
    addAndMakeVisible( m_status );
    for ( auto *b : { &m_undo, &m_redo, &m_help, &m_settings } ) addAndMakeVisible( b );
    addAndMakeVisible( m_save );
    m_save.onClick = [this] { showSaveMenu(); };
    m_undo.onClick = [this]
    {
        flushMatrix();
        apply( m_history.undo(), false );
    };
    m_redo.onClick = [this]
    {
        flushMatrix();
        apply( m_history.redo(), true );
    };
    m_help.onClick = [this] { showHelp( m_helpView == nullptr || !m_helpView->isVisible() ); };
    m_settings.onClick = [this] { showSettings(); };
    addAndMakeVisible( m_cloud );
    addAndMakeVisible( m_canvasButton );
    addAndMakeVisible( m_searchButton );
    m_searchButton.onClick = [this] { openPalette(); };
    addAndMakeVisible( m_compare );
    m_compare.onStateChange = [this] { m_proc.compareWithPreset( m_compare.isDown() && m_compare.isEnabled() ); };
    addAndMakeVisible( m_midi );
    m_midi.setClickingTogglesState( true );
    m_midi.onClick = [this]
    {
        ParamPage::M_ShowMidi = m_midi.getToggleState();
        m_tabs.repaint();
        for ( int t = 0; t < m_tabs.getNumTabs(); ++t )
            if ( auto *view = dynamic_cast<juce::Viewport *>( m_tabs.getTabContentComponent( t ) );
                 view != nullptr && view->getViewedComponent() != nullptr )
                view->getViewedComponent()->repaint();
    };
    addAndMakeVisible( m_meter );
    addChildComponent( m_palette );
    addChildComponent( m_browser );
    m_browser.m_onChoose = [this]( int factory, int user )
    {
        if ( factory >= 0 )
            loadPreset( factory );
        else if ( user >= 0 )
        {
            m_history.begin( "Preset", values(), matrixText(), true );
            m_proc.chooseUserPreset( user );
            endStep();
        }
    };
    m_info = std::make_unique<InfoLine>( m_proc, *this );
    addAndMakeVisible( *m_info );
    addAndMakeVisible( m_nightButton );
    m_nightButton.setClickingTogglesState( false );
    m_nightButton.onClick = [this] { setNight( !m_night ); };
    m_activity.m_last = &m_lastActivity;
    m_lastActivity = juce::Time::getMillisecondCounterHiRes();
    addMouseListener( &m_activity, true );
    m_night = m_proc.userSettings().getBoolValue( "night", false );
    m_nightButton.setToggleState( m_night, juce::dontSendNotification );
    addChildComponent( m_tour );
    m_veil.setInterceptsMouseClicks( false, false );
    addChildComponent( m_veil );   // over the window, then the overlays over it
    m_palette.toFront( false );
    m_browser.toFront( false );
    m_tour.toFront( false );
    m_tour.m_onDone = [this]
    {
        m_proc.userSettings().setValue( "tourDone", true );
        m_proc.userSettings().saveIfNeeded();
    };
    m_canvasButton.onClick = [this] { setCanvas( !m_canvas ); };
    m_cloud.m_onDoubleClick = [this] { setCanvas( !m_canvas ); };

    // The pages: what belongs together sits in one box, coloured by its family. The tabs join the
    // editor first, so every page's controls are made under the frame's look and feel.
    addAndMakeVisible( m_tabs );
    using F = frame::FamilyE;
    auto page = [this]( const juce::String &name, std::vector<ParamPage::Group> groups )
    {
        auto *p = new ParamPage( m_proc, m_actions, std::move( groups ), &m_live );
        m_describe.push_back( [p] { return p->describe(); } );
        m_tabs.addTab( name, additivui::skin().m_panel, new ScrollPage( p, [p]( int w ) { return p->heightFor( w ); } ),
                       true );
    };
    {   // the simple ways in, first
        auto *f = new FocusPage(
            m_proc, m_actions, &m_live,
            [this]( const juce::String &name, std::function<void()> change )
            {
                m_history.begin( name, values(), matrixText(), true );
                change();
                endStep();
            },
            [this] { return m_cloud.thumbnail( 160, 60 ); } );
        m_describe.push_back( [f] { return f->describe(); } );
        m_tabs.addTab( "Focus", additivui::skin().m_panel,
                       new ScrollPage( f, [f]( int w ) { return f->heightFor( w ); } ), true );
    }
    page( "Cloud & Filter",
          { { "Spectrum", F::SOURCE,
              IdsT{ ParamIdE::PARTIALS, ParamIdE::TILT, ParamIdE::BRIGHTNESS, ParamIdE::ODD_EVEN, ParamIdE::INHARMONIC,
                    ParamIdE::STRETCH, ParamIdE::SHIFT } },
            { "Cloud", F::SOURCE,
              IdsT{ ParamIdE::CLOUD_WIDTH, ParamIdE::WIDTH_SLOPE, ParamIdE::DRIFT, ParamIdE::DRIFT_RATE,
                    ParamIdE::SHIMMER, ParamIdE::SHIMMER_RATE, ParamIdE::STEREO } },
            { "Air", F::SOURCE, IdsT{ ParamIdE::AIR, ParamIdE::AIR_CENTER, ParamIdE::AIR_WIDTH, ParamIdE::AIR_JITTER },
              IdsT{ ParamIdE::AIR } },
            { "Turbulence", F::SOURCE, IdsT{ ParamIdE::TURBULENCE, ParamIdE::TURBULENCE_TIME },
              IdsT{ ParamIdE::TURBULENCE } },
            { "Filter", F::FILTER,
              IdsT{ ParamIdE::FILTER_MODEL, ParamIdE::CUTOFF, ParamIdE::RESONANCE, ParamIdE::FILTER_ENV,
                    ParamIdE::KEY_TRACK, ParamIdE::FILTER_DRIFT },
              IdsT{ ParamIdE::FILTER_MODEL } },
            { "Spectral Comb", F::FILTER,
              IdsT{ ParamIdE::COMB_DEPTH, ParamIdE::COMB_DENSITY, ParamIdE::COMB_SWEEP, ParamIdE::COMB_STEREO },
              IdsT{ ParamIdE::COMB_DEPTH } },
            { "Envelope", F::ENVELOPE,
              IdsT{ ParamIdE::ATTACK, ParamIdE::DECAY, ParamIdE::SUSTAIN, ParamIdE::RELEASE } } } );
    page( "Motion",
          { { "Motion", F::MOTION,
              IdsT{ ParamIdE::MOTION, ParamIdE::MOTION_RATE, ParamIdE::MOTION_DEPTH, ParamIdE::LANDSCAPE },
              IdsT{ ParamIdE::MOTION } },
            { "Field", F::MOTION,
              IdsT{ ParamIdE::FIELD_MODE, ParamIdE::FIELD_STRENGTH, ParamIdE::FIELD_RADIUS, ParamIdE::FIELD_SPRING,
                    ParamIdE::FIELD_RATE, ParamIdE::FIELD_COUNT, ParamIdE::PHYSARUM_SHARE, ParamIdE::PHYSARUM_TRAIL },
              IdsT{ ParamIdE::FIELD_MODE } },
            { "Time", F::ENVELOPE, IdsT{ ParamIdE::DISPERSION, ParamIdE::TIME_BEND },
              IdsT{ ParamIdE::DISPERSION, ParamIdE::TIME_BEND } },
            { "Living Landscape", F::SOURCE,
              IdsT{ ParamIdE::LIFE_MODE, ParamIdE::LIFE_DEPTH, ParamIdE::LIFE_RATE, ParamIdE::LIFE_CHARACTER,
                    ParamIdE::LIFE_ZOOM, ParamIdE::LIFE_SIDES, ParamIdE::LIFE_TURN },
              IdsT{ ParamIdE::LIFE_MODE } },
            { "Partial Sync", F::ENVELOPE,
              IdsT{ ParamIdE::SYNC_DEPTH, ParamIdE::SYNC_MODE, ParamIdE::SYNC_COUPLING, ParamIdE::SYNC_RATE,
                    ParamIdE::SYNC_SPREAD, ParamIdE::SYNC_SWARM, ParamIdE::SYNC_REACH, ParamIdE::SYNC_LAG },
              IdsT{ ParamIdE::SYNC_DEPTH } },
            { "Swarmalators", F::ENVELOPE,
              IdsT{ ParamIdE::SWARMAL, ParamIdE::SWARMAL_GATHER, ParamIdE::SWARMAL_SYNC, ParamIdE::SWARMAL_RATE,
                    ParamIdE::SWARMAL_SPEED, ParamIdE::SWARMAL_SPREAD },
              IdsT{ ParamIdE::SWARMAL } } } );
    // Phase 26: the harmonics' own lives, the horizon and the grains (and Phase 27's Talbot timbre; the ripple and its
    // breathers, from the Motion page, Phase 28).
    page( "Harmonics",
          { { "Overblow", F::SOURCE, IdsT{ ParamIdE::OVERBLOW, ParamIdE::OVERBLOW_TIME }, IdsT{ ParamIdE::OVERBLOW } },
            { "Subharmonics", F::SOURCE, IdsT{ ParamIdE::SUBHARMONICS, ParamIdE::SUB_TIME },
              IdsT{ ParamIdE::SUBHARMONICS } },
            { "Horizon", F::SOURCE, IdsT{ ParamIdE::HORIZON, ParamIdE::HORIZON_HEIGHT }, IdsT{ ParamIdE::HORIZON } },
            { "Seasons", F::ENVELOPE, IdsT{ ParamIdE::SEASONS, ParamIdE::SEASON_TIME }, IdsT{ ParamIdE::SEASONS } },
            { "Avalanche", F::MOTION, IdsT{ ParamIdE::AVALANCHE, ParamIdE::AVALANCHE_RATE, ParamIdE::AVALANCHE_GLOW },
              IdsT{ ParamIdE::AVALANCHE } },
            { "Grain", F::SOURCE, IdsT{ ParamIdE::GRAIN, ParamIdE::GRAIN_LENGTH, ParamIdE::GRAIN_DENSITY },
              IdsT{ ParamIdE::GRAIN } },
            { "Talbot", F::ENVELOPE, IdsT{ ParamIdE::TALBOT, ParamIdE::TALBOT_MODE, ParamIdE::TALBOT_TIME },
              IdsT{ ParamIdE::TALBOT } },
            { "Ripple", F::SPACE,
              IdsT{ ParamIdE::RIPPLE_DEPTH, ParamIdE::RIPPLE_SPEED, ParamIdE::RIPPLE_DECAY, ParamIdE::RIPPLE_BEND,
                    ParamIdE::RIPPLE_BREATHERS },
              IdsT{ ParamIdE::RIPPLE_DEPTH } } } );
    // Phase 27: layers that are not notes -- the phantom layer, the dream -- and the premonition; the memory's ghosts
    // (from the Effects page, Phase 28).
    page( "Dreams",
          { { "Phantom", F::FILTER, IdsT{ ParamIdE::PHANTOM, ParamIdE::PHANTOM_LINE, ParamIdE::PHANTOM_WIDTH },
              IdsT{ ParamIdE::PHANTOM } },
            { "Dream", F::ENVELOPE,
              IdsT{ ParamIdE::DREAM, ParamIdE::DREAM_FOCUS, ParamIdE::DREAM_DRIFT, ParamIdE::DREAM_DEPTH,
                    ParamIdE::DREAM_JOURNAL },
              IdsT{ ParamIdE::DREAM } },
            { "Premonition", F::MOTION, IdsT{ ParamIdE::PREMONITION, ParamIdE::PREMONITION_TIME },
              IdsT{ ParamIdE::PREMONITION } },
            { "Memory", F::ENVELOPE,
              IdsT{ ParamIdE::MEMORY_LEVEL, ParamIdE::MEMORY_TIME, ParamIdE::MEMORY_BLOOM, ParamIdE::MEMORY_BLUR,
                    ParamIdE::MEMORY_SINK },
              IdsT{ ParamIdE::MEMORY_LEVEL } },
            { "Streams", F::SOURCE,
              IdsT{ ParamIdE::STREAM_DEPTH, ParamIdE::STREAMS, ParamIdE::STREAM_RATE, ParamIdE::FUSION },
              IdsT{ ParamIdE::STREAM_DEPTH } },
            { "Risset", F::MOTION,
              IdsT{ ParamIdE::RISSET, ParamIdE::RISSET_RATE, ParamIdE::RISSET_TIME, ParamIdE::RISSET_WAY },
              IdsT{ ParamIdE::RISSET } } } );
    // Phase 28: the wind and what it moves -- the chimes (Phase 27), the rain, the leaves, the sea.
    page( "Weather",
          { { "Wind", F::SPACE, IdsT{ ParamIdE::WIND, ParamIdE::GUSTS } },
            { "Chimes", F::SPACE,
              IdsT{ ParamIdE::CHIMES, ParamIdE::CHIME_KIND, ParamIdE::CHIME_PITCH, ParamIdE::CHIME_DECAY },
              IdsT{ ParamIdE::CHIMES } },
            { "Rain", F::FILTER,
              IdsT{ ParamIdE::RAIN, ParamIdE::RAIN_RATE, ParamIdE::RAIN_SURFACE, ParamIdE::RAIN_SHOWERS,
                    ParamIdE::RAIN_TUNE },
              IdsT{ ParamIdE::RAIN } },
            { "Leaves", F::SOURCE, IdsT{ ParamIdE::LEAVES, ParamIdE::FOLIAGE }, IdsT{ ParamIdE::LEAVES } },
            { "Surf", F::MOTION, IdsT{ ParamIdE::SURF, ParamIdE::SURF_SWELL, ParamIdE::SHORE, ParamIdE::SURF_DISTANCE },
              IdsT{ ParamIdE::SURF } } } );
    // Phase 30: the night -- the Aeolian harp in the wind, thunder, the singing ice, the night arc from the real sun.
    page( "Night",
          { { "Aeolian Harp", F::SOURCE, IdsT{ ParamIdE::HARP, ParamIdE::HARP_PITCH, ParamIdE::HARP_GAUGE },
              IdsT{ ParamIdE::HARP } },
            { "Thunder", F::SPACE, IdsT{ ParamIdE::THUNDER, ParamIdE::THUNDER_DISTANCE, ParamIdE::THUNDER_RUMBLE },
              IdsT{ ParamIdE::THUNDER } },
            { "Singing Ice", F::FILTER, IdsT{ ParamIdE::ICE, ParamIdE::ICE_THICKNESS, ParamIdE::ICE_COLD },
              IdsT{ ParamIdE::ICE } },
            { "Night Chorus", F::MOTION,
              IdsT{ ParamIdE::CHORUS, ParamIdE::CHORUS_KIND, ParamIdE::CHORUS_WARMTH, ParamIdE::CHORUS_SYNC,
                    ParamIdE::CHORUS_TUNE },
              IdsT{ ParamIdE::CHORUS } },
            { "Night", F::ENVELOPE, IdsT{ ParamIdE::NIGHT_ARC }, IdsT{ ParamIdE::NIGHT_ARC } } } );
    {   // the tuning, with its picture: the octave over the root, the scale, the voices and Self-Tune's pull
        auto *tp = new TuningPage( m_proc, m_actions, &m_live );
        m_describe.push_back( [tp] { return tp->describe(); } );
        m_tabs.addTab( "Tuning", additivui::skin().m_panel,
                       new ScrollPage( tp, [tp]( int w ) { return tp->heightFor( w ); } ), true );
    }
    page( "Conductor",
          { { "Conductor",
              F::MOTION,
              IdsT{ ParamIdE::BRAIN_ON, ParamIdE::AUTO_MODE, ParamIdE::BRAIN_RATE, ParamIdE::AUTO_RATE,
                    ParamIdE::BRAIN_DENSITY, ParamIdE::BRAIN_HOLD_MIN, ParamIdE::BRAIN_HOLD_MAX },
              IdsT{ ParamIdE::BRAIN_ON },
              { { ParamIdE::BRAIN_HOLD_MIN, ParamIdE::BRAIN_HOLD_MAX } } },
            { "Register",
              F::MOTION,
              IdsT{ ParamIdE::BRAIN_LOW, ParamIdE::BRAIN_HIGH },
              {},
              { { ParamIdE::BRAIN_LOW, ParamIdE::BRAIN_HIGH } } },
            { "Floor", F::ENVELOPE,
              IdsT{ ParamIdE::BRAIN_FOUNDATION, ParamIdE::FLOOR, ParamIdE::PEDAL, ParamIdE::PRESSURE,
                    ParamIdE::PRESSURE_RATE },
              IdsT{ ParamIdE::BRAIN_FOUNDATION } },
            { "Sub & Fifth", F::ENVELOPE, IdsT{ ParamIdE::FLOOR_SUB, ParamIdE::FLOOR_FIFTH, ParamIdE::FLOOR_BREATH } },
            { "Harmony", F::MOTION,
              IdsT{ ParamIdE::BRAIN_CONSONANCE, ParamIdE::BRAIN_WANDER, ParamIdE::AUTO_LEAD, ParamIdE::AUTO_TENSION,
                    ParamIdE::AUTO_ROOT_MOVE, ParamIdE::BRAIN_TIMBRE, ParamIdE::BRAIN_CASCADE, ParamIdE::BRAIN_LIGHT,
                    ParamIdE::AUTO_STEP } },
            { "Brain 2", F::ENVELOPE,
              IdsT{ ParamIdE::BRAIN2_ON, ParamIdE::BRAIN2_INTERVAL, ParamIdE::BRAIN2_DENSITY, ParamIdE::BRAIN2_RATE,
                    ParamIdE::BRAIN2_DEPTH },
              IdsT{ ParamIdE::BRAIN2_ON } },
            { "MPE", F::FILTER,
              IdsT{ ParamIdE::MPE_ON, ParamIdE::MPE_BEND, ParamIdE::MPE_PRESS, ParamIdE::MPE_PRESS_DEPTH,
                    ParamIdE::MPE_SLIDE, ParamIdE::MPE_SLIDE_DEPTH },
              IdsT{ ParamIdE::MPE_ON } },
            { "Quantum Walk", F::ENVELOPE, IdsT{ ParamIdE::QUANTUM, ParamIdE::QUANTUM_TIME, ParamIdE::QUANTUM_SPREAD },
              IdsT{ ParamIdE::QUANTUM } } } );
    {
        auto *s = new SpacePage( m_proc, m_actions, &m_live );
        m_describe.push_back( [s] { return s->describe(); } );
        m_tabs.addTab( "Space", additivui::skin().m_panel,
                       new ScrollPage( s, [s]( int w ) { return s->heightFor( w ); } ), true );
    }
    page( "Effects", { { "Echo",
                         F::SPACE,
                         IdsT{ ParamIdE::DELAY_MIX, ParamIdE::DELAY_TIME_L, ParamIdE::DELAY_TIME_R,
                               ParamIdE::DELAY_FEEDBACK, ParamIdE::DELAY_CROSS, ParamIdE::DELAY_DAMP,
                               ParamIdE::DELAY_ABSORB, ParamIdE::DELAY_DUCK, ParamIdE::DELAY_TO_FAR },
                         IdsT{ ParamIdE::DELAY_MIX },
                         {},
                         { { ParamIdE::DELAY_TIME_L, ParamIdE::DELAY_TIME_R } } },
                       { "Echo 2",
                         F::SPACE,
                         IdsT{ ParamIdE::DELAY2_MIX, ParamIdE::DELAY2_TIME_L, ParamIdE::DELAY2_TIME_R,
                               ParamIdE::DELAY2_FEEDBACK, ParamIdE::DELAY2_DAMP, ParamIdE::DELAY2_TO_FAR },
                         IdsT{ ParamIdE::DELAY2_MIX },
                         {},
                         { { ParamIdE::DELAY2_TIME_L, ParamIdE::DELAY2_TIME_R } } },
                       { "Room", F::ENVELOPE,
                         IdsT{ ParamIdE::EARLY_LEVEL, ParamIdE::EARLY_SIZE, ParamIdE::EARLY_ABSORB, ParamIdE::NEAR_MIX,
                               ParamIdE::NEAR_DECAY, ParamIdE::NEAR_DAMP },
                         IdsT{ ParamIdE::EARLY_LEVEL, ParamIdE::NEAR_MIX } },
                       { "Spectral Echo", F::SPACE,
                         IdsT{ ParamIdE::SPEC_MIX, ParamIdE::SPEC_TIME, ParamIdE::SPEC_TILT, ParamIdE::SPEC_SCATTER,
                               ParamIdE::SPEC_FEEDBACK, ParamIdE::SPEC_DAMP, ParamIdE::SPEC_TO_FAR },
                         IdsT{ ParamIdE::SPEC_MIX } },
                       { "Melt", F::MOTION,
                         IdsT{ ParamIdE::MELT_MIX, ParamIdE::MELT_TIME, ParamIdE::MELT_SPREAD, ParamIdE::MELT_TO_FAR },
                         IdsT{ ParamIdE::MELT_MIX } },
                       { "Bloom", F::SPACE,
                         IdsT{ ParamIdE::BLOOM, ParamIdE::BLOOM_SIZE, ParamIdE::BLOOM_DECAY, ParamIdE::BLOOM_DRIVE,
                               ParamIdE::BLOOM_GLIDE },
                         IdsT{ ParamIdE::BLOOM } } } );
    // Phase 29: the wave room and the topological halo.
    page(
        "Rooms",
        { { "Wave Room", F::SPACE,
            IdsT{ ParamIdE::WAVE_ROOM, ParamIdE::ROOM_SIZE, ParamIdE::ROOM_ABSORB, ParamIdE::ROOM_WALK },
            IdsT{ ParamIdE::WAVE_ROOM } },
          { "Halo", F::MOTION,
            IdsT{ ParamIdE::HALO, ParamIdE::HALO_SPEED, ParamIdE::HALO_TOPOLOGY, ParamIdE::HALO_DISORDER,
                  ParamIdE::HALO_TURN },
            IdsT{ ParamIdE::HALO } },
          { "Time Mirror", F::SPACE, IdsT{ ParamIdE::MIRROR, ParamIdE::MIRROR_TIME }, IdsT{ ParamIdE::MIRROR } },
          { "Headset", F::FILTER, IdsT{ ParamIdE::VR_HANDS, ParamIdE::HEAD_TRACKING }, IdsT{ ParamIdE::VR_HANDS } } } );
    {
        auto *h = new HallPage( m_proc, m_actions, &m_live );
        m_describe.push_back( [h] { return h->describe(); } );
        m_tabs.addTab( "Hall", additivui::skin().m_panel,
                       new ScrollPage( h, [h]( int w ) { return h->heightFor( w ); } ), true );
    }
    {
        auto *x = new MatrixPage( m_proc, [this]( additiv::ModSourceE s ) { chooseSource( s, false ); } );
        m_matrixPage = x;
        m_describe.push_back( [x] { return x->describe(); } );
        m_tabs.addTab( "Modulation", additivui::skin().m_panel,
                       new ScrollPage( x, [x]( int w ) { return x->heightFor( w ); } ), true );
    }
    {
        auto *j = new JourneyPage( m_proc, m_actions, &m_live );
        m_describe.push_back( [j] { return j->describe(); } );
        m_tabs.addTab( "Journey", additivui::skin().m_panel,
                       new ScrollPage( j, [j]( int w ) { return j->heightFor( w ); } ), true );
    }
    {
        auto *m = new MapPage( m_proc, m_actions, &m_live );
        m_describe.push_back( [m] { return m->describe(); } );
        m_tabs.addTab( "Map", additivui::skin().m_panel,
                       new ScrollPage( m, [m]( int w ) { return m->heightFor( w ); } ), true );
    }
    m_tabs.setTabBarDepth( 0 );   // the strip shows the pages
    m_tabs.setOutline( 0 );
    sendLookAndFeelChange();

    // The pages' strip: the pages in their sections, each with what is on in it.
    {
        const juce::StringArray         names = m_tabs.getTabNames();
        auto                            at = [&names]( const char *n ) { return names.indexOf( n ); };
        std::vector<PageStrip::Section> sections = {
            { "", { at( "Focus" ) } },
            { "Sound",
              { at( "Cloud & Filter" ), at( "Motion" ), at( "Harmonics" ), at( "Dreams" ), at( "Weather" ),
                at( "Night" ) } },
            { "Harmony", { at( "Tuning" ), at( "Conductor" ) } },
            { "Space", { at( "Space" ), at( "Effects" ), at( "Rooms" ), at( "Hall" ) } },
            { "Modulation", { at( "Modulation" ) } },
            { "Explore", { at( "Journey" ), at( "Map" ) } }
        };
        for ( auto &s : sections )
            s.m_pages.erase( std::remove( s.m_pages.begin(), s.m_pages.end(), -1 ), s.m_pages.end() );
        m_strip.setPages( names, std::move( sections ) );
        m_strip.m_statusOf = [this]( int t ) { return pageStatus( t ); };
        m_strip.m_onSelect = [this]( int t )
        {
            if ( m_canvas ) setCanvas( false );
            if ( m_helpView != nullptr && m_helpView->isVisible() ) showHelp( false );
            m_tabs.setCurrentTabIndex( t );
            m_strip.setCurrent( t );
            if ( m_rememberSize ) m_proc.userSettings().setValue( "page", m_tabs.getTabNames()[t] );
        };
        addAndMakeVisible( m_strip );
        m_strip.refresh();
        const int last = m_tabs.getTabNames().indexOf( m_proc.userSettings().getValue( "page", "Focus" ) );
        if ( last > 0 && juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT", {} ).isEmpty() )
            m_tabs.setCurrentTabIndex( last );
    }
    // The modulation dock and its inspector.
    m_dock = std::make_unique<ModDock>( m_proc );
    addAndMakeVisible( *m_dock );
    m_inspector = std::make_unique<SourceInspector>( m_proc, m_actions, *m_dock );
    addChildComponent( *m_inspector );
    m_dock->sendLookAndFeelChange();   // both made before they had the frame's look
    m_inspector->sendLookAndFeelChange();
    m_inspectorOpen = m_proc.userSettings().getBoolValue( "inspector", true );
    m_dock->m_onChoose = [this]( additiv::ModSourceE s ) { chooseSource( s, true ); };
    m_dock->m_onToggle = [this] { setInspector( !( m_inspectorDocked ? m_inspectorOpen : m_overlayOpen ) ); };
    m_dock->setChosen( m_inspector->source() );
    if ( m_matrixPage != nullptr ) static_cast<MatrixPage *>( m_matrixPage )->setChosen( m_inspector->source() );
    m_inspector->m_onClose = [this] { setInspector( false ); };
    m_inspector->m_onReveal = [this]( int id ) { revealParam( id ); };
    m_inspector->m_onMatrix = [this]
    {
        const int t = m_tabs.getTabNames().indexOf( "Modulation" );
        if ( t >= 0 ) m_strip.m_onSelect( t );
    };
    // The night's veil over all of it, the overlays over the veil.
    m_veil.toFront( false );
    m_palette.toFront( false );
    m_browser.toFront( false );
    m_tour.toFront( false );

    for ( auto *p : m_proc.getParameters() ) p->addListener( this );
    frame::keepKeysForEditor( *this );   // the keys stay the editor's (Space plays, not the switch last clicked)
    setWantsKeyboardFocus( true );
    setResizable( true, true );
    setResizeLimits( 800, 520, 4800, 3100 );
    m_zoom = std::clamp( static_cast<float>( m_proc.userSettings().getDoubleValue( "zoom", 1.0 ) ), 1.0f, 1.5f );
    if ( const juce::String z = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_ZOOM", {} ); z.isNotEmpty() )
        m_zoom = std::clamp( z.getFloatValue(), 1.0f, 1.5f );
    const auto size = startSize();
    setSize( size.x, size.y );
    if ( m_zoom != 1.0f ) juce::AudioProcessorEditor::setScaleFactor( m_hostScale * m_zoom );
    m_shotPrefix = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT", {} );
    m_rememberSize =
        m_shotPrefix.isEmpty() && juce::SystemStats::getEnvironmentVariable( "ADDITIV_LAYOUT", {} ).isEmpty();
    if ( !m_rememberSize ) m_tourAsked = true;   // no tour in the screenshot and layout modes
    startTimerHz( 15 );
    timerCallback();
}

std::vector<ParamPage *> AdditivEditor::pagesOn( int tab )
{
    // Looked up each time: a page that rebuilds its knobs (Focus) makes new ones.
    std::vector<ParamPage *> pages;
    if ( auto *view = dynamic_cast<juce::Viewport *>( m_tabs.getTabContentComponent( tab ) ) )
        findParamPages( view->getViewedComponent(), pages );
    return pages;
}

void AdditivEditor::modDropped( additiv::ModSourceE s, int, int row )
{
    if ( row < 0 )
        juce::AlertWindow::showMessageBoxAsync(
            juce::MessageBoxIconType::NoIcon, "Empyrean",
            "All sixteen rows of the matrix are taken: free one on the Modulation page.", "Close" );
    else
        chooseSource( s, false );
}

AdditivEditor::~AdditivEditor()
{
    if ( m_windowKeeper.m_window != nullptr ) m_windowKeeper.m_window->removeComponentListener( &m_windowKeeper );
    m_proc.userSettings().saveIfNeeded();   // the window's size and place
    removeMouseListener( &m_activity );
    for ( auto *p : m_proc.getParameters() ) p->removeListener( this );
    stopTimer();
    setLookAndFeel( nullptr );
}

void AdditivEditor::bindChoice( frame::Choice &box, ParamIdE id )
{
    const additiv::ParamDesc &d = additiv::paramDesc( id );
    for ( int i = 0; i < d.m_numChoices; ++i ) box.addItem( d.m_choices[i], i + 1 );
    box.setColour( juce::ComboBox::arrowColourId, additivui::skin().m_accent );
    m_headerAttach.push_back(
        std::make_unique<juce::ComboBoxParameterAttachment>( *m_proc.parameter( static_cast<int>( id ) ), box ) );
    addAndMakeVisible( box );
}

std::vector<float> AdditivEditor::values() const
{
    std::vector<float> v( static_cast<size_t>( additiv::NumParams ) );
    for ( int i = 0; i < additiv::NumParams; ++i ) v[static_cast<size_t>( i )] = m_proc.value( i );
    return v;
}

void AdditivEditor::parameterGestureChanged( int index, bool starting )
{
    // Gestures from the panel come on the message thread; a host's from anywhere -- those are its own to undo.
    if ( m_applying || !juce::MessageManager::getInstance()->isThisTheMessageThread() ) return;
    if ( starting )
    {
        if ( !m_history.open() )
            m_history.begin( additiv::paramDesc( static_cast<ParamIdE>( index ) ).m_name, values(), matrixText(),
                             false );
        m_history.touch( index );
    }
    else if ( m_history.open() )
    {
        endStep();
    }
}

void AdditivEditor::apply( const frame::UndoStep *step, bool after )
{
    if ( step == nullptr ) return;
    m_applying = true;
    for ( const auto &[id, before, now] : step->m_values ) m_proc.setValue( id, after ? now : before );
    if ( step->m_extraBefore != step->m_extraAfter )
        m_proc.engine().setModText( ( after ? step->m_extraAfter : step->m_extraBefore ).toStdString() );
    m_applying = false;
    m_lastMatrix = matrixText();
    m_pendingBefore.clear();
}

void AdditivEditor::endStep()
{
    m_history.end( values(), matrixText() );
    m_lastMatrix = matrixText();
    m_pendingBefore.clear();
}

void AdditivEditor::watchMatrix()
{
    const juce::String now = matrixText();
    const juce::String key = juce::String( m_proc.currentPreset() ) + "|" + m_proc.currentUserPreset();
    // What a preset, a journey, the map, a morph or a preview does to the matrix is theirs, not an edit.
    const bool blended = m_proc.value( static_cast<int>( ParamIdE::MAP_ACTIVE ) ) >= 0.5f ||
                         m_proc.value( static_cast<int>( ParamIdE::MORPH_ACTIVE ) ) >= 0.5f ||
                         m_proc.journeyStatus().isNotEmpty() || m_proc.comparing() || m_proc.previewing() >= 0;
    if ( key != m_presetKey || blended || m_applying || m_history.open() )
    {
        m_presetKey = key;
        m_lastMatrix = now;
        m_pendingBefore.clear();
        return;
    }
    if ( now != m_lastMatrix )
    {
        if ( m_pendingBefore.isEmpty() ) m_pendingBefore = m_lastMatrix;
        m_pendingAt = juce::Time::getMillisecondCounterHiRes();
        m_lastMatrix = now;
    }
    if ( m_pendingBefore.isNotEmpty() && juce::Time::getMillisecondCounterHiRes() - m_pendingAt > 700.0 ) flushMatrix();
}

void AdditivEditor::flushMatrix()
{
    if ( m_pendingBefore.isEmpty() ) return;
    const juce::String before = m_pendingBefore;
    m_pendingBefore.clear();
    if ( before == matrixText() ) return;
    m_history.begin( "Modulation", values(), before, true );
    m_history.end( values(), matrixText() );
    m_lastMatrix = matrixText();
}

void AdditivEditor::timerCallback()
{
    additiv::Engine &e = m_proc.engine();
    const bool       on = m_proc.value( static_cast<int>( ParamIdE::BRAIN_ON ) ) >= 0.5f;
    m_play.show( on ? frame::IconButton::IconE::STOP : frame::IconButton::IconE::PLAY,
                 on ? "Autoplay: stop the conductor" : "Autoplay: start the conductor" );
    const int    cp = m_proc.currentPreset();
    juce::String name = cp >= 0 ? juce::String( additiv::PresetGroupNames[additiv::preset( cp ).m_group] ) + ": " +
                                      additiv::preset( cp ).m_name
                        : m_proc.currentUserPreset().isNotEmpty() ? "Yours: " + m_proc.currentUserPreset()
                                                                  : juce::String( "Presets ..." );
    if ( ( cp >= 0 || m_proc.currentUserPreset().isNotEmpty() ) && m_proc.modified() )
        name << " *";   // moved from its preset
    if ( m_presetName.getButtonText() != name ) m_presetName.setButtonText( name );
    m_undo.setEnabled( m_history.canUndo() );
    m_redo.setEnabled( m_history.canRedo() );
    juce::String s;
    if ( m_proc.learning() >= 0 )
    {
        s << "MIDI learn: move a controller for "
          << additiv::paramDesc( static_cast<ParamIdE>( m_proc.learning() ) ).m_name;
    }
    else if ( !e.prepared() )
    {
        const juce::String why = m_proc.engineError();
        s << "The GPU engine is not running: " << ( why.isEmpty() ? juce::String( "the audio is off" ) : why )
          << "  (an NVIDIA card, GeForce RTX 20 or newer, with a current driver)";
    }
    else
    {
        static const char *const Names[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
        const int                root = e.brainRoot();
        s << e.soundingVoices() << " voices (" << e.soundingFrom( 1 ) << " conductor, " << e.soundingFrom( 2 )
          << " brain 2, " << e.soundingFrom( 0 ) << " played)   root " << Names[( ( root % 12 ) + 12 ) % 12]
          << ( root / 12 - 1 );
        // The machine only when it matters: a block that took more than half its time, or audio that broke.
        const double blockMs = 1000.0 * e.config().m_gpuBlock / std::max( e.config().m_sampleRate, 1.0 );
        if ( e.lastGpuMs() > 0.5 * blockMs || e.underruns() > 0 )
            s << "   GPU " << juce::String( e.lastGpuMs(), 2 ) << " ms";
        if ( e.underruns() > 0 ) s << "   underruns " << static_cast<int>( e.underruns() );
        m_status.setTooltip( "GPU " + juce::String( e.lastGpuMs(), 3 ) + " ms a block of " +
                             juce::String( blockMs, 2 ) + " ms   latency " +
                             juce::String( e.latencySamples() * 1000.0 / std::max( e.config().m_sampleRate, 1.0 ), 1 ) +
                             " ms   " + e.deviceName() );
        const juce::String journey = m_proc.journeyStatus();
        if ( journey.isNotEmpty() )
            s << "      Journey: " << journey;
        else if ( e.morphProgress() < 1.0f )
            s << "      gliding " << juce::roundToInt( 100.0f * e.morphProgress() ) << " %";
    }
    m_status.setText( s, juce::dontSendNotification );
    {   // the meter, the comparison, the Step's countdown, the matrix's undo
        float pl = 0.0f, pr = 0.0f;
        m_proc.takePeaks( pl, pr );
        m_meter.push( pl, pr );
        m_compare.setEnabled( m_proc.modified() || m_proc.comparing() );
        const float next = e.conductorNext(), span = std::max( e.conductorSpan(), 0.001f );
        m_stepLeft = on && e.prepared() ? std::clamp( next / span, 0.0f, 1.0f ) : -1.0f;
        m_step.setTooltip( m_stepLeft >= 0.0f ? "Autoplay: exchange one voice of the chord now (the next in " +
                                                    juce::String( juce::roundToInt( next ) ) + " s)"
                                              : juce::String( "Autoplay: exchange one voice of the chord now" ) );
        repaint( m_step.getBounds().expanded( 2, 4 ) );
        watchMatrix();
    }
    m_cloud.setPage( m_canvas ? juce::String()
                              : m_tabs.getCurrentTabName() );   // the picture shows what the page in front edits
    m_strip.setCurrent( m_tabs.getCurrentTabIndex() );
    if ( m_logTick % 4 == 0 ) m_strip.refresh();   // what is on in every page, four times a second
    // The night: the picture calmer (half as often), the veil -- deeper in a canvas left alone for a minute and a half.
    if ( !m_night || ( ++m_refreshTick & 1 ) == 0 ) m_cloud.refresh();
    float veil = 0.0f;
    if ( m_night )
    {
        veil = 0.5f;
        const double idle = ( juce::Time::getMillisecondCounterHiRes() - m_lastActivity ) * 0.001;
        if ( m_canvas && idle > 90.0 ) veil = static_cast<float>( std::min( 0.9, 0.5 + ( idle - 90.0 ) / 20.0 * 0.4 ) );
    }
    if ( std::abs( veil - m_veil.m_alpha ) > 0.004f )
    {
        m_veil.m_alpha = veil;
        m_veil.setVisible( veil > 0.0f );
        m_veil.repaint();
    }
    // ADDITIV_LAYOUT: the pages measured, then the standalone quits.
    if ( const juce::String layout = juce::SystemStats::getEnvironmentVariable( "ADDITIV_LAYOUT", {} );
         layout.isNotEmpty() && !m_layoutChecked && isShowing() )
    {
        m_layoutChecked = true;
        layoutCheck( juce::File( layout ) );
        return;
    }
    // The first steps, once, on the first start (not in the shot mode).
    if ( !m_tourAsked && m_shotPrefix.isEmpty() && !m_layoutChecked )
    {
        m_tourAsked = true;
        if ( !m_proc.userSettings().getBoolValue( "tourDone", false ) )
            juce::Timer::callAfterDelay( 1200,
                                         [safe = juce::Component::SafePointer<AdditivEditor>( this )]
                                         {
                                             if ( safe != nullptr ) safe->startTour();
                                         } );
    }
    shootStep();
    // ADDITIV_LOG: a line a second on the real-time path (the slowest block, the emptiest ring, the
    // largest callback, the underruns so far), for finding what makes it stutter; the slowest block
    // is split into its preparation, its wait for the GPU and its effects (each its own slowest).
    if ( ++m_logTick % 15 == 0 )
    {
        const juce::String path = juce::SystemStats::getEnvironmentVariable( "ADDITIV_LOG", {} );
        if ( path.isNotEmpty() && e.prepared() )
        {
            float prep = 0.0f, gpu = 0.0f, fx = 0.0f;
            e.takeWorstParts( prep, gpu, fx );
            juce::File( path ).appendText(
                juce::String( m_logTick / 15 ) + " s: worst block " + juce::String( e.takeWorstBlockMs(), 3 ) +
                " ms (prep " + juce::String( prep, 3 ) + ", gpu " + juce::String( gpu, 3 ) + ", effects " +
                juce::String( fx, 3 ) + "), min fill " + juce::String( e.takeMinFill() ) + " of " +
                juce::String( e.latencySamples() ) + ", callback " + juce::String( e.takeMaxCallback() ) +
                ", underruns " + juce::String( static_cast<int>( e.underruns() ) ) + "\n" );
        }
    }
}

void AdditivEditor::shootStep()
{
    // ADDITIV_SHOT_AFTER seconds (at least 3) for the engine to sound, then every page: shown on one tick,
    // photographed three ticks later.
    const int start =
        15 * std::max( 3, juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_AFTER", "3" ).getIntValue() );
    // ADDITIV_PRESET (a name or a number): the preset the pictures show, loaded before the engine is heard.
    if ( m_shotPrefix.isNotEmpty() && m_shotTick == 0 )
    {
        // From the defaults, whatever the standalone's last session left (the map on, a morph, a
        // radius): the pictures are the same every time. What it left is given back before quitting.
        m_proc.getStateInformation( m_shotSaved );
        for ( int i = 0; i < additiv::NumParams; ++i ) m_proc.resetToDefault( i );
        const juce::String want = juce::SystemStats::getEnvironmentVariable( "ADDITIV_PRESET", {} );
        int                index = -1;
        for ( int i = 0; i < additiv::numPresets() && index < 0 && want.isNotEmpty(); ++i )
            if ( want == additiv::preset( i ).m_name ) index = i;
        if ( index < 0 && want.containsOnly( "0123456789" ) && want.isNotEmpty() ) index = want.getIntValue();
        if ( index >= 0 ) m_proc.applyPreset( index );
        // ADDITIV_ROUTE: a route by name, walked from the start.
        const juce::String routeName = juce::SystemStats::getEnvironmentVariable( "ADDITIV_ROUTE", {} );
        for ( const auto &r : m_proc.routes() )
            if ( routeName.isNotEmpty() && r.m_name == routeName && m_proc.loadRoute( r ) ) m_proc.startRoute();
        // ADDITIV_LANDSCAPE: a picture or a recording as the player's own landscape.
        const juce::String land = juce::SystemStats::getEnvironmentVariable( "ADDITIV_LANDSCAPE", {} );
        if ( land.isNotEmpty() )
        {
            juce::String why;
            m_proc.loadLandscape( juce::File( land ), true, &why );
        }
        // ADDITIV_SET ("key=value;key=value"): parameters after the preset, a choice by its name or its index.
        for ( const juce::String &item :
              juce::StringArray::fromTokens( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SET", {} ), ";", "" ) )
            if ( const additiv::ParamDesc *d =
                     additiv::findParam( item.upToFirstOccurrenceOf( "=", false, false ).trim().toRawUTF8() ) )
            {
                const juce::String val = item.fromFirstOccurrenceOf( "=", false, false ).trim();
                float              v = val.getFloatValue();
                for ( int c = 0; c < d->m_numChoices; ++c )
                    if ( val == d->m_choices[c] ) v = static_cast<float>( c );
                m_proc.setValue( static_cast<int>( d->m_id ), v );
            }
        // ADDITIV_SHOT_SAVE: the sound saved as the player's preset of that name, then Tilt moved (the star, the marks).
        const juce::String saveName = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_SAVE", {} );
        if ( saveName.isNotEmpty() )
        {
            juce::String why;
            m_proc.saveUserPreset( saveName, why );
            m_proc.setValue( static_cast<int>( ParamIdE::TILT ),
                             m_proc.value( static_cast<int>( ParamIdE::TILT ) ) + 0.4f );
        }
        // ADDITIV_SHOT_SHOW: a knob shown as the palette shows it (its page, lit, told about in the info line).
        if ( const additiv::ParamDesc *shown = additiv::findParam(
                 juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_SHOW", {} ).toRawUTF8() ) )
        {
            const int id = static_cast<int>( shown->m_id );
            juce::Timer::callAfterDelay( 3000,
                                         [safe = juce::Component::SafePointer<AdditivEditor>( this ), id]
                                         {
                                             if ( safe != nullptr ) safe->revealParam( id );
                                         } );
        }
        // ADDITIV_SHOT_NIGHT: the night mode, for this run only; ADDITIV_SHOT_SNAP: two snapshots a few seconds in;
        // ADDITIV_SHOT_TOUR: the tour's first step.
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_NIGHT", {} ).isNotEmpty() ) m_night = true;
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_SNAP", {} ).isNotEmpty() )
            for ( int ms : { 2000, 2600 } )
                juce::Timer::callAfterDelay( ms,
                                             [safe = juce::Component::SafePointer<AdditivEditor>( this )]
                                             {
                                                 if ( safe != nullptr )
                                                     safe->m_proc.takeSnapshot( safe->m_cloud.thumbnail( 160, 60 ) );
                                             } );
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_TOUR", {} ).isNotEmpty() )
            juce::Timer::callAfterDelay( 1500,
                                         [safe = juce::Component::SafePointer<AdditivEditor>( this )]
                                         {
                                             if ( safe != nullptr ) safe->startTour();
                                         } );
        // ADDITIV_SHOT_SOURCE: a source in the inspector (by its name), opened.
        const juce::String sourceName = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_SOURCE", {} );
        for ( int k = 1; k < additiv::NumModSources && sourceName.isNotEmpty(); ++k )
            if ( sourceName == additiv::ModSourceNames[k] )
                juce::Timer::callAfterDelay( 1000,
                                             [safe = juce::Component::SafePointer<AdditivEditor>( this ), k]
                                             {
                                                 if ( safe != nullptr )
                                                     safe->chooseSource( static_cast<additiv::ModSourceE>( k ), false );
                                             } );
        // ADDITIV_SHOT_DROP ("LFO 2>cutoff"): a source dropped on a knob, as the dock's chip drops it.
        const juce::String drop = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_DROP", {} );
        if ( drop.contains( ">" ) )
            juce::Timer::callAfterDelay(
                1500,
                [safe = juce::Component::SafePointer<AdditivEditor>( this ), drop]
                {
                    if ( safe == nullptr ) return;
                    const additiv::ParamDesc *d =
                        additiv::findParam( drop.fromFirstOccurrenceOf( ">", false, false ).trim().toRawUTF8() );
                    int src = 0;
                    for ( int k = 1; k < additiv::NumModSources; ++k )
                        if ( drop.upToFirstOccurrenceOf( ">", false, false ).trim() == additiv::ModSourceNames[k] )
                            src = k;
                    if ( d == nullptr || src == 0 || !safe->revealParam( static_cast<int>( d->m_id ) ) ) return;
                    for ( ParamPage *p :
                          safe->pagesOn( safe->m_tabs.getCurrentTabIndex() ) )   // the page revealParam showed
                        if ( juce::Component *c = p->controlFor( static_cast<int>( d->m_id ) ) )
                        {
                            p->itemDropped( { knobmod::dragOf( static_cast<additiv::ModSourceE>( src ) ),
                                              safe->m_dock.get(), c->getBounds().getCentre() } );
                            break;
                        }
                } );
        // ADDITIV_SHOT_JOURNEY: a journey under way, by its name (the timeline's picture).
        const juce::String journeyName = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_JOURNEY", {} );
        for ( const auto &entry : m_proc.journeys() )
            if ( journeyName.isNotEmpty() && entry.m_name == journeyName )
            {
                m_proc.startJourney( entry );
                // ADDITIV_SHOT_DRAFT: and a copy of it as the journey being written (the timeline's second band).
                if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_DRAFT", {} ).isNotEmpty() )
                {
                    additiv::Journey copy = m_proc.journeyPlayer().journey();
                    copy.m_name += " (mine)";
                    m_proc.setJourneyDraft( copy );
                }
                break;
            }
        // ADDITIV_SHOT_UNDO: after a drop (ADDITIV_SHOT_DROP), Undo at four seconds -- the matrix as it was.
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_UNDO", {} ).isNotEmpty() )
            juce::Timer::callAfterDelay( 4000,
                                         [safe = juce::Component::SafePointer<AdditivEditor>( this )]
                                         {
                                             if ( safe == nullptr ) return;
                                             safe->flushMatrix();
                                             safe->apply( safe->m_history.undo(), false );
                                         } );
        // ADDITIV_SHOT_BROWSER: the preset browser open.
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_BROWSER", {} ).isNotEmpty() )
            juce::MessageManager::callAsync( [this] { m_browser.open(); } );
        // ADDITIV_SHOT_PALETTE: the command palette open with that typed into it.
        const juce::String query = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_PALETTE", {} );
        if ( query.isNotEmpty() )
            juce::MessageManager::callAsync(
                [this, query]
                {
                    openPalette();
                    m_palette.type( query );
                } );
        // ADDITIV_SHOT_VARY: the Focus page's six variations made, for their picture.
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_VARY", {} ).isNotEmpty() )
            m_proc.makeVariations( 0.3f );
    }
    if ( m_shotPrefix.isEmpty() || ++m_shotTick < start || ( m_shotTick - start ) % 3 != 0 ) return;
    const int t = ( m_shotTick - start ) / 3;
    if ( t > 0 )
    {
        const juce::Image img = createComponentSnapshot( getLocalBounds() );
        const juce::File  f( m_shotPrefix + "-" + juce::String( t - 1 ) + "-" +
                             ( m_canvas ? juce::String( "Canvas" ) : m_tabs.getCurrentTabName() ) + ".png" );
        f.deleteFile();
        juce::FileOutputStream out( f );
        if ( out.openedOk() ) juce::PNGImageFormat().writeImageToStream( img, out );
    }
    if ( t == m_tabs.getNumTabs() && !m_canvas &&
         juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_CANVAS", {} ).isNotEmpty() )
    {
        setCanvas( true );   // ADDITIV_SHOT_CANVAS: the canvas last, photographed on the next round
        m_shotTick -= 3;
        return;
    }
    if ( t < m_tabs.getNumTabs() )
    {
        m_tabs.setCurrentTabIndex( t );
        // ADDITIV_SHOT_FULL: the window as tall as the page, so that nothing of it scrolls out of the picture.
        if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_FULL", {} ).isNotEmpty() )
            if ( auto *sp = dynamic_cast<ScrollPage *>( m_tabs.getTabContentComponent( t ) ) )
                setSize( getWidth(),
                         std::clamp( getHeight() - sp->getMaximumVisibleHeight() + sp->pageHeight() + 16, 860, 3100 ) );
    }
    else
    {
        m_shotPrefix = {};
        if ( m_shotSaved.getSize() > 0 )
            m_proc.setStateInformation( m_shotSaved.getData(), static_cast<int>( m_shotSaved.getSize() ) );
        if ( standalone() ) juce::JUCEApplicationBase::quit();
    }
}

void AdditivEditor::layoutCheck( const juce::File &out )
{
    const auto   before = getBounds();
    juce::String report;
    int          scrolling = 0;
    const int    sizes[][2] = { { 1280, 860 }, { 1280, 800 }, { 1280, 760 }, { 1440, 900 }, { 1920, 1000 } };
    for ( const auto &s : sizes )
    {
        setSize( s[0], s[1] );
        report << s[0] << " x " << s[1] << "\n";
        for ( int t = 0; t < m_tabs.getNumTabs(); ++t )
            if ( auto *sp = dynamic_cast<ScrollPage *>( m_tabs.getTabContentComponent( t ) ) )
            {
                const int need = sp->heightAt( sp->getWidth() ), room = sp->getHeight();
                report << "    " << m_tabs.getTabNames()[t].paddedRight( ' ', 12 ) << " needs " << need << ", has "
                       << room << ( need > room ? "   SCROLLS by " + juce::String( need - room ) : juce::String() )
                       << "\n";
                if ( need > room && s[0] == 1280 && s[1] == 860 ) ++scrolling;
            }
    }
    {   // and the kinds of audio device the standalone offers (ASIO among them on Windows)
        juce::AudioDeviceManager                  devices;
        juce::OwnedArray<juce::AudioIODeviceType> types;
        devices.createAudioDeviceTypes( types );
        report << "on screen " << getScreenBounds().getWidth() << " x " << getScreenBounds().getHeight() << " at zoom "
               << m_zoom << "\n";
        report << "audio device types:";
        for ( auto *type : types ) report << " " << type->getTypeName() << ";";
        report << "\n";
    }
    out.replaceWithText( report );
    setBounds( before );
    if ( standalone() )
    {
        juce::JUCEApplicationBase::getInstance()->setApplicationReturnValue( scrolling );
        juce::JUCEApplicationBase::quit();
    }
}

void AdditivEditor::paintOverChildren( juce::Graphics &g )
{
    if ( m_stepLeft < 0.0f || m_canvas || !m_step.isShowing() || m_palette.isVisible() || m_browser.isVisible() ||
         m_tour.isVisible() )
        return;
    const auto  b = m_step.getBounds().toFloat();
    const float w = ( b.getWidth() - 8.0f ) * m_stepLeft;
    g.setColour( additivui::skin().m_accent.withAlpha( 0.8f ) );
    g.fillRoundedRectangle( b.getX() + 4.0f, b.getBottom() + 1.0f, w, 2.0f, 1.0f );
}

void AdditivEditor::setNight( bool on )
{
    m_night = on;
    m_nightButton.setToggleState( on, juce::dontSendNotification );
    m_proc.userSettings().setValue( "night", on );
    m_proc.userSettings().saveIfNeeded();
    m_lastActivity = juce::Time::getMillisecondCounterHiRes();
}

void AdditivEditor::startTour()
{
    if ( m_canvas ) setCanvas( false );
    m_tabs.setCurrentTabIndex( 0 );
    std::vector<Tour::Step> s;
    s.push_back(
        { &m_play, "Play",
          "The conductor plays by itself: chords in just intonation that change one voice at a time. Space starts and stops it in the standalone; play the keys over it." } );
    s.push_back(
        { &m_presetName, "13312 presets",
          "Click the name to browse them: by name and group, by how they sound (darker, brighter, stiller, more moving ...), by your stars and the ones played lately. Hold a row's play mark to listen." } );
    s.push_back(
        { &m_cloud, "The partials themselves",
          "Twenty thousand sines a voice, as the graphics card has them. The page in front draws what it edits here -- the filter's curve, the register, the air -- and you can drag it. Double-click: the canvas." } );
    s.push_back(
        { &m_strip, "Focus",
          "The simple ways in: nudge the sound brighter, calmer, wider or airier; the few knobs that matter for it; snapshots; variations to evolve it by ear; macros taught by example; presets found by a recording." } );
    s.push_back(
        { &m_searchButton, "Search (Ctrl+K)",
          "Type a few letters to find any knob (its page opens, it lights up; \"cutoff 800\" sets it), any preset or action." } );
    if ( m_info != nullptr )
        s.push_back( { m_info.get(), "The info line",
                       "Point at any control: what it does, its value and what moves it are told here." } );
    s.push_back(
        { &m_save, "Your own presets",
          "Save the sound as your own (Ctrl+S); a star by the name tells that it has moved from its preset, a notch on a knob where the preset had it." } );
    m_tour.start( std::move( s ) );
}

void AdditivEditor::paint( juce::Graphics &g )
{
    m_backdrop.paint( g, getLocalBounds(), m_headerBottom, additivui::skin(), frame::Settings::of( App ).backdrop() );
    if ( additivui::skin().m_logo ) additivui::skin().m_logo( g, m_logo );
}

void AdditivEditor::resized()
{
    if ( m_dock == nullptr ) return;   // not built yet
    auto full = getLocalBounds();
    if ( m_info != nullptr )
    {   // the info line along the bottom (not in the canvas)
        m_info->setVisible( !m_canvas );
        if ( !m_canvas ) m_info->setBounds( full.removeFromBottom( 24 ) );
    }
    auto          area = full.reduced( 12, 8 );
    frame::Header h;
    h.m_logo = &m_logo;
    h.m_title = &m_title;
    h.m_titleWidth = static_cast<int>( frame::titleWidth( additivui::skin(), frame::HeaderRow * 0.62f ) ) + 8;
    h.m_choices = { { &m_prevPreset, 28 }, { &m_presetName, 250 }, { &m_nextPreset, 28 }, { &m_compare, 32 },
                    { &m_mode, 92 },       { &m_scale, 150 },      { &m_root, 58 },       { &m_motion, 104 } };
    m_landscape.setVisible( false );   // on the Motion page; the header has room for the bank
    h.m_actions = { { &m_step, 64 } };
    h.m_play = &m_play;
    h.m_status = &m_status;
    h.m_tools = { &m_undo, &m_redo, nullptr, &m_searchButton, &m_canvasButton, &m_nightButton,
                  &m_midi, nullptr, &m_save, &m_help,         &m_settings };
    frame::layoutHeader( area, h );
    {   // the meter at the status line's right end
        auto sb = m_status.getBounds();
        m_meter.setBounds( sb.removeFromRight( 110 ).reduced( 4, 5 ) );
        m_status.setBounds( sb );
    }
    m_headerBottom = area.getY();
    m_palette.setBounds( getLocalBounds() );
    m_veil.setBounds( getLocalBounds() );
    m_browser.setBounds( getLocalBounds() );
    m_strip.setVisible( !m_canvas );
    m_dock->setVisible( !m_canvas );
    if ( m_canvas ) m_inspector->setVisible( false );
    if ( m_rememberSize && juce::Desktop::getInstance().getKioskModeComponent() == nullptr )
    {
        // The size it is left at -- not a maximised one: that is kept in the window's state (restoreWindow).
        const auto *window = findParentComponentOfClass<juce::ResizableWindow>();
        if ( window == nullptr || !window->isFullScreen() )
        {
            m_proc.userSettings().setValue( "windowWidth", getWidth() );
            m_proc.userSettings().setValue( "windowHeight", getHeight() );
        }
    }
    if ( m_canvas )
    {   // the picture is the window
        m_cloud.setBounds( area );
        m_tour.setBounds( getLocalBounds() );
        return;
    }
    // The dock along the bottom. The picture a third of the room (at most 420 high), but never what a
    // page needs: the tallest page at this width fits under it while the picture can be 100 high. The inspector docked
    // over the dock where that leaves the picture 150 high, else it opens over the pages when wanted.
    const auto dockRow = area.removeFromBottom( ModDock::Height );
    m_dock->setBounds( dockRow );
    area.removeFromBottom( 4 );
    int need = 0;
    for ( int t = 0; t < m_tabs.getNumTabs(); ++t )
        if ( auto *sp = dynamic_cast<ScrollPage *>( m_tabs.getTabContentComponent( t ) ) )
            need = std::max( need, sp->heightAt( area.getWidth() ) );
    const int under = 8 + PageStrip::Height + need;
    m_inspectorDocked = area.getHeight() - under - ( SourceInspector::Height + 6 ) >= 150;
    const bool inspect = m_inspectorDocked ? m_inspectorOpen : m_overlayOpen;
    if ( m_inspectorDocked && inspect )
    {
        m_inspector->setBounds( area.removeFromBottom( SourceInspector::Height ) );
        area.removeFromBottom( 6 );
    }
    else if ( !m_inspectorDocked )
    {
        m_inspector->setBounds( dockRow.getX(), dockRow.getY() - 4 - SourceInspector::Height, dockRow.getWidth(),
                                SourceInspector::Height );
    }
    m_inspector->setOverlay( !m_inspectorDocked );
    m_inspector->setVisible( inspect );
    m_dock->setOpen( inspect );
    const int rest = area.getHeight() - under;
    m_cloud.setBounds( area.removeFromTop( juce::jlimit( 100, 420, std::min( area.getHeight() / 3, rest ) ) ) );
    m_tour.setBounds( getLocalBounds() );
    area.removeFromTop( 8 );
    const auto pagesArea = area;
    m_strip.setBounds( area.removeFromTop( PageStrip::Height ) );
    m_tabs.setBounds( area );
    if ( m_helpView != nullptr ) m_helpView->setBounds( pagesArea );
}

juce::Point<int> AdditivEditor::startSize()
{
    if ( juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT", {} ).isNotEmpty() ||
         juce::SystemStats::getEnvironmentVariable( "ADDITIV_LAYOUT", {} ).isNotEmpty() )
    {
        // ADDITIV_SHOT_SIZE ("1920x1200"): the pictures at another size.
        const juce::String size = juce::SystemStats::getEnvironmentVariable( "ADDITIV_SHOT_SIZE", {} );
        const int          w = size.upToFirstOccurrenceOf( "x", false, false ).getIntValue(),
                           h = size.fromFirstOccurrenceOf( "x", false, false ).getIntValue();
        return w >= 800 && h >= 520 ? juce::Point<int>( w, h ) : juce::Point<int>( 1280, 860 );
    }
    const juce::Displays::Display *d = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay();
    const auto screen = d != nullptr ? d->userBounds.toNearestInt() : juce::Rectangle<int>( 0, 0, 1920, 1040 );
    const auto user = juce::Rectangle<int>( juce::roundToInt( static_cast<float>( screen.getWidth() ) / m_zoom ),
                                            juce::roundToInt( static_cast<float>( screen.getHeight() ) / m_zoom ) );
    const int  maxW = std::max( 800, user.getWidth() - 40 ),
               maxH = std::max( 520, user.getHeight() - 80 );   // room for the host's frame
    const int  lastW = m_proc.userSettings().getIntValue( "windowWidth", 0 ),
               lastH = m_proc.userSettings().getIntValue( "windowHeight", 0 );
    // The size it was left at, where it fits a screen: any of them, as it may have been on a larger one than the main one.
    int roomW = user.getWidth(), roomH = user.getHeight();
    for ( const auto &other : juce::Desktop::getInstance().getDisplays().displays )
    {
        roomW = std::max( roomW, juce::roundToInt( static_cast<float>( other.userBounds.getWidth() ) / m_zoom ) );
        roomH = std::max( roomH, juce::roundToInt( static_cast<float>( other.userBounds.getHeight() ) / m_zoom ) );
    }
    if ( lastW >= 800 && lastH >= 520 && lastW <= roomW && lastH <= roomH ) return { lastW, lastH };
    const int w = std::clamp( juce::roundToInt( static_cast<float>( user.getWidth() ) * 0.75f ), 1280, 2200 );
    const int h = std::max( 860, juce::roundToInt( static_cast<float>( w ) * 0.625f ) );
    return { std::max( 800, std::min( w, maxW ) ), std::max( 520, std::min( h, maxH ) ) };
}

PageStrip::Status AdditivEditor::pageStatus( int tab )
{
    PageStrip::Status s;
    if ( tab < 0 || tab >= m_tabs.getNumTabs() ) return s;
    juce::StringArray on, off;
    bool              free = false;   // a group that never rests
    for ( ParamPage *p : pagesOn( tab ) )
        for ( const ParamPage::GroupState &g : p->groupStates() )
        {
            if ( !g.m_gated )
            {
                free = true;
                continue;
            }
            s.m_dots.push_back( { g.m_colour, g.m_awake } );
            ( g.m_awake ? on : off ).add( g.m_title );
        }
    const juce::String name = m_tabs.getTabNames()[tab];
    if ( name == "Modulation" )
    {
        int rows = 0;
        for ( int i = 0; i < additiv::ModRows; ++i )
        {
            const additiv::ModRow r = m_proc.engine().modRow( i );
            if ( r.m_source != additiv::ModSourceE::OFF && r.m_target >= 0 ) ++rows;
        }
        if ( rows > 0 ) s.m_badge = juce::String( rows );
        s.m_tip = rows == 0 ? juce::String( "no row in use" )
                            : juce::String( rows ) + ( rows == 1 ? " row" : " rows" ) + " of 16 in use";
        return s;
    }
    if ( name == "Journey" && m_proc.journeyStatus().isNotEmpty() )
    {
        s.m_dots.push_back( { additivui::skin().m_accent, true } );
        on.add( "a journey (" + m_proc.journeyStatus() + ")" );
    }
    if ( name == "Map" && m_proc.routeRunning() )
    {
        s.m_dots.push_back( { additivui::skin().m_accent, true } );
        on.add( "a route" );
    }
    s.m_quiet = !free && !s.m_dots.empty() && on.isEmpty();
    if ( !on.isEmpty() ) s.m_tip << "on: " << on.joinIntoString( ", " );
    if ( !off.isEmpty() ) s.m_tip << ( s.m_tip.isEmpty() ? "" : "; " ) << "off: " << off.joinIntoString( ", " );
    return s;
}

void AdditivEditor::chooseSource( additiv::ModSourceE s, bool toggle )
{
    const bool open = m_inspectorDocked ? m_inspectorOpen : m_overlayOpen;
    if ( toggle && open && m_inspector->source() == s )
    {
        setInspector( false );
        return;
    }
    m_inspector->show( s );
    m_dock->setChosen( s );
    if ( m_matrixPage != nullptr ) static_cast<MatrixPage *>( m_matrixPage )->setChosen( s );
    setInspector( true );
}

void AdditivEditor::setInspector( bool open )
{
    if ( m_inspectorDocked )
    {
        m_inspectorOpen = open;
        if ( m_rememberSize ) m_proc.userSettings().setValue( "inspector", open );   // not from the screenshot mode
    }
    else
    {
        m_overlayOpen = open;
    }
    resized();
}

bool AdditivEditor::keyPressed( const juce::KeyPress &key )
{
    if ( key == juce::KeyPress( 'k', juce::ModifierKeys::ctrlModifier, 0 ) ||
         key == juce::KeyPress( 'K', juce::ModifierKeys::ctrlModifier, 0 ) )
    {
        openPalette();
        return true;
    }
    frame::Keys k;
    if ( standalone() )
    {   // in a host, Space and F11 are the host's
        k.m_playStop = [this] { m_play.triggerClick(); };
        k.m_fullScreen = [this] { toggleFullScreen(); };
    }
    k.m_save = [this] { quickSave(); };
    k.m_undo = [this]
    {
        flushMatrix();
        apply( m_history.undo(), false );
    };
    k.m_redo = [this]
    {
        flushMatrix();
        apply( m_history.redo(), true );
    };
    k.m_help = [this] { showHelp( m_helpView == nullptr || !m_helpView->isVisible() ); };
    k.m_escape = [this]
    {
        if ( m_overlayOpen && !m_inspectorDocked )
            setInspector( false );
        else if ( m_canvas )
            setCanvas( false );
        else if ( m_helpView != nullptr && m_helpView->isVisible() )
            showHelp( false );
        else if ( juce::Desktop::getInstance().getKioskModeComponent() != nullptr )
            toggleFullScreen();
    };
    return frame::handleKey( key, k );
}

void AdditivEditor::setScaleFactor( float newScale )
{
    m_hostScale = newScale;
    juce::AudioProcessorEditor::setScaleFactor( m_hostScale * m_zoom );
}

void AdditivEditor::setZoom( float z )
{
    m_zoom = std::clamp( z, 1.0f, 1.5f );
    if ( m_rememberSize ) m_proc.userSettings().setValue( "zoom", static_cast<double>( m_zoom ) );
    juce::AudioProcessorEditor::setScaleFactor( m_hostScale * m_zoom );
}

void AdditivEditor::parentHierarchyChanged()
{
    // On the next turn of the message loop: the standalone's window is still putting its content in
    // when this is called. In a host the plug-in lives in the host's window and this finds nothing.
    juce::MessageManager::callAsync(
        [safe = juce::Component::SafePointer<AdditivEditor>( this )]
        {
            if ( safe == nullptr ) return;
            if ( auto *window = safe->findParentComponentOfClass<juce::DocumentWindow>() )
            {
                window->setTitleBarButtonsRequired( juce::DocumentWindow::minimiseButton |
                                                        juce::DocumentWindow::maximiseButton |
                                                        juce::DocumentWindow::closeButton,
                                                    false );
                safe->restoreWindow( *window );
            }
        } );
}

void AdditivEditor::restoreWindow( juce::ResizableWindow &window )
{
    // Only the standalone's own window, once; never in the screenshot and layout modes.
    if ( !standalone() || !m_rememberSize || m_windowKeeper.m_window != nullptr ) return;
    // JUCE's window has put itself on one screen at the size the editor asked for; the state kept says exactly
    // where it was -- across two screens, maximised -- and is brought back onto a screen should that one be gone.
    const juce::String state = m_proc.userSettings().getValue( "windowState" );
    if ( state.isNotEmpty() ) window.restoreWindowStateFromString( state );
    m_windowKeeper.m_owner = this;
    m_windowKeeper.m_window = &window;
    window.addComponentListener( &m_windowKeeper );
}

void AdditivEditor::WindowKeeper::componentMovedOrResized( juce::Component &, bool, bool )
{
    // Not while minimised or in full screen (the state would be the screen's); maximised is kept as such.
    if ( m_owner == nullptr || m_window == nullptr || m_window->isMinimised() ||
         juce::Desktop::getInstance().getKioskModeComponent() != nullptr )
        return;
    m_owner->m_proc.userSettings().setValue( "windowState", m_window->getWindowStateAsString() );
}

void AdditivEditor::setCanvas( bool on )
{
    m_canvas = on;
    if ( on && m_helpView != nullptr && m_helpView->isVisible() ) showHelp( false );
    m_tabs.setVisible( !on );
    m_canvasButton.setToggleState( on, juce::dontSendNotification );
    m_cloud.setCanvas( on );
    resized();
}

/**
 * @brief Every ParamPage under a component, depth first.
 * @param[in,out] c    the component
 * @param[out]    out  receives them
 */
static void findParamPages( juce::Component *c, std::vector<ParamPage *> &out )
{
    if ( c == nullptr ) return;
    if ( auto *p = dynamic_cast<ParamPage *>( c ) ) out.push_back( p );
    for ( juce::Component *child : c->getChildren() ) findParamPages( child, out );
}

bool AdditivEditor::revealParam( int id )
{
    if ( m_canvas ) setCanvas( false );
    for ( int t = 0; t < m_tabs.getNumTabs(); ++t )
    {
        auto *view = dynamic_cast<juce::Viewport *>( m_tabs.getTabContentComponent( t ) );
        if ( view == nullptr ) continue;
        std::vector<ParamPage *> pages;
        findParamPages( view->getViewedComponent(), pages );
        for ( ParamPage *page : pages )
            if ( juce::Component *control = page->controlFor( id ) )
            {
                m_tabs.setCurrentTabIndex( t );
                const auto at = view->getViewedComponent()->getLocalPoint( control, juce::Point<int>( 0, 0 ) );
                view->setViewPosition( 0, std::max( 0, at.y - 80 ) );
                page->flash( id );
                if ( m_info != nullptr ) m_info->showParam( id );
                return true;
            }
    }
    // A modulator's knob: its source in the inspector, on the Modulation page.
    if ( const additiv::ModSourceE s = SourceInspector::sourceOf( id ); s != additiv::ModSourceE::OFF )
    {
        const int t = m_tabs.getTabNames().indexOf( "Modulation" );
        if ( t >= 0 ) m_tabs.setCurrentTabIndex( t );
        chooseSource( s, false );
        if ( m_info != nullptr ) m_info->showParam( id );
        return true;
    }
    return false;
}

void AdditivEditor::openPalette()
{
    if ( m_helpView != nullptr && m_helpView->isVisible() ) showHelp( false );
    // Words for how a preset sounds -- "dark calm wide", "hell luftig" -- find the presets measured so.
    m_palette.m_fromWords = [this]( const juce::StringArray &words )
    {
        struct Word
        {
            const char *m_text;    ///< the word
            int         m_trait;   ///< 0 brightness, 1 motion, 2 width, 3 noise
            float       m_way;     ///< -1 less, +1 more
        };
        static const Word Words[] = {
            { "dark", 0, -1 },    { "darker", 0, -1 },  { "dunkel", 0, -1 }, { "dusky", 0, -1 },   { "warm", 0, -1 },
            { "bright", 0, 1 },   { "brighter", 0, 1 }, { "hell", 0, 1 },    { "glassy", 0, 1 },   { "shiny", 0, 1 },
            { "calm", 1, -1 },    { "still", 1, -1 },   { "static", 1, -1 }, { "quiet", 1, -1 },   { "ruhig", 1, -1 },
            { "slow", 1, -1 },    { "moving", 1, 1 },   { "lively", 1, 1 },  { "busy", 1, 1 },     { "restless", 1, 1 },
            { "bewegt", 1, 1 },   { "lebendig", 1, 1 }, { "narrow", 2, -1 }, { "mono", 2, -1 },    { "schmal", 2, -1 },
            { "eng", 2, -1 },     { "wide", 2, 1 },     { "broad", 2, 1 },   { "spacious", 2, 1 }, { "breit", 2, 1 },
            { "weit", 2, 1 },     { "pure", 3, -1 },    { "clean", 3, -1 },  { "clear", 3, -1 },   { "rein", 3, -1 },
            { "klar", 3, -1 },    { "airy", 3, 1 },     { "noisy", 3, 1 },   { "breathy", 3, 1 },  { "luftig", 3, 1 },
            { "rauschig", 3, 1 },
        };
        static const char *const Said[4][2] = {
            { "dark", "bright" }, { "calm", "moving" }, { "narrow", "wide" }, { "pure", "airy" }
        };
        float want[4] = { 0, 0, 0, 0 };
        bool  asked = false;
        for ( const juce::String &w : words )
            for ( const Word &k : Words )
                if ( w == k.m_text )
                {
                    want[k.m_trait] = k.m_way;
                    asked = true;
                }
        std::vector<CommandPalette::Command> out;
        if ( !asked ) return out;
        juce::String said;
        for ( int t = 0; t < 4; ++t )
            if ( want[t] != 0.0f ) said << ( said.isEmpty() ? "" : ", " ) << Said[t][want[t] > 0.0f ? 1 : 0];
        // The nearest to a spread and a fifth that way in each trait asked for; the others free.
        std::vector<std::pair<float, int>> near;
        for ( int i = 0; i < additiv::numPresets(); ++i )
        {
            float f[4];
            if ( !additiv::presetFeatures( i, f ) ) continue;
            float d = 0.0f;
            for ( int t = 0; t < 4; ++t )
                if ( want[t] != 0.0f ) d += ( f[t] - 1.2f * want[t] ) * ( f[t] - 1.2f * want[t] );
            near.emplace_back( d, i );
        }
        const size_t n = std::min<size_t>( 10, near.size() );
        std::partial_sort( near.begin(), near.begin() + static_cast<std::ptrdiff_t>( n ), near.end() );
        for ( size_t k = 0; k < n; ++k )
        {
            const int              index = near[k].second;
            const additiv::Preset &p = additiv::preset( index );
            out.push_back( { p.m_name, "sounds " + said + "  (" + additiv::PresetGroupNames[p.m_group] + ")", "Preset",
                             false, [this, index]( const juce::String & ) { loadPreset( index ); } } );
        }
        return out;
    };
    m_palette.open( paletteCommands() );
}

std::vector<CommandPalette::Command> AdditivEditor::paletteCommands()
{
    std::vector<CommandPalette::Command> c;
    auto action = [&c]( const juce::String &title, const juce::String &detail, std::function<void()> f )
    { c.push_back( { title, detail, "Action", false, [f]( const juce::String & ) { f(); } } ); };
    action( "Canvas", "the picture alone, the whole window", [this] { setCanvas( !m_canvas ); } );
    action( "Play / Stop", "the conductor", [this] { m_play.triggerClick(); } );
    action( "Step", "exchange one voice of the chord now", [this] { m_proc.engine().autoplayStep(); } );
    action( "Save", "your own preset (Ctrl+S)", [this] { quickSave(); } );
    action( "Save as ...", "your own preset under a new name", [this] { saveAs(); } );
    action( "Back to the preset", "every knob where the preset had it",
            [this]
            {
                m_history.begin( "Back to the preset", values(), matrixText(), true );
                m_proc.revert();
                endStep();
            } );
    action( "Variations", "six near the sound (Focus page)",
            [this]
            {
                m_proc.makeVariations( 0.3f );
                m_tabs.setCurrentTabIndex( 0 );
            } );
    action( "Help", "the manual (F1)", [this] { showHelp( true ); } );
    action( "Night mode", "the window dim and warm for a dark room", [this] { setNight( !m_night ); } );
    action( "Tour", "the first steps again", [this] { startTour(); } );
    action( "Snapshot", "keep the sound now (Focus page)",
            [this] { m_proc.takeSnapshot( m_cloud.thumbnail( 160, 60 ) ); } );
    action( "Browse presets", "by name, group, star, recent, and how they sound", [this] { m_browser.open(); } );
    action( "Presets menu", "the bank's groups as a menu", [this] { showPresets(); } );
    for ( int t = 0; t < m_tabs.getNumTabs(); ++t )
        action( "Page: " + m_tabs.getTabNames()[t], "show it",
                [this, t]
                {
                    if ( m_canvas ) setCanvas( false );
                    m_tabs.setCurrentTabIndex( t );
                } );
    // Every knob, where it is: its page and its group.
    std::vector<juce::String> where( static_cast<size_t>( additiv::NumParams ) );
    for ( int t = m_tabs.getNumTabs() - 1; t >= 0; --t )
    {   // the first page that shows it wins
        auto *view = dynamic_cast<juce::Viewport *>( m_tabs.getTabContentComponent( t ) );
        if ( view == nullptr ) continue;
        std::vector<ParamPage *> pages;
        findParamPages( view->getViewedComponent(), pages );
        for ( ParamPage *page : pages )
            for ( int id = 0; id < additiv::NumParams; ++id )
                if ( page->controlFor( id ) != nullptr )
                    where[static_cast<size_t>( id )] = m_tabs.getTabNames()[t] + " / " + page->groupOf( id );
    }
    for ( int id = 0; id < additiv::NumParams; ++id )
    {
        if ( const additiv::ModSourceE s = SourceInspector::sourceOf( id );
             where[static_cast<size_t>( id )].isEmpty() && s != additiv::ModSourceE::OFF )
            where[static_cast<size_t>( id )] =
                juce::String( "Modulation / " ) + additiv::ModSourceNames[static_cast<int>( s )];
        if ( where[static_cast<size_t>( id )].isEmpty() ) continue;
        const additiv::ParamDesc &d = additiv::paramDesc( static_cast<ParamIdE>( id ) );
        const bool number = d.m_kind == additiv::ParamKindE::FLOAT || d.m_kind == additiv::ParamKindE::INT;
        c.push_back( { juce::String( d.m_name ), where[static_cast<size_t>( id )], "Knob", number,
                       [this, id]( const juce::String &value )
                       {
                           if ( value.isNotEmpty() )
                           {
                               m_history.begin( additiv::paramDesc( static_cast<ParamIdE>( id ) ).m_name, values(),
                                                matrixText(), true );
                               m_proc.setValue( id, value.getFloatValue() );
                               endStep();
                           }
                           revealParam( id );
                       } } );
    }
    // The player's presets, then the bank's.
    for ( size_t i = 0; i < m_proc.userPresets().size(); ++i )
        c.push_back( { m_proc.userPresets()[i].m_name, "your preset", "Yours", false, [this, i]( const juce::String & )
                       {
                           m_history.begin( "Preset", values(), matrixText(), true );
                           m_proc.chooseUserPreset( static_cast<int>( i ) );
                           endStep();
                       } } );
    for ( int i = 0; i < additiv::numPresets(); ++i )
        c.push_back( { juce::String( additiv::preset( i ).m_name ),
                       juce::String( additiv::PresetGroupNames[additiv::preset( i ).m_group] ), "Preset", false,
                       [this, i]( const juce::String & ) { loadPreset( i ); } } );
    return c;
}

void AdditivEditor::toggleFullScreen()
{
    if ( !standalone() ) return;
    auto &desk = juce::Desktop::getInstance();
    desk.setKioskModeComponent( desk.getKioskModeComponent() == nullptr ? getTopLevelComponent() : nullptr, false );
}

void AdditivEditor::showHelp( bool on )
{
    if ( on && m_helpView == nullptr )
    {
        m_helpView = std::make_unique<frame::HelpView>(
            juce::String::fromUTF8( AdditivData::chapters_txt, AdditivData::chapters_txtSize ), additivui::skin() );
        m_helpView->m_onClose = [this] { showHelp( false ); };
        m_helpView->m_extraTopics = [this]
        {
            std::vector<std::pair<juce::String, juce::String>> t;
            const int                                          i = m_tabs.getCurrentTabIndex();
            if ( i >= 0 && i < static_cast<int>( m_describe.size() ) )
                t.emplace_back( "This tab: " + m_tabs.getTabNames()[i], m_describe[static_cast<size_t>( i )]() );
            t.emplace_back( "Keys", frame::keysText() );
            t.emplace_back(
                "Headset (Meta Quest)",
                frame::headsetGrammar( "Step", "Brain 2 in or out" ) +
                    "\n\nThe left hand's height is the brightness (Macro 2), the right hand's the motion (Macro 1); both hands pinched: the next preset.\n\n" +
                    m_proc.headset().statusText() );
            return t;
        };
        addChildComponent( *m_helpView );
    }
    if ( m_helpView == nullptr ) return;
    m_help.setToggleState( on, juce::dontSendNotification );
    if ( on ) m_helpView->refresh();
    m_helpView->setVisible( on );
    resized();
    if ( on )
        m_helpView->grabKeyboardFocus();
    else
        grabKeyboardFocus();
}

void AdditivEditor::loadPreset( int index )
{
    m_history.begin( "Preset", values(), matrixText(), true );
    m_proc.choosePreset( index );
    endStep();
}

void AdditivEditor::showSaveMenu()
{
    juce::PopupMenu    m;
    const juce::String own = m_proc.currentUserPreset();
    m.addItem( own.isNotEmpty() ? "Save \"" + own + "\"  (Ctrl+S)" : juce::String( "Save  (Ctrl+S)" ),
               [this] { quickSave(); } );
    m.addItem( "Save as ...", [this] { saveAs(); } );
    m.addSeparator();
    m.addItem( "Back to the preset", m_proc.modified(), false,
               [this]
               {
                   m_history.begin( "Back to the preset", values(), matrixText(), true );
                   m_proc.revert();
                   endStep();
               } );
    m.addItem( "Open your presets' folder",
               []
               {
                   AdditivProcessor::presetFolder().createDirectory();
                   AdditivProcessor::presetFolder().startAsProcess();
               } );
    m.showMenuAsync( juce::PopupMenu::Options().withTargetComponent( &m_save ) );
}

void AdditivEditor::quickSave()
{
    const juce::String own = m_proc.currentUserPreset();
    if ( own.isEmpty() )
    {
        saveAs();
        return;
    }
    juce::String why;
    if ( !m_proc.saveUserPreset( own, why ) )
        juce::AlertWindow::showMessageBoxAsync( juce::MessageBoxIconType::WarningIcon, "Save", "Not saved: " + why );
}

void AdditivEditor::saveAs()
{
    const int          cp = m_proc.currentPreset();
    const juce::String suggestion = m_proc.currentUserPreset().isNotEmpty() ? m_proc.currentUserPreset() + " 2"
                                    : cp >= 0 ? juce::String( "My " ) + additiv::preset( cp ).m_name
                                              : juce::String( "My Sound" );
    m_askName = std::make_unique<juce::AlertWindow>(
        "Save as your own preset",
        "Its name (in Documents/Empyrean/Presets; journeys find it by it):", juce::MessageBoxIconType::NoIcon, this );
    m_askName->addTextEditor( "name", suggestion );
    m_askName->addButton( "Save", 1, juce::KeyPress( juce::KeyPress::returnKey ) );
    m_askName->addButton( "Cancel", 0, juce::KeyPress( juce::KeyPress::escapeKey ) );
    m_askName->enterModalState( true,
                                juce::ModalCallbackFunction::create(
                                    [this]( int result )
                                    {
                                        const juce::String name = m_askName != nullptr
                                                                      ? m_askName->getTextEditorContents( "name" )
                                                                      : juce::String();
                                        m_askName.reset();
                                        if ( result != 1 ) return;
                                        juce::String why;
                                        if ( !m_proc.saveUserPreset( name, why ) )
                                            juce::AlertWindow::showMessageBoxAsync(
                                                juce::MessageBoxIconType::WarningIcon, "Save", "Not saved: " + why );
                                    } ),
                                false );
}

void AdditivEditor::showPresets()
{
    juce::PopupMenu bank;
    const int       current = m_proc.currentPreset();
    m_proc.refreshUserPresets();
    if ( !m_proc.userPresets().empty() )
    {   // the player's own first
        bank.addSectionHeader( "Yours" );
        juce::PopupMenu yours;
        for ( size_t i = 0; i < m_proc.userPresets().size(); ++i )
            yours.addItem( 100000 + static_cast<int>( i ) + 1, m_proc.userPresets()[i].m_name, true,
                           m_proc.userPresets()[i].m_name == m_proc.currentUserPreset() );
        bank.addSubMenu( "Your presets", yours, true, nullptr, m_proc.currentUserPreset().isNotEmpty() );
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
            group.addItem( index + 1, additiv::preset( index ).m_name, true, index == current );
        }
        bank.addSubMenu( additiv::PresetGroupNames[g], group, true, nullptr,
                         current >= 0 && additiv::preset( current ).m_group == g );
    }
    bank.showMenuAsync( juce::PopupMenu::Options().withTargetComponent( &m_presetName ),
                        [this]( int id )
                        {
                            if ( id > 100000 )
                            {
                                m_history.begin( "Preset", values(), matrixText(), true );
                                m_proc.chooseUserPreset( id - 100001 );
                                endStep();
                            }
                            else if ( id > 0 )
                            {
                                loadPreset( id - 1 );
                            }
                        } );
}

void AdditivEditor::askPlace()
{
    double lat = 0.0, lon = 0.0;
    m_proc.engine().location( lat, lon );
    auto *w = new juce::AlertWindow( "Night arc", "Where the sun is read: degrees, north and east positive.",
                                     juce::MessageBoxIconType::NoIcon, this );
    w->addTextEditor( "lat", juce::String( lat, 4 ), "Latitude" );
    w->addTextEditor( "lon", juce::String( lon, 4 ), "Longitude" );
    w->addButton( "Keep", 1, juce::KeyPress( juce::KeyPress::returnKey ) );
    w->addButton( "Cancel", 0, juce::KeyPress( juce::KeyPress::escapeKey ) );
    w->enterModalState( true,
                        juce::ModalCallbackFunction::create(
                            [this, w]( int result )
                            {
                                if ( result == 1 )
                                    m_proc.setPlace( w->getTextEditorContents( "lat" ).getDoubleValue(),
                                                     w->getTextEditorContents( "lon" ).getDoubleValue() );
                            } ),
                        true );
}

void AdditivEditor::showSettings()
{
    frame::SettingsMenu m;
    m.m_app = App;
    m.m_version = JucePlugin_VersionString;
    m.m_canFullScreen = [this] { return standalone(); };
    m.m_isFullScreen = [] { return juce::Desktop::getInstance().getKioskModeComponent() != nullptr; };
    m.m_toggleFullScreen = [this] { toggleFullScreen(); };
    m.m_setWindowScale = [this]( float k )
    {
        if ( juce::Desktop::getInstance().getKioskModeComponent() == nullptr )
            setSize( juce::roundToInt( 1280.0f * k ), juce::roundToInt( 860.0f * k ) );
    };
    m.m_headsetStatus = [this] { return m_proc.headset().statusText(); };
    m.m_headsetItems = [this]( juce::PopupMenu &h )
    {   // Phase 30: the head and the hands in the cloud
        h.addItem( m_proc.pose().statusText(), false, false, nullptr );
    };
    m.m_jamStatus = [this] { return m_proc.jamStatus(); };   // the family jam: Off, Lead, Follow
    m.m_moreItems = [this]( juce::PopupMenu &menu )
    {
        juce::PopupMenu zoom;   // the controls larger, beside the window's size
        for ( int pct : { 100, 115, 125, 150 } )
            zoom.addItem( juce::String( pct ) + " %", true, juce::roundToInt( m_zoom * 100.0f ) == pct,
                          [this, pct] { setZoom( static_cast<float>( pct ) / 100.0f ); } );
        menu.addSubMenu( "Zoom", zoom );
        menu.addItem( "Night mode", true, m_night, [this] { setNight( !m_night ); } );
        {   // Phase 30: where the night arc reads the sun
            juce::PopupMenu place;
            double          lat = 0.0, lon = 0.0;
            m_proc.engine().location( lat, lon );
            float       sun = 0.0f;
            const float night = m_proc.engine().nightNow( sun );
            place.addItem( juce::String( std::fabs( lat ), 2 ) + ( lat >= 0.0 ? " N, " : " S, " ) +
                               juce::String( std::fabs( lon ), 2 ) + ( lon >= 0.0 ? " E" : " W" ) + ": the sun at " +
                               juce::String( sun, 0 ) + " degrees, night " + juce::String( night, 2 ),
                           false, false, nullptr );
            place.addItem( "Set the place...", [this] { askPlace(); } );
            place.addItem( "From the time zone", [this] { m_proc.placeFromTimeZone(); } );
            menu.addSubMenu( "Night arc", place );
        }
        menu.addItem( "The first steps (tour)", [this] { startTour(); } );
    };
    m.m_about = [this]
    {
        return juce::String( "The additive ambient synth: 20,000 partials per voice on the GPU, after Noctuary.\n" ) +
               ( m_proc.engine().prepared() ? m_proc.engine().deviceName() : juce::String( "GPU engine not running" ) );
    };
    m.show( m_settings );
}
