/**
 * @file Main.cpp
 * @brief Empyrean Journeys: a small program beside the instrument that renders its journeys into files (WAV, FLAC, Ogg
 *        Vorbis, MP3), faster than real time, each shown with its length.
 *
 *     EmpyreanJourneys                                     the window
 *     EmpyreanJourneys --selftest [S]                      every format, S seconds each, and the queue (ctest journeys)
 *     EmpyreanJourneys --render NAME [--format wav|flac|ogg|mp3] [--out FILE] [--seed N] [--passes P]
 *                      [--seconds S] [--night] [--exact]   one journey without the window
 *     EmpyreanJourneys --shot FILE [--journey NAME]        the window as a PNG, a journey chosen (the manual's)
 * @author Rene Weller
 */
#include "JourneysComponent.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <cstdio>

namespace
{

/**
 * @brief The command line's value after a flag.
 * @param[in] args  the arguments
 * @param[in] flag  the flag
 * @param[in] dflt  when it is not there
 * @return the value
 */
juce::String option( const juce::StringArray &args, const juce::String &flag, const juce::String &dflt )
{
    const int i = args.indexOf( flag );
    return i >= 0 && i + 1 < args.size() ? args[i + 1] : dflt;
}

/**
 * @brief One journey rendered from the command line, its progress printed.
 * @param[in] args  the arguments
 * @return 0 when it was written
 */
int renderFromCommandLine( const juce::StringArray &args )
{
    using namespace journeys;
    const juce::String name = option( args, "--render", {} );
    for ( const JourneyItem &item : listJourneys() )
    {
        if ( item.m_name != name ) continue;
        RenderJob          job;
        const juce::String f = option( args, "--format", "wav" ).toLowerCase();
        job.m_format = f == "flac"  ? FormatE::FLAC
                       : f == "ogg" ? FormatE::OGG
                       : f == "mp3" ? FormatE::MP3
                                    : FormatE::WAV;
        job.m_item = item;
        job.m_seed = static_cast<std::uint64_t>( option( args, "--seed", "1" ).getLargeIntValue() );
        job.m_passes = option( args, "--passes", "1" ).getIntValue();
        job.m_limit = option( args, "--seconds", "0" ).getDoubleValue();
        job.m_night = args.contains( "--night" );
        job.m_exact = args.contains( "--exact" );
        job.m_file = juce::File::getCurrentWorkingDirectory().getChildFile(
            option( args, "--out", juce::File::createLegalFileName( name ) + formatExtension( job.m_format ) ) );
        Renderer     r;
        juce::String why;
        std::printf( "%s: %s into %s\n", name.toRawUTF8(),
                     clockText( planLength( item.m_journey, job.m_seed, job.m_passes ) ).toRawUTF8(),
                     job.m_file.getFullPathName().toRawUTF8() );
        const bool ok = r.render( job, [] { return false; }, why );
        std::printf( ok ? "%s written, %.1fx real time\n" : "failed: %s\n",
                     ok ? clockText( r.m_total.load() ).toRawUTF8() : why.toRawUTF8(), r.m_speed.load() );
        return ok ? 0 : 1;
    }
    std::printf( "no journey %s\n", name.toRawUTF8() );
    return 1;
}

/**
 * @brief The window's queue as the window drives it: two journeys rendered one after the other, then one cancelled
 *        under way (its unfinished file removed); prints what it found (part of --selftest).
 * @return 0 when the queue did all three right
 */
int queueTest()
{
    using namespace journeys;
    const std::vector<JourneyItem> items = listJourneys();
    if ( items.size() < 2 ) return 1;
    const juce::File dir =
        juce::File::getSpecialLocation( juce::File::tempDirectory ).getChildFile( "EmpyreanJourneysQueue" );
    dir.deleteRecursively();
    dir.createDirectory();
    std::vector<RenderJob> jobs( 2 );
    for ( size_t k = 0; k < jobs.size(); ++k )
    {
        jobs[k].m_item = items[k];
        jobs[k].m_limit = 3.0;
        jobs[k].m_file = dir.getChildFile( "q" + juce::String( static_cast<int>( k ) ) + ".wav" );
    }
    RenderQueue  q;
    juce::String name, message;
    int          waiting = 0, finished = 0;
    q.add( jobs );
    for ( int t = 0; t < 1200 && ( q.status( name, waiting, finished ) || waiting > 0 || finished < 2 ); ++t )
        juce::Thread::sleep( 50 );
    message = q.takeMessage();
    const bool both = finished == 2 && jobs[0].m_file.existsAsFile() && jobs[1].m_file.existsAsFile();
    RenderJob  long_ = jobs[0];
    long_.m_limit = 600.0;
    long_.m_file = dir.getChildFile( "cancelled.wav" );
    q.add( { long_ } );
    for ( int t = 0; t < 800 && ( q.m_renderer.m_total.load() < 100.0 || q.m_renderer.m_done.load() < 1.0 ); ++t )
        juce::Thread::sleep( 25 );   // until the long one is a second in
    q.cancelAll();
    for ( int t = 0; t < 400 && q.status( name, waiting, finished ); ++t ) juce::Thread::sleep( 25 );
    const juce::String last = q.takeMessage();
    const bool         cancelled = !q.status( name, waiting, finished ) && last.endsWith( "cancelled" ) &&
                                   !long_.m_file.existsAsFile() && finished == 2;
    std::printf( "  %s  queue: two after each other (%s), one cancelled under way (%s)\n",
                 both && cancelled ? "ok  " : "FAIL", message.toRawUTF8(), last.toRawUTF8() );
    dir.deleteRecursively();
    return both && cancelled ? 0 : 1;
}

/** @brief The window. */
class JourneysWindow : public juce::DocumentWindow
{
public:
    JourneysWindow()
        : juce::DocumentWindow( "Empyrean Journeys", juce::Colour( 0xff07090f ), juce::DocumentWindow::allButtons )
    {
        setUsingNativeTitleBar( true );
        setContentOwned( new journeys::JourneysComponent(), true );
        setResizable( true, true );
        setResizeLimits( 900, 560, 4000, 3000 );
        centreWithSize( getWidth(), getHeight() );
        setVisible( true );
    }
    /** @brief The close button quits. */
    void closeButtonPressed() override { juce::JUCEApplication::getInstance()->systemRequestedQuit(); }
};

/** @brief The application: the window, or a command line's work. */
class JourneysApplication : public juce::JUCEApplication
{
public:
    /** @brief Its name. @return "Empyrean Journeys" */
    const juce::String getApplicationName() override { return "Empyrean Journeys"; }
    /** @brief Its version. @return the project's */
    const juce::String getApplicationVersion() override { return ADDITIV_VERSION; }
    /** @brief Several at once: a command line's rendering beside an open window (else it would hand its arguments to the
     *         window and quit). @return true */
    bool moreThanOneInstanceAllowed() override { return true; }
    /**
     * @brief Opens the window, or does what the command line asks and quits.
     * @param[in] commandLine  the arguments
     */
    void initialise( const juce::String &commandLine ) override
    {
        juce::StringArray args = juce::StringArray::fromTokens( commandLine, true );
        for ( juce::String &a : args ) a = a.unquoted();   // a name with spaces comes in its quotes
        if ( args.contains( "--selftest" ) )
        {
            const double seconds = option( args, "--selftest", "6" ).getDoubleValue();
            const int    formats = journeys::selfTest( seconds > 0.0 ? seconds : 6.0 );
            setApplicationReturnValue( queueTest() != 0 || formats != 0 ? 1 : 0 );
            quit();
            return;
        }
        if ( args.contains( "--render" ) )
        {
            setApplicationReturnValue( renderFromCommandLine( args ) );
            quit();
            return;
        }
        if ( args.contains( "--shot" ) )
        {   // the window as a picture, a journey chosen (the manual's)
            journeys::JourneysComponent c;
            c.choose( option( args, "--journey", "Whole Night - Lullabies" ) );
            const juce::Image shot = c.createComponentSnapshot( c.getLocalBounds() );
            juce::File        out =
                juce::File::getCurrentWorkingDirectory().getChildFile( option( args, "--shot", "journeys.png" ) );
            out.deleteFile();
            juce::FileOutputStream s( out );
            juce::PNGImageFormat   png;
            setApplicationReturnValue( s.openedOk() && png.writeImageToStream( shot, s ) ? 0 : 1 );
            quit();
            return;
        }
        juce::LookAndFeel::getDefaultLookAndFeel().setUsingNativeAlertWindows( true );
        m_window = std::make_unique<JourneysWindow>();
    }
    /** @brief Closes the window. */
    void shutdown() override { m_window.reset(); }
    /** @brief Quits. */
    void systemRequestedQuit() override { quit(); }

private:
    std::unique_ptr<JourneysWindow> m_window;   ///< the window, when there is one
};

}   // namespace

START_JUCE_APPLICATION( JourneysApplication )
