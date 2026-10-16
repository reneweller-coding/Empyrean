/**
 * @file JourneyRender.h
 * @brief Empyrean Journeys: a journey rendered into a file, faster than real time -- the engine offline, the journey's
 *        steps applied as the instrument applies them (each preset glided into, the conductor playing on), written as
 *        WAV, FLAC, Ogg Vorbis or MP3.
 *
 * The journeys are the instrument's own (Journeys.inc) and the player's files in Documents/Empyrean/Journeys; their
 * presets the bank's and the player's own (.empyrean files in Documents/Empyrean/Presets, its subfolders too). A journey
 * draws its dwells and glides from ranges; a seed fixes them, so the length of a rendering is known before it begins
 * (planLength) and the file holds exactly that.
 * @author Rene Weller
 */
#pragma once

#include "additiv/Journey.h"

#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace journeys
{

/** @brief The file formats a journey is written in. */
enum class FormatE : int
{
    WAV = 0,   ///< 24-bit PCM (RF64 beyond 4 GB)
    FLAC,      ///< 24-bit, lossless
    OGG,       ///< Ogg Vorbis, about 320 kbit/s
    MP3,       ///< MPEG layer 3, 320 kbit/s (Windows' own encoder)
    COUNT      ///< number of formats
};

/**
 * @brief A format's name for the menu.
 * @param[in] f  the format
 * @return the name
 */
const char *formatName( FormatE f );

/**
 * @brief A format's file extension, with its dot.
 * @param[in] f  the format
 * @return ".wav", ".flac", ".ogg" or ".mp3"
 */
const char *formatExtension( FormatE f );

/** @brief A journey to choose: one of the instrument's, or a file of the player's. */
struct JourneyItem
{
    juce::String     m_name;             ///< as the list shows it
    additiv::Journey m_journey;          ///< its steps
    bool             m_own = false;      ///< a file of the player's (else one of the instrument's)
    juce::String     m_kind;             ///< "Whole night", "Night", "Journey" or "Yours"
    double           m_shortest = 0.0;   ///< s: a pass with every dwell at its shortest
    double           m_mean = 0.0;       ///< s: a pass at the middle of every range
    double           m_longest = 0.0;    ///< s: a pass with every dwell at its longest
};

/**
 * @brief The journeys to choose from: the instrument's, then the player's files (Documents/Empyrean/Journeys).
 * @return them, each with its lengths worked out
 */
std::vector<JourneyItem> listJourneys();

/**
 * @brief How long a rendering of a journey lasts: its dwells as the seed draws them, for so many passes (one for a
 *        journey that does not cycle) -- exactly what render() then writes.
 * @param[in] journey  the journey
 * @param[in] seed     the dice (JourneyPlayer's)
 * @param[in] passes   passes of a cyclic journey (at least one)
 * @return seconds
 */
double planLength( const additiv::Journey &journey, std::uint64_t seed, int passes );

/**
 * @brief Seconds as h:mm:ss (or m:ss under an hour).
 * @param[in] seconds  the time
 * @return the text
 */
juce::String clockText( double seconds );

/** @brief One rendering: which journey, how, into which file. */
struct RenderJob
{
    JourneyItem   m_item;                    ///< the journey
    std::uint64_t m_seed = 1;                ///< the dice for its dwells and glides
    int           m_passes = 1;              ///< passes of a cyclic journey
    FormatE       m_format = FormatE::WAV;   ///< the file's format
    juce::File    m_file;                    ///< where it goes
    double        m_fade = 20.0;             ///< s: faded out over the end (0: not)
    bool          m_night = false;           ///< the night arc as at 23:00 tonight (else the clock's hour)
    bool          m_exact = false;           ///< GPU blocks of 128 as the instrument plays (else 512: about 3x faster)
    double        m_limit = 0.0;             ///< s: at most this much (0: the whole; the tests ask for seconds)
};

/** @brief Renders jobs, one at a time, and tells how far it is (any thread may read the counters). */
class Renderer
{
public:
    /**
     * @brief Renders a job into its file, as fast as the card allows. Blocks until it is done or cancelled.
     * @param[in]  job        the rendering
     * @param[in]  cancelled  asked between blocks: true stops it (the file is closed as far as it went)
     * @param[out] why        receives why it failed
     * @return true when the whole was written
     */
    bool render( const RenderJob &job, const std::function<bool()> &cancelled, juce::String &why );

    std::atomic<double> m_done{ 0.0 };    ///< s rendered of the job under way
    std::atomic<double> m_total{ 0.0 };   ///< s it will last
    std::atomic<double> m_speed{ 0.0 };   ///< how many times faster than real time, so far
    std::atomic<int>    m_step{ -1 };     ///< the journey's step under way
};

/**
 * @brief Renders seconds of the first of the instrument's journeys into every format in a temporary folder, reads each
 *        back and checks its length and level; prints what it found (the ctest `journeys`).
 * @param[in] seconds  how long each file
 * @return 0 when every format passed
 */
int selfTest( double seconds );

}   // namespace journeys
