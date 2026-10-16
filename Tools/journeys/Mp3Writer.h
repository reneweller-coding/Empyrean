/**
 * @file Mp3Writer.h
 * @brief An MP3 file written through Windows' Media Foundation (its MPEG layer 3 encoder, Windows 8 and later): no
 *        encoder of our own to license, 16-bit PCM in (dithered from float), 320 kbit/s out.
 * @author Rene Weller
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct IMFSinkWriter;

namespace journeys
{

/** @brief Writes stereo float audio into an MP3 file. One thread; open, write, close. */
class Mp3Writer
{
public:
    Mp3Writer() = default;
    /** @brief Closes the file if it is still open. */
    ~Mp3Writer();
    Mp3Writer( const Mp3Writer & ) = delete;
    /** @brief Not copied. @return this */
    Mp3Writer &operator=( const Mp3Writer & ) = delete;

    /**
     * @brief Opens a file for writing.
     * @param[in]  path        the file (UTF-16, as Windows takes it)
     * @param[in]  sampleRate  44100 or 48000
     * @param[in]  kbps        the bit rate (320 at most)
     * @param[out] why         receives why it failed
     * @return true when it is open
     */
    bool open( const std::wstring &path, int sampleRate, int kbps, std::string &why );

    /**
     * @brief Appends samples, dithered (triangular) to 16 bit.
     * @param[in] L  left
     * @param[in] R  right
     * @param[in] n  samples
     * @return false when Media Foundation refused them
     */
    bool write( const float *L, const float *R, int n );

    /**
     * @brief Finishes the file (the encoder flushed, the headers written).
     * @return true when it was finished cleanly
     */
    bool close();

private:
    IMFSinkWriter            *m_writer = nullptr;      ///< the sink writer, null when closed
    unsigned long             m_stream = 0;            ///< its stream
    long long                 m_time = 0;              ///< samples written so far
    int                       m_rate = 48000;          ///< samples a second
    std::vector<std::int16_t> m_pcm;                   ///< the interleaved 16-bit block on its way
    std::uint32_t             m_dither = 0x1234567u;   ///< the dither's generator
    bool                      m_com = false;           ///< this thread's COM was initialised here
};

}   // namespace journeys
