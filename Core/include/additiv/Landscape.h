/**
 * @file Landscape.h
 * @brief Landscapes of time against frequency: an envelope for every partial at once, read at the
 *        partial's own frequency. Painted (any image: time left to right, low frequencies at the
 *        bottom), analysed from a recording, or generated.
 *
 * Brightness and amplitude meet on a decibel scale: black is silence, white full amplitude, and
 * the grey between spans `rangeDb` -- so a picture of a landscape (writeLandscapeBmp) can be
 * painted over and read back (landscapeFromImage) with the same range.
 */
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace additiv {

/** @brief A landscape on the host: rows of time, each `columns` amplitudes from kLandLowHz to kLandHighHz. */
struct Landscape {
    int columns = 0;            ///< frequency columns, logarithmically spaced
    int rows = 0;               ///< time rows; the landscape loops over all of them
    float rowSeconds = 0.1f;    ///< how long one row lasts at rate 1
    std::vector<float> amps;    ///< rows * columns amplitudes 0 .. 1, row by row
};

/** @brief The centre frequency of a column. @param column 0 .. columns-1 @param columns how many @return Hz */
float landscapeColumnHz(int column, int columns);

/** @brief Brightness to amplitude on the decibel scale. @param b 0 .. 1 @param rangeDb the range grey spans @return 0 .. 1 */
float brightnessToAmp(float b, float rangeDb);

/** @brief Amplitude to brightness on the decibel scale. @param a 0 .. 1 @param rangeDb the range grey spans @return 0 .. 1 */
float ampToBrightness(float a, float rangeDb);

/**
 * @brief A generated landscape: slowly swaying bands, one band that climbs through the spectrum
 *        and wraps (Shepard-like), a fluttering high band, over a faint floor. Loops seamlessly.
 * @param loopSeconds  its length
 * @param rangeDb      the decibel range of its brightness
 * @param columns      frequency columns
 * @param rows         time rows
 * @return             the landscape
 */
Landscape landscapeAurora(float loopSeconds, float rangeDb = 40.0f, int columns = 256, int rows = 1024);

constexpr int kNumFactoryLandscapes = 5;                         ///< entries of kLandscapeNames
extern const char* const kLandscapeNames[kNumFactoryLandscapes];  ///< Aurora, Shepard, Tides, Rain, Choir

/**
 * @brief One of the generated landscapes, each looping seamlessly over a floor (motion on a carrying
 *        sound, the user's rule of 12.10.2026):
 *        Aurora -- swaying bands, one climbing and wrapping, a fluttering high band (30 s);
 *        Shepard -- four bands climbing for ever, fading in at the bottom and out at the top (40 s);
 *        Tides -- broad swells rolling up through the spectrum (60 s);
 *        Rain -- sixty short glints at chance places (30 s);
 *        Choir -- the formants of u-o-a-e-i, the vowel turning slowly (48 s).
 * @param index    0 .. kNumFactoryLandscapes-1
 * @param rangeDb  the decibel range of the brightness
 * @return         the landscape (Aurora for an unknown index)
 */
Landscape factoryLandscape(int index, float rangeDb = 40.0f);

/**
 * @brief The RMS of a landscape's amplitudes: how much quieter a voice gets that reads it fully.
 * @param l  the landscape
 * @return   sqrt of the mean squared amplitude (1 for an empty one)
 */
float landscapeRms(const Landscape& l);

/**
 * @brief A landscape from a picture (PNG, JPEG, BMP, TIFF, GIF -- whatever Windows can decode):
 *        time runs left to right over `loopSeconds`, frequency from the bottom (20 Hz) to the top
 *        (20 kHz); brightness is amplitude on the decibel scale.
 * @param path         the image
 * @param loopSeconds  how long the picture lasts from left to right
 * @param rangeDb      the decibel range of its brightness
 * @param out          receives the landscape
 * @param error        receives the reason when it fails (may be null)
 * @return             true when read
 */
bool landscapeFromImage(const std::filesystem::path& path, float loopSeconds, float rangeDb, Landscape& out, std::string* error);

/**
 * @brief A landscape from a recording (WAV or FLAC): a short-time spectrum every 21 ms or so,
 *        folded into logarithmic columns, normalised and made to loop.
 * @param path      the recording
 * @param rangeDb   what lies more than this below the loudest point is silence
 * @param whiten    0 .. 1: how far each column is divided by its own average, so the landscape
 *                  carries the recording's motion rather than its colour
 * @param seamless  0 .. 0.5: the share of the end crossfaded into the beginning, so the loop does not jump
 * @param smooth    seconds over which each column is smoothed (around the loop): the short-time
 *                  spectrum of noise flickers from frame to frame, and every member of a cloud
 *                  reads nearly the same column, so unsmoothed they would flicker together
 * @param out       receives the landscape
 * @param error     receives the reason when it fails (may be null)
 * @return          true when read
 */
bool landscapeFromAudio(const std::filesystem::path& path, float rangeDb, float whiten, float seamless, float smooth,
                        Landscape& out, std::string* error);

/**
 * @brief Crossfades the end of a landscape into its beginning and drops the end, so it loops without a jump.
 * @param l         the landscape
 * @param fraction  0 .. 0.5: the share of rows crossfaded
 */
void makeSeamless(Landscape& l, float fraction);

/**
 * @brief A picture of a landscape as a grey 24-bit BMP: time left to right, low frequencies at the
 *        bottom, amplitude as brightness on the decibel scale (the format landscapeFromImage reads).
 * @param l        the landscape
 * @param path     the file
 * @param rangeDb  the decibel range of the brightness
 * @param error    receives the reason when it fails (may be null)
 * @return         true when written
 */
bool writeLandscapeBmp(const Landscape& l, const std::filesystem::path& path, float rangeDb, std::string* error);

} // namespace additiv
