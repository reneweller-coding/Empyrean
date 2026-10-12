/**
 * @file Landscape.cpp
 * @brief Landscapes: generated, read from pictures (Windows Imaging Component), analysed from
 *        recordings (WAV here, FLAC through dr_flac), and drawn as pictures.
 */
#include "additiv/Landscape.h"

#include "additiv/Cloud.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>

#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#include "dr_flac.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace fs = std::filesystem;

namespace additiv {
namespace {

constexpr double kPi = 3.14159265358979323846;   ///< pi

/** @brief Writes a reason into `error`. @param error the target (may be null) @param what the reason @return false */
bool fail(std::string* error, const std::string& what)
{
    if (error) *error = what;
    return false;
}

/** @brief A whole file into memory. @param path the file @param data receives its bytes @return true when read */
bool readFile(const fs::path& path, std::vector<unsigned char>& data)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize size = f.tellg();
    if (size <= 0) return false;
    data.resize(static_cast<size_t>(size));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return static_cast<bool>(f);
}

/** @brief A little-endian 16-bit value. @param p its bytes @return the value */
std::uint32_t le16(const unsigned char* p) { return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8); }

/** @brief A little-endian 32-bit value. @param p its bytes @return the value */
std::uint32_t le32(const unsigned char* p) { return le16(p) | (le16(p + 2) << 16); }

/**
 * @brief A WAV file (PCM 8/16/24/32 bit or 32-bit float, plain or extensible) as one mono channel.
 * @param d      the file's bytes
 * @param mono   receives the channels' average
 * @param rate   receives the sample rate
 * @param error  receives the reason when it fails
 * @return       true when decoded
 */
bool decodeWav(const std::vector<unsigned char>& d, std::vector<float>& mono, double& rate, std::string* error)
{
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) != 0 || std::memcmp(d.data() + 8, "WAVE", 4) != 0)
        return fail(error, "not a RIFF/WAVE file");
    std::uint32_t format = 0, channels = 0, sampleRate = 0, bits = 0;
    const unsigned char* data = nullptr;
    size_t dataBytes = 0;
    size_t pos = 12;
    while (pos + 8 <= d.size()) {
        const unsigned char* id = d.data() + pos;
        const size_t size = le32(id + 4);
        const size_t body = pos + 8;
        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16 && body + 16 <= d.size()) {
            format = le16(d.data() + body);
            channels = le16(d.data() + body + 2);
            sampleRate = le32(d.data() + body + 4);
            bits = le16(d.data() + body + 14);
            if (format == 0xFFFE && size >= 26 && body + 26 <= d.size()) format = le16(d.data() + body + 24);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = d.data() + std::min(body, d.size());
            dataBytes = std::min(size, d.size() - std::min(body, d.size()));
        }
        pos = body + size + (size & 1);
    }
    const bool pcm = format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
    const bool flt = format == 3 && bits == 32;
    if (!data || channels == 0 || sampleRate == 0 || !(pcm || flt)) return fail(error, "unsupported WAV format (PCM 8/16/24/32 or float 32 only)");
    const size_t bytes = bits / 8;
    const size_t frames = dataBytes / (bytes * channels);
    mono.assign(frames, 0.0f);
    for (size_t f = 0; f < frames; ++f) {
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < channels; ++c) {
            const unsigned char* s = data + (f * channels + c) * bytes;
            float x = 0.0f;
            if (flt) { std::memcpy(&x, s, 4); }
            else if (bits == 8) x = (static_cast<float>(s[0]) - 128.0f) / 128.0f;
            else if (bits == 16) x = static_cast<float>(static_cast<std::int16_t>(le16(s))) / 32768.0f;
            else if (bits == 24) x = static_cast<float>(static_cast<std::int32_t>((static_cast<std::uint32_t>(s[0]) << 8) | (static_cast<std::uint32_t>(s[1]) << 16) | (static_cast<std::uint32_t>(s[2]) << 24)) >> 8) / 8388608.0f;
            else x = static_cast<float>(static_cast<std::int32_t>(le32(s))) / 2147483648.0f;
            sum += x;
        }
        mono[f] = sum / static_cast<float>(channels);
    }
    rate = sampleRate;
    return true;
}

/**
 * @brief A recording (WAV or FLAC, by its contents) as one mono channel.
 * @param path   the file
 * @param mono   receives the channels' average
 * @param rate   receives the sample rate
 * @param error  receives the reason when it fails
 * @return       true when read
 */
bool loadMono(const fs::path& path, std::vector<float>& mono, double& rate, std::string* error)
{
    std::vector<unsigned char> bytes;
    if (!readFile(path, bytes)) return fail(error, "cannot read " + path.string());
    if (bytes.size() >= 4 && std::memcmp(bytes.data(), "fLaC", 4) == 0) {
        unsigned int channels = 0, sampleRate = 0;
        drflac_uint64 frames = 0;
        float* pcm = drflac_open_memory_and_read_pcm_frames_f32(bytes.data(), bytes.size(), &channels, &sampleRate, &frames, nullptr);
        if (!pcm || channels == 0) { if (pcm) drflac_free(pcm, nullptr); return fail(error, "cannot decode the FLAC file " + path.string()); }
        mono.assign(static_cast<size_t>(frames), 0.0f);
        for (size_t f = 0; f < mono.size(); ++f) {
            float sum = 0.0f;
            for (unsigned int c = 0; c < channels; ++c) sum += pcm[f * channels + c];
            mono[f] = sum / static_cast<float>(channels);
        }
        drflac_free(pcm, nullptr);
        rate = sampleRate;
        return true;
    }
    return decodeWav(bytes, mono, rate, error);
}

/**
 * @brief An in-place radix-2 FFT.
 * @param a  the signal, a power of two long; receives its spectrum
 */
void fft(std::vector<std::complex<float>>& a)
{
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * kPi / static_cast<double>(len);
        for (size_t k = 0; k < len / 2; ++k) {
            const std::complex<float> w(static_cast<float>(std::cos(ang * static_cast<double>(k))), static_cast<float>(std::sin(ang * static_cast<double>(k))));
            for (size_t i = 0; i < n; i += len) {
                const std::complex<float> u = a[i + k];
                const std::complex<float> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }
}

/** @brief Releases a COM interface when it goes out of scope. */
template <class T>
struct Com {
    T* p = nullptr;                                ///< the interface, null until created
    ~Com() { if (p) p->Release(); }
    Com() = default;                               ///< empty
    Com(const Com&) = delete;                      ///< owns a reference
    Com& operator=(const Com&) = delete;           ///< owns a reference
    T* operator->() const { return p; }            ///< the interface's members
};

/**
 * @brief Decodes the first frame of a picture to 8-bit grey with the Windows Imaging Component.
 * @param path    the picture
 * @param grey    receives width * height brightness bytes, top row first
 * @param width   receives the width
 * @param height  receives the height
 * @param error   receives the reason when it fails
 * @return        true when decoded
 */
bool decodeGrey(const fs::path& path, std::vector<unsigned char>& grey, UINT& width, UINT& height, std::string* error)
{
    Com<IWICImagingFactory> factory;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory.p))))
        return fail(error, "the Windows Imaging Component is not available");
    Com<IWICBitmapDecoder> decoder;
    if (FAILED(factory->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder.p)))
        return fail(error, "cannot decode the picture " + path.string());
    Com<IWICBitmapFrameDecode> frame;
    if (FAILED(decoder->GetFrame(0, &frame.p))) return fail(error, "the picture has no frame");
    Com<IWICFormatConverter> converter;
    if (FAILED(factory->CreateFormatConverter(&converter.p))
        || FAILED(converter->Initialize(frame.p, GUID_WICPixelFormat8bppGray, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)))
        return fail(error, "cannot convert the picture to grey");
    if (FAILED(converter->GetSize(&width, &height)) || width == 0 || height == 0) return fail(error, "the picture is empty");
    grey.resize(static_cast<size_t>(width) * height);
    if (FAILED(converter->CopyPixels(nullptr, width, static_cast<UINT>(grey.size()), grey.data()))) return fail(error, "cannot read the picture's pixels");
    return true;
}

} // namespace

float landscapeColumnHz(int column, int columns)
{
    const float x = columns > 1 ? static_cast<float>(column) / static_cast<float>(columns - 1) : 0.0f;
    return kLandLowHz * std::pow(kLandHighHz / kLandLowHz, x);
}

float brightnessToAmp(float b, float rangeDb)
{
    if (b <= 0.0f) return 0.0f;
    return std::pow(10.0f, (std::min(b, 1.0f) - 1.0f) * rangeDb / 20.0f);
}

float ampToBrightness(float a, float rangeDb)
{
    if (a <= 0.0f) return 0.0f;
    return std::clamp(1.0f + 20.0f * std::log10(a) / rangeDb, 0.0f, 1.0f);
}

Landscape landscapeAurora(float loopSeconds, float rangeDb, int columns, int rows)
{
    Landscape l;
    l.columns = std::clamp(columns, 2, kLandMaxColumns);
    l.rows = std::clamp(rows, 1, kLandMaxRows);
    l.rowSeconds = loopSeconds / static_cast<float>(l.rows);
    l.amps.resize(static_cast<size_t>(l.columns) * l.rows);
    auto band = [](float y, float centre, float width) { const float d = (y - centre) / width; return std::exp(-0.5f * d * d); };
    const float tau = 2.0f * static_cast<float>(kPi);
    for (int r = 0; r < l.rows; ++r) {
        // Every motion turns a whole number of times per loop, so the loop has no seam.
        const float t = static_cast<float>(r) / static_cast<float>(l.rows);
        const float y1 = 0.35f + 0.15f * std::sin(tau * (t + 0.1f));
        const float s1 = 0.9f * (0.6f + 0.4f * std::sin(tau * 2.0f * t));
        const float y2 = 0.6f + 0.12f * std::sin(tau * (2.0f * t + 0.4f));
        const float s2 = 0.8f * (0.5f + 0.5f * std::sin(tau * (t + 0.6f)));
        float climb = 2.0f * t + 0.2f;
        climb -= std::floor(climb);
        const float y3 = 0.1f + 0.85f * climb;
        const float s3 = std::sin(static_cast<float>(kPi) * climb);   // fades in at the bottom and out at the top
        const float s4 = 0.5f * (0.5f + 0.5f * std::sin(tau * 6.0f * t));
        for (int c = 0; c < l.columns; ++c) {
            const float y = static_cast<float>(c) / static_cast<float>(l.columns - 1);
            float b = 0.25f;
            b = std::max(b, s1 * band(y, y1, 0.04f));
            b = std::max(b, s2 * band(y, y2, 0.03f));
            b = std::max(b, s3 * band(y, y3, 0.025f));
            b = std::max(b, s4 * band(y, 0.82f, 0.05f));
            l.amps[static_cast<size_t>(r) * l.columns + c] = brightnessToAmp(b, rangeDb);
        }
    }
    return l;
}

bool landscapeFromImage(const fs::path& path, float loopSeconds, float rangeDb, Landscape& out, std::string* error)
{
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    std::vector<unsigned char> grey;
    UINT w = 0, h = 0;
    const bool decoded = decodeGrey(path, grey, w, h, error);
    if (SUCCEEDED(init)) CoUninitialize();
    if (!decoded) return false;

    Landscape l;
    l.columns = std::clamp(static_cast<int>(h), 2, kLandMaxColumns);
    l.rows = std::clamp(static_cast<int>(w), 1, kLandMaxRows);
    l.rowSeconds = std::max(loopSeconds, 0.01f) / static_cast<float>(l.rows);
    l.amps.resize(static_cast<size_t>(l.columns) * l.rows);
    auto pixel = [&](int x, int y) { return static_cast<float>(grey[static_cast<size_t>(std::clamp(y, 0, static_cast<int>(h) - 1)) * w + std::clamp(x, 0, static_cast<int>(w) - 1)]) / 255.0f; };
    for (int r = 0; r < l.rows; ++r) {
        const float x = (static_cast<float>(r) + 0.5f) * static_cast<float>(w) / static_cast<float>(l.rows) - 0.5f;
        const int x0 = static_cast<int>(std::floor(x));
        const float fx = x - static_cast<float>(x0);
        for (int c = 0; c < l.columns; ++c) {
            // The top of the picture is the top of the spectrum.
            const float y = static_cast<float>(h) - 0.5f - (static_cast<float>(c) + 0.5f) * static_cast<float>(h) / static_cast<float>(l.columns);
            const int y0 = static_cast<int>(std::floor(y));
            const float fy = y - static_cast<float>(y0);
            const float top = pixel(x0, y0) + (pixel(x0 + 1, y0) - pixel(x0, y0)) * fx;
            const float bottom = pixel(x0, y0 + 1) + (pixel(x0 + 1, y0 + 1) - pixel(x0, y0 + 1)) * fx;
            l.amps[static_cast<size_t>(r) * l.columns + c] = brightnessToAmp(top + (bottom - top) * fy, rangeDb);
        }
    }
    out = std::move(l);
    return true;
}

bool landscapeFromAudio(const fs::path& path, float rangeDb, float whiten, float seamless, float smooth,
                        Landscape& out, std::string* error)
{
    std::vector<float> mono;
    double rate = 48000.0;
    if (!loadMono(path, mono, rate, error)) return false;
    constexpr int kN = 4096;
    if (mono.size() < static_cast<size_t>(kN)) mono.resize(kN, 0.0f);
    const size_t span = mono.size() - kN;
    const size_t hop = std::max<size_t>(1024, (span + kLandMaxRows - 2) / (kLandMaxRows - 1));
    const int frames = static_cast<int>(1 + span / hop);
    constexpr int kColumns = 256;

    // Each column's band in FFT bins: half a column either side of its centre.
    const double halfStep = std::sqrt(std::pow(static_cast<double>(kLandHighHz / kLandLowHz), 1.0 / (kColumns - 1)));
    double kc[kColumns], klo[kColumns], khi[kColumns];
    for (int c = 0; c < kColumns; ++c) {
        kc[c] = landscapeColumnHz(c, kColumns) * kN / rate;
        klo[c] = kc[c] / halfStep;
        khi[c] = kc[c] * halfStep;
    }
    std::vector<float> window(kN);
    for (int i = 0; i < kN; ++i) window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / kN));

    Landscape l;
    l.columns = kColumns;
    l.rows = frames;
    l.rowSeconds = static_cast<float>(static_cast<double>(hop) / rate);
    l.amps.assign(static_cast<size_t>(kColumns) * frames, 0.0f);
    std::vector<std::complex<float>> buf(kN);
    std::vector<float> power(kN / 2 + 1);
    for (int f = 0; f < frames; ++f) {
        const float* x = mono.data() + static_cast<size_t>(f) * hop;
        for (int i = 0; i < kN; ++i) buf[i] = std::complex<float>(x[i] * window[i], 0.0f);
        fft(buf);
        for (int k = 0; k <= kN / 2; ++k) power[k] = std::norm(buf[k]);
        float* row = l.amps.data() + static_cast<size_t>(f) * kColumns;
        for (int c = 0; c < kColumns; ++c) {
            if (kc[c] >= kN / 2) continue;   // above the recording's Nyquist
            const int a = static_cast<int>(std::ceil(klo[c]));
            const int b = std::min(static_cast<int>(std::floor(khi[c])), kN / 2);
            double p = 0.0;
            if (b >= a) {
                for (int k = a; k <= b; ++k) p += power[k];
                p /= (b - a + 1);
            } else {
                const int k0 = static_cast<int>(kc[c]);
                const double fr = kc[c] - k0;
                p = power[k0] + (power[std::min(k0 + 1, kN / 2)] - power[k0]) * fr;
            }
            row[c] = static_cast<float>(std::sqrt(p));
        }
    }

    // Whitening: each column divided by a power of its own average, so a loud low end does not
    // decide the landscape; then the loudest point is 1 and what lies `rangeDb` below it is 0.
    if (whiten > 0.0f) {
        for (int c = 0; c < kColumns; ++c) {
            double mean = 0.0;
            for (int f = 0; f < frames; ++f) mean += l.amps[static_cast<size_t>(f) * kColumns + c];
            mean /= frames;
            if (mean < 1.0e-12) continue;
            const float div = static_cast<float>(std::pow(mean, static_cast<double>(std::clamp(whiten, 0.0f, 1.0f))));
            for (int f = 0; f < frames; ++f) l.amps[static_cast<size_t>(f) * kColumns + c] /= div;
        }
    }
    // The loop first, then the smoothing around it: two passes of a circular box per column, a
    // triangle of `smooth` seconds in all.
    makeSeamless(l, seamless);
    const int half = static_cast<int>(std::lround(0.5 * smooth / l.rowSeconds));
    if (half >= 1 && l.rows > 2 * half + 1) {
        const int rowsN = l.rows;
        std::vector<double> col(rowsN), acc(rowsN);
        for (int c = 0; c < kColumns; ++c) {
            for (int f = 0; f < rowsN; ++f) col[f] = l.amps[static_cast<size_t>(f) * kColumns + c];
            for (int pass = 0; pass < 2; ++pass) {
                double sum = 0.0;
                for (int k = -half; k <= half; ++k) sum += col[(k + rowsN) % rowsN];
                for (int f = 0; f < rowsN; ++f) {
                    acc[f] = sum / (2 * half + 1);
                    sum += col[(f + half + 1) % rowsN] - col[(f - half + rowsN) % rowsN];
                }
                col.swap(acc);
            }
            for (int f = 0; f < rowsN; ++f) l.amps[static_cast<size_t>(f) * kColumns + c] = static_cast<float>(col[f]);
        }
    }

    // Normalised to the 99.5th percentile rather than the maximum: a few breaking waves would
    // otherwise push everything else down (the first sea came out 28 dB below its peaks).
    std::vector<float> sorted(l.amps);
    const size_t at = std::min(sorted.size() - 1, static_cast<size_t>(0.995 * static_cast<double>(sorted.size())));
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(at), sorted.end());
    const float peak = sorted[at];
    if (peak <= 0.0f) return fail(error, "the recording is silent");
    const float floorAmp = std::pow(10.0f, -rangeDb / 20.0f);
    for (float& a : l.amps) a = std::clamp((a / peak - floorAmp) / (1.0f - floorAmp), 0.0f, 1.0f);
    out = std::move(l);
    return true;
}

void makeSeamless(Landscape& l, float fraction)
{
    const int x = static_cast<int>(std::lround(std::clamp(fraction, 0.0f, 0.5f) * static_cast<float>(l.rows)));
    if (x < 1 || l.rows - x < 2) return;
    const int keep = l.rows - x;
    for (int i = 0; i < x; ++i) {
        // Row i of the new loop leans from the old end (just before the wrap) to the old beginning.
        const float w = (static_cast<float>(i) + 0.5f) / static_cast<float>(x);
        float* head = l.amps.data() + static_cast<size_t>(i) * l.columns;
        const float* tail = l.amps.data() + static_cast<size_t>(keep + i) * l.columns;
        for (int c = 0; c < l.columns; ++c) head[c] = tail[c] * (1.0f - w) + head[c] * w;
    }
    l.rows = keep;
    l.amps.resize(static_cast<size_t>(keep) * l.columns);
}

bool writeLandscapeBmp(const Landscape& l, const fs::path& path, float rangeDb, std::string* error)
{
    if (l.columns < 1 || l.rows < 1) return fail(error, "empty landscape");
    const std::uint32_t width = static_cast<std::uint32_t>(l.rows), height = static_cast<std::uint32_t>(l.columns);
    const std::uint32_t stride = (width * 3 + 3) & ~3u;
    const std::uint32_t pixelBytes = stride * height;
    std::vector<unsigned char> bmp(54 + pixelBytes, 0);
    auto put32 = [&](size_t at, std::uint32_t v) { for (int i = 0; i < 4; ++i) bmp[at + i] = static_cast<unsigned char>(v >> (8 * i)); };
    auto put16 = [&](size_t at, std::uint32_t v) { bmp[at] = static_cast<unsigned char>(v); bmp[at + 1] = static_cast<unsigned char>(v >> 8); };
    bmp[0] = 'B'; bmp[1] = 'M';
    put32(2, static_cast<std::uint32_t>(bmp.size())); put32(10, 54);
    put32(14, 40); put32(18, width); put32(22, height); put16(26, 1); put16(28, 24); put32(34, pixelBytes);
    put32(38, 2835); put32(42, 2835);
    // BMP rows run bottom-up: the first is the lowest column.
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x) {
            const float a = l.amps[static_cast<size_t>(x) * l.columns + y];
            const unsigned char g = static_cast<unsigned char>(std::lround(255.0f * ampToBrightness(a, rangeDb)));
            unsigned char* px = bmp.data() + 54 + static_cast<size_t>(y) * stride + static_cast<size_t>(x) * 3;
            px[0] = px[1] = px[2] = g;
        }
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size()));
    if (!f) return fail(error, "cannot write " + path.string());
    return true;
}

} // namespace additiv
