/**
 * @file CloudView.cpp
 * @brief The cloud picture (CloudView.h).
 */
#include "CloudView.h"

#include "Theme.h"

#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Spectrum.h"

#include <cmath>

using additiv::ParamId;

float CloudView::xOf(float hz, juce::Rectangle<float> area)
{
    const float u = std::log2(std::max(hz, 20.0f) / 20.0f) / std::log2(1000.0f);
    return area.getX() + juce::jlimit(0.0f, 1.0f, u) * area.getWidth();
}

void CloudView::refreshLandscape()
{
    const int choice = juce::jlimit(0, additiv::kNumFactoryLandscapes - 1, juce::roundToInt(proc_.value(static_cast<int>(ParamId::Landscape))));
    if (choice == landChoice_) return;
    landChoice_ = choice;
    const additiv::Landscape l = additiv::factoryLandscape(choice);
    landLoop_ = l.rows * l.rowSeconds;
    // Every fourth row and every second column is enough for a picture this size.
    const int w = l.rows / 4, h = l.columns / 2;
    land_ = juce::Image(juce::Image::RGB, w, h, false);
    const juce::Colour lo = additivui::colour::bg, hi = additivui::colour::accent;
    for (int x = 0; x < w; ++x)
        for (int y = 0; y < h; ++y) {
            const float a = l.amps[static_cast<size_t>(x * 4) * l.columns + static_cast<size_t>((h - 1 - y) * 2)];
            land_.setPixelAt(x, y, lo.interpolatedWith(hi, additiv::ampToBrightness(a, 40.0f)));
        }
}

void CloudView::refresh()
{
    for (int i = 0; i < additiv::kMaxViewVoices; ++i) views_[static_cast<size_t>(i)] = proc_.engine().voiceView(i);
    refreshLandscape();
    repaint();
}

void CloudView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto all = getLocalBounds().toFloat();
    g.setColour(skin.group.withAlpha(0.9f));
    g.fillRoundedRectangle(all, skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(all.reduced(0.5f), skin.radius, 1.0f);

    auto inner = all.reduced(10.0f);
    const auto landArea = inner.removeFromRight(inner.getWidth() * 0.28f).withTrimmedLeft(10.0f);
    const auto spec = inner.withTrimmedBottom(14.0f);

    // The axis: 100 Hz, 1 kHz, 10 kHz.
    g.setFont(juce::FontOptions(11.0f));
    for (float f : { 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f }) {
        const float x = xOf(f, spec);
        g.setColour(skin.faint.withAlpha(0.5f));
        g.drawVerticalLine(juce::roundToInt(x), spec.getY(), spec.getBottom());
        g.setColour(skin.dim);
        g.drawText(f >= 1000.0f ? juce::String(f / 1000.0f, 0) + "k" : juce::String(f, 0), juce::Rectangle<float>(x - 20.0f, spec.getBottom(), 40.0f, 14.0f),
                   juce::Justification::centred);
    }

    // The clouds: the voice's own shape, as the GPU draws it, one soft bar per harmonic.
    const float tilt = proc_.value(static_cast<int>(ParamId::Tilt));
    const float oddEven = proc_.value(static_cast<int>(ParamId::OddEven));
    const float bright = proc_.value(static_cast<int>(ParamId::Brightness));
    const float B = additiv::inharmonicB(proc_.value(static_cast<int>(ParamId::Inharmonic)));
    const float width = proc_.value(static_cast<int>(ParamId::CloudWidth)) * (1.0f - 0.6f * proc_.value(static_cast<int>(ParamId::Purity)));
    const float slope = proc_.value(static_cast<int>(ParamId::WidthSlope));
    const float pxPerCent = spec.getWidth() / (1200.0f * std::log2(1000.0f));
    const juce::Colour bySource[4] = { skin.accent, additivui::colour::violet, skin.family(frame::Family::Space),
                                       skin.family(frame::Family::Envelope).withMultipliedAlpha(0.7f) };   // played, conductor, Brain 2, memory
    float firstTime = -1.0f;
    for (const additiv::VoiceView& v : views_) {
        if (v.hz <= 0.0f || v.gain <= 0.0f) continue;
        if (firstTime < 0.0f) firstTime = v.time;
        const juce::Colour c = bySource[juce::jlimit(0, 3, v.source)];
        const float strength = v.gain * (1.0f - 0.6f * v.distance);
        const int H = std::min(160, static_cast<int>(20000.0f / v.hz));
        for (int h = 1; h <= H; ++h) {
            const float a = additiv::harmonicShape(h, tilt, oddEven, bright);
            if (a < 1.0e-3f) continue;
            const float f = v.hz * static_cast<float>(h) * additiv::stretchOf(h, B);
            const float x = xOf(f, spec);
            const float barH = spec.getHeight() * std::pow(a, 0.5f) * strength * 0.95f;
            const float w = std::max(1.5f, width * std::pow(static_cast<float>(h), slope) * pxPerCent);
            g.setColour(c.withAlpha(juce::jlimit(0.05f, 0.85f, 0.25f + 0.6f * strength)));
            g.fillRect(juce::Rectangle<float>(x - 0.5f * w, spec.getBottom() - barH, w, barH));
        }
    }

    // The landscape, and where the voices are in its loop (when the Aurora motion reads it).
    const bool reading = additiv::motionUsesLandscape(static_cast<additiv::MotionKind>(juce::roundToInt(proc_.value(static_cast<int>(ParamId::Motion)))));
    if (land_.isValid()) {
        g.setOpacity(reading ? 1.0f : 0.35f);
        g.drawImage(land_, landArea, juce::RectanglePlacement::stretchToFit);
        g.setOpacity(1.0f);
    }
    g.setColour(skin.edge);
    g.drawRect(landArea, 1.0f);
    g.setColour(skin.dim);
    g.drawText(juce::String(additiv::kLandscapeNames[std::max(landChoice_, 0)]) + (reading ? "" : "  (Motion: Aurora reads it)"),
               landArea.withHeight(14.0f).translated(0.0f, -1.0f), juce::Justification::centredLeft);
    if (reading && firstTime >= 0.0f) {
        const float rate = proc_.value(static_cast<int>(ParamId::MotionRate));
        float u = firstTime * rate / std::max(landLoop_, 0.1f);
        u -= std::floor(u);
        g.setColour(skin.onset);
        g.drawVerticalLine(juce::roundToInt(landArea.getX() + u * landArea.getWidth()), landArea.getY(), landArea.getBottom());
    }
}
