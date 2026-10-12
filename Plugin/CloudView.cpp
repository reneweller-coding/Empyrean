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
    if (proc_.engine().partialSnapshot(snap_, snapPer_)) {
        proc_.engine().lifeSnapshot(lifeCells_);
        if (lifeCells_.size() == static_cast<size_t>(additiv::kLifeW * additiv::kLifeH)) {
            // The living field as a picture: across the spectrum (left low), along a cloud's members (up).
            if (!life_.isValid()) life_ = juce::Image(juce::Image::RGB, additiv::kLifeW, additiv::kLifeH, false);
            const juce::Colour lo = additivui::colour::bg, hi = additivui::colour::violet;
            const juce::Image::BitmapData px(life_, juce::Image::BitmapData::writeOnly);
            for (int y = 0; y < additiv::kLifeH; ++y)
                for (int x = 0; x < additiv::kLifeW; ++x)
                    px.setPixelColour(x, additiv::kLifeH - 1 - y, lo.interpolatedWith(hi, std::sqrt(lifeCells_[static_cast<size_t>(y) * additiv::kLifeW + x])));
        }
    }
    if (!dotsArea_.isEmpty()) plotDots(dotsArea_.getWidth(), dotsArea_.getHeight());
    repaint();
}

void CloudView::plotDots(int w, int h)
{
    if (w <= 0 || h <= 0) return;
    if (dots_.getWidth() != w || dots_.getHeight() != h) {
        dots_ = juce::Image(juce::Image::ARGB, w, h, true);
        glow_.assign(static_cast<size_t>(w) * h * 3, 0.0f);
    }
    // The trails fade by a fifth each frame (about half a second to vanish), then the points come in:
    // each partial at its frequency and loudness (-100 .. -35 dB of a partial's amplitude), its
    // colour by whose note it is, as bright as it is loud.
    for (float& x : glow_) x *= 0.8f;
    const frame::Skin& skin = additivui::skin();
    const juce::Colour bySource[4] = { skin.accent, additivui::colour::violet, skin.family(frame::Family::Space),
                                       skin.dim.brighter(0.4f) };   // played, conductor, Brain 2, the memory's ghosts (pale grey)
    const int voices = snapPer_ > 0 ? static_cast<int>(snap_.size() / 2) / snapPer_ : 0;
    const float logSpan = std::log2(1000.0f);
    for (int v = 0; v < voices; ++v) {
        const int src = v < additiv::kMaxViewVoices ? juce::jlimit(0, 3, views_[static_cast<size_t>(v)].source) : 3;
        const juce::Colour c = bySource[src];
        const float cr = c.getFloatRed(), cg = c.getFloatGreen(), cb = c.getFloatBlue();
        const float* d = snap_.data() + static_cast<size_t>(v) * snapPer_ * 2;
        for (int i = 0; i < snapPer_; ++i) {
            const float hz = d[2 * i], a = d[2 * i + 1];
            if (hz <= 20.0f || a <= 1.0e-6f) continue;
            const float db = 20.0f * std::log10(a);
            const float u = std::log2(hz / 20.0f) / logSpan, l = (db + 100.0f) / 65.0f;
            if (u < 0.0f || u >= 1.0f || l <= 0.0f) continue;
            const int x = static_cast<int>(u * static_cast<float>(w));
            const int y = juce::jlimit(0, h - 1, static_cast<int>((1.0f - std::min(l, 1.0f)) * static_cast<float>(h - 1)));
            const float k = 0.25f + 0.75f * std::min(l, 1.0f);
            float* p = glow_.data() + (static_cast<size_t>(y) * w + x) * 3;   // two pixels wide
            p[0] += cr * k; p[1] += cg * k; p[2] += cb * k;
            if (x + 1 < w) { p[3] += 0.6f * cr * k; p[4] += 0.6f * cg * k; p[5] += 0.6f * cb * k; }
        }
    }
    const juce::Image::BitmapData px(dots_, juce::Image::BitmapData::writeOnly);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* p = glow_.data() + (static_cast<size_t>(y) * w + x) * 3;
            const float m = std::max({ p[0], p[1], p[2] });
            const float s = m > 1.0f ? 1.0f / m : 1.0f;
            px.setPixelColour(x, y, juce::Colour::fromFloatRGBA(p[0] * s, p[1] * s, p[2] * s, std::min(1.0f, m)));
        }
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

    // The partials themselves: every eighth one a point with its trail (plotDots, from the GPU's snapshot).
    dotsArea_ = spec.toNearestInt();
    if (dots_.isValid()) g.drawImage(dots_, spec, juce::RectanglePlacement::stretchToFit);
    float firstTime = -1.0f;
    for (const additiv::VoiceView& v : views_)
        if (v.hz > 0.0f && v.gain > 0.0f) { firstTime = v.time; break; }

    // The landscape, and where the voices are in its loop (when the Aurora motion reads it).
    const bool reading = additiv::motionUsesLandscape(static_cast<additiv::MotionKind>(juce::roundToInt(proc_.value(static_cast<int>(ParamId::Motion)))));
    const bool living = !lifeCells_.empty() && life_.isValid();
    if (living) {   // the living landscape takes the panel while it lives
        g.drawImage(life_, landArea, juce::RectanglePlacement::stretchToFit);
    } else if (land_.isValid()) {
        g.setOpacity(reading ? 1.0f : 0.35f);
        g.drawImage(land_, landArea, juce::RectanglePlacement::stretchToFit);
        g.setOpacity(1.0f);
    }
    g.setColour(skin.edge);
    g.drawRect(landArea, 1.0f);
    g.setColour(skin.dim);
    const int lifeMode = juce::roundToInt(proc_.value(static_cast<int>(ParamId::LifeMode)));
    if (living)
        g.drawText(juce::String("Living landscape: ") + additiv::kLifeModeNames[juce::jlimit(0, additiv::kNumLifeModes - 1, lifeMode)],
                   landArea.withHeight(14.0f).translated(0.0f, -1.0f), juce::Justification::centredLeft);
    else g.drawText(juce::String(additiv::kLandscapeNames[std::max(landChoice_, 0)]) + (reading ? "" : "  (Motion: Aurora reads it)"),
               landArea.withHeight(14.0f).translated(0.0f, -1.0f), juce::Justification::centredLeft);
    if (reading && !living && firstTime >= 0.0f) {
        const float rate = proc_.value(static_cast<int>(ParamId::MotionRate));
        float u = firstTime * rate / std::max(landLoop_, 0.1f);
        u -= std::floor(u);
        g.setColour(skin.onset);
        g.drawVerticalLine(juce::roundToInt(landArea.getX() + u * landArea.getWidth()), landArea.getY(), landArea.getBottom());
    }
}
