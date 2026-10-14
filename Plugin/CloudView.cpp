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

namespace {

constexpr float kHandle = 9.0f;   ///< how near a handle a click takes it, pixels

/** @brief The axis' span: 20 Hz to 20 kHz, in octaves. */
const float kSpan = std::log2(1000.0f);

} // namespace

CloudView::CloudView(AdditivProcessor& proc) : proc_(proc)
{
    static const char* const kMacroNames[4] = { "Macro 1", "Macro 2", "Macro 3", "Macro 4" };
    for (int i = 0; i < 4; ++i) {
        juce::Slider& s = macros_[static_cast<size_t>(i)];
        s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
        s.setColour(juce::Slider::rotarySliderFillColourId, additivui::skin().family(frame::Family::Envelope));
        s.setTooltip(juce::String(kMacroNames[i]) + ": whatever the matrix lets it move");
        const int id = static_cast<int>(ParamId::Macro1) + i;
        if (auto* p = proc_.parameter(id)) macroLinks_.push_back(std::make_unique<juce::SliderParameterAttachment>(*p, s));
        juce::Label& l = macroNames_[static_cast<size_t>(i)];
        l.setText(kMacroNames[i], juce::dontSendNotification);
        l.setColour(juce::Label::textColourId, additivui::skin().dim);
        l.setFont(juce::FontOptions(12.0f));
        l.setJustificationType(juce::Justification::centredLeft);
        addChildComponent(s);
        addChildComponent(l);
    }
}

CloudView::~CloudView()
{
    if (playing_ >= 0) proc_.tapNote(playing_, 0.0f);
}

float CloudView::xOf(float hz, juce::Rectangle<float> area)
{
    const float u = std::log2(std::max(hz, 20.0f) / 20.0f) / kSpan;
    return area.getX() + juce::jlimit(0.0f, 1.0f, u) * area.getWidth();
}

float CloudView::hzAt(float x, juce::Rectangle<float> area)
{
    const float u = juce::jlimit(0.0f, 1.0f, (x - area.getX()) / std::max(area.getWidth(), 1.0f));
    return 20.0f * std::exp2(u * kSpan);
}

int CloudView::keyAt(float x) const
{
    const float hz = hzAt(x, spec_);
    return juce::jlimit(21, 108, juce::roundToInt(69.0f + 12.0f * std::log2(hz / 440.0f)));
}

void CloudView::setCanvas(bool on)
{
    canvas_ = on;
    for (juce::Slider& s : macros_) s.setVisible(on);
    for (juce::Label& l : macroNames_) l.setVisible(on);
    if (!on && playing_ >= 0) {
        proc_.tapNote(playing_, 0.0f);
        playing_ = -1;
    }
    resized();
    repaint();
}

void CloudView::resized()
{
    // The macros under the keys, in the canvas: four knobs side by side on the left, each with its name beside it.
    const int size = 54, y = getHeight() - size - 8, w = 190;
    for (int i = 0; i < 4; ++i) {
        macros_[static_cast<size_t>(i)].setBounds(20 + i * w, y, size, size);
        macroNames_[static_cast<size_t>(i)].setBounds(20 + i * w + size + 4, y + size / 2 - 9, w - size - 10, 18);
    }
}

void CloudView::paintKeys(juce::Graphics& g, juce::Rectangle<float> keys, juce::Rectangle<float> spec)
{
    const frame::Skin& skin = additivui::skin();
    const juce::Colour bySource[3] = { skin.accent, additivui::colour::violet, skin.family(frame::Family::Space) };   // played, conductor, Brain 2
    // Each sounding key once: its loudest voice, near and loud brightest.
    float level[128] = {};
    int source[128] = {};
    for (const additiv::VoiceView& v : views_) {
        if (v.note < 0 || v.note > 127 || v.hz <= 0.0f || v.gain <= 0.0f || v.source > 2) continue;
        const float l = juce::jlimit(0.0f, 1.0f, std::sqrt(v.gain) * (1.0f - 0.6f * juce::jlimit(0.0f, 1.0f, v.distance)));
        if (l > level[v.note]) { level[v.note] = std::max(l, 0.05f); source[v.note] = juce::jlimit(0, 2, v.source); }
    }
    constexpr int kFirst = 21, kLast = 108;   // A0 .. C8
    auto edge = [&](float n) { return xOf(440.0f * std::exp2((n - 69.0f) / 12.0f), spec); };
    // As a keyboard: the white keys one strip, a line where two white keys meet (B-C, E-F), the
    // black keys short and dark over it; a sounding key lit in its part.
    const float left = edge(static_cast<float>(kFirst) - 0.5f), right = edge(static_cast<float>(kLast) + 0.5f);
    const float blackH = std::round(keys.getHeight() * 0.62f);
    g.setColour(skin.dim.interpolatedWith(skin.faint, 0.45f));
    g.fillRect(left, keys.getY(), right - left, keys.getHeight());
    for (int n = kFirst; n <= kLast; ++n) {
        const float x0 = edge(static_cast<float>(n) - 0.5f), x1 = edge(static_cast<float>(n) + 0.5f);
        const bool black = juce::MidiMessage::isMidiNoteBlack(n);
        if (black) {
            g.setColour(level[n] > 0.0f ? skin.bg.interpolatedWith(bySource[source[n]], 0.55f + 0.45f * level[n]) : skin.bg);
            g.fillRect(x0, keys.getY(), x1 - x0, blackH);
            continue;
        }
        if (level[n] > 0.0f) {
            // A white key reaches under its black neighbours' halves; lit, it is its own semitone.
            g.setColour(skin.faint.interpolatedWith(bySource[source[n]], 0.55f + 0.45f * level[n]));
            g.fillRect(x0, keys.getY(), x1 - x0, keys.getHeight());
        }
        if (n > kFirst && !juce::MidiMessage::isMidiNoteBlack(n - 1)) {   // B|C, E|F
            g.setColour(skin.bg.withAlpha(0.9f));
            g.fillRect(x0 - 0.5f, keys.getY(), 1.0f, keys.getHeight());
        }
    }
    // The conductor's root, while it conducts.
    const int root = proc_.engine().brainRoot();
    if (proc_.value(static_cast<int>(ParamId::BrainOn)) >= 0.5f && root >= kFirst && root <= kLast) {
        const float x = 0.5f * (edge(static_cast<float>(root) - 0.5f) + edge(static_cast<float>(root) + 0.5f));
        juce::Path tri;
        tri.addTriangle(x - 3.5f, keys.getY() - 1.0f, x + 3.5f, keys.getY() - 1.0f, x, keys.getY() + 4.0f);
        g.setColour(additivui::colour::root);
        g.fillPath(tri);
    }
    // The names, low to high, each in its colour.
    const juce::Font font(juce::FontOptions(11.0f));
    g.setFont(font);
    float x = spec.getX() + 4.0f;
    for (int n = 0; n < 128; ++n) {
        if (level[n] <= 0.0f) continue;
        const juce::String name = juce::MidiMessage::getMidiNoteName(n, true, true, 4);
        const float w = juce::GlyphArrangement::getStringWidth(font, name) + 8.0f;
        if (x + w > spec.getRight()) break;
        g.setColour(bySource[source[n]].withAlpha(0.45f + 0.55f * level[n]));
        g.drawText(name, juce::Rectangle<float>(x, spec.getY() + 2.0f, w, 14.0f), juce::Justification::centredLeft);
        x += w;
    }
}

void CloudView::refreshLandscape()
{
    int choice = juce::jlimit(0, additiv::kNumLandscapeChoices - 1, juce::roundToInt(proc_.value(static_cast<int>(ParamId::Landscape))));
    const bool own = choice == additiv::kUserLandscape && !proc_.userLandscape().amps.empty();
    if (choice == additiv::kUserLandscape && !own) choice = 0;   // none loaded: the engine plays Aurora
    if (choice == landChoice_ && (!own || landVersion_ == proc_.userLandscapeVersion())) return;
    landChoice_ = choice;
    landVersion_ = proc_.userLandscapeVersion();
    const additiv::Landscape l = own ? proc_.userLandscape() : additiv::factoryLandscape(choice);
    landLoop_ = l.rows * l.rowSeconds;
    // About 256 x 128 points are enough for a picture this size.
    const int sx = std::max(1, l.rows / 256), sy = std::max(1, l.columns / 128);
    const int w = std::max(1, l.rows / sx), h = std::max(1, l.columns / sy);
    land_ = juce::Image(juce::Image::RGB, w, h, false);
    const juce::Colour lo = additivui::colour::bg, hi = additivui::colour::accent;
    for (int x = 0; x < w; ++x)
        for (int y = 0; y < h; ++y) {
            const float a = l.amps[static_cast<size_t>(x * sx) * l.columns + static_cast<size_t>((h - 1 - y) * sy)];
            land_.setPixelAt(x, y, lo.interpolatedWith(hi, additiv::ampToBrightness(a, 40.0f)));
        }
}

std::vector<ParamId> CloudView::paramsOf(Handle h)
{
    switch (h) {
    case Handle::Filter: return { ParamId::Cutoff, ParamId::Resonance };
    case Handle::AirCentre: return { ParamId::AirCenter };
    case Handle::AirEdge: return { ParamId::AirWidth };
    case Handle::Brightness: return { ParamId::Brightness };
    case Handle::RegisterLow: return { ParamId::BrainLow };
    case Handle::RegisterHigh: return { ParamId::BrainHigh };
    case Handle::HallLowest: return { ParamId::ModalLow };
    default: return {};
    }
}

CloudView::Handle CloudView::handleAt(juce::Point<float> p) const
{
    if (canvas_) return Handle::None;
    Handle best = Handle::None;
    float nearest = kHandle;
    for (int i = 1; i < static_cast<int>(Handle::Count); ++i) {
        if (!shown_[static_cast<size_t>(i)]) continue;
        const float d = p.getDistanceFrom(handles_[static_cast<size_t>(i)]);
        if (d < nearest) { nearest = d; best = static_cast<Handle>(i); }
    }
    return best;
}

float CloudView::referenceHz() const
{
    float lowest = 0.0f;
    for (const additiv::VoiceView& v : views_)
        if (v.hz > 0.0f && v.gain > 0.0f && v.source <= 2 && (lowest <= 0.0f || v.hz < lowest)) lowest = v.hz;
    return lowest > 0.0f ? lowest : 130.81f;
}

void CloudView::mouseDown(const juce::MouseEvent& e)
{
    held_ = handleAt(e.position);
    if (held_ != Handle::None) {
        for (ParamId id : paramsOf(held_))
            if (auto* p = proc_.parameter(static_cast<int>(id))) p->beginChangeGesture();
        heldResonance_ = value(ParamId::Resonance);
        heldY_ = e.position.y;
        mouseDrag(e);
        return;
    }
    if (canvas_ && spec_.contains(e.position) && e.mods.isLeftButtonDown()) {   // the canvas plays the key under the mouse
        playing_ = keyAt(e.position.x);
        proc_.tapNote(playing_, juce::jlimit(0.2f, 1.0f, 1.0f - (e.position.y - spec_.getY()) / std::max(spec_.getHeight(), 1.0f)));
    }
}

void CloudView::mouseDrag(const juce::MouseEvent& e)
{
    const float hz = hzAt(e.position.x, spec_);
    switch (held_) {
    case Handle::Filter:
        set(ParamId::Cutoff, hz);
        set(ParamId::Resonance, heldResonance_ + (heldY_ - e.position.y) / 160.0f);
        break;
    case Handle::AirCentre: set(ParamId::AirCenter, hz / referenceHz()); break;
    case Handle::AirEdge: {
        const float centre = referenceHz() * value(ParamId::AirCenter);
        set(ParamId::AirWidth, 1200.0f * std::log2(std::max(hz, centre * 1.01f) / centre));
        break;
    }
    case Handle::Brightness: set(ParamId::Brightness, std::sqrt(std::max(0.0f, hz / referenceHz() - 1.0f) / 200.0f)); break;
    case Handle::RegisterLow: set(ParamId::BrainLow, static_cast<float>(std::min(keyAt(e.position.x), juce::roundToInt(value(ParamId::BrainHigh)) - 1))); break;
    case Handle::RegisterHigh: set(ParamId::BrainHigh, static_cast<float>(std::max(keyAt(e.position.x), juce::roundToInt(value(ParamId::BrainLow)) + 1))); break;
    case Handle::HallLowest: set(ParamId::ModalLow, hz); break;
    default:
        if (playing_ >= 0) {   // the canvas: on to the next key, legato
            const int key = keyAt(e.position.x);
            if (key != playing_) {
                proc_.tapNote(playing_, 0.0f);
                playing_ = key;
                proc_.tapNote(playing_, juce::jlimit(0.2f, 1.0f, 1.0f - (e.position.y - spec_.getY()) / std::max(spec_.getHeight(), 1.0f)));
            }
        }
        return;
    }
    repaint();
}

void CloudView::mouseUp(const juce::MouseEvent& e)
{
    if (held_ != Handle::None) {
        for (ParamId id : paramsOf(held_))
            if (auto* p = proc_.parameter(static_cast<int>(id))) p->endChangeGesture();
        held_ = Handle::None;
        repaint();
        return;
    }
    if (playing_ >= 0) {
        proc_.tapNote(playing_, 0.0f);
        playing_ = -1;
        return;
    }
    if (canvas_ || !landArea_.contains(e.position) || e.getNumberOfClicks() > 1) return;
    chooser_ = std::make_unique<juce::FileChooser>("Your own landscape: a picture (time left to right, low at the bottom) or a recording",
                                                   proc_.userLandscapeFile().existsAsFile() ? proc_.userLandscapeFile()
                                                                                            : juce::File::getSpecialLocation(juce::File::userPicturesDirectory),
                                                   "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.gif;*.wav;*.flac");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this](const juce::FileChooser& fc) {
        const juce::File f = fc.getResult();
        if (f == juce::File()) return;
        juce::String why;
        if (!proc_.loadLandscape(f, true, &why))
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Landscape", "This file could not be read: " + why);
    });
}

void CloudView::mouseMove(const juce::MouseEvent& e)
{
    const Handle h = handleAt(e.position);
    if (h != hover_) {
        hover_ = h;
        setMouseCursor(h != Handle::None ? juce::MouseCursor::DraggingHandCursor : juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void CloudView::mouseDoubleClick(const juce::MouseEvent& e)
{
    if (spec_.contains(e.position) && handleAt(e.position) == Handle::None && onDoubleClick) onDoubleClick();
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
        if (page_ == "Motion") proc_.engine().rippleSnapshot(ripple_);
    }
    if (page_ == "Hall") proc_.engine().hallMemory(hall_, hallAge_);
    if (canvas_)   // the macros' names follow what they move
        for (int i = 0; i < 4; ++i) {
            const juce::String name = proc_.macroName(i);
            if (macroNames_[static_cast<size_t>(i)].getText() != name) macroNames_[static_cast<size_t>(i)].setText(name, juce::dontSendNotification);
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
    // colour by whose note it is, as bright as it is loud. The canvas lets them linger longer.
    const float fade = canvas_ ? 0.9f : 0.8f;
    for (float& x : glow_) x *= fade;
    const frame::Skin& skin = additivui::skin();
    const juce::Colour bySource[4] = { skin.accent, additivui::colour::violet, skin.family(frame::Family::Space),
                                       skin.dim.brighter(0.4f) };   // played, conductor, Brain 2, the memory's ghosts (pale grey)
    const int voices = snapPer_ > 0 ? static_cast<int>(snap_.size() / 2) / snapPer_ : 0;
    for (int v = 0; v < voices; ++v) {
        const int src = v < additiv::kMaxViewVoices ? juce::jlimit(0, 3, views_[static_cast<size_t>(v)].source) : 3;
        const juce::Colour c = bySource[src];
        const float cr = c.getFloatRed(), cg = c.getFloatGreen(), cb = c.getFloatBlue();
        const float* d = snap_.data() + static_cast<size_t>(v) * snapPer_ * 2;
        for (int i = 0; i < snapPer_; ++i) {
            const float hz = d[2 * i], a = d[2 * i + 1];
            if (hz <= 20.0f || a <= 1.0e-6f) continue;
            const float db = 20.0f * std::log10(a);
            const float u = std::log2(hz / 20.0f) / kSpan, l = (db + 100.0f) / 65.0f;
            if (u < 0.0f || u >= 1.0f || l <= 0.0f) continue;
            const int x = static_cast<int>(u * static_cast<float>(w));
            const int y = juce::jlimit(0, h - 1, static_cast<int>((1.0f - std::min(l, 1.0f)) * static_cast<float>(h - 1)));
            const float k = 0.25f + 0.75f * std::min(l, 1.0f);
            float* p = glow_.data() + (static_cast<size_t>(y) * w + x) * 3;   // two pixels wide
            p[0] += cr * k; p[1] += cg * k; p[2] += cb * k;
            if (x + 1 < w) { p[3] += 0.6f * cr * k; p[4] += 0.6f * cg * k; p[5] += 0.6f * cb * k; }
            if (canvas_) {   // the canvas is larger: a point three pixels across
                for (const int dy : { -1, 1 })
                    if (y + dy >= 0 && y + dy < h) {
                        float* q = glow_.data() + (static_cast<size_t>(y + dy) * w + x) * 3;
                        q[0] += 0.5f * cr * k; q[1] += 0.5f * cg * k; q[2] += 0.5f * cb * k;
                    }
                if (x > 0) { p[-3] += 0.5f * cr * k; p[-2] += 0.5f * cg * k; p[-1] += 0.5f * cb * k; }
            }
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

void CloudView::paintHandle(juce::Graphics& g, Handle h, juce::Point<float> at, juce::Colour c)
{
    const size_t i = static_cast<size_t>(h);
    handles_[i] = at;
    shown_[i] = true;
    const bool lit = held_ == h || hover_ == h;
    g.setColour(c.withAlpha(lit ? 0.35f : 0.18f));
    g.fillEllipse(at.x - 7.0f, at.y - 7.0f, 14.0f, 14.0f);
    g.setColour(lit ? c.brighter(0.5f) : c);
    g.drawEllipse(at.x - 5.0f, at.y - 5.0f, 10.0f, 10.0f, 1.6f);
}

void CloudView::paintOverlay(juce::Graphics& g, juce::Rectangle<float> spec, juce::Rectangle<float> keys)
{
    shown_.fill(false);
    if (canvas_) return;
    const frame::Skin& skin = additivui::skin();
    g.setFont(juce::FontOptions(11.0f));
    auto note = [&](juce::Point<float> at, const juce::String& text, juce::Colour c) {   // a handle's value while it is held or hovered
        g.setColour(c);
        g.drawText(text, juce::Rectangle<float>(at.x + 10.0f, at.y - 18.0f, 220.0f, 14.0f), juce::Justification::centredLeft);
    };
    if (page_ == "Filter") {
        const int model = juce::roundToInt(value(ParamId::FilterModel));
        const juce::Colour c = skin.family(frame::Family::Filter);
        if (model > 0) {
            // The filter's response over the axis: +18 dB at the top, -48 at the bottom.
            const float cutoff = value(ParamId::Cutoff), res = value(ParamId::Resonance);
            const float q = additiv::filterQ(model, res), vowel = additiv::vowelOf(cutoff);
            auto yOf = [&](float gain) {
                const float db = juce::jlimit(-48.0f, 18.0f, 20.0f * std::log10(std::max(gain, 1.0e-6f)));
                return spec.getY() + (18.0f - db) / 66.0f * spec.getHeight();
            };
            juce::Path curve, fill;
            for (float x = spec.getX(); x <= spec.getRight(); x += 2.0f) {
                const float y = yOf(additiv::filterResponse(model, hzAt(x, spec), cutoff, q, vowel));
                if (x == spec.getX()) { curve.startNewSubPath(x, y); fill.startNewSubPath(x, spec.getBottom()); }
                else curve.lineTo(x, y);
                fill.lineTo(x, y);
            }
            fill.lineTo(spec.getRight(), spec.getBottom());
            fill.closeSubPath();
            g.setColour(c.withAlpha(0.07f));
            g.fillPath(fill);
            g.setColour(c.withAlpha(0.85f));
            g.strokePath(curve, juce::PathStrokeType(1.5f));
            g.setColour(c.withAlpha(0.35f));
            g.drawHorizontalLine(juce::roundToInt(yOf(1.0f)), spec.getX(), spec.getRight());
            const juce::Point<float> at(xOf(cutoff, spec), yOf(additiv::filterResponse(model, cutoff, cutoff, q, vowel)));
            paintHandle(g, Handle::Filter, at, c);
            if (held_ == Handle::Filter || hover_ == Handle::Filter)
                note(at, "Cutoff " + juce::String(cutoff >= 1000.0f ? juce::String(cutoff / 1000.0f, 2) + " kHz" : juce::String(cutoff, 0) + " Hz")
                             + "   Resonance " + juce::String(res, 2) + "  (drag: across, up)", c);
        } else {
            g.setColour(c.withAlpha(0.6f));
            g.drawText("Filter off: choose a model to draw and take its curve", spec.withTrimmedTop(spec.getHeight() - 16.0f), juce::Justification::centredRight);
        }
        const float depth = value(ParamId::CombDepth);
        if (depth > 0.0f) {   // the comb's notches where they stand now: log2(f) density + phase a whole number
            const float density = value(ParamId::CombDensity), phase = proc_.engine().combPhase();
            g.setColour(c.withAlpha(0.15f + 0.35f * depth));
            for (int k = static_cast<int>(std::floor(std::log2(20.0f) * density)); ; ++k) {
                const float hz = std::exp2((static_cast<float>(k) - phase) / density);
                if (hz > 20000.0f) break;
                if (hz >= 20.0f) g.drawVerticalLine(juce::roundToInt(xOf(hz, spec)), spec.getY(), spec.getBottom());
            }
        }
    } else if (page_ == "Cloud") {
        const float f0 = referenceHz();
        const juce::Colour c = skin.family(frame::Family::Source);
        // The air's band: its partials gather about Centre times the fundamental, Width wide.
        const float air = value(ParamId::Air), centre = f0 * value(ParamId::AirCenter), width = value(ParamId::AirWidth);
        const float lo = centre * std::exp2(-width / 1200.0f), hi = centre * std::exp2(width / 1200.0f);
        const float alpha = air > 0.0f ? 1.0f : 0.45f;
        g.setColour(c.withAlpha(0.08f * alpha));
        g.fillRect(juce::Rectangle<float>::leftTopRightBottom(xOf(lo, spec), spec.getY(), xOf(hi, spec), spec.getBottom()));
        g.setColour(c.withAlpha(0.5f * alpha));
        g.drawVerticalLine(juce::roundToInt(xOf(centre, spec)), spec.getY(), spec.getBottom());
        const juce::Point<float> at(xOf(centre, spec), spec.getY() + spec.getHeight() * 0.22f);
        paintHandle(g, Handle::AirCentre, at, c.withAlpha(alpha));
        paintHandle(g, Handle::AirEdge, { xOf(hi, spec), at.y }, c.withAlpha(alpha));
        g.setColour(c.withAlpha(0.7f * alpha));
        g.drawText(air > 0.0f ? "Air" : "Air (silent at Air 0)", juce::Rectangle<float>(at.x + 9.0f, at.y + 6.0f, 140.0f, 14.0f), juce::Justification::centredLeft);
        if (hover_ == Handle::AirCentre || held_ == Handle::AirCentre)
            note(at, "Centre " + juce::String(value(ParamId::AirCenter), 2) + " x the fundamental", c);
        if (hover_ == Handle::AirEdge || held_ == Handle::AirEdge)
            note({ xOf(hi, spec), at.y }, "Width " + juce::String(width, 0) + " ct", c);
        // Where the brightness lets the harmonics fade: above 1 + b^2 200 times the fundamental.
        const float b = value(ParamId::Brightness);
        const float fade = f0 * (1.0f + b * b * 200.0f);
        if (fade < 20000.0f) {
            const float x = xOf(fade, spec);
            g.setColour(c.withAlpha(0.45f));
            for (float y = spec.getY(); y < spec.getBottom(); y += 6.0f) g.drawVerticalLine(juce::roundToInt(x), y, std::min(y + 3.0f, spec.getBottom()));
            const juce::Point<float> bt(x, spec.getY() + spec.getHeight() * 0.5f);
            paintHandle(g, Handle::Brightness, bt, c);
            g.setColour(c.withAlpha(0.7f));
            g.drawText("Brightness", juce::Rectangle<float>(bt.x + 9.0f, bt.y + 6.0f, 90.0f, 14.0f), juce::Justification::centredLeft);
            if (hover_ == Handle::Brightness || held_ == Handle::Brightness) note(bt, "Brightness " + juce::String(b, 3), c);
        }
    } else if (page_ == "Conductor") {
        // The register as a bracket over the keys, its ends to drag.
        const juce::Colour c = additivui::colour::violet;
        const int low = juce::roundToInt(value(ParamId::BrainLow)), high = juce::roundToInt(value(ParamId::BrainHigh));
        auto edge = [&](float n) { return xOf(440.0f * std::exp2((n - 69.0f) / 12.0f), spec); };
        const float x0 = edge(static_cast<float>(low) - 0.5f), x1 = edge(static_cast<float>(high) + 0.5f), y = keys.getY() - 5.0f;
        g.setColour(c.withAlpha(0.05f));
        g.fillRect(juce::Rectangle<float>::leftTopRightBottom(x0, spec.getY(), x1, spec.getBottom()));
        g.setColour(c.withAlpha(0.8f));
        g.drawLine(x0, y, x1, y, 1.5f);
        g.drawLine(x0, y, x0, keys.getBottom(), 1.5f);
        g.drawLine(x1, y, x1, keys.getBottom(), 1.5f);
        paintHandle(g, Handle::RegisterLow, { x0, y }, c);
        paintHandle(g, Handle::RegisterHigh, { x1, y }, c);
        g.drawText("the conductor's register", juce::Rectangle<float>(x0 + 10.0f, y - 16.0f, 200.0f, 14.0f), juce::Justification::centredLeft);
        if (hover_ == Handle::RegisterLow || held_ == Handle::RegisterLow)
            note({ x0, y - 12.0f }, "Lowest " + juce::MidiMessage::getMidiNoteName(low, true, true, 4), c);
        if (hover_ == Handle::RegisterHigh || held_ == Handle::RegisterHigh)
            note({ x1, y - 12.0f }, "Highest " + juce::MidiMessage::getMidiNoteName(high, true, true, 4), c);
    } else if (page_ == "Motion") {
        // The ripple's string along the spectrum, as the partials ride it.
        const float depth = value(ParamId::RippleDepth);
        if (depth > 0.0f && ripple_.size() == static_cast<size_t>(additiv::kRippleCells)) {
            const juce::Colour c = skin.family(frame::Family::Space);
            const float mid = spec.getCentreY(), amp = spec.getHeight() * 0.3f;
            juce::Path p;
            for (int i = 0; i < additiv::kRippleCells; ++i) {
                const float x = spec.getX() + static_cast<float>(i) / static_cast<float>(additiv::kRippleCells - 1) * spec.getWidth();
                const float y = mid - amp * depth * std::tanh(ripple_[static_cast<size_t>(i)]);
                if (i == 0) p.startNewSubPath(x, y);
                else p.lineTo(x, y);
            }
            g.setColour(c.withAlpha(0.75f));
            g.strokePath(p, juce::PathStrokeType(1.4f));
            g.drawText("Ripple", juce::Rectangle<float>(spec.getX() + 4.0f, mid - 16.0f, 80.0f, 14.0f), juce::Justification::centredLeft);
        }
    } else if (page_ == "Hall") {
        const juce::Colour c = skin.family(frame::Family::Motion);
        // The hall's memory over the axis (its decay's change at Grow 1, the lower half of the picture).
        if (hall_.size() == static_cast<size_t>(additiv::kHallBins) && hallAge_ > 0.0) {
            const float base = spec.getBottom() - spec.getHeight() * 0.125f, span = spec.getHeight() * 0.25f;
            juce::Path p;
            for (int b = 0; b < additiv::kHallBins; b += 2) {
                const float x = spec.getX() + (static_cast<float>(b) + 0.5f) / static_cast<float>(additiv::kHallBins) * spec.getWidth();
                const float oct = juce::jlimit(-0.5f, 1.5f, std::log2(1.0f + std::max(hall_[static_cast<size_t>(b)], 0.0f)) - 1.0f);
                const float y = base - oct * span;
                if (b == 0) p.startNewSubPath(x, y);
                else p.lineTo(x, y);
            }
            g.setColour(c.withAlpha(0.7f));
            g.strokePath(p, juce::PathStrokeType(1.4f));
            g.drawText("the hall's memory", juce::Rectangle<float>(spec.getX() + 4.0f, base - 1.5f * span - 14.0f, 160.0f, 14.0f), juce::Justification::centredLeft);
        }
        // Its lowest mode, to drag.
        const float lowest = value(ParamId::ModalLow), x = xOf(lowest, spec);
        g.setColour(c.withAlpha(0.15f));
        g.fillRect(juce::Rectangle<float>::leftTopRightBottom(spec.getX(), spec.getY(), x, spec.getBottom()));
        g.setColour(c.withAlpha(0.7f));
        g.drawVerticalLine(juce::roundToInt(x), spec.getY(), spec.getBottom());
        const juce::Point<float> at(x, spec.getY() + spec.getHeight() * 0.3f);
        paintHandle(g, Handle::HallLowest, at, c);
        g.drawText("Lowest", juce::Rectangle<float>(at.x + 9.0f, at.y + 6.0f, 80.0f, 14.0f), juce::Justification::centredLeft);
        if (hover_ == Handle::HallLowest || held_ == Handle::HallLowest) note(at, "the hall's lowest mode " + juce::String(lowest, 0) + " Hz", c);
    }
}

void CloudView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto all = getLocalBounds().toFloat();
    g.setColour(canvas_ ? skin.bg : skin.group.withAlpha(0.9f));
    g.fillRoundedRectangle(all, skin.radius);
    if (!canvas_) {
        g.setColour(skin.edge);
        g.drawRoundedRectangle(all.reduced(0.5f), skin.radius, 1.0f);
    }

    auto inner = all.reduced(canvas_ ? 20.0f : 10.0f);
    if (canvas_) inner.removeFromBottom(62.0f);   // the macros
    const auto landArea = canvas_ ? inner : inner.removeFromRight(inner.getWidth() * 0.28f).withTrimmedLeft(10.0f);
    landArea_ = canvas_ ? juce::Rectangle<float>() : landArea;
    const float keyH = canvas_ ? 14.0f : 10.0f;   // the keys' strip under the spectrum
    const auto spec = inner.withTrimmedBottom(14.0f + keyH + 3.0f);
    spec_ = spec;
    const juce::Rectangle<float> keys(spec.getX(), spec.getBottom() + 2.0f, spec.getWidth(), keyH);

    const bool reading = additiv::motionUsesLandscape(static_cast<additiv::MotionKind>(juce::roundToInt(proc_.engine().soundValue(ParamId::Motion))));
    const bool living = !lifeCells_.empty() && life_.isValid();
    if (canvas_) {   // the landscape faintly behind the partials, the whole picture
        if (living) {
            g.setOpacity(0.45f);
            g.drawImage(life_, spec, juce::RectanglePlacement::stretchToFit);
        } else if (land_.isValid()) {
            g.setOpacity(reading ? 0.3f : 0.12f);
            g.drawImage(land_, spec, juce::RectanglePlacement::stretchToFit);
        }
        g.setOpacity(1.0f);
    }

    // The axis: 100 Hz, 1 kHz, 10 kHz.
    g.setFont(juce::FontOptions(11.0f));
    for (float f : { 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f }) {
        const float x = xOf(f, spec);
        g.setColour(skin.faint.withAlpha(canvas_ ? 0.18f : 0.5f));
        g.drawVerticalLine(juce::roundToInt(x), spec.getY(), spec.getBottom());
        g.setColour(canvas_ ? skin.dim.withAlpha(0.5f) : skin.dim);
        g.drawText(f >= 1000.0f ? juce::String(f / 1000.0f, 0) + "k" : juce::String(f, 0), juce::Rectangle<float>(x - 20.0f, keys.getBottom(), 40.0f, 14.0f),
                   juce::Justification::centred);
    }
    paintKeys(g, keys, spec);

    // The partials themselves: every eighth one a point with its trail (plotDots, from the GPU's snapshot).
    dotsArea_ = spec.toNearestInt();
    if (dots_.isValid()) g.drawImage(dots_, spec, juce::RectanglePlacement::stretchToFit);
    paintOverlay(g, spec, keys);
    if (canvas_) {
        g.setColour(skin.dim.withAlpha(0.6f));
        g.drawText("click a key in the spectrum to play it (higher: louder)    double-click or Esc: back",
                   juce::Rectangle<float>(all.getRight() - 520.0f, all.getBottom() - 34.0f, 500.0f, 16.0f), juce::Justification::centredRight);
        return;
    }
    float firstTime = -1.0f;
    for (const additiv::VoiceView& v : views_)
        if (v.hz > 0.0f && v.gain > 0.0f) { firstTime = v.time; break; }

    // The landscape, and where the voices are in its loop (when the Aurora motion reads it).
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
    const int lifeMode = juce::roundToInt(proc_.engine().soundValue(ParamId::LifeMode));
    if (living)
        g.drawText(juce::String("Living landscape: ") + additiv::kLifeModeNames[juce::jlimit(0, additiv::kNumLifeModes - 1, lifeMode)],
                   landArea.withHeight(14.0f).translated(0.0f, -1.0f), juce::Justification::centredLeft);
    else {
        const bool own = landChoice_ == additiv::kUserLandscape;
        const juce::String name = own ? juce::String("Your own: ") + proc_.userLandscapeFile().getFileNameWithoutExtension()
                                      : juce::String(additiv::kLandscapeChoiceNames[std::max(landChoice_, 0)]);
        g.drawText(name + (reading ? "" : "  (Motion: Aurora reads it)"), landArea.withHeight(14.0f).translated(0.0f, -1.0f),
                   juce::Justification::centredLeft);
    }
    if (isMouseOver(true)) {   // what a click does
        g.setColour(skin.dim);
        g.drawText("click: your own picture or recording", landArea.withTrimmedTop(landArea.getHeight() - 14.0f), juce::Justification::centredRight);
    }
    if (reading && !living && firstTime >= 0.0f) {
        const float rate = proc_.engine().soundValue(ParamId::MotionRate);
        float u = firstTime * rate / std::max(landLoop_, 0.1f);
        u -= std::floor(u);
        g.setColour(skin.onset);
        g.drawVerticalLine(juce::roundToInt(landArea.getX() + u * landArea.getWidth()), landArea.getY(), landArea.getBottom());
    }
}
