/**
 * @file Theme.cpp
 * @brief The instrument's skin (Theme.h).
 */
#include "Theme.h"

#include <cmath>

namespace additivui {

void drawLogo(juce::Graphics& g, juce::Rectangle<float> r)
{
    const juce::Point<float> c = r.getCentre();
    const float radius = 0.38f * std::min(r.getWidth(), r.getHeight());
    // Twenty-four partials round a circle, their sizes from a fixed die, the top lit.
    for (int i = 0; i < 24; ++i) {
        const float a = juce::MathConstants<float>::twoPi * static_cast<float>(i) / 24.0f;
        std::uint32_t k = static_cast<std::uint32_t>(i) * 2654435761u;
        k ^= k >> 15;
        const float size = radius * (0.08f + 0.12f * static_cast<float>(k % 1000) / 1000.0f);
        const float lit = 0.35f + 0.65f * 0.5f * (1.0f + std::cos(a));
        const juce::Point<float> p = c + juce::Point<float>(std::sin(a), -std::cos(a)) * radius;
        g.setColour(colour::accent.interpolatedWith(colour::violet, 0.5f * (1.0f - std::cos(a))).withAlpha(lit));
        g.fillEllipse(juce::Rectangle<float>(size, size).withCentre(p));
    }
    g.setColour(colour::accent.withAlpha(0.8f));
    g.fillEllipse(juce::Rectangle<float>(radius * 0.22f, radius * 0.22f).withCentre(c));
}

const frame::Skin& skin()
{
    static const frame::Skin s = [] {
        frame::Skin k;
        k.name = "Additiv Ambient";
        k.bg = colour::bg; k.panel = colour::panel; k.group = colour::group; k.raised = colour::raised; k.edge = colour::edge;
        k.ink = colour::ink; k.dim = colour::dim; k.faint = colour::faint; k.accent = colour::accent; k.onset = colour::onset;
        k.good = colour::good; k.bad = colour::bad;
        k.families[static_cast<int>(frame::Family::Source)] = colour::accent;
        k.families[static_cast<int>(frame::Family::Filter)] = juce::Colour(0xffe0b36a);
        k.families[static_cast<int>(frame::Family::Envelope)] = colour::violet;
        k.families[static_cast<int>(frame::Family::Motion)] = juce::Colour(0xff6fb4ff);
        k.families[static_cast<int>(frame::Family::Space)] = juce::Colour(0xff9fd0e6);
        for (int d = 0; d < 3; ++d) k.decks[d] = colour::accent;
        k.radius = 6.0f;            // soft, as the light is
        k.tracking = 0.22f;
        k.typeface = "Segoe UI Light";
        k.titleBold = false;
        k.glow = true;
        k.logo = [](juce::Graphics& g, juce::Rectangle<float> r) { drawLogo(g, r); };
        return k;
    }();
    return s;
}

} // namespace additivui
