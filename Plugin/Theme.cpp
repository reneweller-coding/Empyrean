/**
 * @file Theme.cpp
 * @brief The instrument's skin (Theme.h).
 * @author Rene Weller
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
        g.setColour(colour::Accent.interpolatedWith(colour::Violet, 0.5f * (1.0f - std::cos(a))).withAlpha(lit));
        g.fillEllipse(juce::Rectangle<float>(size, size).withCentre(p));
    }
    g.setColour(colour::Accent.withAlpha(0.8f));
    g.fillEllipse(juce::Rectangle<float>(radius * 0.22f, radius * 0.22f).withCentre(c));
}

const frame::Skin& skin()
{
    static const frame::Skin s = [] {
        frame::Skin k;
        k.m_name = "Empyrean";
        k.m_bg = colour::Bg; k.m_panel = colour::Panel; k.m_group = colour::GroupColour; k.m_raised = colour::Raised; k.m_edge = colour::Edge;
        k.m_ink = colour::Ink; k.m_dim = colour::Dim; k.m_faint = colour::Faint; k.m_accent = colour::Accent; k.m_onset = colour::Onset;
        k.m_good = colour::Good; k.m_bad = colour::Bad;
        k.m_families[static_cast<int>(frame::FamilyE::SOURCE)] = colour::Accent;
        k.m_families[static_cast<int>(frame::FamilyE::FILTER)] = juce::Colour(0xffe0b36a);
        k.m_families[static_cast<int>(frame::FamilyE::ENVELOPE)] = colour::Violet;
        k.m_families[static_cast<int>(frame::FamilyE::MOTION)] = juce::Colour(0xff6fb4ff);
        k.m_families[static_cast<int>(frame::FamilyE::SPACE)] = juce::Colour(0xff9fd0e6);
        for (int d = 0; d < 3; ++d) k.m_decks[d] = colour::Accent;
        k.m_radius = 6.0f;            // soft, as the light is
        k.m_tracking = 0.22f;
        k.m_typeface = "Segoe UI Light";
        k.m_titleBold = false;
        k.m_glow = true;
        k.m_logo = [](juce::Graphics& g, juce::Rectangle<float> r) { drawLogo(g, r); };
        return k;
    }();
    return s;
}

} // namespace additivui
