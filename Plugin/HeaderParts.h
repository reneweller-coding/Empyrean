/**
 * @file HeaderParts.h
 * @brief Empyrean's own parts of the header beside the family's (Frame.h): glyph buttons drawn as the
 *        frame's IconButton draws its glyphs -- the night (a crescent), the search (a lens), the canvas (a
 *        frame's corners), the comparison with the preset (A | B) -- and the output's meter: two thin bars
 *        from -60 to 0 dBFS with the peak held for a moment, and a mark that stays red once the output has
 *        clipped, until it is clicked.
 */
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

/** @brief A glyph button of the header (see the file comment). */
class ViewButton final : public juce::Button {
public:
    /** @brief The glyphs it can draw. */
    enum class Glyph {
        Night,     ///< a crescent: the night mode
        Search,    ///< a lens: the command palette
        Canvas,    ///< a frame's corners: the picture alone
        Compare    ///< A | B: hold to hear the preset as it was
    };
    /** @brief A button with @p glyph and @p tooltip; it never takes the keyboard focus. @param glyph its glyph @param tooltip its tooltip */
    ViewButton(Glyph glyph, const juce::String& tooltip);
    /** @brief The frame's button background, the glyph over it. @param g where @param highlighted under the mouse @param down pressed */
    void paintButton(juce::Graphics& g, bool highlighted, bool down) override;

private:
    Glyph glyph_;   ///< what it draws
};

/** @brief The output's meter (see the file comment). */
class OutputMeter final : public juce::Component, public juce::SettableTooltipClient {
public:
    OutputMeter();
    /** @brief The peaks since the last look, per side (0 .. 1 and over). @param left left @param right right */
    void push(float left, float right);
    /** @brief The two bars and the clip mark. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief A click puts the clip mark out. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;

private:
    float level_[2] = { 0.0f, 0.0f };   ///< what each bar shows, falling slowly (linear)
    float hold_[2] = { 0.0f, 0.0f };    ///< the held peak
    int holdTicks_[2] = { 0, 0 };       ///< how much longer it is held
    bool clipped_ = false;              ///< the output reached full scale
};
