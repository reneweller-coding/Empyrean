/**
 * @file HeaderParts.h
 * @brief Empyrean's own parts of the header beside the family's (Frame.h): glyph buttons drawn as the
 *        frame's IconButton draws its glyphs -- the night (a crescent), the search (a lens), the canvas (a
 *        frame's corners), the comparison with the preset (A | B) -- and the output's meter: two thin bars
 *        from -60 to 0 dBFS with the peak held for a moment, and a mark that stays red once the output has
 *        clipped, until it is clicked.
 * @author Rene Weller
 */
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

/** @brief A glyph button of the header (see the file comment). */
class ViewButton final : public juce::Button
{
public:
    /** @brief The glyphs it can draw. */
    enum class GlyphE
    {
        NIGHT,    ///< a crescent: the night mode
        SEARCH,   ///< a lens: the command palette
        CANVAS,   ///< a frame's corners: the picture alone
        COMPARE   ///< A | B: hold to hear the preset as it was
    };
    /**
     * @brief A button with @p glyph and @p tooltip; it never takes the keyboard focus.
     * @param[in] glyph    its glyph
     * @param[in] tooltip  its tooltip
     */
    ViewButton( GlyphE glyph, const juce::String &tooltip );
    /**
     * @brief The frame's button background, the glyph over it.
     * @param[in,out] g            where
     * @param[in]     highlighted  under the mouse
     * @param[in]     down         pressed
     */
    void paintButton( juce::Graphics &g, bool highlighted, bool down ) override;

private:
    GlyphE m_glyph;   ///< what it draws
};

/** @brief The output's meter (see the file comment). */
class OutputMeter final : public juce::Component, public juce::SettableTooltipClient
{
public:
    OutputMeter();
    /**
     * @brief The peaks since the last look, per side (0 .. 1 and over).
     * @param[in] left   left
     * @param[in] right  right
     */
    void push( float left, float right );
    /**
     * @brief The two bars and the clip mark.
     * @param[in,out] g  where
     */
    void paint( juce::Graphics &g ) override;
    /**
     * @brief A click puts the clip mark out.
     * @param[in] e  the click
     */
    void mouseDown( const juce::MouseEvent &e ) override;

private:
    float m_level[2] = { 0.0f, 0.0f };   ///< what each bar shows, falling slowly (linear)
    float m_hold[2] = { 0.0f, 0.0f };    ///< the held peak
    int   m_holdTicks[2] = { 0, 0 };     ///< how much longer it is held
    bool  m_clipped = false;             ///< the output reached full scale
};
