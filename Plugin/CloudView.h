/**
 * @file CloudView.h
 * @brief The picture above the pages: the partials themselves, as the GPU has them -- every
 *        eighth partial of every voice a point, its frequency across (log), its loudness up, coloured
 *        by whose note it is (played, the conductor, the second conductor, a ghost of the memory),
 *        leaving a trail that fades; under it the keys, each at its frequency, the notes that sound
 *        lit in the same colours and their names in the corner -- and beside it the landscape the
 *        Aurora motion reads, with where in its loop the voices are.
 */
#pragma once

#include "PluginProcessor.h"

#include <array>

/** @brief The cloud picture (see the file comment). */
class CloudView final : public juce::Component {
public:
    /** @brief A picture of @p proc's voices. @param proc the processor */
    explicit CloudView(AdditivProcessor& proc) : proc_(proc) {}
    /** @brief Reads the voices and the parameters it draws with and repaints (the editor's timer). */
    void refresh();
    /** @brief Draws the clouds and the landscape. */
    void paint(juce::Graphics& g) override;
    /** @brief A click on the landscape opens a picture or a recording as the player's own. @param e the click */
    void mouseUp(const juce::MouseEvent& e) override;

private:
    /** @brief Remakes the landscape's picture when the choice changed. */
    void refreshLandscape();
    /** @brief Fades the trails and plots the newest snapshot into them. @param w width @param h height in pixels */
    void plotDots(int w, int h);
    /** @brief A frequency's place on the axis. @param hz the frequency @param area the axis' area @return x */
    static float xOf(float hz, juce::Rectangle<float> area);
    /**
     * @brief The keys under the spectrum, A0 to C8, each where its pitch lies on the axis: the notes
     *        that sound lit by whose they are (the partials' colours), dimmer the further they stand and
     *        the softer they are; the conductor's root marked; the sounding notes' names in the
     *        spectrum's corner, low to high.
     * @param g     where
     * @param keys  the strip
     * @param spec  the spectrum's area (the axis, and the corner for the names)
     */
    void paintKeys(juce::Graphics& g, juce::Rectangle<float> keys, juce::Rectangle<float> spec);

    AdditivProcessor& proc_;                                            ///< whose voices
    std::array<additiv::VoiceView, additiv::kMaxViewVoices> views_{};  ///< the voices as the last block left them
    int landChoice_ = -1;                                               ///< the landscape pictured
    int landVersion_ = -1;                                              ///< the player's landscape pictured (its version)
    juce::Rectangle<float> landArea_;                                   ///< where the landscape is drawn (a click there opens a file)
    std::unique_ptr<juce::FileChooser> chooser_;                        ///< the open dialog while it is open
    juce::Image land_;                                                  ///< its picture: time left to right, low at the bottom
    float landLoop_ = 30.0f;                                            ///< its loop, seconds
    std::vector<float> snap_;                                           ///< the partials' snapshot: (Hz, amplitude) per point
    int snapPer_ = 0;                                                   ///< its points per voice
    std::vector<float> glow_;                                           ///< the trails: red, green, blue per pixel, fading
    juce::Image dots_;                                                  ///< the trails as a picture
    juce::Rectangle<int> dotsArea_;                                     ///< where the points are drawn (the spectrum's area)
    std::vector<float> lifeCells_;                                      ///< the living field of the last snapshot, empty when off
    juce::Image life_;                                                  ///< its picture
};
