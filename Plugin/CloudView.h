/**
 * @file CloudView.h
 * @brief The picture above the pages: the partials themselves, as the GPU has them -- every
 *        eighth partial of every voice a point, its frequency across (log), its loudness up, coloured
 *        by whose note it is (played, the conductor, the second conductor, a ghost of the memory),
 *        leaving a trail that fades; under it the keys, each at its frequency, the notes that sound
 *        lit in the same colours and their names in the corner -- and beside it the landscape the
 *        Aurora motion reads, with where in its loop the voices are.
 *
 *        It shows what the page in front edits, where it acts, to be taken by the hand: on Filter the
 *        filter's curve (drag its point: across the cutoff, up the resonance) and the comb's notches; on
 *        Cloud the air's band (its centre and its edge) and where the brightness fades; on Conductor
 *        the register as a bracket over the keys; on Motion the ripple's string; on Hall the hall's
 *        memory and its lowest mode.
 *
 *        As a canvas (the editor's Canvas mode) it is the whole window: the living landscape or the
 *        landscape faintly behind the partials, the four macros at the bottom, and a click in the
 *        spectrum plays the key under it (higher up, louder).
 */
#pragma once

#include "PluginProcessor.h"

#include <array>
#include <memory>

/** @brief The cloud picture (see the file comment). */
class CloudView final : public juce::Component {
public:
    /** @brief A picture of @p proc's voices. @param proc the processor */
    explicit CloudView(AdditivProcessor& proc);
    /** @brief Lets go of a key it plays. */
    ~CloudView() override;
    /** @brief Reads the voices and the parameters it draws with and repaints (the editor's timer). */
    void refresh();
    /** @brief The page in front, whose overlay it draws. @param page the tab's name */
    void setPage(const juce::String& page) { page_ = page; }
    /** @brief The whole window, or the strip above the pages. @param on canvas */
    void setCanvas(bool on);
    /** @brief Whether it is the canvas. @return true then */
    bool canvas() const { return canvas_; }
    /** @brief Called on a double click in the spectrum (the editor switches the canvas). */
    std::function<void()> onDoubleClick;
    /** @brief Draws the clouds and the landscape, and the page's overlay. */
    void paint(juce::Graphics& g) override;
    /** @brief The macros' faders in the canvas. */
    void resized() override;
    /** @brief Takes an overlay's handle, or (canvas) plays the key under the mouse. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief Moves the handle taken, or the key played. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override;
    /** @brief Lets go; a click on the landscape opens a picture or a recording as the player's own. @param e the click */
    void mouseUp(const juce::MouseEvent& e) override;
    /** @brief The hand over a handle. @param e the move */
    void mouseMove(const juce::MouseEvent& e) override;
    /** @brief Switches the canvas (onDoubleClick). @param e the click */
    void mouseDoubleClick(const juce::MouseEvent& e) override;

private:
    /** @brief What can be taken in the picture. */
    enum class Handle { None, Filter, AirCentre, AirEdge, Brightness, RegisterLow, RegisterHigh, HallLowest, Count };

    /** @brief Remakes the landscape's picture when the choice changed. */
    void refreshLandscape();
    /** @brief Fades the trails and plots the newest snapshot into them. @param w width @param h height in pixels */
    void plotDots(int w, int h);
    /** @brief A frequency's place on the axis. @param hz the frequency @param area the axis' area @return x */
    static float xOf(float hz, juce::Rectangle<float> area);
    /** @brief The frequency at a place on the axis. @param x pixels @param area the axis' area @return Hz */
    static float hzAt(float x, juce::Rectangle<float> area);
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
    /** @brief The page's overlay, and where its handles are. @param g where @param spec the spectrum @param keys the keys' strip */
    void paintOverlay(juce::Graphics& g, juce::Rectangle<float> spec, juce::Rectangle<float> keys);
    /** @brief A handle as a ring, brighter while held or under the mouse. @param g where @param h which @param at where @param c its colour */
    void paintHandle(juce::Graphics& g, Handle h, juce::Point<float> at, juce::Colour c);
    /** @brief The handle near a point, on the page in front. @param p the point @return it, or None */
    Handle handleAt(juce::Point<float> p) const;
    /** @brief The fundamental the cloud page's overlay is drawn for: the lowest sounding voice's, or C3. @return Hz */
    float referenceHz() const;
    /** @brief A parameter's value. @param id the parameter @return its value */
    float value(additiv::ParamId id) const { return proc_.value(static_cast<int>(id)); }
    /** @brief Sets a parameter (within its range). @param id the parameter @param v the value */
    void set(additiv::ParamId id, float v) { proc_.setValue(static_cast<int>(id), v); }
    /** @brief The parameters a handle moves. @param h the handle @return them (one or two) */
    static std::vector<additiv::ParamId> paramsOf(Handle h);
    /** @brief The key under a place in the spectrum. @param x pixels @return the MIDI note, 21 .. 108 */
    int keyAt(float x) const;

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
    juce::String page_;                                                 ///< the page in front
    bool canvas_ = false;                                               ///< the whole window
    juce::Rectangle<float> spec_;                                       ///< the spectrum's area as last drawn
    std::array<juce::Point<float>, static_cast<size_t>(Handle::Count)> handles_{};   ///< where each handle was drawn
    std::array<bool, static_cast<size_t>(Handle::Count)> shown_{};                    ///< whether it was drawn
    Handle held_ = Handle::None;                                        ///< the handle being dragged
    Handle hover_ = Handle::None;                                       ///< the handle under the mouse
    float heldResonance_ = 0.0f;                                        ///< the resonance when the filter's point was taken
    float heldY_ = 0.0f;                                                ///< and where
    int playing_ = -1;                                                  ///< the key the canvas plays, -1 none
    std::vector<float> ripple_;                                         ///< the ripple's string at the last snapshot
    std::vector<float> hall_;                                           ///< the hall's memory as last fetched
    double hallAge_ = 0.0;                                              ///< its seconds learnt
    std::array<juce::Slider, 4> macros_;                                ///< the four macros as knobs, in the canvas
    std::array<juce::Label, 4> macroNames_;                             ///< their names
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> macroLinks_;   ///< the faders to the parameters
};
