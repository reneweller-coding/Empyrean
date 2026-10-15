/**
 * @file HallView.h
 * @brief The Hall page: the modal hall's knobs, its growth (Grow, Memory) and its memory drawn band by
 *        band -- what it has heard over the hours, where it rings longer (growth) or shorter (erosion)
 *        -- with saving and loading a hall (Documents/Empyrean/Halls) and starting a new one.
 * @author Rene Weller
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** @brief The hall's memory drawn, its age, and the buttons that keep it (see the file comment). */
class HallView final : public juce::Component, private juce::Timer
{
public:
    /**
     * @brief The memory of @p proc's engine.
     * @param[in,out] proc  the processor
     */
    explicit HallView( AdditivProcessor &proc );
    /** @brief The buttons and the status. */
    void resized() override;
    /**
     * @brief The box, the curve, the octaves.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override;
    /** @brief What it shows, in words. @return the text */
    juce::String describe() const;
    /**
     * @brief A stroke begins: the memory painted where the mouse is.
     * @param[in] e  the press
     */
    void mouseDown( const juce::MouseEvent &e ) override;
    /**
     * @brief The stroke goes on, every band between the last place and this one painted.
     * @param[in] e  the drag
     */
    void mouseDrag( const juce::MouseEvent &e ) override;
    /**
     * @brief The stroke ends.
     * @param[in] e  the release
     */
    void mouseUp( const juce::MouseEvent &e ) override;

private:
    /**
     * @brief Paints the memory along a stroke, a soft brush round each band.
     * @param[in] from  where the stroke was
     * @param[in] to    where it is
     */
    void paintStroke( juce::Point<float> from, juce::Point<float> to );
    /** @brief The memory fetched anew, a few times a second. */
    void timerCallback() override;
    /** @brief The curve's area in the box. @return pixels */
    juce::Rectangle<float> plot() const;
    /** @brief Asks for a name and writes the hall into the halls folder. */
    void save();
    /** @brief Asks for a hall file and hears it in. */
    void load();
    /** @brief The status line: name, age, what Grow does with it. @return the text */
    juce::String statusText() const;

    AdditivProcessor  &m_proc;               ///< the processor
    std::vector<float> m_memory;             ///< the memory as last fetched, HallBins bands, 1 the average
    double             m_age = 0.0;          ///< its seconds learnt
    bool               m_painting = false;   ///< a stroke is under way: the memory is not fetched over it
    double             m_paintedAt = 0.0;    ///< when the last stroke ended (ms): the engine takes it at its next block
    juce::Point<float> m_last;               ///< where the stroke was
    juce::String       m_note;               ///< what the painting did besides (Grow switched on ...)
    /** @brief Saves the hall. */
    frame::IconButton m_save{ frame::IconButton::IconE::SAVE,
                              "Writes the hall's memory into Documents/Empyrean/Halls under a name" };
    /** @brief Loads a hall. */
    frame::IconButton m_load{ frame::IconButton::IconE::OPEN,
                              "Loads a hall's memory from a file: the hall rings as that one had grown" };
    juce::TextButton  m_new{ "New hall" };          ///< forgets everything: every band at the average
    juce::Label       m_status;                     ///< name, age, what Grow does
    std::unique_ptr<juce::FileChooser> m_chooser;   ///< the dialog while it is open
};

/** @brief The Hall page: the modal hall's knobs and its growth above, the memory below. */
class HallPage final : public juce::Component
{
public:
    /**
     * @brief Builds the page.
     * @param[in,out] proc     the processor
     * @param[in]     actions  the controls' right-click menu
     * @param[in,out] live     the live rings
     */
    HallPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live );
    /**
     * @brief The height the page needs at a width.
     * @param[in] w  pixels
     * @return pixels
     */
    int heightFor( int w ) const;
    /** @brief The knobs at the top, the memory below. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage m_params;   ///< Modal Hall; Growth
    HallView  m_view;     ///< the memory
};
