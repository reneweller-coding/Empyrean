/**
 * @file SpaceView.h
 * @brief The Space page: its knobs above, and the space itself below, seen from above -- you at the
 *        middle of the bottom edge, the near plane close to you, the far plane at the rim; every voice a
 *        light at its plane (as Breath lets it wander, leaving a trail), across from its pitch (low
 *        notes to the left, high to the right), as large as it is loud, in the colour of whose note it
 *        is. The conductor's depth and the played notes' depth are rings to drag.
 * @author Rene Weller
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <array>

/** @brief The space from above (see the file comment). */
class SpaceView final : public juce::Component, private juce::Timer
{
public:
    /**
     * @brief The voices of @p proc.
     * @param[in,out] proc  the processor
     */
    explicit SpaceView( AdditivProcessor &proc );
    /**
     * @brief The half disc, the planes, the voices and their trails, the two depths.
     * @param[in,out] g  the graphics context
     */
    void paint( juce::Graphics &g ) override;
    /**
     * @brief Takes a depth's ring.
     * @param[in] e  the click
     */
    void mouseDown( const juce::MouseEvent &e ) override;
    /**
     * @brief Moves the ring taken.
     * @param[in] e  the drag
     */
    void mouseDrag( const juce::MouseEvent &e ) override;
    /**
     * @brief Lets go of it.
     * @param[in] e  the release
     */
    void mouseUp( const juce::MouseEvent &e ) override;
    /** @brief What it shows, in words. @return the text */
    juce::String describe() const;

private:
    /** @brief The voices anew, their trails a step older. */
    void timerCallback() override;
    /** @brief Where you are: the middle of the half disc's straight edge. @return pixels */
    juce::Point<float> ear() const;
    /** @brief The half disc's radius. @return pixels */
    float radius() const;
    /**
     * @brief A plane's radius from you: the ear's head room, then the planes out to the rim.
     * @param[in] depth  0 .. 1
     * @return pixels
     */
    float planeRadius( float depth ) const;
    /**
     * @brief The depth at a distance from you.
     * @param[in] r  pixels
     * @return 0 .. 1
     */
    float depthAt( float r ) const;

    static constexpr int Trail = 24;   ///< positions a voice's trail keeps (about three seconds)
    AdditivProcessor    &m_proc;       ///< whose voices
    std::array<std::array<juce::Point<float>, Trail>, additiv::MaxViewVoices>
        m_trails{};   ///< per voice its last places (depth, pitch 0 .. 1)
    std::array<additiv::VoiceView, additiv::MaxViewVoices> m_views{};    ///< the voices now
    std::array<int, additiv::MaxViewVoices>                m_seen{};     ///< how many places each trail holds
    int                                                    m_head = 0;   ///< where the next places go
    int                                                    m_tick = 0;   ///< timer ticks
    int m_held = -1;   ///< the ring taken: 0 Depth, 1 Keys Depth, -1 none
};

/** @brief The Space page: its knobs, and the space below them. */
class SpacePage final : public juce::Component
{
public:
    /**
     * @brief Builds the page.
     * @param[in,out] proc     the processor
     * @param[in]     actions  the controls' right-click menu
     * @param[in,out] live     the live rings
     */
    SpacePage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live );
    /**
     * @brief The height the page needs at a width.
     * @param[in] w  pixels
     * @return pixels
     */
    int heightFor( int w ) const;
    /** @brief The knobs at the top, the space below. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage m_params;   ///< Planes, Near, Far Reverb, Master, Binaural
    SpaceView m_view;     ///< the space
};
