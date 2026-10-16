/**
 * @file Pose.h
 * @brief The headset in the cloud (Phase 30): the head's turn and both hands, over OSC from Empyrean's Quest app,
 *        beside the family's hands (Frame.h's Headset): "/pose" on the headset's port plus 100.
 * @author Rene Weller
 */
#pragma once

#include "additiv/Engine.h"

#include <juce_osc/juce_osc.h>

/**
 * @brief Receives "/pose" with twelve floats -- the head's yaw and pitch (radians), the left hand's x, y, z and the
 *        right's (metres from the head, in the room as it stood when the app began: x right, y up, z ahead), the
 *        pinches (0 .. 1) and whether each hand is seen (0 or 1) -- about sixty times a second. A pose older than a
 *        second is none.
 */
class PoseReceiver final : private juce::OSCReceiver::Listener<juce::OSCReceiver::MessageLoopCallback>
{
public:
    PoseReceiver();
    ~PoseReceiver() override;
    PoseReceiver( const PoseReceiver & ) = delete;              ///< owns a socket: not copyable
    PoseReceiver &operator=( const PoseReceiver & ) = delete;   ///< owns a socket: not copyable
    /**
     * @brief Listens on @p port (0: not at all).
     * @param[in] port  the UDP port, 0 for none
     * @pre called on the message thread
     */
    void listen( int port );
    /**
     * @brief The pose as it arrived last; not valid when none came in the last second.
     * @return the pose
     * @pre called on the message thread
     */
    additiv::HeadsetPose pose() const;
    /**
     * @brief A line for the settings menu: where it listens and what arrives.
     * @return the line
     */
    juce::String statusText() const;

private:
    /**
     * @brief Takes a "/pose" message (anything else is ignored).
     * @param[in] m  the message
     */
    void oscMessageReceived( const juce::OSCMessage &m ) override;

    juce::OSCReceiver    m_receiver;       ///< the socket
    int                  m_port = 0;       ///< the port it listens on, 0 none
    juce::String         m_error;          ///< why it does not, empty when it does
    additiv::HeadsetPose m_pose;           ///< the last pose
    double               m_last = 0.0;     ///< ms (the high-resolution counter) it arrived
    long long            m_messages = 0;   ///< poses received
};
