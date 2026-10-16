/**
 * @file Pose.cpp
 * @brief The headset in the cloud (Pose.h).
 * @author Rene Weller
 */
#include "Pose.h"

#include <cmath>

PoseReceiver::PoseReceiver()
{
    m_receiver.addListener( this );
}

PoseReceiver::~PoseReceiver()
{
    m_receiver.removeListener( this );
    m_receiver.disconnect();
}

void PoseReceiver::listen( int port )
{
    if ( port == m_port && ( port == 0 || m_error.isEmpty() ) ) return;
    m_receiver.disconnect();
    m_port = port;
    m_error = {};
    if ( port <= 0 ) return;
    if ( !m_receiver.connect( port ) ) m_error = "port " + juce::String( port ) + " is taken";
}

additiv::HeadsetPose PoseReceiver::pose() const
{
    additiv::HeadsetPose p = m_pose;
    p.m_valid = m_messages > 0 && juce::Time::getMillisecondCounterHiRes() - m_last < 1000.0;
    return p;
}

juce::String PoseReceiver::statusText() const
{
    if ( m_port <= 0 ) return "the head and hands: not listening (Off)";
    if ( m_error.isNotEmpty() ) return "the head and hands: " + m_error;
    if ( pose().m_valid ) return "the head and hands arrive (port " + juce::String( m_port ) + ")";
    return "listening for the head and hands on port " + juce::String( m_port );
}

void PoseReceiver::oscMessageReceived( const juce::OSCMessage &m )
{
    if ( m.getAddressPattern().toString() != "/pose" || m.size() < 12 ) return;
    auto f = [&]( int i )
    {
        const float v = m[i].isFloat32() ? m[i].getFloat32()
                        : m[i].isInt32() ? static_cast<float>( m[i].getInt32() )
                                         : 0.0f;
        return std::isfinite( v ) ? v : 0.0f;
    };
    additiv::HeadsetPose p;
    p.m_yaw = f( 0 );
    p.m_pitch = f( 1 );
    for ( int h = 0; h < 2; ++h )
    {
        for ( int a = 0; a < 3; ++a ) p.m_hand[h][a] = juce::jlimit( -3.0f, 3.0f, f( 2 + 3 * h + a ) );
        p.m_pinch[h] = juce::jlimit( 0.0f, 1.0f, f( 8 + h ) );
        p.m_tracked[h] = f( 10 + h ) > 0.5f;
    }
    p.m_valid = true;
    m_pose = p;
    m_last = juce::Time::getMillisecondCounterHiRes();
    ++m_messages;
}
