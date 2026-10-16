/**
 * @file main.cpp
 * @brief Empyrean for Meta Quest (Phase 30): you in the cloud -- the head and the hands sent to the desktop's Empyrean,
 *        which computes the sound on its graphics card.
 *
 * The cloud's twenty thousand partials a voice need the desktop's GPU, so this app makes no sound. It is the frame of
 * the family's Quest apps -- NativeActivity + android_native_app_glue, EGL, GLES 3, the Khronos OpenXR loader,
 * `XR_EXT_hand_tracking`, the point renderer and its font, the hands, the bridge (Totality's app, after Ephemeris'
 * and Phosphene's) -- without a synthesizer, with a second message:
 *
 *  - "/hands" to the headset's port (9105): the family's grammar, the same six floats every generator's app sends
 *    (Plugin/Frame.h, Headset): the pinches play and stop the conductor, step it, bring Brain 2 in and out, both hands
 *    the next preset; the hands' heights the brightness and the motion.
 *  - "/pose" to the port plus 100 (9205): the head's turn and pitch and where both hands are, in the room as it stood
 *    when the app began (recentred when the headset is recentred), sixty times a second (Plugin/Pose.h). Then the
 *    binaural cloud and the thunder stay put as the head turns, and the hands reach into the wind, the halo, the wave
 *    room and the slime mould's trail (the Rooms page: Hands, Head Tracking).
 *
 * In the headset: a sky of slow points round you, a ring at arm's reach where the hands act, the hands themselves, and
 * a panel that says where it sends.
 *
 * **`empy.cfg`** in `<externalDataPath>` (`/sdcard/Android/data/com.reneweller.empyrean.quest/files`):
 * @code
 *   bridge_host=192.168.1.20   the desktop running Empyrean (a dotted IPv4 address); nothing is sent without it
 *   bridge_port=9105           its headset port (the desktop's settings, Headset); the pose goes to this plus 100
 * @endcode
 * @author Rene Weller
 */

#include <android/log.h>
#include <android_native_app_glue.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <jni.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

/** @brief An info line in the Android log, tag "Empyrean" (printf style). */
#define LOGI( ... ) __android_log_print( ANDROID_LOG_INFO, "Empyrean", __VA_ARGS__ )
/** @brief An error line in the Android log, tag "Empyrean" (printf style). */
#define LOGE( ... ) __android_log_print( ANDROID_LOG_ERROR, "Empyrean", __VA_ARGS__ )

namespace
{

// ---------------------------------------------------------------- small math

/** @brief Column-major 4x4 matrix, the layout GL wants. */
struct Mat4
{
    float m_m[16];   ///< the sixteen values, column by column
};
/** @brief A point or direction. */
struct Vec3
{
    float m_x;   ///< right
    float m_y;   ///< up
    float m_z;   ///< back (towards the viewer)
};

/**
 * @brief The identity matrix.
 * @return the identity matrix
 */
Mat4 identity()
{
    Mat4 r{};
    r.m_m[0] = r.m_m[5] = r.m_m[10] = r.m_m[15] = 1.0f;
    return r;
}

/**
 * @brief The product @p a times @p b (b applied first).
 * @param[in] a  the left factor
 * @param[in] b  the right factor, applied first
 * @return the product
 */
Mat4 multiply( const Mat4 &a, const Mat4 &b )
{
    Mat4 r{};
    for ( int c = 0; c < 4; ++c )
        for ( int rr = 0; rr < 4; ++rr )
            r.m_m[c * 4 + rr] = a.m_m[0 * 4 + rr] * b.m_m[c * 4 + 0] + a.m_m[1 * 4 + rr] * b.m_m[c * 4 + 1] +
                                a.m_m[2 * 4 + rr] * b.m_m[c * 4 + 2] + a.m_m[3 * 4 + rr] * b.m_m[c * 4 + 3];
    return r;
}

/**
 * @brief Asymmetric projection from OpenXR's four half-angles.
 * @param[in] fov    the four half-angles
 * @param[in] nearZ  the near plane's distance
 * @param[in] farZ   the far plane's distance
 * @return the projection matrix
 */
Mat4 projectionFromFov( const XrFovf &fov, float nearZ, float farZ )
{
    const float l = std::tan( fov.angleLeft ), r = std::tan( fov.angleRight );
    const float u = std::tan( fov.angleUp ), d = std::tan( fov.angleDown );
    const float w = r - l, h = u - d;
    Mat4        p{};
    p.m_m[0] = 2.0f / w;
    p.m_m[8] = ( r + l ) / w;
    p.m_m[5] = 2.0f / h;
    p.m_m[9] = ( u + d ) / h;
    p.m_m[10] = -( farZ + nearZ ) / ( farZ - nearZ );
    p.m_m[11] = -1.0f;
    p.m_m[14] = -( 2.0f * farZ * nearZ ) / ( farZ - nearZ );
    return p;
}

/**
 * @brief The rotation of unit quaternion @p q as a matrix.
 * @param[in] q  the unit quaternion
 * @return the rotation matrix
 */
Mat4 rotationFromQuat( const XrQuaternionf &q )
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    Mat4        r = identity();
    r.m_m[0] = 1 - 2 * ( y * y + z * z );
    r.m_m[4] = 2 * ( x * y - z * w );
    r.m_m[8] = 2 * ( x * z + y * w );
    r.m_m[1] = 2 * ( x * y + z * w );
    r.m_m[5] = 1 - 2 * ( x * x + z * z );
    r.m_m[9] = 2 * ( y * z - x * w );
    r.m_m[2] = 2 * ( x * z - y * w );
    r.m_m[6] = 2 * ( y * z + x * w );
    r.m_m[10] = 1 - 2 * ( x * x + y * y );
    return r;
}

/**
 * @brief Vector @p v rotated by @p q.
 * @param[in] q  the unit quaternion
 * @param[in] v  the vector
 * @return the rotated vector
 */
Vec3 rotate( const XrQuaternionf &q, Vec3 v )
{
    const Mat4 r = rotationFromQuat( q );
    return { r.m_m[0] * v.m_x + r.m_m[4] * v.m_y + r.m_m[8] * v.m_z,
             r.m_m[1] * v.m_x + r.m_m[5] * v.m_y + r.m_m[9] * v.m_z,
             r.m_m[2] * v.m_x + r.m_m[6] * v.m_y + r.m_m[10] * v.m_z };
}

/**
 * @brief The view matrix of an eye pose: the inverse of the pose.
 * @param[in] pose  the eye's pose
 * @return the view matrix
 */
Mat4 viewFromPose( const XrPosef &pose )
{
    const Mat4 r = rotationFromQuat( pose.orientation );
    Mat4       rt = identity();
    for ( int c = 0; c < 3; ++c )
        for ( int rr = 0; rr < 3; ++rr ) rt.m_m[c * 4 + rr] = r.m_m[rr * 4 + c];
    Mat4 t = identity();
    t.m_m[12] = -pose.position.x;
    t.m_m[13] = -pose.position.y;
    t.m_m[14] = -pose.position.z;
    return multiply( rt, t );
}

/**
 * @brief @p x limited to 0 .. 1.
 * @param[in] x  the value
 * @return the value limited to 0 .. 1
 */
float clamp01( float x )
{
    return x < 0.0f ? 0.0f : ( x > 1.0f ? 1.0f : x );
}

/**
 * @brief The head's turn about the vertical: its forward direction's angle, 0 along -z, positive to the right.
 * @param[in] q  the head's orientation
 * @return radians
 */
float yawOf( const XrQuaternionf &q )
{
    const Vec3 f = rotate( q, { 0.0f, 0.0f, -1.0f } );
    return std::atan2( f.m_x, -f.m_z );
}

/**
 * @brief An integer hash with good avalanche (lowbias32): the sky's dice.
 * @param[in] x  any value
 * @return its hash
 */
uint32_t hash32( uint32_t x )
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/**
 * @brief A uniform number in [0, 1) from two dice.
 * @param[in] a  the first
 * @param[in] b  the second
 * @return 0 .. 1
 */
float uniform( uint32_t a, uint32_t b )
{
    return static_cast<float>( hash32( a * 0x9E3779B9U ^ hash32( b ) ) >> 8 ) * ( 1.0f / 16777216.0f );
}

// ---------------------------------------------------------------- 5x7 point font

/**
 * @brief One glyph as seven rows of five bits (bit 4 = left column), or the blank for the unknown (Noctuary's table,
 *        measured legible at arm's length).
 * @param[in] c  the character
 * @return its seven rows
 */
const unsigned char *glyph( char c )
{
    static const unsigned char Font[][7] = {
        { 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 }, { 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E },
        { 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E },   // A B C
        { 0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E }, { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F },
        { 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10 },   // D E F
        { 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F }, { 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 },
        { 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E },   // G H I
        { 0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E }, { 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 },
        { 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F },   // J K L
        { 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11 }, { 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11 },
        { 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },   // M N O
        { 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 }, { 0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D },
        { 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 },   // P Q R
        { 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E }, { 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
        { 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },   // S T U
        { 0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04 }, { 0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A },
        { 0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11 },                                                 // V W X
        { 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04 }, { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F },   // Y Z
        { 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E }, { 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E },
        { 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F },   // 0 1 2
        { 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E }, { 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 },
        { 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E },   // 3 4 5
        { 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E }, { 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 },
        { 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E },   // 6 7 8
        { 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C },   // 9
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, { 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C },   // space - .
        { 0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00 }, { 0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },   // : / unknown
    };
    if ( c >= 'a' && c <= 'z' ) c = static_cast<char>( c - 'a' + 'A' );
    if ( c >= 'A' && c <= 'Z' ) return Font[c - 'A'];
    if ( c >= '0' && c <= '9' ) return Font[26 + ( c - '0' )];
    switch ( c )
    {
    case ' ': return Font[36];
    case '-': return Font[37];
    case '.': return Font[38];
    case ':': return Font[39];
    case '/': return Font[40];
    default: break;
    }
    return Font[41];
}

// ---------------------------------------------------------------- config

/** @brief What `empy.cfg` can say. */
struct Config
{
    std::string m_bridgeHost;          ///< the desktop running Empyrean; empty: nothing is sent
    int         m_bridgePort = 9105;   ///< its headset port (the desktop's settings, Headset); the pose to this + 100
};

/**
 * @brief Reads `<dir>/empy.cfg`; every key is optional.
 * @param[in] dir  the folder of empy.cfg
 * @return the configuration
 */
Config readConfig( const char *dir )
{
    Config c;
    if ( dir == nullptr ) return c;
    const std::string path = std::string( dir ) + "/empy.cfg";
    FILE             *f = std::fopen( path.c_str(), "r" );
    if ( f == nullptr )
    {
        LOGI( "no config at %s", path.c_str() );
        return c;
    }
    char line[512];
    while ( std::fgets( line, sizeof( line ), f ) )
    {
        std::string s( line );
        while ( !s.empty() && ( s.back() == '\n' || s.back() == '\r' || s.back() == ' ' ) ) s.pop_back();
        if ( s.empty() || s[0] == '#' ) continue;
        const size_t eq = s.find( '=' );
        if ( eq == std::string::npos ) continue;
        const std::string k = s.substr( 0, eq ), v = s.substr( eq + 1 );
        if ( k == "bridge_host" )
            c.m_bridgeHost = v;
        else if ( k == "bridge_port" )
            c.m_bridgePort = std::atoi( v.c_str() );
        else
            LOGE( "empy.cfg: unknown key %s", k.c_str() );
    }
    std::fclose( f );
    return c;
}

// ---------------------------------------------------------------- GL scene

/** @brief The vertex shader: a point at its position, sized by its distance. */
const char *kVertexShader = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aCol;
layout(location = 2) in float aSize;
uniform mat4 uVP;
out vec4 vCol;
void main() {
    vec4 p = uVP * vec4(aPos, 1.0);
    gl_Position = p;
    gl_PointSize = aSize / max(p.w, 0.2);
    vCol = aCol;
})";

/** @brief The fragment shader: a soft round dot, added to what is there. */
const char *kFragmentShader = R"(#version 300 es
precision mediump float;
in vec4 vCol;
out vec4 o;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d) * 2.0;
    float a = smoothstep(1.0, 0.15, r);
    o = vec4(vCol.rgb * a * vCol.a, 1.0);
})";

/** @brief One soft round point: the only primitive the scene has. */
struct Point
{
    float m_x;      ///< position in the room, metres
    float m_y;      ///< height in the room, metres
    float m_z;      ///< depth in the room, metres
    float m_r;      ///< red 0..1
    float m_g;      ///< green 0..1
    float m_b;      ///< blue 0..1
    float m_a;      ///< brightness 0..1
    float m_size;   ///< diameter, pixels at one metre
};

/** @brief A head-locked plane to lay text out on: origin plus a right and an up axis. */
struct Panel
{
    Vec3 m_origin;   ///< the top left of the text, in the room
    Vec3 m_right;    ///< one metre to the right along the panel
    Vec3 m_up;       ///< one metre up along the panel
};

/**
 * @brief A shader of @p type compiled from @p src; a failure is logged.
 * @param[in] type  the shader's type
 * @param[in] src   its source
 * @return the shader, 0 on failure
 */
GLuint compile( GLenum type, const char *src )
{
    const GLuint s = glCreateShader( type );
    glShaderSource( s, 1, &src, nullptr );
    glCompileShader( s );
    GLint ok = 0;
    glGetShaderiv( s, GL_COMPILE_STATUS, &ok );
    if ( !ok )
    {
        char log[1024];
        glGetShaderInfoLog( s, sizeof( log ), nullptr, log );
        LOGE( "shader: %s", log );
    }
    return s;
}

/** @brief Everything visible, as additive points (nothing about the camera moves by itself). */
class Scene
{
public:
    /** @brief Display pixels per radian on the Quest's render target (about 1830 px over 90 degrees). */
    static constexpr float PixelsPerRadian = 1150.0f;

    /**
     * @brief Links the shaders and sets up the point buffer; false if the program does not link.
     * @return false when the program does not link
     */
    bool init()
    {
        m_program = glCreateProgram();
        glAttachShader( m_program, compile( GL_VERTEX_SHADER, kVertexShader ) );
        glAttachShader( m_program, compile( GL_FRAGMENT_SHADER, kFragmentShader ) );
        glLinkProgram( m_program );
        GLint ok = 0;
        glGetProgramiv( m_program, GL_LINK_STATUS, &ok );
        if ( !ok )
        {
            LOGE( "program link failed" );
            return false;
        }
        m_uVP = glGetUniformLocation( m_program, "uVP" );
        glGenBuffers( 1, &m_vbo );
        glGenVertexArrays( 1, &m_vao );
        glBindVertexArray( m_vao );
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof( Point ), reinterpret_cast<void *>( 0 ) );
        glEnableVertexAttribArray( 1 );
        glVertexAttribPointer( 1, 4, GL_FLOAT, GL_FALSE, sizeof( Point ), reinterpret_cast<void *>( 12 ) );
        glEnableVertexAttribArray( 2 );
        glVertexAttribPointer( 2, 1, GL_FLOAT, GL_FALSE, sizeof( Point ), reinterpret_cast<void *>( 28 ) );
        glBindVertexArray( 0 );
        return true;
    }

    /** @brief Starts a new picture: forgets the points of the last frame. */
    void begin() { m_points.clear(); }

    /**
     * @brief One point in the room.
     * @param[in] x     x in metres
     * @param[in] y     y in metres
     * @param[in] z     z in metres
     * @param[in] r     red, 0..1
     * @param[in] g     green, 0..1
     * @param[in] b     blue, 0..1
     * @param[in] a     brightness, 0..1
     * @param[in] size  the diameter in pixels at one metre
     */
    void add( float x, float y, float z, float r, float g, float b, float a, float size )
    {
        m_points.push_back( { x, y, z, r, g, b, a, size } );
    }

    /**
     * @brief Text on a panel, one point per lit pixel of the 5x7 font; @p cell is the pixel pitch.
     * @param[in] p     the panel
     * @param[in] u     metres to the right
     * @param[in] v     metres up
     * @param[in] cell  the pitch of one font pixel in metres
     * @param[in] text  the text
     * @param[in] r     red, 0..1
     * @param[in] g     green, 0..1
     * @param[in] b     blue, 0..1
     * @param[in] a     brightness, 0..1
     */
    void addText( const Panel &p, float u, float v, float cell, const char *text, float r, float g, float b, float a )
    {
        for ( const char *c = text; *c; ++c )
        {
            const unsigned char *gl = glyph( *c );
            for ( int row = 0; row < 7; ++row )
                for ( int col = 0; col < 5; ++col )
                    if ( gl[row] & ( 0x10 >> col ) )
                    {
                        const float uu = u + static_cast<float>( col ) * cell,
                                    vv = v - static_cast<float>( row ) * cell;
                        add( p.m_origin.m_x + p.m_right.m_x * uu + p.m_up.m_x * vv,
                             p.m_origin.m_y + p.m_right.m_y * uu + p.m_up.m_y * vv,
                             p.m_origin.m_z + p.m_right.m_z * uu + p.m_up.m_z * vv, r, g, b, a,
                             cell * PixelsPerRadian );
                    }
            u += 6.0f * cell;
        }
    }

    /** @brief Sends this frame's points to the GPU. */
    void upload()
    {
        glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
        glBufferData( GL_ARRAY_BUFFER, static_cast<GLsizeiptr>( m_points.size() * sizeof( Point ) ), m_points.data(),
                      GL_DYNAMIC_DRAW );
    }

    /**
     * @brief Draws the points for one eye with view-projection @p vp into a @p width by @p height target.
     * @param[in] vp      the view-projection matrix
     * @param[in] width   the target's width
     * @param[in] height  the target's height
     */
    void draw( const Mat4 &vp, int width, int height )
    {
        glViewport( 0, 0, width, height );
        glClearColor( 0.006f, 0.008f, 0.016f, 1.0f );   // a night sky's blue, nearly black
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glDisable( GL_DEPTH_TEST );
        glEnable( GL_BLEND );
        glBlendFunc( GL_ONE, GL_ONE );   // additive: points add light, they never occlude
        glUseProgram( m_program );
        glUniformMatrix4fv( m_uVP, 1, GL_FALSE, vp.m_m );
        glBindVertexArray( m_vao );
        glDrawArrays( GL_POINTS, 0, static_cast<GLsizei>( m_points.size() ) );
        glBindVertexArray( 0 );
    }

private:
    GLuint             m_program = 0;   ///< the shader program
    GLuint             m_vbo = 0;       ///< the points' buffer
    GLuint             m_vao = 0;       ///< the points' vertex layout
    GLint              m_uVP = -1;      ///< the location of the view-projection uniform
    std::vector<Point> m_points;        ///< this frame's points
};

// ---------------------------------------------------------------- hands

/**
 * @brief The hands as tracked: each palm, its pinch with a Schmitt trigger (closed under 22 mm between the thumb's and
 *        the index's tips, open over 38 mm), its height against the head (`(palm.y - (head.y - 1.0)) / 0.8`, the
 *        family's measure), and the joints for the picture.
 */
class Hands
{
public:
    /** @brief One hand's measurement of this frame. */
    struct Hand
    {
        bool       m_valid = false;                          ///< tracked this frame
        XrVector3f m_palm{};                                 ///< the palm's position in the room
        float      m_pinch = 0.0f;                           ///< 0 open .. 1 closed
        bool       m_closed = false;                         ///< after the Schmitt trigger
        float      m_height = 0.5f;                          ///< 0..1 against the head
        XrVector3f m_joints[XR_HAND_JOINT_COUNT_EXT] = {};   ///< every joint, for the picture
    };

    /**
     * @brief Feeds one hand.
     * @param[in] h             the hand, 0 left, 1 right
     * @param[in] joints        its joints this frame (XR_HAND_JOINT_COUNT_EXT)
     * @param[in] thumbToIndex  the distance of the thumb's and the index's tips
     * @param[in] headY         the head's height in the same space
     */
    void setHand( int h, const XrHandJointLocationEXT *joints, float thumbToIndex, float headY )
    {
        Hand &s = m_hand[h];
        s.m_valid = true;
        s.m_palm = joints[XR_HAND_JOINT_PALM_EXT].pose.position;
        for ( int j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j ) s.m_joints[j] = joints[j].pose.position;
        s.m_pinch = clamp01( 1.0f - ( thumbToIndex - 0.015f ) / 0.035f );
        if ( thumbToIndex < 0.022f )
            s.m_closed = true;
        else if ( thumbToIndex > 0.038f )
            s.m_closed = false;
        s.m_height = clamp01( ( s.m_palm.y - ( headY - 1.0f ) ) / 0.8f );
    }
    /**
     * @brief Marks a hand as not tracked this frame.
     * @param[in] h  the hand, 0 left, 1 right
     */
    void lost( int h )
    {
        m_hand[h].m_valid = false;
        m_hand[h].m_closed = false;
    }
    /**
     * @brief Hand @p h: 0 left, 1 right.
     * @param[in] h  the hand, 0 left, 1 right
     * @return the hand
     */
    const Hand &hand( int h ) const { return m_hand[h]; }

private:
    Hand m_hand[2];   ///< left, right
};

// ---------------------------------------------------------------- the bridge

/**
 * @brief The bridge to the desktop's Empyrean over OSC (UDP), fire and forget: "/hands" to the headset's port, the
 *        family's six floats (left height, right height, left pinch, right pinch, left tracked, right tracked), thirty
 *        times a second; "/pose" to the port plus 100, twelve floats (the head's yaw and pitch, the left hand's x, y, z
 *        and the right's from the head in the room as it stood when the app began, the pinches, whether each is seen),
 *        sixty times a second.
 */
class Bridge
{
public:
    Bridge() = default;
    Bridge( const Bridge & ) = delete;              ///< owns a socket: not copyable
    Bridge &operator=( const Bridge & ) = delete;   ///< owns a socket: not copyable
    ~Bridge() { close(); }
    /**
     * @brief Opens the UDP socket to @p host (a dotted IPv4 address, no names) and @p port; false if it cannot.
     * @param[in] host  the dotted IPv4 address
     * @param[in] port  the headset port (the pose goes to this plus 100)
     * @return false when the socket cannot be opened
     */
    bool open( const std::string &host, int port )
    {
        close();
        m_sock = ::socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
        if ( m_sock < 0 ) return false;
        std::memset( &m_hands, 0, sizeof( m_hands ) );
        m_hands.sin_family = AF_INET;
        m_hands.sin_port = htons( static_cast<uint16_t>( port ) );
        if ( inet_pton( AF_INET, host.c_str(), &m_hands.sin_addr ) != 1 )
        {
            close();
            return false;
        }
        m_pose = m_hands;
        m_pose.sin_port = htons( static_cast<uint16_t>( port + 100 ) );
        return true;
    }
    /** @brief Closes the socket; send() then does nothing. */
    void close()
    {
        if ( m_sock >= 0 ) ::close( m_sock );
        m_sock = -1;
    }
    /**
     * @brief Whether the bridge is on.
     * @return true while the bridge is on
     */
    bool ok() const { return m_sock >= 0; }
    /**
     * @brief Sends the hands (thirty times a second) and the pose (sixty).
     * @param[in] hands  the hands
     * @param[in] pose   the pose's twelve floats
     * @param[in] dt     the seconds since the last frame
     * @pre called on the render thread
     */
    void send( const Hands &hands, const float *pose, double dt )
    {
        if ( m_sock < 0 ) return;
        m_sinceHands += dt;
        m_sincePose += dt;
        if ( m_sinceHands >= 1.0 / 30.0 )
        {
            m_sinceHands = 0.0;
            float a[6];
            for ( int h = 0; h < 2; ++h )
            {
                const Hands::Hand &s = hands.hand( h );
                a[h] = s.m_height;
                a[2 + h] = s.m_valid && s.m_closed ? 1.0f : 0.0f;
                a[4 + h] = s.m_valid ? 1.0f : 0.0f;
            }
            message( "/hands", a, 6, m_hands );
        }
        if ( m_sincePose >= 1.0 / 60.0 )
        {
            m_sincePose = 0.0;
            message( "/pose", pose, 12, m_pose );
        }
    }

private:
    /**
     * @brief One OSC message of floats to @p to.
     * @param[in] address  its address
     * @param[in] values   the floats
     * @param[in] count    how many (at most 16)
     * @param[in] to       where
     */
    void message( const char *address, const float *values, int count, const sockaddr_in &to )
    {
        char   buf[160];
        size_t pos = 0;
        auto   putStr = [&]( const char *t )
        {
            const size_t l = std::strlen( t ) + 1;
            std::memcpy( buf + pos, t, l );
            pos += l;
            while ( pos & 3 ) buf[pos++] = 0;
        };
        putStr( address );
        char tags[20] = ",";
        for ( int i = 0; i < count && i < 16; ++i ) tags[1 + i] = 'f';
        putStr( tags );
        for ( int i = 0; i < count && i < 16; ++i )
        {   // big-endian 32-bit words
            uint32_t u;
            std::memcpy( &u, values + i, 4 );
            buf[pos++] = static_cast<char>( u >> 24 );
            buf[pos++] = static_cast<char>( u >> 16 );
            buf[pos++] = static_cast<char>( u >> 8 );
            buf[pos++] = static_cast<char>( u );
        }
        ::sendto( m_sock, buf, pos, 0, reinterpret_cast<const sockaddr *>( &to ), sizeof( to ) );
    }

    int         m_sock = -1;          ///< the UDP socket, or -1 while closed
    sockaddr_in m_hands{};            ///< the desktop's headset port
    sockaddr_in m_pose{};             ///< and the pose's
    double      m_sinceHands = 0.0;   ///< seconds since the last "/hands"
    double      m_sincePose = 0.0;    ///< seconds since the last "/pose"
};

// ---------------------------------------------------------------- the app

/** @brief One eye's swapchain and the framebuffer it is rendered through. */
struct SwapchainTarget
{
    XrSwapchain                              m_swapchain = XR_NULL_HANDLE;   ///< the eye's swapchain
    int                                      m_width = 0;                    ///< its width, pixels
    int                                      m_height = 0;                   ///< its height, pixels
    std::vector<XrSwapchainImageOpenGLESKHR> m_images;                       ///< the swapchain's images
    GLuint                                   m_fbo = 0;                      ///< the framebuffer drawn through
    GLuint                                   m_depth = 0;                    ///< its depth buffer
};

/** @brief The Quest app: the OpenXR session, the hands, the picture, and the bridge to the desktop. */
class App
{
public:
    /**
     * @brief The app on the native activity @p app.
     * @param[in,out] app  the native activity
     */
    explicit App( android_app *app ) : m_app( app ) {}

    /**
     * @brief Reads empy.cfg, opens the bridge, sets up OpenXR, EGL, the hands and the scene; false where something
     *        essential failed.
     * @return false where something essential failed
     */
    bool init()
    {
        m_dataDir = m_app->activity->externalDataPath ? m_app->activity->externalDataPath : "";
        m_config = readConfig( m_dataDir.c_str() );
        if ( !m_config.m_bridgeHost.empty() )
        {
            if ( m_bridge.open( m_config.m_bridgeHost, m_config.m_bridgePort ) )
                LOGI( "bridge: the hands to %s:%d, the pose to %d", m_config.m_bridgeHost.c_str(),
                      m_config.m_bridgePort, m_config.m_bridgePort + 100 );
            else
                LOGE( "bridge: bad host %s (a dotted IPv4 address)", m_config.m_bridgeHost.c_str() );
        }
        if ( !initLoader() ) return false;
        if ( !initInstance() ) return false;
        if ( !initEgl() ) return false;
        if ( !initSession() ) return false;
        if ( !initHands() ) LOGE( "hand tracking unavailable: only the head is sent" );
        return m_scene.init();
    }

    /** @brief The main loop: Android's events, OpenXR's events, a frame while the session runs; until the app is closed. */
    void run()
    {
        while ( !m_app->destroyRequested )
        {
            int                  events;
            android_poll_source *source;
            const int            timeout = ( m_sessionRunning || m_app->window == nullptr ) ? 0 : -1;
            while ( ALooper_pollOnce( timeout, nullptr, &events, reinterpret_cast<void **>( &source ) ) >= 0 )
            {
                if ( source ) source->process( m_app, source );
                if ( m_app->destroyRequested ) break;
            }
            pollXrEvents();
            if ( m_quit ) break;
            if ( m_sessionRunning ) frame();
        }
    }

    /** @brief Destroys what OpenXR made. */
    void shutdown()
    {
        m_bridge.close();
        for ( SwapchainTarget &t : m_targets )
            if ( t.m_swapchain != XR_NULL_HANDLE ) xrDestroySwapchain( t.m_swapchain );
        for ( int h = 0; h < 2; ++h )
            if ( m_handTracker[h] != XR_NULL_HANDLE && m_pfnDestroyHandTracker )
                m_pfnDestroyHandTracker( m_handTracker[h] );
        if ( m_viewSpace != XR_NULL_HANDLE ) xrDestroySpace( m_viewSpace );
        if ( m_stageSpace != XR_NULL_HANDLE ) xrDestroySpace( m_stageSpace );
        if ( m_session != XR_NULL_HANDLE ) xrDestroySession( m_session );
        if ( m_instance != XR_NULL_HANDLE ) xrDestroyInstance( m_instance );
    }

private:
    // ------------------------------------------------ OpenXR setup

    /**
     * @brief Initialises Android's OpenXR loader.
     * @return false when the loader fails
     */
    bool initLoader()
    {
        PFN_xrInitializeLoaderKHR initLoader = nullptr;
        xrGetInstanceProcAddr( XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                               reinterpret_cast<PFN_xrVoidFunction *>( &initLoader ) );
        if ( initLoader == nullptr )
        {
            LOGE( "no xrInitializeLoaderKHR" );
            return false;
        }
        XrLoaderInitInfoAndroidKHR li{ XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR };
        li.applicationVM = m_app->activity->vm;
        li.applicationContext = m_app->activity->clazz;
        return XR_SUCCEEDED( initLoader( reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR *>( &li ) ) );
    }

    /**
     * @brief Creates the OpenXR instance with GLES and hand tracking and finds the headset; notes whether it tracks
     *        hands.
     * @return false when there is no instance or no headset
     */
    bool initInstance()
    {
        const char *exts[] = { XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME, XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME,
                               XR_EXT_HAND_TRACKING_EXTENSION_NAME };
        XrInstanceCreateInfoAndroidKHR android{ XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR };
        android.applicationVM = m_app->activity->vm;
        android.applicationActivity = m_app->activity->clazz;
        XrInstanceCreateInfo ci{ XR_TYPE_INSTANCE_CREATE_INFO, &android };
        std::strncpy( ci.applicationInfo.applicationName, "Empyrean", XR_MAX_APPLICATION_NAME_SIZE - 1 );
        ci.applicationInfo.applicationVersion = 1;
        std::strncpy( ci.applicationInfo.engineName, "EmpyreanPresence", XR_MAX_ENGINE_NAME_SIZE - 1 );
        ci.applicationInfo.apiVersion = XR_API_VERSION_1_0;
        ci.enabledExtensionCount = 3;
        ci.enabledExtensionNames = exts;
        if ( XR_FAILED( xrCreateInstance( &ci, &m_instance ) ) )
        {
            LOGE( "xrCreateInstance failed" );
            return false;
        }
        XrSystemGetInfo sgi{ XR_TYPE_SYSTEM_GET_INFO };
        sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        if ( XR_FAILED( xrGetSystem( m_instance, &sgi, &m_system ) ) )
        {
            LOGE( "xrGetSystem failed" );
            return false;
        }
        XrSystemHandTrackingPropertiesEXT ht{ XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT };
        XrSystemProperties                sp{ XR_TYPE_SYSTEM_PROPERTIES, &ht };
        xrGetSystemProperties( m_instance, m_system, &sp );
        m_handsSupported = ht.supportsHandTracking == XR_TRUE;
        LOGI( "system: %s, hand tracking %d", sp.systemName, m_handsSupported ? 1 : 0 );
        return true;
    }

    /**
     * @brief An EGL context for GLES 3 on a small pbuffer, as OpenXR's GLES binding needs.
     * @return false when there is no context
     */
    bool initEgl()
    {
        PFN_xrGetOpenGLESGraphicsRequirementsKHR getReq = nullptr;
        xrGetInstanceProcAddr( m_instance, "xrGetOpenGLESGraphicsRequirementsKHR",
                               reinterpret_cast<PFN_xrVoidFunction *>( &getReq ) );
        XrGraphicsRequirementsOpenGLESKHR req{ XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR };
        if ( getReq == nullptr || XR_FAILED( getReq( m_instance, m_system, &req ) ) )
        {
            LOGE( "GLES requirements failed" );
            return false;
        }
        m_display = eglGetDisplay( EGL_DEFAULT_DISPLAY );
        EGLint major = 0, minor = 0;
        if ( !eglInitialize( m_display, &major, &minor ) )
        {
            LOGE( "eglInitialize failed" );
            return false;
        }
        const EGLint attribs[] = { EGL_RENDERABLE_TYPE,
                                   EGL_OPENGL_ES3_BIT,
                                   EGL_SURFACE_TYPE,
                                   EGL_PBUFFER_BIT,
                                   EGL_RED_SIZE,
                                   8,
                                   EGL_GREEN_SIZE,
                                   8,
                                   EGL_BLUE_SIZE,
                                   8,
                                   EGL_ALPHA_SIZE,
                                   8,
                                   EGL_DEPTH_SIZE,
                                   0,
                                   EGL_NONE };
        EGLint       count = 0;
        if ( !eglChooseConfig( m_display, attribs, &m_eglConfig, 1, &count ) || count == 0 )
        {
            LOGE( "eglChooseConfig failed" );
            return false;
        }
        const EGLint pbuf[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
        m_surface = eglCreatePbufferSurface( m_display, m_eglConfig, pbuf );
        const EGLint ctx[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        m_context = eglCreateContext( m_display, m_eglConfig, EGL_NO_CONTEXT, ctx );
        if ( m_context == EGL_NO_CONTEXT || !eglMakeCurrent( m_display, m_surface, m_surface, m_context ) )
        {
            LOGE( "EGL context failed" );
            return false;
        }
        LOGI( "EGL %d.%d, GL %s", major, minor, glGetString( GL_VERSION ) );
        return true;
    }

    /**
     * @brief The session, the room (stage, else local) and head spaces, and a swapchain with framebuffer and depth per
     *        eye.
     * @return false when the session or the swapchains fail
     */
    bool initSession()
    {
        XrGraphicsBindingOpenGLESAndroidKHR gb{ XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR };
        gb.display = m_display;
        gb.config = m_eglConfig;
        gb.context = m_context;
        XrSessionCreateInfo sci{ XR_TYPE_SESSION_CREATE_INFO, &gb };
        sci.systemId = m_system;
        if ( XR_FAILED( xrCreateSession( m_instance, &sci, &m_session ) ) )
        {
            LOGE( "xrCreateSession failed" );
            return false;
        }
        XrReferenceSpaceCreateInfo rs{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        rs.poseInReferenceSpace.orientation.w = 1.0f;
        rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        if ( XR_FAILED( xrCreateReferenceSpace( m_session, &rs, &m_stageSpace ) ) )
        {
            rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            if ( XR_FAILED( xrCreateReferenceSpace( m_session, &rs, &m_stageSpace ) ) )
            {
                LOGE( "no reference space" );
                return false;
            }
        }
        rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        xrCreateReferenceSpace( m_session, &rs, &m_viewSpace );

        uint32_t viewCount = 0;
        xrEnumerateViewConfigurationViews( m_instance, m_system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0,
                                           &viewCount, nullptr );
        std::vector<XrViewConfigurationView> cfg( viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW } );
        xrEnumerateViewConfigurationViews( m_instance, m_system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount,
                                           &viewCount, cfg.data() );
        m_views.assign( viewCount, { XR_TYPE_VIEW } );

        uint32_t fmtCount = 0;
        xrEnumerateSwapchainFormats( m_session, 0, &fmtCount, nullptr );
        std::vector<int64_t> formats( fmtCount );
        xrEnumerateSwapchainFormats( m_session, fmtCount, &fmtCount, formats.data() );
        int64_t format = formats.empty() ? GL_RGBA8 : formats[0];
        for ( int64_t f : formats )
            if ( f == GL_SRGB8_ALPHA8 )
            {
                format = f;
                break;
            }
        m_targets.resize( viewCount );
        for ( uint32_t v = 0; v < viewCount; ++v )
        {
            SwapchainTarget      &t = m_targets[v];
            XrSwapchainCreateInfo sc{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
            sc.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
            sc.format = format;
            sc.sampleCount = 1;
            sc.width = cfg[v].recommendedImageRectWidth;
            sc.height = cfg[v].recommendedImageRectHeight;
            sc.faceCount = 1;
            sc.arraySize = 1;
            sc.mipCount = 1;
            if ( XR_FAILED( xrCreateSwapchain( m_session, &sc, &t.m_swapchain ) ) )
            {
                LOGE( "xrCreateSwapchain failed" );
                return false;
            }
            t.m_width = static_cast<int>( sc.width );
            t.m_height = static_cast<int>( sc.height );
            uint32_t imgCount = 0;
            xrEnumerateSwapchainImages( t.m_swapchain, 0, &imgCount, nullptr );
            t.m_images.assign( imgCount, { XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR } );
            xrEnumerateSwapchainImages( t.m_swapchain, imgCount, &imgCount,
                                        reinterpret_cast<XrSwapchainImageBaseHeader *>( t.m_images.data() ) );
            glGenFramebuffers( 1, &t.m_fbo );
            glGenRenderbuffers( 1, &t.m_depth );
            glBindRenderbuffer( GL_RENDERBUFFER, t.m_depth );
            glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, t.m_width, t.m_height );
            LOGI( "view %u: %dx%d, %u images", v, t.m_width, t.m_height, imgCount );
        }
        return true;
    }

    /**
     * @brief A hand tracker per hand; false where the headset or runtime does not track hands.
     * @return false where the hands are not tracked
     */
    bool initHands()
    {
        if ( !m_handsSupported ) return false;
        xrGetInstanceProcAddr( m_instance, "xrCreateHandTrackerEXT",
                               reinterpret_cast<PFN_xrVoidFunction *>( &m_pfnCreateHandTracker ) );
        xrGetInstanceProcAddr( m_instance, "xrLocateHandJointsEXT",
                               reinterpret_cast<PFN_xrVoidFunction *>( &m_pfnLocateHandJoints ) );
        xrGetInstanceProcAddr( m_instance, "xrDestroyHandTrackerEXT",
                               reinterpret_cast<PFN_xrVoidFunction *>( &m_pfnDestroyHandTracker ) );
        if ( !m_pfnCreateHandTracker || !m_pfnLocateHandJoints ) return false;
        for ( int h = 0; h < 2; ++h )
        {
            XrHandTrackerCreateInfoEXT hci{ XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT };
            hci.hand = ( h == 0 ) ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
            hci.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
            if ( XR_FAILED( m_pfnCreateHandTracker( m_session, &hci, &m_handTracker[h] ) ) ) return false;
        }
        return true;
    }

    /**
     * @brief OpenXR's events: begins and ends the session as its state asks, notes when the app is to quit, and
     *        recentres the room's forward when the headset is recentred.
     */
    void pollXrEvents()
    {
        XrEventDataBuffer ev{ XR_TYPE_EVENT_DATA_BUFFER };
        while ( xrPollEvent( m_instance, &ev ) == XR_SUCCESS )
        {
            if ( ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED )
            {
                const auto *sc = reinterpret_cast<const XrEventDataSessionStateChanged *>( &ev );
                switch ( sc->state )
                {
                case XR_SESSION_STATE_READY:
                {
                    XrSessionBeginInfo bi{ XR_TYPE_SESSION_BEGIN_INFO };
                    bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    if ( XR_SUCCEEDED( xrBeginSession( m_session, &bi ) ) ) m_sessionRunning = true;
                    break;
                }
                case XR_SESSION_STATE_STOPPING:
                    xrEndSession( m_session );
                    m_sessionRunning = false;
                    break;
                case XR_SESSION_STATE_EXITING:
                case XR_SESSION_STATE_LOSS_PENDING: m_quit = true; break;
                default: break;
                }
            }
            else if ( ev.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING )
            {
                m_centred = false;   // the headset recentred: the room's forward is taken again
            }
            else if ( ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING )
            {
                m_quit = true;
            }
            ev = { XR_TYPE_EVENT_DATA_BUFFER };
        }
    }

    // ------------------------------------------------ per frame

    /**
     * @brief The head pose at @p time (m_headPose, m_headValid); the room's forward taken from the first one.
     * @param[in] time  the predicted display time
     */
    void updateHead( XrTime time )
    {
        m_headValid = false;
        if ( m_viewSpace == XR_NULL_HANDLE ) return;
        XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
        if ( XR_FAILED( xrLocateSpace( m_viewSpace, m_stageSpace, time, &loc ) ) ) return;
        const XrSpaceLocationFlags need =
            XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
        if ( ( loc.locationFlags & need ) != need ) return;
        m_headPose = loc.pose;
        m_headValid = true;
        if ( !m_centred )
        {
            m_yaw0 = yawOf( m_headPose.orientation );
            m_centre = { m_headPose.position.x, m_headPose.position.y, m_headPose.position.z };
            m_centred = true;
        }
    }

    /**
     * @brief Locates both hands at @p time and feeds their joints and pinches to m_hands.
     * @param[in] time  the predicted display time
     */
    void updateHands( XrTime time )
    {
        for ( int h = 0; h < 2; ++h )
        {
            if ( m_handTracker[h] == XR_NULL_HANDLE )
            {
                m_hands.lost( h );
                continue;
            }
            XrHandJointLocationEXT  joints[XR_HAND_JOINT_COUNT_EXT];
            XrHandJointLocationsEXT locs{ XR_TYPE_HAND_JOINT_LOCATIONS_EXT };
            locs.jointCount = XR_HAND_JOINT_COUNT_EXT;
            locs.jointLocations = joints;
            XrHandJointsLocateInfoEXT li{ XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT };
            li.baseSpace = m_stageSpace;
            li.time = time;
            if ( XR_FAILED( m_pfnLocateHandJoints( m_handTracker[h], &li, &locs ) ) || !locs.isActive )
            {
                m_hands.lost( h );
                continue;
            }
            const XrHandJointLocationEXT &palm = joints[XR_HAND_JOINT_PALM_EXT];
            const XrHandJointLocationEXT &thumb = joints[XR_HAND_JOINT_THUMB_TIP_EXT];
            const XrHandJointLocationEXT &index = joints[XR_HAND_JOINT_INDEX_TIP_EXT];
            const XrSpaceLocationFlags    need =
                XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
            if ( ( palm.locationFlags & need ) != need )
            {
                m_hands.lost( h );
                continue;
            }
            const float dx = thumb.pose.position.x - index.pose.position.x;
            const float dy = thumb.pose.position.y - index.pose.position.y;
            const float dz = thumb.pose.position.z - index.pose.position.z;
            m_hands.setHand( h, joints, std::sqrt( dx * dx + dy * dy + dz * dz ),
                             m_headValid ? m_headPose.position.y : 1.6f );
        }
    }

    /**
     * @brief The pose as "/pose" sends it: the head's yaw against the room's forward and its pitch, then each hand from
     *        the head in the room's axes (x right, y up, z ahead), the pinches and whether each is seen.
     * @param[out] out  receives twelve floats
     */
    void poseOut( float *out ) const
    {
        const Vec3  f = rotate( m_headPose.orientation, { 0.0f, 0.0f, -1.0f } );
        float       yaw = yawOf( m_headPose.orientation ) - m_yaw0;
        const float pi = 3.14159265f;
        while ( yaw > pi ) yaw -= 2.0f * pi;
        while ( yaw < -pi ) yaw += 2.0f * pi;
        out[0] = m_headValid ? yaw : 0.0f;
        out[1] = m_headValid ? std::asin( std::max( -1.0f, std::min( 1.0f, f.m_y ) ) ) : 0.0f;
        const float s = std::sin( m_yaw0 ), c = std::cos( m_yaw0 );
        const Vec3  right{ c, 0.0f, s }, ahead{ s, 0.0f, -c };   // the room's axes as they stood at the start
        for ( int h = 0; h < 2; ++h )
        {
            const Hands::Hand &hand = m_hands.hand( h );
            const float        dx = hand.m_palm.x - m_headPose.position.x, dy = hand.m_palm.y - m_headPose.position.y,
                               dz = hand.m_palm.z - m_headPose.position.z;
            out[2 + 3 * h] = hand.m_valid ? dx * right.m_x + dz * right.m_z : 0.0f;
            out[3 + 3 * h] = hand.m_valid ? dy : 0.0f;
            out[4 + 3 * h] = hand.m_valid ? dx * ahead.m_x + dz * ahead.m_z : 0.0f;
            out[8 + h] = hand.m_valid ? hand.m_pinch : 0.0f;
            out[10 + h] = hand.m_valid && m_headValid ? 1.0f : 0.0f;
        }
    }

    /** @brief Pitch of one font pixel, metres (a glyph about 1.7 degrees tall at the panel's distance). */
    static constexpr float Cell = 0.0042f;
    /** @brief Distance between text rows, metres. */
    static constexpr float Row = 0.040f;
    /** @brief How far in front of the eyes the panel sits, metres. */
    static constexpr float PanelDistance = 1.0f;

    /**
     * @brief The panel, low in front of the eyes, following the head's turn only.
     * @return the panel
     */
    Panel headPanel() const
    {
        Panel       p;
        const Vec3  fwd = rotate( m_headPose.orientation, { 0.0f, 0.0f, -1.0f } );
        const float len = std::fmax( std::sqrt( fwd.m_x * fwd.m_x + fwd.m_z * fwd.m_z ), 1.0e-3f );
        const Vec3  f{ fwd.m_x / len, 0.0f, fwd.m_z / len };
        p.m_right = { -f.m_z, 0.0f, f.m_x };
        p.m_up = { 0.0f, 1.0f, 0.0f };
        const float half = 0.5f * 18.0f * 6.0f * Cell;
        p.m_origin = { m_headPose.position.x + f.m_x * PanelDistance - p.m_right.m_x * half,
                       m_headPose.position.y - 0.25f,
                       m_headPose.position.z + f.m_z * PanelDistance - p.m_right.m_z * half };
        return p;
    }

    /**
     * @brief The picture: the sky turning slowly round the room's centre (a turn in ten minutes, each point twinkling
     *        by its own sine), the ring at arm's reach a little below the head where the hands act, the hands' joints
     *        (warmer as they pinch), and the panel.
     * @param[in] t  seconds since the app began
     */
    void buildScene( double t )
    {
        constexpr int Stars = 900;
        const float   turn = static_cast<float>( t * 6.2831853 / 600.0 );
        for ( int i = 0; i < Stars; ++i )
        {
            const uint32_t u = static_cast<uint32_t>( i );
            const float    z = 2.0f * uniform( u, 1 ) - 1.0f, a = 6.2831853f * uniform( u, 2 ) + turn;
            const float    r = std::sqrt( std::max( 0.0f, 1.0f - z * z ) ) * 8.0f;
            const float    twinkle =
                0.55f + 0.45f * std::sin( static_cast<float>( t ) * ( 0.2f + 0.6f * uniform( u, 3 ) ) +
                                          6.2831853f * uniform( u, 4 ) );
            const float warm = uniform( u, 5 );
            m_scene.add( m_centre.m_x + r * std::cos( a ), m_centre.m_y + 8.0f * std::fabs( z ) * 0.8f - 1.0f,
                         m_centre.m_z + r * std::sin( a ), 0.55f + 0.3f * warm, 0.65f + 0.15f * warm, 1.0f,
                         0.35f * twinkle, 20.0f + 30.0f * uniform( u, 6 ) );
        }
        if ( m_headValid )
        {   // the reach: an arm's length round the head, at the height of the hands' middle
            for ( int k = 0; k < 96; ++k )
            {
                const float a = 6.2831853f * static_cast<float>( k ) / 96.0f;
                m_scene.add( m_headPose.position.x + 0.8f * std::cos( a ), m_headPose.position.y - 0.4f,
                             m_headPose.position.z + 0.8f * std::sin( a ), 0.45f, 0.8f, 0.9f, 0.18f, 6.0f );
            }
        }
        for ( int h = 0; h < 2; ++h )
        {
            const Hands::Hand &hand = m_hands.hand( h );
            if ( !hand.m_valid ) continue;
            for ( int j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j )
                m_scene.add( hand.m_joints[j].x, hand.m_joints[j].y, hand.m_joints[j].z, 0.6f + 0.4f * hand.m_pinch,
                             0.85f, 1.0f - 0.5f * hand.m_pinch, 0.7f, 9.0f );
        }
        const Panel p = headPanel();
        m_scene.addText( p, 0.0f, 0.0f, Cell, "EMPYREAN", 0.6f, 0.85f, 1.0f, 0.9f );
        if ( m_bridge.ok() )
        {
            char line[40];
            std::snprintf( line, sizeof( line ), "TO %s", m_config.m_bridgeHost.c_str() );
            m_scene.addText( p, 0.0f, -Row, Cell, line, 0.5f, 0.75f, 0.9f, 0.7f );
            std::snprintf( line, sizeof( line ), "HANDS %d POSE %d", m_config.m_bridgePort,
                           m_config.m_bridgePort + 100 );
            m_scene.addText( p, 0.0f, -2.0f * Row, Cell, line, 0.5f, 0.75f, 0.9f, 0.7f );
        }
        else
        {
            m_scene.addText( p, 0.0f, -Row, Cell, "NO BRIDGE: EMPY.CFG", 1.0f, 0.6f, 0.4f, 0.8f );
        }
    }

    /** @brief One frame: wait, the head and the hands, the bridge, the picture for both eyes, submit. */
    void frame()
    {
        XrFrameWaitInfo wi{ XR_TYPE_FRAME_WAIT_INFO };
        XrFrameState    fs{ XR_TYPE_FRAME_STATE };
        if ( XR_FAILED( xrWaitFrame( m_session, &wi, &fs ) ) ) return;
        XrFrameBeginInfo bi{ XR_TYPE_FRAME_BEGIN_INFO };
        xrBeginFrame( m_session, &bi );

        const double dt =
            m_lastTime == 0 ? 1.0 / 72.0 : static_cast<double>( fs.predictedDisplayTime - m_lastTime ) * 1.0e-9;
        m_lastTime = fs.predictedDisplayTime;
        m_seconds += std::min( dt, 0.1 );
        updateHead( fs.predictedDisplayTime );
        updateHands( fs.predictedDisplayTime );
        float pose[12];
        poseOut( pose );
        m_bridge.send( m_hands, pose, dt );

        std::vector<XrCompositionLayerProjectionView> projViews;
        XrCompositionLayerProjection                  layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
        const XrCompositionLayerBaseHeader *layers[1] = { reinterpret_cast<XrCompositionLayerBaseHeader *>( &layer ) };
        uint32_t                            layerCount = 0;
        if ( fs.shouldRender )
        {
            XrViewLocateInfo vli{ XR_TYPE_VIEW_LOCATE_INFO };
            vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            vli.displayTime = fs.predictedDisplayTime;
            vli.space = m_stageSpace;
            XrViewState vs{ XR_TYPE_VIEW_STATE };
            uint32_t    viewCount = 0;
            xrLocateViews( m_session, &vli, &vs, static_cast<uint32_t>( m_views.size() ), &viewCount, m_views.data() );
            if ( ( vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT ) &&
                 ( vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT ) )
            {
                m_scene.begin();
                buildScene( m_seconds );
                m_scene.upload();
                projViews.resize( viewCount, { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW } );
                for ( uint32_t v = 0; v < viewCount; ++v )
                {
                    SwapchainTarget            &t = m_targets[v];
                    XrSwapchainImageAcquireInfo ai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
                    uint32_t                    index = 0;
                    xrAcquireSwapchainImage( t.m_swapchain, &ai, &index );
                    XrSwapchainImageWaitInfo swi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
                    swi.timeout = XR_INFINITE_DURATION;
                    xrWaitSwapchainImage( t.m_swapchain, &swi );
                    glBindFramebuffer( GL_FRAMEBUFFER, t.m_fbo );
                    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                            t.m_images[index].image, 0 );
                    glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, t.m_depth );
                    m_scene.draw(
                        multiply( projectionFromFov( m_views[v].fov, 0.05f, 100.0f ), viewFromPose( m_views[v].pose ) ),
                        t.m_width, t.m_height );
                    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
                    XrSwapchainImageReleaseInfo ri{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
                    xrReleaseSwapchainImage( t.m_swapchain, &ri );
                    projViews[v].pose = m_views[v].pose;
                    projViews[v].fov = m_views[v].fov;
                    projViews[v].subImage.swapchain = t.m_swapchain;
                    projViews[v].subImage.imageRect = { { 0, 0 }, { t.m_width, t.m_height } };
                    projViews[v].subImage.imageArrayIndex = 0;
                }
                layer.space = m_stageSpace;
                layer.viewCount = viewCount;
                layer.views = projViews.data();
                layerCount = 1;
            }
        }
        XrFrameEndInfo ei{ XR_TYPE_FRAME_END_INFO };
        ei.displayTime = fs.predictedDisplayTime;
        ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        ei.layerCount = layerCount;
        ei.layers = layers;
        xrEndFrame( m_session, &ei );
    }

    android_app *m_app;                          ///< the native activity
    std::string  m_dataDir;                      ///< the app's external data folder (empy.cfg)
    Config       m_config;                       ///< what empy.cfg says
    Scene        m_scene;                        ///< what is drawn
    Hands        m_hands;                        ///< the hands as tracked
    Bridge       m_bridge;                       ///< the hands and the pose to the desktop's Empyrean
    bool         m_centred = false;              ///< the room's forward has been taken
    float        m_yaw0 = 0.0f;                  ///< the head's turn when it was taken: the room's forward
    Vec3         m_centre{ 0.0f, 1.6f, 0.0f };   ///< the head when it was taken: the sky's centre
    double       m_seconds = 0.0;                ///< seconds the frames have run

    EGLDisplay m_display = EGL_NO_DISPLAY;   ///< the EGL display
    EGLConfig  m_eglConfig = nullptr;        ///< the EGL configuration
    EGLSurface m_surface = EGL_NO_SURFACE;   ///< the small pbuffer surface the context is current on
    EGLContext m_context = EGL_NO_CONTEXT;   ///< the GLES context

    XrInstance                   m_instance = XR_NULL_HANDLE;     ///< the OpenXR instance
    XrSystemId                   m_system = XR_NULL_SYSTEM_ID;    ///< the headset
    XrSession                    m_session = XR_NULL_HANDLE;      ///< the OpenXR session
    XrSpace                      m_stageSpace = XR_NULL_HANDLE;   ///< the room: stage, else local
    XrSpace                      m_viewSpace = XR_NULL_HANDLE;    ///< the head
    bool                         m_sessionRunning = false;        ///< the session runs: frames are drawn
    bool                         m_quit = false;                  ///< the runtime asks the app to end
    bool                         m_handsSupported = false;        ///< the headset tracks hands
    bool                         m_headValid = false;             ///< m_headPose is valid this frame
    std::vector<XrView>          m_views;                         ///< the eyes' poses and fields of view this frame
    std::vector<SwapchainTarget> m_targets;                       ///< per eye: its swapchain and framebuffer
    XrPosef                      m_headPose{ { 0, 0, 0, 1 }, { 0, 0, 0 } };   ///< the head in the room
    XrTime                       m_lastTime = 0;   ///< the last frame's display time (0 before the first)

    PFN_xrCreateHandTrackerEXT  m_pfnCreateHandTracker = nullptr;                        ///< xrCreateHandTrackerEXT
    PFN_xrLocateHandJointsEXT   m_pfnLocateHandJoints = nullptr;                         ///< xrLocateHandJointsEXT
    PFN_xrDestroyHandTrackerEXT m_pfnDestroyHandTracker = nullptr;                       ///< xrDestroyHandTrackerEXT
    XrHandTrackerEXT            m_handTracker[2] = { XR_NULL_HANDLE, XR_NULL_HANDLE };   ///< left, right
};

/** @brief The native app glue's command callback: nothing to do, OpenXR's events drive the app. */
void handleCmd( android_app *, int32_t ) {}

}   // namespace

/**
 * @brief NativeActivity entry point (android_native_app_glue).
 * @param[in,out] app  the native activity
 */
void android_main( android_app *app )
{
    app->onAppCmd = handleCmd;
    JNIEnv *env = nullptr;
    app->activity->vm->AttachCurrentThread( &env, nullptr );
    {
        auto a = std::make_unique<App>( app );
        if ( a->init() )
            a->run();
        else
            LOGE( "init failed" );
        a->shutdown();
    }
    app->activity->vm->DetachCurrentThread();
}
