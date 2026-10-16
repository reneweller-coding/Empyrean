/**
 * @file Life.cu
 * @brief The living field under the partials: Lenia, the Turing field and the animated pictures, the Fluid (a lattice
 *        Boltzmann flow with ink), the Islands (Cahn-Hilliard), and the field's statistics. The update kernel
 *        (Voices.cu) reads a cell where a partial is.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

/** @brief Lenia's ring kernel, normalised, set once at open(). */
__constant__ float cLeniaKernel[( 2 * LifeRadius + 1 ) * ( 2 * LifeRadius + 1 )];

// ---- Animated pictures (KaleidoscopeEnhanced's Scene2D/Chain*.frag, its continuous transforms) ----

/**
 * @brief Smooth value noise, 0 .. 1 (corners hashed).
 * @param[in] x  across
 * @param[in] y  along
 * @return the noise
 */
__device__ float pictureNoise( float x, float y )
{
    const float ix = floorf( x ), iy = floorf( y ), fx = x - ix, fy = y - iy;
    const float sx = fx * fx * ( 3.0f - 2.0f * fx ), sy = fy * fy * ( 3.0f - 2.0f * fy );
    auto        h = []( float a, float b )
    {
        return static_cast<float>( hash32( static_cast<std::uint32_t>( static_cast<int>( a ) ) * 0x8DA6B343U ^
                                           static_cast<std::uint32_t>( static_cast<int>( b ) ) * 0xD8163841U ) >>
                                   8 ) *
               ( 1.0f / 16777216.0f );
    };
    const float a = h( ix, iy ), b = h( ix + 1.0f, iy ), c = h( ix, iy + 1.0f ), d = h( ix + 1.0f, iy + 1.0f );
    return a + ( b - a ) * sx + ( c - a ) * sy + ( a - b - c + d ) * sx * sy;
}

/**
 * @brief Three octaves of noise, 0 .. 1 (the shaders' fbm3).
 * @param[in] x  across
 * @param[in] y  along
 * @return the noise
 */
__device__ float pictureFbm( float x, float y )
{
    float v = 0.0f, a = 0.5f;
    for ( int i = 0; i < 3; ++i )
    {
        v += a * pictureNoise( x, y );
        const float nx = 1.6f * x + 1.2f * y + 2.3f, ny = -1.2f * x + 1.6f * y + 2.3f;
        x = nx;
        y = ny;
        a *= 0.5f;
    }
    return v / 0.875f;
}

/**
 * @brief The shaders' mirror repeat: identity on [0, 1], mirrored beyond, so every fold is seamless.
 * @param[in] u  a coordinate
 * @return 0 .. 1
 */
__device__ __forceinline__ float pictureMirror( float u )
{
    const float f = u * 0.5f - floorf( u * 0.5f );
    return 1.0f - fabsf( f * 2.0f - 1.0f );
}

/**
 * @brief The landscape as a picture at (u, v), mirror-repeated: u its time (left to right), v its
 *        frequency (low at the bottom); its brightness on the landscapes' 40 dB scale.
 * @param[in] hdr   the header (the landscape's size)
 * @param[in] land  the landscape
 * @param[in] u     across
 * @param[in] v     up
 * @return 0 .. 1
 */
__device__ float pictureAt( const BlockHeader *hdr, const float *land, float u, float v )
{
    const int cols = hdr->m_landColumns, rows = hdr->m_landRows;
    if ( cols < 2 || rows < 2 ) return 0.5f;
    const float  fr = pictureMirror( u ) * static_cast<float>( rows - 1 ),
                 fc = pictureMirror( v ) * static_cast<float>( cols - 1 );
    const int    r0 = min( static_cast<int>( fr ), rows - 2 ), c0 = min( static_cast<int>( fc ), cols - 2 );
    const float  ar = fr - static_cast<float>( r0 ), ac = fc - static_cast<float>( c0 );
    const float *p = land + static_cast<size_t>( r0 ) * cols + c0;
    const float  top = p[0] + ( p[1] - p[0] ) * ac, bottom = p[cols] + ( p[cols + 1] - p[cols] ) * ac;
    const float  a = top + ( bottom - top ) * ar;
    return fminf( fmaxf( 1.0f + 20.0f * log10f( fmaxf( a, 1.0e-6f ) ) / 40.0f, 0.0f ), 1.0f );
}

/**
 * @brief One cell of an animated picture. The field is the screen (twice as wide as high); the
 *        transforms are the shaders' (tKaleido, tTunnel, tSpiral, tWarp), the feedback and the
 *        rain read the field before this step.
 * @param[in] hdr   the header (its life: time, angle, sides, character)
 * @param[in] land  the landscape
 * @param[in] src   the field before this step
 * @param[in] x     the cell across
 * @param[in] y     the cell along
 * @return the cell (x: the value, or Rain's height; y: Rain's height before)
 */
__device__ float2 pictureCell( const BlockHeader *hdr, const float *land, const float2 *src, int x, int y )
{
    const LifeField &L = hdr->m_life;
    const float      px = ( static_cast<float>( x ) + 0.5f ) / static_cast<float>( LifeW ),
                     py = ( static_cast<float>( y ) + 0.5f ) / static_cast<float>( LifeH );
    const float      dx = ( px - 0.5f ) * 2.0f, dy = py - 0.5f;   // centred, the aspect kept
    const float      r = fmaxf( sqrtf( dx * dx + dy * dy ), 1.0e-4f ), a = atan2f( dy, dx );
    const float      T = L.m_time, ch = L.m_character, S = fmaxf( L.m_sides, 2.0f );
    const float      Pi = 3.14159265f;
    switch ( L.m_mode )
    {
    case LifeModeE::KALEIDOSCOPE:
    {
        // tKaleido round a centre that wanders slowly through the picture, the picture flowing under it.
        const float cx = 0.5f + 0.2f * __sinf( 0.0107f * L.m_seconds ),
                    cy = 0.5f + 0.2f * __cosf( 0.0131f * L.m_seconds );
        const float sec = 2.0f * Pi / S;
        const float w = fabsf( a - sec * floorf( a / sec ) - 0.5f * sec ) + L.m_angle;
        float       u = cx + 0.6f * r * __cosf( w ) + 0.05f * T, v = cy + 0.6f * r * __sinf( w );
        u += ch * 0.15f * ( pictureFbm( 3.0f * u + T, 3.0f * v ) - 0.5f );
        v += ch * 0.15f * ( pictureFbm( 3.0f * u + 5.2f, 3.0f * v - T ) - 0.5f );
        return make_float2( pictureAt( hdr, land, u, v ), 0.0f );
    }
    case LifeModeE::TUNNEL:
    {
        // tTunnel: the angle round (folded into Sides mirror wedges), the depth 1/r along, flying in.
        const float sec = 2.0f * Pi / S;
        const float w = fabsf( a + L.m_angle - sec * floorf( ( a + L.m_angle ) / sec ) - 0.5f * sec ) / sec;
        const float depth = 0.25f / r + 0.3f * T;
        return make_float2( pictureAt( hdr, land, w + ch * 0.3f * depth, depth ), 0.0f );
    }
    case LifeModeE::SPIRAL:
    {
        // tSpiral: log-polar, the angle sheared into the radius -- an endless zoom along Sides arms.
        const float arms = 2.0f * floorf( S * 0.5f );
        const float u = logf( r ) * ( 0.5f + 0.5f * ch ) - 0.3f * T + ( a + L.m_angle ) / Pi;
        const float v = ( a + L.m_angle ) / Pi * arms * 0.5f;
        return make_float2( pictureAt( hdr, land, u, v ), 0.0f );
    }
    case LifeModeE::FLOW:
    {
        // tWarp, twice: the picture carried on a current of noise that itself drifts.
        float       u = px + 0.05f * T * __cosf( L.m_angle ), v = py + 0.05f * T * __sinf( L.m_angle );
        const float s = 0.15f + 0.45f * ch;
        for ( int k = 0; k < 2; ++k )
        {
            const float wu = pictureFbm( 2.5f * u + 0.3f * T, 2.5f * v ) - 0.5f,
                        wv = pictureFbm( 2.5f * u + 5.2f, 2.5f * v - 0.3f * T ) - 0.5f;
            u += s * wu;
            v += s * wv;
        }
        return make_float2( pictureAt( hdr, land, u, v ), 0.0f );
    }
    case LifeModeE::FEEDBACK:
    {
        // Video feedback: this cell takes the field from a point turned and drawn towards the middle
        // (so everything spirals outward), fading a little, and the picture seeps in.
        // A step every block (375 a second): it zooms out by 0.15 .. 0.6 a second and forgets in about three.
        const float step = 0.0004f * ( 1.0f + 3.0f * ch ), rot = L.m_turnStep;
        const float cs = __cosf( rot ), sn = __sinf( rot );
        const float qx = ( dx * cs - dy * sn ) / ( 1.0f + step ), qy = ( dx * sn + dy * cs ) / ( 1.0f + step );
        const float fx = ( qx * 0.5f + 0.5f ) * static_cast<float>( LifeW ) - 0.5f,
                    fy = ( qy + 0.5f ) * static_cast<float>( LifeH ) - 0.5f;
        const int   x0 = static_cast<int>( floorf( fx ) ), y0 = static_cast<int>( floorf( fy ) );
        const float ax = fx - static_cast<float>( x0 ), ay = fy - static_cast<float>( y0 );
        auto at = [&]( int xx, int yy ) { return src[( yy & ( LifeH - 1 ) ) * LifeW + ( xx & ( LifeW - 1 ) )].x; };
        const float back = ( at( x0, y0 ) * ( 1.0f - ax ) + at( x0 + 1, y0 ) * ax ) * ( 1.0f - ay ) +
                           ( at( x0, y0 + 1 ) * ( 1.0f - ax ) + at( x0 + 1, y0 + 1 ) * ax ) * ay;
        const float seep = pictureAt( hdr, land, px + 0.02f * T, py + 0.5f * __sinf( 0.01f * L.m_seconds ) );
        return make_float2( fminf( fmaxf( 0.997f * back + 0.003f * seep, 0.0f ), 1.0f ), 0.0f );
    }
    case LifeModeE::RAIN:
    {
        // A damped 2D wave (leapfrog, the five-point Laplacian), struck at chance cells.
        const float2 c = src[y * LifeW + x];
        const float  l = src[y * LifeW + ( ( x - 1 ) & ( LifeW - 1 ) )].x,
                     rr = src[y * LifeW + ( ( x + 1 ) & ( LifeW - 1 ) )].x;
        const float  u = src[( ( y - 1 ) & ( LifeH - 1 ) ) * LifeW + x].x,
                     d = src[( ( y + 1 ) & ( LifeH - 1 ) ) * LifeW + x].x;
        // A step every block: the rings cross the field in about three seconds; Character keeps them ringing.
        const float damp = 0.0005f + 0.004f * ( 1.0f - ch );
        float next = ( 2.0f * c.x - ( 1.0f - damp ) * c.y + 0.05f * ( l + rr + u + d - 4.0f * c.x ) ) / ( 1.0f + damp );
        const float roll = static_cast<float>( hash32( static_cast<std::uint32_t>( y * LifeW + x ) * 0x9E3779B9U ^
                                                       hdr->m_blockIndex * 0x85EBCA6BU ) >>
                                               8 ) *
                           ( 1.0f / 16777216.0f );
        if ( roll < L.m_drops ) next += 1.5f;
        return make_float2( fminf( fmaxf( next, -4.0f ), 4.0f ), c.x );
    }
    case LifeModeE::STARFIELD:
    {
        // Stars on a log-polar grid streaming outwards; three layers of different density.
        float v = 0.0f;
        for ( int k = 0; k < 3; ++k )
        {
            const float         density = 6.0f * S * ( 1.0f + static_cast<float>( k ) );
            const float         gu = ( a + L.m_angle * ( 1.0f + 0.3f * k ) ) / ( 2.0f * Pi ) * density,
                                gv = ( logf( r ) - T * ( 0.6f + 0.4f * k ) ) * density * 0.25f;
            const float         cu = floorf( gu ), cv = floorf( gv );
            const std::uint32_t hs =
                hash32( static_cast<std::uint32_t>( static_cast<int>( cu ) ) * 0x27D4EB2DU ^
                        static_cast<std::uint32_t>( static_cast<int>( cv ) ) * 0x165667B1U ^ ( k * 0x9E3779B9U ) );
            if ( ( hs & 255U ) < static_cast<std::uint32_t>( 40 + 60 * ch ) )
            {
                const float su = gu - cu - ( 0.2f + 0.6f * static_cast<float>( ( hs >> 8 ) & 255U ) / 255.0f ),
                            sv = gv - cv - ( 0.2f + 0.6f * static_cast<float>( ( hs >> 16 ) & 255U ) / 255.0f );
                v = fmaxf( v, __expf( -( su * su + sv * sv ) * 25.0f ) * fminf( r * 3.0f, 1.0f ) );
            }
        }
        // A faint glow under the stars: a sky of nothing but points is so sparse that no make-up can
        // keep its level (measured 2 dB under), and the partials between them would all but vanish.
        return make_float2( 0.12f + 0.88f * v, 0.0f );
    }
    case LifeModeE::MOIRE:
    {
        // Two line gratings, nearly alike and turning nearly together: where they agree and where they
        // cancel makes broad bands that wander slowly (the moire), far wider than the lines.
        const float f1 = 2.0f * S, f2 = f1 * ( 1.0f + 0.02f + 0.08f * ch );
        const float a1 = L.m_angle, a2 = 0.9f * L.m_angle + 0.08f;
        const float g1 = 0.5f + 0.5f * __cosf( 2.0f * Pi * f1 * ( dx * __cosf( a1 ) + dy * __sinf( a1 ) ) + T );
        const float g2 = 0.5f + 0.5f * __cosf( 2.0f * Pi * f2 * ( dx * __cosf( a2 ) + dy * __sinf( a2 ) ) - 0.7f * T );
        return make_float2( g1 * g2, 0.0f );
    }
    default: return make_float2( 0.0f, 0.0f );
    }
}

}   // namespace

/**
 * @brief Seeds the living field when asked (hdr->life.reseed): Lenia a few patches of noise, the
 *        Turing field U = 1, V = 0 with a few squares of V -- from a hash of the cell and the salt.
 * @param[in]     hdr  the header
 * @param[in,out] a    buffer A
 * @param[in,out] b    buffer B (the one before the steps is seeded)
 */
__global__ void lifeSeedKernel( const BlockHeader *hdr, float2 *a, float2 *b )
{
    const LifeField &L = hdr->m_life;
    if ( !L.m_reseed || L.m_mode == LifeModeE::OFF ) return;
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    float2             *dst = ( hdr->m_lifeParity & 1 ) ? b : a;
    const int           x = i & ( LifeW - 1 ), y = i >> 8;
    const std::uint32_t salt = static_cast<std::uint32_t>( L.m_salt ) * 0x9E3779B9U;
    float2              c = L.m_mode == LifeModeE::TURING ? make_float2( 1.0f, 0.0f ) : make_float2( 0.0f, 0.0f );
    if ( lifeIsPicture( L.m_mode ) )
    {
        dst[i] = c;
        return;
    }   // the pictures are computed, Rain starts from still water
    // Turing dense (eighty small squares: a sparse field is uneven where the partials read it --
    // measured 3 dB over while it grew from fourteen), Lenia a few large patches of noise.
    const int patches = L.m_mode == LifeModeE::TURING ? 80 : 8;
    for ( int k = 0; k < patches; ++k )
    {
        const std::uint32_t h = hash32( salt + 7919U * static_cast<std::uint32_t>( k ) );
        const int cx = static_cast<int>( h & ( LifeW - 1 ) ), cy = static_cast<int>( ( h >> 8 ) & ( LifeH - 1 ) );
        const int r = L.m_mode == LifeModeE::TURING ? 3 + static_cast<int>( ( h >> 16 ) % 5U )
                                                    : 10 + static_cast<int>( ( h >> 16 ) % 9U );
        int       dx = abs( x - cx ), dy = abs( y - cy );
        dx = min( dx, LifeW - dx );
        dy = min( dy, LifeH - dy );
        if ( dx > r || dy > r ) continue;
        const float noise = static_cast<float>( hash32( salt ^ static_cast<std::uint32_t>( i ) * 2654435761U ) >> 8 ) *
                            ( 1.0f / 16777216.0f );
        if ( L.m_mode == LifeModeE::TURING )
            c = make_float2( 0.5f, 0.25f + 0.05f * noise );
        else
            c.x = fmaxf( c.x, noise );
    }
    dst[i] = c;
}

/**
 * @brief One step of the living field, cell by cell, from one buffer into the other (which one by
 *        the parity and the step's number); a step this block does not have does nothing.
 *
 * Lenia (Chan 2019): the ring kernel's weighted sum u of the neighbourhood, the growth
 * 2 exp(-(u - mu)^2 / 2 sigma^2) - 1, a tenth of it added. Gray-Scott (Pearson 1993):
 * dU = Du lap U - U V^2 + F (1 - U), dV = Dv lap V + U V^2 - (F + k) V, Du 0.2097, Dv 0.105, one
 * unit of time a step, the five-point Laplacian.
 * @param[in]     hdr   the header
 * @param[in]     node  this step's number in the block, 0 or 1
 * @param[in,out] a     buffer A
 * @param[in,out] b     buffer B
 * @param[in]     land  the landscape (the animated pictures read it)
 */
__global__ void lifeStepKernel( const BlockHeader *hdr, int node, float2 *a, float2 *b, const float *land )
{
    const LifeField &L = hdr->m_life;
    if ( L.m_mode == LifeModeE::OFF || L.m_mode == LifeModeE::FLUID || L.m_mode == LifeModeE::ISLANDS ||
         node >= L.m_steps )
        return;   // those two: their own kernels
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    const bool    fromB = ( ( hdr->m_lifeParity + node ) & 1 ) != 0;
    const float2 *src = fromB ? b : a;
    float2       *dst = fromB ? a : b;
    const int     x = i & ( LifeW - 1 ), y = i >> 8;
    if ( lifeIsPicture( L.m_mode ) )
    {
        dst[i] = pictureCell( hdr, land, src, x, y );
    }
    else if ( L.m_mode == LifeModeE::LENIA )
    {
        constexpr int R = LifeRadius, K = 2 * LifeRadius + 1;
        float         u = 0.0f;
        for ( int dy = -R; dy <= R; ++dy )
        {
            const float2 *row = src + static_cast<size_t>( ( y + dy ) & ( LifeH - 1 ) ) * LifeW;
            for ( int dx = -R; dx <= R; ++dx )
            {
                const float k = cLeniaKernel[( dy + R ) * K + ( dx + R )];
                if ( k != 0.0f ) u += k * row[( x + dx ) & ( LifeW - 1 )].x;
            }
        }
        const float d = ( u - L.m_lenMu ) / fmaxf( L.m_lenSigma, 1.0e-4f );
        const float g = 2.0f * __expf( -0.5f * d * d ) - 1.0f;
        dst[i] = make_float2( fminf( fmaxf( src[i].x + 0.1f * g, 0.0f ), 1.0f ), 0.0f );
    }
    else
    {
        const float2 c = src[i];
        const float2 l = src[y * LifeW + ( ( x - 1 ) & ( LifeW - 1 ) )],
                     r = src[y * LifeW + ( ( x + 1 ) & ( LifeW - 1 ) )];
        const float2 u = src[( ( y - 1 ) & ( LifeH - 1 ) ) * LifeW + x],
                     d = src[( ( y + 1 ) & ( LifeH - 1 ) ) * LifeW + x];
        const float  lapU = l.x + r.x + u.x + d.x - 4.0f * c.x, lapV = l.y + r.y + u.y + d.y - 4.0f * c.y;
        const float  uvv = c.x * c.y * c.y;
        dst[i] = make_float2( fminf( fmaxf( c.x + 0.2097f * lapU - uvv + L.m_feed * ( 1.0f - c.x ), 0.0f ), 1.0f ),
                              fminf( fmaxf( c.y + 0.105f * lapV + uvv - ( L.m_feed + L.m_kill ) * c.y, 0.0f ), 1.0f ) );
    }
}

namespace
{

// ---- The Fluid: a lattice Boltzmann fluid (D2Q9, BGK) on the field's torus, with ink ----

__constant__ int cLbmX[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };   ///< the lattice's directions, across

__constant__ int cLbmY[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };   ///< and along

__constant__ float cLbmW[9] = { 4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f, 1.0f / 9.0f,
                                1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };   ///< their weights

/**
 * @brief One direction's equilibrium.
 * @param[in] k    the direction
 * @param[in] rho  the density
 * @param[in] ux   the velocity across
 * @param[in] uy   along
 * @return f_eq
 */
__device__ __forceinline__ float lbmEq( int k, float rho, float ux, float uy )
{
    const float eu = static_cast<float>( cLbmX[k] ) * ux + static_cast<float>( cLbmY[k] ) * uy;
    return cLbmW[k] * rho * ( 1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * ( ux * ux + uy * uy ) );
}

/**
 * @brief The shortest way round the torus.
 * @param[in] d  a difference
 * @param[in] n  the size
 * @return it within -n/2 .. n/2
 */
__device__ __forceinline__ float torusDelta( float d, float n )
{
    return d - n * rintf( d / n );
}

}   // namespace

/**
 * @brief Seeds the Fluid when asked: at rest but for six eddies of either turn (from the salt).
 * @param[in]     hdr  the header
 * @param[in,out] fa   the lattice, buffer A
 * @param[in,out] fb   buffer B
 */
__global__ void fluidSeedKernel( const BlockHeader *hdr, float *fa, float *fb )
{
    const LifeField &L = hdr->m_life;
    if ( !L.m_reseed || L.m_mode != LifeModeE::FLUID ) return;
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    float              *dst = ( hdr->m_lifeParity & 1 ) ? fb : fa;
    const int           x = i & ( LifeW - 1 ), y = i >> 8;
    const std::uint32_t salt = static_cast<std::uint32_t>( L.m_salt ) * 0x9E3779B9U;
    float               ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 6; ++k )
    {
        const std::uint32_t h = hash32( salt + 104729U * static_cast<std::uint32_t>( k + 1 ) );
        const float cx = static_cast<float>( h & ( LifeW - 1 ) ), cy = static_cast<float>( ( h >> 8 ) & ( LifeH - 1 ) );
        const float turn = ( h >> 20 ) & 1U ? 1.0f : -1.0f;
        const float dx = torusDelta( static_cast<float>( x ) - cx, static_cast<float>( LifeW ) );
        const float dy = torusDelta( static_cast<float>( y ) - cy, static_cast<float>( LifeH ) );
        const float g =
            0.006f * turn * __expf( -( dx * dx + dy * dy ) * ( 1.0f / 128.0f ) );   // an eddy eight cells across
        ux -= g * dy;
        uy += g * dx;
    }
    for ( int k = 0; k < 9; ++k ) dst[static_cast<size_t>( k ) * LifeCells + i] = lbmEq( k, 1.0f, ux, uy );
}

/**
 * @brief One step of the Fluid, cell by cell: the lattice streamed (pulled from the neighbours),
 *        pushed by the jets (the velocity shifted by tau F / rho), relaxed to its equilibrium; the
 *        ink carried back along the flow, the jets' ink added, a little of it fading. Reads one
 *        buffer, writes the other, as the field's steps do.
 * @param[in]     hdr    the header
 * @param[in]     node   this step's number in the block, 0 or 1
 * @param[in,out] fa     the lattice, buffer A
 * @param[in,out] fb     buffer B
 * @param[in,out] lifeA  the field (the ink), buffer A
 * @param[in,out] lifeB  buffer B
 */
__global__ void fluidStepKernel( const BlockHeader *hdr, int node, float *fa, float *fb, float2 *lifeA, float2 *lifeB )
{
    const LifeField &L = hdr->m_life;
    if ( L.m_mode != LifeModeE::FLUID || node >= L.m_steps ) return;
    const int i = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= LifeCells ) return;
    const bool    fromB = ( ( hdr->m_lifeParity + node ) & 1 ) != 0;
    const float  *src = fromB ? fb : fa;
    float        *dst = fromB ? fa : fb;
    const float2 *ink = fromB ? lifeB : lifeA;
    float2       *inkOut = fromB ? lifeA : lifeB;
    const int     x = i & ( LifeW - 1 ), y = i >> 8;
    float         f[9], rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for ( int k = 0; k < 9; ++k )
    {
        const int sx = ( x - cLbmX[k] ) & ( LifeW - 1 ), sy = ( y - cLbmY[k] ) & ( LifeH - 1 );
        f[k] = src[static_cast<size_t>( k ) * LifeCells + sy * LifeW + sx];
        rho += f[k];
        ux += f[k] * static_cast<float>( cLbmX[k] );
        uy += f[k] * static_cast<float>( cLbmY[k] );
    }
    if ( !( rho > 0.05f && rho < 20.0f ) )
    {   // a lattice that left the world: at rest again
        rho = 1.0f;
        ux = uy = 0.0f;
        for ( int k = 0; k < 9; ++k ) f[k] = lbmEq( k, 1.0f, 0.0f, 0.0f );
    }
    else
    {
        ux /= rho;
        uy /= rho;
    }
    // The jets: a little above the bottom, each where its note lies across, blowing up with its lean.
    float fx = 0.0f, fy = 0.0f, inkIn = 0.0f;
    for ( int j = 0; j < LifeJets; ++j )
    {
        const float amt = L.m_jetAmt[j];
        if ( amt <= 0.0f ) continue;
        const float dx = torusDelta( static_cast<float>( x ) - L.m_jetX[j], static_cast<float>( LifeW ) );
        const float dy = torusDelta( static_cast<float>( y ) - 6.0f, static_cast<float>( LifeH ) );
        const float g = amt * __expf( -( dx * dx + dy * dy ) * ( 1.0f / 18.0f ) );   // three cells wide
        fx += g * L.m_jetSway[j];
        fy += g;
        inkIn += g;
    }
    const float tau = fmaxf( L.m_tau, 0.502f );
    float       vx = ux + tau * L.m_push * fx / rho, vy = uy + tau * L.m_push * fy / rho;
    const float speed = sqrtf( vx * vx + vy * vy );
    if ( speed > 0.12f )
    {
        vx *= 0.12f / speed;
        vy *= 0.12f / speed;
    }   // under a tenth of the lattice's sound: it stays a fluid
    for ( int k = 0; k < 9; ++k )
        dst[static_cast<size_t>( k ) * LifeCells + i] = f[k] - ( f[k] - lbmEq( k, rho, vx, vy ) ) / tau;
    // The ink, from where the flow brought it (bilinear on the torus).
    const float px = static_cast<float>( x ) - ux, py = static_cast<float>( y ) - uy;
    const int   x0 = static_cast<int>( floorf( px ) ), y0 = static_cast<int>( floorf( py ) );
    const float ax = px - static_cast<float>( x0 ), ay = py - static_cast<float>( y0 );
    auto        at = [&]( int cx, int cy ) { return ink[( cy & ( LifeH - 1 ) ) * LifeW + ( cx & ( LifeW - 1 ) )].x; };
    const float top = at( x0, y0 ) + ( at( x0 + 1, y0 ) - at( x0, y0 ) ) * ax;
    const float bottom = at( x0, y0 + 1 ) + ( at( x0 + 1, y0 + 1 ) - at( x0, y0 + 1 ) ) * ax;
    const float d = fminf( ( top + ( bottom - top ) * ay ) * L.m_fade + 0.03f * inkIn, 1.0f );
    inkOut[i] = make_float2( d, 10.0f * sqrtf( ux * ux + uy * uy ) );
}

// ---- Islands: Cahn-Hilliard separation with Oono and Puri's long-range term, on a coarse grid ----

/**
 * @brief The Islands, in one CUDA block: its 64 x 32 grid in shared memory, `substeps` explicit
 *        steps of dc = lap(c^3 - c - g lap c) - eps (c - mean) (g 0.6, a fiftieth of a unit each, a
 *        grain of noise so it never quite freezes), then the field drawn from it: each of its cells
 *        read bilinearly, drifted across, as 0.5 + 0.5 tanh(2.5 c). Seeded with a little noise about
 *        the mean when asked.
 * @param[in]     hdr    the header
 * @param[in,out] isles  the grid, kept between blocks
 * @param[in,out] lifeA  the field, buffer A
 * @param[in,out] lifeB  buffer B
 */
__global__ void __launch_bounds__( 1024 )
    islandsKernel( const BlockHeader *hdr, float *isles, float2 *lifeA, float2 *lifeB )
{
    const LifeField &L = hdr->m_life;
    if ( L.m_mode != LifeModeE::ISLANDS || L.m_steps < 1 ) return;
    constexpr int       N = IsleW * IsleH;
    __shared__ float    c[N], mu[N];
    const int           t = static_cast<int>( threadIdx.x );
    const std::uint32_t salt = static_cast<std::uint32_t>( L.m_salt ) * 0x9E3779B9U;
    for ( int k = t; k < N; k += blockDim.x )
        c[k] = L.m_reseed
                   ? L.m_mean + 0.2f * ( static_cast<float>(
                                             hash32( salt ^ ( static_cast<std::uint32_t>( k ) * 2654435761U ) ) >> 8 ) *
                                             ( 1.0f / 16777216.0f ) -
                                         0.5f )
                   : isles[k];
    __syncthreads();
    constexpr float g = 0.6f, dt = 0.02f;
    for ( int s = 0; s < L.m_substeps; ++s )
    {
        for ( int k = t; k < N; k += blockDim.x )
        {
            const int   x = k & ( IsleW - 1 ), y = k / IsleW;
            const float lap = c[y * IsleW + ( ( x + 1 ) & ( IsleW - 1 ) )] +
                              c[y * IsleW + ( ( x - 1 ) & ( IsleW - 1 ) )] +
                              c[( ( y + 1 ) & ( IsleH - 1 ) ) * IsleW + x] +
                              c[( ( y - 1 ) & ( IsleH - 1 ) ) * IsleW + x] - 4.0f * c[k];
            mu[k] = c[k] * c[k] * c[k] - c[k] - g * lap;
        }
        __syncthreads();
        for ( int k = t; k < N; k += blockDim.x )
        {
            const int           x = k & ( IsleW - 1 ), y = k / IsleW;
            const float         lap = mu[y * IsleW + ( ( x + 1 ) & ( IsleW - 1 ) )] +
                                      mu[y * IsleW + ( ( x - 1 ) & ( IsleW - 1 ) )] +
                                      mu[( ( y + 1 ) & ( IsleH - 1 ) ) * IsleW + x] +
                                      mu[( ( y - 1 ) & ( IsleH - 1 ) ) * IsleW + x] - 4.0f * mu[k];
            const std::uint32_t h =
                hash32( ( hdr->m_blockIndex * 16U + static_cast<std::uint32_t>( s ) ) * 2654435761U ^
                        static_cast<std::uint32_t>( k ) ^ salt );
            const float noise = 0.004f * ( static_cast<float>( h >> 8 ) * ( 1.0f / 16777216.0f ) - 0.5f );
            c[k] = fminf( fmaxf( c[k] + dt * ( lap - L.m_eps * ( c[k] - L.m_mean ) ) + noise, -1.5f ), 1.5f );
        }
        __syncthreads();
    }
    for ( int k = t; k < N; k += blockDim.x ) isles[k] = c[k];
    // The field (the buffer this block's one step leads to), drawn from the grid.
    float2         *dst = ( hdr->m_lifeParity & 1 ) ? lifeA : lifeB;
    constexpr float sx = static_cast<float>( IsleW ) / static_cast<float>( LifeW ),
                    sy = static_cast<float>( IsleH ) / static_cast<float>( LifeH );
    for ( int i = t; i < LifeCells; i += blockDim.x )
    {
        const float px = ( static_cast<float>( i & ( LifeW - 1 ) ) + 0.5f ) * sx - 0.5f + L.m_drift,
                    py = ( static_cast<float>( i >> 8 ) + 0.5f ) * sy - 0.5f;
        const int   x0 = static_cast<int>( floorf( px ) ), y0 = static_cast<int>( floorf( py ) );
        const float ax = px - static_cast<float>( x0 ), ay = py - static_cast<float>( y0 );
        auto        at = [&]( int cx, int cy ) { return c[( cy & ( IsleH - 1 ) ) * IsleW + ( cx & ( IsleW - 1 ) )]; };
        const float top = at( x0, y0 ) + ( at( x0 + 1, y0 ) - at( x0, y0 ) ) * ax;
        const float bottom = at( x0, y0 + 1 ) + ( at( x0 + 1, y0 + 1 ) - at( x0, y0 + 1 ) ) * ax;
        const float v = top + ( bottom - top ) * ay;
        dst[i] = make_float2( 0.5f + 0.5f * tanhf( 2.5f * v ), v );
    }
}

/**
 * @brief The living field's mean and mean square of what the partials read, in one CUDA block, in
 *        a fixed order (bit-identical runs).
 * @param[in]  hdr    the header
 * @param[in]  a      buffer A
 * @param[in]  b      buffer B
 * @param[out] stats  receives mean, mean square
 */
__global__ void lifeStatsKernel( const BlockHeader *hdr, const float2 *a, const float2 *b, float *stats )
{
    __shared__ float s1[256], s2[256];
    const LifeField &L = hdr->m_life;
    const int        t = static_cast<int>( threadIdx.x );
    float            m = 0.0f, q = 0.0f;
    if ( L.m_mode != LifeModeE::OFF )
    {
        const float2 *cur = lifeCurrent( hdr, a, b );
        for ( int i = t; i < LifeCells; i += 256 )
        {
            const float v = lifeValue( L.m_mode, cur[i] );
            m += v;
            q += v * v;
        }
    }
    s1[t] = m;
    s2[t] = q;
    __syncthreads();
    for ( int s = 128; s > 0; s >>= 1 )
    {
        if ( t < s )
        {
            s1[t] += s1[t + s];
            s2[t] += s2[t + s];
        }
        __syncthreads();
    }
    if ( t == 0 )
    {
        stats[0] = s1[0] / LifeCells;
        stats[1] = s2[0] / LifeCells;
    }
}

/**
 * @brief Lenia's ring kernel into this file's constant memory.
 * @param[in] kernel  (2 LifeRadius + 1)^2 weights, normalised
 * @return the CUDA call's result
 */
cudaError_t uploadLifeTables( const float *kernel )
{
    return cudaMemcpyToSymbol( cLeniaKernel, kernel, sizeof( cLeniaKernel ) );
}

}   // namespace additiv::gpu
