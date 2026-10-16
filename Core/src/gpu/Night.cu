/**
 * @file Night.cu
 * @brief The night (Phase 30): the Aeolian harp in the wind, thunder from a lightning channel, the singing ice, and
 *        their mix into the frames' channels 24 .. 27 (near left and right, far left and right).
 *
 * The harp's strings each shed a vortex street that locks onto one of their harmonics (Facchinetti's wake
 * oscillator); a strike lays out a tortuous channel whose segments each send an N-wave, summed by distance into a ring
 * of whole numbers; the ice's stress grid slips in avalanches (Olami, Feder and Christensen), each heard as a chirp
 * falling from the highs (flexural waves are dispersive). Every sum in a fixed order or in whole numbers: a block
 * renders bit for bit the same every time.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

namespace
{

constexpr float ThunderFixed = 4194304.0f;   ///< the thunder's ring: whole numbers of 2^-22
constexpr float SoundSpeed = 343.0f;         ///< m/s
constexpr float EarHeight = 1.7f;            ///< m: the listener's ears above the ground
constexpr float EarHalf = 0.0875f;           ///< m: half the head's width
constexpr int   IceSteps = 16;               ///< the ice's relaxation steps a block (a crack runs over blocks)

constexpr float IceFall = 1.9f;   ///< the chirp's frequency falls as t^-IceFall (between 5/3 and 2)

/**
 * @brief The wind's velocity at a cell of its lattice: the D2Q9 moments in the order of Weather.cu's tables
 *        (rest, +x, +y, -x, -y, then the diagonals +x+y, -x+y, -x-y, +x-y).
 * @param[in] wind  the lattice (buffer A, after the block's steps)
 * @param[in] idx   the cell
 * @return across and from near to far, lattice units a step
 */
__device__ float2 harpWind( const float *wind, int idx )
{
    float f[9];
    for ( int k = 0; k < 9; ++k ) f[k] = wind[k * WindCells + idx];
    const float rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    const float ux = f[1] - f[3] + f[5] - f[6] - f[7] + f[8];
    const float uy = f[2] - f[4] + f[5] + f[6] - f[7] - f[8];
    return rho > 1.0e-3f ? make_float2( ux / rho, uy / rho ) : make_float2( 0.0f, 0.0f );
}

/**
 * @brief A smooth random vector along a line of segments: value noise at one scale, each knot a die per axis.
 * @param[in] seed   the strike's dice
 * @param[in] i      the segment
 * @param[in] scale  segments between knots
 * @param[in] salt   which noise
 * @return -1 .. 1 per axis
 */
__device__ float3 channelNoise( std::uint32_t seed, int i, int scale, std::uint32_t salt )
{
    const int   k = i / scale;
    const float u = static_cast<float>( i - k * scale ) / static_cast<float>( scale );
    const float w = u * u * ( 3.0f - 2.0f * u );
    float       v[3];
    for ( int a = 0; a < 3; ++a )
    {
        const float h0 =
            2.0f * hash01( seed, static_cast<std::uint32_t>( k ), salt * 8U + static_cast<std::uint32_t>( a ) ) - 1.0f;
        const float h1 =
            2.0f * hash01( seed, static_cast<std::uint32_t>( k + 1 ), salt * 8U + static_cast<std::uint32_t>( a ) ) -
            1.0f;
        v[a] = h0 + ( h1 - h0 ) * w;
    }
    return make_float3( v[0], v[1], v[2] );
}

/**
 * @brief Lays out a stretch of a lightning channel from `from` to `to` in `count` segments, by the 32 lanes of a warp
 *        (count / 32 consecutive segments each): tortuous at three scales (a segment's turn, a few metres', the
 *        stroke's own wander), the wander's mean taken out again so that it ends where it should; where each lane's
 *        stretch begins from a scan over the lanes. Each segment's middle and length, its direction and its weight.
 * @param[in]  T       the thunder
 * @param[in]  from    where it begins, m
 * @param[in]  to      where it ends
 * @param[in]  count   its segments, a multiple of 32
 * @param[in]  weight  its energy per metre against the main stroke's
 * @param[in]  salt    which noise
 * @param[out] seg     its segments: (middle, length), (direction, weight) each
 * @param[out] W       shared scratch of `count` wanders
 * @pre called by all 32 lanes of a warp with the same arguments
 */
__device__ void layChannel( const Thunder &T, float3 from, float3 to, int count, float weight, std::uint32_t salt,
                            float4 *seg, float3 *W )
{
    const std::uint32_t seed = static_cast<std::uint32_t>( T.m_seed );
    const int           lane = static_cast<int>( threadIdx.x ) & 31, per = count / 32, first = lane * per;
    const float3 d = make_float3( ( to.x - from.x ) / count, ( to.y - from.y ) / count, ( to.z - from.z ) / count );
    const float  len = sqrtf( d.x * d.x + d.y * d.y + d.z * d.z );
    const float  t = T.m_tortuosity;
    float3       sum = make_float3( 0.0f, 0.0f, 0.0f );
    for ( int i = first; i < first + per; ++i )
    {
        const float3 a = channelNoise( seed, i, 1, salt ), b = channelNoise( seed, i, 16, salt + 1U ),
                     c = channelNoise( seed, i, 256, salt + 2U );
        const float3 w =
            make_float3( len * t * ( a.x + 2.0f * b.x + 3.0f * c.x ), len * t * ( a.y + 2.0f * b.y + 3.0f * c.y ),
                         len * t * ( a.z + 2.0f * b.z + 3.0f * c.z ) );
        W[i] = w;
        sum = make_float3( sum.x + w.x, sum.y + w.y, sum.z + w.z );
    }
    for ( int o = 16; o > 0; o >>= 1 )
    {   // the lanes' sums to lane 0 in a fixed order, then handed to all
        sum.x += __shfl_down_sync( 0xFFFFFFFFU, sum.x, o );
        sum.y += __shfl_down_sync( 0xFFFFFFFFU, sum.y, o );
        sum.z += __shfl_down_sync( 0xFFFFFFFFU, sum.z, o );
    }
    const float3 mean =
        make_float3( __shfl_sync( 0xFFFFFFFFU, sum.x, 0 ) / count, __shfl_sync( 0xFFFFFFFFU, sum.y, 0 ) / count,
                     __shfl_sync( 0xFFFFFFFFU, sum.z, 0 ) / count );
    float3 mine = make_float3( 0.0f, 0.0f, 0.0f );
    for ( int i = first; i < first + per; ++i )
        mine = make_float3( mine.x + d.x + W[i].x - mean.x, mine.y + d.y + W[i].y - mean.y,
                            mine.z + d.z + W[i].z - mean.z );
    float3 scan = mine;
    for ( int o = 1; o < 32; o <<= 1 )
    {
        const float ux = __shfl_up_sync( 0xFFFFFFFFU, scan.x, o ), uy = __shfl_up_sync( 0xFFFFFFFFU, scan.y, o ),
                    uz = __shfl_up_sync( 0xFFFFFFFFU, scan.z, o );
        if ( lane >= o ) scan = make_float3( scan.x + ux, scan.y + uy, scan.z + uz );
    }
    float3 p = make_float3( from.x + scan.x - mine.x, from.y + scan.y - mine.y, from.z + scan.z - mine.z );
    for ( int i = first; i < first + per; ++i )
    {
        const float3 s = make_float3( d.x + W[i].x - mean.x, d.y + W[i].y - mean.y, d.z + W[i].z - mean.z );
        const float  l = fmaxf( sqrtf( s.x * s.x + s.y * s.y + s.z * s.z ), 1.0e-3f );
        seg[2 * i] = make_float4( p.x + 0.5f * s.x, p.y + 0.5f * s.y, fmaxf( p.z + 0.5f * s.z, 0.0f ), l );
        seg[2 * i + 1] = make_float4( s.x / l, s.y / l, s.z / l, weight );
        p = make_float3( p.x + s.x, p.y + s.y, p.z + s.z );
    }
    __syncwarp();
}

/**
 * @brief An N-wave with sloped edges, piecewise linear: up to 1 over `rise`, down to -1 by `len - rise`, back to 0 at
 *        `len`.
 * @param[in] j     the sample within the wave
 * @param[in] len   its length, samples
 * @param[in] rise  its edges, samples (at least 1)
 * @return -1 .. 1
 */
__device__ __forceinline__ float nWave( float j, float len, float rise )
{
    if ( j < rise ) return j / rise;
    if ( j > len - rise ) return fminf( ( j - len ) / rise, 0.0f );
    return 1.0f - 2.0f * ( j - rise ) / fmaxf( len - 2.0f * rise, 1.0f );
}

}   // namespace

/**
 * @brief The Aeolian harp (Phase 30): one warp a string, one lane a mode (harmonic n = lane + 1, a little stretched).
 *        Per sample every mode turns as a complex one-pole and takes its share of the wake's lift, the warp sums (to
 *        lane 0 in a fixed order, then handed to every lane) the string's acceleration where the wake stands and its
 *        sound at the bridge, and the wake -- a van der Pol oscillator at the vortices' frequency w_f = 2 pi St U / D
 *        -- takes a semi-implicit Euler step driven by that acceleration (Facchinetti, de Langre and Biolley 2004:
 *        q'' + eps w_f (q^2 - 1) q' + w_f^2 q = A y'', the string y'' + .. + w_n^2 y = M w_f^2 q). So a string locks
 *        onto the harmonic nearest its vortices and holds it, and jumps when the wind has moved on.
 * @param[in]     hdr      the block header
 * @param[in]     wind     the wind's lattice (buffer A, after the block's steps)
 * @param[in,out] modes    per string and mode its complex amplitude y - i y'/w (in diameters)
 * @param[in,out] wakes    per string its wake (q, q')
  * @param[out]    scratch  the night's scratch: rows 0 .. 3 the harp's channels, rows NightStringRow .. its strings
 * @param[in]     maxN     the engine's block size (the scratch stride)
 * @param[out]    stats    [56] the strings' mean wake amplitude, [57] the harmonic they sound (by energy, averaged)
 */
__global__ void __launch_bounds__( HarpStrings * 32 )
    harpKernel( const BlockHeader *hdr, const float *wind, float2 *modes, float2 *wakes, float *scratch, int maxN,
                float *stats )
{
    __shared__ float   sQ[HarpStrings];   // per string its wake's amplitude
    __shared__ float   sH[HarpStrings];   // and the harmonic it sounds
    const AeolianHarp &H = hdr->m_harp;
    if ( !H.m_on ) return;
    const int   s = static_cast<int>( threadIdx.x ) >> 5, lane = static_cast<int>( threadIdx.x ) & 31;
    const int   n = hdr->m_n;
    const float fs = hdr->m_sampleRate, dt = 1.0f / fs;
    float2      z = modes[s * HarpModes + lane];
    float2      w = wakes[s];
    if ( H.m_clear )
    {
        z = make_float2( 0.0f, 0.0f );
        w = make_float2( 0.05f + 0.1f * hash01( 0x4A5B6C7DU, static_cast<std::uint32_t>( s ), 1 ), 0.0f );
    }
    // The wind where the string stands (a block of 4 x 3 cells round the harp's), as Strouhal's frequency.
    const int    cx = min( max( H.m_x + ( s & 3 ) - 2, 0 ), WindW - 1 );
    const int    cy = min( max( H.m_y + ( s >> 2 ) - 1, 0 ), WindH - 1 );
    const float2 u = harpWind( wind, cy * WindW + cx );
    const float  speed = sqrtf( u.x * u.x + u.y * u.y ) * H.m_speed;   // m/s
    const float  wf = fminf( 6.2831853f * 0.2f * speed / fmaxf( H.m_diameter[s], 1.0e-4f ),
                             6.2831853f * fminf( 6000.0f, 0.15f * fs ) );
    // The mode: its frequency (a little stiffness), its decay (a few seconds, shorter higher up), its pole, where the
    // wake drives it and how the bridge hears it.
    const float nh = static_cast<float>( lane + 1 );
    const float fn = H.m_hz[s] * nh * sqrtf( 1.0f + 2.0e-5f * nh * nh );
    const bool  live = fn > 20.0f && fn < 0.45f * fs;
    const float wn = 6.2831853f * fn;
    const float sigma = 6.9077553f / ( 4.0f / ( 1.0f + 0.15f * nh ) );
    float       rs, rc;
    sincosf( wn * dt, &rs, &rc );
    const float keep = expf( -sigma * dt );
    const float pr = live ? keep * rc : 0.0f, pi = live ? keep * rs : 0.0f;
    const float phi = live ? sinf( nh * 3.1415927f * 0.31f ) + 0.6f * sinf( nh * 3.1415927f * 0.73f ) : 0.0f;
    const float bridge = live ? rsqrtf( nh ) * fn * 1.0e-3f : 0.0f;
    const float lift = H.m_force * wf * wf, inv = live ? dt / wn : 0.0f;
    float      *mine = scratch + static_cast<size_t>( NightStringRow + s ) * maxN;
    float       energy = 0.0f;
    for ( int i = 0; i < n; ++i )
    {
        // the mode: turned, then the wake's lift as a kick to its velocity (y' += f dt: the imaginary part -f dt / w)
        const float f = lift * w.x * phi;
        const float zr = pr * z.x - pi * z.y, zi = pr * z.y + pi * z.x - f * inv;
        z = make_float2( zr, zi );
        float acc = phi * ( -wn * wn * zr + f );   // its acceleration where the wake stands
        float snd = bridge * zr;
        for ( int o = 16; o > 0; o >>= 1 )
        {
            acc += __shfl_down_sync( 0xFFFFFFFFU, acc, o );
            snd += __shfl_down_sync( 0xFFFFFFFFU, snd, o );
        }
        acc = __shfl_sync( 0xFFFFFFFFU, acc, 0 );
        // the wake: one semi-implicit Euler step (every lane the same, from the same numbers)
        const float q = w.x;
        float       p = w.y + dt * ( -H.m_eps * wf * ( q * q - 1.0f ) * w.y - wf * wf * q + H.m_couple * acc );
        p = fminf( fmaxf( p, -40.0f * wf - 1.0f ), 40.0f * wf + 1.0f );
        w = make_float2( fminf( fmaxf( q + dt * p, -8.0f ), 8.0f ), p );
        if ( lane == 0 ) mine[i] = snd;
        energy += zr * zr;
    }
    if ( !( fabsf( z.x ) < 1.0e3f && fabsf( z.y ) < 1.0e3f ) ) z = make_float2( 0.0f, 0.0f );   // never lost
    if ( !( fabsf( w.x ) <= 8.0f && fabsf( w.y ) < 1.0e9f ) ) w = make_float2( 0.1f, 0.0f );
    modes[s * HarpModes + lane] = z;
    if ( lane == 0 ) wakes[s] = w;
    // which harmonic the string sounds: the energy's mean harmonic number (fixed order)
    float en = energy, eh = energy * nh;
    for ( int o = 16; o > 0; o >>= 1 )
    {
        en += __shfl_down_sync( 0xFFFFFFFFU, en, o );
        eh += __shfl_down_sync( 0xFFFFFFFFU, eh, o );
    }
    if ( lane == 0 )
    {
        sQ[s] = sqrtf( w.x * w.x + ( wf > 0.0f ? w.y * w.y / ( wf * wf ) : 0.0f ) );
        sH[s] = en > 1.0e-20f ? eh / en : 0.0f;
    }
    __syncthreads();
    // the strings into the harp's four channels, each at its place, summed in their order
    for ( int i = static_cast<int>( threadIdx.x ); i < n; i += HarpStrings * 32 )
    {
        float l = 0.0f, r = 0.0f;
        for ( int k = 0; k < HarpStrings; ++k )
        {
            float gl, gr;
            __sincosf( 0.78539816f * ( 1.0f + H.m_place[k] ), &gr, &gl );
            const float v = scratch[static_cast<size_t>( NightStringRow + k ) * maxN + i];
            l += gl * v;
            r += gr * v;
        }
        l = H.m_level * l;
        r = H.m_level * r;
        scratch[0 * maxN + i] = 0.85f * l;
        scratch[1 * maxN + i] = 0.85f * r;
        scratch[2 * maxN + i] = 0.35f * l;
        scratch[3 * maxN + i] = 0.35f * r;
    }
    if ( threadIdx.x == 0 )
    {
        float q = 0.0f, h = 0.0f;
        for ( int k = 0; k < HarpStrings; ++k )
        {
            q += sQ[k];
            h += sH[k];
        }
        stats[56] = q / HarpStrings;
        stats[57] = h / HarpStrings;
    }
}

/**
 * @brief A lightning strike's channel (Phase 30), in the block of the strike, by one warp: the main stroke from the
 *        cloud's base down to its foot (ThunderMain segments), then its branches off the upper half (the rest of the
 *        segments, one after another), every one tortuous at three scales.
 * @param[in]  hdr  the block header
 * @param[out] seg  ThunderSegments segments: (middle, length), (direction, weight) each
 */
__global__ void __launch_bounds__( 32 ) thunderChannelKernel( const BlockHeader *hdr, float4 *seg )
{
    __shared__ float3 W[ThunderMain];
    const Thunder    &T = hdr->m_thunder;
    if ( !T.m_on || !T.m_strike ) return;
    const std::uint32_t seed = static_cast<std::uint32_t>( T.m_seed );
    float               sa, ca;
    sincosf( T.m_azimuth, &sa, &ca );
    const float3 foot = make_float3( T.m_distance * sa, T.m_distance * ca, 0.0f );
    {   // the main stroke, its top a little off over the foot
        const float  r = 0.4f * T.m_height * hash01( seed, 1, 11 ), a = 6.2831853f * hash01( seed, 2, 11 );
        const float3 top = make_float3( foot.x + r * cosf( a ), foot.y + r * sinf( a ), T.m_height );
        layChannel( T, top, foot, ThunderMain, 1.0f, 20U, seg, W );
    }
    constexpr int per = ( ThunderSegments - ThunderMain ) / ThunderBranches;
    for ( int b = 0; b < ThunderBranches; ++b )
    {   // a branch: from a point of the upper half, down and aside, a fifth to a half of the height long
        const std::uint32_t ub = static_cast<std::uint32_t>( b );
        const int at = static_cast<int>( ( 0.1f + 0.4f * hash01( seed, ub, 12 ) ) * static_cast<float>( ThunderMain ) );
        const float4 m = seg[2 * at];
        const float  len = T.m_height * ( 0.2f + 0.3f * hash01( seed, ub, 13 ) );
        const float  a = 6.2831853f * hash01( seed, ub, 14 );
        const float3 from = make_float3( m.x, m.y, m.z );
        const float3 to =
            make_float3( m.x + 0.7f * len * cosf( a ), m.y + 0.7f * len * sinf( a ), fmaxf( m.z - 0.7f * len, 0.0f ) );
        layChannel( T, from, to, per, 0.35f * T.m_branches, 30U + 4U * ub, seg + 2 * ( ThunderMain + b * per ), W );
    }
}

/**
 * @brief A sixteenth of a strike's segments into the thunder's ring (Phase 30): one CUDA block a segment. Its N-wave
 *        (Few 1969: the channel's radius R0 = sqrt(E / (pi p0)) at E = 1e5 J/m, the wave 2.6 R0 / c long, lengthening
 *        slowly on its way -- as the eighth root of the distance, so that its peak lies at 60 .. 100 Hz, as measured) reaches each ear when its distance says, smeared over the time its length
 *        takes along the line of sight -- so a segment across the line of sight claps and one along it rumbles --,
 *        its edges sloped by the air's absorption (the more, the farther), shaded by the head. Whole numbers, so any
 *        order sums the same.
 * @param[in]     hdr   the block header
 * @param[in]     seg   the channel (thunderChannelKernel)
 * @param[in,out] ring  per ear ThunderRing samples of whole numbers
 */
__global__ void __launch_bounds__( 256 ) thunderRenderKernel( const BlockHeader *hdr, const float4 *seg, int *ring )
{
    const Thunder &T = hdr->m_thunder;
    if ( !T.m_on || T.m_chunk < 0 || T.m_chunk >= ThunderChunks ) return;
    const int    k = T.m_chunk * ( ThunderSegments / ThunderChunks ) + static_cast<int>( blockIdx.x );
    const float4 a = seg[2 * k], b = seg[2 * k + 1];
    if ( b.w <= 0.0f ) return;
    const float fs = hdr->m_sampleRate;
    float       sy, cy;
    sincosf( T.m_yaw, &sy, &cy );
    const float3 right = make_float3( cy, -sy, 0.0f );   // the head turned by yaw (0: facing +y)
    const float3 rel = make_float3( a.x, a.y, a.z - EarHeight );
    const float  r = fmaxf( sqrtf( rel.x * rel.x + rel.y * rel.y + rel.z * rel.z ), 1.0f );
    const float3 u = make_float3( rel.x / r, rel.y / r, rel.z / r );
    const float  energy = 1.0e5f * b.w * T.m_energy;   // J/m
    const float  r0 = sqrtf( energy / ( 3.1415927f * 101325.0f ) );
    const float  tau = 2.6f * r0 / SoundSpeed * powf( fmaxf( r / ( 10.0f * r0 ), 1.0f ), 0.125f );
    const float  smear = fabsf( b.x * u.x + b.y * u.y + b.z * u.z ) * a.w / SoundSpeed;
    const float  sigma = 5.4e-6f * sqrtf( r ) + 2.0e-7f * r;   // the air's absorption as the edges' slope, s
    const float  len = ( tau + smear ) * fs, rise = fmaxf( 2.5f * sigma * fs + 0.5f * smear * fs, 1.0f );
    const float  amp = sqrtf( b.w * T.m_energy ) * ( 100.0f / r ) * tau / ( tau + smear ) * ThunderFixed;
    for ( int e = 0; e < 2; ++e )
    {
        const float  side = e == 0 ? -1.0f : 1.0f;
        const float3 ear = make_float3( side * EarHalf * right.x, side * EarHalf * right.y, 0.0f );
        const float  dx = rel.x - ear.x, dy = rel.y - ear.y, dz = rel.z;
        const float  re = sqrtf( dx * dx + dy * dy + dz * dz );
        const float  shade = fmaxf( 0.6f + 0.4f * side * ( u.x * right.x + u.y * right.y ), 0.2f );
        const int    at = T.m_origin + static_cast<int>( re / SoundSpeed * fs + 0.5f );
        const int    count = min( static_cast<int>( len ) + 1, 48000 );
        int         *mine = ring + static_cast<size_t>( e ) * ThunderRing;
        for ( int j = static_cast<int>( threadIdx.x ); j < count; j += 256 )
        {
            const int v = __float2int_rn( amp * shade * nWave( static_cast<float>( j ), len, rise ) );
            if ( v != 0 ) atomicAdd( mine + ( ( at + j ) & ( ThunderRing - 1 ) ), v );
        }
    }
}

/**
 * @brief The thunder's ring read out (Phase 30): this block's samples of each ear, cleared behind the reading, softly
 *        bounded, into the night's scratch (rows 4 .. 7) -- mostly the far plane, the near one as near as the strikes.
 * @param[in]     hdr      the block header
 * @param[in,out] ring     per ear ThunderRing samples of whole numbers
 * @param[out]    scratch  the night's scratch
 * @param[in]     maxN     the engine's block size (the scratch stride)
 */
__global__ void thunderOutKernel( const BlockHeader *hdr, int *ring, float *scratch, int maxN )
{
    const Thunder &T = hdr->m_thunder;
    if ( !T.m_on ) return;
    const int n = hdr->m_n;
    for ( int i = static_cast<int>( threadIdx.x ); i < n; i += static_cast<int>( blockDim.x ) )
    {
        const int idx = ( T.m_read + i ) & ( ThunderRing - 1 );
        float     v[2];
        for ( int e = 0; e < 2; ++e )
        {
            int *at = ring + static_cast<size_t>( e ) * ThunderRing + idx;
            v[e] = T.m_level * tanhf( static_cast<float>( *at ) * ( 1.0f / ThunderFixed ) );
            *at = 0;
        }
        scratch[4 * maxN + i] = T.m_near * v[0];
        scratch[5 * maxN + i] = T.m_near * v[1];
        scratch[6 * maxN + i] = ( 1.0f - T.m_near ) * v[0];
        scratch[7 * maxN + i] = ( 1.0f - T.m_near ) * v[1];
    }
}

/**
 * @brief The ice sheet's stress (Phase 30), one CUDA block, a cell a thread: when the cold brings a crack, every cell
 *        is raised alike until the most stressed one passes its threshold (a reduction in a fixed order); then
 *        IceSteps steps of Olami, Feder and Christensen's relaxation -- every cell past 1 slips at once, to 0, handing
 *        IceShare of its stress to each of its four neighbours (the rim's lost) --, the slips counted in whole numbers.
 *        A crack that has run (a block without a slip after slips) becomes a chirp in the next slot: from where its
 *        slips' middle is on the lake, its loudness the slips' count to the power 0.35 (a crack a hundred times as
 *        large five times as loud).
 * @param[in]     hdr     the block header
 * @param[in,out] stress  the cells' stress
 * @param[in,out] chirps  IceChirps slots: (age s, arrival s, amplitude, top rad/s), (pan, near share, end s, -)
 * @param[in,out] meta    the running crack (slips, sum of x, sum of y), -, the chirps begun
 * @param[out]    stats   [58] the last crack's size, [59] the chirps begun
 */
__global__ void __launch_bounds__( 1024 )
    iceKernel( const BlockHeader *hdr, float *stress, float4 *chirps, int *meta, float *stats )
{
    __shared__ float  F[IceN * IceN];
    __shared__ float  G[IceN * IceN];
    __shared__ float  top[IceN * IceN];
    __shared__ int    sCount, sX, sY;
    const SingingIce &I = hdr->m_ice;
    if ( !I.m_on ) return;
    const int c = static_cast<int>( threadIdx.x );
    const int x = c & ( IceN - 1 ), y = c / IceN;
    if ( c == 0 ) sCount = sX = sY = 0;
    float f = stress[c];
    if ( I.m_crack )
    {   // the cold: all raised alike until the most stressed cell slips
        top[c] = f;
        __syncthreads();
        for ( int s = IceN * IceN / 2; s > 0; s >>= 1 )
        {
            if ( c < s ) top[c] = fmaxf( top[c], top[c + s] );
            __syncthreads();
        }
        f += fmaxf( 1.0f - top[0], 0.0f ) + 1.0e-6f;
    }
    F[c] = f;
    __syncthreads();
    int count = 0;
    for ( int step = 0; step < IceSteps; ++step )
    {
        G[c] = F[c] >= 1.0f ? F[c] : 0.0f;
        __syncthreads();
        float g = 0.0f;
        if ( x > 0 ) g += G[c - 1];
        if ( x < IceN - 1 ) g += G[c + 1];
        if ( y > 0 ) g += G[c - IceN];
        if ( y < IceN - 1 ) g += G[c + IceN];
        if ( G[c] > 0.0f ) ++count;
        F[c] = ( G[c] > 0.0f ? 0.0f : F[c] ) + IceShare * g;
        __syncthreads();
    }
    if ( count > 0 )
    {
        atomicAdd( &sCount, count );
        atomicAdd( &sX, count * x );
        atomicAdd( &sY, count * y );
    }
    stress[c] = F[c];
    __syncthreads();
    if ( c != 0 ) return;
    if ( I.m_reset ) meta[0] = meta[1] = meta[2] = 0;
    if ( sCount > 0 )
    {   // the crack runs on
        meta[0] += sCount;
        meta[1] += sX;
        meta[2] += sY;
    }
    else if ( meta[0] > 0 )
    {   // it has run: its chirp, from its middle on the lake (the listener on the shore, the lake ahead from 20 m)
        const float size = static_cast<float>( meta[0] );
        const float px = ( ( static_cast<float>( meta[1] ) / size + 0.5f ) / IceN - 0.5f ) * I.m_size;
        const float py = 20.0f + ( static_cast<float>( meta[2] ) / size + 0.5f ) / IceN * I.m_size;
        const float d = sqrtf( px * px + py * py );
        const float fs = hdr->m_sampleRate;
        const float hi = 6.2831853f * fminf( 8000.0f, 0.4f * fs ), low = 6.2831853f * 50.0f;
        const float arrive = d / ( 2.0f * sqrtf( I.m_beta * hi ) );   // the highs' group velocity: 2 sqrt(beta w)
        const float end = arrive * powf( hi / low, 1.0f / IceFall );
        const int   slot = meta[4] & ( IceChirps - 1 );
        chirps[2 * slot] = make_float4( 0.0f, arrive, powf( size, 0.35f ) * rsqrtf( 1.0f + d / 100.0f ), hi );
        chirps[2 * slot + 1] =
            make_float4( fminf( fmaxf( 1.4f * px / d, -1.0f ), 1.0f ), expf( -d / 150.0f ), end, 0.0f );
        stats[58] = size;
        meta[4] += 1;
        meta[0] = meta[1] = meta[2] = 0;
    }
    stats[59] = static_cast<float>( meta[4] );
}

/**
 * @brief The ice's chirps (Phase 30), one thread a slot: silent until its highs arrive, then a tone whose frequency
 *        falls as t^-IceFall from the top (8 kHz) to 50 Hz -- its phase the frequency's integral, closed form --
 *        under a quick rise and a long fall; mixed through shared memory in a fixed order into the night's scratch
 *        (rows 8 .. 11).
 * @param[in]     hdr      the block header
 * @param[in,out] chirps   the slots (their ages advanced by the block)
 * @param[out]    scratch  the night's scratch
 * @param[in]     maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__( IceChirps )
    iceSingKernel( const BlockHeader *hdr, float4 *chirps, float *scratch, int maxN )
{
    __shared__ float  sV[IceChirps][MixTile + 1];
    __shared__ float  sG[IceChirps][4];
    const SingingIce &I = hdr->m_ice;
    if ( !I.m_on ) return;
    const int    k = static_cast<int>( threadIdx.x );
    const int    n = hdr->m_n;
    const float  fs = hdr->m_sampleRate;
    float4       a = chirps[2 * k];
    const float4 b = chirps[2 * k + 1];
    if ( I.m_reset ) a.z = 0.0f;   // begun again: every chirp silent
    const bool live = a.z > 0.0f && a.x < b.z;
    {
        float gl, gr, gn, gf;
        __sincosf( 0.78539816f * ( 1.0f + b.x ), &gr, &gl );
        __sincosf( 1.57079633f * ( 1.0f - b.y ), &gf, &gn );
        const float g = live ? a.z * I.m_level : 0.0f;
        sG[k][0] = g * gl * gn;
        sG[k][1] = g * gr * gn;
        sG[k][2] = g * gl * gf;
        sG[k][3] = g * gr * gf;
    }
    const float tmin = fmaxf( a.y, 1.0e-4f ), wtop = a.w, span = fmaxf( b.z - tmin, 1.0e-3f );
    const float coef = wtop * tmin / ( 1.0f - IceFall );   // the phase: coef ((t / tmin)^(1 - p) - 1)
    for ( int t0 = 0; t0 < n; t0 += MixTile )
    {
        for ( int i = 0; i < MixTile; ++i )
        {
            float v = 0.0f;
            if ( live && t0 + i < n )
            {
                const float t = a.x + static_cast<float>( t0 + i ) / fs;
                if ( t >= tmin && t < b.z )
                {
                    float ph = coef * ( __powf( t / tmin, 1.0f - IceFall ) - 1.0f );
                    ph -= 6.2831853f * floorf( ph * 0.15915494f );
                    const float x = ( t - tmin ) / span;
                    const float env = 2.2f * sqrtf( sqrtf( x ) ) * ( 1.0f - x ) * sqrtf( 1.0f - x );
                    v = env * __sinf( ph );
                }
            }
            sV[k][i] = v;
        }
        __syncthreads();
        for ( int o = k; o < 4 * MixTile; o += IceChirps )
        {
            const int ch = o / MixTile, i = o - ch * MixTile;
            if ( t0 + i >= n ) continue;
            float acc = 0.0f;
            for ( int q = 0; q < IceChirps; ++q ) acc += sG[q][ch] * sV[q][i];
            scratch[static_cast<size_t>( 8 + ch ) * maxN + t0 + i] = acc;
        }
        __syncthreads();
    }
    if ( live )
    {
        a.x += static_cast<float>( n ) / fs;
        if ( a.x >= b.z ) a.z = 0.0f;   // done: the slot is free
    }
    chirps[2 * k] = a;
}

namespace
{

/** @brief A kind of caller of the night chorus: its carrier, its call, its rhythm. */
struct ChorusKind
{
    float m_hz;          ///< the carrier, Hz
    float m_spread;      ///< the callers' spread about it (a share)
    float m_fm;          ///< the carrier's glide over a call (a share: +0.08 rises by 8 %)
    float m_pulseRate;   ///< pulses a second within a call (at 20 degrees)
    float m_pulses;      ///< pulses a call
    float m_duty;        ///< the share of a pulse's time it sounds
    float m_period;      ///< s between calls at 20 degrees
    float m_warm;        ///< how much the warmth quickens it (1: Dolbear's crickets, 0: not at all)
    float m_level;       ///< its loudness
    float m_r2;          ///< its second partial's ratio to the carrier
    float m_h2;          ///< and level
    float m_r3;          ///< its third partial's ratio
    float m_h3;          ///< and level
    float m_pad0;        ///< keeps the struct 16-byte aligned
    float m_pad1;        ///< likewise
    float m_pad2;        ///< likewise
};

/**
 * @brief The eight kinds: field cricket, snowy tree cricket, katydid, mole cricket, spring peeper, tree frog, bullfrog,
 *        natterjack toad (their bands and calls after recordings' published figures, rounded).
 */
__constant__ ChorusKind cChorus[ChorusSpecies] = {
    { 4600.0f, 0.04f, 0.0f, 30.0f, 4.0f, 0.5f, 0.55f, 1.0f, 0.7f, 2.0f, 0.15f, 3.0f, 0.05f, 0.0f, 0.0f, 0.0f },
    { 2800.0f, 0.03f, 0.0f, 55.0f, 9.0f, 0.6f, 0.55f, 1.0f, 0.6f, 2.0f, 0.10f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 9500.0f, 0.08f, 0.0f, 70.0f, 6.0f, 0.3f, 1.3f, 0.8f, 0.5f, 1.37f, 0.6f, 1.81f, 0.4f, 0.0f, 0.0f, 0.0f },
    { 2300.0f, 0.03f, 0.0f, 65.0f, 120.0f, 0.6f, 4.0f, 0.7f, 0.5f, 2.0f, 0.1f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 2900.0f, 0.05f, 0.08f, 8.3f, 1.0f, 1.0f, 1.1f, 0.4f, 0.8f, 2.0f, 0.05f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 1100.0f, 0.05f, 0.0f, 25.0f, 9.0f, 0.5f, 2.2f, 0.5f, 0.8f, 2.0f, 0.6f, 3.0f, 0.2f, 0.0f, 0.0f, 0.0f },
    { 230.0f, 0.08f, -0.05f, 70.0f, 45.0f, 0.6f, 7.0f, 0.2f, 1.0f, 2.0f, 0.8f, 3.0f, 0.6f, 0.0f, 0.0f, 0.0f },
    { 1250.0f, 0.05f, 0.0f, 15.0f, 30.0f, 0.5f, 9.0f, 0.4f, 0.7f, 2.0f, 0.4f, 3.0f, 0.1f, 0.0f, 0.0f, 0.0f },
};

/**
 * @brief How much the night's warmth quickens a kind (Dolbear's law for the crickets, milder for the rest): its rates
 *        against 20 degrees.
 * @param[in] k       the kind
 * @param[in] warmth  degrees Celsius
 * @return the factor (at least 0.15)
 */
__device__ __forceinline__ float chorusWarm( const ChorusKind &k, float warmth )
{
    return fmaxf( 0.15f, 1.0f + k.m_warm * ( warmth - 20.0f ) / 15.0f );
}

/**
 * @brief Which kind a caller is, by who calls: the meadow's, the treetops', the pond's, or all, each kind's share
 *        weighted.
 * @param[in] mix  ChorusKindE
 * @param[in] u    the caller's die, 0 .. 1
 * @return the kind, 0 .. ChorusSpecies - 1
 */
__device__ int chorusKindOf( int mix, float u )
{
    const float  meadow[ChorusSpecies] = { 0.5f, 0.3f, 0.0f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f };
    const float  treetops[ChorusSpecies] = { 0.0f, 0.4f, 0.6f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    const float  pond[ChorusSpecies] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.45f, 0.3f, 0.1f, 0.15f };
    const float  all[ChorusSpecies] = { 0.2f, 0.15f, 0.15f, 0.05f, 0.2f, 0.12f, 0.05f, 0.08f };
    const float *w = mix == 0 ? meadow : mix == 1 ? treetops : mix == 2 ? pond : all;
    float        acc = 0.0f;
    for ( int k = 0; k < ChorusSpecies; ++k )
    {
        acc += w[k];
        if ( u < acc ) return k;
    }
    return ChorusSpecies - 1;
}

}   // namespace

/**
 * @brief The night chorus's clocks, one thread a caller (Phase 31): what each heard in the last block -- from every
 *        caller whose call began then, by its distance; and the loudness in its band of the other kinds calling --
 *        then its clock: a call of its own kind heard close enough resets it towards its start by Sync (Greenfield's
 *        inhibitory resetting, unless its own call is already under way), a clock come round calls, unless a louder
 *        caller of another kind fills its band (then it waits, half a period at most). Reads the last block's states,
 *        writes this block's: every caller the same picture of the field, so any order gives the same.
 * @param[in]     hdr    the block header
 * @param[in,out] state  per caller two entries in two buffers (by the block's parity): (clock, waited s, carrier Hz,
 *                       level), (across, from near to far, kind, s since its call began at this block's start or -1)
 * @param[in,out] stats  [26 .. 28] gather the calls begun, the callers calling and those waiting (whole numbers)
 */
__global__ void __launch_bounds__( 256 ) chorusStepKernel( const BlockHeader *hdr, float4 *state, float *stats )
{
    __shared__ float4  tile[256];
    __shared__ float4  tileB[256];
    __shared__ int     sBegun, sCalling, sWaiting;
    const NightChorus &C = hdr->m_chorus;
    if ( !C.m_on ) return;
    const int     i = static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x );
    const bool    odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4 *src = state + ( odd ? 2 * ChorusCallers : 0 );
    float4       *dst = state + ( odd ? 0 : 2 * ChorusCallers );
    const float   fs = hdr->m_sampleRate, dt = static_cast<float>( hdr->m_n ) / fs;
    if ( threadIdx.x == 0 ) sBegun = sCalling = sWaiting = 0;
    float4 a = src[2 * i], b = src[2 * i + 1];
    if ( C.m_reset )
    {   // scattered over the field, each its kind, carrier, loudness, its clock anywhere
        const std::uint32_t u = static_cast<std::uint32_t>( i );
        const int           kind = chorusKindOf( C.m_kind, hash01( 0xC40125U, u, 1 ) );
        const ChorusKind   &K = cChorus[kind];
        float               hz = K.m_hz * ( 1.0f + K.m_spread * ( 2.0f * hash01( 0xC40125U, u, 2 ) - 1.0f ) );
        if ( C.m_tune > 0.0f && C.m_pitchCount > 0 )
        {   // pulled to the nearest of the scale's pitches
            float best = C.m_pitch[0];
            for ( int q = 1; q < min( C.m_pitchCount, ChorusPitches ); ++q )
                if ( fabsf( logf( C.m_pitch[q] / hz ) ) < fabsf( logf( best / hz ) ) ) best = C.m_pitch[q];
            hz = hz + C.m_tune * ( best - hz );
        }
        a = make_float4( hash01( 0xC40125U, u, 3 ), 0.0f, hz, K.m_level * ( 0.4f + 0.6f * hash01( 0xC40125U, u, 4 ) ) );
        b = make_float4( hash01( 0xC40125U, u, 5 ), hash01( 0xC40125U, u, 6 ), static_cast<float>( kind ), -1.0f );
    }
    const int         kind = min( max( static_cast<int>( b.z ), 0 ), ChorusSpecies - 1 );
    const ChorusKind &K = cChorus[kind];
    const float       warm = chorusWarm( K, C.m_warmth );
    const float       callLen = K.m_pulses / ( K.m_pulseRate * ( 0.5f + 0.5f * warm ) );
    const float       period =
        fmaxf( K.m_period / warm * ( 0.85f + 0.3f * hash01( 0xC40125U, static_cast<std::uint32_t>( i ), 7 ) ),
               callLen * 1.2f );
    // What it heard: the calls of its kind that began in the last block, the other kinds calling in its band.
    float heard = 0.0f, masked = 0.0f;
    __syncthreads();
    for ( int base = 0; base < ChorusCallers; base += 256 )
    {
        tile[threadIdx.x] = src[2 * ( base + static_cast<int>( threadIdx.x ) )];
        tileB[threadIdx.x] = src[2 * ( base + static_cast<int>( threadIdx.x ) ) + 1];
        __syncthreads();
        if ( !C.m_reset )
            for ( int j = 0; j < 256; ++j )
            {
                const float4 oa = tileB[j];
                if ( oa.w < 0.0f || base + j == i ) continue;   // silent, or itself
                const float dx = oa.x - b.x, dy = oa.y - b.y;
                const float w = tile[j].w / ( 1.0f + ( dx * dx + dy * dy ) * ( 1.0f / 0.0064f ) );
                if ( static_cast<int>( oa.z ) == kind )
                {
                    if ( oa.w < dt * 1.5f ) heard += w;   // a call of its kind, begun in the last block
                }
                else if ( fabsf( __logf( tile[j].z / a.z ) ) < 0.35f )
                {
                    masked += w;   // another kind in its band
                }
            }
        __syncthreads();
    }
    // Its clock.
    float phase = a.x, waited = a.y, callT = b.w;
    if ( callT >= 0.0f ) callT += dt;
    if ( callT >= callLen ) callT = -1.0f;   // its call is over
    phase += dt / period;
    if ( heard > 0.25f && callT < 0.0f && phase < 0.92f ) phase = phase * ( 1.0f - C.m_sync ) - 0.05f * C.m_sync;
    bool begun = false, waiting = false;
    if ( phase >= 1.0f )
    {
        if ( masked > 0.6f * a.w && waited < 0.5f * period )
        {   // a louder caller of another kind fills its band: it waits for a gap
            waited += dt;
            waiting = true;
        }
        else
        {
            phase = fminf( phase - 1.0f, 0.2f );
            waited = 0.0f;
            callT = 0.0f;
            begun = true;
        }
    }
    dst[2 * i] = make_float4( phase, waited, a.z, a.w );
    dst[2 * i + 1] = make_float4( b.x, b.y, b.z, callT );
    // The block's counts in whole numbers (any order the same), gathered in [26 .. 28] for the order's kernel.
    if ( begun ) atomicAdd( &sBegun, 1 );
    if ( callT >= 0.0f ) atomicAdd( &sCalling, 1 );
    if ( waiting ) atomicAdd( &sWaiting, 1 );
    __syncthreads();
    if ( threadIdx.x == 0 )
    {
        atomicAdd( &stats[26], static_cast<float>( sBegun ) );
        atomicAdd( &stats[27], static_cast<float>( sCalling ) );
        atomicAdd( &stats[28], static_cast<float>( sWaiting ) );
    }
}

/**
 * @brief The night chorus's order (Phase 31), one CUDA block of 1024: the most numerous kind's clocks as phasors, their
 *        mean's length (1: all in step, 0: spread), summed in a fixed order; and the step's counts moved to where the
 *        host reads them, cleared for the next block.
 * @param[in]     hdr    the block header
 * @param[in]     state  the callers (this block's buffer)
 * @param[in,out] stats  [62] receives the order, [60], [61], [63] the calls begun, calling and waiting ([26 .. 28],
 *                       cleared)
 */
__global__ void __launch_bounds__( 1024 ) chorusOrderKernel( const BlockHeader *hdr, const float4 *state, float *stats )
{
    __shared__ float   re[1024], im[1024];
    __shared__ int     count[ChorusSpecies];
    const NightChorus &C = hdr->m_chorus;
    if ( !C.m_on ) return;
    const bool    odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4 *now = state + ( odd ? 0 : 2 * ChorusCallers );
    const int     t = static_cast<int>( threadIdx.x );
    if ( t < ChorusSpecies ) count[t] = 0;
    __syncthreads();
    for ( int i = t; i < ChorusCallers; i += 1024 )
        atomicAdd( &count[min( max( static_cast<int>( now[2 * i + 1].z ), 0 ), ChorusSpecies - 1 )], 1 );
    __syncthreads();
    int most = 0;
    for ( int k = 1; k < ChorusSpecies; ++k )
        if ( count[k] > count[most] ) most = k;
    float r = 0.0f, s = 0.0f;
    for ( int i = t; i < ChorusCallers; i += 1024 )
        if ( static_cast<int>( now[2 * i + 1].z ) == most )
        {
            float sn, cs;
            __sincosf( 6.2831853f * now[2 * i].x, &sn, &cs );
            r += cs;
            s += sn;
        }
    re[t] = r;
    im[t] = s;
    __syncthreads();
    for ( int h = 512; h > 0; h >>= 1 )
    {
        if ( t < h )
        {
            re[t] += re[t + h];
            im[t] += im[t + h];
        }
        __syncthreads();
    }
    if ( t == 0 )
    {
        stats[62] = count[most] > 0 ? sqrtf( re[0] * re[0] + im[0] * im[0] ) / static_cast<float>( count[most] ) : 0.0f;
        stats[60] = stats[26];
        stats[61] = stats[27];
        stats[63] = stats[28];
        stats[26] = stats[27] = stats[28] = 0.0f;
    }
}

/**
 * @brief The night chorus's calls (Phase 31), a thread a caller: within its call each pulse a raised sine of its kind's
 *        duty, the carrier gliding by its kind's sweep with two partials more; placed by where it sits (across: the
 *        pan, from near to far: the planes and the distance's loss); mixed through shared memory in a fixed order into
 *        its CUDA block's four channels.
 * @param[in]  hdr      the block header
 * @param[in]  state    the callers (this block's buffer)
 * @param[out] scratch  per CUDA block four channels of the block
 * @param[in]  maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__( ChorusThreads )
    chorusSoundKernel( const BlockHeader *hdr, const float4 *state, float *scratch, int maxN )
{
    __shared__ float   sV[ChorusThreads][MixTile + 1];
    __shared__ float   sG[ChorusThreads][4];
    const NightChorus &C = hdr->m_chorus;
    if ( !C.m_on ) return;
    const int         k = static_cast<int>( threadIdx.x );
    const int         i = static_cast<int>( blockIdx.x ) * ChorusThreads + k;
    const bool        odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4     *now = state + ( odd ? 0 : 2 * ChorusCallers );
    const float4      a = now[2 * i], b = now[2 * i + 1];
    const int         n = hdr->m_n;
    const float       fs = hdr->m_sampleRate;
    const int         kind = min( max( static_cast<int>( b.z ), 0 ), ChorusSpecies - 1 );
    const ChorusKind &K = cChorus[kind];
    const float       warm = chorusWarm( K, C.m_warmth );
    const float       rate = K.m_pulseRate * ( 0.5f + 0.5f * warm );
    const float       callLen = K.m_pulses / rate;
    {
        float gl, gr, gn, gf;
        __sincosf( 0.78539816f * ( 1.0f + 0.9f * ( 2.0f * b.x - 1.0f ) ), &gr, &gl );
        __sincosf( 1.57079633f * fminf( b.y * 1.2f, 1.0f ), &gf, &gn );
        const float g = b.w >= 0.0f ? C.m_level * a.w / ( 1.0f + 2.5f * b.y ) : 0.0f;
        sG[k][0] = g * gl * gn;
        sG[k][1] = g * gr * gn;
        sG[k][2] = g * gl * gf;
        sG[k][3] = g * gr * gf;
    }
    const float  w = 6.2831853f * a.z / fs;
    float *const out = scratch + static_cast<size_t>( blockIdx.x ) * 4 * maxN;
    for ( int t0 = 0; t0 < n; t0 += MixTile )
    {
        for ( int q = 0; q < MixTile; ++q )
        {
            float       v = 0.0f;
            const float tc = b.w + static_cast<float>( t0 + q ) / fs;   // s into its call
            if ( b.w >= 0.0f && t0 + q < n && tc < callLen )
            {
                const float pu = tc * rate;
                const float u = pu - floorf( pu );
                const float env = u < K.m_duty ? __sinf( 3.1415927f * u / K.m_duty ) : 0.0f;
                const float edge = fminf( fminf( tc, callLen - tc ) * 200.0f, 1.0f );   // 5 ms in and out
                float       ph = w * fs * ( tc + 0.5f * K.m_fm * tc * tc / callLen );
                ph -= 6.2831853f * floorf( ph * 0.15915494f );
                const float p2 = ph * K.m_r2, p3 = ph * K.m_r3;
                v = env * env * edge *
                    ( __sinf( ph ) + K.m_h2 * __sinf( p2 - 6.2831853f * floorf( p2 * 0.15915494f ) ) +
                      K.m_h3 * __sinf( p3 - 6.2831853f * floorf( p3 * 0.15915494f ) ) );
            }
            sV[k][q] = v;
        }
        __syncthreads();
        for ( int o = k; o < 4 * MixTile; o += ChorusThreads )
        {
            const int ch = o / MixTile, q = o - ch * MixTile;
            if ( t0 + q >= n ) continue;
            float acc = 0.0f;
            for ( int c = 0; c < ChorusThreads; ++c ) acc += sG[c][ch] * sV[c][q];
            out[static_cast<size_t>( ch ) * maxN + t0 + q] = acc;
        }
        __syncthreads();
    }
}

/**
 * @brief The night chorus's reduce (Phase 31): its CUDA blocks' four channels added sample by sample in their order into
 *        the night's scratch (rows 12 .. 15).
 * @param[in]  hdr      the block header
 * @param[in]  partial  per CUDA block of the chorus's sound four channels
 * @param[out] scratch  the night's scratch
 * @param[in]  maxN     the engine's block size (the scratch stride)
 */
__global__ void chorusReduceKernel( const BlockHeader *hdr, const float *partial, float *scratch, int maxN )
{
    const int t = static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x );
    const int ch = t / maxN, s = t - ch * maxN;
    if ( !hdr->m_chorus.m_on || ch >= 4 || s >= hdr->m_n ) return;
    float acc = 0.0f;
    for ( int b = 0; b < ChorusBlocks; ++b ) acc += partial[static_cast<size_t>( b ) * 4 * maxN + t];
    scratch[static_cast<size_t>( 12 + ch ) * maxN + s] = acc;
}

/**
  * @brief The night's mix (Phases 30, 31): the harp's, the thunder's, the ice's and the chorus's four channels added, in
 *        that order, into the frames' channels 24 .. 27 (near left and right, far left and right); a source at rest
 *        adds nothing.
 * @param[in]  hdr      the block header
 * @param[in]  scratch  the night's scratch
 * @param[in]  maxN     the engine's block size (the scratch stride)
 * @param[out] out      the frames
 */
__global__ void nightMixKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out )
{
    const int i = static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x );
    if ( i >= hdr->m_n ) return;
    const bool on[4] = { hdr->m_harp.m_on != 0, hdr->m_thunder.m_on != 0, hdr->m_ice.m_on != 0,
                         hdr->m_chorus.m_on != 0 };
    for ( int c = 0; c < 4; ++c )
    {
        float v = 0.0f;
        for ( int src = 0; src < 4; ++src )
            if ( on[src] ) v += scratch[static_cast<size_t>( 4 * src + c ) * maxN + i];
        out[i * CloudOutChannels + 24 + c] = v;
    }
}

}   // namespace additiv::gpu
