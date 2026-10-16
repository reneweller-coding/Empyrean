/**
 * @file Agents.cu
 * @brief The agents (Phase 28): the swarmalators the partials ride (O'Keeffe, Ha and Strogatz), and the physarum's
 *        trail its agents -- the partials of the Physarum field (Voices.cu) -- leave and follow.
 * @author Rene Weller
 */
#include "Kernels.cuh"

namespace additiv::gpu
{

/**
 * @brief The swarmalators' forces (Phase 28; O'Keeffe, Ha and Strogatz 2017, with A = B = 1), a chunk of the partners
 *        at a time: every one pulled to every other by 1 + J cos(their phases' difference), pushed off by one over their
 *        distance squared (so they keep apart), its phase pulled by sin(the difference) over the distance. The
 *        partners split into SwarmChunks chunks, one CUDA block a chunk and a tile of swarmalators -- 64 blocks (all
 *        pairs in four blocks were bound by latency on four multiprocessors: 60 us a block). The step sums the chunks.
 * @param[in]  hdr      the block header
 * @param[in]  a        the swarm (x, y, phase, -), buffer A
 * @param[in]  b        buffer B
 * @param[out] partial  per chunk and swarmalator the sums (pull across, pull from near to far, phase pull, -)
 */
__global__ void __launch_bounds__( SwarmThreads )
    swarmForceKernel( const BlockHeader *hdr, const float4 *a, const float4 *b, float4 *partial )
{
    __shared__ float4 tile[SwarmN / SwarmChunks];
    const SwarmSpace &S = hdr->m_swarm;
    if ( !S.m_on || S.m_reset ) return;
    constexpr int Per = SwarmN / SwarmChunks;
    const int     i = static_cast<int>( blockIdx.x ) * SwarmThreads + static_cast<int>( threadIdx.x );
    const int     chunk = static_cast<int>( blockIdx.y );
    const float4 *src = ( hdr->m_blockIndex & 1U ) ? b : a;
    if ( static_cast<int>( threadIdx.x ) < Per )
    {   // the chunk with its phases as cosine and sine (cos and sin of a difference then by the identities)
        const float4 o = src[chunk * Per + static_cast<int>( threadIdx.x )];
        float        os, oc;
        __sincosf( o.z, &os, &oc );
        tile[threadIdx.x] = make_float4( o.x, o.y, oc, os );
    }
    __syncthreads();
    const float4 me = src[i];
    float        ms, mc;
    __sincosf( me.z, &ms, &mc );
    float vx = 0.0f, vy = 0.0f, dth = 0.0f;
    for ( int j = 0; j < Per; ++j )
    {
        const float4 o = tile[j];
        const float  dx = o.x - me.x, dy = o.y - me.y;
        const float  inv = rsqrtf( dx * dx + dy * dy + 1.0e-4f );
        const float  c = o.z * mc + o.w * ms;   // cos(theta_j - theta_i)
        const float  s = o.w * mc - o.z * ms;   // sin(theta_j - theta_i)
        const float  pull = ( 1.0f + S.m_j * c ) * inv - inv * inv;
        vx = fmaf( dx, pull, vx );
        vy = fmaf( dy, pull, vy );
        dth = fmaf( s, inv, dth );
    }
    partial[static_cast<size_t>( chunk ) * SwarmN + static_cast<size_t>( i )] = make_float4( vx, vy, dth, 0.0f );
}

/**
 * @brief One step of the swarmalators (Phase 28): the chunks' forces summed in their order, every one moved and its
 *        phase turned at its own rate and pulled by K; the step is the block's time by Speed. Two buffers, by the
 *        block's parity.
 * @param[in]     hdr      the block header
 * @param[in,out] a        the swarm (x, y, phase, -), buffer A
 * @param[in,out] b        buffer B
 * @param[in]     partial  the forces by chunk (swarmForceKernel)
 */
__global__ void __launch_bounds__( SwarmThreads )
    swarmStepKernel( const BlockHeader *hdr, float4 *a, float4 *b, const float4 *partial )
{
    const SwarmSpace &S = hdr->m_swarm;
    if ( !S.m_on ) return;
    const int           i = static_cast<int>( blockIdx.x ) * SwarmThreads + static_cast<int>( threadIdx.x );
    const bool          odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4       *src = odd ? b : a;
    float4             *dst = odd ? a : b;
    const std::uint32_t ui = static_cast<std::uint32_t>( i );
    if ( S.m_reset )
    {   // scattered over the unit disc, at random phases
        const float r = sqrtf( hash01( 0x5A4A5A4AU, ui, 1 ) ), ang = 6.2831853f * hash01( 0x5A4A5A4AU, ui, 2 );
        dst[i] = make_float4( r * __cosf( ang ), r * __sinf( ang ), 6.2831853f * hash01( 0x5A4A5A4AU, ui, 3 ), 0.0f );
        return;
    }
    const float4 me = src[i];
    float        vx = 0.0f, vy = 0.0f, dth = 0.0f;
    for ( int c = 0; c < SwarmChunks; ++c )
    {
        const float4 p = partial[static_cast<size_t>( c ) * SwarmN + static_cast<size_t>( i )];
        vx += p.x;
        vy += p.y;
        dth += p.z;
    }
    const float h = S.m_speed * static_cast<float>( hdr->m_n ) / hdr->m_sampleRate;
    const float omega = S.m_spread * ( 2.0f * hash01( 0x5A4A5A4AU, ui, 4 ) - 1.0f );
    const float inv = 1.0f / static_cast<float>( SwarmN );
    float       th = me.z + h * ( omega + S.m_k * dth * inv );
    th -= 6.2831853f * floorf( th * 0.15915494f );
    float x = me.x + h * vx * inv, y = me.y + h * vy * inv;
    if ( !( fabsf( x ) < 100.0f && fabsf( y ) < 100.0f ) )
    {   // never lost: one gone wrong starts again near the middle
        x = 0.1f * ( 2.0f * hash01( 0x5A4A5A4AU, ui, 5 ) - 1.0f );
        y = 0.1f * ( 2.0f * hash01( 0x5A4A5A4AU, ui, 6 ) - 1.0f );
    }
    dst[i] = make_float4( x, y, th, 0.0f );
}

/**
 * @brief The swarm as the partials ride it (Phase 28): one CUDA block of SwarmN threads finds its middle and its
 *        size (the root mean square of the distances), and writes every swarmalator's place in the field, -1 .. 1
 *        across and from near to far (the swarm fills 2/3 of the field's reach), and its phase. And its orders:
 *        R, how much the phases agree, and S, how much the phase follows the angle round the middle (a phase wave).
 * @param[in]  hdr    the block header
 * @param[in]  a      the swarm, buffer A
 * @param[in]  b      buffer B
 * @param[out] view   per swarmalator (across, near to far, phase, -)
 * @param[out] stats  [22] R, [23] S
 */
__global__ void __launch_bounds__( SwarmN )
    swarmViewKernel( const BlockHeader *hdr, const float4 *a, const float4 *b, float4 *view, float *stats )
{
    __shared__ float  sx[SwarmN], sy[SwarmN], sz[SwarmN];
    const SwarmSpace &S = hdr->m_swarm;
    if ( !S.m_on ) return;
    const int    i = static_cast<int>( threadIdx.x );
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float4 me = ( odd ? a : b )[i];   // what the step just wrote
    auto         sum3 = [&]( float x, float y, float z, float &ox, float &oy, float &oz )
    {
        sx[i] = x;
        sy[i] = y;
        sz[i] = z;
        __syncthreads();
        for ( int s = SwarmN / 2; s > 0; s >>= 1 )
        {
            if ( i < s )
            {
                sx[i] += sx[i + s];
                sy[i] += sy[i + s];
                sz[i] += sz[i + s];
            }
            __syncthreads();
        }
        ox = sx[0];
        oy = sy[0];
        oz = sz[0];
        __syncthreads();
    };
    float cx, cy, unused;
    sum3( me.x, me.y, 0.0f, cx, cy, unused );
    cx *= 1.0f / SwarmN;
    cy *= 1.0f / SwarmN;
    const float dx = me.x - cx, dy = me.y - cy;
    float       r2, rs, rc;
    float       ts, tc;
    __sincosf( me.z, &ts, &tc );
    sum3( dx * dx + dy * dy, ts, tc, r2, rs, rc );
    const float radius = sqrtf( fmaxf( r2 * ( 1.0f / SwarmN ), 1.0e-8f ) );
    const float k = 1.0f / ( 1.5f * radius );
    view[i] = make_float4( fminf( fmaxf( dx * k, -1.0f ), 1.0f ), fminf( fmaxf( dy * k, -1.0f ), 1.0f ), me.z, 0.0f );
    // S: the phase against the angle round the middle, either way round.
    const float phi = atan2f( dy, dx );
    float       ps, pc, ms, mc;
    __sincosf( phi + me.z, &ps, &pc );
    __sincosf( phi - me.z, &ms, &mc );
    float a1, a2, a3, b1, b2, b3;
    sum3( ps, pc, 0.0f, a1, a2, a3 );
    sum3( ms, mc, 0.0f, b1, b2, b3 );
    if ( i == 0 )
    {
        stats[22] = sqrtf( rs * rs + rc * rc ) * ( 1.0f / SwarmN );
        stats[23] = fmaxf( sqrtf( a1 * a1 + a2 * a2 ), sqrtf( b1 * b1 + b2 * b2 ) ) * ( 1.0f / SwarmN );
    }
}

/**
 * @brief The physarum's trail (Phase 28): each block it spreads (a share to the mean of its 3 x 3 neighbourhood),
 *        fades, takes in what the agents left in the last block (whole numbers, so in any order the same) and the
 *        food at the notes' harmonics -- a line across the field at each, two cells wide, a fifth of a visit a block
 *        (stronger, it held the agents on the lines: measured 0.1 % of the power between the harmonics; so, with a
 *        sensor of 200 cents, a quarter). Two buffers, by the block's parity; the update kernel reads the one written
 *        here.
 * @param[in]     hdr      the block header
 * @param[in,out] trail    two buffers of PhysCells
 * @param[in,out] deposit  per cell what the agents left, 256 a visit (cleared here)
 */
__global__ void physTrailKernel( const BlockHeader *hdr, float *trail, unsigned int *deposit )
{
    const Physarum &P = hdr->m_phys;
    if ( !P.m_on ) return;
    const int idx = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( idx >= PhysCells ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *src = trail + ( odd ? PhysCells : 0 );
    float       *dst = trail + ( odd ? 0 : PhysCells );
    if ( P.m_fresh )
    {
        dst[idx] = 0.0f;
        deposit[idx] = 0U;
        return;
    }
    const int x = idx % PhysW, y = idx / PhysW;
    float     sum = 0.0f;
    for ( int dy = -1; dy <= 1; ++dy )
        for ( int dx = -1; dx <= 1; ++dx )
        {
            const int xx = min( max( x + dx, 0 ), PhysW - 1 ), yy = min( max( y + dy, 0 ), PhysH - 1 );
            sum += src[yy * PhysW + xx];
        }
    float v = ( 1.0f - P.m_spread ) * src[idx] + P.m_spread * sum * ( 1.0f / 9.0f );
    v = v * P.m_keep + static_cast<float>( deposit[idx] ) * ( 1.0f / 256.0f );
    const float fx = static_cast<float>( x ) + 0.5f;
    for ( int k = 0; k < min( P.m_count, PhysFood ); ++k )
    {
        const float d = fabsf( fx - P.m_food[k] );
        if ( d < 2.0f ) v += 0.2f * P.m_foodWeight[k] * ( 1.0f - 0.5f * d );
    }
    dst[idx] = v;
    deposit[idx] = 0U;
}

/**
 * @brief The physarum's trail in two numbers (Phase 28): its mean and its spread (standard deviation over the
 *        mean), one CUDA block of 1024 over the trail just written.
 * @param[in]  hdr    the block header
 * @param[in]  trail  the two buffers
 * @param[out] stats  [24] the mean, [25] the spread
 */
__global__ void __launch_bounds__( 1024 ) physStatsKernel( const BlockHeader *hdr, const float *trail, float *stats )
{
    __shared__ float s1[1024], s2[1024];
    const Physarum  &P = hdr->m_phys;
    if ( !P.m_on ) return;
    const bool   odd = ( hdr->m_blockIndex & 1U ) != 0;
    const float *t = trail + ( odd ? 0 : PhysCells );
    float        a = 0.0f, b = 0.0f;
    for ( int k = static_cast<int>( threadIdx.x ); k < PhysCells; k += 1024 )
    {
        a += t[k];
        b += t[k] * t[k];
    }
    s1[threadIdx.x] = a;
    s2[threadIdx.x] = b;
    __syncthreads();
    for ( int s = 512; s > 0; s >>= 1 )
    {
        if ( static_cast<int>( threadIdx.x ) < s )
        {
            s1[threadIdx.x] += s1[threadIdx.x + s];
            s2[threadIdx.x] += s2[threadIdx.x + s];
        }
        __syncthreads();
    }
    if ( threadIdx.x == 0 )
    {
        const float mean = s1[0] / PhysCells, var = fmaxf( s2[0] / PhysCells - mean * mean, 0.0f );
        stats[24] = mean;
        stats[25] = mean > 1.0e-9f ? sqrtf( var ) / mean : 0.0f;
    }
}

}   // namespace additiv::gpu
