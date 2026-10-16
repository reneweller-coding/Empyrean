/**
 * @file Device.cuh
 * @brief What the cloud's kernels share: the block header, the layout's constants, the dice and the few readers that
 *        more than one file of gpu/ calls.
 *
 * Every .cu file of gpu/ is compiled on its own (no relocatable device code, so the compiler sees a kernel and all it
 * calls at once): what two of them call is here, inline; a table in constant memory lives in the file whose kernels
 * read it and is filled at open() through that file's upload function (Kernels.cuh).
 * @author Rene Weller
 */
#pragma once

#include "additiv/Cloud.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace additiv::gpu
{

constexpr int Chunk = 512;          ///< partials per CUDA block of the render kernel
constexpr int Threads = 256;        ///< threads per CUDA block of the update and render kernels
constexpr int ReduceLanes = 32;     ///< samples per CUDA block of the reduce kernel
constexpr int ReduceGroups = 16;    ///< threads per sample in the reduce kernel, each summing every 16th chunk
constexpr int ModalThreads = 256;   ///< threads per CUDA block of the modal kernel
constexpr int ModalPer = 4;         ///< modes a thread rings (their sum goes through the warp's reduction once)
constexpr int ModalBlocks = ModalModes / ( ModalThreads * ModalPer );   ///< CUDA blocks of the modal kernel
constexpr int ModalTile = 128;                                          ///< samples the modal kernel sums per pass
static_assert( ModalModes % ModalThreads == 0, "the modes fill whole CUDA blocks" );
constexpr int EchoN = 1024;               ///< the spectral echo's FFT size
constexpr int EchoBins = EchoN / 2 + 1;   ///< its bands, 0 .. N/2
constexpr int EchoSlots = 2048;           ///< blocks each band's delay line holds (5.4 s at 128)
constexpr int EchoThreads = EchoN / 2;    ///< threads of its one CUDA block: one butterfly each
constexpr int MeltBands = MeltN / 2;      ///< the melt's bands that live, 0 .. N/2 - 1 (0, the DC, kept silent)
constexpr int MeltThreads = 1024;         ///< threads of its one CUDA block: two bands each
static_assert( MeltBands == 2 * MeltThreads, "two bands a thread" );
constexpr int LifeCells = LifeW * LifeH;   ///< cells of the living field
static_assert( ( LifeW & ( LifeW - 1 ) ) == 0 && ( LifeH & ( LifeH - 1 ) ) == 0, "the torus wraps by masks" );
constexpr float AirShare = 0.2f;   ///< the share of a voice's partials that become its air band when Air is up
constexpr int   PhysCells = PhysW * PhysH;                  ///< cells of the physarum's trail (Phase 28)
constexpr float PhysCellsPerOctave = PhysW / 9.96578428f;   ///< its trail cells an octave (log2 1000)

// ---- Phase 27 (16.10.2026): the chime field in its wind, the dream.

constexpr int   WindCells = WindW * WindH;                      ///< cells of the wind's lattice
constexpr float WindTau = 0.53f;                                ///< the wind's relaxation time (its viscosity)
constexpr int   ChimeThreads = 256;                             ///< modes per CUDA block of the chime kernel
constexpr int   ChimeBlocks = ChimeModesTotal / ChimeThreads;   ///< CUDA blocks of the chime kernel
constexpr int   ChimeTile = 128;                                ///< samples the chime kernel sums per pass
constexpr int   ChimesAcross = 64;                              ///< chimes from left to right (32 from near to far)
constexpr int   DreamThreads = 256;                             ///< memories per CUDA block of the dream's likeness
static_assert( ChimeModesTotal % ChimeThreads == 0, "the chimes' modes fill whole CUDA blocks" );
static_assert( DreamMemories % DreamThreads == 0, "the memories fill whole CUDA blocks" );
static_assert( HallBins == 2 * DreamBands, "a dream band is two of the hall's" );

// ---- Phase 28 (16.10.2026): the weather in the wind, the bloom, the swarmalators, the physarum.

constexpr int   RainThreads = 256;                         ///< drops per CUDA block of the rain kernel
constexpr int   RainBlocks = RainSlots / RainThreads;      ///< CUDA blocks of the rain kernel
constexpr int   LeafThreads = 256;                         ///< clusters per CUDA block of the leaf kernel
constexpr int   LeafBlocks = LeafClusters / LeafThreads;   ///< CUDA blocks of the leaf kernel
constexpr int   WeatherTile = 128;                         ///< samples the rain kernel sums per pass
constexpr int   MixTile = 32;   ///< samples the leaves and the sea mix through shared memory per pass
constexpr int   WeatherBlocks = RainBlocks + LeafBlocks + 1;   ///< the weather's scratch: rain, leaves, the sea
constexpr int   BloomThreads = 256;                            ///< modes per CUDA block of the bloom kernel
constexpr int   BloomBlocks = BloomModes / BloomThreads;       ///< CUDA blocks of the bloom kernel
constexpr int   BloomTile = 128;                               ///< samples the bloom kernel sums per pass
constexpr float BloomLow = 30.0f;                              ///< the bloom's lowest mode that sounds, Hz
constexpr float BloomHigh = 18000.0f;                          ///< its highest, Hz
constexpr int   SwarmChunks = 16;                              ///< the swarm's partners split over CUDA blocks
constexpr int   SwarmThreads = 256;                            ///< swarmalators per CUDA block
constexpr int   StatsFloats = 64;                              ///< the stats brought down every block
static_assert( RainSlots % RainThreads == 0 && LeafClusters % LeafThreads == 0, "whole CUDA blocks" );
static_assert( BloomModes % BloomThreads == 0 && SwarmN % SwarmThreads == 0, "whole CUDA blocks" );
static_assert( SwarmN == 1024, "the swarm's view is reduced by one CUDA block of 1024" );

// ---- Phase 29 (16.10.2026): the topological halo, the wave room.

constexpr int HaloThreads = 1024;   ///< a halo carrier's CUDA block: a thread a component
constexpr int RoomFirTaps = 64;     ///< the wave room's low pass, down and up
constexpr int RoomHist = 128;       ///< samples of history a source keeps for the low pass
static_assert( HaloComponents <= HaloThreads, "a carrier's lattice in one CUDA block" );
static_assert( RoomFirTaps % RoomDecim == 0, "the interpolation's phases are whole" );

// ---- Phase 30 (16.10.2026): the night -- the Aeolian harp, thunder, the singing ice.

constexpr int NightStringRow = 16;                        ///< the night's scratch: the harp's strings from this row
constexpr int NightRows = NightStringRow + HarpStrings;   ///< harp, thunder, ice, chorus (four rows each), the strings
constexpr int ChorusThreads = 64;                         ///< callers per CUDA block of the chorus's sound
constexpr int ChorusBlocks = ChorusCallers / ChorusThreads;   ///< CUDA blocks of the chorus's sound
constexpr int StarThreads = 64;                               ///< modes per CUDA block of the star's sound
constexpr int StarBlocks = StarModes / StarThreads;           ///< CUDA blocks of the star's sound
static_assert( ChorusCallers % 256 == 0 && ChorusCallers % ChorusThreads == 0, "the callers fill whole CUDA blocks" );
static_assert( HarpModes == 32, "a string's modes are a warp's lanes" );
static_assert( ThunderMain % 32 == 0 && ( ThunderSegments - ThunderMain ) % ( 32 * ThunderBranches ) == 0,
               "the channel's stretches split over a warp's lanes" );
static_assert( ThunderMain <= 2048, "the main stroke's wander fits in shared memory" );
static_assert( ThunderSegments % ThunderChunks == 0, "the strike's segments split into whole chunks" );
static_assert( ( ThunderRing & ( ThunderRing - 1 ) ) == 0 && ( IceN & ( IceN - 1 ) ) == 0,
               "rings and grids wrap by masks" );
static_assert( ( IceChirps & ( IceChirps - 1 ) ) == 0 && IceChirps % 32 == 0, "the chirps' slots wrap by a mask" );
static_assert( IceN * IceN == 1024, "the ice sheet is one CUDA block, a cell a thread" );

/** @brief What every kernel needs to know about the block; uploaded in front of the voices. */
struct BlockHeader
{
    int           m_n;                ///< samples in this block
    int           m_nPrev;            ///< samples of the previous block, by which the phases advance
    float         m_sampleRate;       ///< Hz
    int           m_voices;           ///< voices in the table
    int           m_shapeCount;       ///< shapes in the bank
    int           m_landColumns;      ///< the landscape's frequency columns (0 = none)
    int           m_landRows;         ///< its time rows
    float         m_landRowSeconds;   ///< how long one row lasts
    std::uint32_t m_blockIndex;       ///< blocks launched since open(): where this block's levels go in the ring
    std::int32_t  m_phantomVoice;     ///< the GPU voice of the phantom layer (its chunks reduced apart), -1 none
    std::int32_t  m_dreamVoice;       ///< the GPU voice of the dream, -1 none
    std::uint32_t m_hpad2;            ///< likewise
    ModalHall     m_modal;            ///< the modal hall's settings for this block
    SpectralField m_field;            ///< the field and the dispersion for this block
    Binaural      m_binaural;         ///< the binaural placement for this block
    SpectralEcho  m_echo;             ///< the spectral echo for this block
    LifeField     m_life;             ///< the living landscape for this block
    PartialSync   m_sync;             ///< the partial sync for this block
    Ripple        m_ripple;           ///< the ripple for this block
    Reservoir     m_reservoir;        ///< the reservoir for this block
    Melt          m_melt;             ///< the melt for this block
    SpectralComb  m_comb;             ///< the spectral comb for this block
    Turbulence    m_turb;             ///< the energy cascade for this block
    ChimeField    m_chimes;           ///< the chime field for this block (Phase 27)
    Dream         m_dream;            ///< the dream for this block (Phase 27)
    Weather       m_weather;          ///< the weather for this block (Phase 28)
    Bloom         m_bloom;            ///< the bloom for this block (Phase 28)
    SwarmSpace    m_swarm;            ///< the swarmalators for this block (Phase 28)
    Physarum      m_phys;             ///< the physarum for this block (Phase 28)
    Halo          m_halo;             ///< the topological halo for this block (Phase 29)
    WaveRoom      m_room;             ///< the wave room for this block (Phase 29)
    AeolianHarp   m_harp;             ///< the Aeolian harp for this block (Phase 30)
    Thunder       m_thunder;          ///< thunder for this block (Phase 30)
    SingingIce    m_ice;              ///< the singing ice for this block (Phase 30)
    NightChorus   m_chorus;           ///< the night chorus for this block (Phase 31)
    StarField     m_star;             ///< the star for this block (Phase 33)
    BlackHole     m_hole;             ///< the black hole for this block (Phase 33)
    Scintillation m_scint;            ///< the scintillation for this block (Phase 34)
    Absorption    m_absorb;           ///< the absorption for this block (Phase 34)

    std::int32_t m_lifeParity;   ///< which of the two field buffers holds the field before this block's steps
    std::int32_t m_lpad0;        ///< keeps the header 16-byte aligned
    std::int32_t m_lpad1;        ///< likewise
    std::int32_t m_lpad2;        ///< likewise
};
static_assert( sizeof( BlockHeader ) % 16 == 0, "the voices behind the header stay 16-byte aligned" );
static_assert( sizeof( CloudVoice ) % 16 == 0, "CloudVoice is uploaded as an array" );
static_assert( sizeof( EnvLane ) == 64, "EnvLane keeps its layout" );
static_assert( sizeof( EnvShape ) % 16 == 0, "EnvShape is uploaded as an array" );

/**
 * @brief An integer hash with good avalanche (lowbias32, C. Wellons); the dice of the cloud.
 * @param[in] x  any value
 * @return its hash
 */
__host__ __device__ __forceinline__ std::uint32_t hash32( std::uint32_t x )
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/**
 * @brief A uniform number in [0, 1) for one partial of one note.
 * @param[in] seed  the note's seed
 * @param[in] p     the partial within the voice
 * @param[in] salt  which of the partial's dice (offset, drift rates, phases, pan, envelopes, ...)
 * @return 0 .. 1
 */
__device__ __forceinline__ float hash01( std::uint32_t seed, std::uint32_t p, std::uint32_t salt )
{
    return static_cast<float>( hash32( seed * 0x9E3779B9U ^ hash32( p * 0x85EBCA6BU + salt * 0xC2B2AE35U ) ) >> 8 ) *
           ( 1.0f / 16777216.0f );
}

/**
 * @brief The bend of an envelope segment.
 * @param[in] x  0 .. 1 through the segment
 * @param[in] c  -1 .. 1: 0 straight, positive starts slowly, negative starts fast
 * @return 0 .. 1
 */
__device__ __forceinline__ float bend( float x, float c )
{
    if ( c > 0.001f ) return __powf( x, 1.0f + 5.0f * c );
    if ( c < -0.001f ) return 1.0f - __powf( 1.0f - x, 1.0f - 5.0f * c );
    return x;
}

/**
 * @brief A frequency's band on the hall's log axis (HallBins over 20 Hz .. 20 kHz).
 * @param[in] f  Hz
 * @return the band
 */
__device__ __forceinline__ int hallBin( float f )
{
    return min( max( static_cast<int>( log2f( fmaxf( f, 20.0f ) * ( 1.0f / 20.0f ) ) *
                                       ( static_cast<float>( HallBins ) / 9.96578428f ) ),
                     0 ),
                HallBins - 1 );
}

/**
 * @brief What a partial reads from a cell of the living field, 0 .. 1: Lenia's value, or the
 *        Turing field's second chemical (V, which lives under 0.4) scaled up.
 * @param[in] mode  the field's mode
 * @param[in] c     the cell (Lenia: x; Turing: U, V)
 * @return 0 .. 1
 */
__device__ __forceinline__ float lifeValue( LifeModeE mode, float2 c )
{
    if ( mode == LifeModeE::TURING ) return fminf( c.y * 3.0f, 1.0f );
    if ( mode == LifeModeE::RAIN ) return 0.5f + 0.5f * tanhf( 3.0f * c.x );   // the water's height about its rest
    return c.x;
}

/**
 * @brief The field buffer that holds the field after this block's steps.
 * @param[in] hdr  the header
 * @param[in] a    buffer A
 * @param[in] b    buffer B
 * @return the current one
 */
__device__ __forceinline__ const float2 *lifeCurrent( const BlockHeader *hdr, const float2 *a, const float2 *b )
{
    return ( ( hdr->m_lifeParity + hdr->m_life.m_steps ) & 1 ) ? b : a;
}

}   // namespace additiv::gpu
