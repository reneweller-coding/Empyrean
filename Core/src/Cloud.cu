/**
 * @file Cloud.cu
 * @brief The partial cloud's engine (CloudEngine): its buffers on the card, the block captured once as a CUDA graph and
 *        launched every block, the features that rest switched off in it, and what comes back.
 *
 * The kernels are in gpu/, a file a subject, each compiled on its own:
 *   - Voices.cu     the partials: update, render, reduce, and what the update reads
 *   - Hall.cu       the hall's ear and memory, the modal hall
 *   - Life.cu       the living field, the Fluid, the Islands
 *   - Weather.cu    the wind, the chimes, the rain, the leaves, the sea
 *   - Dream.cu      the dream's memory and recall
 *   - Bloom.cu      the bloom
 *   - Agents.cu     the swarmalators, the physarum
 *   - Rooms.cu      the topological halo, the wave room
 *   - Spectral.cu   the spectral echo, the melt, the instrument listening to itself
 *   - Reservoir.cu  the reservoir
 * gpu/Device.cuh holds what they share (the block header, the layout's constants, the dice), gpu/Kernels.cuh what
 * this file launches.
 * @author Rene Weller
 */
#include "additiv/Cloud.h"
#include "additiv/Spectrum.h"
#include "gpu/Kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace additiv
{
using namespace gpu;

namespace
{

/**
 * @brief Writes a failed CUDA call into `error`.
 * @param[in]  e      the call's result
 * @param[in]  what   the call, for the message
 * @param[out] error  receives the message (may be null)
 * @return true when the call succeeded
 */
bool ok( cudaError_t e, const char *what, std::string *error )
{
    if ( e == cudaSuccess ) return true;
    if ( error ) *error = std::string( what ) + ": " + cudaGetErrorString( e );
    return false;
}

}   // namespace

/** @brief The CUDA side of CloudEngine. */
struct CloudEngine::Impl
{
    CloudConfig             m_cfg;          ///< the layout
    int                     m_pcap = 0;     ///< partials per voice in the tables, a multiple of Chunk
    int                     m_chunks = 0;   ///< render CUDA blocks: voices * pcap / Chunk
    std::vector<CloudVoice> m_voices;       ///< the host's voices, edited by the caller
    /** @brief What a voice's power sum was computed for: when none of it changed, the sum is reused. */
    struct ShapeKey
    {
        float m_hz = -1.0f;            ///< fundamental
        int   m_harmonics = 0;         ///< harmonics summed
        float m_tilt = 0.0f;           ///< spectral tilt
        float m_oddEven = 0.0f;        ///< odd/even weight
        float m_brightness = 0.0f;     ///< brightness window
        float m_inharmonic = 0.0f;     ///< stretch
        float m_presenceDb = 0.0f;     ///< presence bell
        float m_distance = 0.0f;       ///< the presence's distance
        float m_lowCut = 0.0f;         ///< pad low cut
        float m_stretch = 1.0f;        ///< the harmonics' stretch exponent
        float m_shift = 0.0f;          ///< the partials' shift, Hz
        float m_horizon = 0.0f;        ///< the horizon's amount
        float m_horizonRatio = 1.0f;   ///< the horizon over the fundamental
        /**
         * @brief Equal in every field.
         * @param[in] o  the other key
         * @return true when the sum can be reused
         */
        bool operator==( const ShapeKey &o ) const
        {
            return m_hz == o.m_hz && m_harmonics == o.m_harmonics && m_tilt == o.m_tilt && m_oddEven == o.m_oddEven &&
                   m_brightness == o.m_brightness && m_inharmonic == o.m_inharmonic && m_presenceDb == o.m_presenceDb &&
                   m_distance == o.m_distance && m_lowCut == o.m_lowCut && m_stretch == o.m_stretch &&
                   m_shift == o.m_shift && m_horizon == o.m_horizon && m_horizonRatio == o.m_horizonRatio;
        }
    };
    std::vector<ShapeKey> m_shapeKey;   ///< per voice: what powerSum was made for
    std::vector<double>
        m_powerSum;   ///< per voice: the sum over its harmonics of the squared shape (tilt, odd/even, brightness, presence, low cut)
    int             m_nPrev = 0;               ///< the previous block's length, sent with the next
    int             m_shapeCount = 0;          ///< shapes in the bank on the GPU
    int             m_landColumns = 0;         ///< the landscape on the GPU: columns (0 = none)
    int             m_landRows = 0;            ///< its rows
    float           m_landRowSeconds = 0.1f;   ///< seconds per row
    int             m_device = 0;              ///< the CUDA device
    cudaDeviceProp  m_props{};                 ///< its properties
    cudaStream_t    m_stream = nullptr;        ///< the stream the graph runs on (highest priority)
    cudaEvent_t     m_evStart = nullptr;       ///< recorded before the graph
    cudaEvent_t     m_evStop = nullptr;        ///< recorded after it: wait() waits for this one
    cudaGraph_t     m_graph = nullptr;         ///< the captured block
    cudaGraphExec_t m_exec = nullptr;          ///< its executable form
    unsigned char  *m_hUpload = nullptr;       ///< pinned host: header + voices as the GPU will read them
    unsigned char  *m_dUpload = nullptr;       ///< device: the same
    float          *m_hOut = nullptr;          ///< pinned host: the finished block
    float          *m_dOut = nullptr;          ///< device: the finished block
    float          *m_dScratch = nullptr;      ///< device: per chunk a stereo block
    uint4          *m_dRun = nullptr;          ///< device: per partial start phase, step, glide
    float4 *m_dAmp = nullptr;   ///< device: per partial the four amplitudes (near L/R, far L/R) the last block ended on
    float4 *m_dGains =
        nullptr;   ///< device: per partial three entries, the near, the far and the near cosine ramps of this block
    float2 *m_dAmpQ =
        nullptr;   ///< device: per partial the near plane's cosine parts the last block ended on (binaural)
    float       *m_dEchoHist = nullptr;    ///< device: the spectral echo's input history, two rings of EchoN
    float       *m_dEchoAcc = nullptr;     ///< device: its overlap, two rings of EchoN
    float2      *m_dEchoRing = nullptr;    ///< device: its delay lines, per ear and band EchoSlots spectra
    int         *m_dEchoState = nullptr;   ///< device: its history index and frame count
    float2      *m_dFlockA = nullptr;      ///< device: the flock's state (offset, velocity), buffer A
    float2      *m_dFlockB = nullptr;      ///< device: buffer B
    float       *m_dSyncTheta = nullptr;   ///< device: per partial the sync's phase
    float       *m_dSyncX = nullptr;       ///< device: per partial its place on its circle (Swarm)
    float2      *m_dSyncOrder = nullptr;   ///< device: per voice and group the sync's mean phasors, three planes
    PartialSync  m_sync;                   ///< the host's copy of the partial sync's settings
    Ripple       m_ripple;                 ///< the host's copy of the ripple
    Reservoir    m_reservoir;              ///< the host's copy of the reservoir's settings
    Melt         m_melt;                   ///< the host's copy of the melt's settings
    SpectralComb m_comb;                   ///< the host's copy of the spectral comb
    Turbulence   m_turb;                   ///< the host's copy of the energy cascade's settings
    float       *m_dCascadeA =
        nullptr;   ///< device: per voice and harmonic its power against the spectrum's, buffer A (even blocks read it)
    float              *m_dCascadeB = nullptr;      ///< device: buffer B
    float              *m_dFilterTable = nullptr;   ///< device: per voice the transported filter, FilterBins gains
    unsigned long long *m_dHallHist = nullptr;      ///< device: per band the partials' power this block (fixed point)
    float              *m_dHallMemory = nullptr;    ///< device: the hall's memory, per band
    float              *m_dDensity = nullptr;       ///< device: the spectrum's density, per band
    float              *m_dTau = nullptr;           ///< device: per partial its own clock
    float4             *m_dOrbitA = nullptr;        ///< device: the Orbit field's state, buffer A
    float4             *m_dOrbitB = nullptr;        ///< device: buffer B
    float              *m_hHallMemory = nullptr;    ///< pinned host: the memory as the last snapshot brought it down
    float              *m_hHallUp = nullptr;        ///< pinned host: a memory to send up
    bool                m_hallUp = false;           ///< the next launch sends hHallUp up first
    bool                m_hallDown = false;         ///< hHallMemory holds a memory
    float              *m_dMeltHist = nullptr;      ///< device: the melt's input history, two rings of MeltN
    float              *m_dMeltAcc = nullptr;       ///< device: its overlap, two rings of MeltN
    float              *m_dMeltEnergy = nullptr;    ///< device: its held energy, per ear and band
    float              *m_dMeltOmega = nullptr;     ///< device: its bands' sung frequencies
    float              *m_dMeltPhase = nullptr;     ///< device: their sung phases, per ear
    float              *m_dMeltHeard = nullptr;     ///< device: their last heard phases, per ear
    int                *m_dMeltState = nullptr;     ///< device: its history index and the samples since its last frame
    float              *m_dResW = nullptr;          ///< device: the reservoir's coupling, N x N
    float              *m_dResIn = nullptr;         ///< device: its input weights
    float              *m_dResOut = nullptr;        ///< device: its readouts
    float              *m_dResX = nullptr;          ///< device: its state, two buffers
    float2             *m_dLifeA = nullptr;         ///< device: the living field, buffer A
    float2             *m_dLifeB = nullptr;         ///< device: buffer B
    float              *m_dFluidA = nullptr;        ///< device: the Fluid's lattice, nine directions a cell, buffer A
    float              *m_dFluidB = nullptr;        ///< device: buffer B
    float              *m_dIsles = nullptr;         ///< device: the Islands' grid
    float  *m_dLifeStats = nullptr;   ///< device: its mean and mean square, then what was heard (level, centroid, flux)
    float  *m_dListenHist = nullptr;          ///< device: the listening's history, two rings of EchoN
    float  *m_dListenPrev = nullptr;          ///< device: its last frame's magnitudes
    int    *m_dListenState = nullptr;         ///< device: its history index
    float  *m_hLifeStats = nullptr;           ///< pinned host: the same, brought down every block
    float2 *m_hLife = nullptr;                ///< pinned host: the field brought down with a snapshot
    std::vector<float> m_lifeCells;           ///< the field as the partials read it, for the picture
    bool               m_lifeShown = false;   ///< lifeCells holds a field
    LifeField          m_life;                ///< the host's copy of the living landscape's settings
    int                m_lifeParity = 0;      ///< which buffer holds the field before the next block's steps
    float2       *m_dSnap = nullptr;   ///< device: the picture's snapshot, every SnapStride-th partial (Hz, amplitude)
    float2       *m_hSnap = nullptr;   ///< pinned host: the snapshot brought down
    bool          m_snapAsked = false;           ///< the next launch brings it down
    bool          m_snapInFlight = false;        ///< the launched block brings it down
    bool          m_snapReady = false;           ///< a snapshot came down and was not taken yet
    Binaural      m_binaural;                    ///< the host's copy of the binaural placement
    SpectralEcho  m_echo;                        ///< the host's copy of the spectral echo's settings
    EnvShape     *m_dShapes = nullptr;           ///< device: the envelope bank, EnvShapes
    float        *m_dLand = nullptr;             ///< device: the landscape, room for LandMaxColumns * LandMaxRows
    float2       *m_dModal = nullptr;            ///< device: the modal hall's states, ModalModes
    float        *m_dGainRing = nullptr;         ///< device: per voice GainRing levels, for the dispersion
    SpectralField m_field;                       ///< the host's copy of the field and the dispersion
    std::uint32_t m_blockIndex = 0;              ///< blocks launched since open()
    float        *m_dModalScratch = nullptr;     ///< device: per modal CUDA block two channels of sums
    float        *m_dWindA = nullptr;            ///< device: the wind's lattice, nine directions a cell, buffer A
    float        *m_dWindB = nullptr;            ///< device: buffer B
    float4       *m_dChimeState = nullptr;       ///< device: per chime its clapper (angle, speed, slow wind)
    float2       *m_dChimeHit = nullptr;         ///< device: per chime this block's strike
    float2       *m_dChimeModes = nullptr;       ///< device: per chime mode its complex state
    float        *m_dChimeScratch = nullptr;     ///< device: per chime CUDA block four channels of sums
    float        *m_dDreamAcc = nullptr;         ///< device: the hall's bands gathered for the dream's next memory
    float        *m_dDreamMem = nullptr;         ///< device: the dream's memories, DreamBands x DreamMemories
    float        *m_dDreamState = nullptr;       ///< device: the dream's state, DreamBands
    float        *m_dDreamSims = nullptr;        ///< device: the memories' likenesses, two sets
    int          *m_dMeta = nullptr;             ///< device: the counters (Phase 27: memories, strikes, next slot; 28)
    float        *m_dShowers = nullptr;          ///< device: the showers' density, two buffers of WindCells
    float4       *m_dDrop = nullptr;             ///< device: per rain slot its ring
    int2         *m_dBirth = nullptr;            ///< device: per rain slot when its drop fell
    float2       *m_dDropAt = nullptr;           ///< device: per rain slot where its drop fell
    float4       *m_dLeaf = nullptr;             ///< device: per leaf cluster its ring and its slow wind
    float4       *m_dSurf = nullptr;             ///< device: per stretch of shore two entries
    float        *m_dWeatherScratch = nullptr;   ///< device: per weather CUDA block four channels of sums
    float2       *m_dBloom = nullptr;            ///< device: per bloom mode its complex state
    float        *m_dBloomBands = nullptr;       ///< device: the bloom's bands, the active modes, the energy
    float        *m_dBloomScratch = nullptr;     ///< device: per bloom CUDA block two channels of sums
    float4       *m_dSwarmA = nullptr;           ///< device: the swarmalators, buffer A
    float4       *m_dSwarmB = nullptr;           ///< device: buffer B
    float4       *m_dSwarmView = nullptr;        ///< device: the swarm as the partials ride it
    float4       *m_dSwarmForce = nullptr;       ///< device: the swarm's forces by chunk of partners
    float        *m_dPhysTrail = nullptr;        ///< device: the physarum's trail, two buffers of PhysCells
    unsigned int *m_dPhysDeposit = nullptr;      ///< device: what its agents left, per cell
    Weather       m_weather;                     ///< the host's copy of the weather's settings
    Bloom         m_bloom;                       ///< the host's copy of the bloom's settings
    SwarmSpace    m_swarm;                       ///< the host's copy of the swarmalators' settings
    Physarum      m_phys;                        ///< the host's copy of the physarum's settings and food
    float2       *m_dHalo = nullptr;             ///< device: the halo's lattices, HaloCarriers x HaloComponents
    float        *m_dRoom = nullptr;             ///< device: the wave room's three pressure fields
    float        *m_dRoomHist = nullptr;         ///< device: per source its low pass's history
    float        *m_dRoomIn = nullptr;           ///< device: per source the block at the room's rate
    float        *m_dRoomEars = nullptr;         ///< device: the ears' block at the room's rate
    float        *m_dRoomOutHist = nullptr;      ///< device: the ears' last samples, for the way up
    Halo          m_halo;                        ///< the host's copy of the halo's settings
    WaveRoom      m_room;                        ///< the host's copy of the wave room's settings
    float2       *m_dHarpModes = nullptr;        ///< device: per harp string and mode its complex amplitude
    float2       *m_dHarpWakes = nullptr;        ///< device: per harp string its wake (q, q')
    float4       *m_dThunderSeg = nullptr;       ///< device: the last strike's channel, two entries a segment
    int          *m_dThunderRing = nullptr;      ///< device: the thunder's ring, per ear ThunderRing whole numbers
    float        *m_dIceStress = nullptr;        ///< device: the ice sheet's stress, IceN x IceN
    float4       *m_dIceChirps = nullptr;        ///< device: the ice's chirps, two entries a slot
    int          *m_dIceMeta = nullptr;          ///< device: the ice's running crack and its counters
    float        *m_dNightScratch = nullptr;     ///< device: the night's scratch, NightRows rows of the block
    AeolianHarp   m_harp;                        ///< the host's copy of the harp's settings
    Thunder       m_thunder;                     ///< the host's copy of thunder's settings
    SingingIce    m_ice;                         ///< the host's copy of the ice's settings
    int           m_thunderRead = 0;             ///< where the thunder's ring is read next
    int           m_thunderOrigin = 0;           ///< where in it the last strike's moment is
    int           m_thunderChunk = ThunderChunks;   ///< the last strike's next part into the ring (ThunderChunks: done)
    bool          m_thunderWas = false;         ///< thunder was on in the last block (the ring is cleared as it begins)
    float        *m_hIceUp = nullptr;           ///< pinned host: an ice sheet's stress to send up
    float4       *m_dChorus = nullptr;          ///< device: the chorus's callers, two entries each, two buffers
    float        *m_dChorusScratch = nullptr;   ///< device: per chorus CUDA block four channels of the block
    float        *m_dMirror = nullptr;          ///< device: the time mirror's recording
    NightChorus   m_chorus;                     ///< the host's copy of the night chorus's settings
    float4       *m_dStarModes = nullptr;       ///< device: the star's modes (Phase 33)
    float4       *m_dStar = nullptr;            ///< device: per star mode its amplitudes and its phase
    float        *m_dStarScratch = nullptr;     ///< device: per star CUDA block four channels of the block
    float        *m_hStarUp = nullptr;          ///< pinned host: a star's modes to send up
    StarField     m_star;                       ///< the host's copy of the star's settings
    Scintillation m_scint;                      ///< the host's copy of the scintillation's settings (Phase 34)
    Absorption    m_absorb;                     ///< the host's copy of the absorption's settings (Phase 34)
    BlackHole     m_hole;                       ///< the host's copy of the black hole's settings
    bool          m_starUp = false;             ///< the next launch sends hStarUp up first
    bool          m_mirrorWas = false;          ///< the time mirror ran in the last block (cleared as it begins)
    bool          m_iceUp = false;              ///< the next launch sends hIceUp up first

    ChimeField m_chimes;              ///< the host's copy of the chime field's settings
    Dream      m_dream;               ///< the host's copy of the dream's settings
    int        m_phantomVoice = -1;   ///< the phantom layer's GPU voice, -1 none
    int        m_dreamVoice = -1;     ///< the dream's GPU voice, -1 none
    ModalHall  m_modal;               ///< the host's copy of the modal hall's settings
    /**
     * @brief A feature's kernels in the graph: switched off in the executable graph while the feature rests
     *        (cudaGraphNodeSetEnabled), so a block pays only for what sounds. A feature stays on three blocks after
     *        it rests, so its kernels clear their state and write their silence as they always did.
     */
    struct Gate
    {
        std::vector<cudaGraphNode_t> m_nodes;            ///< its kernels' nodes in the graph
        bool                         m_enabled = true;   ///< as the executable graph has them
        int                          m_rest = 0;         ///< blocks the feature has rested
    };
    std::array<Gate, 23> m_gates;   ///< life, wind, chimes, weather, swarm, physarum, halo, room, modal hall, bloom,
                                    ///< dream, phantom, reservoir, spectral echo, melt, harp, thunder, ice, the night's
                                    ///< mix, the chorus, the star, the scintillation, the absorption
    float m_gpuMs = 0.0f;           ///< GPU time of the last finished block
    bool  m_launched = false;       ///< a block is in flight

    /** @brief Bytes of the upload: the header and the voices. @return bytes */
    size_t uploadBytes() const
    {
        return sizeof( BlockHeader ) + sizeof( CloudVoice ) * static_cast<size_t>( m_cfg.m_voices );
    }

    /** @brief Frees whatever was allocated, in any state. */
    void release()
    {
        if ( m_stream ) cudaStreamSynchronize( m_stream );
        if ( m_exec ) cudaGraphExecDestroy( m_exec );
        if ( m_graph ) cudaGraphDestroy( m_graph );
        if ( m_evStart ) cudaEventDestroy( m_evStart );
        if ( m_evStop ) cudaEventDestroy( m_evStop );
        if ( m_stream ) cudaStreamDestroy( m_stream );
        if ( m_hUpload ) cudaFreeHost( m_hUpload );
        if ( m_hOut ) cudaFreeHost( m_hOut );
        if ( m_hSnap ) cudaFreeHost( m_hSnap );
        if ( m_hLifeStats ) cudaFreeHost( m_hLifeStats );
        if ( m_hLife ) cudaFreeHost( m_hLife );
        for ( void *p : { static_cast<void *>( m_dUpload ),
                          static_cast<void *>( m_dOut ),
                          static_cast<void *>( m_dScratch ),
                          static_cast<void *>( m_dRun ),
                          static_cast<void *>( m_dAmp ),
                          static_cast<void *>( m_dGains ),
                          static_cast<void *>( m_dShapes ),
                          static_cast<void *>( m_dLand ),
                          static_cast<void *>( m_dModal ),
                          static_cast<void *>( m_dModalScratch ),
                          static_cast<void *>( m_dGainRing ),
                          static_cast<void *>( m_dAmpQ ),
                          static_cast<void *>( m_dEchoHist ),
                          static_cast<void *>( m_dEchoAcc ),
                          static_cast<void *>( m_dEchoRing ),
                          static_cast<void *>( m_dEchoState ),
                          static_cast<void *>( m_dSnap ),
                          static_cast<void *>( m_dLifeA ),
                          static_cast<void *>( m_dLifeB ),
                          static_cast<void *>( m_dLifeStats ),
                          static_cast<void *>( m_dSyncTheta ),
                          static_cast<void *>( m_dSyncOrder ),
                          static_cast<void *>( m_dFlockA ),
                          static_cast<void *>( m_dFlockB ),
                          static_cast<void *>( m_dListenHist ),
                          static_cast<void *>( m_dListenPrev ),
                          static_cast<void *>( m_dListenState ),
                          static_cast<void *>( m_dResW ),
                          static_cast<void *>( m_dResIn ),
                          static_cast<void *>( m_dResOut ),
                          static_cast<void *>( m_dResX ),
                          static_cast<void *>( m_dMeltHist ),
                          static_cast<void *>( m_dMeltAcc ),
                          static_cast<void *>( m_dMeltEnergy ),
                          static_cast<void *>( m_dMeltOmega ),
                          static_cast<void *>( m_dMeltPhase ),
                          static_cast<void *>( m_dMeltHeard ),
                          static_cast<void *>( m_dMeltState ),
                          static_cast<void *>( m_dSyncX ),
                          static_cast<void *>( m_dFluidA ),
                          static_cast<void *>( m_dFluidB ),
                          static_cast<void *>( m_dIsles ),
                          static_cast<void *>( m_dCascadeA ),
                          static_cast<void *>( m_dCascadeB ),
                          static_cast<void *>( m_dFilterTable ),
                          static_cast<void *>( m_dHallHist ),
                          static_cast<void *>( m_dHallMemory ),
                          static_cast<void *>( m_dDensity ),
                          static_cast<void *>( m_dTau ),
                          static_cast<void *>( m_dOrbitA ),
                          static_cast<void *>( m_dOrbitB ),
                          static_cast<void *>( m_dWindA ),
                          static_cast<void *>( m_dWindB ),
                          static_cast<void *>( m_dChimeState ),
                          static_cast<void *>( m_dChimeHit ),
                          static_cast<void *>( m_dChimeModes ),
                          static_cast<void *>( m_dChimeScratch ),
                          static_cast<void *>( m_dDreamAcc ),
                          static_cast<void *>( m_dDreamMem ),
                          static_cast<void *>( m_dDreamState ),
                          static_cast<void *>( m_dDreamSims ),
                          static_cast<void *>( m_dMeta ),
                          static_cast<void *>( m_dShowers ),
                          static_cast<void *>( m_dDrop ),
                          static_cast<void *>( m_dBirth ),
                          static_cast<void *>( m_dLeaf ),
                          static_cast<void *>( m_dDropAt ),
                          static_cast<void *>( m_dSurf ),
                          static_cast<void *>( m_dWeatherScratch ),
                          static_cast<void *>( m_dBloom ),
                          static_cast<void *>( m_dBloomBands ),
                          static_cast<void *>( m_dBloomScratch ),
                          static_cast<void *>( m_dSwarmA ),
                          static_cast<void *>( m_dSwarmB ),
                          static_cast<void *>( m_dSwarmView ),
                          static_cast<void *>( m_dSwarmForce ),
                          static_cast<void *>( m_dPhysTrail ),
                          static_cast<void *>( m_dPhysDeposit ),
                          static_cast<void *>( m_dHalo ),
                          static_cast<void *>( m_dRoom ),
                          static_cast<void *>( m_dRoomHist ),
                          static_cast<void *>( m_dRoomIn ),
                          static_cast<void *>( m_dRoomEars ),
                          static_cast<void *>( m_dRoomOutHist ),
                          static_cast<void *>( m_dHarpModes ),
                          static_cast<void *>( m_dHarpWakes ),
                          static_cast<void *>( m_dThunderSeg ),
                          static_cast<void *>( m_dThunderRing ),
                          static_cast<void *>( m_dIceStress ),
                          static_cast<void *>( m_dIceChirps ),
                          static_cast<void *>( m_dIceMeta ),
                          static_cast<void *>( m_dNightScratch ),
                          static_cast<void *>( m_dChorus ),
                          static_cast<void *>( m_dChorusScratch ),
                          static_cast<void *>( m_dMirror ),
                          static_cast<void *>( m_dStarModes ),
                          static_cast<void *>( m_dStar ),
                          static_cast<void *>( m_dStarScratch ) } )
            if ( p ) cudaFree( p );
        m_exec = nullptr;
        m_graph = nullptr;
        m_evStart = m_evStop = nullptr;
        m_stream = nullptr;
        m_hUpload = m_dUpload = nullptr;
        m_hOut = m_dOut = m_dScratch = m_dLand = nullptr;
        m_dRun = nullptr;
        m_dAmp = nullptr;
        m_dGains = nullptr;
        m_dShapes = nullptr;
        m_dModal = nullptr;
        m_dModalScratch = nullptr;
        m_dGainRing = nullptr;
        m_dAmpQ = nullptr;
        m_dEchoHist = nullptr;
        m_dEchoAcc = nullptr;
        m_dEchoRing = nullptr;
        m_dEchoState = nullptr;
        m_dSnap = nullptr;
        m_hSnap = nullptr;
        m_dLifeA = m_dLifeB = nullptr;
        m_dLifeStats = nullptr;
        m_hLifeStats = nullptr;
        m_hLife = nullptr;
        m_dSyncTheta = nullptr;
        m_dSyncOrder = nullptr;
        m_dFlockA = nullptr;
        m_dFlockB = nullptr;
        m_dListenHist = nullptr;
        m_dListenPrev = nullptr;
        m_dListenState = nullptr;
        m_dResW = nullptr;
        m_dResIn = nullptr;
        m_dResOut = nullptr;
        m_dResX = nullptr;
        m_dMeltHist = m_dMeltAcc = m_dMeltEnergy = m_dMeltOmega = m_dMeltPhase = m_dMeltHeard = nullptr;
        m_dMeltState = nullptr;
        m_dSyncX = nullptr;
        m_dFluidA = m_dFluidB = nullptr;
        m_dIsles = nullptr;
        m_dCascadeA = m_dCascadeB = nullptr;
        m_dFilterTable = nullptr;
        if ( m_hHallMemory ) cudaFreeHost( m_hHallMemory );
        if ( m_hHallUp ) cudaFreeHost( m_hHallUp );
        if ( m_hIceUp ) cudaFreeHost( m_hIceUp );
        m_hIceUp = nullptr;
        if ( m_hStarUp ) cudaFreeHost( m_hStarUp );
        m_hStarUp = nullptr;
        m_starUp = false;
        m_iceUp = false;
        m_dHallHist = nullptr;
        m_dHallMemory = m_dDensity = m_dTau = nullptr;
        m_dOrbitA = m_dOrbitB = nullptr;
        m_hHallMemory = m_hHallUp = nullptr;
        m_dWindA = m_dWindB = nullptr;
        m_dChimeState = nullptr;
        m_dChimeHit = nullptr;
        m_dChimeModes = nullptr;
        m_dChimeScratch = nullptr;
        m_dDreamAcc = m_dDreamMem = m_dDreamState = m_dDreamSims = nullptr;
        m_dMeta = nullptr;
        m_dShowers = m_dWeatherScratch = m_dBloomBands = m_dBloomScratch = m_dPhysTrail = nullptr;
        m_dDrop = m_dLeaf = m_dSurf = m_dSwarmA = m_dSwarmB = m_dSwarmView = m_dSwarmForce = nullptr;
        m_dBirth = nullptr;
        m_dDropAt = nullptr;
        m_dBloom = nullptr;
        m_dPhysDeposit = nullptr;
        m_dHalo = nullptr;
        m_dRoom = m_dRoomHist = m_dRoomIn = m_dRoomEars = m_dRoomOutHist = nullptr;
        m_dHarpModes = m_dHarpWakes = nullptr;
        m_dThunderSeg = m_dIceChirps = nullptr;
        m_dThunderRing = m_dIceMeta = nullptr;
        m_dIceStress = m_dNightScratch = nullptr;
        m_dChorus = nullptr;
        m_dChorusScratch = m_dMirror = nullptr;
        m_dStarModes = m_dStar = nullptr;
        m_dStarScratch = nullptr;
    }

    /** @brief Waits for a block in flight, so the tables may be replaced. */
    void settle()
    {
        if ( m_stream ) cudaStreamSynchronize( m_stream );
        if ( m_launched )
        {
            cudaEventElapsedTime( &m_gpuMs, m_evStart, m_evStop );
            m_launched = false;
        }
    }

    /**
     * @brief Fills a voice's derived fields: how its partials divide between the harmonic clouds and
     *        the air band, the harmonics below the cutoff (with the stiff string's stretch), members
     *        per harmonic, and the scales that make the voice's RMS its level -- normalised after the
     *        whole spectral shape, as Noctuary does, so brightness, odd/even, presence and the low cut
     *        colour the voice without changing its level (the filter, the distance and the motions do).
     * @param[in]     i  the voice
     * @param[in,out] v  its upload copy, completed here
     */
    void derive( int i, CloudVoice &v )
    {
        const float fs = static_cast<float>( m_cfg.m_sampleRate );
        const float cutoff = std::min( 20000.0f, 0.43f * fs );
        const int   partials = std::clamp( v.m_partials, 1, m_cfg.m_partialsPerVoice );
        v.m_partials = partials;
        if ( v.m_kind == VoiceKindE::HALO )
        {   // the halo (Phase 29): every partial's amplitude its site's in the lattice, times the level
            v.m_harmonics = v.m_members = v.m_harmonicPartials = v.m_subClouds = 0;
            v.m_scale = v.m_level;
            v.m_airScale = 0.0f;
            return;
        }
        if ( v.m_kind != VoiceKindE::NOTE )
        {   // the phantom layer and the dream (Phase 27): every partial a sine of one amplitude, RMS `level`
            const int per = v.m_kind == VoiceKindE::DREAM ? std::max( partials / DreamBands, 1 ) : partials;
            v.m_harmonics = 0;
            v.m_members = 0;
            v.m_harmonicPartials = 0;
            v.m_subClouds = 0;
            v.m_scale = static_cast<float>( v.m_level * std::sqrt( 2.0 / per ) );
            v.m_airScale = 0.0f;
            return;
        }
        const int airPartials =
            ( v.m_air > 0.0f && partials >= 2 )
                ? std::clamp( static_cast<int>( std::lround( partials * AirShare ) ), 1, partials - 1 )
                : 0;
        const int   HP = partials - airPartials;
        const float hz = std::max( v.m_hz, 1.0f );
        // Harmonics below the cutoff: f0 h^s sqrt(1 + B h^2) + shift < cutoff -- rising in h, so the
        // last one below is found by halving.
        const float s = std::clamp( v.m_stretch, 0.5f, 2.0f );
        const float hzn = std::clamp( v.m_horizon, 0.0f, 1.0f ), R = std::max( v.m_horizonRatio, 1.0f );
        auto        count = [&]( float amount )
        {
            int  lo = 1, hi = HP;
            auto below = [&]( int h )
            { return hz * partialRatio( h, std::max( 0.0f, v.m_inharmonic ), s, amount, R ) + v.m_shift < cutoff; };
            if ( below( hi ) )
                lo = hi;
            else
                while ( hi - lo > 1 )
                {
                    const int mid = ( lo + hi ) / 2;
                    ( below( mid ) ? lo : hi ) = mid;
                }
            return lo;
        };
        int H = count( 0.0f );
        // Bent towards a horizon below the cutoff no harmonic ever reaches it: as many as unbent (at
        // least 48), the rest of them crowded under the horizon.
        if ( hzn > 0.0f ) H = std::min( count( hzn ), std::max( H, 48 ) );
        H = std::clamp( H, 1, HP );
        // Subharmonics: their clouds after the harmonics', as many members each.
        const int S = v.m_subClouds > 0 && HP > SubClouds ? SubClouds : 0;
        if ( S > 0 ) H = std::min( H, HP - S );
        v.m_subClouds = S;
        v.m_harmonics = H;
        v.m_members = std::max( HP / ( H + S ), 1 );
        v.m_harmonicPartials = HP;
        const ShapeKey key{ hz,
                            H,
                            v.m_tilt,
                            v.m_oddEven,
                            v.m_brightness,
                            v.m_inharmonic,
                            v.m_presenceDb,
                            std::clamp( v.m_distance, 0.0f, 1.0f ),
                            v.m_lowCut,
                            s,
                            v.m_shift,
                            hzn,
                            R };
        if ( !( m_shapeKey[i] == key ) )
        {
            double sum = 0.0;
            for ( int h = 1; h <= H; ++h )
            {
                const float  fh = std::max( hz * partialRatio( h, v.m_inharmonic, s, hzn, R ) + v.m_shift, 1.0f );
                const double a = static_cast<double>( harmonicShape( h, v.m_tilt, v.m_oddEven, v.m_brightness ) ) *
                                 presenceGain( fh, v.m_presenceDb, key.m_distance ) * lowCutGain( fh, v.m_lowCut );
                sum += a * a;
            }
            m_powerSum[i] = std::max( sum, 1.0e-30 );
            m_shapeKey[i] = key;
        }
        // Power of the voice: every member a sine (mean square 1/2), the shimmer's two sines add
        // their own mean square (0.7^2 + 0.3^2) / 2 times its depth squared. The envelopes are
        // the sound's own motion and are not compensated.
        const float shimmerMs = 1.0f + 0.29f * v.m_shimmer * v.m_shimmer;
        v.m_scale = static_cast<float>( v.m_level / std::sqrt( 0.5 * v.m_members * m_powerSum[i] * shimmerMs ) );
        // The air band at `air` times the voice's level; its flicker's mean square is 1 + 0.29 * 0.8^2.
        v.m_airScale =
            airPartials > 0
                ? static_cast<float>( v.m_air * v.m_level / std::sqrt( 0.5 * airPartials * ( 1.0 + 0.29 * 0.64 ) ) )
                : 0.0f;
    }
};

CloudEngine::CloudEngine() = default;
CloudEngine::~CloudEngine()
{
    close();
}

bool CloudEngine::open( const CloudConfig &config, std::string *error )
{
    close();
    const int n = config.m_blockSize;
    if ( n < 32 || n > 4096 || ( n & ( n - 1 ) ) != 0 )
    {
        if ( error ) *error = "blockSize must be a power of two, 32 .. 4096";
        return false;
    }
    if ( config.m_voices < 1 || config.m_partialsPerVoice < 1 || config.m_sampleRate < 8000.0 )
    {
        if ( error ) *error = "voices, partials and sample rate must be positive";
        return false;
    }

    auto im = std::make_unique<Impl>();
    im->m_cfg = config;
    im->m_pcap = ( config.m_partialsPerVoice + Chunk - 1 ) / Chunk * Chunk;
    im->m_chunks = config.m_voices * im->m_pcap / Chunk;
    im->m_voices.assign( config.m_voices, CloudVoice{} );
    im->m_shapeKey.assign( config.m_voices, Impl::ShapeKey{} );
    im->m_powerSum.assign( config.m_voices, 1.0 );

    int count = 0;
    if ( !ok( cudaGetDeviceCount( &count ), "cudaGetDeviceCount", error ) ) return false;
    if ( count < 1 )
    {
        if ( error ) *error = "no CUDA device";
        return false;
    }
    // The flags only take before the context exists; a second engine in the process keeps the first's.
    cudaSetDeviceFlags( config.m_spinWait ? cudaDeviceScheduleSpin : cudaDeviceScheduleBlockingSync );
    cudaGetLastError();
    Impl &m = *im;
    bool  good = ok( cudaSetDevice( m.m_device ), "cudaSetDevice", error ) &&
                 ok( cudaGetDeviceProperties( &m.m_props, m.m_device ), "cudaGetDeviceProperties", error );
    // A card this build has no code for (CMAKE_CUDA_ARCHITECTURES) says so here, by name, not at the
    // first block's launch.
    if ( cudaFuncAttributes attr{}; good && cudaFuncGetAttributes( &attr, updateKernel ) != cudaSuccess )
    {
        cudaGetLastError();
        if ( error )
            *error = std::string( m.m_props.name ) + " (compute " + std::to_string( m.m_props.major ) + "." +
                     std::to_string( m.m_props.minor ) + ") is not among the cards this build is made for";
        return false;
    }
    int lo = 0, hi = 0;
    good = good && ok( cudaDeviceGetStreamPriorityRange( &lo, &hi ), "cudaDeviceGetStreamPriorityRange", error ) &&
           ok( cudaStreamCreateWithPriority( &m.m_stream, cudaStreamNonBlocking, hi ), "cudaStreamCreateWithPriority",
               error ) &&
           ok( cudaEventCreate( &m.m_evStart ), "cudaEventCreate", error ) &&
           ok( cudaEventCreateWithFlags( &m.m_evStop, config.m_spinWait ? cudaEventDefault : cudaEventBlockingSync ),
               "cudaEventCreate", error );
    const size_t partialsTotal = static_cast<size_t>( config.m_voices ) * m.m_pcap;
    const size_t landFloats = static_cast<size_t>( LandMaxColumns ) * LandMaxRows;
    good = good &&
           ok( cudaHostAlloc( &m.m_hUpload, m.uploadBytes(), cudaHostAllocDefault ), "cudaHostAlloc upload", error ) &&
           ok( cudaHostAlloc( &m.m_hOut, sizeof( float ) * CloudOutChannels * n, cudaHostAllocDefault ),
               "cudaHostAlloc out", error ) &&
           ok( cudaMalloc( &m.m_dUpload, m.uploadBytes() ), "cudaMalloc upload", error ) &&
           ok( cudaMalloc( &m.m_dOut, sizeof( float ) * CloudOutChannels * n ), "cudaMalloc out", error ) &&
           ok( cudaMalloc( &m.m_dScratch, sizeof( float ) * CloudChannels * n * static_cast<size_t>( m.m_chunks ) ),
               "cudaMalloc scratch", error ) &&
           ok( cudaMalloc( &m.m_dRun, sizeof( uint4 ) * partialsTotal ), "cudaMalloc run", error ) &&
           ok( cudaMalloc( &m.m_dAmp, sizeof( float4 ) * partialsTotal ), "cudaMalloc amp", error ) &&
           ok( cudaMalloc( &m.m_dGains, sizeof( float4 ) * 3 * partialsTotal ), "cudaMalloc gains", error ) &&
           ok( cudaMalloc( &m.m_dAmpQ, sizeof( float2 ) * partialsTotal ), "cudaMalloc ampQ", error ) &&
           ok( cudaMalloc( &m.m_dEchoHist, sizeof( float ) * 2 * EchoN ), "cudaMalloc echo history", error ) &&
           ok( cudaMalloc( &m.m_dEchoAcc, sizeof( float ) * 2 * EchoN ), "cudaMalloc echo overlap", error ) &&
           ok( cudaMalloc( &m.m_dEchoRing, sizeof( float2 ) * 2 * EchoBins * static_cast<size_t>( EchoSlots ) ),
               "cudaMalloc echo lines", error ) &&
           ok( cudaMalloc( &m.m_dEchoState, sizeof( int ) * 2 ), "cudaMalloc echo state", error ) &&
           ok( cudaMalloc( &m.m_dSnap, sizeof( float2 ) * ( partialsTotal / SnapStride ) ), "cudaMalloc snapshot",
               error ) &&
           ok( cudaHostAlloc( &m.m_hSnap, sizeof( float2 ) * ( partialsTotal / SnapStride ), cudaHostAllocDefault ),
               "cudaHostAlloc snapshot", error ) &&
           ok( cudaMalloc( &m.m_dLifeA, sizeof( float2 ) * LifeCells ), "cudaMalloc life", error ) &&
           ok( cudaMalloc( &m.m_dLifeB, sizeof( float2 ) * LifeCells ), "cudaMalloc life", error ) &&
           ok( cudaMalloc( &m.m_dFluidA, sizeof( float ) * 9 * LifeCells ), "cudaMalloc fluid", error ) &&
           ok( cudaMalloc( &m.m_dFluidB, sizeof( float ) * 9 * LifeCells ), "cudaMalloc fluid", error ) &&
           ok( cudaMalloc( &m.m_dIsles, sizeof( float ) * IsleW * IsleH ), "cudaMalloc islands", error ) &&
           ok( cudaMalloc( &m.m_dCascadeA, sizeof( float ) * partialsTotal ), "cudaMalloc cascade", error ) &&
           ok( cudaMalloc( &m.m_dCascadeB, sizeof( float ) * partialsTotal ), "cudaMalloc cascade", error ) &&
           ok( cudaMalloc( &m.m_dFilterTable, sizeof( float ) * FilterBins * static_cast<size_t>( config.m_voices ) ),
               "cudaMalloc filter table", error ) &&
           ok( cudaMalloc( &m.m_dHallHist, sizeof( unsigned long long ) * HallBins ), "cudaMalloc hall", error ) &&
           ok( cudaMalloc( &m.m_dHallMemory, sizeof( float ) * HallBins ), "cudaMalloc hall", error ) &&
           ok( cudaMalloc( &m.m_dDensity, sizeof( float ) * HallBins ), "cudaMalloc density", error ) &&
           ok( cudaMalloc( &m.m_dTau, sizeof( float ) * partialsTotal ), "cudaMalloc clocks", error ) &&
           ok( cudaMalloc( &m.m_dOrbitA, sizeof( float4 ) * partialsTotal ), "cudaMalloc orbit", error ) &&
           ok( cudaMalloc( &m.m_dOrbitB, sizeof( float4 ) * partialsTotal ), "cudaMalloc orbit", error ) &&
           ok( cudaHostAlloc( &m.m_hHallMemory, sizeof( float ) * HallBins, cudaHostAllocDefault ),
               "cudaHostAlloc hall", error ) &&
           ok( cudaHostAlloc( &m.m_hIceUp, sizeof( float ) * IceN * IceN, cudaHostAllocDefault ), "cudaHostAlloc ice",
               error ) &&
           ok( cudaHostAlloc( &m.m_hHallUp, sizeof( float ) * HallBins, cudaHostAllocDefault ), "cudaHostAlloc hall",
               error ) &&
           ok( cudaMalloc( &m.m_dLifeStats, sizeof( float ) * StatsFloats ), "cudaMalloc life stats", error ) &&
           ok( cudaMalloc( &m.m_dResW, sizeof( float ) * ReservoirN * ReservoirN ), "cudaMalloc reservoir", error ) &&
           ok( cudaMalloc( &m.m_dResIn, sizeof( float ) * ReservoirN * ReservoirInputs ), "cudaMalloc reservoir",
               error ) &&
           ok( cudaMalloc( &m.m_dResOut, sizeof( float ) * ReservoirN * ReservoirOutputs ), "cudaMalloc reservoir",
               error ) &&
           ok( cudaMalloc( &m.m_dResX, sizeof( float ) * 2 * ReservoirN ), "cudaMalloc reservoir", error ) &&
           ok( cudaMalloc( &m.m_dMeltHist, sizeof( float ) * 2 * MeltN ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltAcc, sizeof( float ) * 2 * MeltN ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltEnergy, sizeof( float ) * 2 * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltOmega, sizeof( float ) * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltPhase, sizeof( float ) * 2 * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltHeard, sizeof( float ) * 2 * MeltBands ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dMeltState, sizeof( int ) * 2 ), "cudaMalloc melt", error ) &&
           ok( cudaMalloc( &m.m_dListenHist, sizeof( float ) * 2 * EchoN ), "cudaMalloc listen", error ) &&
           ok( cudaMalloc( &m.m_dListenPrev, sizeof( float ) * EchoBins ), "cudaMalloc listen", error ) &&
           ok( cudaMalloc( &m.m_dListenState, sizeof( int ) * 2 ), "cudaMalloc listen", error ) &&
           ok( cudaMalloc( &m.m_dSyncTheta, sizeof( float ) * partialsTotal ), "cudaMalloc sync", error ) &&
           ok( cudaMalloc( &m.m_dFlockA, sizeof( float2 ) * partialsTotal ), "cudaMalloc flock", error ) &&
           ok( cudaMalloc( &m.m_dFlockB, sizeof( float2 ) * partialsTotal ), "cudaMalloc flock", error ) &&
           ok( cudaMalloc( &m.m_dSyncOrder, sizeof( float2 ) * 3 * partialsTotal ), "cudaMalloc sync order", error ) &&
           ok( cudaMalloc( &m.m_dSyncX, sizeof( float ) * partialsTotal ), "cudaMalloc sync places", error ) &&
           ok( cudaHostAlloc( &m.m_hLifeStats, sizeof( float ) * StatsFloats, cudaHostAllocDefault ),
               "cudaHostAlloc life stats", error ) &&
           ok( cudaHostAlloc( &m.m_hLife, sizeof( float2 ) * LifeCells, cudaHostAllocDefault ), "cudaHostAlloc life",
               error ) &&
           ok( cudaMalloc( &m.m_dShapes, sizeof( EnvShape ) * EnvShapes ), "cudaMalloc shapes", error ) &&
           ok( cudaMalloc( &m.m_dLand, sizeof( float ) * landFloats ), "cudaMalloc landscape", error ) &&
           ok( cudaMalloc( &m.m_dModal, sizeof( float2 ) * ModalModes ), "cudaMalloc modal", error ) &&
           ok( cudaMalloc( &m.m_dModalScratch, sizeof( float ) * 2 * n * static_cast<size_t>( ModalBlocks ) ),
               "cudaMalloc modal scratch", error ) &&
           ok( cudaMalloc( &m.m_dGainRing, sizeof( float ) * GainRing * static_cast<size_t>( config.m_voices ) ),
               "cudaMalloc gain ring", error ) &&
           ok( cudaMalloc( &m.m_dWindA, sizeof( float ) * 9 * WindCells ), "cudaMalloc wind", error ) &&
           ok( cudaMalloc( &m.m_dWindB, sizeof( float ) * 9 * WindCells ), "cudaMalloc wind", error ) &&
           ok( cudaMalloc( &m.m_dChimeState, sizeof( float4 ) * ChimeCount ), "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dChimeHit, sizeof( float2 ) * ChimeCount ), "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dChimeModes, sizeof( float2 ) * ChimeModesTotal ), "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dChimeScratch, sizeof( float ) * 4 * n * static_cast<size_t>( ChimeBlocks ) ),
               "cudaMalloc chimes", error ) &&
           ok( cudaMalloc( &m.m_dDreamAcc, sizeof( float ) * HallBins ), "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dDreamMem, sizeof( float ) * DreamBands * static_cast<size_t>( DreamMemories ) ),
               "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dDreamState, sizeof( float ) * DreamBands ), "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dDreamSims, sizeof( float ) * 2 * DreamMemories ), "cudaMalloc dream", error ) &&
           ok( cudaMalloc( &m.m_dMeta, sizeof( int ) * 16 ), "cudaMalloc counters", error ) &&
           ok( cudaMalloc( &m.m_dShowers, sizeof( float ) * 2 * WindCells ), "cudaMalloc showers", error ) &&
           ok( cudaMalloc( &m.m_dDrop, sizeof( float4 ) * RainSlots ), "cudaMalloc rain", error ) &&
           ok( cudaMalloc( &m.m_dBirth, sizeof( int2 ) * RainSlots ), "cudaMalloc rain", error ) &&
           ok( cudaMalloc( &m.m_dDropAt, sizeof( float2 ) * RainSlots ), "cudaMalloc rain", error ) &&
           ok( cudaMalloc( &m.m_dLeaf, sizeof( float4 ) * LeafClusters ), "cudaMalloc leaves", error ) &&
           ok( cudaMalloc( &m.m_dSurf, sizeof( float4 ) * 2 * SurfSegments ), "cudaMalloc sea", error ) &&
           ok( cudaMalloc( &m.m_dWeatherScratch, sizeof( float ) * 4 * n * static_cast<size_t>( WeatherBlocks ) ),
               "cudaMalloc weather", error ) &&
           ok( cudaMalloc( &m.m_dBloom, sizeof( float2 ) * BloomModes ), "cudaMalloc bloom", error ) &&
           ok( cudaMalloc( &m.m_dBloomBands, sizeof( float ) * ( 2 * BloomBands + 2 ) ), "cudaMalloc bloom", error ) &&
           ok( cudaMalloc( &m.m_dBloomScratch, sizeof( float ) * 2 * n * static_cast<size_t>( BloomBlocks ) ),
               "cudaMalloc bloom", error ) &&
           ok( cudaMalloc( &m.m_dSwarmA, sizeof( float4 ) * SwarmN ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dSwarmB, sizeof( float4 ) * SwarmN ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dSwarmView, sizeof( float4 ) * SwarmN ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dSwarmForce, sizeof( float4 ) * SwarmN * SwarmChunks ), "cudaMalloc swarm", error ) &&
           ok( cudaMalloc( &m.m_dPhysTrail, sizeof( float ) * 2 * PhysCells ), "cudaMalloc physarum", error ) &&
           ok( cudaMalloc( &m.m_dPhysDeposit, sizeof( unsigned int ) * PhysCells ), "cudaMalloc physarum", error ) &&
           ok( cudaMalloc( &m.m_dHalo, sizeof( float2 ) * HaloCarriers * HaloComponents ), "cudaMalloc halo", error ) &&
           ok( cudaMalloc( &m.m_dRoom, sizeof( float ) * 3 * static_cast<size_t>( RoomCells ) ), "cudaMalloc room",
               error ) &&
           ok( cudaMalloc( &m.m_dRoomHist, sizeof( float ) * RoomSources * RoomHist ), "cudaMalloc room", error ) &&
           ok( cudaMalloc( &m.m_dRoomIn, sizeof( float ) * RoomSources * ( 4096 / RoomDecim ) ), "cudaMalloc room",
               error ) &&
           ok( cudaMalloc( &m.m_dRoomEars, sizeof( float ) * 2 * ( 4096 / RoomDecim ) ), "cudaMalloc room", error ) &&
           ok( cudaMalloc( &m.m_dRoomOutHist, sizeof( float ) * 2 * ( RoomFirTaps / RoomDecim ) ), "cudaMalloc room",
               error ) &&
           ok( cudaMalloc( &m.m_dHarpModes, sizeof( float2 ) * HarpStrings * HarpModes ), "cudaMalloc harp", error ) &&
           ok( cudaMalloc( &m.m_dHarpWakes, sizeof( float2 ) * HarpStrings ), "cudaMalloc harp", error ) &&
           ok( cudaMalloc( &m.m_dThunderSeg, sizeof( float4 ) * 2 * ThunderSegments ), "cudaMalloc thunder", error ) &&
           ok( cudaMalloc( &m.m_dThunderRing, sizeof( int ) * 2 * static_cast<size_t>( ThunderRing ) ),
               "cudaMalloc thunder", error ) &&
           ok( cudaMalloc( &m.m_dIceStress, sizeof( float ) * IceN * IceN ), "cudaMalloc ice", error ) &&
           ok( cudaMalloc( &m.m_dIceChirps, sizeof( float4 ) * 2 * IceChirps ), "cudaMalloc ice", error ) &&
           ok( cudaMalloc( &m.m_dIceMeta, sizeof( int ) * 8 ), "cudaMalloc ice", error ) &&
           ok( cudaMalloc( &m.m_dNightScratch, sizeof( float ) * NightRows * static_cast<size_t>( n ) ),
               "cudaMalloc night", error ) &&
           ok( cudaMalloc( &m.m_dChorus, sizeof( float4 ) * 4 * ChorusCallers ), "cudaMalloc chorus", error ) &&
           ok( cudaMalloc( &m.m_dChorusScratch, sizeof( float ) * 4 * static_cast<size_t>( n ) * ChorusBlocks ),
               "cudaMalloc chorus", error ) &&
           ok( cudaMalloc( &m.m_dMirror, sizeof( float ) * RoomMirrors * static_cast<size_t>( MirrorMax ) ),
               "cudaMalloc mirror", error ) &&
           ok( cudaMalloc( &m.m_dStarModes, sizeof( float4 ) * StarModes ), "cudaMalloc star", error ) &&
           ok( cudaMalloc( &m.m_dStar, sizeof( float4 ) * 2 * StarModes ), "cudaMalloc star", error ) &&
           ok( cudaMalloc( &m.m_dStarScratch, sizeof( float ) * 4 * static_cast<size_t>( n ) * StarBlocks ),
               "cudaMalloc star", error ) &&
           ok( cudaHostAlloc( &m.m_hStarUp, sizeof( float4 ) * StarModes, cudaHostAllocDefault ), "cudaHostAlloc star",
               error );
    // Cleared on our own stream and waited for: the stream does not synchronise with the default
    // stream, so a plain cudaMemset could still be running when the first block's update kernel
    // writes the same tables (it did: one render in three differed).
    good =
        good &&
        ok( cudaMemsetAsync( m.m_dRun, 0, sizeof( uint4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dAmp, 0, sizeof( float4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dGains, 0, sizeof( float4 ) * 3 * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dAmpQ, 0, sizeof( float2 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoHist, 0, sizeof( float ) * 2 * EchoN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoAcc, 0, sizeof( float ) * 2 * EchoN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoRing, 0, sizeof( float2 ) * 2 * EchoBins * static_cast<size_t>( EchoSlots ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dEchoState, 0, sizeof( int ) * 2, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSnap, 0, sizeof( float2 ) * ( partialsTotal / SnapStride ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLifeA, 0, sizeof( float2 ) * LifeCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLifeB, 0, sizeof( float2 ) * LifeCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dFluidA, 0, sizeof( float ) * 9 * LifeCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFluidB, 0, sizeof( float ) * 9 * LifeCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dIsles, 0, sizeof( float ) * IsleW * IsleH, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dCascadeA, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dCascadeB, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFilterTable, 0, sizeof( float ) * FilterBins * static_cast<size_t>( config.m_voices ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dHallHist, 0, sizeof( unsigned long long ) * HallBins, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDensity, 0, sizeof( float ) * HallBins, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dTau, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dOrbitA, 0, sizeof( float4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dOrbitB, 0, sizeof( float4 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dLifeStats, 0, sizeof( float ) * StatsFloats, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dListenHist, 0, sizeof( float ) * 2 * EchoN, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dListenPrev, 0, sizeof( float ) * EchoBins, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dListenState, 0, sizeof( int ) * 2, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSyncTheta, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFlockA, 0, sizeof( float2 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dFlockB, 0, sizeof( float2 ) * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dSyncOrder, 0, sizeof( float2 ) * 3 * partialsTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dSyncX, 0, sizeof( float ) * partialsTotal, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dShapes, 0, sizeof( EnvShape ) * EnvShapes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLand, 0, sizeof( float ) * landFloats, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dModal, 0, sizeof( float2 ) * ModalModes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dGainRing, 0, sizeof( float ) * GainRing * static_cast<size_t>( config.m_voices ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dWindA, 0, sizeof( float ) * 9 * WindCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dWindB, 0, sizeof( float ) * 9 * WindCells, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dChimeState, 0, sizeof( float4 ) * ChimeCount, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dChimeHit, 0, sizeof( float2 ) * ChimeCount, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dChimeModes, 0, sizeof( float2 ) * ChimeModesTotal, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDreamAcc, 0, sizeof( float ) * HallBins, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dDreamMem, 0, sizeof( float ) * DreamBands * static_cast<size_t>( DreamMemories ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dDreamState, 0, sizeof( float ) * DreamBands, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDreamSims, 0, sizeof( float ) * 2 * DreamMemories, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dMeta, 0, sizeof( int ) * 16, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dShowers, 0, sizeof( float ) * 2 * WindCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dDrop, 0, sizeof( float4 ) * RainSlots, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dBirth, 0xFF, sizeof( int2 ) * RainSlots, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dDropAt, 0, sizeof( float2 ) * RainSlots, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dLeaf, 0, sizeof( float4 ) * LeafClusters, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSurf, 0, sizeof( float4 ) * 2 * SurfSegments, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dWeatherScratch, 0, sizeof( float ) * 4 * n * static_cast<size_t>( WeatherBlocks ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dBloom, 0, sizeof( float2 ) * BloomModes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dBloomBands, 0, sizeof( float ) * ( 2 * BloomBands + 2 ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSwarmA, 0, sizeof( float4 ) * SwarmN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSwarmB, 0, sizeof( float4 ) * SwarmN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dSwarmView, 0, sizeof( float4 ) * SwarmN, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dPhysTrail, 0, sizeof( float ) * 2 * PhysCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dPhysDeposit, 0, sizeof( unsigned int ) * PhysCells, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dHalo, 0, sizeof( float2 ) * HaloCarriers * HaloComponents, m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoom, 0, sizeof( float ) * 3 * static_cast<size_t>( RoomCells ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomHist, 0, sizeof( float ) * RoomSources * RoomHist, m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomIn, 0, sizeof( float ) * RoomSources * ( 4096 / RoomDecim ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomEars, 0, sizeof( float ) * 2 * ( 4096 / RoomDecim ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dRoomOutHist, 0, sizeof( float ) * 2 * ( RoomFirTaps / RoomDecim ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dHarpModes, 0, sizeof( float2 ) * HarpStrings * HarpModes, m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dHarpWakes, 0, sizeof( float2 ) * HarpStrings, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dThunderSeg, 0, sizeof( float4 ) * 2 * ThunderSegments, m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dThunderRing, 0, sizeof( int ) * 2 * static_cast<size_t>( ThunderRing ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dIceStress, 0, sizeof( float ) * IceN * IceN, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dIceChirps, 0, sizeof( float4 ) * 2 * IceChirps, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dIceMeta, 0, sizeof( int ) * 8, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dNightScratch, 0, sizeof( float ) * NightRows * static_cast<size_t>( n ), m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dChorus, 0, sizeof( float4 ) * 4 * ChorusCallers, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dChorusScratch, 0, sizeof( float ) * 4 * static_cast<size_t>( n ) * ChorusBlocks,
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dMirror, 0, sizeof( float ) * RoomMirrors * static_cast<size_t>( MirrorMax ),
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dStarModes, 0, sizeof( float4 ) * StarModes, m.m_stream ), "cudaMemsetAsync",
            error ) &&
        ok( cudaMemsetAsync( m.m_dStar, 0, sizeof( float4 ) * 2 * StarModes, m.m_stream ), "cudaMemsetAsync", error ) &&
        ok( cudaMemsetAsync( m.m_dStarScratch, 0, sizeof( float ) * 4 * static_cast<size_t>( n ) * StarBlocks,
                             m.m_stream ),
            "cudaMemsetAsync", error ) &&
        ok( cudaStreamSynchronize( m.m_stream ), "cudaStreamSynchronize", error );
    if ( good )
    {
        // Lenia's ring kernel: a bell around half the radius, normalised (Noctuary's, on a larger torus).
        constexpr int R = LifeRadius, K = 2 * LifeRadius + 1;
        float         kernel[K * K];
        float         sum = 0.0f;
        for ( int dy = -R; dy <= R; ++dy )
            for ( int dx = -R; dx <= R; ++dx )
            {
                const float r = std::sqrt( static_cast<float>( dx * dx + dy * dy ) ) / static_cast<float>( R );
                const float d = ( r - 0.5f ) / 0.15f;
                const float k = r <= 1.0f ? std::exp( -0.5f * d * d ) : 0.0f;
                kernel[( dy + R ) * K + ( dx + R )] = k;
                sum += k;
            }
        for ( float &k : kernel ) k /= sum;
        good = ok( uploadLifeTables( kernel ), "cudaMemcpyToSymbol lenia", error );
        {   // the halo's disc (Phase 29): its sites row by row, their neighbours
            signed char xy[HaloSites][2];
            short       nbr[HaloSites][4];
            int         at[2 * HaloRadius + 1][2 * HaloRadius + 1];
            int         sites = 0;
            for ( int y = -HaloRadius; y <= HaloRadius; ++y )
                for ( int x = -HaloRadius; x <= HaloRadius; ++x )
                {
                    at[y + HaloRadius][x + HaloRadius] = -1;
                    if ( x * x + y * y <= HaloRadius * HaloRadius && sites < HaloSites )
                    {
                        xy[sites][0] = static_cast<signed char>( x );
                        xy[sites][1] = static_cast<signed char>( y );
                        at[y + HaloRadius][x + HaloRadius] = sites++;
                    }
                }
            auto siteAt = [&]( int x, int y )
            {
                if ( x < -HaloRadius || x > HaloRadius || y < -HaloRadius || y > HaloRadius ) return -1;
                return at[y + HaloRadius][x + HaloRadius];
            };
            for ( int s = 0; s < HaloSites; ++s )
            {
                const int x = xy[s][0], y = xy[s][1];
                nbr[s][0] = static_cast<short>( siteAt( x + 1, y ) );
                nbr[s][1] = static_cast<short>( siteAt( x - 1, y ) );
                nbr[s][2] = static_cast<short>( siteAt( x, y + 1 ) );
                nbr[s][3] = static_cast<short>( siteAt( x, y - 1 ) );
            }
            good = good && sites == HaloSites;
            // The wave room's low pass: a windowed sinc (Blackman) at 2.2 kHz, unity at DC.
            float        fir[RoomFirTaps];
            double       firSum = 0.0;
            const double fc = 2200.0 / config.m_sampleRate;
            for ( int k = 0; k < RoomFirTaps; ++k )
            {
                const double x = k - 0.5 * ( RoomFirTaps - 1 );
                const double s =
                    x == 0.0 ? 2.0 * fc : std::sin( 6.283185307179586 * fc * x ) / ( 3.141592653589793 * x );
                const double w = 0.42 - 0.5 * std::cos( 6.283185307179586 * k / ( RoomFirTaps - 1 ) ) +
                                 0.08 * std::cos( 12.566370614359172 * k / ( RoomFirTaps - 1 ) );
                fir[k] = static_cast<float>( s * w );
                firSum += s * w;
            }
            for ( float &h : fir ) h = static_cast<float>( h / firSum );
            // Phase 34: hydrogen's lines (five series of thirty, rising, over Pfund alpha) and the scintillation's
            // Kolmogorov screen (a power spectrum k^(-11/3) under an outer scale: amplitudes (k^2 + k0^2)^(-11/12), the
            // phases a seeded die), unit RMS.
            std::vector<double> lines;
            for ( int n1 = 1; n1 <= 5; ++n1 )
                for ( int n2 = n1 + 1; n2 <= n1 + 30; ++n2 )
                    lines.push_back( 1.0 / ( n1 * n1 ) - 1.0 / ( static_cast<double>( n2 ) * n2 ) );
            std::sort( lines.begin(), lines.end() );
            float linesF[LineCount];
            for ( int i = 0; i < LineCount; ++i )
                linesF[i] = static_cast<float>( lines[static_cast<size_t>( i )] / lines[0] );
            std::vector<float> screen( ScintScreen, 0.0f );
            {
                std::uint32_t die = 0x5C1A7u;
                auto          next = [&die]()
                {
                    die = die * 1664525u + 1013904223u;
                    return static_cast<double>( die >> 8 ) / 16777216.0;
                };
                std::vector<double> amp( ScintScreen / 2 ), ph( ScintScreen / 2 );
                for ( int k = 1; k < ScintScreen / 2; ++k )
                {
                    // von Karman: Kolmogorov's k^(-11/6) under an outer scale of 64 points (two Fresnel radii),
                    // so the small whirls carry the phase -- a pure k^(-11/6) is all its longest wave, a ramp
                    // across the Fresnel zone that only moves the pattern
                    amp[static_cast<size_t>( k )] =
                        std::pow( static_cast<double>( k ) * k + 32.0 * 32.0, -11.0 / 12.0 );
                    ph[static_cast<size_t>( k )] = 6.283185307179586 * next();
                }
                double sq = 0.0;
                for ( int x = 0; x < ScintScreen; ++x )
                {
                    double v = 0.0;
                    for ( int k = 1; k < ScintScreen / 2; ++k )
                        v += amp[static_cast<size_t>( k )] *
                             std::cos( 6.283185307179586 * k * x / ScintScreen + ph[static_cast<size_t>( k )] );
                    screen[static_cast<size_t>( x )] = static_cast<float>( v );
                    sq += v * v;
                }
                const double rms = std::sqrt( sq / ScintScreen );
                for ( float &v : screen ) v = static_cast<float>( v / std::max( rms, 1e-12 ) );
            }
            good = good && ok( uploadVoiceTables( xy, linesF, screen.data() ), "cudaMemcpyToSymbol halo", error ) &&
                   ok( uploadRoomTables( xy, nbr, fir ), "cudaMemcpyToSymbol halo, room", error );
        }
        m.m_lifeCells.assign( static_cast<size_t>( LifeCells ), 0.0f );
        reservoirInitKernel<<<static_cast<unsigned>( ( static_cast<size_t>( ReservoirN ) * ReservoirN + 255 ) / 256 ),
                              256, 0, m.m_stream>>>( m.m_dResW, m.m_dResIn, m.m_dResOut, m.m_dResX );
        good = ok( cudaStreamSynchronize( m.m_stream ), "reservoir init", error );
        std::memset( m.m_hLifeStats, 0, sizeof( float ) * StatsFloats );
        m.m_melt.m_clear = 1;                                         // the first block fills the melt's tables
        for ( int b = 0; b < HallBins; ++b ) m.m_hHallUp[b] = 1.0f;   // the hall new: every band at the average
        good =
            good && ok( cudaMemcpy( m.m_dHallMemory, m.m_hHallUp, sizeof( float ) * HallBins, cudaMemcpyHostToDevice ),
                        "cudaMemcpy hall", error );
    }
    if ( good )
    {
        std::memset( m.m_hUpload, 0, m.uploadBytes() );
        std::memset( m.m_hOut, 0, sizeof( float ) * CloudOutChannels * n );
        // The block as one graph: voices up, the levels' ring, three kernels for the voices, two for
        // the modal hall and one for the spectral echo (which hear the voices' block on the card),
        // the block (eight channels) down.
        const auto *hdr = reinterpret_cast<const BlockHeader *>( m.m_dUpload );
        const auto *voices = reinterpret_cast<const CloudVoice *>( m.m_dUpload + sizeof( BlockHeader ) );
        good = ok( cudaStreamBeginCapture( m.m_stream, cudaStreamCaptureModeThreadLocal ), "cudaStreamBeginCapture",
                   error );
        if ( good )
        {
            cudaMemcpyAsync( m.m_dUpload, m.m_hUpload, m.uploadBytes(), cudaMemcpyHostToDevice, m.m_stream );
            // The frames cleared first: a feature's kernels switched off (the gates) write nothing, and its
            // channels must read silence.
            cudaMemsetAsync( m.m_dOut, 0, sizeof( float ) * CloudOutChannels * n, m.m_stream );
            constexpr unsigned lifeGrid = ( LifeCells + 255 ) / 256;
            lifeSeedKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, m.m_dLifeA, m.m_dLifeB );
            lifeStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 0, m.m_dLifeA, m.m_dLifeB, m.m_dLand );
            lifeStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 1, m.m_dLifeA, m.m_dLifeB, m.m_dLand );
            fluidSeedKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, m.m_dFluidA, m.m_dFluidB );
            fluidStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 0, m.m_dFluidA, m.m_dFluidB, m.m_dLifeA,
                                                               m.m_dLifeB );
            fluidStepKernel<<<lifeGrid, 256, 0, m.m_stream>>>( hdr, 1, m.m_dFluidA, m.m_dFluidB, m.m_dLifeA,
                                                               m.m_dLifeB );
            islandsKernel<<<1, 1024, 0, m.m_stream>>>( hdr, m.m_dIsles, m.m_dLifeA, m.m_dLifeB );
            lifeStatsKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dLifeA, m.m_dLifeB, m.m_dLifeStats );
            windStepKernel<<<( WindCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, 0, m.m_dWindA, m.m_dWindB );
            windStepKernel<<<( WindCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, 1, m.m_dWindA, m.m_dWindB );
            chimeStepKernel<<<( ChimeCount + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dChimeState,
                                                                                 m.m_dChimeHit, m.m_dMeta );
            rainFieldKernel<<<( WindCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dShowers );
            swarmForceKernel<<<dim3( SwarmN / SwarmThreads, SwarmChunks ), SwarmThreads, 0, m.m_stream>>>(
                hdr, m.m_dSwarmA, m.m_dSwarmB, m.m_dSwarmForce );
            swarmStepKernel<<<SwarmN / SwarmThreads, SwarmThreads, 0, m.m_stream>>>( hdr, m.m_dSwarmA, m.m_dSwarmB,
                                                                                     m.m_dSwarmForce );
            swarmViewKernel<<<1, SwarmN, 0, m.m_stream>>>( hdr, m.m_dSwarmA, m.m_dSwarmB, m.m_dSwarmView,
                                                           m.m_dLifeStats );
            physTrailKernel<<<( PhysCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, m.m_dPhysTrail, m.m_dPhysDeposit );
            haloKernel<<<HaloCarriers, HaloThreads, 0, m.m_stream>>>( hdr, m.m_dHalo, m.m_dLifeStats );

            // Phase 34: the screen's intensity per band and the absorption's transmission, read by the voices.
            scintKernel<<<ScintBins, 256, 0, m.m_stream>>>( hdr );
            absorbKernel<<<AbsorbBins / 256, 256, 0, m.m_stream>>>( hdr );
            gainRingKernel<<<static_cast<unsigned>( config.m_voices ), 256, 0, m.m_stream>>>( hdr, voices,
                                                                                              m.m_dGainRing );
            cascadeKernel<<<static_cast<unsigned>( ( partialsTotal + Threads - 1 ) / Threads ), Threads, 0,
                            m.m_stream>>>( hdr, voices, m.m_pcap, m.m_dCascadeA, m.m_dCascadeB );
            filterTransportKernel<<<static_cast<unsigned>( config.m_voices ), 256, 0, m.m_stream>>>( hdr, voices,
                                                                                                     m.m_dFilterTable );
            syncOrderKernel<<<static_cast<unsigned>( ( partialsTotal + Threads - 1 ) / Threads ), Threads, 0,
                              m.m_stream>>>( hdr, voices, m.m_pcap, m.m_dSyncTheta, m.m_dSyncX, m.m_dSyncOrder,
                                             partialsTotal );
            updateKernel<<<static_cast<unsigned>( ( partialsTotal + Threads - 1 ) / Threads ), Threads, 0,
                           m.m_stream>>>( hdr, voices, m.m_pcap, m.m_dShapes, m.m_dLand, m.m_dRun, m.m_dAmp, m.m_dGains,
                                          m.m_dGainRing, m.m_dAmpQ, m.m_dSnap, m.m_dLifeA, m.m_dLifeB, m.m_dSyncTheta,
                                          m.m_dSyncX, m.m_dSyncOrder, partialsTotal, m.m_dFlockA, m.m_dFlockB,
                                          m.m_dCascadeA, m.m_dCascadeB, m.m_dFilterTable, m.m_dHallHist, m.m_dDensity,
                                          m.m_dTau, m.m_dOrbitA, m.m_dOrbitB, m.m_dDreamState, m.m_dSwarmView,
                                          m.m_dPhysTrail, m.m_dPhysDeposit, m.m_dHalo );
            hallKernel<<<1, HallBins, 0, m.m_stream>>>( hdr, m.m_dHallHist, m.m_dHallMemory, m.m_dDensity,
                                                        m.m_dDreamAcc );
            dreamSimKernel<<<DreamMemories / DreamThreads, DreamThreads, 0, m.m_stream>>>(
                hdr, m.m_dDreamAcc, m.m_dDreamMem, m.m_dDreamState, m.m_dDreamSims, m.m_dMeta );
            dreamRecallKernel<<<1, DreamBands, 0, m.m_stream>>>( hdr, m.m_dDreamAcc, m.m_dDreamMem, m.m_dDreamState,
                                                                 m.m_dDreamSims, m.m_dMeta );
            renderKernel<<<static_cast<unsigned>( m.m_chunks ), Threads, 0, m.m_stream>>>(
                hdr, voices, m.m_pcap, m.m_dRun, m.m_dGains, m.m_dScratch, n );
            reduceKernel<<<static_cast<unsigned>( CloudChannels * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ),
                           0, m.m_stream>>>( hdr, m.m_dScratch, m.m_chunks, m.m_pcap / Chunk, n, m.m_dOut );
            phantomReduceKernel<<<static_cast<unsigned>( ( n + 127 ) / 128 ), 128, 0, m.m_stream>>>(
                hdr, m.m_dScratch, m.m_pcap / Chunk, n, m.m_dOut );
            // The wave room (Phase 29): its sources from the voices' chunks, its steps, its ears.
            roomSourceKernel<<<RoomSources, 128, 0, m.m_stream>>>( hdr, m.m_dScratch, m.m_pcap / Chunk, n,
                                                                   m.m_dRoomHist, m.m_dRoomIn );
            for ( int step = 0; step < n / RoomDecim; ++step )
                roomStepKernel<<<( RoomCells + 255 ) / 256, 256, 0, m.m_stream>>>( hdr, step, m.m_dRoom, m.m_dRoomIn,
                                                                                   m.m_dRoomEars, m.m_dMirror );
            roomOutKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dRoomEars, m.m_dRoomOutHist, m.m_dOut );

            chimeKernel<<<static_cast<unsigned>( ChimeBlocks ), ChimeThreads, 0, m.m_stream>>>(
                hdr, m.m_dChimeHit, m.m_dChimeModes, m.m_dChimeScratch, n );
            chimeReduceKernel<<<static_cast<unsigned>( 4 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dChimeScratch, n, m.m_dOut );
            rainSpawnKernel<<<1, 32, 0, m.m_stream>>>( hdr, m.m_dShowers, m.m_dBirth, m.m_dDropAt, m.m_dMeta );
            rainKernel<<<RainBlocks, RainThreads, 0, m.m_stream>>>( hdr, m.m_dDropAt, m.m_dDrop, m.m_dBirth,
                                                                    m.m_dWeatherScratch, n, m.m_dMeta );
            leafKernel<<<LeafBlocks, LeafThreads, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dLeaf, m.m_dWeatherScratch, n,
                                                                    m.m_dMeta );
            surfKernel<<<1, SurfSegments, 0, m.m_stream>>>( hdr, m.m_dSurf, m.m_dWeatherScratch, n, m.m_dMeta );
            weatherReduceKernel<<<static_cast<unsigned>( 4 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                  m.m_stream>>>( hdr, m.m_dWeatherScratch, n, m.m_dOut );
            // The night (Phase 30): the harp in the wind, a strike's channel and its sixteenth into the ring, the
            // ring read, the ice's stress and its chirps, their mix.
            harpKernel<<<1, HarpStrings * 32, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dHarpModes, m.m_dHarpWakes,
                                                                m.m_dNightScratch, n, m.m_dLifeStats );
            thunderChannelKernel<<<1, 32, 0, m.m_stream>>>( hdr, m.m_dThunderSeg );
            thunderRenderKernel<<<ThunderSegments / ThunderChunks, 256, 0, m.m_stream>>>( hdr, m.m_dThunderSeg,
                                                                                          m.m_dThunderRing );
            thunderOutKernel<<<1, 128, 0, m.m_stream>>>( hdr, m.m_dThunderRing, m.m_dNightScratch, n );
            iceKernel<<<1, 1024, 0, m.m_stream>>>( hdr, m.m_dIceStress, m.m_dIceChirps, m.m_dIceMeta, m.m_dLifeStats );
            iceSingKernel<<<1, IceChirps, 0, m.m_stream>>>( hdr, m.m_dIceChirps, m.m_dNightScratch, n );
            // The night chorus (Phase 31): the callers' clocks, their order, their calls, summed.
            chorusStepKernel<<<ChorusCallers / 256, 256, 0, m.m_stream>>>( hdr, m.m_dChorus, m.m_dLifeStats );
            chorusOrderKernel<<<1, 1024, 0, m.m_stream>>>( hdr, m.m_dChorus, m.m_dLifeStats );
            chorusSoundKernel<<<ChorusBlocks, ChorusThreads, 0, m.m_stream>>>( hdr, m.m_dChorus, m.m_dChorusScratch,
                                                                               n );
            chorusReduceKernel<<<static_cast<unsigned>( ( 4 * n + 127 ) / 128 ), 128, 0, m.m_stream>>>(
                hdr, m.m_dChorusScratch, m.m_dNightScratch, n );
            nightMixKernel<<<static_cast<unsigned>( ( n + 127 ) / 128 ), 128, 0, m.m_stream>>>( hdr, m.m_dNightScratch,
                                                                                                n, m.m_dOut );
            // The star (Phase 33): its modes' amplitudes walk, they sound, added after the night's.
            starStepKernel<<<StarModes / 256, 256, 0, m.m_stream>>>( hdr, m.m_dStarModes, m.m_dStar );
            starSoundKernel<<<StarBlocks, StarThreads, 0, m.m_stream>>>( hdr, m.m_dStarModes, m.m_dStar,
                                                                         m.m_dStarScratch, n );
            spaceMixKernel<<<static_cast<unsigned>( ( n + 127 ) / 128 ), 128, 0, m.m_stream>>>( hdr, m.m_dStarScratch,
                                                                                                n, m.m_dOut );
            modalKernel<<<static_cast<unsigned>( ModalBlocks ), ModalThreads, 0, m.m_stream>>>(
                hdr, m.m_dOut, m.m_dModal, m.m_dModalScratch, n, m.m_dLand, m.m_dHallMemory );
            modalReduceKernel<<<static_cast<unsigned>( 2 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dModalScratch, n, m.m_dOut );
            bloomKernel<<<BloomBlocks, BloomThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dBloom, m.m_dBloomBands,
                                                                       m.m_dBloomScratch, n );
            bloomBandKernel<<<BloomBands + 1, 256, 0, m.m_stream>>>( hdr, m.m_dBloom, m.m_dBloomBands, m.m_dLifeStats );
            bloomReduceKernel<<<static_cast<unsigned>( 2 * n / ReduceLanes ), dim3( ReduceLanes, ReduceGroups ), 0,
                                m.m_stream>>>( hdr, m.m_dBloomScratch, n, m.m_dOut );
            spectralEchoKernel<<<1, EchoThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dEchoHist, m.m_dEchoAcc,
                                                                   m.m_dEchoRing, m.m_dEchoState );
            meltKernel<<<1, MeltThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dMeltHist, m.m_dMeltAcc, m.m_dMeltEnergy,
                                                           m.m_dMeltOmega, m.m_dMeltPhase, m.m_dMeltHeard,
                                                           m.m_dMeltState );
            listenKernel<<<1, EchoThreads, 0, m.m_stream>>>( hdr, m.m_dOut, m.m_dListenHist, m.m_dListenPrev,
                                                             m.m_dListenState, m.m_dLifeStats );
            reservoirStepKernel<<<static_cast<unsigned>( ReservoirN * 32 / 256 ), 256, 0, m.m_stream>>>(
                hdr, m.m_dResW, m.m_dResIn, m.m_dResX );
            reservoirReadKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dResOut, m.m_dResX, m.m_dLifeStats );
            phase27StatsKernel<<<1, 32, 0, m.m_stream>>>( m.m_dMeta, m.m_dLifeStats );
            phase28StatsKernel<<<1, 256, 0, m.m_stream>>>( hdr, m.m_dWindA, m.m_dMeta, m.m_dLifeStats );
            physStatsKernel<<<1, 1024, 0, m.m_stream>>>( hdr, m.m_dPhysTrail, m.m_dLifeStats );
            cudaMemcpyAsync( m.m_hLifeStats, m.m_dLifeStats, sizeof( float ) * StatsFloats, cudaMemcpyDeviceToHost,
                             m.m_stream );
            cudaMemcpyAsync( m.m_hOut, m.m_dOut, sizeof( float ) * CloudOutChannels * n, cudaMemcpyDeviceToHost,
                             m.m_stream );
            good = ok( cudaStreamEndCapture( m.m_stream, &m.m_graph ), "cudaStreamEndCapture", error ) &&
                   ok( cudaGraphInstantiate( &m.m_exec, m.m_graph, 0 ), "cudaGraphInstantiate", error );
            // The features' kernels, found in the graph by their functions, for the gates.
            if ( good )
            {
                const std::pair<const void *, int> owners[] = {
                    { reinterpret_cast<const void *>( &lifeSeedKernel ), 0 },
                    { reinterpret_cast<const void *>( &lifeStepKernel ), 0 },
                    { reinterpret_cast<const void *>( &fluidSeedKernel ), 0 },
                    { reinterpret_cast<const void *>( &fluidStepKernel ), 0 },
                    { reinterpret_cast<const void *>( &islandsKernel ), 0 },
                    { reinterpret_cast<const void *>( &lifeStatsKernel ), 0 },
                    { reinterpret_cast<const void *>( &windStepKernel ), 1 },
                    { reinterpret_cast<const void *>( &phase28StatsKernel ), 1 },
                    { reinterpret_cast<const void *>( &chimeStepKernel ), 2 },
                    { reinterpret_cast<const void *>( &chimeKernel ), 2 },
                    { reinterpret_cast<const void *>( &chimeReduceKernel ), 2 },
                    { reinterpret_cast<const void *>( &rainFieldKernel ), 3 },
                    { reinterpret_cast<const void *>( &rainSpawnKernel ), 3 },
                    { reinterpret_cast<const void *>( &rainKernel ), 3 },
                    { reinterpret_cast<const void *>( &leafKernel ), 3 },
                    { reinterpret_cast<const void *>( &surfKernel ), 3 },
                    { reinterpret_cast<const void *>( &weatherReduceKernel ), 3 },
                    { reinterpret_cast<const void *>( &swarmForceKernel ), 4 },
                    { reinterpret_cast<const void *>( &swarmStepKernel ), 4 },
                    { reinterpret_cast<const void *>( &swarmViewKernel ), 4 },
                    { reinterpret_cast<const void *>( &physTrailKernel ), 5 },
                    { reinterpret_cast<const void *>( &physStatsKernel ), 5 },
                    { reinterpret_cast<const void *>( &haloKernel ), 6 },
                    { reinterpret_cast<const void *>( &roomSourceKernel ), 7 },
                    { reinterpret_cast<const void *>( &roomStepKernel ), 7 },
                    { reinterpret_cast<const void *>( &roomOutKernel ), 7 },
                    { reinterpret_cast<const void *>( &modalKernel ), 8 },
                    { reinterpret_cast<const void *>( &modalReduceKernel ), 8 },
                    { reinterpret_cast<const void *>( &bloomKernel ), 9 },
                    { reinterpret_cast<const void *>( &bloomBandKernel ), 9 },
                    { reinterpret_cast<const void *>( &bloomReduceKernel ), 9 },
                    { reinterpret_cast<const void *>( &dreamSimKernel ), 10 },
                    { reinterpret_cast<const void *>( &dreamRecallKernel ), 10 },
                    { reinterpret_cast<const void *>( &phantomReduceKernel ), 11 },
                    { reinterpret_cast<const void *>( &reservoirStepKernel ), 12 },
                    { reinterpret_cast<const void *>( &reservoirReadKernel ), 12 },
                    { reinterpret_cast<const void *>( &spectralEchoKernel ), 13 },
                    { reinterpret_cast<const void *>( &meltKernel ), 14 },
                    { reinterpret_cast<const void *>( &harpKernel ), 15 },
                    { reinterpret_cast<const void *>( &thunderChannelKernel ), 16 },
                    { reinterpret_cast<const void *>( &thunderRenderKernel ), 16 },
                    { reinterpret_cast<const void *>( &thunderOutKernel ), 16 },
                    { reinterpret_cast<const void *>( &iceKernel ), 17 },
                    { reinterpret_cast<const void *>( &iceSingKernel ), 17 },
                    { reinterpret_cast<const void *>( &nightMixKernel ), 18 },
                    { reinterpret_cast<const void *>( &chorusStepKernel ), 19 },
                    { reinterpret_cast<const void *>( &chorusOrderKernel ), 19 },
                    { reinterpret_cast<const void *>( &chorusSoundKernel ), 19 },
                    { reinterpret_cast<const void *>( &chorusReduceKernel ), 19 },
                    { reinterpret_cast<const void *>( &starStepKernel ), 20 },
                    { reinterpret_cast<const void *>( &starSoundKernel ), 20 },
                    { reinterpret_cast<const void *>( &spaceMixKernel ), 20 },
                    { reinterpret_cast<const void *>( &scintKernel ), 21 },
                    { reinterpret_cast<const void *>( &absorbKernel ), 22 },
                };
                size_t nodeCount = 0;
                good = ok( cudaGraphGetNodes( m.m_graph, nullptr, &nodeCount ), "cudaGraphGetNodes", error );
                std::vector<cudaGraphNode_t> nodes( nodeCount );
                good =
                    good && ok( cudaGraphGetNodes( m.m_graph, nodes.data(), &nodeCount ), "cudaGraphGetNodes", error );
                for ( size_t k = 0; good && k < nodeCount; ++k )
                {
                    cudaGraphNodeType type{};
                    if ( cudaGraphNodeGetType( nodes[k], &type ) != cudaSuccess || type != cudaGraphNodeTypeKernel )
                        continue;
                    cudaKernelNodeParams p{};
                    if ( cudaGraphKernelNodeGetParams( nodes[k], &p ) != cudaSuccess ) continue;
                    for ( const auto &[fn, gate] : owners )
                        if ( p.func == fn ) m.m_gates[static_cast<size_t>( gate )].m_nodes.push_back( nodes[k] );
                }
            }
        }
    }
    if ( !good )
    {
        m.release();
        return false;
    }
    m_impl = std::move( im );
    return true;
}

void CloudEngine::close()
{
    if ( !m_impl ) return;
    m_impl->release();
    m_impl.reset();
}

const CloudConfig &CloudEngine::config() const
{
    return m_impl->m_cfg;
}

bool CloudEngine::setShapes( const EnvShape *shapes, int count )
{
    Impl &m = *m_impl;
    count = std::clamp( count, 0, EnvShapes );
    m.settle();
    if ( count > 0 && cudaMemcpyAsync( m.m_dShapes, shapes, sizeof( EnvShape ) * count, cudaMemcpyHostToDevice,
                                       m.m_stream ) != cudaSuccess )
        return false;
    if ( cudaStreamSynchronize( m.m_stream ) != cudaSuccess ) return false;
    m.m_shapeCount = count;
    return true;
}

bool CloudEngine::setLandscape( const float *amps, int columns, int rows, float rowSeconds )
{
    Impl &m = *m_impl;
    if ( columns < 2 || columns > LandMaxColumns || rows < 1 || rows > LandMaxRows || !( rowSeconds > 0.0f ) || !amps )
        return false;
    m.settle();
    if ( cudaMemcpyAsync( m.m_dLand, amps, sizeof( float ) * static_cast<size_t>( columns ) * rows,
                          cudaMemcpyHostToDevice, m.m_stream ) != cudaSuccess )
        return false;
    if ( cudaStreamSynchronize( m.m_stream ) != cudaSuccess ) return false;
    m.m_landColumns = columns;
    m.m_landRows = rows;
    m.m_landRowSeconds = rowSeconds;
    return true;
}

CloudVoice &CloudEngine::voice( int i )
{
    return m_impl->m_voices[std::clamp( i, 0, m_impl->m_cfg.m_voices - 1 )];
}

ModalHall &CloudEngine::modal()
{
    return m_impl->m_modal;
}

SpectralField &CloudEngine::field()
{
    return m_impl->m_field;
}

Binaural &CloudEngine::binaural()
{
    return m_impl->m_binaural;
}

SpectralEcho &CloudEngine::spectralEcho()
{
    return m_impl->m_echo;
}

void CloudEngine::requestSnapshot()
{
    m_impl->m_snapAsked = true;
}

LifeField &CloudEngine::life()
{
    return m_impl->m_life;
}

PartialSync &CloudEngine::sync()
{
    return m_impl->m_sync;
}

Ripple &CloudEngine::ripple()
{
    return m_impl->m_ripple;
}

Reservoir &CloudEngine::reservoir()
{
    return m_impl->m_reservoir;
}

Melt &CloudEngine::melt()
{
    return m_impl->m_melt;
}

Turbulence &CloudEngine::turbulence()
{
    return m_impl->m_turb;
}

ChimeField &CloudEngine::chimes()
{
    return m_impl->m_chimes;
}

Dream &CloudEngine::dream()
{
    return m_impl->m_dream;
}

int CloudEngine::dreamMemories() const
{
    return m_impl ? static_cast<int>( m_impl->m_hLifeStats[12] ) : 0;
}

int CloudEngine::chimeStrikes() const
{
    return m_impl ? static_cast<int>( m_impl->m_hLifeStats[13] ) : 0;
}

Weather &CloudEngine::weather()
{
    return m_impl->m_weather;
}

Bloom &CloudEngine::bloom()
{
    return m_impl->m_bloom;
}

SwarmSpace &CloudEngine::swarm()
{
    return m_impl->m_swarm;
}

Physarum &CloudEngine::physarum()
{
    return m_impl->m_phys;
}

void CloudEngine::weatherStats( float *out ) const
{
    for ( int k = 0; k < 8; ++k ) out[k] = m_impl ? m_impl->m_hLifeStats[16 + k] : 0.0f;
}

Halo &CloudEngine::halo()
{
    return m_impl->m_halo;
}

WaveRoom &CloudEngine::room()
{
    return m_impl->m_room;
}

AeolianHarp &CloudEngine::harp()
{
    return m_impl->m_harp;
}

Thunder &CloudEngine::thunder()
{
    return m_impl->m_thunder;
}

SingingIce &CloudEngine::ice()
{
    return m_impl->m_ice;
}

int CloudEngine::dreamRead( std::vector<float> &out, int most )
{
    out.clear();
    if ( !m_impl || !m_impl->m_dDreamMem ) return 0;
    Impl &m = *m_impl;
    m.settle();
    int meta[3] = { 0, 0, 0 };
    if ( cudaMemcpy( meta, m.m_dMeta, sizeof( meta ), cudaMemcpyDeviceToHost ) != cudaSuccess ) return 0;
    const int count = std::clamp( std::min( meta[0], most ), 0, DreamMemories );
    const int next = ( ( meta[2] % DreamMemories ) + DreamMemories ) % DreamMemories;
    if ( count == 0 ) return 0;
    // The newest `count` slots are one stretch before `next`, or two when it wraps: each brought down as a 2D copy
    // (the memories are stored band by band), then turned memory by memory, the newest first.
    std::vector<float> bands( static_cast<size_t>( count ) * DreamBands );
    const int          first = next - count;
    struct Stretch
    {
        int m_from;    ///< its first slot
        int m_count;   ///< its slots
        int m_at;      ///< where it lands among the copied columns
    };
    const Stretch parts[2] = { first >= 0 ? Stretch{ first, count, 0 } : Stretch{ first + DreamMemories, -first, 0 },
                               first >= 0 ? Stretch{ 0, 0, 0 } : Stretch{ 0, next, -first } };
    for ( const Stretch &s : parts )
    {
        if ( s.m_count <= 0 ) continue;
        if ( cudaMemcpy2D( bands.data() + s.m_at, sizeof( float ) * static_cast<size_t>( count ),
                           m.m_dDreamMem + s.m_from, sizeof( float ) * DreamMemories,
                           sizeof( float ) * static_cast<size_t>( s.m_count ), DreamBands,
                           cudaMemcpyDeviceToHost ) != cudaSuccess )
            return 0;
    }
    out.resize( static_cast<size_t>( count ) * DreamBands );
    for ( int j = 0; j < count; ++j )   // column count - 1 is the newest
        for ( int b = 0; b < DreamBands; ++b )
            out[static_cast<size_t>( j ) * DreamBands + b] =
                bands[static_cast<size_t>( b ) * count + static_cast<size_t>( count - 1 - j )];
    return count;
}

void CloudEngine::dreamWrite( const float *in, int count )
{
    if ( !m_impl || !m_impl->m_dDreamMem ) return;
    Impl &m = *m_impl;
    m.settle();
    count = std::clamp( count, 0, DreamMemories );
    // Stored oldest first, so that the newest is the slot before the next one, as the dream itself keeps them.
    std::vector<float> bands( static_cast<size_t>( std::max( count, 1 ) ) * DreamBands );
    for ( int j = 0; j < count; ++j )
        for ( int b = 0; b < DreamBands; ++b )
            bands[static_cast<size_t>( b ) * count + static_cast<size_t>( count - 1 - j )] =
                in[static_cast<size_t>( j ) * DreamBands + b];
    if ( count > 0 )
        cudaMemcpy2D( m.m_dDreamMem, sizeof( float ) * DreamMemories, bands.data(),
                      sizeof( float ) * static_cast<size_t>( count ), sizeof( float ) * static_cast<size_t>( count ),
                      DreamBands, cudaMemcpyHostToDevice );
    const int counts[3] = { count, 0, count % DreamMemories };
    cudaMemcpy( m.m_dMeta, &counts[0], sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( m.m_dMeta + 2, &counts[2], sizeof( int ), cudaMemcpyHostToDevice );
}

NightChorus &CloudEngine::chorus()
{
    return m_impl->m_chorus;
}

void CloudEngine::chorusStats( float *out ) const
{
    for ( int k = 0; k < 4; ++k ) out[k] = m_impl ? m_impl->m_hLifeStats[60 + k] : 0.0f;
    out[3] /= static_cast<float>( ChorusCallers );   // the waiting ones as a share
}

StarField &CloudEngine::star()
{
    return m_impl->m_star;
}

BlackHole &CloudEngine::hole()
{
    return m_impl->m_hole;
}

Scintillation &CloudEngine::scintillation()
{
    return m_impl->m_scint;
}

Absorption &CloudEngine::absorption()
{
    return m_impl->m_absorb;
}

void CloudEngine::setStarModes( const float *in, int count )
{
    if ( !m_impl || !m_impl->m_hStarUp ) return;
    const int c = std::clamp( count, 0, StarModes );
    std::memset( m_impl->m_hStarUp, 0, sizeof( float4 ) * StarModes );
    std::memcpy( m_impl->m_hStarUp, in, sizeof( float4 ) * static_cast<size_t>( c ) );
    m_impl->m_starUp = true;
}

void CloudEngine::setIceStress( const float *in )
{
    if ( !m_impl || !m_impl->m_hIceUp ) return;
    std::memcpy( m_impl->m_hIceUp, in, sizeof( float ) * IceN * IceN );
    m_impl->m_iceUp = true;
}

void CloudEngine::nightStats( float *out ) const
{
    for ( int k = 0; k < 4; ++k ) out[k] = m_impl ? m_impl->m_hLifeStats[56 + k] : 0.0f;
}

void CloudEngine::haloStats( float *energy, float *angle, float *rim ) const
{
    for ( int c = 0; c < HaloCarriers; ++c )
    {
        energy[c] = m_impl ? m_impl->m_hLifeStats[32 + c] : 0.0f;
        angle[c] = m_impl ? m_impl->m_hLifeStats[40 + c] : 0.0f;
        rim[c] = m_impl ? m_impl->m_hLifeStats[48 + c] : 0.0f;
    }
}

void CloudEngine::physarumStats( float &mean, float &spread ) const
{
    mean = m_impl ? m_impl->m_hLifeStats[24] : 0.0f;
    spread = m_impl ? m_impl->m_hLifeStats[25] : 0.0f;
}

void CloudEngine::setLayers( int phantomVoice, int dreamVoice )
{
    m_impl->m_phantomVoice = phantomVoice;
    m_impl->m_dreamVoice = dreamVoice;
}

bool CloudEngine::hallMemory( float *out ) const
{
    if ( !m_impl || !m_impl->m_hallDown ) return false;
    std::memcpy( out, m_impl->m_hHallMemory, sizeof( float ) * HallBins );
    return true;
}

void CloudEngine::setHallMemory( const float *in )
{
    if ( !m_impl ) return;
    std::memcpy( m_impl->m_hHallUp, in, sizeof( float ) * HallBins );
    m_impl->m_hallUp = true;
}

SpectralComb &CloudEngine::comb()
{
    return m_impl->m_comb;
}

void CloudEngine::reservoirOut( float *out ) const
{
    for ( int k = 0; k < ReservoirOutputs; ++k ) out[k] = m_impl->m_hLifeStats[5 + k];
}

void CloudEngine::lifeStats( float &mean, float &meanSq ) const
{
    mean = m_impl->m_hLifeStats[0];
    meanSq = m_impl->m_hLifeStats[1];
}

void CloudEngine::heard( float &levelDb, float &centroid, float &flux ) const
{
    levelDb = m_impl->m_hLifeStats[2];
    centroid = m_impl->m_hLifeStats[3];
    flux = m_impl->m_hLifeStats[4];
}

const float *CloudEngine::lifePicture() const
{
    return m_impl->m_lifeShown ? m_impl->m_lifeCells.data() : nullptr;
}

bool CloudEngine::takeSnapshot( const float *&dots, int &perVoice )
{
    Impl &m = *m_impl;
    if ( !m.m_snapReady ) return false;
    m.m_snapReady = false;
    dots = reinterpret_cast<const float *>( m.m_hSnap );
    perVoice = m.m_pcap / SnapStride;
    return true;
}

void CloudEngine::launch()
{
    Impl     &m = *m_impl;
    const int n = m.m_cfg.m_blockSize;
    auto     *hdr = reinterpret_cast<BlockHeader *>( m.m_hUpload );
    hdr->m_n = n;
    hdr->m_nPrev = m.m_nPrev;
    hdr->m_sampleRate = static_cast<float>( m.m_cfg.m_sampleRate );
    hdr->m_voices = m.m_cfg.m_voices;
    hdr->m_shapeCount = m.m_shapeCount;
    hdr->m_landColumns = m.m_landColumns;
    hdr->m_landRows = m.m_landRows;
    hdr->m_landRowSeconds = m.m_landRowSeconds;
    hdr->m_modal = m.m_modal;
    hdr->m_modal.m_tunedCount = std::clamp( m.m_modal.m_tunedCount, 0, ModalTuned );
    hdr->m_field = m.m_field;
    const bool pitches = m.m_field.m_mode == FieldModeE::SCALE ||
                         m.m_field.m_mode == FieldModeE::ORBIT;   // the scale's pitches, not attractors
    hdr->m_field.m_count = std::clamp( m.m_field.m_count, 0, pitches ? FieldPoints : FieldAttractors );
    hdr->m_field.m_dispersion = std::clamp( m.m_field.m_dispersion, -2.0f, 2.0f );
    hdr->m_blockIndex = m.m_blockIndex++;
    hdr->m_binaural = m.m_binaural;
    hdr->m_echo = m.m_echo;
    m.m_echo.m_clear = 0;   // asked for once
    hdr->m_sync = m.m_sync;
    hdr->m_ripple = m.m_ripple;
    hdr->m_reservoir = m.m_reservoir;
    hdr->m_melt = m.m_melt;
    hdr->m_comb = m.m_comb;
    hdr->m_turb = m.m_turb;
    hdr->m_chimes = m.m_chimes;
    hdr->m_chimes.m_pitchCount = std::clamp( m.m_chimes.m_pitchCount, 0, ChimePitches );
    hdr->m_chimes.m_kind = std::clamp( m.m_chimes.m_kind, 0, static_cast<int>( ChimeKindE::COUNT ) - 1 );
    m.m_chimes.m_reset = 0;   // asked for once
    hdr->m_dream = m.m_dream;
    m.m_dream.m_store = 0;   // asked for once
    m.m_dream.m_clear = 0;
    hdr->m_weather = m.m_weather;
    hdr->m_weather.m_pitchCount = std::clamp( m.m_weather.m_pitchCount, 0, RainPitches );
    hdr->m_weather.m_wind = ( m.m_weather.m_wind || m.m_chimes.m_on || m.m_harp.m_on ) ? 1 : 0;
    m.m_weather.m_reset = 0;   // asked for once
    m.m_weather.m_rainFresh = 0;
    m.m_weather.m_windFresh = 0;
    hdr->m_bloom = m.m_bloom;
    hdr->m_swarm = m.m_swarm;
    m.m_swarm.m_reset = 0;   // asked for once
    hdr->m_phys = m.m_phys;
    hdr->m_phys.m_on = m.m_field.m_mode == FieldModeE::PHYSARUM ? m.m_phys.m_on : 0;
    hdr->m_phys.m_count = std::clamp( m.m_phys.m_count, 0, PhysFood );
    m.m_phys.m_fresh = 0;   // asked for once
    hdr->m_halo = m.m_halo;
    m.m_halo.m_clear = 0;   // asked for once
    hdr->m_room = m.m_room;
    hdr->m_room.m_count = std::clamp( m.m_room.m_count, 0, RoomSources );
    hdr->m_room.m_nx = std::clamp( m.m_room.m_nx, 3, RoomNX );
    hdr->m_room.m_ny = std::clamp( m.m_room.m_ny, 3, RoomNY );
    hdr->m_room.m_nz = std::clamp( m.m_room.m_nz, 3, RoomNZ );
    m.m_room.m_clear = 0;   // asked for once
    hdr->m_harp = m.m_harp;
    hdr->m_harp.m_x = std::clamp( m.m_harp.m_x, 0, WindW - 1 );
    hdr->m_harp.m_y = std::clamp( m.m_harp.m_y, 0, WindH - 1 );
    m.m_harp.m_clear = 0;   // asked for once
    {   // thunder: the ring read on as the blocks pass; a strike's channel laid out, then sent in over ThunderChunks
        // blocks (a strike asked for while one is still on its way waits for it)
        hdr->m_thunder = m.m_thunder;
        hdr->m_thunder.m_strike = 0;
        hdr->m_thunder.m_read = m.m_thunderRead;
        m.m_thunderRead = ( m.m_thunderRead + n ) & ( ThunderRing - 1 );
        if ( m.m_thunder.m_on && m.m_thunder.m_strike && m.m_thunderChunk >= ThunderChunks )
        {
            hdr->m_thunder.m_strike = 1;
            m.m_thunderOrigin = hdr->m_thunder.m_read;
            m.m_thunderChunk = 0;
            m.m_thunder.m_strike = 0;   // asked for once
        }
        hdr->m_thunder.m_origin = m.m_thunderOrigin;
        hdr->m_thunder.m_chunk = m.m_thunderChunk < ThunderChunks ? m.m_thunderChunk++ : -1;
        if ( !m.m_thunder.m_on ) m.m_thunderChunk = ThunderChunks;   // a strike on its way is dropped with the ring
        if ( m.m_thunder.m_on && !m.m_thunderWas )                   // begins from a silent ring
            cudaMemsetAsync( m.m_dThunderRing, 0, sizeof( int ) * 2 * static_cast<size_t>( ThunderRing ), m.m_stream );
        m.m_thunderWas = m.m_thunder.m_on != 0;
    }
    hdr->m_ice = m.m_ice;
    hdr->m_chorus = m.m_chorus;
    hdr->m_chorus.m_pitchCount = std::clamp( m.m_chorus.m_pitchCount, 0, ChorusPitches );
    hdr->m_chorus.m_kind = std::clamp( m.m_chorus.m_kind, 0, static_cast<int>( ChorusKindE::COUNT ) - 1 );
    m.m_chorus.m_reset = 0;   // asked for once
    hdr->m_star = m.m_star;   // the star and the black hole (Phase 33)
    hdr->m_star.m_count = std::clamp( m.m_star.m_count, 0, StarModes );
    m.m_star.m_reset = 0;   // asked for once
    hdr->m_hole = m.m_hole;
    hdr->m_scint = m.m_scint;   // Phase 34
    hdr->m_scint.m_offset -= std::floor( m.m_scint.m_offset / ScintScreen ) * ScintScreen;
    hdr->m_absorb = m.m_absorb;
    hdr->m_hole.m_start = std::max( m.m_hole.m_start, 1.5f );
    {   // the time mirror (Phase 31): its window within bounds; it begins from silent recordings
        const bool mirror = m.m_room.m_on && m.m_room.m_mirrorOn;
        hdr->m_room.m_mirrorOn = mirror ? 1 : 0;
        hdr->m_room.m_mirrorLen = std::clamp( m.m_room.m_mirrorLen, 1, MirrorMax );
        hdr->m_room.m_mirrorPos = std::clamp( m.m_room.m_mirrorPos, 0, hdr->m_room.m_mirrorLen - 1 );
        hdr->m_room.m_mirrorBuf &= 1;
        hdr->m_room.m_mirrorGain = std::max( m.m_room.m_mirrorGain, 0.0f );
        if ( mirror && !m.m_mirrorWas )
            cudaMemsetAsync( m.m_dMirror, 0, sizeof( float ) * RoomMirrors * static_cast<size_t>( MirrorMax ),
                             m.m_stream );
        m.m_mirrorWas = mirror;
    }
    m.m_ice.m_reset = 0;   // asked for once
    m.m_ice.m_crack = 0;
    hdr->m_phantomVoice = m.m_phantomVoice >= 0 && m.m_phantomVoice < m.m_cfg.m_voices ? m.m_phantomVoice : -1;
    hdr->m_dreamVoice = m.m_dreamVoice >= 0 && m.m_dreamVoice < m.m_cfg.m_voices ? m.m_dreamVoice : -1;
    m.m_melt.m_clear = 0;   // asked for once
    hdr->m_life = m.m_life;
    hdr->m_life.m_steps = m.m_life.m_mode == LifeModeE::OFF ? 0 : std::clamp( m.m_life.m_steps, 0, 2 );
    hdr->m_lifeParity = m.m_lifeParity;
    m.m_lifeParity = ( m.m_lifeParity + hdr->m_life.m_steps ) & 1;   // where the field is after this block
    m.m_life.m_reseed = 0;                                           // asked for once
    auto       *up = reinterpret_cast<CloudVoice *>( m.m_hUpload + sizeof( BlockHeader ) );
    const float dt = static_cast<float>( n / m.m_cfg.m_sampleRate );
    for ( int i = 0; i < m.m_cfg.m_voices; ++i )
    {
        CloudVoice &v = m.m_voices[i];
        if ( v.m_restart ) v.m_time = 0.0f;
        up[i] = v;
        if ( v.m_active )
        {
            m.derive( i, up[i] );
            v.m_harmonics = up[i].m_harmonics;   // back to the host's copy, for logs and pictures
            v.m_members = up[i].m_members;
            v.m_scale = up[i].m_scale;
        }
        v.m_restart = 0;
        if ( v.m_active ) v.m_time += dt;
    }
    if ( m.m_starUp )
    {   // a star's modes: up before the block
        cudaMemcpyAsync( m.m_dStarModes, m.m_hStarUp, sizeof( float4 ) * StarModes, cudaMemcpyHostToDevice,
                         m.m_stream );
        m.m_starUp = false;
    }
    if ( m.m_iceUp )
    {   // an ice sheet: up before the block
        cudaMemcpyAsync( m.m_dIceStress, m.m_hIceUp, sizeof( float ) * IceN * IceN, cudaMemcpyHostToDevice,
                         m.m_stream );
        m.m_iceUp = false;
    }
    if ( m.m_hallUp )
    {   // a memory loaded (or a new hall): up before the block
        cudaMemcpyAsync( m.m_dHallMemory, m.m_hHallUp, sizeof( float ) * HallBins, cudaMemcpyHostToDevice, m.m_stream );
        m.m_hallUp = false;
    }
    // The gates: a resting feature's kernels switched off in the graph (on again at once when it wakes).
    {
        const int  pv = hdr->m_phantomVoice;
        const bool want[23] = { hdr->m_life.m_mode != LifeModeE::OFF,
                                hdr->m_weather.m_wind != 0,
                                hdr->m_chimes.m_on != 0,
                                hdr->m_weather.m_rainOn || hdr->m_weather.m_leavesOn || hdr->m_weather.m_surfOn,
                                hdr->m_swarm.m_on != 0,
                                hdr->m_phys.m_on != 0,
                                hdr->m_halo.m_on != 0,
                                hdr->m_room.m_on != 0,
                                hdr->m_modal.m_on != 0,
                                hdr->m_bloom.m_on != 0,
                                hdr->m_dream.m_on != 0,
                                pv >= 0 && m.m_voices[static_cast<size_t>( pv )].m_active != 0,
                                hdr->m_reservoir.m_on != 0,
                                hdr->m_echo.m_on != 0,
                                hdr->m_melt.m_on != 0,
                                hdr->m_harp.m_on != 0,
                                hdr->m_thunder.m_on != 0,
                                hdr->m_ice.m_on != 0,
                                hdr->m_harp.m_on || hdr->m_thunder.m_on || hdr->m_ice.m_on || hdr->m_chorus.m_on,
                                hdr->m_chorus.m_on != 0,
                                hdr->m_star.m_on != 0,
                                hdr->m_scint.m_on != 0,
                                hdr->m_absorb.m_on != 0 };
        for ( size_t g = 0; g < m.m_gates.size(); ++g )
        {
            Impl::Gate &gate = m.m_gates[g];
            gate.m_rest = want[g] ? 0 : std::min( gate.m_rest + 1, 1000 );
            const bool on = want[g] || gate.m_rest < 3;
            if ( on == gate.m_enabled ) continue;
            for ( cudaGraphNode_t node : gate.m_nodes ) cudaGraphNodeSetEnabled( m.m_exec, node, on ? 1U : 0U );
            gate.m_enabled = on;
        }
    }
    cudaEventRecord( m.m_evStart, m.m_stream );
    cudaGraphLaunch( m.m_exec, m.m_stream );
    m.m_snapInFlight = m.m_snapAsked;
    if ( m.m_snapAsked )
    {   // the picture's snapshot behind the block, before the event wait() waits for
        cudaMemcpyAsync( m.m_hSnap, m.m_dSnap,
                         sizeof( float2 ) * ( static_cast<size_t>( m.m_cfg.m_voices ) * m.m_pcap / SnapStride ),
                         cudaMemcpyDeviceToHost, m.m_stream );
        cudaMemcpyAsync( m.m_hHallMemory, m.m_dHallMemory, sizeof( float ) * HallBins, cudaMemcpyDeviceToHost,
                         m.m_stream );             // and the hall's memory
        if ( m.m_life.m_mode != LifeModeE::OFF )   // and the living field, as it stands after this block's steps
            cudaMemcpyAsync( m.m_hLife, m.m_lifeParity ? m.m_dLifeB : m.m_dLifeA, sizeof( float2 ) * LifeCells,
                             cudaMemcpyDeviceToHost, m.m_stream );
        m.m_snapAsked = false;
    }
    cudaEventRecord( m.m_evStop, m.m_stream );
    m.m_nPrev = n;
    m.m_launched = true;
}

bool CloudEngine::ready() const
{
    return !m_impl->m_launched || cudaEventQuery( m_impl->m_evStop ) != cudaErrorNotReady;
}

const float *CloudEngine::wait()
{
    Impl &m = *m_impl;
    if ( m.m_launched )
    {
        if ( m.m_cfg.m_spinWait )
        {
            while ( cudaEventQuery( m.m_evStop ) == cudaErrorNotReady ) _mm_pause();
        }
        else
        {
            cudaEventSynchronize( m.m_evStop );
        }
        cudaEventElapsedTime( &m.m_gpuMs, m.m_evStart, m.m_evStop );
        m.m_launched = false;
        if ( m.m_snapInFlight )
        {
            m.m_snapReady = true;
            m.m_hallDown = true;
            m.m_snapInFlight = false;
            m.m_lifeShown = m.m_life.m_mode != LifeModeE::OFF;
            if ( m.m_lifeShown )
                for ( int i = 0; i < LifeCells; ++i )
                    m.m_lifeCells[static_cast<size_t>( i )] =
                        m.m_life.m_mode == LifeModeE::TURING ? std::min( m.m_hLife[i].y * 3.0f, 1.0f ) : m.m_hLife[i].x;
        }
    }
    return m.m_hOut;
}

float CloudEngine::lastGpuMs() const
{
    return m_impl->m_gpuMs;
}

std::string CloudEngine::deviceName() const
{
    const cudaDeviceProp &p = m_impl->m_props;
    return std::string( p.name ) + " (sm_" + std::to_string( p.major ) + std::to_string( p.minor ) + ")";
}

}   // namespace additiv
