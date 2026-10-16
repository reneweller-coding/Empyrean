/**
 * @file Kernels.cuh
 * @brief The cloud's kernels as the engine (Cloud.cu) launches them, file by file, and the functions that fill each
 *        file's tables in constant memory. Each kernel is documented where it is defined.
 * @author Rene Weller
 */
#pragma once

#include "Device.cuh"

namespace additiv::gpu
{

// Voices.cu
__global__ void syncOrderKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const float *theta,
                                 const float *sx, float2 *order, size_t plane );
__global__ void gainRingKernel( const BlockHeader *hdr, const CloudVoice *voices, float *ring );
__global__ void filterTransportKernel( const BlockHeader *hdr, const CloudVoice *voices, float *table );
__global__ void cascadeKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, float *qa, float *qb );
__global__ void updateKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const EnvShape *shapes,
                              const float *land, uint4 *run, float4 *amp, float4 *gains, const float *ring,
                              float2 *ampQ, float2 *snap, const float2 *lifeA, const float2 *lifeB, float *theta,
                              float *sx, const float2 *order, size_t plane, float2 *flockA, float2 *flockB,
                              const float *qa, const float *qb, const float *filterTable, unsigned long long *hallHist,
                              const float *density, float *tau, float4 *orbitA, float4 *orbitB, const float *dreamState,
                              const float4 *swarmView, const float *physTrail, unsigned int *physDeposit,
                              const float2 *haloState );
__global__ void renderKernel( const BlockHeader *hdr, const CloudVoice *voices, int pcap, const uint4 *run,
                              const float4 *gains, float *scratch, int maxN );
__global__ void reduceKernel( const BlockHeader *hdr, const float *scratch, int chunks, int perVoice, int maxN,
                              float *out );
__global__ void phantomReduceKernel( const BlockHeader *hdr, const float *scratch, int perVoice, int maxN, float *out );
cudaError_t uploadVoiceTables( const signed char ( *xy )[2] );

// Hall.cu
__global__ void hallKernel( const BlockHeader *hdr, unsigned long long *hist, float *memory, float *density,
                            float *dreamAcc );
__global__ void modalKernel( const BlockHeader *hdr, const float *in, float2 *state, float *scratch, int maxN,
                             const float *land, const float *memory );
__global__ void modalReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out );

// Life.cu
__global__ void lifeSeedKernel( const BlockHeader *hdr, float2 *a, float2 *b );
__global__ void lifeStepKernel( const BlockHeader *hdr, int node, float2 *a, float2 *b, const float *land );
__global__ void fluidSeedKernel( const BlockHeader *hdr, float *fa, float *fb );
__global__ void fluidStepKernel( const BlockHeader *hdr, int node, float *fa, float *fb, float2 *lifeA, float2 *lifeB );
__global__ void islandsKernel( const BlockHeader *hdr, float *isles, float2 *lifeA, float2 *lifeB );
__global__ void lifeStatsKernel( const BlockHeader *hdr, const float2 *a, const float2 *b, float *stats );
cudaError_t uploadLifeTables( const float *kernel );

// Weather.cu
__global__ void windStepKernel( const BlockHeader *hdr, int node, float *a, float *b );
__global__ void chimeStepKernel( const BlockHeader *hdr, const float *wind, float4 *state, float2 *impulse, int *meta );
__global__ void chimeKernel( const BlockHeader *hdr, const float2 *impulse, float2 *state, float *scratch, int maxN );
__global__ void chimeReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out );
__global__ void phase27StatsKernel( int *meta, float *stats );
__global__ void rainFieldKernel( const BlockHeader *hdr, const float *wind, float *showers );
__global__ void rainSpawnKernel( const BlockHeader *hdr, const float *showers, int2 *birth, float2 *at, int *meta );
__global__ void rainKernel( const BlockHeader *hdr, const float2 *at, float4 *drop, int2 *birth, float *scratch,
                            int maxN, int *meta );
__global__ void leafKernel( const BlockHeader *hdr, const float *wind, float4 *state, float *scratch, int maxN,
                            int *meta );
__global__ void surfKernel( const BlockHeader *hdr, float4 *state, float *scratch, int maxN, int *meta );
__global__ void weatherReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out );
__global__ void phase28StatsKernel( const BlockHeader *hdr, const float *wind, int *meta, float *stats );

// Dream.cu
__global__ void dreamSimKernel( const BlockHeader *hdr, const float *acc, const float *mem, const float *state,
                                float *sims, const int *meta );
__global__ void dreamRecallKernel( const BlockHeader *hdr, float *acc, float *mem, float *state, const float *sims,
                                   int *meta );

// Bloom.cu
__global__ void bloomKernel( const BlockHeader *hdr, const float *in, float2 *state, const float *bands, float *scratch,
                             int maxN );
__global__ void bloomBandKernel( const BlockHeader *hdr, const float2 *state, float *bands, float *stats );
__global__ void bloomReduceKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out );

// Agents.cu
__global__ void swarmForceKernel( const BlockHeader *hdr, const float4 *a, const float4 *b, float4 *partial );
__global__ void swarmStepKernel( const BlockHeader *hdr, float4 *a, float4 *b, const float4 *partial );
__global__ void swarmViewKernel( const BlockHeader *hdr, const float4 *a, const float4 *b, float4 *view, float *stats );
__global__ void physTrailKernel( const BlockHeader *hdr, float *trail, unsigned int *deposit );
__global__ void physStatsKernel( const BlockHeader *hdr, const float *trail, float *stats );

// Rooms.cu
__global__ void haloKernel( const BlockHeader *hdr, float2 *state, float *stats );
__global__ void roomSourceKernel( const BlockHeader *hdr, const float *scratch, int perVoice, int maxN, float *hist,
                                  float *in );
__global__ void roomStepKernel( const BlockHeader *hdr, int step, float *field, const float *in, float *ears );
__global__ void roomOutKernel( const BlockHeader *hdr, const float *ears, float *hist, float *out );
cudaError_t uploadRoomTables( const signed char ( *xy )[2], const short ( *nbr )[4], const float *fir );

// Night.cu
__global__ void harpKernel( const BlockHeader *hdr, const float *wind, float2 *modes, float2 *wakes, float *scratch,
                            int maxN, float *stats );
__global__ void thunderChannelKernel( const BlockHeader *hdr, float4 *seg );
__global__ void thunderRenderKernel( const BlockHeader *hdr, const float4 *seg, int *ring );
__global__ void thunderOutKernel( const BlockHeader *hdr, int *ring, float *scratch, int maxN );
__global__ void iceKernel( const BlockHeader *hdr, float *stress, float4 *chirps, int *meta, float *stats );
__global__ void iceSingKernel( const BlockHeader *hdr, float4 *chirps, float *scratch, int maxN );
__global__ void nightMixKernel( const BlockHeader *hdr, const float *scratch, int maxN, float *out );

// Spectral.cu
__global__ void spectralEchoKernel( const BlockHeader *hdr, float *io, float *hist, float *acc, float2 *ring, int *st );
__global__ void meltKernel( const BlockHeader *hdr, float *io, float *hist, float *acc, float *energy, float *omega,
                            float *phase, float *heard, int *st );
__global__ void listenKernel( const BlockHeader *hdr, const float *io, float *hist, float *prev, int *st,
                              float *stats );

// Reservoir.cu
__global__ void reservoirInitKernel( float *w, float *win, float *rout, float *x );
__global__ void reservoirStepKernel( const BlockHeader *hdr, const float *w, const float *win, float *x );
__global__ void reservoirReadKernel( const BlockHeader *hdr, const float *rout, const float *x, float *stats );

}   // namespace additiv::gpu
