<img src="Deploy/empyrean.png" width="96" align="left" alt="Empyrean" />

# Empyrean

An additive ambient synthesizer: every voice is **twenty thousand sine waves**, computed on the graphics card. Each
harmonic of a note is not one sine but a cloud of slightly detuned ones, every partial with envelopes of its own --
its loudness, its pitch and its place in the stereo field each follow a loop of their own -- so that a held chord never
stands still. It plays by itself, after Noctuary's conductor: a chord in just intonation that changes one voice at a
time, minutes apart, in the spirit of Robert Rich's sleep concerts.

**VST3 plugin and standalone application** for Windows (x64) with an **NVIDIA GeForce RTX** card (20 series or newer),
and a command-line renderer. Licence: AGPL-3.0. The empyrean is the highest heaven, the one made of light.

<br clear="left" />

![The Cloud page: every sounding voice as its spectrum of clouds, the landscape beside it](docs/screenshot.png)

## Download

Not released yet: build it from source (below). The release will bring a setup with the standalone, the VST3, the
offline renderer and the manual, and a portable zip.

## How it is put together

- **The GPU** (`Core/src/Cloud.cu`, CUDA 13.4): 16 voices x 20,000 partials. Per block of 128 samples one CUDA graph:
  an update kernel per partial (its frequency and glide, four amplitude ramps, its three envelopes, the filter's
  response at its frequency, its distance), a render kernel (a thread per sample, 512 partials per chunk, phases in
  32-bit fixed point), a reduction in a fixed order (bit-identical renders); then, in the same graph, the **modal
  hall**: 65,536 decaying resonances, a thread each, fed by the block the voices just made, summed with warp shuffles.
  0.06 to 0.09 ms of GPU time per block of 2.7 ms on an RTX 5090, the hall included.
- **The real-time path** (`Core/src/Engine.cpp`): a worker thread (MMCSS Pro Audio) renders ahead into a ring by two
  host blocks and ten milliseconds; the audio callback stamps notes and copies. A note begins on its exact sample. The
  standalone puts the process into Windows' REALTIME GPU priority class: without it another program's graphics could
  hold the card for 20 ms.
- **On the CPU**, what works on the sum: Noctuary's echoes, near room, diffusion, far reverb and unmask, the bass
  made mono, mid/side, the soft clip -- and the conductor, the modulation, the tuning.

## The sound

- **The spectrum** (Noctuary's formula, per partial): Tilt, Brightness, Odd / Even, Inharmonic (a stiff string's
  stretch), Presence, Pad Low Cut. The voice is normalised after the whole shape: colour changes the sound, not the level.
- **The cloud:** its width in cents and how it widens to the top, every partial's drift and shimmer, its place in the
  stereo field; **Air**, a band of flickering partials heard as noise.
- **Motion:** Random, Wave, Pendulum, Breathe, Glissando, Aurora -- the partials' envelopes set all at once; Aurora
  reads a **landscape** (time against frequency): Aurora, Shepard, Tides, Rain, Choir.
- **The field**, a force on every partial's frequency, carried from block to block on the GPU: Scale gathers the
  cloud into the key, Swarm lets attractors on chaotic (Lorenz) paths gather the partials near them, Scatter parts
  them like dust. **Dispersion** makes each partial's clock late by its pitch: attacks and glints fall as cascades.
- **The filter** as an exact frequency response at every partial: LP 6/12/24, HP 12, BP 12, Notch, Peak, Formant.
- **Tuning:** equal temperament, the just scales, the harmonic series, slendro, Bohlen-Pierce, and Timbre -- the scale
  the cloud's own spectrum asks for (Sethares). Purity narrows the clouds so that just intervals can lock.
- **Space** (Noctuary's): every note on a plane between the ear and the far background, quieter, darker and heard only
  through the far reverb the farther it stands; Breath, Doppler. The far reverb with Noctuary's diffusion, pre-delay,
  unmask, its four networks and freeze.
- **Echoes and rooms** (Noctuary's): two asymmetric delays in series, receding into the far plane; early reflections
  and a small reverb for the near plane.
- **The modal hall:** a reverb made the way the instrument is made, sixty-five thousand resonances that die away, on
  the GPU. Tuned, a share of them stands on the pitches of the scale and the hall rings in key, like sympathetic
  strings; Decay up to a minute, Damping, Spread, Drift, Width, Lowest, Breath (its colour moves), Freeze.
- **The conductor:** Free or Chords (Autoplay), Step, and Brain 2, a second conductor deep in the background.
- **Modulation:** four LFOs without hard edges, two envelopes, four macros, mod wheel, pressure, velocity and a random
  curve, through a matrix of sixteen rows.
- **1024 presets** in sixteen groups, levelled by their loudest second.

## Build

Windows, Visual Studio 2026 with C++, CUDA 13.4 (in its default folder), CMake 3.30 or newer; JUCE is fetched.

```
cmake --preset msvc
cmake --build --preset msvc
ctest --preset msvc
```

The programs land in `bin/msvc`. The tests: the engine's self-test, the GPU probe's, the documentation check (every
entity documented, Doxygen) and pluginval at strictness 10 (where it is unpacked). `Deploy/build_release.ps1` builds
the release (`release` preset: static runtime, code for every RTX generation), runs the tests, makes the manual,
stages, checks and packs it into `dist/`.

## Render

```
empy_render --seconds 600 --preset "Velvet Dreaming" --out night.wav
empy_render --seconds 60 --no-brain --chord 50,57,62,66 --hold 40 --set motion=Aurora
```

`--set` takes any key of the parameter table; `--dump-params` writes the table as JSON (the manual is made from it).

## The family

Noctuary, Ephemeris, Phosphene, Totality, Parhelion: the same window frame, the same keys, the family jam (one
leads, the others follow its root) and the Meta Quest app's hands in bridge mode. The Quest itself has no card for
twenty thousand partials per voice; this instrument stays on the computer.

## Licence

AGPL-3.0 (`LICENSE`). Built with JUCE and NVIDIA's CUDA; the CUDA runtime is linked statically under NVIDIA's licence
for it.
