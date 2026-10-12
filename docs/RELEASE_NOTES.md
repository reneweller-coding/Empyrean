# Release notes

## 0.8.0 (12.10.2026, not released)

Forces on the partials.

- **The field** (Motion page): every partial keeps an offset from its own place, moved each block by what pulls it
  and drawn back by a spring. *Scale* gathers the cloud onto the scale's pitches (a wide or stretched cloud collapses
  into the key, keeping a 2-cent shimmer around each pitch); *Swarm* lets up to eight attractors wander through the
  spectrum on Lorenz paths and gathers the partials near them; *Scatter* pushes the partials away from them. Pull,
  Radius, Spring, Wander, Attractors. The level stays (measured within 0.6 dB): partials that gather keep their
  phases and a narrow spread, so they do not fuse into one sine of a frozen random level.
- **Dispersion**: each partial's clock -- its envelopes, the landscape it reads and the voice's level, of which the GPU
  keeps a history -- runs late by its pitch, up to 1.5 s. Attacks and the landscapes' glints fall as cascades from
  the top (or rise, negative). A voice stays until its latest partial has heard the end of its envelope.
- **The modal hall breathes**: every resonance's decay wanders slowly by up to an octave, so the hall's colour moves.
- The factory presets use them where they fit (the field in 159, dispersion in 149, breath in 114), and the bank is
  levelled again. Neither costs measurable GPU time.

## 0.7.0 (12.10.2026, not released)

Echoes, rooms and a hall of its own.

- **The modal hall**, on the GPU in the same graph as the voices: 65,536 resonances, each a sine that dies away,
  woken by everything the voices play (near and far). Decay (0.5 to 60 s), Damping (the upper octaves die sooner),
  Tuned (the share of the resonances on the scale's pitches: the hall rings in key and retunes with the scale),
  Spread, Drift, Width, Lowest and Freeze. It costs about 0.025 ms of GPU time per block. White noise comes out at the
  level it went in, whatever the decay; tuned resonances are weighted by how close they stand, so Tuned colours the
  hall rather than making it louder (measured within 1 dB).
- **Echoes** (Noctuary's): two asymmetric delays in series, with Cross, Damping, Absorb, Duck and To Far.
- **The near room**: early reflections from a room that follows where the voices stand, and a small reverb.
- **The far hall** gains Noctuary's Diffusion, Pre-Delay, Unmask, its four networks (Colourless by default) and Freeze.
- A new page, *Effects*; every effect knob is a target of the modulation matrix (the switches and Far Mode are not).
  The factory presets have each group's echoes, room and hall (804 of 1024 a modal hall), and the bank is levelled
  again with them.

## 0.6.0 (12.10.2026, not released)

The whole instrument, and its name: **Empyrean** (the working name was AdditivAmbientSynth; the code keeps it).

- **The GPU engine:** 16 voices x 20,000 partials on an NVIDIA GeForce RTX card, every partial with envelopes of
  its own (loudness, pitch, place), 0.03 to 0.08 ms of GPU time per block of 2.7 ms. The standalone takes Windows'
  REALTIME GPU class. A card the build has no code for, or a computer without NVIDIA's driver, is named in the
  status line; the setup asks before installing there.
- **The sound:** Noctuary's spectral formula per partial, clouds, air, six motions and five landscapes, the filter as
  an exact response per partial (LP 6/12/24, HP, BP, Notch, Peak, Formant), Noctuary's tunings and the Timbre scale,
  Noctuary's space (near and far planes, far reverb, Breath, Doppler, Tide).
- **The conductor:** Noctuary's Autoplay (Chords) and Free modes, Step, and Brain 2. After a preset change both
  conductors fill their chords within seconds (as in Noctuary), rather than at their own event rate.
- **Modulation:** four LFOs without hard edges, two envelopes, four macros, mod wheel, pressure, velocity and a slow
  random curve through a sixteen-row matrix; live rings on the modulated knobs.
- **1024 presets** in sixteen groups, levelled by their loudest second to -20 dBFS (measured -30 .. -4 before).
- **The family:** the window frame, keys, the family jam (follow the leader's root and section energy, or lead) and
  the Meta Quest app's hands in bridge mode.
- **The release:** VST3, standalone, `empy_render` (now with `--preset` and `--dump-params`), the manual built out
  of the program, a setup and a portable zip (`Deploy/build_release.ps1`).

Found on the way: the preset generator wrote whole numbers that ended in zero without it (a cutoff of 180 Hz became
18 Hz): 24 presets had a cutoff a decade too low (seven Dark Drones were silent), 9 an air band a decade too narrow.
