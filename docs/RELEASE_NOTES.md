# Release notes

## 0.9.1 (12.10.2026, not released)

- **The picture shows the partials themselves.** It used to draw each voice's shape from the knobs, one quiet bar per
  harmonic, so nothing of what the partials do was seen. Now the update kernel keeps every eighth partial's frequency
  and amplitude each block, the engine brings the 51,200 points down about thirty times a second (400 KB, behind the
  block, before the event the worker waits for), and the picture draws them as points with fading trails: loudness
  up, frequency across. Measured how much they move: with a preset that reads a landscape (Velvet Dreaming, Aurora)
  single harmonics' bands swing by 15 to 22 dB within twenty seconds, with the default Breathe motion 5 to 8 dB.

## 0.9.0 (12.10.2026, not released)

Five ideas for the GPU.

- **Binaural partial cloud** (Space page; headphones): every partial of the near plane at a place of its own round the
  head. Being a sine, a partial needs only one complex gain per ear -- the head's shadow and the pinna (Brown and
  Duda), the time round the head (Woodworth) as a phase -- so the render kernel adds a cosine to the sine it had.
  Surround, Height, Orbit. Measured at 90 degrees: the 8th harmonic 13 dB louder at the near ear, the fundamental
  0.85 ms sooner; in front 0.0 dB and 0.00 ms; the level as in stereo. Presets never switch it: it is the listener's.
- **Memory** (Effects page): every note let go leaves a ghost on four extra GPU voices -- it rises over Bloom after the
  note, fades over Memory Time, widens, drifts and darkens (Blur), recedes (Sink); its own phases and its own dice.
- **Self-Tune** (Tuning page): voices whose harmonics nearly meet (Reach) pull together, weighted by loudness and by
  how low the harmonics are; Anchor draws them back to the scale. An equal-tempered C major triad: third 400.0 ->
  386.8 cents, fifth 700.0 -> 701.9.
- **Painted hall**: the modal hall heard through the landscape at each mode's frequency (Paint).
- **Spectral echo** (Effects page): a 1024-point FFT in one CUDA block (no cuFFT: it would have added some hundred
  megabytes), every band of each ear through a delay line of its own with feedback; Time, Tilt, Scatter, Feedback,
  Damping, To Far.
- The presets use them where they fit (memory 189, spectral echo 181, self-tune 101, paint 94); the bank is levelled
  again. GPU time with twelve voices and Brain 2 sounding: 0.10 ms per block of 2.7 ms, 0.17 ms with every effect on.

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
