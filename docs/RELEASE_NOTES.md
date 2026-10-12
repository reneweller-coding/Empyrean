# Release notes

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
