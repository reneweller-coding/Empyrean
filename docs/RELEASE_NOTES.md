# Release notes

## 0.15.0 (14.10.2026, not released)

A hall that grows into the music played in it, each partial's own time, a ripple that lingers, orbits, and 1024 more
presets.

- **The growing hall** (new Hall page): with Grow the modal hall remembers what it hears, band by band over the
  octaves, over Memory (a minute to four hours), and rings longer where it heard much (growth) or shorter (erosion).
  The memory stays through presets, is kept with the session, and can be saved, loaded or begun anew
  (Documents/Empyrean/Halls); the page draws it.
- **Time Bend** (Motion > Time): each partial's clock -- its drift, shimmer and sync -- slower where the spectrum is
  dense, or, below nothing, faster.
- **Ripple Bend:** the ripple's wave slower where the notes sound, so it lingers and piles up there.
- **Orbit** (a field mode): the scale's pitches as wells in pitch and place, a magnetic field bending every move into
  circles -- fast small ones riding a slow one. Pull, Radius, Spring and Wander set wells, size, damping and field.
- **6144 presets:** sixteen new groups of halls and orbits (Growing Halls, Eroded Halls, Cathedral Years, Worn Stone,
  Slow Clocks, Quickened Air, Gravity Lenses, Lingering Waves, Orbits, Cyclotrons, Planetaria, Magnetic Choirs,
  Event Horizons, Memory Palaces, Celestial Mechanics, Living Architecture); 307 of the presets before took the new
  means where they fit (every value they had stays). A whole night of halls and orbits, a route over them.
- **The picture takes the hand:** it shows what the page in front edits, where it acts -- the filter's curve and the
  comb's notches, the air's band and the brightness' edge, the conductor's register over the keys, the ripple's
  string, the hall's memory and its lowest mode -- and the curve's point, the band, the edges and the bracket can be
  dragged.
- **Canvas:** the picture alone, the whole window, the landscape behind the partials and the four macros under them;
  a click in the spectrum plays the key under it.
- **Modulation at the knob:** every matrix row on a knob is an arc round it in its source's colour (its span, and a
  dot where it moves the knob now); a knob's right-click menu lists its rows with their depths and offers every
  source to modulate it with.
- **The groups:** a group's main controls are marked, and a group whose main controls stand at nothing rests, dimmed.
- **Follow Light** (Conductor): the conductors choose notes whose harmonics fall where the landscape (or the living
  field) is bright.
- **The Focus page**, the simple ways in, first among the pages: **Nudge** (darker/brighter, calmer/livelier,
  narrower/wider, purer/airier: every heard knob a little the measured way, the level held), **Focus** (the knobs that do
  the most for this sound), **Variations** (six near the sound, hold to listen, Keep to evolve it), **Teach** (a macro
  learns a gesture by example) and **Find** (the presets that sound like a recording, ringed on the map). Built on a
  measure of what every shaping knob does around 48 presets.
- **Your own presets:** save (Ctrl+S), save as, back to the preset; a star when the sound has moved from its preset, a
  notch on each knob where the preset had it; journeys play them by name.
- **The preset browser:** by name, section, group, stars, recent and the measured traits (darker/brighter,
  stiller/more moving, narrower/wider, purer/airier), and by who plays; hold to listen.
- **The command palette** (Ctrl+K): any knob (its page opens, it lights up; "cutoff 800" sets it), preset or action.
- **The info line:** what the control under the mouse does, its value and what moves it -- a sentence for all 210.
- **Macros as large knobs**, named by what they move; **snapshots** with pictures; **MPE seen at the keys**;
  a **night mode** for dark rooms; **first steps** on the first start.
- **Every page fits the window** at its usual size (1280 x 860, and still at 800 high) without scrolling: the boxes
  fill the rows as tightly as they can, Focus stands in two columns, the matrix has a page of its own with every
  source live beside its rows, the space's and the hall's pictures take what is left, and the picture at the top
  gives way where a page needs the room. A test (ctest `layout`) keeps it so.
- **The whole bank levelled anew:** 3530 presets from before 0.14.0 measured again with today's engine (some had
  drifted by several dB); the map and the search read the new measures.
- **The space from above** on the Space page: the voices at their planes and pitches, Breath's wandering as trails;
  Depth and Keys Depth as rings to drag.

## 0.14.0 (14.10.2026, not released)

Swarms and chimeras, two living landscapes that answer the playing, turbulence, the filter morphing by transport,
MPE, and 1024 more presets.

- **Partial Sync's modes:** Chimera (each partial coupled to its neighbours round its cloud with a lag: an ordered,
  pulsing core beside a flickering mist, its border wandering) and Swarm (swarmalators: every partial also has a place
  on a circle in pitch and field, and place and phase pull each other -- a phase wave circling through the cloud at the
  rate, two groups pulsing against each other, broken waves, or each for itself), with Swarm, Reach and Lag.
- **Two living landscapes:** Fluid (a lattice Boltzmann fluid the notes blow ink and momentum into at their
  frequencies: plumes, eddies, vortex streets) and Islands (Cahn-Hilliard separation: a fine field gathers into a few
  large islands drifting across the spectrum).
- **Turbulence** (Cloud page): the energy cascade of a struck gong -- a loud harmonic hands its energy upwards, the
  louder the more, and the spectrum settles back over Settle. Brighter, not louder: a soft chord stays pure.
- **The filter morphs by transport:** in a glide, an A/B morph or on the map, different filters move into each other
  -- their power curves read as masses, the blend their Wasserstein barycentre -- so a pass band or a formant wanders
  to its new place instead of fading out while the other fades in.
- **MPE:** the lower zone, every note on channels 2 .. 16 with its own bend (48 semitones), pressure and slide
  (controller 74); channel 1 the master. Press and Slide each move a target: level, filter, brightness, width,
  nearness, shimmer or motion. RPN 0 sets the bend range, a host's MPE configuration switches it on or off.
- **5120 presets:** sixteen new groups of swarms and currents (Swarms, Splinters, Chimeras, Twin Swarms, Currents,
  Vortex Streets, Ink Plumes, Islands, Labyrinths, Archipelagos, Gong Blooms, Bronze Swells, Shimmer Cascades,
  Pressure Pads, Expressive Voices, Living Storms); 324 of the presets before took the new means where they fit (every
  value they had stays). 168 journeys, thirteen routes; the map measured anew.

## 0.13.1 (13.10.2026, not released)

Corrections after listening: the preset menu, the jumps in a morph with modulation, the map.

- **A preset you choose is the one you hear.** Under the A/B morph a preset from the list only moved
  the knobs, which the morph does not play; with the map on it only moved the cursor, and there a
  blend of six presets played. Now the list, the arrows, a program change and the headset play the
  preset: Morph switches off, a journey or a route stops, the map lets go (its cursor goes to the
  preset's point), and the sound glides in.
- **Morphing with modulation has no steps.** Each sound brings its own LFOs and envelopes: on the way
  the rows of the sound being left keep running on its modulators while they fade (before, LFO 1
  changed its shape at the middle, and a slow breath on the brightness quickened into the next
  sound's flutter). A motion, a living landscape or a field passes through nothing where its kind
  changes with whatever the matrix adds to its depth (before, 0.30 of a depth stood there), and a
  picture does so where its landscape changes. Following the map's cursor, a kind fades out before
  it changes, and whole numbers glide (before, a far jump moved the partials by 91 % of their range
  in one block).
- **The map as the presets sound.** Every preset was measured (its spectrum's centre, how much it
  changes over a quarter of a second, a second and three seconds): across darker to brighter, up
  still to moving, half rank and half value as in Noctuary. The first map ranked guesses from the
  parameters and filled the square evenly; its axes agreed with the measured ones by a correlation
  of 0.24 and 0.16. Now the bank's darker, gently moving middle is crowded and its edges thin.
- **Radius in the spacing of the presets** round the cursor (0.1 .. 4, default 0.4): on a preset's
  own point that preset plays (0.98 of the blend), half way to a neighbour the two share; in an empty
  stretch the blend widens with the distances. The blend's weights are continuous everywhere: a
  preset enters and leaves the blend at nothing.
- The Map page: the map from the top, as large as the window shows; the knobs and the routes beside it.
- **The keys** under the picture of the partials, each at its pitch on the same axis: the notes that sound lit in their
  colours (yours, the conductor's, the second conductor's), the conductor's root marked, their names in the corner.
- **A maximise button** on the standalone's window, as Noctuary and Totality have.

## 0.13.0 (13.10.2026, not released)

Spectral means, moving pictures, the preset map, and 1024 more presets.

- **Stretch** (Cloud page): every octave of the harmonics spans that many cents -- with the Timbre scale, which is
  made from the spectrum, a stretched spectrum gets the stretched tuning in which it is consonant again.
  **Shift** moves every partial by the same number of hertz: harmonic sounds turn metallic and bell-like.
- **The spectral comb** (Filter page): notches evenly spaced per octave, read at every partial; Sweep moves them
  through the spectrum without end (a phaser that rises or falls for ever), Stereo combs left and right apart. The
  level is kept over the sweep.
- **Your own landscape:** click the landscape beside the picture and open a picture or a recording.
- **Animated pictures** in the living landscape, after the continuous transforms of the Kaleidoscope visualizer:
  Kaleidoscope (your landscape folded into turning mirror wedges), Tunnel, Spiral (an endless zoom: patterns travel
  through the spectrum without end), Flow (like ink in water), Feedback, Rain (rings on a pond), Starfield and
  Moire, with Sides and Turn. Every partial reads the picture: across it is heard as the colour wandering, along
  it as the cloud's shimmer and width. They cost next to nothing on the GPU.
- **The preset map** (Map page): every preset on a square, across from darker to brighter, up from still to
  moving. With Map on, the sound is the blend of the presets round the cursor, followed over Map Glide. Click to go
  to a preset, drag to wander. **Routes** walk the cursor: waypoints with a travel time (how long the sound takes to
  turn into the next), a hold time and a radius; twelve come with it, yours are saved as text. empy_render plays
  them too (`--route`).
- **4096 presets:** sixteen new groups of visions (Kaleidoscopes, Tunnels, Spirals, Ink & Flow, Feedback Loops,
  Raindrops, Star Streams, Moire Veils, Stretched Worlds, Shifted Metals, Comb Phasers, Barber Poles, Painted
  Light, Prisms, Alien Harmony, Dream Machines); 899 of the presets before use the new means where they fit (every
  value they had stays). 135 journeys.
- Not built: a "coherent start" (every partial starting in phase). In a cloud the hundreds of members of a harmonic
  lie within cents of each other, so in phase they add in amplitude -- measured 19 dB louder for half a second, and
  a plain sine rather than a pulse.

## 0.12.0 (12.10.2026, not released)

Sleep concerts: 1024 more presets, sixteen groups made for the night.

- **Sixteen groups of sleep concerts:** Lullabies, Slow Breath, Deep Sleep, Dream Drift, Moonlight, Night Rain, Ocean
  Sleep, Cradle, Starfield, Hypnagogia, Twilight Choirs, Warm Drones, Snowfall, Forest Night, Slow Tides, Afterglow.
  Each starts from one of the sixteen sounds and is made slow (attacks of six seconds and more, releases of fifteen),
  darker, with a slow conductor (a change every one and a half to five minutes, few voices, low) and nothing sudden.
- **Modulations made for the night**, four or five a preset on top of the bank's own, every one slow and smooth: a
  breath of five to seven a minute (Slow Breath lets the room, the harmonics' pulse and the colour breathe at the same
  rate), tides of two to eight minutes, a smooth random wander, Lorenz and Roessler over ten to thirty minutes, the
  Kuramoto oscillators well in step, slow Lenia, the night deepening while the harmony stands (Root Age makes it darker
  and farther), calming answers to what is heard (louder becomes darker, change stills the shimmer, roughness tunes it
  purer), a reservoir with a long memory -- many of them through a via source.
- **102 journeys:** a Journey and a Night for every group, and six Whole Nights, two of them (Lullabies, Deep Sleep)
  from the sleep concerts.
- The 2048 presets before are unchanged; the new ones are measured and levelled like them.

## 0.11.0 (12.10.2026, not released)

Glide, morph and journeys: one sound becoming another, and sleep concerts for a whole night.

- **Glide** (Journey page): a preset chosen -- arrows, list, program change, headset -- travels from the sound before
  it over that many seconds instead of jumping. Every voice is a spectrum written afresh every block, so the sound
  itself moves: every number on its way in its own range, the partials sliding into a new scale, the voice filter
  (a gain at every partial) fading from one filter into the other, a motion or a living landscape fading out where
  it changes and in again, the matrix as two weighted sums. The knobs show where it is going; a knob turned on the
  way changes where it goes.
- **Morph A/B:** two slots (the sound as it plays, or a preset); with Morph on, A - B moves the whole instrument
  between them, the matrix included. Switching it on and off glides. The slots are kept with the project.
- **Journeys:** presets in a row, each held for a time drawn from its range and glided into the next, round and
  round -- Noctuary's format and player, but each step a glide rather than a crossfade of two engines. 68 come with
  the instrument: for each group a Journey (twelve of its different faces) and a Night (its twelve stillest, ten to
  eighteen minutes each, glides of two to four), and four Whole Nights across groups (about six and a half hours
  round). Write your own with + and save them; the files are plain text.
- **empy_render** plays journeys (`--journey`), and writes long renders as it goes, an hour a file: a whole night
  offline in about forty minutes.

## 0.10.0 (12.10.2026, not released)

Living systems: Noctuary's modulators, five new things on the GPU, and twice the presets.

- **Noctuary's sources** in the matrix: Lenia (a living field), Lorenz and Roessler (two strange attractors), four
  Kuramoto oscillators falling into step, the Beat (the rate the lowest interval beats: it turns in equal temperament,
  stands in just intonation), the conductor's Cascade, Root Age, Note, Distance, Random per Note. Every row of the
  matrix can now have a **via** source that scales its depth.
- **What it hears:** the instrument listens to itself on the GPU -- Loudness, Heard Brightness, Change and Roughness
  (Sethares' dissonance: a fifth reads 0.08, a second 0.54) are sources too.
- **The reservoir:** an echo state network of 2048 neurons on the GPU, listening to what is heard and played; its
  four readouts are smooth curves that never repeat and remember the music. Chaos (below 1 it forgets, above it runs
  on by itself), Memory, Input.
- **Living landscape** (Motion page): Lenia or a Turing pattern (spots, mitosis, coral, worms, mazes) grows on a
  256 x 128 field on the GPU, and every partial's loudness follows the field where it stands. The level is kept.
- **Partial sync:** each partial a Kuramoto oscillator coupled to the others of its harmonic: coupled, a harmonic's
  members pulse as one (the 8th harmonic swings 10.7 dB instead of 5.8).
- **Flock** (the field's fourth mode): each harmonic's cloud moves like a flock of birds on a Lorenz wind, keeping its
  width (its centre wandered 42.8 cents, its spread stayed 7.1).
- **Ripple:** every new note strikes a string stretched along the spectrum; the waves run up and down, reflect,
  cross, and every partial's loudness rides them.
- **Melt** (Effects page): the spectrum held and melting into the space between the notes -- 4096-point frames on
  the GPU, each band's energy held over Melt Time and diffused across the spectrum (Spread, in octaves after a
  second, energy kept), sung again by a phase vocoder at the frequencies it heard.
- **The sources, live:** the Modulation page draws every source's last four seconds.
- **2048 presets** in 32 groups. The first sixteen use the new systems where they fit (761 of them; every value they
  had stays). Sixteen new groups put one system in front each: Living Fields, Turing Gardens, Synchrony, Flocks,
  Ripples, Melting Spectra, Spectral Cascades, Resonant Halls, Ghosts & Memory, Strange Attractors, Reservoir Minds,
  Self-Listening, Gravity Wells, Smear & Dispersion, Lattices & Beats, Night Gardens. All measured and levelled again.
- GPU time with twelve voices: about 0.1 ms per block of 2.7 ms; the reservoir adds 0.0065 ms, the melt 0.007 ms.
- **No dropout at the start:** the engine renders a second latency's worth ahead before the first callback and lets
  the callbacks use it up, so a device that fills its buffer in a burst no longer finds the ring short (the
  standalone had one underrun in its first second). The latency the host is told stays.

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
