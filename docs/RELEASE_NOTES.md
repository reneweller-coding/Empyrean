# Release notes

## 1.0.0 (17.10.2026)

The first public release: an additive synthesizer that does on a graphics card what no processor could -- and then
keeps going.

- **320,000 partials in real time.** Sixteen voices of twenty thousand sine waves each, every partial with its own
  frequency and glide, three envelopes (loudness, pitch, place), the filter's exact response at its frequency and its
  distance -- in one CUDA graph per block: on an RTX 5090, twelve voices sounding, 0.10 ms of GPU time per block of
  2.7 ms, 0.17 ms with every effect on. The sum in a fixed order, so a render is bit-identical every time.
- **A reverb made of the instrument's own stuff:** a modal hall of 65,536 decaying resonances on the card, tuned to the
  scale, that remembers what it heard over minutes to hours and rings longer there; a spectral echo of 500 bands.
- **Physics as sound:** a lattice-Boltzmann fluid the notes blow ink into, a room simulated as air (the wave equation
  on 250,000 cells, with a time-reversal mirror), Lenia and Turing fields, Kuramoto sync, chimeras and swarmalators,
  a slime mould between the harmonics, a gong's energy cascade, an Aeolian harp whose vortices lock onto a harmonic,
  thunder from a lightning channel laid out anew at every strike, a frozen lake cracking in its critical state, a
  night chorus of two thousand insects and frogs.
- **The sky and the deep:** a star's oscillations (asteroseismology: a sun, a red giant, a white dwarf), a black hole
  that redshifts the partials falling in and lenses the rest into a ring, black holes merging, hydrogen's spectral
  lines with their Zeeman triplets, twinkling through Fresnel turbulence, the ocean's deep sound channel traced ray by
  ray, whistlers in the magnetosphere, giant vocal tracts singing kargyraa.
- **The ear:** binaural per partial (head shadow, pinna, the time round the head, each of 20,000 partials at its own
  point), a melody heard only between the ears (dichotic pitch), streams that part and fuse by common fate, Risset's
  rhythm that slows for ever.
- **Tuning that listens:** a conductor in just intonation that changes one voice at a time; Self-Tune pulls nearly
  meeting harmonics together; Sethares' scales from the cloud's own spectrum; the root walking as a quantum wave on the
  Tonnetz.
- **A mind of its own:** a reservoir of 2048 neurons on the card as a modulation source, the instrument dreaming of
  what it played (a modern Hopfield network over its own spectra), macros taught by example, presets found by a
  recording, a preset map that grows itself (MAP-Elites).
- **16384 presets** in 256 groups, each levelled by its loudness as it is played (ITU-R BS.1770) and placed on a map
  by how it sounds; glides in which the partials themselves travel and filters move into each other by transport; 531
  journeys, nineteen of them whole nights.
- **Real time on Windows:** a render-ahead ring (two host blocks and ten milliseconds), MMCSS, and Windows' REALTIME
  GPU class, so another program's graphics cannot hold the card.
- **New since 0.26.0:** the render app writes its score cues and chapters beside a journey (`--cues`, `--chapters`)
  for KaleidoscopeEnhanced and Heliacal; the signal flow (`docs/flow.png`), the demos (`Tools/demo/make_demos.py`), the
  social preview, and the publish script.

VST3 and standalone for Windows 10 or 11 (x64) with an NVIDIA GeForce RTX card (20 series or newer) and a current
driver; developed and tested on an RTX 5090 -- on a smaller card lower *Partials* (Cloud page) and leave the largest
simulations off. The installer is not code-signed.

## 0.26.0 (16.10.2026, not released)

Journeys to take along.

- **Empyrean Journeys:** a small program beside the instrument (Start menu, EmpyreanJourneys.exe) that renders
  journeys into files -- WAV or FLAC at 24 bits, Ogg Vorbis or MP3 at 320 kbit/s -- faster than real time, the
  instrument running offline as a journey plays it (every preset glided into, the conductor playing on). A table of
  every journey, the instrument's and yours, with its steps, its length and its shortest to longest, searched and
  sorted; several chosen are rendered one after another, with progress, speed and time left, and can be cancelled.
  The dice are cast before a rendering begins, so the panel tells its exact length; the passes of a cyclic journey,
  a fade over the last twenty seconds, the night arc as from 23:00. It sets every partial 94 times a second instead
  of the instrument's 375 -- about three times faster, the same sound (only Scintillation's quickest twinkle a little
  smoother); "Exactly as the instrument plays" renders sample for sample. The same from the command line
  (`--render NAME --format wav|flac|ogg|mp3`, `--list`), and a self-test of every format and the queue (ctest
  `renderapp`). Built on the family's render app (Tools/renderapp: RenderKit.h, the window, the queue, the files),
  which the other instruments share with their own sources.
- **Journey lengths:** the instrument's journey box shows every journey with its length (a pass at the middle of its
  ranges). The timeline under it counts a glide within its step, as a journey plays it -- it drew the glides ahead of
  the stays, and every journey looked longer by them (Whole Night - Plasma & Voice: 7:54:00 instead of 6:24:00).
- **The bank levelled as it is played:** every preset's level measured as the panel plays it -- its conductor on, the
  chord at once, 45 seconds for the halls, the floor and the conductor to build up -- instead of a fixed chord with the
  conductor off, which put presets up to 6 dB (in places 25) above or below their neighbours. The loudest were
  measured at their own gain, past the output's soft clip, which had made them seem quieter than they play (by up to
  10 dB). Only the master gain moves; the bank's middle stays where it was.
- **Time Mirror fixed:** with the conductor playing, 44 of the 48 mirror presets in a wave room rose to full scale
  within a minute -- the room's uniform pressure, never heard, sent back on every step of the window. The mirror's
  recording and the room's output are high-passed now, and a governor takes the mirror back should a room ring on.
  The mirror is subtler for it: much of its swell was that pressure.

## 0.25.0 (16.10.2026, not released)

Plasma and voice. 1024 presets; 16384 in all.

- **Magnetosphere** (Night): the Earth's magnetic field as a VLF radio hears it -- whistlers (a lightning click drawn
  out along a field line into a falling tone, with its echoes; every strike of Thunder sends one), the dawn chorus
  (rising tones in bursts, like birds), plasmaspheric hiss. Whistlers, Dispersion, Dawn Chorus, Gyrofrequency, Hiss.
- **Throat** (Dreams): the throat of the abyss -- up to four vocal tracts of a giant's length (Kelly-Lochbaum), singing
  the root, its fifth, its octave; Kargyraa closes the ventricular folds every other cycle, an octave below, as in
  Tuvan and Tibetan chant. Tract, Kargyraa, Vowel, Throats.
- **16384 presets:** sixteen new groups of plasma and voice (Whistler Nights, Sferic Storms, Dawn Choruses, Radio
  Auroras, Plasma Hiss, VLF Receivers, Monk Drones, Kargyraa Halls, Giant Throats, Vowel Tides, Abyssal Choirs,
  Overtone Caves, Throat & Thunder, Cosmic Chant, Ionosphere, Voices of the Void); 531 journeys, nineteen whole nights.

## 0.24.0 (16.10.2026, not released)

Light and depth. 1024 presets.

- **Scintillation** (Cosmos): the partials twinkle as starlight does -- a drifting screen of turbulence, every band
  heard through its own Fresnel zone, bright and dark patches wandering over pitch and time. Scattering, Drift.
- **Spectral Lines** (Cosmos): Lines draws every harmonic to a line of hydrogen's spectrum (Pfund at the note up to
  Lyman six octaves above); Heat is the lines' Doppler width; Zeeman splits them into slowly beating triplets;
  Absorption carves hydrogen's dark lines, on the key's tonic, into everything that sounds.
- **Wreck** (Cosmos): a derelict's structure -- hull, tank, truss, plate, its modes worked out from a mass-spring
  model (Tools/presets/make_wreck.py) -- struck by micrometeorites and creaking in thermal snaps, most at dusk and dawn
  with the night arc. Structure, Wreck Size, Hits, Snaps, Ring.
- **Deep Channel** (Rooms): the ocean's sound channel -- rays traced through Munk's profile; a sound comes back as a
  swell of arrivals that ends at once, darkened by the water. Range, Off Axis.
- **15360 presets:** sixteen new groups of light and depth (Twinkling Stars, Turbulent Light, Hydrogen Choirs, Balmer
  Bells, Zeeman Fields, Dark Lines, Nebula Glow, Derelict Hulls, Orbital Tanks, Broken Trusses, Perforated Plates,
  Thermal Snaps, Deep Channels, Ocean Crossings, Whale Roads, Lost Signals); 498 journeys, eighteen whole nights.

## 0.23.0 (16.10.2026, not released)

Stars and gravity: a page Cosmos, the redshift hall. 1024 presets.

- **Star** (Cosmos): a star's oscillations, as asteroseismology hears them, transposed into the audible -- hundreds of
  modes, each kept ringing by the convection's noise (a Lorentzian line). A Sun (an almost even comb, its pairs of
  lines drawing together with age), a Red Giant (clusters of narrow mixed modes), a White Dwarf (gravity modes evenly
  spaced in period: an undertone series). Star Age, Spin (the azimuthal orders, a slow chorus), Star Pitch.
- **Black Hole** (Cosmos): partials fall into it -- redshifted, dimmed, drawn to it, frozen at the horizon --, the
  light behind it is bent into an Einstein ring and brightened. Fall, Hole Place.
- **Merger** (Cosmos): now and then two black holes merge -- the wave rises for minutes from half a hertz (felt first
  as the field's width breathing faster and faster), becomes a tone in the bass and rings down in the hole's
  quasi-normal modes. Ringdown, Mass Ratio, Inspiral; the source Merger.
- **Redshift** (Hall): the modal hall's modes glide down for ever and are reborn at the top -- every tail sinks.
  Redshift Time.
- **14336 presets:** sixteen new groups of stars and gravity (Sun Songs, Red Giants, White Dwarfs, Spinning Stars,
  Ageing Stars, Star Choirs, Accretion Disks, Falling Light, Einstein Rings, Mergers, Distant Mergers, Redshift Halls,
  Receding Light, Cosmic Floors, Quiet Universe, Gravity's Choir); 465 journeys, seventeen whole nights.

## 0.22.0 (16.10.2026, not released)

The floor. Measured with the conductor playing, a quarter of the bank had a thin bass: its foundation went through
formant and band-pass filters, or rain and harps covered it. Now every preset stands on one. 1024 presets.

- **The foundation keeps its depth:** whatever filter the sound goes through, the foundation's fundamental and second
  harmonic come through it.
- **Floor** (Conductor): the share of the sound under 150 Hz the foundation holds at the least -- raised slowly, by
  up to 12 dB, while the rest covers it.
- **Pedal:** the foundation holds its note while the root moves, for up to three minutes, or for ever on the key's
  tonic: an organ point.
- **Pressure:** a share of the foundation's partials a beat above the rest -- the floor pulses slowly, as two organ
  pipes a hair apart do. Pressure Rate.
- **Sub Octave, Fifth, Breath:** a voice an octave under the foundation, one a fifth above it, swelling and ebbing.
- **Every preset has a floor:** a foundation wherever the conductor plays (2387 presets that floated got one) and
  Floor's share; nothing else changed.
- **13312 presets:** sixteen new groups of deep floors for dark ambient (Organ Points, Pressure Drones, Sub Abysses,
  Fifth Pillars, Breathing Floors, Cathedral Depths, Abyssal Plains, Black Monoliths, Iron Pedals, Tectonic Hum,
  Cavern Drones, Undertow, Low Chapels, Night Pressure, Deep Bells, Floors & Storms); 432 journeys, sixteen whole
  nights. The Nocturnes' "Nachtmusik" presets are now "Small-Hours Air".

## 0.21.0 (16.10.2026, not released)

A night chorus of insects and frogs, a time-reversal mirror in the wave room, a map that grows itself, and the whole
bank levelled anew by its loudness. 1024 presets.

- **Night Chorus** (Night): two thousand insects and frogs over the field, eight kinds -- field and tree crickets,
  katydids, mole crickets, peepers, tree frogs, bullfrogs, toads --, each calling in a band of its own (the acoustic
  niche). Every caller is a clock; a call of its own kind heard close enough resets it (Greenfield's inhibitory
  resetting), so neighbours fall into step or into turns, and a caller about to call into a louder one of another kind
  in its band waits for a gap. Who Calls (a meadow, the treetops, a pond, all), Warmth (warm insects call faster:
  Dolbear's law; the night arc cools the night), Answering, Tuned Calls.
- **Time Mirror** (Rooms): sixty-four cells round the listener in the wave room listen for a window, then send it back
  reversed -- the waves run back to where they came from (Fink's time-reversal mirror): every sound returns as a swell
  converging on its source. Two strokes, so it never hears itself. Time Mirror, Mirror Window.
- **The map grows itself:** `Tools/presets/grow.py` (MAP-Elites) keeps the best sound in every cell of the map,
  mutates them, renders and measures the children as the bank is measured (`additiv_presets --evaluate`) and keeps the
  new ones -- nearer the player's stars when there are any -- as user presets in Documents/Empyrean/Presets/Grown,
  levelled like the bank. The plugin now lists user presets from subfolders too.
- **Loudness:** the bank measured anew, every preset by its loudest three seconds weighted as the ear weighs them
  (ITU-R BS.1770, LUFS) rather than its loudest second's RMS, which weighed a wave room's bass as much as a rain's
  glints; measured by day, so the night arc does not count.
- **12288 presets:** sixteen new groups of choruses and mirrors (Cricket Meadows, Katydid Treetops, Frog Ponds, Summer
  Nights, Warm Evenings, Cool Nights, Answering Choirs, Tuned Choruses, Time Mirrors, Reverse Halls, Mirror Chapels,
  Converging Halos, Storm Choruses, Lake Nights, Night Concerts, Nocturnes); 399 journeys, fifteen whole nights.

## 0.20.0 (16.10.2026, not released; its presets came with 0.21.0)

The night: an Aeolian harp in the wind, thunder, singing ice; streams that part a voice or fuse it; Risset's rhythm;
the real night where you are; the dream's journal; the headset in the cloud. A page Night, 1024 presets.

- **Aeolian Harp** (the new page Night): twelve strings in the field's wind, tuned to the chord. Each sheds vortices
  that lock onto one of its harmonics -- Facchinetti's wake oscillator, a van der Pol oscillator coupled to the string
  -- and jump to the next as the wind changes. Harp Pitch, Gauge.
- **Thunder** (Night): every strike lays out a lightning channel of its own, tortuous at every scale and branched; each
  of its four thousand segments sends an N-wave (Few), clapping across the line of sight, rumbling along it, heard as
  late as it is far and with the air's absorption. Storm Distance, Rumble; the source Lightning flashes at the strike.
- **Singing Ice** (Night): a frozen lake as a sheet of slider blocks in its critical state (Olami, Feder and
  Christensen), cracking in avalanches of every size; each crack a chirp falling from the highs, as bending waves in
  ice are dispersive. Ice Thickness, Cold.
- **Streams** (Dreams): the partials parted into streams -- odd and even, by three, registers, members -- each with a
  vibrato and tremolo of its own, so the ear hears several sounds (Bregman's common fate); Fusion moves them as one
  again.
- **Risset** (Dreams): the voices pulse in Risset's rhythm that slows (or quickens) for ever and never arrives.
- **Night Arc** (Night): the sun's height where you are (Settings > Night arc; at first the main city of your time
  zone) as a night of 0 .. 1 that darkens, slows and widens the sound; the source Night for the matrix.
- **Dream Journal** (Dreams): the dream's newest thousand memories kept with the project, and dreamt on when it opens.
- **Headset** (Rooms) and **Empyrean for Meta Quest** (Quest/): the app sends the head's turn and the hands; Head
  Tracking keeps the binaural cloud and the thunder in place, Hands puts them into the wind, the halo, the wave room
  and the slime mould's trail. Built, not yet tried on a headset.
- **Faster:** a resting feature's kernels are switched off in the CUDA graph; the modal hall rings four modes a thread;
  the swarm's forces spread over 64 blocks; the presets measured in parallel. The kernels live in Core/src/gpu, a file
  a subject.
- **11264 presets:** sixteen new groups of night and illusions (Aeolian Harps, Harp Gardens, Distant Storms, Summer
  Storms, Rolling Thunder, Singing Lakes, Frozen Nights, Thin Ice, Twin Streams, Fusion Choirs, Parting Clouds, Endless
  Ritardando, Endless Accelerando, Night Arcs, Inside the Cloud, Night Weather).

## 0.19.0 (16.10.2026, not released)

A room simulated as air, a topological insulator of sound round the listener, and a conductor whose root walks as a
quantum wave on the Tonnetz. A page Rooms, 1024 presets.

- **Wave Room** (the new page Rooms): the voices sound in a room simulated as air -- the wave equation on a grid of a
  quarter of a million cells, eleven centimetres apart, six thousand steps a second, on the graphics card. Its modes,
  its standing waves and its flutter happen of themselves below about 1.5 kHz. Every voice stands where it is, the
  listener's two ears hear it; Room Size, Walls, and Listener -- walk through the room and the bass swells and thins.
- **Halo** (Rooms): a topological insulator made of sound (Qi, Wu and Zhang's Chern insulator on a disc of 441 sites
  round the listener). Every note of the chord strikes its rim from a direction of its own, and what it starts runs
  round you one way only, past any disorder inside (topological protection); made trivial, it stays where it was
  struck. Once Round, Way Round, Topology, Disorder.
- **Quantum Walk** (Conductor): the conductor's root moves as a continuous-time quantum walk on the Tonnetz (fifths
  and thirds): a wave over the twelve pitch classes that spreads and interferes, looked at every so often -- the root
  moves to where it is found and the wave partly collapses there, so it remembers where it might have gone.
- **10240 presets:** sixteen new groups of rooms and topology (Chapel Air, Stone Halls, Curtained Rooms, Walking Rooms,
  Standing Waves, Halo Rings, Counter Halos, Edge States, Still Halos, Halo Chapels, Quantum Harmony, Interference
  Nights, Tonnetz Walks, Quantum Halos, Resonant Spaces, Topologies); the presets before are as they were.
- **empy_render --log S:** a line of its numbers every S seconds (the halo's and the walk's among them).

## 0.18.0 (16.10.2026, not released; its presets came with 0.19.0)

The wind brings weather -- rain, leaves, a sea --, a plate everything rings blooms, the partials swarm round the
listener or crawl as a slime mould between the harmonics. A page Weather, 1024 presets.

- **Rain** (the new page Weather, with the Wind and the Chimes): drops fall over the field where the showers are, and
  the wind carries the showers across. On water each drop traps a bubble that sings (Minnaert's resonance, as van den
  Doel measured it, rising as it rises), on leaves it ticks, on stone it clicks, on a tin roof it rings the sheet; tuned,
  the bubbles and the roof sing the scale.
- **Leaves** (Weather): four thousand clusters over the field rustle as the wind where they stand blows, harder in its
  gusts -- broad leaves, pine needles, grass, dry leaves.
- **Surf** (Weather): sixty-four trains of swell run onto a shore along the field and come in sets; the crests foam where
  they break -- sand hisses, pebbles crackle in the backwash, rocks boom. The wind raises the waves.
- **Bloom** (Effects): a large plate of 32768 modes that the voices, the chimes and the weather ring. Its energy climbs
  into the high modes, as a tam-tam's shimmer blooms after the stroke (the cascade of weak wave turbulence, the energy
  kept), and its pitch rises with the energy like an opera gong's.
- **Swarmalators** (Motion): a thousand oscillators swarm in the plane round the listener while they synchronise; the
  partials ride them, placed and pulsing -- a ring whose pulse circles you, all pulsing as one, turning clusters, a still
  disc.
- **Physarum** (a mode of the Field): a share of the partials crawl as a slime mould over a trail the harmonics feed;
  veins grow between the harmonics and the partials glide along them.
- **The pages:** Ripple moved to Harmonics, Memory to Dreams.
- **9216 presets:** sixteen new groups of weather and growth (Rain on Water, Tin Roofs, Forest Rain, Stone Gardens,
  Rustling Woods, Pine Winds, Seashores, Pebble Beaches, Storm Coasts, Tam-Tam Halls, Thunder Sheets, Swarm Rings,
  Firefly Swarms, Slime Veins, Mycelium Nights, Weather Gardens); the presets before are as they were.

## 0.17.0 (16.10.2026, not released)

Four layers that are not notes, made for the night -- a field of chimes in a wind, a melody heard only between the ears,
a dream of what was played, the conductor's notes heard coming -- and timbres that come back. A page Dreams, 1024 presets.

- **Chimes** (the new page Dreams): 2048 tuned chimes hang over the field, left to right and near to far, in a real
  simulated wind (a lattice Boltzmann flow, vortex streets shed from three posts). Each clapper is a pendulum the wind
  swings; gusts cross the field and the chimes ring in waves. Tubes, glass, bamboo or bells, tuned to the scale; Wind
  and Gusts.
- **Phantom** (Dreams): noise the same in both ears except where a pitch turns the phase between them -- the brain
  hears a faint, clear tone there that is in neither ear (Huggins' pitch). With headphones. It carries a slow melody
  of its own, the notes that just ended, or the chord; heard dry, after the rooms and echoes.
- **Dream** (Dreams): every three seconds the instrument keeps what it heard, if it is new, and a dream wanders among
  these memories by association (a modern Hopfield network), sung far off: over a night the harmonies of hours ago
  come back, blurred and linked.
- **Premonition** (Dreams): the conductor's notes are heard coming -- their cloud swells wide and far before they
  begin, as a reverb backwards, and is drawn into them.
- **Talbot** (Harmonics): the first 24 harmonics evolve as a wave function. Revival: the timbre falls apart, stands
  upside down half way and is whole again after the period, lingering at each (Talbot's effect); Bloch: the energy
  climbs the harmonics and comes back.
- **8192 presets:** sixteen new groups of dreams and phantoms (Phantom Choirs, Inner Voices, Talbot Carpets, Bloch
  Tides, Glass Forests, Bamboo Groves, Wind Tubes, Temple Winds, Storm Gardens, Dreaming Rooms, Night Memories,
  Premonitions, Breathing In, Phantom Gardens, Lucid Dreams, Elsewhere); the presets before are as they were.
- **Icon:** the program carries the icon drawn for every size (Deploy/empyrean.ico); a build could keep the old one,
  JUCE making it only when CMake configures.

## 0.16.0 (15.10.2026, not released)

The harmonics get lives of their own: registers that overblow, seasons of overtone families, avalanches, a floor that
opens beneath a held chord. And grains, a horizon, breathers on the ripple, 1024 more presets.

- **Overblow** (the new page Harmonics): every note's registers -- its fundamental, its octave, its twelfth, its double
  octave -- compete for a breath, as in a flute. The register that sounds holds on, so a note jumps up later than it
  falls back, and one near the edge flickers between two. The breath: the knob, how hard the note is played, and MPE
  pressure (the new target Breath).
- **Subharmonics** (Harmonics): a loud harmonic pumps a cloud at half its frequency, and that one a cloud at half
  again -- the period doubles, as in a Faraday experiment. It grows from nothing, slowly near its threshold: hold a loud
  chord and a dark floor opens beneath it; let go and it closes. The power comes from the pumping harmonic.
- **Horizon** (Harmonics): the harmonics bend towards a horizon they never pass; the low ones stay, the high ones
  crowd below it into a halo. The conductors and Self-Tune hear the bent harmonics.
- **Seasons** (Harmonics): five families of harmonics -- the octaves, fifths, thirds, sevenths and ninths over the
  fundamental -- take turns in a slow round, as in rock-paper-scissors, never settling and never dying out.
- **Avalanche** (Harmonics): a rice pile on every note's harmonics (the Oslo model): mostly small glints, now and then a
  slide through a dozen harmonics, rarely one through all of them -- the statistics of sand piles and earthquakes.
- **Grain** (Harmonics): a share of the partials sounds in grains, each born at a new place in its cloud and the field.
  With the full twenty thousand partials it makes the cloud breathe and rustle; with a few hundred it glitters.
- **Breathers** (Motion > Ripple): the ripple's string as a sine-Gordon chain -- a note's strike stays where it fell as
  a breather, pulsing there and drifting slowly, instead of running off as a wave.
- **7168 presets:** sixteen new groups of harmonic lives (Overblown Flutes, Breath Registers, Four Seasons, Turning
  Years, Avalanches, Glacier Calving, Subterranea, Period Doubling, Halo Horizons, Aurora Curtains, Rainfall Grains,
  Breathing Grains, Breather Chains, Harmonic Ecology, Night Ecology, Wild Harmonics); 395 of the presets before took
  the new means where they fit -- only added: every value they had stays.

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
- **ASIO** in the standalone's audio dialog (Windows), with Steinberg's ASIO SDK under its GPLv3 option.
- **The header tidied:** search, canvas, night and the MIDI marks as glyphs beside undo and help; a meter of the output
  with a clip mark; A | B held lets you hear the preset as it was; a line under Step runs down to the conductor's next
  exchange; the status tells what sounds (the GPU and the device in its tooltip, the GPU only when it is tight).
- **Focus, quieter:** the variations are there on their own (made anew whenever the sound moves), Keep once a tile was
  heard; a "+" in the first free snapshot; teaching in steps (Teach, then Set here or Cancel, a small x to forget).
- **One Modulation page:** the matrix and every source live; a click on a source opens it in the inspector, where the
  modulators' knobs live now. **Alt-drag** any knob: the inspector's source modulates it by the drag. **Undo** takes back
  every change to the matrix -- and a preset's matrix with the preset, which it did not before.
- **Spans and links:** Lowest - Highest and Hold Min - Max as one bar with two ends; a chain links an echo's Time L and R.
- **Tuning shows the tuning:** the octave over the root with the scale's pitches in cents off the steps, and the voices
  where they sound, Self-Tune's pull as a tail; a click on a line plays it, Shift-click holds it, so an interval can
  be heard locking. **Journeys** get a timeline of their steps: a click goes to a step; the journey being written is
  changed there by hand (a step's edge, its glide's ramp, its place; right-click removes it), and Edit a copy makes one
  of the instrument's journeys yours.
- **By hand:** the hall's memory painted with the mouse (Grow comes on at 0.4 when it was at nothing); the map's
  waypoints dragged (snapping to a preset) and removed with a right-click; a morph slot takes a snapshot of the Focus
  page.
- **The map zooms:** the wheel (or a pinch) zooms round the mouse up to 32 times, a right- or middle-drag moves the
  view; an overview in the corner, Whole map, the presets' names where few are in view, and the view follows the cursor
  (or the preset playing) when it walks out.
- **The window comes back where it was left:** the standalone keeps its place, its size and whether it was maximised,
  across screens; a size left on a screen larger than the main one is kept too.
- **Words in the palette:** "dark calm wide" (or dunkel, ruhig, breit ...) finds the presets measured so.
- **Zoom** (100 - 150 %) in the settings, beside the window's size; the last page comes back; the foundation's key is
  underlined in the picture.
- **A floor under the conductor (Foundation, Conductor > Register):** one voice holds the root in the octave from C2 up
  (from Lowest, when lower) while the root stands and moves with it; the chord is drawn from a fifth above. The conductor's notes
  had no bass of their own -- in Chords mode a candidate was scored by consonance and voice leading only, the root even
  halved. 4956 of the 6144 presets have a foundation now (drawn last: everything else they had stays); measured over 48
  of them, the lowest note is the root 82 % of the time instead of 22 %. Its level follows the chord.
- **Every page fits the window** (1280 x 860 and more, still at 800 high) without scrolling: the boxes fill the rows
  as tightly as they can, Focus stands in two columns, the matrix has a page of its own with every source live beside
  its rows, the space's and the hall's pictures take what is left, and the picture at the top gives way where a page
  needs the room. A test (ctest `layout`) keeps it so.
- **The window opens as large as the screen allows** (three quarters of its width, at least 1280 x 860) and comes back
  at the size it was left at; in a large window the picture grows.
- **The pages in sections, with what is on:** Focus; Sound (Cloud & Filter -- now one page --, Motion); Harmony;
  Space (Space, Effects, Hall); Modulation (Modulation, Matrix); Explore. A dot beside a page's name for every group
  with main controls, lit while it plays; a page whose groups all rest is dimmed; Matrix counts its rows.
- **The modulation dock** along the bottom: every source a chip with its live trace -- drag it onto any knob to
  modulate it; click it for the inspector: its trace, its own knobs, every knob it moves with depth and removal.
- **The whole bank levelled anew:** 3530 presets from before 0.14.0 measured again with today's engine (some had
  drifted by several dB); the map and the search read the new measures.
- **The space from above** on the Space page: the voices at their planes and pitches, Breath's wandering as trails;
  Depth and Keys Depth as rings to drag.
- **Its own look** (15.10.2026): a picture behind the panel, as the siblings have one -- a sea of clouds at night under
  streams of light -- and a new logo, the overtone ladder (the partials of one tone as lights on stalks, falling as the
  harmonic series does), in the header, the icon and the setup.

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
