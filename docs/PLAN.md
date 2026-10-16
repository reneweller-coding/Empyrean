# Empyrean: Plan für den additiven GPU-Synth (Arbeitsname AdditivAmbientSynth)

Stand 11.10.2026, Phase 0 (Messprogramm) gebaut. Repo `G:\Tools\VRAudio\AdditivAmbientSynth` (eigenes Projekt neben
Noctuary), Namensraum vorläufig `additiv::`; der Name in der Familie der Himmels- und Lichtwörter ist offen. Code-Kommentare
im Doxygen-Format wie bei den Geschwistern, Plan und Journal auf Deutsch. Zahlen: **[M]** hier gemessen, **[I]** inferiert.

## Idee

Noctuarys additive Bank (32 Partials, bis zu sechs verstimmte Strands, also höchstens 192 Phasoren pro Stimme) ist der
Lieblingsklang des Nutzers. Darauf aufbauend: **20.000 Partials pro Stimme, komplett auf der GPU (CUDA)**.

Unter 20 kHz passen bei A2 nur 181 Harmonische. 20.000 Partials heißen deshalb: Jede Harmonische wird zu einer **Wolke**
aus verstimmten Partials. Das ist das PADsynth-Prinzip, nur live statt als vorberechnete Schleife: Jeder Partial hat
seinen eigenen Platz in der Glockenkurve der Wolke, eine eigene langsame Tonhöhendrift, eigenen Amplituden-Shimmer und
einen eigenen Ort im Stereofeld (später im Raum). Die 20.000 sind ein Budget, kein Ziel: Ab einer gewissen Dichte pro
kritischem Band klingt mehr wie gefiltertes Rauschen. Wo die Grenze liegt, entscheidet das Gehör (Hörvergleich unten).

Entscheidungen des Nutzers (11.10.2026): CUDA; der neueste nvcc (13.4); kein macOS-Pfad.

## Toolchain

- CUDA 13.4, im Preset über den Ordner gewählt (`toolset: cuda=C:/.../CUDA/v13.4`). `CUDA_PATH` alter Shells zeigt auf
  13.3, das `nvcc` im PATH ist 11.0 und kennt Blackwell nicht.
- `CMAKE_CUDA_ARCHITECTURES 120` (RTX 5090), Visual Studio 2026 (MSVC 19.51), CMake 4.4.
- `.\build.ps1 msvc -Test` baut und startet den Selbsttest; die Programme liegen in `bin\msvc`.

## Phase 0: das Messprogramm `probe` (gebaut)

`Core/` (bis 12.10.2026 `Gpu/`) ist der Kern der Engine: `CloudVoice` (die wenigen Zahlen einer Stimme), `CloudEngine` (Tabellen,
Kernel, ein CUDA-Graph pro Block). Pro Block laufen drei Kernel: **update** (ein Thread pro Partial: Frequenz,
Amplitudenrampe und Startphase aus den Stimmparametern), **render** (ein CUDA-Block pro 512 Partials, ein Thread pro
Sample, Partials im Shared Memory) und **reduce** (die Teilsummen immer in derselben Reihenfolge, also bitgleich).
Phasen sind 32-Bit-Festkomma und laufen über Stunden ohne Drift.

`Tools/probe` misst: `selftest`, `bench`, `latency` (getaktete Blöcke aus einem MMCSS-"Pro Audio"-Thread, sofort
abgewartet), `ahead` (das geplante Design: ein Worker rendert in einen Ring voraus, ein getakteter Konsument nimmt pro
Periode einen Block), `render` (Hörvergleich). `--gpu-priority` setzt die GPU-Planungsklasse des Prozesses.

### Messungen [M] (11.10.2026; RTX 5090, i9-12900K, Windows 11 26100, WDDM; 16 Stimmen x 20.000 Partials)

| Messung | Ergebnis |
|---|---|
| Selbsttest | ein Partial: −128 dB Restfehler gegen den exakten Sinus; volle Stimme +0,12 dB vom Sollpegel; bitgleich |
| `bench`, Block 128 (2.667 µs) | GPU 47 µs, Rundlauf 55 µs, **48-fache Echtzeit**; 16 x 100.000 noch 20-fach |
| `latency` 60 s, Desktop ruhig | Mittel 168 µs, max 956 µs, kein Block über dem Budget |
| `ahead` 10,7 ms und 5,3 ms, ruhig | 0 Unterläufe; einzelne Worker-Blöcke bis 3,5 ms |
| unter Fremdlast¹, normale Klasse | `latency`: 462 von 11.250 Blöcken zu spät (bis 5,3 ms); `ahead` 10,7 ms: ein Hänger von 20 ms (4 Unterläufe) |
| unter Fremdlast¹, Klasse HIGH | keine Besserung (636 zu spät) |
| unter Fremdlast¹, Klasse **REALTIME** | `latency` 40 s: max 720 µs, **0 zu spät**; `ahead` 5,3 ms: 0 Unterläufe, der Ring nie unter voll |

¹ ein zweiter CUDA-Prozess, der die GPU dauerhaft füllt (`probe bench --block 1024 --voices 64 --partials 100000`).
Windows teilt die GPU zwischen Prozessen in Zeitscheiben; die Klasse REALTIME (`D3DKMTSetProcessSchedulingPriorityClass`)
lässt unsere Blöcke vor. Sie wurde ohne Administratorrechte angenommen.

Offen: dieselben Läufe unter **Grafiklast** (Spiel, Quest Link, Browser-Video), z. B.
`probe ahead --seconds 600 --gpu-priority realtime`.

### Hörvergleich

`probe render --partials 200,2000,20000` schreibt `work/renders/cloud/cloud-<n>.wav`: Akkord A2 E3 B3 C4 E4, 30 s,
alle auf −20 dBFS RMS. Bei 200 Partials hat jede Harmonische 1 bis 3 Partials (fast Noctuarys Bank), bei 2.000 sind es
11 bis 33, bei 20.000 110 bis 333. Stellschrauben: `--width` (Cent, 25), `--slope` (0,3), `--drift` (Cent, 3),
`--shimmer` (0,4).

**Urteil des Nutzers (11.10.2026):** Zwischen 2.000 und 20.000 ist der Unterschied schwer zu hören; weil die Leistung
gleich ist (16 Stimmen: 40 gegen 47 µs GPU pro Block [M], 0,3 % des Budgets), gilt **20.000 Partials pro Stimme**.
Die nächste Frage des Nutzers: eine eigene, loopbare, komplexe Hüllkurve für jeden Sinus (Phase 0b).

## Phase 0b: eine Hüllkurve für jeden Partial (gebaut, 11.10.2026)

Der Nutzer: "Könnten wir eigentlich auch jedem Sinus eine eigene (loopbare) komplexe Hüllkurve zuweisen?" und
"baue das alles ein".

**Der Haken, der das Design bestimmt.** Unabhängige Zufallshüllkurven auf 110 bis 333 Partials pro Harmonische mitteln
sich weg; die Harmonische schwankt dann nur noch um etwa 1/√K. Das Spektrogramm der Szene `random` bestätigt es [M]: kaum
von `none` zu unterscheiden. Hörbare Struktur entsteht nur, wenn die Hüllkurven der Partials **zusammenhängen**.

**Modell** (`Core/include/additiv/Cloud.h`):
- Eine **Bank** von bis zu 64 Formen (`EnvShape`): bis zu 16 Punkte mit Kurve, ein Loop, nach dem Loop ein Release-Schwanz
  (beim Loslassen verlässt die Form ihren Loop dort, wo sie gerade ist). Acht Werksformen in `Shapes.cpp`.
- Drei **Spuren** pro Stimme (`EnvLane`): Lautstärke, Tonhöhe (Cent), Ort im Stereofeld. Jede Spur sagt, wie die
  Partials die Formen teilen: welche Form, welches Tempo (`rateSpread`), welcher Versatz im Loop (`phase`), welche
  Verzögerung, welche Tiefe (`depthTilt`), jeweils aus einer **Quelle** pro Partial: Tonhöhe (log), Harmonische, Platz
  in der Wolke, Zufall (pro Parameter ein eigener Würfel), gerade/ungerade. Dazu `chaos`: vom Gleichschritt bis "jeder
  für sich". Eine Spur mit `amount` 0 kostet nichts.
- Eine **Landschaft** aus Zeit mal Frequenz (bis 512 x 8192, logarithmisch 20 Hz bis 20 kHz, loopt): jeder Partial liest
  die Spalte seiner Frequenz. Quellen: gemalt (jedes Bild, das Windows lesen kann: Zeit nach rechts, Frequenz nach oben,
  Helligkeit = Amplitude in dB über `--range`), aus einer Aufnahme analysiert (WAV/FLAC: STFT, 256 Spalten, Weißung,
  zyklische Glättung 0,15 s, nahtloser Loop, Normierung auf das 99,5. Perzentil) oder erzeugt (`aurora`). Jede
  Landschaft wird als BMP nach `work/landscapes` gezeichnet und kann übermalt und wieder geladen werden.
- Auswertung einmal pro Block und Partial als reine Funktion der Zeit (kein Zustand, bitgleich). Die Tonhöhe **gleitet**
  im Block linear (quadratische Phase), Links und Rechts rampen getrennt: keine Stufen, keine Sprünge.

**Messungen [M]** (16 Stimmen x 20.000 Partials, Block 128): ohne Hüllkurven 55 µs GPU pro Block (vorher 47: das
Gleiten und die getrennten Kanalrampen), mit Hüllkurven 53 bis 60 µs, alle drei Spuren zusammen 60 µs; rund 40-fache
Echtzeit. Selbsttest: Dreieck auf der Lautstärkespur auf 0,03 % genau, Glissando ohne Sprung und exakt gelandet
(−132 dB), Panorama mit konstanter Leistung, Landschaft und Release-Schwanz auf ihren Sollwerten, alles bitgleich.

**Szenen** (`probe scenes`, `work/renders/scenes/<name>.wav`, Akkord A2 E3 B3 C4 E4, 40 s): `none`, `random`
(unabhängiger Zufall), `wave` (eine Welle steigt durchs Spektrum), `pendulum` (Pendelwelle über die Harmonischen; im
Spektrogramm die stärkste Wirkung), `breathe` (Lautstärke, Ort und Tonhöhe wandern durch jede Wolke), `deepnote`
(jeder Partial gleitet aus bis zu 15 Halbtönen in den Akkord), `aurora` (gemalte Landschaft), `recording`
(Meeresaufnahme aus Noctuarys FieldRecordings als Landschaft). **Hörurteil des Nutzers (12.10.2026):** "klingt alles
schon mal sehr interessant und ziemlich anders als was man von normalen Synthesizern so kennt"; **Aurora gefällt besser
als Pendulum, das zwischendurch eher dünn wird.**

Warum Pendulum dünn wird [A]: Die Form Pulse fällt nach 0,25 s Anstieg wie (1−x)^4 und liegt rund 70 % ihrer Zeit nahe
null, die Tiefe ist 1. Die tiefen Harmonischen haben fast dieselbe Rate und sind deshalb periodisch gemeinsam leise:
Der Bass fällt aus (im Spektrogramm dunkle senkrechte Lücken unten bei etwa 10, 20, 29 und 37 s). Aurora dagegen setzt
seine Bänder auf einen Boden bei −30 dB; die Wolke ist nie leer. Folgerung: Bewegung soll auf einem tragenden Klang
liegen, nicht ihn ersetzen; Landschaften sind ein Kernstück (Bibliothek, Editor in der GUI).

Beobachtet: Landschaften senken den Pegel stark (aurora −42 dBFS roh, weil der Boden bei −30 dB liegt); im Synth
braucht die Landschaft eine Pegelkorrektur.

## Folgerungen für die Architektur [I]

1. **Der Audio-Callback wartet nie auf die GPU.** Ein Worker rendert feste GPU-Blöcke (128 Samples) in einen Ring mit
   Vorlauf (Start 10 ms, einstellbar); das Plugin meldet die Latenz dem Host. Noten und Parameter tragen Zeitstempel.
2. **GPU-Klasse REALTIME.** Sie gilt pro Prozess: Im Standalone unproblematisch, im Plugin beträfe sie den ganzen
   DAW-Prozess. Zu entscheiden, wenn das Plugin kommt: Klasse im Host-Prozess setzen oder ein kleiner GPU-Server-Prozess
   (ein CUDA-Kontext für alle Instanzen, Ring im Shared Memory).
3. Die Stimme bleibt ein paar Dutzend Zahlen; die GPU rechnet daraus die Partials. Presets und Modulation arbeiten wie in
   der Familie auf diesen Zahlen.
4. Eine CPU-Referenz nur für Tests und CI (GitHub-Runner haben keine GPU), kein macOS.

## Was von Noctuary kommt, und wo es läuft (Entscheidung 12.10.2026)

Der Nutzer fragte, wie viel wir von Noctuary übernehmen, ob die Robert-Rich-Idee passt ("JA absolut"), ob der Autoplay
mitkommt (ja) und ob die entscheidenden Module auf die GPU können. Die Regel: **Was pro Stimme oder pro Partial im
Spektrum wirkt, rechnet die GPU** (dort wird es fast kostenlos und genauer: für einen Sinus ist jedes lineare Filter ein
Faktor und eine Phase pro Ohr); **was rekursiv auf der Summe läuft, bleibt auf der CPU** (jedes Sample hängt am
vorigen; ein GPU-Thread wäre langsamer als ein Kern, und die ganze Effektkette kostet in Noctuary 1 % eines Kerns).

| Noctuary | hier | wo |
|---|---|---|
| Tuning (reine Skalen, Scala, Purity), ClusterBrain (Free, Autoplay/Chords, Brain 2, Arc), Clock, Dsp | Kopie mit Herkunftsvermerk | CPU |
| Raummodell: nah/fern, Luftdämpfung, Presence, Breath, Doppler, Tide, Phase Width, Externalise | pro **Partial** statt pro Stimme; später binaural (Brown/Duda) | GPU |
| Filter: Z-Plane, Formant, lineare Antwort der Circuit-Filter | Frequenzgang am Partial ausgewertet | GPU |
| Air, Ghost | breite, schnell driftende Wolken | GPU |
| Strike | Modalsynthese aus abklingenden Partials | GPU |
| Rückkopplung "To Pitch" | letzter GPU-Block moduliert den nächsten | GPU |
| Far-Reverb (FDN), Delays, Convolution-Room, Body, Mid/Side, Patina, Soft Clip | Kopie | CPU |
| Klangquellen (Harmonic, FM, Wavetable, Texture, Bow, Spectral, ...) | ersetzt durch die Wolke | -- |
| Near-Layer, Cosmos, GrainCloud, Presets | vorerst nicht | -- |

Rich passt: Das Atmen der Partials ist hier das Grundprinzip; die Konsonanz nach Sethares (`BrainSpectrum`,
`makeTimbreScale`) kann am exakt bekannten Spektrum der Wolke messen; "Raum ist eine Landschaft" wird pro Partial möglich.
**Haken:** reine Intervalle rasten nur ein, wenn die Wolken schmaler sind als der Abstand rein/temperiert (große Terz
14 Cent, Wolke 25 Cent): *Purity* soll die Wolken mit verengen. Identität: Rich ist der musikalische Rahmen, die eigene
Stimme sind Wolken, Hüllkurven pro Partial und Landschaften.

## Phase 1: das spielbare Instrument (gebaut, 12.10.2026)

**Aus Noctuary kopiert** (d263f54, Herkunftsvermerk im Kopf jeder Datei, Namensraum `additiv`, Makros `ADDITIV_`):
`Dsp.h`, `Simd.h`, `Oversample.h`, `Effects.h/.cpp`, `Tuning.h/.cpp` mit `HarmonicEntropy.inc`, `ClusterBrain.h`;
`cmake/Family.cmake` unverändert aus Totality. `Gpu/` heißt jetzt `Core/` (Familienlayout, Doccheck).

**Neu:** `Params.h/.cpp` (eine Tabelle, 45 Parameter, Noctuarys Standardwerte; Autoplay startet in *Chords*, die
Bewegung in *Breathe*), `Motion.h/.cpp` (die Szenen aus Phase 0b als Wahl des Instruments), `Engine.h/.cpp`:
- Ein **Worker** (MMCSS Pro Audio) rendert GPU-Blöcke von 128 Samples in einen Ring und hält ihn um Hostblock plus
  10 ms voraus; der Callback stempelt Noten (Offset + Latenz) und kopiert nur. Ein Ton beginnt exakt auf seinem Sample
  (`CloudVoice::onset`, Rampe ab dem Einsatz). Offline rendert `process()` dieselbe Zeitlinie selbst. Latenz an den
  Host: 768 Samples (16 ms) bei Block 256, 1024 bei 512.
- **Stimmen:** 16, Hüllkurve aus Noctuarys `Envelope` mit Blockrate, Stehlen mit einem Block Ausblenden, Release,
  Frequenzen gleiten bei Stimmungswechseln (0,3 s).
- **Stimmung:** Noctuarys Skalen, Purity mischt 12-TET und Skala; **Purity verengt die Wolken** (bei 1 auf 40 %).
- **Dirigent:** `ClusterBrain` (Free, Chords = Autoplay), `BrainSpectrum` aus dem Spektrum der Wolke; seine Töne
  stehen bimodal nah/fern wie in Noctuary (40 % nah, 60 % tief).
- **GPU:** vier Kanäle (nah L/R, fern L/R); pro Partial `cos/sin(d·π/2)`, −6 dB je Einheit und Luftdämpfung
  (Tiefpass erster Ordnung, Ecke ab 16 kHz bis fünf Oktaven tiefer, relativ zur Entfernung 0, also am Ohr neutral).
- **Mischung:** fern nur über den Far-Reverb (Noctuarys FDN, 100 % nass), nah trocken, dann Mid/Side, Master,
  Soft Clip. **GPU-Klasse REALTIME** im Standalone; das Plugin lässt seinen Host vorerst in Ruhe.

**Werkzeuge:** `additiv_render` (Nacht ohne Fenster, `--set key=value`), Plugin `AdditivAmbientSynth` (VST3 und
Standalone, JUCEs generische Oberfläche nach Abschnitten gruppiert; Arbeitsname bis zum eigentlichen Namen).

**Gemessen [M]:** Selbsttest 10/10 (Tabelle, reine Quinte 1,5 exakt, Stille bitgenau, Einsatz auf dem Sample,
Determinismus mit Dirigent, Autoplay füllt, Stehlen, Release, 6 s Echtzeit mit kommenden und gehenden Noten:
0 Unterläufe, 0 verspätet); Probe-Selbsttest 12/12; Doccheck grün; **pluginval Stufe 10 bestanden**. Autoplay,
240 s mit Every 20 s: −22 dBFS RMS, Spitzen −8 dBFS (Noctuarys Bereich −21 bis −28), 0,05 ms GPU pro Block,
37-fache Echtzeit offline. Der Standalone startet und läuft (239 MB).

**Pendulum nach dem Urteil des Nutzers:** Die Tiefe wächst jetzt mit der Tonhöhe (ein Viertel bei 110 Hz, die Hälfte
bei 1 kHz). Nach der Nummer der Harmonischen war es zuerst versucht: Dann bekamen die starken tiefen Harmonischen nur
ein Fünftel der Tiefe, und die Pendelwelle verschwand fast.

**Offen aus Phase 1:** Brain 2, Scala- und Timbre-Skala (bis dahin 12-TET in diesen Slots), ein Step-Knopf und
Clock-Sync für den Autoplay, MPE und Controller, Pegelausgleich der Landschaft (Aurora), die Frage der GPU-Klasse im
Plugin, der Name.

## Phase 2: das Klangmodell (gebaut, 12.10.2026)

Nach der Regel "pro Partial im Spektrum auf die GPU, rekursiv auf der Summe auf die CPU":
- **Noctuarys Spektralformel pro Partial** (`Spectrum.h`, für GPU und Host dieselben Funktionen): Tilt, Odd/Even,
  Brightness als Fenster über bis zu 200 Harmonische, Inharmonic als steife Saite (die Zahl der Harmonischen unter der
  Grenze mit Streckung gerechnet), Presence (Glocke 1,8 bis 5,6 kHz, nur nah), Pad Low Cut (12 dB/Okt). Die Stimme
  wird wie in Noctuary **nach** der ganzen Form normiert: Farbe ändert den Klang, nicht den Pegel.
- **Das Stimmfilter als exakter Frequenzgang am Partial:** LP 6/12/24, HP 12, BP 12, Notch, Peak, Formant (u-o-a-e-i,
  die Ecke wählt den Vokal); Ecke mit Key Track, Hüllkurve (Oktaven) und eigener Drift.
- **Air:** 20 % der Partials als Rauschband um ein Vielfaches des Grundtons, mit 40-Cent-Flackern in Tonhöhe und Pegel.
- **Raum auf der CPU pro Stimme:** Breath (die Ebene wandert), Doppler (näher = höher), Tide (das ganze Instrument
  neigt sich im Minutentakt); Seitenstrom für alle Drifter, damit sie die Würfel der Noten nie verschieben.
- **Brain 2** (Noctuarys zweiter Dirigent, Free, eigener Strom, Grundton = erster + Intervall, tief im Raum), **Step**
  als Schalter für den Autoplay, **Timbre-Skala** (Sethares) aus dem Spektrum der Wolke.
- **Landschaftsbibliothek** (fünf erzeugte, beim Start gebaut): Aurora, Shepard, Tides, Rain, Choir; **Pegelausgleich**
  nach ihrem RMS (bis 18 dB). Wave hat einen Boden (80 % Tiefe).

**Gemessen [M]:** Probe-Selbsttest 18/18 (u. a. LP12 eine Oktave über der Ecke 0,02425 = Soll, Spitze bei Q 4
genau 0,4; Presence und Low Cut auf vier Stellen; Brightness und Odd/Even unter 1e-6; Streckung exakt; Air +3,2 dB
statt +3,0; Ferne exakt); Engine-Selbsttest 14/14 (Brain 2 spielt, Timbre-Skala mit 9 Stufen, Aurora 2,1 dB neben
der nackten Wolke, alles gleichzeitig endlich und vernünftig). Volle Last (16 Stimmen, Air, Formant, Brain 2,
Breathe): 0,077 ms GPU pro Block, 3 % des Budgets. Hörbeispiele: `work/renders/phase2/` (choir, shepard, rain,
air-formant-brain2).

**Offen:** binaural pro Partial (Brown/Duda) mit der Quest; eigene Bilder und Aufnahmen als Landschaft brauchen den
Dateidialog der GUI (Phase 3); Z-Plane und Strike als Modalsynthese.

## Phase 3: die Oberfläche (gebaut, 12.10.2026)

Auf dem Rahmen der Familie (`Plugin/Frame.h/.cpp`, unverändert aus Totality, in allen fünf Repos gleich):
- **Skin** "Aurora bei Nacht" (Nachtblau, Türkis, Violett), Fenster 1280 x 860. Seit 15.10.2026 als Logo die
  Obertonleiter (zehn Partials als Lichter auf Stielen, Höhen wie die Obertonreihe; vorher ein Ring aus Partials, der
  Phosphenes Mandala zu ähnlich war) und als Bild hinter dem Panel ein Wolkenmeer bei Nacht unter Strömen aus
  Lichtpunkten (`Plugin/Resources/backdrop.jpg`, Gemini 2.5 Flash Image wie bei den Geschwistern).
- **Kopfzeile:** Autoplay Start/Stopp (Leertaste im Standalone), Modus, Step, Skala, Grundton, Bewegung, Landschaft;
  Statuszeile (Stimmen nach Herkunft, Grundton, GPU-Zeit, Latenz, Gerät, Unterläufe); Rückgängig, Wiederholen,
  Hilfe (F1, Text aus `Plugin/Resources/chapters.txt` plus "Diese Seite" mit allen Parametern), Einstellungen.
- **Wolkenbild:** jede klingende Stimme als Spektrum ihrer Wolken (gespielt grün, Dirigent violett, Brain 2 blass),
  daneben die Landschaft mit der Stelle, an der die Stimmen in ihrem Loop sind.
- **Seiten:** Cloud, Motion, Filter (mit Hüllkurve), Tuning, Conductor (mit Brain 2), Space (mit Master); jeder
  Regler mit dem Rechtsklick-Menü der Familie (MIDI Learn, Vergessen, Standard); Rückgängig über die Gesten.
- **MIDI:** Haltepedal (CC 64), CC 74 auf das Filter, MIDI Learn (im Zustand gespeichert).
- **Bildschirmfotos** ohne Desktop-Steuerung: `ADDITIV_SHOT=<Präfix>` (und `ADDITIV_SHOT_AFTER=<s>`) fotografiert jede
  Seite und beendet den Standalone; `ADDITIV_LOG=<Datei>` schreibt sekündlich den Echtzeitpfad mit.

**Gefunden und behoben [M]:** Im Standalone wuchsen die Unterläufe auf etwa einen pro Sekunde (61 nach 60 s), obwohl
der langsamste Block nur 2 ms brauchte: WASAPI lieferte zwei Callbacks (je 480 Samples) direkt hintereinander, und
der Ring stand dann bei 448. Der Vorlauf deckt jetzt zwei Hostblöcke plus 10 ms (Latenz 32 ms bei 480, 21 ms bei
256): 0 Unterläufe in 31 s, der Ring nie unter 576. Dazu füllt `prepare()` den Ring vor dem ersten Callback.

**Gemessen:** Doccheck, Selbsttest 14/14, Probe 18/18, **pluginval Stufe 10 mit dem Editor** bestanden.

## Phase 4: Modulation und Presets (gebaut, 12.10.2026)

- **Modulation** (`Modulation.h/.cpp`): vier LFOs (Sine, Triangle, Ramp Up/Down, Square, Random, Steps; Raten im
  goldenen Schnitt, 0,05 bis 0,21 Hz; keine harten Kanten: Rücksprünge über 7 % des Zyklus, Steps gleitet), zwei
  Hüllkurven aus der Formenbank (neu ab einer Note in die Stille), vier Makros, Modrad, Druck, Anschlag, eine langsame
  Zufallskurve; **Matrix aus 16 Zeilen** als Text im Zustand (`lfo1>cutoff:0.35;...`), Tiefe im Reglerweg (normiert,
  mit Skew). Auf der Seite *Modulation* die Quellen und die Matrix; jeder modulierte Regler zeigt einen Ring.
- **1024 Presets** in 16 Gruppen zu 64 (`Tools/presets/make_presets.py` schreibt `Core/src/Presets.inc`): Rezepte je
  Gruppe, Namen aus Adjektiv x Nomen, gemeinsame Bedeutung der Makros (1 Bewegung, 2 Helligkeit oder Filter, 3 Air,
  4 Tiefe), Modrad und Druck. **Pegel:** `additiv_presets` misst jedes Preset (Akkord 50/57/62/66, ohne Dirigent)
  an seiner **lautesten Sekunde** (Glocken und Glas würden sonst im Abklingen gemessen), `make_presets.py` zieht alle
  auf −20 dBFS (Master −18 bis +12 dB). Formant brauchte ×3 (die Chöre lagen 18 dB unter dem Rest).
- **Gefunden [M]:** `f(x, 0)` strich bei ganzen Zahlen die Endnullen (180 Hz Cutoff wurde 18 Hz): 24 Presets mit einem
  Cutoff eine Dekade zu tief (sieben Dark Drones bei −60 bis −84 dBFS, also stumm), 9 mit zu schmalem Air-Band.
  Nach der Korrektur liegt die ganze Bank vor dem Ausgleich bei −30 bis −4 dBFS (Median −20,3), also im Bereich der
  Korrektur.
- Programmwechsel mit Bank (CC 0 = Gruppe), Kopfzeile mit Vor/Zurück und Menü nach Gruppen. **Nach einem
  Presetwechsel füllen beide Dirigenten ihren Akkord im Drittelsekundentakt** (`Engine::requestFill`, Noctuarys
  `requestBrainFill`): Velvet Dreaming hatte nach 12 s eine Stimme, jetzt nach 8 s sechs.

## Phase 5: Familie (gebaut, 12.10.2026)

- **Familien-Jam** (`Plugin/Jam.h/.cpp`, unverändert aus Totality): als Folger hält der Dirigent seinen Grundton auf
  der Tonklasse des Leaders (die nächste Note dieser Klasse, die eigene Stimmung bleibt) und öffnet oder schließt
  Helligkeit (±40 %) und Cutoff (±1 Oktave) mit der Energie des Leader-Abschnitts; als Leader meldet er Grundton und
  Skala. Selbsttest: Grundton 30 s auf der Tonklasse des Leaders gehalten, obwohl Root Move 1 wandern will.
- **Headset** im Bridge-Modus: Die Quest hat keine CUDA-Karte, das Instrument bleibt am PC; die Hände kommen über
  `frame::Headset` (Port 9100 bis zum Namen, `portOf` in `Frame.cpp` aller fünf Repos). Linke Hand Höhe = Makro 2
  (Helligkeit), rechte = Makro 1 (Bewegung), Pinches: Play/Stop, Step, Brain 2, beide = nächstes Preset.
- Hilfe: Kapitel *Modulation*, *Presets*, *Family jam and headset*.

## Phase 6: Release (vorbereitet, 12.10.2026; nicht veröffentlicht)

- `release`-Preset: statische Laufzeit, **Code für jede RTX-Generation** (sm_75, 86, 89, 120 und PTX 120). Eine Karte
  ohne Code oder ein Rechner ohne NVIDIA-Treiber wird mit Namen in der Statuszeile gemeldet (`cudaFuncGetAttributes`
  in `CloudEngine::open`); das Setup fragt nach, wenn `nvcuda.dll` fehlt oder AVX2.
- `Deploy/build_release.ps1`: Build, ctest, pluginval mit zwei weiteren Seeds, Bildschirmfotos (`Tools/manual/
  make_shots.py`, stumm über `ADDITIV_MUTE`, `ADDITIV_PRESET`, `ADDITIV_SHOT_FULL`), Handbuch (`make_manual.py`: Text
  aus `chapters.txt`, Tabellen aus `empy_render --dump-params`, Anhang MIDI/Presets/Anforderungen), Stage,
  dumpbin-Prüfung (keine MSVC- oder CUDA-Laufzeit-DLL), SHA256SUMS, Zip, Inno-Setup 7. Unsigniert.
- `README.md`, `docs/RELEASE_NOTES.md`, Version 0.6.0.

**Der Name (12.10.2026): Empyrean**, der höchste Himmel, der aus Licht (Aurora war besetzt: NI/iZotope, UVI;
deutsche Namen mag der Nutzer nicht). Gefunden nur ein Zebra-Soundset, EmpyreanFX (Gitarren-Patches) und Empyrealm
(Kontakt). Was ausgeliefert wird, heißt so: `Empyrean.exe`, `Empyrean.vst3` (Code `Empy`, `com.reneweller.empyrean`),
`empy_render`, `Empyrean-Manual.pdf`, `empyrean.ico`, Einstellungen und Jam unter "Empyrean", Headset-Port **9105**
(`portOf` in allen sechs Kopien von `Frame.cpp`). Der Code behält den Arbeitsnamen des Ordners (Namensraum `additiv`,
`ADDITIV_`-Optionen und -Variablen, `additiv_presets`, `additiv_selftest`), wie Noctuarys Code `ambient` heißt.

**Entschieden (12.10.2026):** AGPL mit der statisch gelinkten CUDA-Laufzeit ist für den Nutzer in Ordnung.
**Offen:** der Startunterlauf (siehe Phase 9: fremde GPU-Last); Hören auf älteren Karten (nur RTX 5090 gemessen);
Linux optional.

## Phase 7: Echos, Räume und der modale Hall (gebaut, 12.10.2026)

Auf die Frage des Nutzers, ob Delays und Reverbs schon drin seien: Nur der Far-Reverb war es. Jetzt:
- **CPU, aus Noctuarys `Effects.h`** (schon kopiert, bisher nicht verdrahtet): zwei `StereoDelay` in Serie auf der
  nahen Ebene (Mix und To Far je Block über den Block gerampt), `EarlyRoom` aus den trockenen Stimmen (Quelle: mittlere
  Entfernung nach Lautstärke) und ein kleiner `Reverb` als Near-Reverb; vor dem Far-Reverb der `Diffuser`, danach
  `Unmask` (Seitenkette: die nahe Ebene); Far Mode (Standard Colourless), Pre-Delay, Freeze.
- **GPU: der modale Hall** (`Cloud.cu`, `modalKernel` und `modalReduceKernel` im selben Graphen hinter der Reduktion,
  Eingang: der fertige Block der Stimmen auf der Karte, Ausgang: Kanäle 5 und 6 des Blocks). 65.536 komplexe
  Einpoler, ein Thread je Mode, 128 Samples sequentiell, Summe pro Sample über Warp-Shuffles und dann die acht Warps in
  fester Reihenfolge (bitgleich). Jede Mode errechnet Ort, Abklingen und Phasen jeden Block aus einem Hash und dem
  Kopf: gleichmäßig in Hz zwischen Lowest und 16 kHz, oder (Anteil Tuned) auf einer der Tonhöhen der Skala (bis 128,
  vom Engine aus `frequencyOf` mit Tide) mit Dreiecksstreuung. Drift: eigene Sinusrate k/1000 Hz, die Uhr springt bei
  1000 s ohne Sprung. T60 unter 250 Hz = Decay, jede Oktave darüber 2^Damping schneller. Gewicht sqrt(2(1-r²)/N):
  weißes Rauschen kommt mit seiner Leistung heraus; gestimmte Moden zusätzlich mal sqrt(Band der Tonhöhen / Bereich),
  höchstens 2. Freeze: r = 1 − 1,2e-7 (24 Minuten auf −60 dB), Eingang zu. Ruht (on = 0), sobald der Pegel einen
  ganzen Block 0 war: dann gelöscht, kaum GPU-Zeit.
- Seite **Effects** (Echo, Echo 2, Room, Modal Hall), Far-Gruppe auf Space erweitert; Kapitel in der Hilfe.
- Presets: Effekt-Profil je Gruppe in `make_presets.py` (`FX`), gewürfelt nach dem Rezept, damit die Klänge bleiben.

**Gemessen [M]:** GPU 0,06 → 0,085 ms pro Block mit Hall (1 % des Budgets). Akkord 50/57/62/66, Hall auf 1: Tuned 0
und 1 innerhalb 1 dB (die Dichte-Gewichtung trägt), Decay 40 s nicht lauter als 8 s, nur länger. Selbsttest 24/24:
Abklingen 29,4 dB in 1 s bei Decay 2 s (Soll 30), gestimmt 15 Cent neben der 4. Harmonischen −69,5 dB gegen −19,0 dB
ungestimmt, Freeze hält auf 0,8 dB, das Echo kommt nach 0,5 s, alles zugleich endlich und vernünftig. Der LFO-Test
misst jetzt ohne Effekte (Raum und Hall glätten die Pegelschwankung, die er prüft).

## Phase 8: Kräfte auf den Partials (gebaut, 12.10.2026)

Aus den Ideen des Nutzers (spektrale Attraktoren, große FDNs, Dispersion) die zwei, die zur Architektur passen; das
große FDN nicht, denn der modale Hall ist schon eine Summe von Moden (Begründung im Gespräch).
- **Feld** (`SpectralField` in `Cloud.h`, `fieldStep` in `Cloud.cu`): der Versatz jedes Partials in Cent liegt im bisher
  freien `run.w`, bei einer neuen Note 0. Jeden Block bewegt er sich auf das zu, was zieht (Scale: der nächste Ton der
  Skala per binärer Suche; Swarm/Scatter: bis zu acht Attraktoren, Glocke der Breite Radius), höchstens Pull Cent pro
  Sekunde und nie darüber hinaus (Relaxation 1 − e^(−Pull/Radius·dt)); die Feder zieht mit e^(−Spring·dt) heim. Feld
  aus: in etwa einer Sekunde heim. Attraktoren: acht Lorenz-Systeme auf dem Host (Mittelpunktregel, Schritte ≤ 0,002),
  x über sieben Oktaven ab 60 Hz, z als Gewicht.
- **Gefunden [M]:** Scale ließ den Pegel um 3 dB fallen und das Schimmern verschwinden: Die Mitglieder einer
  Harmonischen fielen auf eine Frequenz und summierten sich zu einem Sinus mit eingefrorener Zufallsamplitude (im Mittel
  gleich, aber die tiefen Harmonischen tragen fast alles). Jetzt behält jeder Partial seinen Platz in der Glocke,
  gestaucht auf 2 Cent je Einheit: 0,3 dB Unterschied, die Wolke schwebt weiter.
- **Dispersion**: die Uhr jedes Partials um D·(1 − Tonhöhenlage) Sekunden später (negativ: um |D|·Lage), für Lanes und
  Landschaft über `c.t`, für den Pegel der Stimme über einen Ring auf der Karte (`gainRingKernel`, 1024 Blöcke je
  Stimme, beim Notenstart gelöscht). Die Stimme bleibt nach dem Ende ihrer Hüllkurve |D| länger aktiv.
- **Atmender Hall**: das T60 jeder Mode wandert um bis zu eine Oktave (Sinus k/1000 Hz).
- Presets: `MOTION_FX` in `make_presets.py`. Hilfe: Absätze im Kapitel Motion, Breath im Kapitel Effects.

**Gemessen [M]:** GPU unverändert 0,06 ms pro Block. Selbsttest 27/27: Scale 20 Cent neben der 4. Harmonischen −11,0 →
−69,9 dB bei −30,6 / −30,0 dBFS; Swarm/Scatter −24,7 / −25,0 gegen −25,1 dBFS, gleicher Seed bitgleich; Dispersion 1 s:
Grundton −35,7 → +46,7 dB, 8. Harmonische +19,4 → −51,2 dB zwischen 0,36–0,56 s und 0,66–0,86 s.

## Phase 9: fünf Ideen für die GPU (gebaut, 12.10.2026)

Meine Vorschläge, vom Nutzer alle angenommen ("Das klingt alles super, bitte baue es ein"):
- **Binaurale Partialwolke** (`Binaural` in `Cloud.h`, `earGains` in `Cloud.cu`): pro Partial und Block je Ohr ein
  komplexer Gain aus Kopfschatten (Brown/Duda: Pol und Nullstelle, α nach dem Einfallswinkel), Ohrmuschel (fünf
  Reflexionen nach Azimut und Elevation, normiert auf ihre mittlere Leistung 2,625) und Woodworth-Laufzeit als Phase;
  das Paar auf die Leistung des Pan-Gesetzes normiert. Gains jetzt drei `float4` pro Partial (der dritte: die
  Kosinusanteile der nahen Ebene), `ampQ` für ihre Rampen; der Render-Kernel rechnet binaural `__sincosf` statt
  `__sinf`. Azimut = Platz · (90° … 180°) + Orbit, Elevation aus dem Hash ±60° · Height. Die ferne Ebene bleibt
  (Far-Reverb). Master: kein Mono-Bass, Breite 1. `keptByPresets`: Presets schalten Binaural nie.
- **Gedächtnis**: vier Geisterstimmen hinter den 16 (`EngineConfig::ghosts`, GPU-Stimmen 20). `stopNote` fängt die
  Stimme beim Loslassen (`captureGhost`: freier Platz oder der leiseste), eigener Seed und eigener Zufallsstrom.
  Hüllkurve: Anstieg über Bloom, −60 dB über Memory Time; Breite ×(1 + 3 Blur x), Drift + 12 Blur x, Helligkeit
  ×(1 − 0,7 Blur x), Entfernung gegen 1 mit Sink. Im Bild als vierte Farbe.
- **Selbststimmung** (CPU): je Paar von Stimmen die Harmonischen 1–8 × 1–8, Abstand d in Cent unter Reach: Gewicht
  Lautstärke · Glocke · 1/(ha·hb); jede Stimme geht die Hälfte des gewichteten Mittels (Rate 1 − e^(−2·Gravity·dt)),
  Anchor zieht zurück. Der Ton beginnt auf der Skala.
- **Gemalter Hall**: jede Mode durch die Landschaft bei ihrer Frequenz, eigene Uhr (gewickelt auf die Schleife der
  Landschaft), Ausgleich 1/RMS der Landschaft.
- **Spektrales Echo** (`spectralEchoKernel`, ein CUDA-Block mit 512 Threads): Verlauf und Overlap als Ringe auf
  denselben Indizes, Hann-Fenster, eigene FFT (Bitumkehr, zehn Stufen, `sincospif`), links als Real-, rechts als
  Imaginärteil, je Band und Ohr eine Verzögerungsleitung (2048 Blöcke) mit Rückkopplung; Overlap-Add mit
  1/(N · 3/8 · N/n). Latenz 1024 − 128 Samples, von der Zeit abgezogen. Kanäle 7 und 8 des Blocks (jetzt 8).

**Gemessen [M]:** Selbsttest 32/32 (Zahlen in den Release Notes). GPU mit vier Stimmen: binaural +0,005 ms,
spektrales Echo +0,01 ms. Volle Last (60 s Dirigent mit 12 Stimmen und Brain 2): 0,097 ms pro Block, mit allem
zugleich (binaural, Gedächtnis, spektrales Echo, gemalter Hall, Swarm, Self-Tune, Dispersion) 0,165 ms, 6 % des
Blocks; die vier Geisterstimmen kosten wie vier Stimmen. Pegel: binaural 0,3 dB
unter Stereo, gemalter Hall 0,1 dB, spektrales Echo bei Mix 0,6 +1 dB. Ein gelöster Akkord: 20 s danach −33,8 statt
−49,4 dBFS mit Gedächtnis.

**Der Startunterlauf, aufgeklärt [M]:** Das Log teilt den langsamsten Block jetzt in Vorbereitung, Warten auf die
GPU und Effekte. Vorbereitung bis 1 ms, Effekte bis 0,65 ms, das **Warten auf die GPU bis 2,5 ms**, obwohl die Kernels
0,1–0,17 ms rechnen: Die Karte war zu rund 50 % mit anderen Programmen belegt (Firefox, Teams u. a.; `nvidia-smi`),
deren Grafikarbeit sich gröber unterbrechen lässt als die CUDA-Last aus Phase 0, auch gegen die REALTIME-Klasse. Im
Betrieb hält der Ring (nie unter 960 von 1536), nur beim Start nach den zwei WASAPI-Callbacks wird es eng. Nebenbei:
Denormals jetzt je Block aus (`NoDenormals`, FTZ und DAZ), denn der Worker ist nicht JUCEs Audio-Thread. Mehr
Vorlauf nur beim Start: in Phase 10 gebaut (Startreserve).

## 0.9.1: das Bild zeigt die Partials (12.10.2026)

Der Nutzer: "die Partials selbst bleiben immer konstant". Gemessen bewegen sie sich (Bandenergie einzelner
Harmonischer über 20 s: ruhende Wolke 1–4 dB in den oberen Harmonischen, Breathe 5–8 dB, Velvet Dreaming mit Aurora
15–22 dB) -- aber das Bild zeichnete nur die Form jeder Stimme aus den Reglern. Jetzt schreibt der Update-Kernel für
jeden achten Partial (f, Amplitude) in `dSnap`; `launch()` kopiert sie, wenn verlangt, hinter dem Graphen vor dem
Ereignis, auf das `wait()` wartet; das Engine verlangt es etwa alle 33 ms und reicht die 51.200 Punkte über einen
Mutex weiter, den der Worker nur versucht (nie wartet). Die Ansicht: Punkte (log f quer, −100 … −35 dB hoch) in
einem Glühpuffer, der je Bild auf 80 % fällt. Selbsttest 33: 2499 von 2560 Punkten der klingenden Stimme, alle an
ihren Harmonischen, keiner bei stummen Stimmen.

## Phase 10: lebende Systeme (gebaut, 12.10.2026)

Der Nutzer fragte nach weiteren GPU-Effekten und nach Noctuarys Modulatoren ("Da hatten wir doch bei Noctuary so
eine Art 2D Game-Of-Life-Modulator") und nahm alle Vorschläge an außer Paulstretch ("eher was für Samples": die
Töne werden ohnehin neu erzeugt). Je Baustein ein Commit, ein Selbsttest, die Hilfe.
- **10a Noctuarys Quellen:** Lenia 1–4 (32×32 auf der CPU, Ringkern, Wachstumsglocke), Lorenz X/Y/Z und Rössler
  X/Y/Z (RK4, Chaos Period), Kuramoto 1–4 (Coherence, Coherence Rate), Beat (die Schwebung der zwei tiefsten Töne),
  Cascade (die Erregung des Dirigenten, Brain Cascade), Root Age, Note, Distance, Random per Note; jede Matrixzeile
  mit einer **Via**-Quelle, die die Tiefe skaliert (Text `src>ziel:tiefe:via`). Die Attraktoren laufen beim Reset
  25 Lorenz-Zeiteinheiten vor, Lenia zwanzig Generationen: sie leben ab der ersten Sekunde.
- **10b Lebende Landschaft** (`LifeField`): Lenia oder Gray-Scott (Turing; Character läuft über Pearsons Karte
  Flecken → Mitose → Koralle → Würmer → Labyrinth) auf einem 256×128-Torus auf der GPU, jeder Partial liest ihn bei
  (log f · Zoom, Platz in seiner Wolke). Ausgleich exakt aus E[(1−d+d·v·m)²] = 1, über 0,5 s geglättet; Neusaat,
  wenn das Feld stirbt oder voll ist. −25,4 (Lenia) / −25,5 (Turing) gegen −26,3 dBFS frei.
- **10c Partial-Sync:** je Partial ein Kuramoto-Oszillator mit dem Mittelfeld seiner Harmonischen; Faktor
  (1 + d sin θ)/√(1 + d²/2). Gekoppelt pulsiert die 8. Harmonische mit 10,7 statt 5,8 dB.
- **10d Selbsthören** (`listenKernel`, 1024er-FFT der Stimmen je Block): Loudness, Heard Brightness (Schwerpunkt
  auf der log-Achse), Change (Fluss), Roughness (Sethares über 8×8 Harmonische, ×16). Eine Quinte 0,08, eine
  Sekunde 0,54.
- **10e Schwarm** (Field-Modus Flock): Boids je Wolke -- Zusammenhalt und Ausrichtung auf den Versätzen, Abstand
  ab dem halben Heimatabstand (die Reihenfolge bleibt), Lorenz-Wind, Feder, Reibung. Die Mitte der 4. Harmonischen
  wandert 42,8 Cent, die Breite bleibt 7,1 Cent.
- **10f Welle** (`Ripple`): gedämpfte Wellengleichung auf 256 Zellen über das Spektrum (CPU, Leapfrog,
  reflektierende Enden), jede neue Note schlägt sie an ihren ersten acht Harmonischen an (Ricker-Wavelet in Ruhe);
  jeder Partial liest (1 + d·tanh u) · Ausgleich. Die 4. Harmonische schwankt 8,5 statt 3,3 dB.
- **10g Reservoir:** Echo-State-Netz mit 2048 Neuronen auf der GPU (Kopplung gleichverteilt ±√(3/N), Spektralradius
  = Chaos; ein Warp je Zeile; Leck aus Memory), gefüttert mit acht Größen (was es hört, Beat, Note, Kaskade, Lenia),
  jede gegen ihre letzte halbe Minute standardisiert; vier feste Auslesungen sind Reservoir 1–4. Läuft nur, wenn
  eine Zeile es liest (oder das Fenster offen ist); 6,5 µs pro Block. Ausschlag 0,50 mit dem Dirigenten, allein
  ruhig bei Chaos 0,6 (0,0001), frei laufend bei 1,4 (0,74), bitgleich wiederholbar.
- **10h Melt** (`meltKernel`, ein CUDA-Block mit 1024 Threads, eigene 4096er-FFT): alle 512 Samples ein Frame; jedes
  Band hält seine Energie (Melt Time bis −60 dB), das gehaltene Spektrum diffundiert auf der log-Achse (Spread:
  Oktaven nach einer Sekunde) -- jede Quelle streut ihre Energie gleichmäßig über 1 + 2κx Bänder, gesammelt über
  Präfixsummen in Double-Float (zwei floats; double läuft auf GeForce mit 1/64 und kostete 25 µs), also
  energieerhaltend. Ein Phasen-Vocoder singt es neu: jedes Band auf der gehörten Frequenz, beim Schmelzen zur
  eigenen Mitte gezogen, je Ohr leicht verstimmt. Kanäle 8/9 (`kCloudOutChannels` 10), Mix und To Far. Nachklang
  −26,7 dBFS bei −26,4 gehalten (−28,4 mit Spread 1,5); zwischen den Harmonischen −75,4 → −8,7 dB; 7 µs pro Block.
- **Die Quellen live** (`SourcesView`): auf der Modulation-Seite alle 42 Quellen als Spur ihrer letzten vier
  Sekunden, gefärbt nach Art; das Engine veröffentlicht sie je Block (`sourceValue`).
- **Presets:** 2048 in 32 Gruppen. Die ersten sechzehn bekamen Phase 10, wo sie passt -- gezogen nach allem
  anderen, sodass jeder bisherige Wert bleibt (761 von 1024: Melt 173, Ripple 160, Sync 140, Life 112, Reservoir 72,
  Flock 28; 516 neue Matrixzeilen). Sechzehn neue Gruppen, die lebenden Systeme, beginnen je mit dem Klang einer
  der ersten sechzehn und stellen ein System nach vorn: Living Fields, Turing Gardens, Synchrony, Flocks, Ripples,
  Melting Spectra, Spectral Cascades, Resonant Halls, Ghosts & Memory, Strange Attractors, Reservoir Minds,
  Self-Listening, Gravity Wells, Smear & Dispersion, Lattices & Beats, Night Gardens. Alle neu gemessen und auf
  −20 dBFS gebracht.

- **Der Startunterlauf, behoben:** vor dem ersten Callback rendert der Worker eine zweite Latenzlänge voraus
  (`startReserve_`) und ersetzt sie nicht, bis die Callbacks sie verbraucht haben; ein startendes Gerät darf
  seinen Puffer in einem Schub füllen. Die gemeldete Latenz bleibt; ein Ereignis in den ersten ~32 ms kommt um den
  unverbrauchten Rest der Reserve später (`stampFor`), nie zu spät. Die Rauheit (bis 15.000 `exp` je Block bei 16
  Stimmen) wird nur jeden achten Block gerechnet.

**Gemessen [M]:** Selbsttest 44/44. GPU mit zwölf Stimmen (probe bench): 0,091–0,103 ms ohne, +0,0065 ms Reservoir,
+0,007 ms Melt. Standalone mit Nightblooming Grove (Life, Melt, Gedächtnis, modaler Hall) neben einer zweiten
GPU-Last: langsamster Block 0,6–0,9 ms (Vorbereitung bis 0,39, GPU-Warten bis 0,57, Effekte bis 0,26); drei Starts
ohne Unterlauf, Mindestfüllung in der ersten Sekunde 960–1056 von 1536 (vorher 224 und ein Unterlauf).

## Phase 11: Morph und Journeys (gebaut, 12.10.2026)

Der Nutzer: "In Noctuary haben wir doch auch diese tolle Möglichkeit zwischen Presets zu interpolieren und ganze
Nächte lang Sleep-Konzerte laufen zu lassen ... bei rein additiver Synthese doch viel einfacher". Noctuary hat drei
Dinge: Morph A/B (zwei Parameter-Schnappschüsse, lerp), die Preset-Karte mit Route, und Journeys (Presets in Reihe,
Verweildauer und Überblendung gewürfelt, zyklisch) -- und dort überblendet ein Journey-Schritt zwei ganze Engines als
Audio, weil Samples, Wavetables und Räume nicht interpolierbar sind. Hier wird jede Stimme jeden Block neu als
Spektrum geschrieben, also kann der Klang selbst wandern. Ganz trivial ist es nicht: Auswahlparameter springen, und
die Matrix ist keine Zahl. Deshalb:
- **Morph.h** (`MorphState`: alle Parameter, der Anteil des Filters, bis 64 gewichtete Matrixzeilen; `blendMorph`):
  Zahlen gleiten in ihrer eigenen (schiefen) Normierung; Auswahl und Schalter wechseln in der Mitte; eine Auswahl
  mit Menge (Motion/Landscape über Motion Depth, Life über Life Depth, Field über Pull) blendet ihre Menge aus und
  ein -- von Off sofort mit Menge 0, nach Off am Ende, zwischen zweien in der Mitte bei 0. Der **Filter** wird pro
  Partial als Verstärkung gelesen (`filterGain` auf der GPU) und jetzt mit `filterMix` potenziert (in Dezibel
  geteilt): ein Filter blendet aus, ein anderes ein, ohne Sprung -- das ist der additive Vorteil. Die **Matrix**
  als zwei gewichtete Zeilensummen (gleiche Quelle/Ziel/Via werden addiert).
- **Engine**: jeder Block zuerst `updateSound()`: Ziel = die Regler und die Matrix, oder mit Morph an die Mischung
  der Slots A/B bei A - B; ein Gleiten (`beginMorph`) startet von dem, was der Block davor gespielt hat (`sound_`),
  und folgt dem Ziel mit Smoothstep. Der Host bittet um das Gleiten und setzt dann das neue Preset: die Regler zeigen
  das Ziel, ein gedrehter Regler ändert es. Kein Wettlauf: der Worker nimmt den Ausgangspunkt aus seinem eigenen
  letzten Block. Während des Gleitens folgt die Stimmung langsam (ein Zehntel des Gleitens, höchstens 6 s): eine neue
  Skala wird erreicht, indem die Partials hineingleiten. Morph an/aus und neue Slots gleiten ebenfalls (Glide, mind.
  1 s). Neue Parameter `morph_active`, `morph_pos`, `preset_glide` (keptByPresets, keine Modulationsziele).
- **Journey.h** aus Noctuary übernommen (Format, Zeitparser jetzt mit strtod, Player mit Seed); ein Schritt ist hier
  ein Gleiten. **68 eingebaute Journeys** (`Tools/presets/make_journeys.py` → `Journeys.inc`): je Gruppe eine
  Journey (zwölf verschiedene Gesichter per Farthest-Point-Sampling, als Kette nächster Nachbarn; 6–12 min, gleiten
  1–2:30) und eine Night (die zwölf ruhigsten; 10–18 min, gleiten 2–4), dazu vier Whole Nights über Gruppen (24
  Presets, die ruhigsten je Gruppe; 12–20 min, gleiten 2:30–5; eine Runde ≈ 6:24 h). Merkmale aus den
  Preset-Parametern (es gibt hier keine vermessene Bibliothek).
- **Oberfläche:** neue Seite Journey (Glide, Morph, A - B; Slots A/B aus dem Klang oder einem Preset; Journey-Box mit
  Abschnitten, Start/Stopp, Status, + dieses Preset, Speichern nach Documents/Empyrean/Journeys); die Statuszeile
  zeigt die Journey oder das Gleiten. Slots im Zustand gespeichert.
- **empy_render** `--journey NAME|DATEI --journey-seed N`, schreibt die WAV fortlaufend und ab einer Stunde in
  Stundendateien: eine ganze Nacht lässt sich offline rendern (11-fache Echtzeit).

**Gemessen [M]:** Selbsttest 47/47. Gleiten 4 s von Breathe/ohne Filter/12-TET zu Aurora/LP 24/JI: Tilt auf dem Weg
exakt in der Normierung, Filteranteil exakt w, Motion Depth 0,01 in der Mitte (Motion und Skala wechseln dort),
größter Pegelschritt zwischen Zehntelsekunden 3,2 dB gleitend gegen 0,8 vorher und 5,6 danach (Aurora selbst);
am Ende exakt das Ziel. A/B bei 0,25: Tilt und Filteranteil exakt. Journeys: 68 lesbar, 864 Schritte, alle Presets
bekannt, gleich geseedete Player gleich.

## Phase 12: Schlafkonzerte (gebaut, 12.10.2026)

Der Nutzer: "noch 1024 Presets ... mit vielfältigen Modulationen, die sich besonders für Schlafkonzerte eignen".
Sechzehn Gruppen (Lullabies, Slow Breath, Deep Sleep, Dream Drift, Moonlight, Night Rain, Ocean Sleep, Cradle,
Starfield, Hypnagogia, Twilight Choirs, Warm Drones, Snowfall, Forest Night, Slow Tides, Afterglow), je vom Klang
einer der ersten sechzehn aus. `sleepy()`: Attack ≥ 6 s, Release ≥ 15 s, Brightness ≤ 0,6, Tilt ≥ 1,4, Shimmer ≤ 0,5,
Presence ≤ 2 dB, Motion Rate ≤ 0,6; Dirigent an, Chords, alle 90–300 s, 3–5 Stimmen, Lage 31–74; Echo-Feedback ≤ 0,5,
Echo ≤ 0,15, spektrales Echo ≤ 0,2, Ripple ≤ 0,4 und langsam, Sync ≤ 1 Hz. Schlafmodulationen nach Art (`SLEEP_POOLS`,
je Art ihre Quelle einmal eingestellt): Atem (LFO 4 Sinus 0,08–0,12 Hz), Gezeiten (LFO 3 Sinus/Dreieck 0,002–0,008 Hz),
Wanderung (LFO 2 Random 0,01–0,04 Hz), Chaos (Periode 10–30 min), Kuramoto (Kohärenz 0,5–0,9), Lenia (0,5–2 Hz),
Nacht (Root Age → dunkler, ferner, mehr Gedächtnis und Melt), Beruhigen (Loudness → dunkler, Change → weniger
Shimmer, Roughness → Self-Tune), Reservoir (Gedächtnis 10–30 s); Via aus Root Age, Lenia 2, Kura 3. Slow Breath teilt
eine Atemrate zwischen Raumatem, Partial-Sync und LFO 4. 1024 Presets mit 11–15 Zeilen (Mittel 13,8), 878 Zeilen mit
Via; die alten 2048 unverändert. Zwei neue Whole Nights (Lullabies, Deep Sleep): 102 Journeys. Gemessen wurden nur
die neuen (`additiv_presets --from 2048 --to 3071`, führt in levels.json zusammen).

## Phase 13: spektrale Mittel und bewegte Bilder (gebaut, 13.10.2026)

Ein Text über fehlende Funktionen im Vergleich zu Razor/Alchemy/Loom; meine Einschätzung: Phase und Partial-Limit
bringen in einer Wolke nichts, mitschwingende Resonanz ist der modale Hall schon, gebaut wurde:
- **13a Stretch und Shift:** `partialRatio(h, B, s)` = h^s · √(1+Bh²) überall (GPU, Obertonzahl per Halbierung,
  Leistungssumme, Timbre-Skala, Selbststimmung, Rauheit); Shift in Hz auf jeden Partial, unter 20 Hz ausgeblendet.
  Der kohärente Anfang wurde gebaut, gemessen und verworfen: phasengleiche Mitglieder addieren sich in der Amplitude
  (+19 dB über eine halbe Sekunde, ein Sinus statt eines Impulses; als Sinus gestartet sogar Sägezahn-Crest 4 dB).
- **13b Spektraler Kamm:** erhöhter Kosinus über log2 f · Dichte + Phase (+ Stereo · Platz/4), Ausgleich exakt
  1/√((1−d/2)² + d²/8); über zwei Durchläufe −26,2 gegen −26,7 dBFS.
- **13c Eigene Landschaft:** `setUserLandscape` (lange Aufnahmen auf 8192 Zeilen gemittelt), Wahl „Your Own“, Klick
  auf das Landschaftsfeld, Pfad im Zustand.
- **13d Animierte Bilder:** acht Life-Modi, berechnet in `lifeStepKernel` über `pictureCell` mit den Transformationen
  aus KaleidoscopeEnhanced (Scene2D/ChainKaleidoTunnel.frag: tKaleido, tTunnel, tSpiral, tWarp/fbm, mirrorUV), Quelle
  die hochgeladene Landschaft (Helligkeit auf 40 dB); Feedback und Rain lesen den Puffer davor (375 Schritte/s:
  Zoom 0,15–0,6/s, Vergessen ~3 s; Rain 0,5–30 Tropfen/s); Starfield mit Grundschimmer (sonst 2 dB unter dem Pegel).
  GPU ≤ 0,005 ms pro Block. Regler Sides, Turn.
- **13e Presets:** 1024 Visions in 16 Gruppen; 899 der 3072 bekamen Phase 13 (nach allem anderen gezogen).
  `additiv_presets --names` misst nur die geänderten und neuen (1923).

## Phase 35: Plasma und Stimme (gebaut, 16.10.2026, 0.25.0)

Die letzten beiden Ideen: die Magnetosphäre und die Kehle des Abgrunds -- beide auf der CPU.

- **35a Magnetosphere (Night):** wie ein VLF-Empfänger sie hört. Whistler: der Klick eines Blitzes läuft entlang einer
  Feldlinie durchs Plasma und kommt dispergiert an, f(t) = (D / t)², von 10 kHz bis unter 200 Hz, mit Echos bei 3 D und
  5 D (Hin und Her zwischen den Hemisphären); Whistlers je Minute als Poissonstrom, und jeder Einschlag von Thunder
  schickt einen. Dawn Chorus: steigende Elemente in Bündeln (drei bis zehn, 0,08 .. 0,3 s auseinander) von 0,1 .. 0,3
  der Gyrofrequenz, je lauter, desto schneller steigend (Omuras nichtlineares Wachstum). Hiss: zwei Rauschquellen
  durch einen Bandpass um 700 Hz. Eigener Würfel.
- **35b Throat (Dreams):** bis zu vier Kelly-Lochbaum-Vokaltrakte, bei halber Rate gerechnet (je Abschnitt 7,1 mm, 24
  .. 63 Abschnitte für 17 .. 45 cm), mit Verlust 0,999 je Stoß, Glottis 0,75, Lippen -0,85 und abgeleiteter
  Abstrahlung; angeregt von einem Rosenberg-Puls auf dem Grundton (Oktave ab C2), seiner Quinte, Oktave, der Quinte
  darüber, mit langsamem Vibrato. Kargyraa: die Taschenfalten schwächen jeden zweiten Puls (bis 75 %) -- die
  Periodenverdopplung des Tuva- und Gyuto-Gesangs, eine Oktave darunter. Vowel u o a e i über Flächenfunktionen nach
  Fant und Story (vereinfacht, acht Punkte), langsam driftend.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Plasma & Voice“, 240 .. 255): Whistler Nights, Sferic Storms, Dawn
  Choruses, Radio Auroras, Plasma Hiss, VLF Receivers, Monk Drones, Kargyraa Halls, Giant Throats, Vowel Tides,
  Abyssal Choirs, Overtone Caves, Throat & Thunder, Cosmic Chant, Ionosphere, Voices of the Void. 531 Reisen,
  neunzehn ganze Nächte. Damit 16384 Presets in 256 Gruppen.

**Gemessen [M]:** Selbsttest 109, 110 (Whistler: der lauteste Ton fällt in 23 % der Schritte von 50 ms, steigt in 4 %; der Dawn
Chorus steigt in 82 %, fällt in 15 %; die Kehle auf D2: f0/2 gegen f0 -10,6 dB mit Kargyraa, -83,9 dB ohne; 1 .. 2,5
kHz gegen 150 .. 600 Hz bei i +1,8 dB, bei u -7,4 dB). **Gefunden:** `dispersion` gab es schon (Phase 8, die Uhr der
Teiltöne) -- der neue heißt `whistler_dispersion`; die Kehle war über die Ableitung der Abstrahlung im Bass zu leise
(x 16 ausgeglichen). **Bass-Prüfung [M]** (64 der neuen, Dirigent an): unter 150 Hz Median 17,2 %, 10. Perzentil
10,8 %, zwei unter 8 %.

## Phase 34: Licht und Tiefe (gebaut, 16.10.2026, 0.24.0)

Die nächsten vier Ideen: Szintillation, Spektrallinien (mit Absorption), Wrack, Tiefsee-Kanal.

- **34a Scintillation (`Voices.cu`):** ein dünner Phasenschirm (von Kármán: Kolmogorovs k^(-11/6) unter einer äußeren
  Skala von 64 Punkten, 2048 Punkte, Einheits-RMS) zieht durch die Sichtlinie. Je Band (256, 40 Hz .. 16 kHz) das
  Fresnel-Integral E = sum w(x) exp(i (s phi(x) + pi x² / rF²)) über eine Gauß-Apertur von zwei Fresnelradien, die
  Phase mit der Wellenlänge (s = 1 kHz / f, wie im Plasma), rF mit ihrer Wurzel; |E|² über dieselbe Summe ohne Schirm,
  dann über ihr eigenes Mittel der letzten 8 s. Die Teiltöne lesen sqrt(I) bei ihrer Frequenz. **Gefunden [M]:** mit
  reinem Kolmogorov war alles Phase in der längsten Welle -- eine Rampe über die Fresnelzone, die das Muster nur
  verschiebt: nichts funkelte; ohne die Normierung verlor die endliche Apertur bei starker Streuung 10 dB.
- **34b Lines (`Voices.cu`):** die h-te Harmonische zur h-ten Linie des Wasserstoffs (fünf Serien Lyman .. Pfund,
  je dreißig Linien, aufsteigend über Pfund alpha auf der Note: sechs Oktaven, jede Serie zu ihrer Grenze
  gedrängt); Heat ist die Dopplerbreite (2 .. 40 Cent, an jeder Linie gleich in Cent); Zeeman ein Triplett: je ein
  Viertel der Glieder Zeeman Hz tiefer und höher.
- **34c Absorption:** die Transmission exp(-tau Summe der Linienprofile), tau 4, Halbwertsbreite 0,1 .. 1,3 % (Heat),
  auf der Tonika der Tonart (Pfund alpha in der Oktave ab C1), für alle Teiltöne, die Luft eingeschlossen. **Gefunden
  [M]:** 1024 Bänder über zehn Oktaven waren bei 417 Hz 2,8 Hz breit, die Linie 2 Hz: der Einbruch 5 dB -- jetzt 8192
  Bänder; das Flackern der Luft (±40 Cent bei 8 Hz) verschmiert die Linie als FM, der Test misst mit langsamem Flackern.
- **34d Wreck:** `Tools/presets/make_wreck.py` rechnet die Moden von vier Feder-Masse-Modellen (Rumpf 142 Knoten /
  798 Federn, Tank 146 / 1488, Fachwerk 84 / 265, Platte zweilagig 286 / 1599; Biegung über übernächste Nachbarn und
  Diagonalen) mit numpy (eigh), ohne die freien Starrkörpermoden und einen weichen Mechanismus der Platte (Schwelle
  1e-5 des größten Eigenwerts): je Form 64 Moden, wie stark sie an 16 Stellen angeschlagen werden, zwei Tonabnehmer --
  `Core/src/Wreck.inc`. Die Engine lässt sie als komplexe Einpoler klingen (T60 = Ring x Verhältnis^-0,4, die tiefste
  auf dem Grundton x 2^Size); Mikrometeoriten als Poissonstrom (Hits je Minute, log-normal), thermische Knacke als
  Läufe von Knarzen aus winzigen Rutschern 3 .. 12 ms auseinander, sechsmal so oft in der Dämmerung mit dem
  Nachtbogen; eigener Würfel. Pegel: 60 Treffer je Minute bei Wreck 1 -26 dBFS (zuerst -38).
- **34e Deep Channel (CPU):** Munks Profil c(z) = 1500 (1 + 0,00737 (eta + e^-eta - 1)), 400 Strahlen ±14°, 4-km-Schritte
  über Snells Invariante, Oberfläche (-0,9) und Boden (0,3) reflektierend; Eigenstrahlen, wo Nachbarn den Empfänger auf
  der Achse einklammern, so laut, wie ihre Strahlröhre schmal ist; Abgriffe relativ zur ersten Ankunft auf einem Ring
  von 43 s, das Wasser dunkelt ab (8 kHz x (100 km / R)^0,9). Rechenzeit wenige Millisekunden, nur wenn Range oder
  Off Axis sich bewegen.
- **Seiten:** Cosmos bekommt Scintillation, Spectral Lines, Wreck; Rooms den Deep Channel.
- **Selbsttest:** der Rahmen von `main` hält eine Engine je alten Testblock (MSVC legt sie nicht übereinander) und
  überlief nun auch 8 MB, noch vor dem ersten Test -- jetzt 32 MB Stapel (`/STACK`); die neuen Tests halten ihre Engine
  auf dem Heap.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Light & Depth“, 224 .. 239): Twinkling Stars, Turbulent Light,
  Hydrogen Choirs, Balmer Bells, Zeeman Fields, Dark Lines, Nebula Glow, Derelict Hulls, Orbital Tanks, Broken
  Trusses, Perforated Plates, Thermal Snaps, Deep Channels, Ocean Crossings, Whale Roads, Lost Signals. 498 Reisen,
  achtzehn ganze Nächte.

**Gemessen [M]:** Selbsttest 104 .. 108 (Szintillation: mit gegen ohne 4,5 dB hin und her, -1,2 dB im Ganzen; Lines:
Balmer alpha gegen die 11. Harmonische +59,5 dB mit, -86,4 ohne; Absorption: die Linie -15,2 dB gegen die Luft daneben,
ohne +0,2; Wrack: die tiefsten acht Moden 15,3 dB über den Lücken; Tiefsee: nach 800 km die lauteste Ankunft 10,2 s
nach der Note, 0,2 s später 30 dB leiser -- ein Anschwellen, das jäh endet). Wrack und Tiefsee-Kanal kosten kaum messbar CPU (30 s Klang mit und ohne gleich schnell gerendert).
**Bass-Prüfung [M]** (64 der neuen, Dirigent an): unter 150 Hz Median 20,0 %, 10. Perzentil 12,9 %, eines unter 8 %
(„Long-Range Transit“, 6,7 %: der Kanal dunkelt die Mischung ab, mit ihr die Tiefe, die er nicht weiterträgt).

## Phase 33: Sterne und Schwerkraft (gebaut, 16.10.2026, 0.23.0)

Die ersten vier der Ideen für Dark und Space Ambient (siehe Phase 32): Sternbeben, schwarzes Loch, Verschmelzung,
Rotverschiebungshall. Neue Seite Cosmos (nach Night).

- **33a Star (`gpu/Space.cu`):** die Moden eines Sterns, wie die Asteroseismologie sie mit Kepler und TESS hört, ins
  Hörbare versetzt. Die Tabelle rechnet die Engine (`starModes`): Sonne -- p-Moden nu = dnu (n + l/2 + eps) - l (l + 1)
  D0, Grade 0 .. 3 mit Sichtbarkeiten 1 / 1,5 / 0,5 / 0,04, Ordnungen um n_max = 21 unter einer Gaußglocke von 2,4,
  kleine Trennung dnu02 = 6 D0 von 0,09 auf 0,03 dnu mit dem Alter, Linienbreiten mit nu^4 bis 0,008 dnu bei n_max;
  Roter Riese -- n_max 10, jede Dipolmode als gemischte Mode in neun Linien (je näher der p-Mode, desto stärker und
  breiter), dichter mit dem Alter; Weißer Zwerg -- g-Moden, gleich weit in der Periode (eine Untertonreihe), nur
  manche angeregt, längere Perioden mit dem Alter. Jede Mode in ihre azimutalen Ordnungen gespalten (Spin), gesehen
  unter 60° (Gizon und Solanki 2003), quer nach m/l platziert. Auf der GPU je Mode ein Ornstein-Uhlenbeck-Prozess in
  der Ebene (gedämpft mit der Linienbreite, angestoßen mit Box-Muller-Rauschen aus Block und Mode) -- eine
  Lorentz-Linie --, die Amplitude über den Block gleitend, gemischt in festen Reihenfolgen auf die Kanäle 24 .. 27
  nach der Nacht. dnu ist der Grundton (Oktave ab C2) mal 2^Star Pitch.
- **33b Black Hole (`Voices.cu`):** ein Anteil der Teiltöne fällt (6/10 bei 1): r^(3/2) schrumpft gleichmäßig über
  Fall von 30 Schwarzschild-Radien bis zum Horizont, von fern gesehen rotverschoben (sqrt(1 - rs/r)), gedämpft (das
  Quadrat), zum Loch gezogen, eingefroren, weit draußen neu geboren. Der Rest wird gelinst (Punktlinse): hinaus zu
  seinem Bild jenseits des Einsteinrings (ein Viertel des Felds bei 1), heller um die Vergrößerung (höchstens 6 dB).
- **33c Merger (CPU, `renderMerger`):** ein Poissonstrom (Merger 1: alle drei Minuten im Mittel). Das Massenverhältnis
  gibt den Spin des Rests (a_f = 2 sqrt 3 eta - 3,871 eta² + 4,028 eta³), der Spin die Quasinormalmoden (Berti,
  Cardoso und Will: l = m = 2 und l = m = 3); Ringdown setzt die Tonhöhe der 22-Mode und damit die letzte stabile
  Bahn (f_isco = f_ring / (6^1,5 F / 2)). Das Einspiralen f = f_isco (tau_isco / tau)^(3/8) ab einem halben Hertz,
  die Amplitude mit f^(2/3); unhörbar langsam dehnt und staucht es die Breite des Felds (Plus-Polarisation auf die
  Seiten), hörbar wird es ein Ton im Bass (Plus in der Mitte, Kreuz an den Seiten), dann der Sturz und der Ringdown.
  Quelle Merger. **Gefunden [M]:** die Dehnung mit der physikalischen Amplitude war im frühen Einspiralen ein
  Hundertstel des Merger und nicht zu spüren -- jetzt mit der vierten Wurzel; der erste Pegel (14 x VoiceLevel)
  hätte über 0 dBFS gespitzt, jetzt 1,8 x: -11,6 dBFS.
- **33d Redshift (`Hall.cu`):** jede Mode des modalen Halls gleitet über drei Oktaven abwärts, jede von einem eigenen
  Platz, oben neu geboren, ein- und ausgeblendet mit einem Sinus² über den Weg (Ein- und Ausgang) -- ein
  Shepard-Nachhall; Redshift Time je Oktave.
- **Gefunden [M]:** die Wartezeit bis zum nächsten Merger zog anfangs aus dem Würfel der Engine, auch mit Merger
  aus -- das verschob jede Zufallsentscheidung danach (Entfernungen, Dirigent), und sechs alte Selbsttests fielen
  (Freeze, Scale, Bilder, Orbit, Find, Headset). Jetzt ein eigener Würfel (`m_eventRng`), gezogen nur mit Merger.
- **Selbsttest:** die Engines der Tests ab 99 auf dem Heap -- `main` lief mit 8 MB Stapel über (0xC00000FD), weil
  jeder Testblock seine Engine im Rahmen von `main` hält.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Stars & Gravity“, 208 .. 223): Sun Songs, Red Giants, White Dwarfs,
  Spinning Stars, Ageing Stars, Star Choirs, Accretion Disks, Falling Light, Einstein Rings, Mergers, Distant
  Mergers, Redshift Halls, Receding Light, Cosmic Floors, Quiet Universe, Gravity's Choir; jedes mit dem Boden
  (floor_rule). 465 Reisen, siebzehn ganze Nächte.

**Gemessen [M]:** Selbsttest 100 .. 103 (Sonne bei dnu 18,35 Hz: die Radialmoden 25,3 dB über den Lücken, der Stern
-28,7 dBFS; schwarzes Loch: 150 .. 300 Hz unter einer Note bei 440 Hz -36,9 dB gegen -125 ohne; Merger: 70 Hz auf
der Höhe 44,5 dB, vorher -75,8; Breitenschwankung 4,76 gegen 3,91 dB; Redshift: Mitte des Nachhalls 1,64 gegen 2,41
Oktaven über 60 Hz). GPU: der Stern etwa 0,02 .. 0,03 ms je Block (0,104 -> 0,126 .. 0,138 ms mit dem Standardklang).
**Bass-Prüfung [M]** (64 der neuen, Dirigent an): unter 150 Hz Median 17,2 %, 10. Perzentil 11,8 %, eines unter 8 %
(„Turbulent-Ring Halo Ring“, 6,8 %); der Grundton im Median 100 % der Zeit unten.

## Phase 32: Das Fundament (gebaut, 16.10.2026, 0.22.0)

Der Nutzer, nach meinen Ideen für Dark und Space Ambient: „beachte bei den Presets bitte auf das Bass-Fundament“ und
„Lass uns gerne gleich alles umsetzen“. Die Bass-Prüfung mit Dirigent (Phase 31) hatte gezeigt: in einem Viertel der
Bank ist der Boden dünn, auch wo der Grundton unten liegt -- die Foundation ging durch Formant- und Bandpassfilter, und
laute Schichten deckten sie zu. Also zuerst das Fundament, dann die Ideen (Phasen 33 ..).

- **32a Die Foundation behält ihre Tiefe:** unter 2,2 x ihrem Grundton nehmen ihr Filter, Kamm, Low Cut und Präsenz
  nichts (`CloudVoice::m_floorHz`, im Update-Kernel als Untergrenze der Farbe, über die Oktave darüber zurück).
- **32b Floor:** der Anteil unter 150 Hz, den das Fundament mindestens hält (0,25 x Floor). Die Engine misst den Mix
  vor dem Master (Tiefpass zweiter Ordnung wie `--register`, über 2 s geglättet) und hebt die Foundation-Stimmen mit
  einem langsamen Integrator um bis zu 12 dB, solange der Rest sie zudeckt.
- **32c Pedal:** die Foundation hält ihre Note gegen die Bewegungen des Grundtons (bis zu Pedal² x 3 min), bei 1 ein
  Orgelpunkt auf der Tonika der Tonart (Root).
- **32d Pressure:** ein Anteil der Teiltöne der Foundation (bis zur Hälfte) steht Pressure Rate höher (x Harmonische):
  zwei Pfeifen, ein Haar auseinander. **Gefunden [M]:** die Wolkenbreite (±0,5 Hz bei 73 Hz) und Drift verwischten
  die Schwebung in eine Bande von Schwebungen -- bei Druck werden Breite und Drift der Foundation auf 3 %, ihr
  Schimmer auf 10 % genommen; die Bewegung „Breathe“ deckt sie trotzdem zu, also begrenzen die Rezepte bei Druck die
  Bewegungstiefe auf 0,15 .. 0,4.
- **32e Sub Octave, Fifth, Breath:** zwei Foundation-Stimmen daneben (eine Oktave tiefer, eine Quinte höher; dann
  beginnt der Akkord eine Oktave über der Foundation), Breath lässt sie in 26 und 42 s an- und abschwellen.
- **Seite:** auf Conductor die Kästen Floor (Foundation, Floor, Pedal, Pressure, Rate) und Sub & Fifth (sonst
  scrollte die Seite bei 1280 x 860 um 10 px).
- **Presets:** `floor_rule` zuletzt für alle: jedes Preset mit Dirigent bekommt eine Foundation (2387 hatten keine)
  und Floor 0,35 .. 0,65; sonst bleibt alles, wie es war. 1024 neue in 16 Gruppen (Abschnitt „Deep Floors“: Organ
  Points, Pressure Drones, Sub Abysses, Fifth Pillars, Breathing Floors, Cathedral Depths, Abyssal Plains, Black
  Monoliths, Iron Pedals, Tectonic Hum, Cavern Drones, Undertow, Low Chapels, Night Pressure, Deep Bells, Floors &
  Storms). Ein deutsches Wort im Namen („Nachtmusik“ in Nocturnes) heißt jetzt „Small-Hours Air“. 432 Reisen,
  sechzehn ganze Nächte.

**Gemessen [M]:** Selbsttest 99 (Formantfilter: unter 150 Hz 8,9 % mit Foundation gegen 1,3 % ohne; eine Harfe darüber
7,0 %, mit Floor 1 27,1 %; Pedal 1: 100 % der Blicke auf der Tonika, während der Quantum Walk den Grundton bewegt;
Druck: Autokorrelation der Hüllkurve bei 4 s minus bei 2 s 1,50 gegen 0,40 ohne; Sub Octave: unter 55 Hz -9,7 gegen
-17,2 dB des Ganzen). **Bass-Prüfung [M]** (`--register`, Dirigent an, 60 s; dieselben 78 Presets wie in Phase 31 und 64 der neuen):
in der alten Bank 10. Perzentil unter 150 Hz 2,0 -> 13,5 %, unter 8 % 8 -> 1 von 26; in Night & Illusions und Chorus
& Mirrors 1,4 -> 10,3 %, unter 8 % 19 -> 0 von 52 (Median 12,1 -> 17,1 %); Deep Floors Median 21,5 %, 10. Perzentil
15,4 %. **Gefunden:** Floor hob zuerst die ganze Foundation-Stimme (bis 12 dB) -- ihre Obertöne mit, im Formantfilter
die lautesten: „Moored Diapason“ wurde 3 dB lauter, der Anteil unter 150 Hz stieg nur von 0,2 auf 1,6 %. Jetzt hebt
Floor nur ihre Tiefe (die Teiltöne unter 2,2 f0, `m_floorGain`, bis 18 dB): 6,8 % bei +2,6 dB. Am dünnsten bleibt
„Climbing Ladder“ (Shepard & Tides, 4,9 %).

## Phase 31: Nachtchor, Zeitspiegel, die Karte wächst selbst (gebaut, 16.10.2026, 0.21.0)

Zweiter Teil von „Bitte alles einbauen, das klingt alles spannend.“ (siehe Phase 30): der Chor, der Spiegel, das
Wachsen und die neue Einpegelung der ganzen Bank.

- **31a Nachtchor:** 2048 Rufer (Zustand als float4 × 4 je Rufer), acht Arten -- Feldgrille, Baumgrille,
  Laubheuschrecke, Maulwurfsgrille, Frühlingspeeper, Laubfrosch, Ochsenfrosch, Kröte --, jede mit eigenem Träger,
  Puls und Rhythmus in einem eigenen Band (akustische Nische). Wer ruft: Wiese, Baumkronen, Teich, alle. Die Raten nach
  Dolbears Gesetz (Wärme 5 .. 30 °C, der Nachtbogen kühlt), milder für die Frösche. Jeder Rufer ist eine Uhr; ein Ruf
  der eigenen Art, nah und laut genug gehört, setzt sie zurück (Greenfields inhibitorisches Zurücksetzen) -- Nachbarn
  fallen in Gleichtakt oder in Wechsel; wer in einen lauteren fremden Rufer im eigenen Band riefe, wartet auf eine
  Lücke. Tuned Calls zieht die Träger zu den Tönen der Skala. Vier Kernel (Uhren 64 Threads je Block, Ordnung als
  Phasoren der häufigsten Art, Rufe als Pulse aus angehobenem Sinus, Summe in fester Reihenfolge); die Zählungen in
  ganzen Zahlen (eine gebrochene float-Atomic wäre nicht deterministisch gewesen). Kanäle mit Harfe, Donner, Eis in
  24 .. 27. GPU etwa 0,12 ms je Block.
- **31b Zeitspiegel:** 64 Zellen rund um den Hörer im Wellenraum (Phase 29). Zwei Takte: einen Fenster lang (1 .. 6 s)
  hören, im nächsten rückwärts zurücksenden -- die Wellen laufen zurück, woher sie kamen (Finks Zeitumkehrspiegel),
  jeder Klang kehrt als Anschwellen zu seiner Quelle zurück. **Gefunden [M]:** die erste Fassung nahm ihr eigenes
  Zurücksenden mit auf und schaukelte sich auf (-3,7 dBFS); jetzt hört der Spiegel nur im ersten Takt, Anteil 0,08.
- **31c MAP-Elites (`Tools/presets/grow.py`):** Mouret und Clune. Das Archiv (`work/grow/archive.json`) hält je Zelle
  der Karte (Helligkeit × Bewegung, wie `make_map`) den besten Klang; Kinder entstehen durch Mutation weniger Regler
  (in einem Viertel der Fälle mit einer anderen Elite gekreuzt, auch die Matrix), `additiv_presets --evaluate` misst
  sie wie die Bank (LUFS und die vier Beschreiber, parallel); ein Kind ersetzt die Elite seiner Zelle, wenn die leer
  ist oder es besser ist: näher an den Sternen des Spielers (`player.settings`), weiter von jedem Werkspreset, weder
  still noch am Clipping. Ausgabe als eigene Presets in Documents/Empyrean/Presets/Grown („Grown Dusky Drifting
  7-12“), eingepegelt wie die Bank; das Plugin liest den Ordner jetzt mit Unterordnern. Ein nächster Lauf setzt das
  Archiv fort.
- **31d Lautheit:** die ganze Bank neu gemessen, jedes Preset als K-gewichtete Lautheit (ITU-R BS.1770) der
  lautesten drei Sekunden statt des RMS der lautesten Sekunde, bei Tag (der Nachtbogen zählt nicht). TARGET so
  gewählt, dass der mittlere master_gain bleibt.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Chorus & Mirrors“, Gruppen 176 .. 191); 399 Reisen, fünfzehn ganze
  Nächte.

**Gemessen [M]:** Selbsttest 98/98 (97 Nachtchor: 1072 Rufe/s bei 28 °C, 312 bei 10 °C; Ordnung 0,02 ohne, 0,49 mit
Antworten; 98 Zeitspiegel: der späte Teil schwillt von -30,4 auf -21,3 dBFS, der Akkord -20). Referenzklänge bitgleich.
Bank: alle 11264 in 4261 s neu in LUFS gemessen (zehn Engines), die 1024 neuen in 481 s. TARGET -18,0 LUFS (Median
RMS -19,95 dB, Median LUFS -17,99; LUFS minus RMS 5 % +0,7, 95 % +2,9 dB): die alten Presets ändern nur master_gain,
5 % -0,9 .. 95 % +1,3 dB, höchstens ±3,9; sechs bleiben am Anschlag von +12 dB (Feverish Spectra -39,8 LUFS, schon
vorher -40 dB RMS). **Bass-Prüfung [M]** (`--register`, Dirigent an, 60 s, Stichprobe von 78): Median 18,9 % der
Leistung unter 150 Hz in der alten Bank, 12,1 % in Night & Illusions, 15 von 44 unter 8 % -- auch mit dem Grundton
unten: Formant- und Bandpassfilter nehmen der Foundation ihren Bass, laute Schichten (Harfe, Regen) decken ihn zu.
Folgt in Phase 32.

## Phase 30: die Nacht, Streams, Risset, Nachtbogen, Traumtagebuch, Headset (gebaut, 16.10.2026, 0.20.0)

Der Nutzer zu meiner Liste von Ideen (Optimierungen, Kopf und Hände aus der Quest in der Wolke, Äolsharfe im
LBM-Wind, Nachtchor, Zeitumkehr-Raum, Donner, singendes Eis, Fusion/Fission, Risset, Traumtagebuch, Nachtbogen nach
der echten Sonne, MAP-Elites, je Phase 1024 Presets und eine neue LUFS-Messung): „Bitte alles einbauen, das klingt
alles spannend.“ Phase 30 nahm die Nacht und die Illusionen, Phase 31 den Rest.

- **30a Äolsharfe:** zwölf Saiten im Wind des LBM-Gitters (Phase 27), auf den Akkord gestimmt, je Saite eine Warp,
  je Lane eine Mode (32, leicht gestreckt). Facchinettis Nachlaufoszillator (van der Pol, an die Beschleunigung der
  Saite gekoppelt, Strouhal 0,2) rastet auf die Harmonische nahe der Wirbelfrequenz ein und springt mit dem Wind.
  **Gefunden:** stumm, solange der Wind nur für Windspiele und Wetter wehte -- jetzt auch für die Harfe.
- **30b Donner:** je Einschlag ein eigener Blitzkanal (2048 Segmente Hauptkanal, vier Äste, tortuos auf drei Skalen,
  warp-parallel ausgelegt), je Segment eine N-Welle nach Few (Dauer τ = 2,6 R0/c · (r/10 R0)^(1/8)), verschmiert
  entlang der Sichtlinie, Luftdämpfung; in einen Ring ganzer Zahlen (2^21) summiert, sechzehn Stücke je Block,
  bitgleich. Erster Schlag 0,5 .. 2 s nach dem Einschalten; die Quelle Lightning blitzt beim Einschlag. **Gefunden
  [M]:** mit Fews Exponent 1/4 grollte er bei 15 .. 23 Hz; mit 1/8 bei 44 .. 112 Hz. Der Test des Kopfdrehens maß
  kaum Pegelunterschied, weil Bass Mono bei 150 Hz das Grollen faltet -- der Test setzt 40 Hz.
- **30c Singendes Eis:** OFC-Gitter (Olami, Feder, Christensen) 32 × 32, α = 0,2, auf der CPU mit 30000
  quasistatischen Lawinen in den kritischen Zustand vorgewärmt, dann Poisson-Risse (0,15 · 40^Kälte je s). Je Riss ein
  fallender Chirp (f ∝ t^-1,9: Biegewellen im Eis sind dispersiv), Amplitude ∝ Größe^0,35. **Gefunden:** 64 × 64 ohne
  Vorwärmen brauchte Minuten bis zum kritischen Zustand.
- **30d Streams:** die Teiltöne in Ströme geteilt (ungerade/gerade, nach drei, Register, Mitglieder), je Strom eigenes
  Vibrato und Tremolo auf der Stimmuhr -- das Ohr hört mehrere Klänge (Bregmans gemeinsames Schicksal); Fusion bewegt
  sie wieder gemeinsam.
- **30e Risset:** fünf Pulsschichten; an der Oktavgrenze verdoppelt oder halbiert sich die Phase -- ewig langsamer
  (oder schneller), ohne anzukommen. Quelle Risset Pulse.
- **30f Nachtbogen:** Sonnenstand nach dem Astronomical Almanac (`Sun.h`), Ort aus den Einstellungen oder der
  Hauptstadt der Zeitzone; Nacht = Smoothstep bis -18°. Sie dunkelt (Brightness -0,35, Tilt +0,15), verlangsamt
  (Brain Rate -0,3, Density -0,2) und weitet (Far Level +0,15). Quelle Night.
- **30g Traumtagebuch:** die neuesten 1024 Erinnerungen des Traums (ein Byte je Band), gzip und Base64 im Zustand;
  ein anderes Projekt träumt davon weiter.
- **30h Headset:** `/pose` auf Port + 100 (`Plugin/Pose.h`), Hände an 9105. Die Kopfdrehung hält die binaurale Wolke
  und den Donner im Raum; die Hände sind Pfosten im Wind, Defekte im Halo, Absorber oder Quellen im Wellenraum, Futter
  für den Schleimpilz. Neue Quest-App (`Quest/`, OpenXR, NDK r27, 877 KB): gebaut, noch auf keinem Headset gelaufen.
- **Seiten:** neue Seite Night (Aeolian Harp, Thunder, Singing Ice, Night Chorus, Night) nach Weather; Streams, Risset
  und das Tagebuch auf Dreams; Time Mirror und Headset auf Rooms.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Night & Illusions“, Gruppen 160 .. 175).

**Gemessen [M]:** Selbsttest 96/96 (89 Harfe: Harmonische 21,6 bei Wind 0,3, 25,2 bei 0,9; 90 Donner: 6,4 s nach dem
Blitz bei 2,1 km (erwartet 6,2), 97 .. 99 % unter 400 Hz; 91 Eis: 10 gegen 77 Chirps bei mild und kalt, 7 von 7
Einsätzen fallend; 92 Streams: Hüllkurven-Korrelation 0,01 getrennt, 1,00 verschmolzen; 93 Risset: der Tempokamm
gleitet 0,29 / 0,68 wie berechnet; 94 Sonne: Berlin zur Sonnenwende 60,9° mittags, -60,9° um Mitternacht, Nacht:
Anteil über 2 kHz 4,0 → 2,1 %; 95 Tagebuch: 12 Erinnerungen, 6204 Byte; 96 Headset: Pegelunterschied -1,5 → +2,5 dB
beim Kopfdrehen, Hände als Futter +44 dB). Die bestehenden Klänge bitgleich.

## Optimierungen und Cloud.cu aufgeteilt (gebaut, 16.10.2026, 0.20.0)

- **Ruhende Features aus dem Graphen:** Leben, Wind, Windspiele, Wetter, Schwarm, Physarum, Halo, Wellenraum, modaler
  Hall, Bloom, Traum, Phantom, Reservoir, Echo, Melt (und mit Phase 30/31 Harfe, Donner, Eis, Chor, Nachtmischung)
  werden im ausführbaren Graphen abgeschaltet (`cudaGraphNodeSetEnabled`, drei Blöcke nach dem Ruhen; die Frames
  vorher geleert). **Gemessen [M]:** ein schlichter Akkord 0,135 → 0,10 ms je Block (die 16 leeren Raumschritte allein
  waren 13 % der GPU-Zeit).
- **Modaler Hall:** vier Moden je Thread, eine Warp-Summe für alle vier.
- **Mit allem an (Nsight Systems):** Schwarmkräfte über 64 statt 4 CUDA-Blöcke, Regen mit einem Spawner, Laub und
  Brandung über geteilten Speicher gemischt statt Warp-Summen je Sample, die mittlere Windgeschwindigkeit aus jeder
  achten Zelle.
- **Phantomschicht:** keine Heap-Allokation je Block mehr.
- **Presets parallel gemessen:** `additiv_presets --jobs` (10 Engines): 48 Presets 58 → 18 s, gleiche Zahlen.
- **Lautheit:** ITU-R BS.1770 (K-Gewichtung, lauteste drei Sekunden, LUFS) statt RMS der lautesten Sekunde.
- **Cloud.cu aufgeteilt:** die Kernel in `Core/src/gpu` (Device.cuh, Kernels.cuh, Voices, Hall, Life, Weather, Dream,
  Bloom, Agents, Rooms, Spectral, Reservoir, später Night), Namensraum `additiv::gpu`; jede Datei füllt ihre Tabellen im
  Konstantenspeicher über eine eigene Upload-Funktion. Cloud.cu behält Engine, Graph und API. Drei Renderings mit fast
  allen GPU-Features bitgleich.

## Phase 29: Wellenraum, topologischer Halo, Quantum Walk (gebaut, 16.10.2026, 0.19.0)

Der zweite Teil der Nacht (siehe Phase 28). Absorb (Sidechain-Resynthese) blieb wie empfohlen außen vor.

- **29a Wellenraum:** 3D-Wellengleichung (Standard-Leapfrog, Courant ~0,5) auf 88 × 70 × 40 Zellen à 11,4 cm bei
  fs/8 (6 kHz bei 48 kHz), 16 Kernel-Schritte je Block im Graphen, drei Druckfelder rotierend. Wände: Bilbaos
  Finite-Volumen-Rand mit einer Admittanz ((1 + l b (6-K)/2) p+ = (2 - K l²) p + l² Σ - (1 - l b (6-K)/2) p-).
  Quellen: jede Stimme aus ihren Chunks summiert, Tiefpass (64-Tap-Sinc, 2,2 kHz) und Dezimation, an ihrer Zelle auf
  einer Höhenebene (sonst prüfte jede Zelle 24 Quellen); zwei Ohren je eine Zelle neben dem Hörer, polyphase
  hochgerechnet. Kanäle 22, 23. **Gemessen [M]:** durch bitgleiche Differenz isoliert (mit minus ohne Raum) bei
  Pegel 1 gleichauf mit dem trockenen Akkord (-28 gegen -29 dBFS), Band 100 Hz .. 3 kHz.
- **29b Halo:** Qi-Wu-Zhang-Modell auf einer Scheibe (Radius 12, 441 Plätze, zwei Orbitale), je Träger (8) ein Gitter,
  RK4 je Block im mitdrehenden Rahmen, eine CUDA-Block je Träger. Antrieb in der Lückenmitte an der Randstelle in
  Richtung der Note (goldener Winkel), 0,4 s kräftig, dann sanft. Eine dritte Schichtstimme (HALO) singt es: je Platz
  und Träger ein Sinus mit der komplexen Amplitude des Platzes (Sinus- und Kosinusanteil, gemeinsame Phase je Träger).
  **Prototyp [M]:** u = +1 läuft gegen den Uhrzeigersinn, 98 % am Rand, ein Umlauf 12,5 s bei Omega 10; u = -1 im
  Uhrzeigersinn; Unordnung 1,5 innen: gleiche Richtung, ~50 % am Rand; u = 3 (trivial): keine Zustände in der Lücke.
  In der Engine 99 % am Rand.
- **29c Quantum Walk:** 12 komplexe Amplituden auf den Tonklassen, Tonnetz-Kanten ±3, ±4, ±7 (zirkulant: Schritt im
  Fourierraum), Phase zwischen zwei Blicken 0,3 .. 6 rad (Spread); beim Blick Wurf nach |psi|², der Grundton
  (setRoot wie der Jam-Folger) springt zur nächsten Note dieser Klasse, psi zu 70 % kollabiert. Ein Jam-Leiter hat
  Vorrang.
- **Seiten:** neue Seite Rooms (Wave Room, Halo) im Abschnitt Space; Quantum Walk auf Conductor.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Rooms & Topology“); die bestehenden unverändert.

**Gemessen [M]:** Selbsttest 88/88 (86 Wellenraum: Ausklang eine Sekunde nach dem Akkord zwischen harten Wänden -29,8,
mit weichen -56,7 dBFS, 99 % unter 2,5 kHz; 87 Halo: in drei Sekunden +142° gegen den Uhrzeigersinn (86 % am Rand),
-145° im Uhrzeigersinn, trivial -35° -- was sich da bewegt, sind die Flanken des Anschlags außerhalb der Lücke; 88
Quantum Walk: bei 0,5 rad 74 Schritte, 49 % zu Tonnetz-Nachbarn, bei 2 rad 78 Schritte, 76 % -- simuliert 47 und 78 %:
Interferenz). Gefunden: der Dirigent bewegt den Grundton auch selbst; ohne Neuzentrierung maß der Walk Sprünge von
einem Grundton aus, an dem er nicht stand. Bank: 1040 Presets in 1449 s gemessen (die 16 umbenannten -- „Opera Opera
Gong“, „Sandy Sands“ -- mit), 0 alte verändert. 333 Reisen.

## Phase 28: Wetter, Bloom, Swarmalators, Physarum (gebaut, 16.10.2026, 0.18.0)

Der Nutzer fragte nach weiteren Ideen jenseits des Stands der Technik; von meinen sieben Vorschlägen (Weather, Bloom,
Swarmalators, Physarum, Topological Halo, Quantum Walk, Absorb; dazu der Wellenraum) empfahl ich 28 und 29, er: „führe
über Nacht die nächsten Phasen 28 und 29 aus“. Die Swarmalators gab es als Sync-Modus auf dem Ring je Wolke schon
(Phase 16); gebaut ist die zweidimensionale, instrumentweite Fassung mit Paarkräften.

- **28a Wetter:** Der Wind (LBM aus Phase 27) bläst für Windspiele oder Wetter (`Weather::m_wind`, eigener Neustart
  `m_windFresh`). **Regen:** 16384 Plätze, ein freier fängt einen Tropfen mit der Wahrscheinlichkeit Rate · dt / freie
  Plätze · Schauerdichte; Wasser: Minnaert 3,26 kHz / r(mm), Dämpfung 0,043 f + 0,0014 f^1,5 (van den Doel), Anstieg;
  Blätter, Stein, Blechdach (gestimmt: Skalentöne). Schauer: Dichte auf dem Windgitter, semi-Lagrange mit 0,15 der
  Windgeschwindigkeit verfrachtet, links gespeist von vier OU-Zufällen (10 s). **Gefunden [M]:** die Rückkehr zur
  Gleichverteilung mit 0,0005 je Block (5 s) löschte jeden Schauer, bevor er ein Viertel des Feldes kreuzte; jetzt
  0,00002 -- links/rechts schwankt 1,3 dB (gleichmäßiger Regen 0,47). **Laub:** 4096 Gruppen, Klickrate ∝ ((|u| + 2
  Böe) / 0,08)², gemessen mittlere Windgeschwindigkeit 0,026 / 0,050 / 0,069 bei Wind 0,2 / 0,5 / 0,8. **Brandung:**
  128 Uferstücke, 64 Wellenzüge um die Periode (Tiefwasser), Brechen über 1,4 σ · Höhe, Schaum (3 s), Zischen,
  Rückstrom-Knistern (Kiesel), Brandungsdonner (Felsen). Kanäle 16 .. 19.
- **28b Bloom:** 32768 komplexe Einpoler gleichverteilt in Hz (Weyl, Abstand 3,24 / Fläche), angeregt von Stimmen,
  Windspielen, Wetter; je Block die Energie von 64 Bändern (feste Reihenfolge), Fluss nach oben Drive · 80 · (ē / ē_ref)
  · dt (höchstens die Hälfte), als Dämpfung abgezogen und dem Band darüber als Rauschen gleicher Leistung gegeben;
  Berger-Glide. **Gefunden [M]:** die Kaskadendämpfung ging in die Eingangsnormierung ein und öffnete die Platte für mehr
  Eingang (lauter statt verschoben) -- Eingang jetzt mit dem ungedämpften Pol. Ein echtes FD-Gitter (von Kármán) bräuchte
  je Sample eine Biharmonik-Lösung: in Echtzeit zu teuer. Kanäle 20, 21.
- **28c Swarmalators:** 1024 Agenten, alle Paare in Kacheln, Schritt = Blockzeit · Speed; Ansicht normiert auf ihren
  Schwerpunkt und Radius; R und S nach O'Keeffe. Partials reiten einen (Ort, Tiefe, Puls). **Gemessen [M]:** J 1, K 0:
  S 1,00, R 0,01; K 1: R 1,00; J 0,5, K -1: R 0,00, S 0,00. **Gefunden:** J 1, K -1 ist die aktive Phasenwelle (S 0,55,
  der Ring dreht), die stille Scheibe liegt erst jenseits von K = -1,2 J.
- **28d Physarum (Feldmodus):** Spur 1024 × 32 (log-Frequenz × Feld), Agenten nach Jones (Sensor 0,7 rad, Drehung 45°
  je Zelle), Ablage in ganzen Zahlen, Futter an den ersten acht Harmonischen. **Gefunden [M]:** Futter 4 je Block hielt
  die Agenten auf den Linien (0,1 % der Leistung zwischen den Harmonischen); mit 0,2 und Radius 200 Cent 26 .. 36 %.
- **Seiten:** neue Seite Weather (Wind, Chimes, Rain, Leaves, Surf); Ripple nach Harmonics, Memory nach Dreams (Motion
  und Effects waren bei 1280 × 860 10 px zu hoch).
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Weather & Growth“); die bestehenden unverändert.

**Gemessen [M]:** Selbsttest 85/85 (80 Regen: Blechdach 150 → 600 Tropfen/s, 218 → 864 klingend (4,0×); 81 Laub: Stille,
Wind 0,3 -25,9, 0,9 -16,5 dBFS; 82 Brandung: Sätze über 11,3 dB, Wind 0,6 gegen Flaute +16 dB; 83 Bloom: über 2 kHz
37,5 % gegen 0,2 %, Ausklang leiser; 84 Swarmalators wie oben; 85 Physarum: 36 % zwischen den Harmonischen, Spur-Spread
3,46). GPU mit allem an 0,17 ms je Block (Bloom 0,07).

## Phase 27: Windspiele, Phantommelodien, Traum, Vorahnung, Talbot (gebaut, 16.10.2026, 0.17.0)

Der Nutzer fragte nach Ideen weit über den Stand der Synthese hinaus, für Ambient auf der GPU; meine Vorschläge (im
Gespräch) und sein „bitte baue das ein und erstelle gleich noch 1024 neue Presets“. Die Wellensimulation eines Raums
(3D-FDTD) blieb für später.

- **27a Phantom:** eine Sonderstimme (nach den Geistern) aus 8192 Rauschpartials 150 Hz .. 3 kHz, gleich je Oktave,
  flackernd wie Air, in beiden Ohren gleich -- außer dass um jede Zieltonhöhe die Phase des rechten Ohrs einmal ganz
  herumdreht (2 pi über Phantom Width, tanh). Huggins-Tonhöhe (Cramer & Huggins 1958), nur über Kopfhörer. Eigene
  Kanäle (14, 15): `reduceKernel` lässt ihre Chunks aus, `phantomReduceKernel` summiert sie; die Engine mischt sie
  nach Mid/Side trocken dazu (Echos, Räume, Breite verwischten die Phase). Der Render-Kernel rechnet ihre
  Kosinusanteile auch ohne Binaural. **Gefunden:** zuerst trug sie die Töne des Akkords -- dort klingen die echten
  Töne schon, die Phantomtonhöhe verschmolz mit ihnen (Kreuzspektrum dort ~0). Jetzt Phantom Line: Melody (eigene
  Linie in der Skala, ein bis zwei Stufen alle 4 .. 8 s, Überblendung 1,5 s), Echo (die zuletzt verklungenen Noten,
  30 s), Chord.
- **27b Talbot:** die ersten 24 Harmonischen als Wellenfunktion ab Notenbeginn, im Eigensystem (`talbotSystem`,
  zwischengespeichert): Revival = Kosinusbasis mit n² w (exakte Wiederkehr, bei T/2 gespiegelt, bei T/4 beides zur
  Hälfte); Bloch = Kette mit Hüpfen J = wM/8 und Neigung w je Harmonischer (Jacobi-Rotationen), Eigenwerte auf die
  exakte Leiter w n gesetzt (an den Kettenenden lagen sie daneben: die Rückkehr verfehlte um drei Harmonische).
  **Gefunden:** mit −J lief das ruhende Paket nach unten in die Wand am Grundton und blieb; mit +J steigt es. Leistung
  exakt erhalten (unitär), Depth mischt nach Leistung. **Gefunden beim Hören [M]:** die Wiederkehr ist scharf -- die
  23. Mode dreht 529-mal so schnell wie die erste, das Bild steht nur ein 3300stel der Periode (bei 30 s 9 ms); über
  2 s gemittelt lag der Grundton über der 23. nur bei +18 → ±0 → −14 → +18 dB, dazwischen Rauschen. Jetzt eine
  verzerrte Uhr (`talbotClock`): je Viertelperiode langsam an den Rändern, exponentiell schnell dazwischen (17
  e-Faltungen), so steht jede Marke -- ganz, halb und halb (bei T/4 exakt: Leistung = (Bild + Spiegel)/2),
  gespiegelt -- eine Weile, und das Bild zerfällt von den feinsten Einzelheiten her. Prototyp: +27 (3 s) → +6 →
  −22 (5 s) → +6 → +27 dB; im Render (A2 in der Skala 108,75 Hz, Velvet Dreaming mit Hall und Gedächtnis): +26 →
  bis −21 → +28 dB, Bloch-Schwerpunkt 3,1 → 12,5 → 3,1.
- **27c Chimes:** 2048 Windspiele (64 × 32, gejittert) × 16 Moden (acht, je ein Dublett ±1,5 Cent) = 32768 komplexe
  Einpoler (`chimeKernel`, Bauart des modalen Halls, ruhende Blöcke schreiben nur Stille); Wind = D2Q9 BGK auf 128 ×
  64 (tau 0,53, zwei Schritte je Block, Kraft nach rechts, drei Pfosten mit Wirbelstraßen, Pull-Streaming mit
  Bounce-back); Klöppel als Pendel (0,6 .. 1,4 Hz) gestoßen von Querwind und Böigkeit, Anschlag beim Spalt mit
  Samplelage. Böe: Ornstein-Uhlenbeck (3 s). Arten: Rohr (freier Stab 1, 2,76, 5,40 ..), Glas, Bambus (kurz), Glocke.
  Gestimmt auf die Skala im Register der Art, Chime Pitch. Kanäle 10 .. 13 (nah/fern). Kalibriert [M]: Wind 0,3 →
  0,8 Anschläge/s, 0,5 → 3,6, 0,6 → 18, 0,8 → 63, 1,0 → 71 (Klöppel 16, Kraft 2e-5 (2 Wind)^1,2); der Python-
  Prototyp lag mit Klöppel 30 bei 10/s, die Engine bei 79 -- der Unterschied steckt in der Böe.
- **27d Dream:** der Hall hört (listen) und sammelt 1024 Bänder; alle 3 s wird das Gesammelte (als 512 Bänder, Länge
  1) eine Erinnerung, wenn es keiner zu 98,5 % gleicht (bis 4096, die älteste geht). `dreamSimKernel` (16 Blöcke)
  vergleicht alle Erinnerungen mit der Anfrage (Zustand + Rauschen), `dreamRecallKernel` (ein Block) nimmt die 32
  nächsten, mischt sie nach softmax(beta Ähnlichkeit) und zieht den Zustand ein Stück hin. Eine Sonderstimme (16384
  Partials über die 512 Bänder) singt ihn, fern. Feste Reihenfolge: bitgleich. Die Traumstimme selbst wird nicht
  gehört (keine Rückkopplung).
- **27e Premonition:** eine Note des Dirigenten beginnt mit ihrem Anschwellen (e^(-4 Rest/Länge), Wolke × (1 + 4 p)
  breit, fern, dunkler), die Hüllkurve erst nach Premonition Time; danach zieht sich der Schwall in 0,15 s in die
  Note. Gespielte Tasten nicht (ihre Zukunft kennt das Instrument nicht).
- **Seiten:** Talbot auf Harmonics; neue Seite **Dreams** (Sound): Chimes, Phantom, Dream, Premonition.
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Dreams & Phantoms“); die bestehenden unverändert.

**Gemessen [M]:** Selbsttest 79/79 (75 Phantom: an der Tonhöhe 2,91 rad zwischen den Ohren, eine Quinte darüber 0,00,
ein Ohr allein −2,1 dB gegen daneben; 76 Talbot: Grundton über der 24. +14,9 → −15,6 → +14,9 dB, Leistung ±0,00;
Bloch-Schwerpunkt 2,4 → 11,6 → 2,4; 77 Chimes: bei Wind 0,6 61 Anschläge in 10 s, ohne Wind keiner und Stille; 78
Dream: 3 Erinnerungen, 61 % der Traumleistung an den Harmonischen der gespielten Noten; 79 Premonition: Wolke 12,4
gegen 2,5 Cent, Pegel 0,07 gegen 1,00).

## Phase 26: Die eigenen Leben der Harmonischen, Körner, Horizont, Breather (gebaut, 15.10.2026, 0.16.0)

Drei Texte des Nutzers über neue Klangerzeuger (Gabor-Atome, Schrödinger-Spektren, Weierstraß-Kaskaden,
Möbius-Oszillator, Rijke-Rohr/Überblasen; Spin-Glas, Klein-Gordon-Membran, Billard, holographische Beugung,
Lotka-Volterra; subharmonische Kaskaden, erregbare Medien, Sandhaufen, hyperbolische Spektren, Voice-Braiding).
Meine Einschätzung im Gespräch: gebaut, was im 128er-Block auf den Harmonischen arbeitet, sich nicht über die 110 bis
333 Partials einer Harmonischen wegmittelt (Phase 0b) und kein bestehendes Modul wiederholt. Nicht gebaut: Möbius
(Moorers Discrete Summation Formula von 1976; aliast und versteckt Obertöne vor Filter, Landschaft, Turbulence),
Spin-Glas (Kuramoto, Chimera, Swarm gibt es; gleichgerichtete Spins sind ein Sinus), Klein-Gordon-Membran und Billard
(eigene Instrumente; Strahlen machen keine Scars), Beugung (je Partial 10^10 Operationen; per FFT ein Weichzeichner der
Landschaft), Weierstraß (= Shepard), erregbare Medien (Flock, Swarm, Orbit bewegen die Wolken schon), Braiding.
Prototypen der Dynamik in Python (scratch), dann Kern, Selbsttests, Seiten, Hilfe, 1024 Presets.

- **26a Overblow:** vier Register (1, 2, 3, 4 × f0) als Anteile, Replikatordynamik; jedes bevorzugt einen Atem (0,
  0,45, 0,72, 0,95), das klingende hält durch seinen Anteil (Hold 0,12: Hysterese), Mutation 0,0005. Atem = Knopf ×
  (0,55 + 0,45 Anschlag + 0,5 MPE-Breath) + eigene langsame Unruhe (±0,04 × Knopf). Harmonische h klingt mit
  √(Summe der Anteile der Register, die h teilen) (`registerGain`). Erster Versuch nach Lambs Modenkonkurrenz
  (Wachstumsraten mit Kreuzsättigung): wechselte erst, wenn alle Register leise waren -- verworfen.
- **26b Seasons:** fünf Familien nach dem ungeraden Teil von h (Oktaven, Quinten, Terzen, Septimen, Nonen;
  `seasonFamily`) in einem zyklischen Turnier (jede schlägt die nächsten zwei mit 1, verliert gegen die zwei davor mit
  2), Replikatordynamik mit Mutation 0,01: der heterokline Zyklus zieht an, die Mutation hält ihn regelmäßig (Anteile
  0,002 .. 0,91; Gewinn > Verlust dämpft dagegen auf den Mittelpunkt -- in Python gemessen). Eine Runde 12,9
  Zeiteinheiten, auf Year skaliert. Global (jede Stimme in derselben Jahreszeit; je Stimme mittelte es sich weg).
  Grundton bleibt; Amplitude der Familie 1 − d + d √(5x).
- **26c Avalanche:** Oslo-Reishaufen (Christensen et al. 1996; in 1D kritisch, der BTW-Sandhaufen nicht) auf den
  Harmonischen 2 .. 49 je Stimme, Schwellen 1/2 nach jedem Kippen neu, eine Generation je 35 ms (die Lawine läuft
  hörbar), jedes Kippen leuchtet seine Harmonische (bis +9,5 dB, Glow, 40 ms weich). Gemessen (Python): eine Lawine
  kippt im Mittel so oft, wie der Haufen lang ist -- bei 1 Korn/s und Glow 1,5 s leuchtet die 2. Harmonische 70 % der
  Zeit; daher 0,3 Körner/s und Glow 0,6 s als Vorgabe (12 % / 3 %, zwölf oder mehr gemeinsam 5 % der Zeit).
- **26d Subharmonics:** acht Wolken nach den Harmonischen (½, 3/2, 5/2, 7/2 gepumpt von 1, 3, 5, 7; ¼ .. 7/4 von
  diesen), Partials gleichmäßig (`m_members = HP / (H + 8)`), je Note bei ihrem Beginn reserviert (nie umverteilt).
  Parametrisch: da/dt = a ((D A − 0,25) − 1,5 (a/A)²) / τ ab 10⁻³ A, die Leistung kommt aus der Pumpe
  (`m_pumpKeep`); unter 20 .. 40 Hz nicht gepumpt. D = Knopf × 1,3 × Stärke² × Hüllkurve × (1 + Breath).
- **26e Breathers:** der Ripple als Sine-Gordon-Kette (Masse bis 1 Hz); Schläge als Glocken (Breite c/m), je Harmonische
  1 .. 4 mit kleiner Drift (0,05 .. 0,15 c); gelesen als 2 sin(u/2) (ein Kink hinterlässt nichts). Gemessen
  (Python): ein angepasster Schlag bleibt bei 1 Hz ~18 s lokalisiert; bei 0,5 Hz zerlief er wie eine lineare Welle.
- **26f Horizon:** `horizonBend`: 1 + (R − 1) tanh((r − 1)/(R − 1)), Steigung 1 am Grundton, überall (`partialRatio`
  mit Horizont: Kernel, Normierung, Dirigent, Self-Tune, Rauheit, Ripple). Unter dem Horizont erreichen die
  Harmonischen den Cutoff nie: so viele wie ungebogen (mindestens 48).
- **26g Grains:** ein Anteil der Partials als Körner (Hann², Länge ±40 %, Dichte = Anteil der Zeit an, eigene Phase),
  jedes Korn an einem neuen Ort der Wolke und im Feld (springt, während es still ist), Make-up 1/√(3 Dichte / 8).
  Hörversuch: bei 20.000 Partials klingen auch bei 5 % Dichte Tausende Körner zugleich -- der Pegel schwankt in 20 ms
  nicht mehr als ohne (3,2 gegen 3,5 dB); hörbar körnig erst mit wenigen Partials (2000: 4,2 dB, 1000: 5,4 dB).
- **Kern:** `CloudVoice` trägt je Stimme die Anteile, Familien, das Glühen (48), die Subwolken, den Horizont und die
  Körner (768 Bytes); `Engine::stepHarmonics` rechnet Register, Haufen, Subharmonische und das Make-up (die Leistung der
  ersten 256 Harmonischen bleibt), `stepSeasons` die Jahreszeiten. Der Update-Kernel liest es je Harmonische; der
  Sync-Tisch kennt die Subwolken (Zeilen H+1 ..). MPE-Ziel **Breath** (Überblasen und Pumpe).
- **Seiten:** eine neue Seite **Harmonics** (Abschnitt Sound) mit Overblow, Subharmonics, Horizon, Seasons,
  Avalanche und Grain; Breathers im Ripple (Motion). Erst auf Cloud & Filter und Motion verteilt: beide wurden 10
  Pixel zu hoch (ctest `layout`).
- **Presets:** 1024 in 16 Gruppen (Abschnitt „Harmonic Lives“): Overblown Flutes, Breath Registers, Four Seasons,
  Turning Years, Avalanches, Glacier Calving, Subterranea (hoch gesetzter Dirigent, Root Age vertieft den Boden), Period
  Doubling, Halo Horizons, Aurora Curtains, Rainfall Grains, Breathing Grains, Breather Chains, Harmonic Ecology, Night
  Ecology, Wild Harmonics; 395 der 6144 bestehenden bekamen die neuen Mittel, gezogen nach allem anderen, auch nach dem
  Fundament -- nur hinzugefügt, kein Wert, den sie hatten, ändert sich (geprüft). Gefunden: `make_journeys.param_table`
  las seit der CGVR-Formatierung keinen Parameter mehr (`F( ParamIdE::`); Karte, Reisen und Nudges wären ohne Werte
  gerechnet worden -- die Muster vertragen jetzt Leerraum.

**Gemessen [M]:** Selbsttest 74/74 (68 Overblow: Grundton −31,9 dB unter der Oktave, Leistung −0,06 dB, nach dem
Überblasen bei wenig Atem −22,6, von Anfang an +7,2; 69 Seasons: Harmonische 3 und 5 schwingen 23 / 26 dB, Leistung
±0,02 dB; 70 Avalanche: 22 Glanzlichter und 64 Momente mit sechs oder mehr in einer Minute, ohne 0, bitgleich; 71
Subharmonics: −40,4 → +3,3 dB nach 20 s, −24,0 drei Sekunden nach dem Pumpenstopp, Leistung −0,03 dB; 72 Horizon:
höchster Partial 885 Hz bei 881 Hz Horizont; 73 Grains: 90 % der Partials still im Bild, Pegel +0,65 dB; 74 Breathers:
93 gegen 55 % der Energie in einer Oktave um den Schlag). Gemessen wird am Ziel jedes Partials (Schnappschuss), nicht
am Klang: die Mitglieder einer schmalen Wolke schweben gegeneinander, ihr Pegel wandert im Sekundenfenster um dB.
Hörbeispiele: `work/renders/phase26/` (Referenz, Overblow mit LFO-Atem, Seasons, Avalanche, Subharmonics unter einem
hohen Akkord, Horizon, Körner dicht/atmend/glitzernd/spärlich, Ripple mit und ohne Breather).

## Phase 25: Die Karte zoomt, das Fenster bleibt, wo es war (gebaut, 15.10.2026, 0.15.0)

Nutzer: „Bitte baue das mit dem Zoom noch ein.“ und „Beim Schließen sollte sich der Synth die Größe und Position merken“:
- **Karten-Zoom:** `MapView::view_` (Ausschnitt in Kartenkoordinaten), `toScreen`/`toMap` rechnen über ihn, alles
  andere folgt. Mausrad/Pinch um die Maus (1–32x, ein Rastschritt ≈ 1,4x), Rechts-/Mittel-Ziehen verschiebt
  (Rechtsklick auf Wegpunkt bleibt dessen Menü), Übersicht oben rechts (Klick/Ziehen), „Whole map“. Punkte-Cache nur
  für den Ausschnitt, Namen erst in Ruhe (180 ms) und nur bei ≤ 400 sichtbaren Punkten, höchstens 60, von der Mitte
  aus, ohne Überlappung (bei ~4,5x waren es sonst Hunderte). Der Ausschnitt gleitet dem Fokus nach (Cursor bei Map an,
  sonst das spielende Preset), wenn dieser von selbst hinausläuft; Randmarke zeigt ihn.
- **Fenster:** JUCE merkt sich nur windowX/Y und schiebt das Fenster beim Start ganz auf einen Bildschirm; die Größe
  galt nur, wenn sie auf den *Haupt*bildschirm passte. Jetzt `windowState` (ResizableWindow::getWindowStateAsString,
  „fs“ = maximiert) bei jeder Bewegung (ComponentListener, nicht minimiert/Vollbild) und `restoreWindowStateFromString`
  nach dem Start (nur Standalone, nur einmal); Breite/Höhe nicht im maximierten Zustand, Prüfung gegen alle Bildschirme.
  Geprüft: verschoben + geschlossen + neu gestartet = gleiche Lage. DisplayFusion teilt den 5120er-Bildschirm des
  Nutzers: „Maximieren“ ergibt dort 1920 breit ohne „fs“ – die Lage kommt trotzdem genau so wieder.

## Phase 24: Hand anlegen -- Zeitleiste, Hallgedächtnis, Wegpunkte, Hörlineal (gebaut, 15.10.2026, 0.15.0)

Nutzer: „Ja bitte, setze das um.“ (vierte Kritik, mein angepasster Plan; keine Routenvorschau, kein Auto-Füllen):
- **Journey-Zeitleiste:** oben die gewählte/laufende Reise, Klick auf einen Schritt springt dorthin
  (`skipJourneyTo`, startet sie, wenn keine läuft); unten der Entwurf (`journeyDraft`/`setJourneyDraft`): rechte
  Kante = Verweilen, Rampe = Gleiten (nach links länger), Block = Reihenfolge (Einfügemarke), Rechtsklick entfernt;
  Spannen lo/hi skalieren proportional, Pixelmaß beim Ziehbeginn fest. „Edit a copy“ übernimmt die obere Reise.
- **Hallgedächtnis malen:** Gauß-Pinsel über 10 Bänder, Höhe -> Oktaven -0,5..1,5 -> Gedächtnis `2^(oct+1)-1`
  (`Engine::setHallMemory`); Grow 0 -> 0,4 mit Hinweis, Modal Hall 0 -> Hinweis.
- **Wegpunkte:** ziehen (rastet auf Presets ein), Rechtsklick entfernt; Menüs an der Maus statt unter der Fläche.
- **Hörlineal (Tuning):** Klick auf eine Skalenlinie spielt die nächste Taste dieser Tonhöhe (`Engine::noteHz`, aus der
  Tastenzuordnung gerechnet) in der Oktave der Höhe; Shift hält, Verlassen der Seite lässt los.
- **Snapshots** im „Preset ...“-Menü der Morph-Slots (zuerst, `snapshotToSlot`).
- Mit echter Maus geprüft. Fallstrick beim Testen: ein synthetischer Rechtsklick auf ein JUCE-Menü lässt die rechte
  Taste in Windows gedrückt (GetAsyncKeyState), danach verwirft JUCE jeden Linksklick als Zweitklick -- vor dem
  Weitertesten `mouse_event(RIGHTUP)`; Menüpunkte per Tastatur oder Down/Up mit Pause wählen.

## Phase 23: Feinschliff nach der dritten Kritik, ASIO (gebaut, 15.10.2026, 0.15.0)

Nutzer: „baue das alles ein, auch das ASIO SDK“ (die Kritikpunkte und meine Vorschläge):
- **ASIO:** Steinbergs SDK 2.3.4 (GPLv3-Option seit 29.10.2025) in `_Sources/ASIO-SDK`, `JUCE_ASIO=1`, sonst JUCEs
  identische Kopie; der Layout-Bericht listet die Gerätetypen (Windows Audio, ..., ASIO).
- **Kopfzeile:** eigene Glyphen (HeaderParts.h: Mond, Lupe, Rahmen, A|B), Pegelanzeige mit Clip-Marke, A|B halten =
  Preset hören (`compareWithPreset`), Countdown unter Step (`ClusterBrain::untilNext`), Statuszeile ruhig.
- **Focus:** Variationen selbst gewürfelt (wenn sich der Klang bewegt hat, nicht in Blends), Keep erst nach Anhören,
  „+ Snapshot“, Teach in Schritten (`teachCancel`, `taught`), pulsierender Ring.
- **Modulation + Matrix** ein Tab (12 Tabs), Klick auf Quelle -> Inspektor; Palette/revealParam finden Modulator-Regler
  über `SourceInspector::sourceOf`. **Alt-Ziehen** am Knopf (Frame.h `ControlActions::altDrag`, zweiter optionaler
  Haken). **Undo der Matrix:** jeder Schritt trägt den Matrixtext (`extraBefore/After` war leer: Preset-Undo stellte die
  Matrix nicht zurück); Änderungen außerhalb eines Schritts werden nach 0,7 s Ruhe ein Schritt „Modulation".
- **Spannen und Ketten** in ParamPage (`Group::spans`, `links`); **TuningView** (Engine: `scaleCents`, `rootHz`,
  `VoiceView::tune`); **JourneyTimeline**; **Palette-Wörter** (Merkmale aus Find.inc); **Zoom** über
  `setScaleFactor` (Host-Skala darunter erhalten); letzte Seite gemerkt; Fundament-Taste unterstrichen; MIDI-Marken.

## Phase 22: Ein Fundament unter dem Dirigenten (gebaut, 14.10.2026, 0.15.0)

Nutzer: „Der Conductor scheint immer relativ hohe Noten zu spielen, aber selten ein solides Bassfundament.“
Befund: Empyrean reicht dem ClusterBrain keine Registerrollen durch (Layers, Bass Hold ... stehen auf aus), und im
Chords-Modus (fast alle Presets) bewertet `chooseNote` nur Konsonanz und Stimmführung; der Grundton wird sogar
halbiert. Gemessen mit `additiv_presets --register` (je Gruppe ein Preset, 3 min; work/register/).
- **Foundation** (`brain_foundation`, 0..1, Conductor > Register): die Engine (nicht das geteilte ClusterBrain) hält
  eine Stimme auf dem Grundton in der Oktave ab C2 (ab Lowest, wenn tiefer; Quelle 4, als Dirigent gezeigt, wird beim
  Stimmenklau verschont, kommt wieder, wenn sie doch genommen wurde); der Akkord wird ab einer Quinte darüber gezogen
  (mindestens eine Oktave Raum). Pegel (0,6 + 0,4 x Foundation) x halbe Summenleistung der Akkordstimmen (als Amplitude,
  über 4 s geglättet), nah (Tiefe x 0,15). Erst stand es nah und lauter: unter einem noch einziehenden Akkord
  (Lunar Hall: 24 s Attack, eine Note alle 163 s) war es +9 dB und 81 % der Leistung unter 150 Hz.
- Messwerkzeuge laden Presets jetzt wie das Panel mit `requestFill()` (sonst misst man den langsamen Einstieg:
  im Mittel 1,1 Noten); empy_render `--fill`.
- Presets: `foundation(r)` zuletzt gezogen, 85 % bekommen 0,45..0,85 (keins mit Lowest >= 55): 4956 von 6144.
- Selbsttest 67: tiefste Note = Grundton unter C3 in 100 % der Blicke (ohne: 47 %), kein Akkordton näher als eine Quinte.
- Gemessen an 48 Presets (je jede zweite Gruppe, 150 s, Akkord sofort voll): tiefste Note = Grundton 22 % -> 82 % der
  Zeit, auf H2 oder tiefer 66 % -> 90 %, Leistung unter 150 Hz im Median gleich (14,2 -> 14,4 %): der Bass war oft da,
  aber nicht der Grundton und nicht stehend. Pegel: Median -0,1 dB (-2,1 .. +4,1). Zehn Presets bleiben fast ohne
  Bass: drei ohne Fundament (Lowest >= 55), die übrigen formen ihre Teiltöne nach einer unten dunklen Landschaft.
  Hörbeispiele: work/renders/foundation (Lunar Hall, Starry Milky Way, je vorher/nachher).

## Phase 21k: Mehr Platz, Seiten mit Status, Modulationsleiste (gebaut, 14.10.2026, 0.15.0)

Nach einer zweiten HCI-Kritik (blinde Tabs, Tab-Wechsel, Modulation nur auf eigener Seite; Vital/Pigments als
Vorbild) und dem Hinweis des Nutzers, Musiker hätten mindestens 1920 x 1080:
- **Startgröße nach Bildschirm:** drei Viertel der Breite des Arbeitsbereichs, Seitenverhältnis 0,625, mindestens
  1280 x 860, höchstens 2200 breit; die zuletzt verlassene Größe kommt wieder (player.settings windowWidth/Height).
  Das Bild oben wächst mit (ein Drittel, bis 420 px). Untergrenze bleibt 1280 x 860 (ctest `layout`).
- **PageStrip** statt der JUCE-Tableiste: Abschnitte Sound, Harmony, Space, Modulation, Explore; je Seite ein Punkt
  pro Gruppe mit Hauptreglern (`ParamPage::groupStates`, auch für verborgene Seiten), Name gedimmt wenn alles ruht,
  Matrix zählt Zeilen, Journey/Map zeigen laufende Reise/Route; Tooltip „on: ...; off: ...". Morph hat jetzt ein Gate.
- **Cloud & Filter** auf einer Seite (3 Reihen); die Bild-Overlays beider Seiten zusammen.
- **ModDock:** Chips mit Live-Kurve (klassische 14, dazu jede Quelle, die eine Zeile liest, eine per „+");
  Ziehen auf einen Knopf → `knobmod::addRow` (Tiefe 0,2; dieselbe Quelle auf denselben Knopf gibt keine zweite Zeile);
  ParamPage ist DragAndDropTarget mit Vorschau-Ring. **SourceInspector:** Kurve, eigene Regler, alle Ziele mit Tiefe
  und Entfernen; angedockt, wo das Fenster hoch genug ist (ab etwa 970 px), sonst über den Seiten.

## Phase 21j: Jede Seite passt ins Fenster (gebaut, 14.10.2026, 0.15.0)

Der Nutzer musste je nach Tab scrollen oder das Fenster vergrößern. Gemessen (ADDITIV_LAYOUT, jetzt ctest `layout`)
bei 1280 x 860, 527 px unter den Tabs: Focus brauchte 922, Modulation 1050, Space 772, Effects 586.
- `ParamPage::flow`: die wenigsten Reihen; ein späterer Kasten rückt nur in die Lücke einer früheren Reihe, wenn das
  eine Reihe spart (Effects 4 → 3, Space 3 → 2, Modulation 3 → 2).
- Focus in zwei Spalten ab 1150 px Breite: links Nudge, Fokus, Makros; rechts Schnappschüsse, Variationen, Suche.
- Modulation geteilt: die Modulatoren mit ihren 28 Quellen live; neue Seite Matrix: 16 Zeilen in einer Spalte, alle
  42 Quellen live daneben. Die Quellen-Kacheln passen ihre Spalten der Höhe an.
- Raum- und Hallbild nehmen den Rest (mindestens 200/220 px); das Bild oben weicht der höchsten Seite (120..260 px).
- Ergebnis: bei 1280 x 860 und 1280 x 800 scrollt nichts; erst bei 760 px Höhe drei Seiten um 12-20 px.

## Phase 21: Eine Oberfläche zum Wiederfinden (gebaut, 14.10.2026, 0.15.0)

Meine Vorschläge, vom Nutzer alle gewünscht (vorerst nur in Empyrean, dann Entscheidung über die Familie):
- **21a Eigene Presets** (es gab keine!): `Documents/Empyrean/Presets/*.empyrean` (XML: Werte, Matrix), Speichern
  Strg+S, Speichern unter (Werks-Namen abgelehnt: Reisen finden beim Namen), Referenz je Preset → Stern im Namen,
  Kerbe am Knob, Zurück für Knob und Ganzes. Matrixvergleich über packModRow (wie die Engine schreibt).
- **21b Makros** als große Drehknöpfe (Literatur: kein Befund für kleine Slider; Fitts, Hardware-Encoder), benannt
  nach dem stärksten Ziel (Lektion oder Zeile), gleichnamige Parameter mit Bereich (Motion Depth, Space Depth).
- **21c Befehlspalette** Strg+K: Regler (Seite, Aufleuchten, Wert), Presets, Aktionen; verstreute Buchstaben.
- **21d Info-Zeile** (Ableton Info View): 210 Sätze in `ParamInfo.cpp`, Wert, Matrix-Quellen; sonst Tooltip.
- **21e Preset-Browser**: Kategorien, Suche, Eigenschaftsbereiche (aus `Find.inc`), „wer spielt“, Sterne und
  Verlauf (`player.settings`), Vorhören per gehaltenem ▶.
- **21f–i**: Nachtmodus (eigener Schleier unter den Überlagerungen), Schnappschüsse (Bild aus der CloudView),
  MPE an den Tasten (VoiceView: press, slide, bend), erste Schritte (Tour, sieben Hinweise).

## Phase 20: Einfache Wege hinein -- die Fokus-Seite (gebaut, 14.10.2026, 0.15.0)

Ideen aus der Forschung, vom Nutzer gewählt (ohne CLAP-Textsuche): semantische Stupser (Social-EQ, SAFE), Fokus
(schrittweises Aufdecken), Variationen mit Vorhören (interaktive Evolution nach Dahlstedt, Side Views nach Terry &
Mynatt), Makro aus Beispielen (Wekinator, Fiebrink), Suche nach Beispiel (SynthAssist, Cartwright & Pardo).
- **Messung:** `additiv_presets --sensitivity`: um 48 Presets (je Gruppe zweite) jeder der 47 formenden Regler um 0,15
  seines Wegs verstellt, Helligkeit/Bewegung/Breite/Rauschen/Pegel neu gemessen; `make_nudges.py` mittelt je Regler über
  die Presets, in denen er zu hören ist (Gate offen), und schreibt `Nudges.inc` samt Gates. `describe()` liegt jetzt im
  Kern (`Describe.h`).
- **Stupser:** erst kürzester Schritt über alle gehörten Regler mit Pegel und anderen Maßen per Gram-Schmidt
  herausprojiziert -- Prüfung (`--nudgecheck`, 8 Presets × 8 Stupser): ein Viertel falsche Richtung, Sprünge bis 4 dB.
  Jetzt: die bis zu vier spezifischsten Regler (viel für das Maß, wenig für die anderen und den Pegel), Pegel gehalten,
  ein Mengenregler auf null bleibt null (Air von 0 auf 0,07 machte einen Chor 5,9 dB lauter) → Helligkeit und Breite
  16/16, Rauschen 15/16 richtig, Pegel meist ±1 dB. Bewegung blieb 10/16 (8-s-Fenster gegen minutenlange Zyklen):
  ruhiger/lebendiger jetzt als Regel (alle Raten ×/÷ 1,23, Drift und Schimmern ±0,04). **Fokus:** Summe der Empfindlichkeiten in
  Spreizungen, höchstens drei je Gruppe. **Variationen:** 4–8 gehörte Regler zufällig, Pegel über Master nach der
  gemessenen Pegel-Empfindlichkeit. **Makro-Lektionen:** relative Versätze je Ort (bis 4 Orte), in der Engine wie
  Matrixzeilen vor den Gates, im Zustand gespeichert. **Suche:** vier Merkmale (Bewegung log), je Spreizung skaliert
  (`make_find.py` → `Find.inc`), WAV/FLAC stereo (`loadStereo`).
- **Bank neu gemessen:** der Selbsttest der Suche fand „Pale Choir“ nicht an seinem eigenen Klang -- die Engine-
  Änderungen seit Phase 16 hatten die 3530 älteren Presets verschoben (863 über 1 dB, 173 über 2 dB, bis 9,7 dB neben
  dem Pegel der Bank). Neu gemessen (75 min), gepegelt, Karte und Suchtabelle neu. Lehre: nach Engine-Änderungen, die
  den Klang aller Presets berühren, eine Stichprobe alter Presets nachmessen.

## Phase 19: Der Dirigent folgt dem Licht; Kristallisation als Hörversuch (gebaut, 14.10.2026, 0.15.0)

- **Follow Light:** je Kandidat das Licht an seinen Harmonischen (Klangspektrum des Dirigenten) -- lebende Landschaft
  über die Spalten gemittelt, sonst die Landschaft der Aurora-Bewegung über die ersten 8 s; Faktor (Licht/Mittel des
  Registers)^(3·Follow Light) in beiden Zügen des Dirigenten. Gemessen: Band 200–400 Hz hell → 38 % ohne, 69 % mit.
- **Kristallisation** ohne neuen Code als Hörproben (`work/renders/crystal`): LFO-3-Gezeiten (8 min) über Feldstärke,
  Purity, Wolkenbreite, Schimmern und Kopplung, Lautheit schmilzt; und Root Age (je länger die Harmonie steht, desto
  gefrorener). `empy_render --mod` für Matrixzeilen von der Kommandozeile. Ob eine eigene Front-Dynamik folgt, entscheidet
  das Anhören.

## Phase 18: Die Oberfläche nimmt die Hand (gebaut, 14.10.2026, 0.15.0)

Eine Kritik der Oberfläche nach HCI-Maßstäben (Hierarchie, direkte Manipulation, Modulation am Regler, 2D-Raum,
Zen-Modus); umgesetzt, verbunden mit eigenen Ideen. `Frame.h` bekam einen optionalen Haken (`ControlActions::more`) für
eigene Menüeinträge, sonst bleibt der Rahmen der Familie unberührt.
- **18a Modulation am Knob** (`KnobMod`): je Zeile ein Bogen in der Farbe ihrer Quelle (Spanne, Punkt für jetzt) über
  `ParamPage::paintOverChildren`; das Rechtsklickmenü listet die Zeilen (Tiefe ziehen, entfernen) und „Modulate with“
  legt eine Zeile in der ersten freien an. Gruppen haben ein `gate`: stehen alle auf null (und keine Zeile bewegt sie),
  ruht die Gruppe gedimmt; die Gate-Regler sind betont (Name in Gruppenfarbe, Halo).
- **18b Das Bild zeigt die Seite vorn** und lässt sich greifen: Filterkurve (die Formel jetzt einmal in `Spectrum.h`,
  GPU und Bild rechnen dasselbe), Kamm-Kerben (Phase aus der Engine), Air-Band und Helligkeitsgrenze, Register-
  Klammer, Ripple-Saite (Snapshot), Hallgedächtnis und Lowest. **Canvas:** das Bild allein, Landschaft dahinter, Makros,
  Tasten spielen per Klick (`MidiKeyboardState` in den MIDI-Puffer).
- **18c Space von oben** (`SpaceView`): Stimmen auf ihrer Ebene und Tonhöhe mit Spuren; Depth und Keys Depth als Ringe.

## Phase 17: Die wachsende Halle, Eigenzeit, verweilender Ripple, Orbit (gebaut, 14.10.2026, 0.15.0)

Ein zweiter Ideentext (spektraler Magnetismus, spektrale Chemie, Zeitdilatation und Linsen, vorausschauender Dirigent,
Morphogenese der Halle); gebaut, was sich messbar und hörbar umsetzen lässt. Offen und mit dem Nutzer zu klären:
Kristallisation als Experiment, ein Dirigent, der hellen Stellen der Landschaft folgt.
- **17a Halle:** `updateKernel` sammelt die Amplitude jedes Partials ganzzahlig (`atomicAdd`, uint64, 1e-12) in 1024
  log-Bändern 20 Hz .. 20 kHz; `hallKernel` (ein Block, feste Reihenfolge) macht daraus die relative Dichte (Mittel 1,
  über drei Bänder geglättet), folgt ihr über 0,3 s und schreibt das Gedächtnis M mit 1 − exp(−dt/Memory), solange
  etwas klingt. `modalKernel`: T ·= 2^(Grow · clamp(log2(1 + M) − 1, −0,5, 1,5)). Amplituden statt Leistungen: mit
  Leistung zählten nur die lautesten Bänder, die Obertöne eines Akkords lagen unter dem Mittel und wuchsen nicht.
  Gedächtnis im Plugin-Zustand (base64), .hall-Dateien, bleibt über `prepare()` hinweg.
- **17b Eigenzeit:** τ je Partial, Rate 1/(1 + D·s) bzw. 1 + |D|·s mit s = clamp(log2(1 + ρ) − 1, 0, 4); Drift,
  Schimmern und Sync laufen auf τ. **Ripple Bend:** Dichte der klingenden Stimmen (16 Harmonische, σ = 4 Zellen),
  c_i = c/(1 + 4·Bend·s); konservative Form (c²u_x)_x -- in c²u_xx verließen Wellen die langsame Zone lauter
  (gemessen: Energie 164 % statt 35 %).
- **17c Orbit:** Zustand (Versatz, Seite, zwei Geschwindigkeiten) je Partial, Ping-Pong; Mulde am nächsten Skalenton,
  Drehung der Geschwindigkeit um ω_c·dt exakt, Rauschen σ = ½R·√(2γk·dt) (mittleres Quadrat ½R nach dem
  Gleichverteilungssatz, in Python geprüft). Fehler gefunden: der Header kappte `count` außer bei Scale auf 8.
- **17d Presets:** 1024 in 16 Gruppen (Abschnitt „Halls & Orbits“), 307 bestehende ergänzt (nach allem anderen
  gezogen); gemessen wurden die 1331 neuen und geänderten.

**Gemessen [M]:** Selbsttest 61/61: 58 Gedächtnis 15,8 am Grundton, Ausklang 11,2 dB/s neu, 4,9 gewachsen,
16,1 erodiert, 3,3 nach dem Laden in eine neue Engine; 59 Drift 0,94 → 0,45 (langsam) / 2,58 (schnell) ct je Bild;
60 Grundton kreist 6,9 ct um seinen Ton, 11 Nulldurchgänge in 10 s, Pegel ±0,1 dB; 61 Ripple nach 0,6 s 23 % ohne,
70 % mit Bend am Grundton.

## Phase 16: Schwärme, Strömungen, Turbulenz, Transport, MPE (gebaut, 14.10.2026, 0.14.0)

Ein Text über Forschung zu GPU-Audio, DDSP, physikalischer Modellierung und Schwarmdynamik; meine Einschätzung:
gebaut wurde, was zum Instrument passt (Tensor-Kerne, Reibung, Raytracing je Partial und Verdeckungs-Culling nicht).
- **16a Sync-Modi:** Chimera (Kopplung an die Nachbarn rund um den Ring der Wolke, 1 + 0,995 cos, Verzug 1,25 .. 1,55,
  Start nach Abrams/Strogatz) und Swarm (Swarmalators auf dem Ring nach O'Keeffe et al.: Platz x auf einem Kreis in
  Tonhöhe Reach·cos x und Seite ½·sin x; drei mittlere Zeiger je Wolke). Zustände simuliert (Python): Phasenwelle bei
  K = 0, J > 0 oder K ≈ 1, J < 0; zwei Gruppen bei K, J > 0; zersplittert an den Grenzen; verstreut bei J < 0.
- **16b Fluid und Islands:** Lattice-Boltzmann D2Q9 (BGK) auf dem Torus des Felds, Düsen der acht lautesten Noten an
  Frequenz und Oktave, Tinte semi-Lagrange; Cahn-Hilliard mit Oono-Puri auf 64×32 in einem Block (Shared Memory).
- **16c Turbulence:** Leistung je Harmonischer auf der GPU, Fluss drive·p² (7/10 nach h+1, 3/10 nach 2h), Rückkehr über
  Settle/(1 + h/8); Antrieb kubisch in Notenstärke und Lautheit.
- **16d Filter-Transport:** MorphState hält gewichtete Filter (einer je Modell); `filterTransportKernel` bildet das
  Wasserstein-Baryzentrum der Leistungskurven (1024 Bänder, Mittel der Quantilfunktionen); ein Filter: direkt.
- **16e MPE:** Kanäle 2..16 je eine Note (Biegung, Druck, CC 74), Kanal 1 Master; RPN 0/6; Press/Slide-Ziele.
- **16f Presets:** 1024 in 16 Gruppen; 324 der 4096 bekamen die neuen Mittel (nach allem anderen gezogen).
  `additiv_presets --describe` führt die Beschreibungen jetzt zusammen; gemessen wurden die 1348 neuen und geänderten.

**Gemessen [M]:** Selbsttest 57/57: 53 Chimäre 0,49 der Nachbarn im Gleichschritt, Schwarm gleich weit (0,3 ct) bzw.
verstreut (11,9 ct); 54 Tinte 0,79 an den Noten gegen 0,28, Inseln 578 → 370 Grenzen; 55 Schwerpunkt laut +1,23,
leise +0,52 Oktaven; 56 halbwegs tief −27,6 / Mitte +1,9 / hoch −27,2 dB; 57 Biegung 200 ct, Druck +3,00 dB.

## Phase 15: Korrekturen nach dem Anhören (gebaut, 13.10.2026, 0.13.1)

Der Nutzer: "das Laden der Presets über das Menü kaputt", "sind die Presets wirklich so regelmäßig
verteilt?", "beim Morphen mit den Modulationen ... Unstetigkeiten".
- **Menü:** `choosePreset` (Menü, Pfeile, Programmwechsel, Headset): Journey und Pfad halten, Morph aus,
  die Karte lässt los (Cursor auf den Punkt), dann `applyPreset`. Ursache: unter A/B nur die Regler,
  bei Karte nur der Cursor, und Radius 0,05 war das Dreifache des Punktabstands (sechs Presets gemischt);
  Selbsttest 51 prüfte "auf dem Punkt" nur mit 0,004.
- **Modulatoren je Klang:** `ModSetup` (LFO- und Hüllkurven-Formen und -Raten) in `MorphState::setups`,
  `ModRow::setup`; die Engine lässt je Setup eine `OwnModulators`-Einheit laufen (eine Einheit, die
  keiner mehr liest, übernimmt ein neues Setup und läuft weiter; eine neue startet als Kopie der
  stärksten). **Gates** (`MorphState::gate`) für Bewegung, Leben, Feld: der Matrix-Anteil der Menge
  wird damit multipliziert, durch null am Wechsel. **approachMorph:** eine andere Art blendet erst aus
  (doppeltes Tempo), Begleiter bleiben, Wechsel unter 0,005; ganze Zahlen gleiten ungerundet (die
  Engine rundet). **Landschaft:** wechselt nur, wo alle Leser (Aurora, Bilder) bei null sind.
- **Karte gemessen:** `additiv_presets --describe` (Schwerpunkt in Oktaven über 40 Hz; Bewegung =
  Betragsänderung von 36 Bändern über 0,25/1/3 s relativ; Breite; Flachheit), 4096 Presets in 111 min
  → `work/presets/descriptors.json`; `make_map.py`: halb Rang, halb Wert (Perzentile 1/99 auf 0,05/0,95,
  tanh-Ränder), leichte Abstoßung. Korrelation der alten Achsen mit den gemessenen: 0,24 / 0,16.
- **Radius relativ:** σ = Radius × Abstand zum zweitnächsten Preset (0,1 .. 4, Standard 0,4: auf dem
  Punkt 0,977 des Gewichts); Gewicht des siebten Nachbarn und 2 % werden abgezogen (stetig). Pfade
  in derselben Einheit (0,5 .. 1,5).

**Gemessen [M]:** Selbsttest 52 vorher/nachher: Helligkeit je Block 0,0041 → 0,0017 (As LFO selbst
0,0020); Bewegungstiefe am Wechsel 0,30 → 0,000; Kartensprung Partials 0,91 → 0,004 ihres Bereichs je
Block, Menge am Artwechsel 1,0 → 0,005; Bildtiefe am Landschaftswechsel 0,004.

## Phase 14: die Preset-Karte und ihre Pfade (gebaut, 13.10.2026)

Der Nutzer: "eine Preset-Karte wie in Noctuary ... Pfade definieren ... Zeiten für die Dauer der Morphs".
- **Karte** (`make_map.py` → `PresetMap.inc`): quer dunkel→hell, hoch still→bewegt (Rang, hoch innerhalb von 32
  Helligkeitsstreifen, damit keine Ecken leer bleiben; 40 Abstoßungsschritte).
- **Mischung** (`PresetMap.cpp`): sechs nächste Nachbarn mit Gauß-Gewichten, in Lücken wächst der Radius (Noctuary);
  `MapBlender` mischt sie nacheinander mit `blendMorph` (gewichtetes Mittel), Nachbarn unter 2 % bleiben weg. Die
  Engine bekommt das Ziel als dritten Slot und folgt ihm blockweise (`approachMorph`, Map Glide), ganze Zahlen und
  Umschalter sofort (das Mischen hat ihre Mengen schon durch null geführt).
- **Pfade** (`Route.h` aus Noctuary, gleiches Textformat): Wegpunkte (Preset oder x,y) mit Reise-, Haltezeit und
  Radius; zwölf mitgelieferte aus `make_map.py`, eigene in Documents/Empyrean/Routes. Der Processor läuft sie im
  30-Hz-Takt ab (Route Speed). `empy_render --route`.
- **Seite Map:** Karte (Klick, Ziehen, Shift-Klick für Wegpunkte, Nachbarn markiert, Name unter der Maus) und
  Pfad-Editor (m:ss). Preset-Wahl bei eingeschalteter Karte fährt den Cursor hin, Karte aus lässt die Regler beim
  nächsten Preset, eine Journey verlässt die Karte.

**Gemessen [M]:** Selbsttest 51/51 (47 Stretch/Shift, 48 Kamm, 49 eigene Landschaft, 50 Bilder, 51 Karte).

## Phasen (Vorschlag)

0. Messprogramm und Hörvergleich: gebaut, 20.000 entschieden. 0b. Hüllkurven pro Partial: gebaut, vom Nutzer gut
   aufgenommen (12.10.2026).
1. **Das spielbare Instrument** (erweitert 12.10.2026; **gebaut**, siehe oben): Stimmverwaltung (Note an/aus, Stehlen mit Ausblenden,
   Anschlag = Nähe), Hüllkurve pro Stimme, Vorausrendern (Worker, Ring, Zeitstempel, Einsatz samplegenau im GPU-Block,
   Latenz an den Host), REALTIME-Klasse; **reine Stimmung** und **Dirigent mit Autoplay** aus Noctuary; die GPU liefert
   **nah und fern** (4 Kanäle), die Entfernung ist eine Eigenschaft des Partials; Far-Reverb, Mid/Side und Soft Clip aus
   Noctuary; Parametertabelle; ein schlichter Standalone und VST3 (JUCE, generische Oberfläche); ein Renderer für die
   Nacht ohne Oberfläche; Selbsttests.
2. **Gebaut (siehe oben).** Klangmodell: Noctuarys Spektralformel (Tilt, Brightness, Odd/Even, Inharmonic, Match auf die Skala) mal Wolke
   (Breite, Steigung, Dichte), Drift, Shimmer; Raummodell und Filter pro Partial auf der GPU; Landschaften als
   Kernstück (Bibliothek, Pegelausgleich); Werksformen nach "Bewegung auf tragendem Klang"; Rauschanteile.
3. **Gebaut (siehe oben).** Plugin (VST3, Standalone) mit der Frame.h-GUI der Familie und einem Bild der Partials.
4. **Gebaut (siehe oben).** Modulation und Presets (1024 wie die Geschwister; seit Phase 10 2048).
5. **Gebaut (siehe oben).** Familie: Jam, Headset im Bridge-Modus (eine Quest-App mit CUDA gibt es nicht).
6. **Vorbereitet (siehe oben).** Release für Windows (Linux optional); veröffentlicht wird nach dem Namen.
