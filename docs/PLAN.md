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
- **Skin** "Aurora bei Nacht" (Nachtblau, Türkis, Violett; Logo: ein Ring aus Partials), Fenster 1280 x 860.
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
Denormals jetzt je Block aus (`NoDenormals`, FTZ und DAZ), denn der Worker ist nicht JUCEs Audio-Thread. **Offen:**
mehr Vorlauf nur beim Start oder eine Einstellung für den Vorlauf (10/20/40 ms).

## 0.9.1: das Bild zeigt die Partials (12.10.2026)

Der Nutzer: "die Partials selbst bleiben immer konstant". Gemessen bewegen sie sich (Bandenergie einzelner
Harmonischer über 20 s: ruhende Wolke 1–4 dB in den oberen Harmonischen, Breathe 5–8 dB, Velvet Dreaming mit Aurora
15–22 dB) -- aber das Bild zeichnete nur die Form jeder Stimme aus den Reglern. Jetzt schreibt der Update-Kernel für
jeden achten Partial (f, Amplitude) in `dSnap`; `launch()` kopiert sie, wenn verlangt, hinter dem Graphen vor dem
Ereignis, auf das `wait()` wartet; das Engine verlangt es etwa alle 33 ms und reicht die 51.200 Punkte über einen
Mutex weiter, den der Worker nur versucht (nie wartet). Die Ansicht: Punkte (log f quer, −100 … −35 dB hoch) in
einem Glühpuffer, der je Bild auf 80 % fällt. Selbsttest 33: 2499 von 2560 Punkten der klingenden Stimme, alle an
ihren Harmonischen, keiner bei stummen Stimmen.

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
4. **Gebaut (siehe oben).** Modulation und Presets (1024 wie die Geschwister).
5. **Gebaut (siehe oben).** Familie: Jam, Headset im Bridge-Modus (eine Quest-App mit CUDA gibt es nicht).
6. **Vorbereitet (siehe oben).** Release für Windows (Linux optional); veröffentlicht wird nach dem Namen.
