# AdditivAmbientSynth: Plan für den additiven GPU-Synth

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

`Gpu/` ist der Kern der späteren Engine: `CloudVoice` (die wenigen Zahlen einer Stimme), `CloudEngine` (Tabellen,
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

**Urteil des Nutzers (11.10.2026):** Zwischen 2.000 und 20.000 ist der Unterschied schwer zu hören; weil die Leistung
gleich ist (16 Stimmen: 40 gegen 47 µs GPU pro Block [M], 0,3 % des Budgets), gilt **20.000 Partials pro Stimme**.
Die nächste Frage des Nutzers: eine eigene, loopbare, komplexe Hüllkurve für jeden Sinus (Phase 0b).

## Phase 0b: eine Hüllkurve für jeden Partial (gebaut, 11.10.2026)

Der Nutzer: "Könnten wir eigentlich auch jedem Sinus eine eigene (loopbare) komplexe Hüllkurve zuweisen?" und
"baue das alles ein".

**Der Haken, der das Design bestimmt.** Unabhängige Zufallshüllkurven auf 110 bis 333 Partials pro Harmonische mitteln
sich weg; die Harmonische schwankt dann nur noch um etwa 1/√K. Das Spektrogramm der Szene `random` bestätigt es [M]: kaum
von `none` zu unterscheiden. Hörbare Struktur entsteht nur, wenn die Hüllkurven der Partials **zusammenhängen**.

**Modell** (`Gpu/include/additiv/Cloud.h`):
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
schon mal sehr interessant und ziemlich anders als was man von normalen Synthesizern so kennt"; ein Urteil über einzelne
Szenen steht aus.

Beobachtet: Landschaften senken den Pegel stark (aurora −42 dBFS roh, weil der Boden bei −30 dB liegt); im Synth
braucht die Landschaft eine Pegelkorrektur.


`probe render --partials 200,2000,20000` schreibt `work/renders/cloud/cloud-<n>.wav`: Akkord A2 E3 B3 C4 E4, 30 s,
alle auf −20 dBFS RMS. Bei 200 Partials hat jede Harmonische 1 bis 3 Partials (fast Noctuarys Bank), bei 2.000 sind es
11 bis 33, bei 20.000 110 bis 333. Stellschrauben: `--width` (Cent, 25), `--slope` (0,3), `--drift` (Cent, 3),
`--shimmer` (0,4).

## Folgerungen für die Architektur [I]

1. **Der Audio-Callback wartet nie auf die GPU.** Ein Worker rendert feste GPU-Blöcke (128 Samples) in einen Ring mit
   Vorlauf (Start 10 ms, einstellbar); das Plugin meldet die Latenz dem Host. Noten und Parameter tragen Zeitstempel.
2. **GPU-Klasse REALTIME.** Sie gilt pro Prozess: Im Standalone unproblematisch, im Plugin beträfe sie den ganzen
   DAW-Prozess. Zu entscheiden, wenn das Plugin kommt: Klasse im Host-Prozess setzen oder ein kleiner GPU-Server-Prozess
   (ein CUDA-Kontext für alle Instanzen, Ring im Shared Memory).
3. Die Stimme bleibt ein paar Dutzend Zahlen; die GPU rechnet daraus die Partials. Presets und Modulation arbeiten wie in
   der Familie auf diesen Zahlen.
4. Eine CPU-Referenz nur für Tests und CI (GitHub-Runner haben keine GPU), kein macOS.

## Phasen (Vorschlag)

0. Messprogramm und Hörvergleich: gebaut, 20.000 entschieden. 0b. Hüllkurven pro Partial: gebaut, vom Nutzer gut
   aufgenommen (12.10.2026).
1. Engine: Stimmverwaltung, Hüllkurven, Ring und Worker, REALTIME-Klasse, Parameterglättung, Selbsttests.
2. Klangmodell: Noctuarys Spektralformel (Tilt, Brightness, Odd/Even, Inharmonic, Match auf die Skala) mal Wolke
   (Breite, Steigung, Dichte), Drift, Shimmer, Ort pro Partial; Rauschanteile.
3. Plugin (VST3, Standalone) mit der Frame.h-GUI der Familie und einem Bild der Partials.
4. Modulation und Presets (rund 1024 wie die Geschwister).
5. Familie: Jam, Keyboard-Spiel, Quest.
6. Release für Windows (Linux optional).
