# FocusWatch Dataset — Design für die Datenpublikation (Elsevier Data in Brief)

Status: Entwurf zur Review
Datum: 2026-08-14

## 1. Ziel

Der zusammengeführte Korpus aus drei Erhebungen wird als **Data Descriptor bei
Elsevier "Data in Brief"** publiziert. Deliverable ist der Datensatz selbst,
nicht das ML-Ergebnis.

Dafür entsteht ein **neues, eigenständiges Repository in der GitHub-Org
`CH-GE-Focus-Watch`** mit einem Python-Package, das alle Quellformate über
Adapter einliest, gegen einen physikalischen Validator prüft und ein einheitlich
beschriebenes Parquet-Bundle schreibt. Nachnutzer stellen sich ihren
Arbeitsdatensatz über Flags im Manifest selbst zusammen, statt drei Quellschemata
kennen zu müssen.

Der Zielzustand ist **nicht** eine dritte Kopie der ML-Pipeline. Merge,
Alignment, Label-Ableitung und Feature-Extraktion bleiben Methode und wandern
als Code ins Package, nicht als Artefakte in den Datensatz.

## 2. Korpus

Der publizierte Korpus ist **exakt das Google-Drive-Bundle** (lokal `alldata/`,
gitignored). 67 Recordings aus vier Quell-Pipelines:

| Kohorte | Recordings | Modalitäten | Ground Truth |
|---|---:|---|---|
| ML4SCS | 33 | Watch + Pen + Marker (lückenlos) | Pen `dot_type`, Task-Marker |
| ETH Ege-Pipeline | 2 (T6, T7) | Watch + Head + Pen + Web-Events | Pen-Events, Phasen-Events |
| ETH SensorLogger | 7 (E1–E3, T8–T10, S3) | Wrist + Head (+ Roh-Accel in 6/7) | Pen-Events im Session-JSON (nur E1–E3), Phasen-Events |
| AirPods-Attention | 25 (P1–P19, P21–P26) | Head-IMU (~25 Hz) | Beobachter-Intervalle |

Gemessene Modalitäts-Abdeckung — sie ist bewusst lückenhaft, und genau das
begründet das Flag-Modell:

| Tabelle | Recordings | Zusammensetzung |
|---|---:|---|
| `watch/` | 42 | 33 ML4SCS + 2 Ege + 7 SensorLogger |
| `watch_rawaccel/` | 6 | SensorLogger ohne E3 (dort fehlt die Datei) |
| `headimu/` | 34 | 25 AirPods + 2 Ege + 7 SensorLogger |
| `pen/` | 38 | 33 ML4SCS + 2 Ege + 3 SensorLogger (E1–E3) |
| `markers/` | 42 | 33 ML4SCS + 2 Ege + 7 SensorLogger |
| `attention/` | 25 | AirPods |

**T8, T9, T10 und S3 tragen keine Pen-Strokes** — ihr Session-JSON enthält nur
ein `pen_session_sync`-Event, keine `pen_down`/`pen_move`/`pen_up`. Das erklärt,
warum der bestehende `foreign_adapter` sie nie verarbeitet hat: es gibt nichts
zu alignen. Sie werden trotzdem publiziert (`has_pen = false`): rund vier
Stunden Watch- und Head-IMU mit Phasen-Markern sind für selbstüberwachtes
Vortrainieren wertvoll und für einen überwachten Benchmark unbrauchbar — die
Flag entscheidet, welcher Nachnutzer sie bekommt.

Die Pen-Coverage der Pen-tragenden Kohorten ist außerdem sehr ungleich:
ML4SCS liefert kontinuierliche Moleskine-Ströme, Ege und SensorLogger liefern
Web-App-Events (T6: 2.516 `pen_dot`, 236 Strokes; E1: 16.530 `pen_move`,
1.159 Strokes).

Die 33 ML4SCS-Sessions stammen von 33 **verschiedenen** Personen. Verteilung
der Aufnahmeparameter — die beiden Achsen sind **unabhängig**:

| | `ml4scs_v1` | `ml4scs_v2` |
|---|---:|---:|
| 50 Hz | 9 | 0 |
| 100 Hz + Gravity | 4 | 20 |

Die vier `100hz_grav` + `v1`-Sessions sind der Grund, warum `watch_hz` und
`protocol_id` getrennte Flags sein müssen: ein Nachnutzer, der sie über eine
einzige "Profil"-Spalte sucht, findet sie nicht.

### 2.1 Zwei Ground-Truth-Semantiken

Der Korpus trägt zwei verschiedene Forschungsfragen:

- **Schreiben vs. nicht schreiben** — Ground Truth ist der Pen (`dot_type`),
  sample-genau, in ML4SCS und beiden ETH-Pipelines.
- **Fokussiert vs. abgelenkt** — Ground Truth ist eine Beobachter-Annotation in
  Minuten-Intervallen, nur in der AirPods-Kohorte.

Beide werden publiziert, aber **niemals in eine gemeinsame `label`-Spalte
gelegt**. Es sind verschiedene Achsen; eine gemeinsame Spalte würde Nachnutzer
einladen, sie zusammenzuwerfen.

### 2.2 ID-Namensräume

Die Personen-IDs der Kohorten kollidieren (AirPods-`P17` ist ein anderer Mensch
als ML4SCS-`P17`). Alle IDs werden deshalb kohorten-präfigiert:

- `recording_id`: `ML4SCS-S008`, `ETH-EGE-T6`, `ETH-SL-E1`, `AIRPODS-P17`
- `participant_id`: `ML4SCS-P07`, `ETH-T6`, `AIRPODS-P17`

## 3. Getroffene Entscheidungen

| # | Entscheidung | Begründung |
|---|---|---|
| D1 | Consent deckt öffentliche Publikation ab; CC BY 4.0 | vom Nutzer bestätigt |
| D2 | **Nur Rohdaten.** Kein `derived/`, keine Windows, keine gemergten Labels | Derivate sind Methode; das Package erzeugt sie reproduzierbar |
| D3 | **Parquet only** (kein csv.gz-Mirror) | Nutzerentscheidung; Volumen unkritisch |
| D4 | Pen-`x/y` wird voll publiziert; Redaktion als konfigurierbare Pipeline-Stufe, Default aus | Nutzerentscheidung, mit offener Option auf ein späteres v2 |
| D5 | δ wird **nicht** eingebacken; rohe Geräteuhren + δ/σ als Metadatum | S094 und S062/P33 waren monatelang unentdeckte δ-Artefakte |
| D6 | Signal-**Semantik steht im Spaltennamen**, nicht in einem Flag | ein Flag kann man überlesen, einen Spaltennamen nicht |
| D7 | Roh-Accel der SensorLogger-Sessions bekommt eine **eigene Tabelle** | verschiedene Rate + Zeitachse; kein Stream darf verloren gehen |
| D8 | Einheiten werden harmonisiert, **Semantik nicht konvertiert** | siehe §5 |
| D9 | Qualitätsmetriken sind Manifest-**Spalten**, kein Einschlussfilter | ein σ-Gate ist ein ML-Kriterium, kein Datenqualitätskriterium |

Offen und außerhalb dieses Designs zu klären: Archiv (Zenodo vs. Mendeley
Data), Ethikkommissions-Zulassung inkl. Protokollnummer (vom DIB-Template im
Wortlaut verlangt), Rechte und Co-Autorschaft an den ETH- und AirPods-Daten.
Das Design bleibt archiv-agnostisch.

## 4. Bundle-Struktur

Flach, eine Datei pro Recording pro Modalität. Data in Brief verlangt, dass
**jede Datei im Artikel einzeln beschrieben wird** — ein Hive-Partitionsbaum
würde die Data Description zur Baum-Exegese machen. Kohortenzugehörigkeit ist
eine Manifest-Spalte, kein Pfadsegment.

```
focuswatch-dataset-v1.0/
  README.md                        Einstieg + Rezept-Kapitel
  data_dictionary.md               jede Spalte: Größe, Einheit, Frame, Semantik
  datapackage.json                 Frictionless-Deskriptor (generiert)
  validation_report.json           Messwerte aller Physik-Checks (generiert)
  sessions.parquet                 DAS MANIFEST — einzige Wahrheit für Flags
  sessions.csv                     dasselbe lesbar (klein, für Reviewer)
  channels.parquet                 pro (recording, modality, column): Einheit + Semantik
  watch/{recording_id}.parquet             42
  watch_rawaccel/{recording_id}.parquet     6
  headimu/{recording_id}.parquet           34
  pen/{recording_id}.parquet               38
  markers/{recording_id}.parquet           42
  attention/{recording_id}.parquet         25
```

`validation_report.json` wird bewusst mitpubliziert: damit wird die
Qualitätsprüfung von "wir haben einmal richtig gelegen" zu reviewbarer Evidenz,
auf die der DIB-Artikel verweisen kann.

## 5. Der Einheiten- und Semantik-Vertrag

Es gibt zwei Sorten von Vereinheitlichung, und sie sind unterschiedlich
zulässig. Das Kriterium ist **nicht** Umkehrbarkeit (`x / 9.80665` ist in
float64 nicht bit-umkehrbar), sondern:

> Eine Transformation ist zulässig, wenn sie **parameterfrei, per-Spalte affin,
> mit publizierter Konstante und ohne Per-Sample-Zustand** ist.

- **Einheiten werden harmonisiert.** Ein Skalar pro Spalte, die Konstante steht
  im Manifest (`unit_conversion_factor`). Zulässig.
- **Semantik wird nicht konvertiert.** `total → user` erfordert eine
  Orientierungsschätzung pro Sample, also ein Modell. Unzulässig unter D2.
  Stattdessen: die Semantik steht im Spaltennamen, und das Package bietet
  `to_user_acceleration()` für Nachnutzer, die konvertieren wollen.

Die puristische Alternative "gar nichts anfassen, nur deklarieren" wird
verworfen: die Messung (§5.1) zeigt, dass die Rohexporte flag-abhängige und
innerhalb einer Datei gemischte Einheiten tragen. "Nur deklarieren" exportiert
dieses Chaos an jeden Nachnutzer.

**Kanonische Einheiten:** Beschleunigung und Gravity in **g**, Winkel­geschwindigkeit
in **rad/s**, Quaternionen dimensionslos in **xyzw**, Zeit als **int64 Unix-Nanosekunden**
(für Millisekunden-Quellen exakt ×10⁶, für SensorLogger nativ). Die Uhren-*Domäne*
bleibt nativ und wird per `time_domain` deklariert — das ist D5.

### 5.0 Zwei Alignment-Regime

Der Korpus trägt zwei strukturell verschiedene Zeit-Situationen, und ein
Nachnutzer muss das wissen, bevor er Kohorten poolt:

| Regime | Kohorten | Bedeutung |
|---|---|---|
| `shared_clock` | ETH Ege, ETH SensorLogger | Backend stempelt **alle** Modalitäten auf dieselbe Unix-ms-Wall-Clock. Gemessen T6: IMU 1780577357025–1780578155747, Pen 1780577371079–1780578103584, Head deckungsgleich; Überlapp 732,5 s bzw. 1621,0 s bei voller Deckung. **Kein δ nötig.** |
| `estimated_delta` | ML4SCS | Die Watch-Capture-Uhr `ts` ist eine andere als die Server-Uhr, auf der der Pen gestempelt wird. δ wird per Stroke-Varianz-Minimierung **geschätzt** und als `pen_delta_s`/`pen_delta_sigma` deklariert, nicht angewandt (D5). |

Das Feld heißt `time_alignment`. Der Unterschied ist keine Randnotiz: bei
ML4SCS ist die Pen↔Watch-Zuordnung eine Schätzung mit Konfidenz, bei den
ETH-Sessions eine Konstruktion.

`pen_session_sync` ist **kein** Uhren-Anker — sein Payload
(`pen_connected_t_ms`, `session_start_t_ms`, `pen_minus_session_ms`) hält
lediglich fest, wann der Pen relativ zum Sessionstart verbunden wurde; beide
Werte liegen bereits auf der Wall-Clock. Der eigentliche Zweit-Uhren-Träger ist
unauffälliger: **jedes** ETH-Pen-Event führt in `payload.timestamp` zusätzlich
die Pen-Geräteuhr (Beispiel E2: `t_ms` 1780853816507 gegen `payload.timestamp`
1716121921598, ≈ 749 Tage Versatz). Diese bleibt als Metadatum erhalten, analog
zu `pen_logger.py`s Trennung von `local_ts_ms` und `timestamp`.

### 5.1 Gemessene Ausgangslage

Alle Werte sind Vektornormen über echte Dateien, nicht Annahmen:

| Quelle | ‖accel‖ median | Semantik | ‖gravity‖ | Gyro-Spalte |
|---|---:|---|---:|---|
| ML4SCS 100 Hz (S096, n=151.880) | 0,0393 | user | 1,0000 (g) | `rx/ry/rz` |
| ML4SCS 50 Hz (S008, n=46.114) | 0,0377 | user | *fehlt* | `rx/ry/rz` |
| ETH Ege wrist (T6, n=80.002) | **0,9946** | **total** | *fehlt* | heißt **`gx/gy/gz`** |
| ETH Ege head (T6, n=38.406) | 1,0003 | total | *fehlt* | *fehlt* |
| SensorLogger WristMotion (E2, n=101.898) | 0,0371 | user | 1,0000 (g) | `rotationRate*` |
| SensorLogger RawAccel (E2, n=101.967) | 0,9969 | total/raw | — | — |
| SensorLogger Headphone (E2, n=50.520) | 0,0214 | user | **9,8067 (m/s²)** | `rotationRate*` |
| AirPods (P1, n=26.126) | 0,0156 | user | 1,0000 (g) | `rotation_rate_*` |

Drei Fallen, von denen **keine an Spaltennamen erkennbar ist**:

1. **Ege rechnet die Schwerkraft mit ein** (‖accel‖ ≈ 0,99 statt ≈ 0,04) **und**
   nennt das Gyroskop `gx/gy/gz` — dieselbe Buchstabenfolge, die im ML4SCS-Schema
   die Gravity bezeichnet.
2. **SensorLogger ist in sich inkonsistent:** WristMotion-Gravity in g,
   Headphone-Gravity in m/s², dieselbe Datei-Familie derselben Aufnahme. Das ist
   [dokumentiertes App-Verhalten](https://github.com/tszheichoi/awesome-sensor-logger/blob/main/UNITS.md):
   Headphone-Gravity wird *immer* in m/s² geschrieben, Acceleration dagegen
   abhängig vom `standardisation`-Flag in `Metadata.csv`.
3. **Semantik ist eine Kanal-, nicht eine Session-Eigenschaft.** E2 trägt
   gleichzeitig user-Accel (WristMotion), total-Accel (RawAccel) und user-Accel
   *mit* m/s²-Gravity (Headphone).

### 5.2 Verifizierte Konventionen

| Frage | Test | Ergebnis |
|---|---|---|
| Quaternion-Reihenfolge | Gravity aus Quaternion rekonstruieren, Winkel gegen gemessene Gravity | **Alle Quellen `xyzw`**, Medianfehler 0,000° (Ege 0,18°). `wxyz`-Fehlinterpretation ergäbe 31–125°. |
| Gravity-Vorzeichen | dito, gegen `[0,0,−1]` vs. `[0,0,+1]` | **`[0,0,−1]`** überall; `+1` ergibt exakt 180° |
| Ege: raw oder fused? | ‖accel‖ und Residuum gegen R(q)·[0,0,−1] in Ruhefenstern (Gyro-Norm < 0,05 rad/s) | **raw uncalibrated.** T6: 0,99541 ± 0,00429; T7: 0,99315 ± 0,00404. Eine Rekombination `user + gravity` müsste exakt 1,000 ergeben — der Bias persistiert und ist je Session verschieden. |

`scipy.spatial.transform.Rotation.from_quat` erwartet **xyzw** (scalar-last).
Da die Quellen ihre Spalten unterschiedlich benennen und ordnen (`qw,qx,qy,qz`
bei Ege, `quaternionW/X/Y/Z` bei SensorLogger, `qx,qy,qz,qw` bei ML4SCS), ist
das die wahrscheinlichste stille Verwechslung des ganzen Vorhabens — sie
verändert keine Norm und wirft keine Exception.

## 6. Kanonisches Schema

### `watch/` und `headimu/`

| Spalte | Typ | Einheit | Bedeutung |
|---|---|---|---|
| `t_ns` | int64 | Unix-ns | Capture-Zeit in der nativen Uhren-Domäne |
| `accel_user_{x,y,z}` | float64 | g | Beschleunigung **ohne** Schwerkraft |
| `accel_total_{x,y,z}` | float64 | g | Beschleunigung **mit** Schwerkraft |
| `gyro_{x,y,z}` | float64 | rad/s | Winkelgeschwindigkeit |
| `gravity_{x,y,z}` | float64 | g | Schwerkraftrichtung, Einheitsvektor |
| `quat_{x,y,z,w}` | float64 | — | Attitude, xyzw, normiert |

Eine Datei enthält **entweder** `accel_user_*` **oder** `accel_total_*`, nie
beide unter demselben Namen. Nicht vorhandene Kanäle erzeugen keine Spalte
(kein NULL-Padding). Welche Spalten existieren, steht im Manifest und in
`channels.parquet`.

Quellspezifische Provenienz-Spalten (`local_ts_ms`, `sequence`,
`server_received_ms`, …) werden mit Präfix `src_` erhalten. Sie kosten
komprimiert fast nichts und dokumentieren die Capture-Kette.

### `watch_rawaccel/`

`t_ns`, `accel_total_{x,y,z}` — eigene Tabelle, weil Rate und Zeitachse von der
fusionierten `watch/`-Tabelle abweichen.

### `pen/`

`t_ns`, `dot_type`, `x`, `y`, `pressure`, `tilt_x`, `tilt_y`.

Die Moleskine-Framing-Rows (`x = y = −1`) bleiben als Sentinel erhalten und
werden im Data Dictionary hervorgehoben — sie zu Null zu machen wäre eine
Semantikänderung, sie zu übersehen ruiniert jede `mean(x)`-Statistik.

Pen-Koordinaten und Druck werden **nicht** harmonisiert: Moleskine liefert
Ncode-Rastereinheiten, Ege liefert `x/y/force` in unbekannter Skala. Hier fehlt
die publizierbare Konstante, also fällt es per §5-Kriterium auf die
Deklarations-Seite (`pen_xy_unit`, `pen_pressure_scale` im Manifest).

Die Redaktions-Policy aus D4 gilt für **alle** Pen-Quellen, nicht nur für
Moleskine. Die ETH-Pen-Events tragen ebenfalls Koordinaten (`x`, `y`, `force`,
`tilt`), aus denen freies Schreiben genauso rekonstruierbar ist. `redact.py`
arbeitet daher auf der kanonischen `pen/`-Tabelle, nach dem Adapter — nicht
quellenspezifisch davor.

### `markers/`

`t_ns`, `event`, `task_id`, `task_name`, `task_index`, `task_category`,
`protocol_id`.

### `attention/`

`t_start_ns`, `t_end_ns`, `label` — die **Intervalle**, nicht die
Per-Sample-Expansion. Gemessen: Gerrits `label`-Spalte ist exakt das
`.txt`-Intervallprotokoll, auf 26.000 Samples ausgerollt. Das Intervall ist die
rohe Annotation, die Expansion ist ein Derivat (fällt unter D2). Nebeneffekt:
sichtbar wird, dass die Annotation minutengenau ist und nicht sample-genau.

### `channels.parquet`

Eine Zeile pro `(recording_id, modality, column)`: `quantity`, `unit`,
`semantics`, `frame`, `sample_rate_hz`, `unit_conversion_factor`. Das ist das
BIDS-`channels.tsv`-Muster, flach gehalten. ~1.000 Zeilen.

`unit` benutzt ein geschlossenes, einheitlich angewendetes Vokabular über
alle nicht-physikalischen Spalten hinweg (physikalische Größen tragen ihre
SI-nahe Einheit aus `schema.UNITS`, z. B. `g`, `rad/s`):

| Wert | Bedeutung |
|---|---|
| `category` | einer aus einer festen, aufgezählten Menge von Strings |
| `ordinal` | Position eines Elements in einer Sequenz, keine Magnitude (z. B. `task_index`) |
| `device_native` | Rohwert in der Skala der Quelle selbst; für Pen-`x`/`y`/`pressure` steht die tatsächliche Einheit zusätzlich pro Recording im Manifest (`pen_xy_unit`/`pen_pressure_scale` — Moleskine-Ncode-Raster und ETH-Web-App-Pixel/-Force sind unterschiedliche Skalen unter demselben Spaltennamen) |
| `source_native` | `src_`-Provenienz-Spalte: der unveränderte Wert des Quellgeräts, nur zur Nachvollziehbarkeit erhalten, nie neu interpretiert |
| `n/a` | keine physikalische oder Quell-Einheit anwendbar (z. B. eine kategoriale ID) |

`device_native` und `source_native` klingen ähnlich, meinen aber
Verschiedenes: Ersteres ist eine echte Quell-Messskala (mit dokumentierter
Referenz im Manifest), Letzteres ist reine Audit-Provenienz ohne eigenen
fachlichen Wert.

## 7. Manifest (`sessions.parquet`)

Alle Flags leben ausschließlich hier. Die Parquet-Dateien tragen in ihren
Key-Value-Metadaten nur `schema_version` und `recording_id`, generiert aus dem
Manifest — es gibt keinen zweiten, unabhängig gepflegten Ort.

| Gruppe | Spalten |
|---|---|
| Identität | `recording_id`, `participant_id`, `cohort`, `pipeline` |
| Modalitäten | `has_watch`, `has_watch_rawaccel`, `has_headimu`, `has_pen`, `has_markers`, `has_attention` |
| Watch-Capabilities | `watch_hz_nominal`, `watch_hz_measured`, `has_gravity`, `has_quaternion`, `accel_semantics`, `accel_calibration`, `accel_still_bias`, `gravity_source` |
| Head-Capabilities | `head_hz_nominal`, `head_hz_measured`, `has_head_gravity`, `has_head_quaternion`, `has_head_gyro` |
| | *Die Head-Rate schwankt stark und darf nicht deklariert, sondern muss gemessen werden: Ege T6 125 Hz (Median-Δt 8 ms), T7 62,5 Hz (16 ms), AirPods ~25 Hz.* |
| | *`has_head_gyro` ist ein DATEN-Fakt, keine strukturelle Eigenschaft von `headimu`: SensorLogger und AirPods führen ein Kopf-Gyroskop (bei AirPods das einzige Bewegungssignal überhaupt), Ege nicht.* |
| Zeit | `time_domain`, `time_alignment`, `t_start_ns`, `t_end_ns`, `duration_s` |
| Protokoll | `protocol_id`, `study_mode`, `subject_index`, `n_writing_tasks`, `n_idle_tasks` |
| | *`n_writing_tasks`/`n_idle_tasks` sind ML4SCS-skopiert — nur `adapters/ml4scs.py`s Markers tragen eine echte `task_category`-Taxonomie (`writing`/`idle`). Ege und SensorLogger schreiben ein reines Session-Event-Log ohne Task-Kategorie, dort steht `None` (nicht `0` — ein `0` würde fälschlich „Protokoll ohne Schreib-Tasks" statt „Taxonomie nicht anwendbar" bedeuten).* |
| Träger-Kontext | `watch_wrist_side` ∈ {`left`, `right`, `unknown`} |
| Pen | `pen_xy_unit`, `pen_pressure_scale`, `pen_delta_s`, `pen_delta_sigma`, `delta_applied` (immer `false`), `alignment_note` |
| Qualität | `n_samples_watch`, `n_samples_pen`, `n_samples_head`, `issue_codes` |
| Provenienz | `source_pipeline`, `schema_version`, `redaction_policy`, `build_git_sha` |

`accel_semantics` und `accel_calibration` stehen hier **zusätzlich** zum
Spaltennamen, als durchsuchbare Facette — nicht als deren Ersatz. Bei Recordings
mit mehreren Accel-Streams (SensorLogger) beschreibt das Manifest-Feld den
`watch/`-Stream; die vollständige Wahrheit steht in `channels.parquet`.

`watch_wrist_side` wird für die meisten Recordings `unknown` sein. Das Feld
existiert trotzdem: die Handgelenk-Seite spiegelt Achsen relativ zur
Schreibbewegung und ist nachträglich kaum rekonstruierbar — ohne das Feld fragt
niemand danach.

### 7.1 Die Query-Oberfläche

```python
from focuswatch_dataset import load_manifest, load_recording

root = "focuswatch-dataset-v1.0"
m = load_manifest(root)

m.query("has_watch and has_pen and watch_hz_nominal == 100 and has_gravity")
m.query("has_watch and accel_semantics == 'user'")
m.query("has_headimu")
m.query("has_watch_rawaccel")
m.query("protocol_id == 'ml4scs_v2'")

df = load_recording(root, "ML4SCS-S096", modality="watch")
```

`load_recording` takes `root` explicitly rather than remembering it from a
prior `load_manifest` call. The alternative - a stateful "dataset" object
returned by `load_manifest` that closes over its root - was rejected: every
other function in this API (`load_channels`, `load_recordings`, `by_flags`)
already takes its root/manifest explicitly, a reuser inspecting two mirrors
of the bundle side by side needs two roots live at once, and the explicit
form is what every test in this package's own suite already exercises with a
`tmp_path` root - no hidden global to reset between fixtures.

`load_recordings(subset)` prüft beim Laden mehrerer Recordings auf
Semantik-Homogenität und wirft, wenn `user`- und `total`-Accel gemischt würden.

## 8. Adapter-Spezifikation

| Quelle | Zentrale Transformationen |
|---|---|
| **ML4SCS** | `ax/ay/az` → `accel_user_*`; `rx/ry/rz` → `gyro_*`; `gx/gy/gz` → `gravity_*`; `qx..qw` → `quat_*`; `ts` (ms) → `t_ns`; Marker unverändert |
| **ETH Ege** | `ax/ay/az` → `accel_**total**_*`; **`gx/gy/gz` → `gyro_*`** (Namenskollisions-Fix, nur mit Physik-Gate); `qw,qx,qy,qz` → `quat_{x,y,z,w}` (**Reihenfolge!**); **`t_ms` → `t_ns`** (verifiziert Unix-ms; `t_session_ms` ist session-relativ, 0…800.106, und wird als `src_t_session_ms` erhalten); `pen_events.type` → `dot_type`-Vokabular; `events.csv` → `markers/` |
| **ETH SensorLogger** | Einheit aus `Metadata.csv:standardisation` **ableiten, nie hartkodieren**; Headphone-Gravity ÷ 9,80665; `quaternionW/X/Y/Z` → `quat_{x,y,z,w}`; `WatchAccelerometerUncalibrated` → eigene `watch_rawaccel/`-Tabelle; `time` (ns) → `t_ns`; **Pen und Marker kommen aus dem Session-JSON**, nicht aus `Annotation.csv` (die ist leer) — `events[].t_ms` ist bereits wall-clock-aligned, `payload.timestamp` ist die unaligned Pen-Geräteuhr und bleibt Metadatum |
| **AirPods** | `user_acceleration_*_g` → `accel_user_*`; `gravity_*_g` → `gravity_*`; `rotation_rate_*_rad_s` → `gyro_*`; `quaternion_*` → `quat_*`; Intervall-`.txt` → `attention/`; Per-Sample-`label` **verwerfen** (Derivat) |

Gravity ist bei Ege nicht gemessen, aber aus dem Quaternion rekonstruierbar.
`foreign_adapter/gravity_quat.py` tut das bereits (validiert gegen
SensorLogger-Ground-Truth, mittlerer Absolutfehler ~2e-8) und wandert ins
Package. **Die rekonstruierte Gravity wird nicht als Datenspalte publiziert**
(sie wäre ein Derivat), sondern nur intern für die Validierung genutzt;
`gravity_source` dokumentiert `none` für Ege.

### 8.1 Pen-Vokabular — empirisch geklärt

Beide ETH-Pipelines stammen von derselben Web-App, speichern die Events aber
verschieden (Ege in einer Supabase-Tabelle, SensorLogger im Session-JSON).
Häufigkeitsanalyse T6: `pen_dot` 2.516, `pen_down` 236, `pen_up` 236,
`pen_paper_info` 228. Die dominante Klasse mit Abtastintervall ist `pen_dot`;
dieselbe App schreibt im SensorLogger-JSON dafür `pen_move`. `pen_paper_info`
tritt etwa einmal pro Stroke auf und trägt keine Position.

| Quelle | Event | → kanonisch |
|---|---|---|
| Moleskine | `PEN_DOWN` / `PEN_MOVE` / `PEN_UP` / `PEN_HOVER` | unverändert |
| Ege (Supabase) | `pen_down` / `pen_dot` / `pen_up` | `PEN_DOWN` / `PEN_MOVE` / `PEN_UP` |
| Ege (Supabase) | `pen_paper_info` | Framing-Event, Position `NaN`, `dot_type` erhalten |
| SensorLogger (JSON) | `pen_down` / `pen_move` / `pen_up` | `PEN_DOWN` / `PEN_MOVE` / `PEN_UP` |
| beide | `pen_session_sync` | nicht in `pen/`; als Marker-Event erhalten |

`PEN_HOVER` existiert nur in der Moleskine-Quelle. Es wird nicht synthetisiert.

### 8.2 Vor der Implementierung noch zu verifizieren

- **Uhren-Domäne der Attention-Annotation.** Die Beobachter-Intervalle laufen auf
  einer nicht deklarierten Uhr. Ohne `time_domain`-Deklaration plus Überlapp-Test
  gegen die AirPods-Zeitachse ist das dieselbe Fallenklasse wie Pen↔Watch, nur
  ohne δ-Verfahren als Netz.
- **Pen-Koordinatenskalen.** Moleskine-Ncode-Raster gegen Ege-`x/y/force`:
  Wertebereiche und Seitenverhältnis gegen das physische Papierformat prüfen,
  und je Quelle einen Stroke-Plot rendern. Sind Buchstaben lesbar, nicht
  gespiegelt oder gestaucht? Das ist der billigste und schärfste Test, und sein
  Ergebnis gehört ins Data Dictionary, nicht in eine Annahme.
- **`accel_still_bias` je Ege-Session** als Diagnosewert ins Manifest
  (gemessen: T6 −0,46 %, T7 −0,69 %) — dokumentieren, nicht korrigieren.

## 9. Validator

Der Validator prüft **Physik, nicht Spaltennamen**. Keine der drei Fallen aus
§5.1 ist an Namen erkennbar; alle drei sind an Vektornormen erkennbar.

### 9.1 Build-Gate (echte Daten, bricht hart)

| Check | Toleranz |
|---|---|
| `‖gravity‖` nach Harmonisierung | Median ∈ [0,99; 1,01], IQR < 0,01 |
| **Verbotene Bänder** für ‖accel‖-Median | muss in [0; 0,2] (user) **oder** [0,9; 1,1] (total); (0,2; 0,9) und [5; 15] brechen den Build |
| `‖gyro‖` | Median ∈ [0,005; 2] rad/s, p95 < 20 (deg/s-Daten reißen das) |
| `‖quat‖` | ∈ [0,999; 1,001] |
| Winkel(gemessene Gravity, R(quat)·[0,0,−1]) | Median < 2° (real gemessen: 0,000°) |
| dito ohne Gravity-Spalte, gegen Ruhefenster-Accel | Median < 5° |
| Flachlage-Fenster (\|gx\|,\|gy\| < 0,1) | `gz` ≈ **−1**, nicht +1 |
| `t_ns` nach **stabilem** Sort | streng monoton |
| Zeit-Magnitudenklasse | passt zu deklarierter Einheit (~1,7e9 s / e12 ms / e15 µs / e18 ns; < 1e8 = session-relativ) |
| Modalitäts-Zeitbereiche je Recording | überlappen |
| Spill-Guard | kein Sample < `t_start_ns` − 60 s |
| `hz_measured` vs. `hz_nominal` | Abweichung < 20 % |
| Manifest-Konsistenz | `has_X` ⟺ Datei existiert; `n_samples` stimmt; jede `channels`-Zeile hat Einheit und Semantik ≠ leer |
| Determinismus | zweimal bauen → identische Checksummen |

Die verbotenen Bänder sind der eigentliche Anti-Verwechslungs-Mechanismus: ein
vergessenes ÷9,80665 landet bei ~9,8, eine user/total-Verwechslung bei ~0,5 im
Mittel — beides im verbotenen Bereich.

### 9.2 Repo-CI (nur synthetische Fixtures)

Rohdaten betreten den Git-Baum nie. Die CI testet die **Logik**:

- Einheiten-Konversion inkl. beider `standardisation`-Zweige
- Quaternion-Reihenfolgen-Detektor: synthetische Rotation, beide Reihenfolgen
  eingespeist → richtige wird gewählt
- **Negativ-Tests der verbotenen Bänder**: synthetische Dateien mit ‖accel‖-Median
  0,5 bzw. 9,8 → der Validator **muss** failen. Ein Validator, dessen Failen nie
  getestet wurde, ist Dekoration.
- Adapter-Mapping (Ege `gx`→`gyro_x`), Sentinel-Erhalt, Zeit-Klassifikator,
  Redaktionsstufe
- Roundtrip write→read bit-identisch (float64)
- Golden Files pro Quellformat auf synthetischen Mini-Recordings
- Property-based (hypothesis) auf der Zeitachsen-Normalisierung: Ties,
  Duplikate, Rückwärtssprünge → Monotonie und Zeilenerhalt

Die Schwellwerte selbst leben als Konstanten in `schema.py` und werden von
beiden Ebenen importiert, damit CI und Build-Gate nicht auseinanderdriften.

## 10. Package- und Repo-Architektur

Neues Repo `CH-GE-Focus-Watch/focuswatch-dataset`, `src`-Layout,
pip-installierbar, **strikt code-only**.

```
src/focuswatch_dataset/
  schema.py        kanonische Spalten, dtypes, Einheiten, Schwellwerte
  adapters/
    base.py        Protocol: discover(root) -> [RecordingRef]; load(ref) -> Bundle
    ml4scs.py
    ege.py
    sensorlogger.py
    airpods.py
  validate.py      strukturell + physikalisch, beide Ebenen
  manifest.py      sessions.parquet + channels.parquet, Konsistenzprüfung
  write.py         deterministischer Parquet-Writer
  redact.py        Pen-x/y-Policy (Default aus)
  select.py        Flag-Query
  load.py          Consumer-API inkl. Homogenitätsprüfung
  cli.py           fw build | validate | report
```

**Parquet-Writer-Konfiguration:** zstd, `BYTE_STREAM_SPLIT` für Sensor-Floats,
`DELTA_BINARY_PACKED` für `t_ns`, float64 durchgehend. Gemessen auf S080
(101 MB CSV): 15,7 MB, Faktor 6,4. Hochgerechnet ~310 MB für den ganzen Korpus.
float32 spart weitere 40 %, roundtrippt aber nicht bit-identisch — für eine
Datenpublikation nicht akzeptabel.

**Datenpolitik.** Der Build liest von einem externen Pfad; das Output-Verzeichnis
liegt außerhalb des Repos. Tests laufen ausschließlich auf synthetischen
Fixtures. Ein Pre-Commit-Guard blockt Dateien > 1 MB und bekannte Datenpfad-Muster.
Grund: am 08.08.2026 sind bereits einmal Probandendaten in beide Remotes
gelangt und mussten mit `git filter-repo` entfernt werden; `origin` ist public.

**Lizenz-Split:** Code Apache-2.0 im GitHub-Repo, Daten CC BY 4.0 im Daten-Repo.
`CITATION.cff` plus Zenodo-GitHub-Integration, damit das Code-Release einen
eigenen DOI bekommt und im Artikel neben dem Daten-DOI zitierbar ist.

## 11. Risiko-Ranking

Nach Wahrscheinlichkeit × Stille sortiert — "still" heißt: produziert falsche
Ergebnisse, ohne dass etwas crasht.

1. **Quaternion-Reihenfolge.** Die Norm ist so oder so 1,0; drei Quellen, drei
   Benennungskonventionen, scipy-Default `xyzw`. Empirisch für alle Quellen
   geklärt (§5.2) und als Assertion fixiert.
2. **Einheiten-/Semantik-Mix unter identischem Spaltennamen.** Verstärker:
   **per-Session-Z-Score maskiert Skalenfehler vollständig.** Wer ML4SCS-`ax`
   (0,04) mit Ege-`ax` (0,99) poolt und z-scored, bekommt keine Warnung — nur
   eine etwas schlechtere Metrik ohne erkennbare Ursache. Adressiert durch D6
   (Semantik im Spaltennamen) und die verbotenen Bänder.
3. **`standardisation`-Flag nicht gebrancht.** Heute korrekt, bricht still bei
   der nächsten gespendeten SensorLogger-Aufnahme mit anderem Flag.
   Zeitbomben-Risiko.
4. **Handgelenk-Seite unerfasst.** Degradiert Cross-Subject-Transfer dauerhaft,
   nachträglich kaum rekonstruierbar, wirft nie eine Exception.
5. **Pen-Skalen und Sentinels.** Falsche Stroke-Geometrie, verschobene
   Druckstatistik — sichtbar nur, wenn jemand rendert.
6. **Zeit-Epochen.** Meist laut (leere Joins); die stille Variante ist
   session-relativ als Epochenzeit gemappt und wird vom Überlapp-Test gefangen.

## 12. Was dieses Design bewusst nicht tut

- Kein Resampling auf eine gemeinsame Rate. Native Raten bleiben nativ.
- Keine gemergten Labels, keine Feature-Fenster, keine δ-korrigierten Zeitachsen.
- Keine Vereinheitlichung von Pen-Koordinaten über die zwei Pen-Quellen.
- Keine Motion-BIDS-Konformität. Das Sidecar-Muster und die Channels-Beschreibung
  werden übernommen, die Verzeichnisgrammatik nicht — BIDS ist für
  Neuro-Workflows gebaut und hat für die Pen-Modalität keinen Platz.
