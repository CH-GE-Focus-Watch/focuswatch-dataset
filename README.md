# FocusWatch Dataset

Wrist motion, head motion, and digital-pen events recorded with consumer
wearables during desk-based writing, non-writing, and attention tasks. This
repository holds the software that built the dataset and that loads, filters,
and labels it.

- **Article:** *A Multimodal Wrist and Head Motion Dataset for Handwriting and
  Focus Detection with Consumer Wearables*, Data in Brief. DOI: TODO.
- **Dataset:** Mendeley Data, CC BY-NC 4.0. DOI: TODO.
- **Software:** this repository, Apache-2.0.

The recordings come from four cohorts with different devices, sampling rates,
and clocks. They were converted into one Apache Parquet format with
harmonized units. Each recording keeps its native sampling rate, and a
manifest declares which modalities, channels, devices, and clocks it has.

## Dataset scope

The dataset contains 88 recordings with a total duration of 33.7 h.

| Cohort | `cohort` | Participants | Recordings | W | H | P | M | A |
| --- | --- | ---: | ---: | :-: | :-: | :-: | :-: | :-: |
| Scripted writing | `ML4SCS` | 33 | 33 | ● | – | ● | ● | – |
| Multimodal | `ZHAW` | TODO | 9 | ● | ● | ● | ● | – |
| Attention | `AIRPODS` | 38 | 38 | – | ● | – | – | ● |
| Unscripted writing | `MOLESKINE` | 8 | 8 | ● | – | ● | – | – |

W: wrist motion (Apple Watch). H: head motion (AirPods Pro 3). P: pen events,
which also provide the handwriting ground truth. M: protocol markers. A:
attention intervals.

Participant identifiers are numbered within each cohort, so the same number in
two cohorts says nothing about the person. No mapping across cohorts exists.

Each modality of a recording is one Parquet file:

```text
sessions.parquet / .csv   manifest, one row per recording
channels.parquet          unit, sampling rate, and clock of every column
data_dictionary.md        definitions of all units and terms
watch/                    W   {recording_id}.parquet
watch_rawaccel/           uncalibrated wrist accelerometer, 6 ZHAW recordings
headimu/                  H
pen/                      P
markers/                  M
attention/                A
```

`recording_id` joins a recording's files across all directories.

## How recordings differ

The cohorts were recorded with different apps and settings. The bundle keeps
these differences and declares them per recording in `sessions.parquet`.
Filter on these columns before you combine recordings.

| | Manifest column | `ML4SCS` (33) | `ZHAW` (9) | `AIRPODS` (38) | `MOLESKINE` (8) |
| --- | --- | --- | --- | --- | --- |
| Acceleration | `accel_semantics` | `user` | `user` 7, `total` 2 | `user` | `total` |
| Wrist rate | `watch_hz_measured` | 50 Hz 9, 100 Hz 24 | 100 Hz | – | 50 Hz |
| Wrist gyroscope | `has_watch_gyro` | 33 | 9 | – | 0 |
| Wrist gravity | `has_gravity` | 24 | 7 | – | 0 |
| Wrist quaternion | `has_quaternion` | 19 | 9 | – | 0 |
| Head rate | `head_hz_measured` | – | 36 or 50 Hz | 27–29 Hz | – |
| Head gyroscope, gravity, quaternion | `has_head_*` | – | 7, 7, 9 | 38, 38, 38 | – |
| Clock alignment | `time_alignment` | `estimated_delta` | `shared_clock` | `shared_clock` | `overlap_only` |
| Task protocol | `protocol_id` | `ml4scs_v1` 13, `ml4scs_v2` 20 | `zhaw_web` | `airpods_p` 28, `airpods_t` 10 | – |

Numbers in a cell are recordings.

**Acceleration.** `user` acceleration has gravity removed, `total` includes it.
Do not mix the two. `load_recordings` refuses to, and `fw preprocess` refuses
unless told otherwise. `to_user_acceleration` derives `user` from `total` when
the recording has a quaternion, which holds for the two `ZHAW` recordings. The
`MOLESKINE` recordings have no quaternion and stay `total`.

**Sampling rate.** Every stream keeps its native rate. To combine 50 Hz and
100 Hz recordings, load them with `target_hz=50`, which downsamples with an
anti-aliasing filter. The `_hz_measured` columns come from the median sample
interval. AirPods deliver head motion on an uneven grid: in the `AIRPODS`
cohort the median interval is 34 ms, but a quarter of all intervals are 56 ms
or longer.

**Clocks.** `time_alignment` states how the pen relates to the motion stream:

- `shared_clock`: all modalities use one clock and can be joined directly.
- `estimated_delta`: pen and wrist use different clocks. The offset was
  estimated but is not applied, and `pen_delta_s` is empty on purpose.
- `overlap_only`: pen and wrist were paired only because their wall-clock
  ranges overlap. No offset was estimated, so sample-level joins are
  approximate.

Each recording's `alignment_note` explains its case.

## Devices

| Cohort | Wrist | Head | Pen | Recording software |
| --- | --- | --- | --- | --- |
| `ML4SCS` | Apple Watch Series 6 | – | Moleskine Smart Pen NWP-F130 | TODO |
| `ZHAW` | Apple Watch SE (2nd generation) | AirPods Pro 3 | Neo smartpen M1 NWP-F50 (5), LAMY safari all black ncode NWP-F80 (3), unknown (1) | custom watchOS app (2), Sensor Logger (7) |
| `AIRPODS` | – | AirPods Pro 3 | – | TODO |
| `MOLESKINE` | Apple Watch SE (2nd generation) | – | Moleskine Smart Pen NWP-F130 | Sensor Logger, Moleskine app |

`watch_model`, `head_model`, and `pen_model` in `sessions.parquet` name the
device of each recording. One `ZHAW` recording logged no pen name, so its pen
stays `unknown`.

Wrist and head motion come from Apple Core Motion. Head motion has no
magnetometer reference, so the yaw of the head orientation drifts. All pens are
NeoLAB Ncode pens and report the contact states `PEN_DOWN`, `PEN_MOVE`, and
`PEN_UP`. Pen pressure is device-native and not comparable across pen
models: `pen_pressure_scale` names the scale of each recording.

## Pen sampling rate

A pen reports dots only while it touches the paper, so it has no fixed rate,
and none of the manufacturers we checked states one. `pen_hz_measured` gives
the dots per second within strokes. `pen_hz_clock` names the clock it was
measured on: the pen's own clock (`src_timestamp`) where the source logs it,
otherwise `t_ns` if it stamps dots one by one.

| Pen model | Clock | Recordings | Median | Range |
| --- | --- | ---: | ---: | --- |
| Moleskine Smart Pen NWP-F130 (`ML4SCS`) | `src_timestamp` | 33 | 79.2 Hz | 68.2–83.5 Hz |
| Moleskine Smart Pen NWP-F130 (`MOLESKINE`) | `t_ns` | 8 | 39.0 Hz | 31.1–39.5 Hz |
| Neo smartpen M1 NWP-F50 | `src_timestamp` | 1 | 70.0 Hz | – |
| LAMY safari all black ncode NWP-F80 | `src_timestamp` | 1 | 45.1 Hz | – |
| LAMY safari all black ncode NWP-F80 | `t_ns` | 2 | 48.7 Hz | 46.9–50.5 Hz |
| unknown | `src_timestamp` | 1 | 69.1 Hz | – |

Four NWP-F50 recordings have no rate, because their `t_ns` arrives in
Bluetooth batches that share one timestamp. The `MOLESKINE` recordings show
about half the NWP-F130 rate of `ML4SCS`. TODO: the cause is not known yet.

## Use the bundle

Read the manifest first, choose recordings, then load only their samples.

```python
from focuswatch_dataset import by_flags, load_manifest, load_recording, load_recordings

root = "PATH/TO/DATASET"
manifest = load_manifest(root)

# Shorthand for equality filters
writing = by_flags(manifest, has_watch=True, has_pen=True, accel_semantics="user")
# Any pandas expression over the manifest columns
fast = manifest.query("cohort == 'ML4SCS' and watch_hz_measured > 90")

pen = load_recording(root, writing["recording_id"].iloc[0], modality="pen")
watch = load_recordings(root, writing["recording_id"].tolist(), modality="watch", target_hz=50)
```

`load_recordings` joins the tables of several recordings and adds a
`recording_id` column. It refuses selections that mix data that does not
belong together:

- Wrist acceleration with and without gravity. Convert first with
  `to_user_acceleration(df)`, which needs a quaternion.
- Motion streams whose rates differ by more than 10 %. Pass `target_hz` to
  downsample with an anti-aliasing filter, or `allow_mixed_rates=True` to keep
  the native rates and get a `sample_rate_hz` column.
- Pen tables with different coordinate or pressure scales.
  `allow_mixed_pen_units=True` keeps them and adds the scale columns.

The published files are never changed.

To filter without code, `fw explore PATH/TO/DATASET` opens a terminal
assistant. Choose a goal (writing detection, attention, or everything),
narrow by signals, rates, and timing, and export the selection to a new
directory. Each filter shows how many recordings it leaves, and `c` copies the
equivalent pandas query.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
git clone https://github.com/CH-GE-Focus-Watch/focuswatch-dataset.git
cd focuswatch-dataset
uv sync --extra tui
```

The `tui` extra is needed for `fw explore`. Use `uv sync --all-extras` to also
install the test dependencies.

## Derive training labels

The bundle publishes ground truth as recorded: pen contact events and
attention intervals. `fw preprocess` turns them into per-sample labels in a
new directory and records every parameter in `preprocess.json`.

```bash
uv run fw preprocess --dataset PATH/TO/DATASET --task writing \
  --gap-closing-s 3 --target-hz 50 --cohort ML4SCS --out labels-writing
uv run fw preprocess --dataset PATH/TO/DATASET --task attention \
  --out labels-attention
```

- `writing` adds `label_writing` to each watch table. A sample counts as
  writing when the nearest pen event within 40 ms is `PEN_DOWN` or
  `PEN_MOVE`. Pauses between contacts up to `--gap-closing-s` also count as
  writing. This option has no default, because it defines what "writing"
  means; the project uses 3 s.
- `attention` adds `label_attention` (`focused` or `distracted`) to each head
  motion table, from the interval that contains the sample.
- Pen timestamps are joined as published. `--pen-shift-s` shifts them.
  `overlap_only` recordings are skipped unless `--include-overlap-only` is
  given.
- `writing` refuses to mix `user` and `total` acceleration unless
  `--allow-mixed-accel` is given. `ZHAW` contains both, so the example above
  selects `ML4SCS` only.
- `--cohort` and `--recording` restrict the selection.

The two label columns are never merged: writing and attention are different
questions with different ground truth. The same steps are available in Python
through `focuswatch_dataset.preprocess.PreprocessConfig` and
`preprocess_recordings`.

## Build and validate a bundle

This step is only needed to rebuild the dataset from the raw sources, which
are not public. Pass every source explicitly:

```bash
uv run fw build \
  --source ml4scs=PATH/TO/ml4scs_partner_export \
  --source ege=PATH/TO/data_ege_pipeline \
  --source sensorlogger=PATH/TO/data_ege_sensorlogger_pipeline \
  --source airpods=PATH/TO/Headphone_6Tasks_38People \
  --source moleskine=PATH/TO/raw_moleskine \
  --out out/focuswatch-dataset \
  --redact all_xy \
  --time-origin recording_start

uv run fw validate --dataset out/focuswatch-dataset
uv run fw report --dataset out/focuswatch-dataset
```

`--redact` and `--time-origin` have no default, so a release cannot inherit a
privacy policy by accident. `all_xy` removes all pen coordinates.
`recording_start` makes every timestamp count from the start of its
recording, so the bundle does not reveal when someone was recorded. If any
check fails, the build writes nothing and the previous output stays in place.
`--no-strict` is for diagnostics only.

## Limitations

- Attention labels describe the instructed task, not a verified attentional
  state.
- Participants are healthy adults recruited by convenience sampling. Sex and
  gender were not recorded.
- Pen coordinates and recording dates are not published, so handwriting shape
  and time of day cannot be analyzed.
- Validation covers structure, timing, physical plausibility, and redaction.
  It does not establish consent, redistribution rights, or scientific
  validity.

## Release checklist

1. Build from a clean commit on `main`, so `build_git_sha` has no `-dirty`
   suffix, with `--redact all_xy --time-origin recording_start`.
2. Run `uv run pytest` and `uv run fw validate --dataset PATH`; both must
   pass without failures.
3. Confirm consent, redistribution rights, and the CC BY-NC 4.0 license for
   every cohort.
4. Record a SHA-256 checksum for every file of the archive.
5. Replace the TODO markers in this README and the DOI placeholder in
   `CITATION.cff`.
6. Confirm that no raw data is tracked:

   ```bash
   git ls-files | xargs uv run python scripts/check_no_data.py
   ```

## License and citation

The software is Apache-2.0, see [`LICENSE`](LICENSE). The dataset is CC BY-NC
4.0. When you use the data, cite the article and the dataset (DOIs: TODO).
`CITATION.cff` describes the software.
