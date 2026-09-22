# YOLO26 One-Shot Person Re-Identification (face-based)

Register a person from **one** photograph, then find that person in live camera
streams, video files, images and image directories.

Identity comes from the **face**, so it does not change when the person changes
clothes. Measured on the bundled demo, recolouring everything below the chin
leaves the match at **0.98**; the same image under whole-body appearance ReID
collapses from 0.96 to 0.43 and the identity is lost.

The system keeps four different problems apart, because conflating them is the
usual way a "ReID system" ends up identifying outfits:

| Question | Answered by |
| --- | --- |
| *Is there a person here?* | YOLO26 detection |
| *Is this the same moving blob as last frame?* | BoT-SORT tracking (a temporary, source-local id) |
| *Is this person's face visible, and how much of it?* | SCRFD face detection + 5-point alignment + visibility |
| *Which registered person is this — if any?* | Face embedding compared against the identity gallery |

**Enrolling from a passport photograph?** That is what the
[face identity engine](docs/identity.md) is for: calibrated thresholds per
visibility class, quality-weighted temporal evidence, and per-condition
measurement across glasses, masks, pose, light, distance and occlusion. Start
with `python main.py identity build`.

Three guarantees follow from that separation:

* A tracker id is never treated as an identity.
* A detection is never forced onto the nearest registered person —
  **`Unknown` is a first-class result.**
* When a face is not visible the answer is **`No face`**, never a guess from
  clothing. That state is distinct from `Unknown`: "unknown" means a face was
  compared and matched nobody; "no face" means there was nothing to compare.

---

## Table of contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model setup](#model-setup)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Person enrollment](#person-enrollment)
- [Face identity engine (passport photos)](#face-identity-engine-passport-photos)
- [Gallery creation](#gallery-creation)
- [Webcam usage](#webcam-usage)
- [Video usage](#video-usage)
- [Image usage](#image-usage)
- [Directory processing](#directory-processing)
- [Output structure](#output-structure)
- [Threshold configuration](#threshold-configuration)
- [Visualization](#visualization)
- [Snapshots](#snapshots)
- [Recording](#recording)
- [Events](#events)
- [How recognition works](#how-recognition-works)
- [How tracking works](#how-tracking-works)
- [Debugging](#debugging)
- [Benchmarking](#benchmarking)
- [Testing](#testing)
- [REST API](#rest-api)
- [Performance tuning](#performance-tuning)
- [Edge deployment (NVIDIA Jetson)](#edge-deployment-nvidia-jetson)
- [GPU setup](#gpu-setup)
- [CPU setup](#cpu-setup)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Privacy considerations](#privacy-considerations)
- [Project structure](#project-structure)

---

## Architecture

```
                ┌─────────────────┐
                │ Input Source    │   webcam / video / image /
                │                 │   directory / RTSP stream
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ YOLO26 Detector │   person class only
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Face detection  │   YuNet: face box + 5 landmarks,
                │ + alignment     │   warped onto the 112x112 template
                └────────┬────────┘   (no face -> "No face", never a guess)
                         ▼
                ┌─────────────────┐
                │ Face Encoder    │   ArcFace w600k_r50 → 512-D vector
                └────────┬────────┘   (or SFace → 128-D)
                    embedding (L2-normalised)
                         ▼
              ┌──────────────────────┐
              │ Identity Gallery     │   [N, D] matrix, cached on disk
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ Similarity Matcher   │   one matmul, cosine
              └──────────┬───────────┘
                  recognised / unknown
                         ▼
              ┌──────────────────────┐
              │ Tracker + Temporal   │   BoT-SORT + identity stabilization
              │ Identity Stability   │
              └──────────┬───────────┘
                         ▼
         ┌────────────────────────────────┐
         │ Renderer / Events / Recording  │
         │ Snapshots / Video / Metadata   │
         └────────────────────────────────┘
```

Each stage is an interface (`Detector`, `FaceDetector`, `ReIDEncoder`,
`Tracker`, `BaseSource`, `EmbeddingStore`, …) wired together by a single
composition root (`src/pipeline/engine.py`). Swapping a backend — or the whole
recognition modality — does not touch the pipeline.

### Recognition modes

`recognition.mode` is always explicit; the system never switches silently.

| | `face` (default) | `person_reid` |
| --- | --- | --- |
| Identity from | the face alone | whole-body appearance |
| Change of clothing | **no effect** | identity lost |
| Person facing away | reports `No face` | still works |
| Distance | needs a face ≳36 px | works further away |
| Encoder | ArcFace / SFace | `yolo26*-reid.onnx` |

Both are fully implemented. Switch with one line, and the gallery re-enrols
itself automatically because the encoder fingerprint changes.

---

## Requirements

- Python **3.11+**
- ~2 GB of disk for the model weights and dependencies
- Optional: an NVIDIA GPU with a CUDA build of PyTorch, or Apple Silicon (MPS)

Everything runs on CPU; the GPU is an optimisation, not a requirement.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For development (tests + linting):

```bash
pip install -r requirements-dev.txt
```

---

## Model setup

```bash
# Person detector (YOLO26, ~5 MB) - downloads automatically on first use
python -c "import os; os.makedirs('models', exist_ok=True); os.chdir('models'); \
from ultralytics.utils.downloads import attempt_download_asset; \
attempt_download_asset('yolo26n.pt')" 2>/dev/null || \
  python main.py config validate --check-models   # also triggers the download

# Face detector + face recogniser
python scripts/fetch_face_models.py              # YuNet + ArcFace  (~175 MB)
python scripts/fetch_face_models.py --tier lite  # YuNet + SFace    (~39 MB)
```

| Role | Default | Alternatives |
| --- | --- | --- |
| Person detector | `yolo26n.pt` | `yolo26s/m/l/x.pt` |
| Face detector | `face_detection_yunet_2023mar.onnx` | any YuNet build |
| Face encoder | `w600k_r50.onnx` (ArcFace, 512-D) | `face_recognition_sface_2021dec.onnx` (SFace, 128-D), `glintr100.onnx`, any ArcFace-format ONNX |
| Body encoder (`person_reid` mode only) | `yolo26n-reid.onnx` | `yolo26s/m/l/x-reid.onnx`, any `.onnx`, or a `.pt` Ultralytics checkpoint |

Measured on the bundled demo set, both face encoders separate genuine from
impostor faces cleanly:

| Encoder | Size | Genuine floor | Impostor ceiling | Gap |
| --- | --- | --- | --- | --- |
| ArcFace `w600k_r50` | 174 MB | 0.911 | 0.154 | **0.757** |
| SFace | 39 MB | 0.726 | 0.138 | 0.588 |

The shipped threshold of `0.45` sits inside both gaps. ArcFace is the default
because the margin is wider; SFace is roughly 1.8x faster and much smaller.

Every path is set in `config.yaml` - **no model path or size is hard-coded
anywhere in the code.**

---

## Quick start

```bash
# 1. Build the bundled demo dataset (two people + test images + a test clip)
python scripts/build_demo.py

# 2. Check the configuration
python main.py config validate --config config.yaml

# 3. Enroll everybody listed under people: in config.yaml
python main.py gallery build --config config.yaml

# 4. Run it
python main.py images --input data/demo/test_images
python main.py video  --input data/demo/test_video/walkthrough.mp4
python main.py webcam --device 0
```

Expected output from step 4:

```
Images processed  : 6
Detections        : 10 (5 recognized, 5 unknown)
Recognized        : person_a=3, person_b=2
  01_person_a.jpg: Person A (0.96)
  02_person_b.jpg: Person B (0.95)
  03_a_and_b.jpg: Person A (0.98), Person B (0.94)
  04_unknown_only.jpg: Unknown (-0.03), Unknown (-0.07), No face, No face
  05_person_a_different_clothes.jpg: Person A (0.98)
  06_person_a_face_hidden.jpg: No face
```

The last two lines are the point of the whole design:

* **05** recolours everything below the chin and leaves the face untouched -
  the identity is unchanged (0.98).
* **06** obscures the face and leaves the clothing untouched - the system
  reports `No face` instead of naming the person. A system that still said
  "Person A" here would be reading the clothes.

---

## Configuration

YAML drives everything. Three files ship with the project:

| File | Purpose |
| --- | --- |
| `config.yaml` | the working configuration you edit |
| `configs/default.yaml` | every option, documented with its default |
| `configs/production.yaml` | headless profile: event recording, retention, stricter thresholds |

Anything you omit falls back to the schema default, so `config.yaml` only needs
what you actually change. Relative paths resolve against **the configuration
file's own directory**, never the working directory.

Validation happens once at startup and produces actionable messages:

```bash
$ python main.py config validate --config config.yaml
Configuration OK: config.yaml
  people          : 2 (2 enabled)
  recognition     : face  (face only -- independent of clothing)
  person detector : models/yolo26n.pt
  face detector   : models/face_detection_yunet_2023mar.onnx
  face encoder    : models/w600k_r50.onnx
  min face size   : 36px (identity held 45 frames while the face is hidden)
  threshold       : 0.45 (high 0.6)
  tracking        : on (botsort.yaml)
  output directory: /…/data/output

$ python main.py config show --config config.yaml --section matching
```

Environment variables can be interpolated with `${VAR}` or `${VAR:-default}` —
see `.env.example`.

---

## Person enrollment

People are declared in YAML. **Person data is never written into Python source.**

```yaml
people:
  - id: "john_doe"          # unique, filename-safe
    name: "John Doe"        # required
    title: "Manager"        # optional
    image_path: "data/persons/john_doe.jpg"   # exactly ONE reference image
    enabled: true
```

Given that, the system automatically:

1. loads the reference image (read-only — **your original file is never modified**),
2. runs YOLO26 to find the person,
3. selects the right detection (see below),
4. crops and preprocesses it,
5. finds and aligns the face, then runs the face encoder,
6. L2-normalises the embedding,
7. stores it in the gallery with metadata.

You never have to crop the person yourself.

**When the reference image contains several people**, the default is to refuse
rather than silently register the wrong person:

```yaml
enrollment:
  require_single_person: true            # refuse if >1 person is detected
  selection_strategy: "largest_person"   # largest_person | highest_confidence | center_most
```

```
error: 4 people detected in the reference image for 'john_doe'
(data/persons/group.jpg); refusing to guess which one to register.
Boxes: [49,398,240,902] conf=0.87, … Either crop the image to a single
person, or set enrollment.require_single_person: false to use the
'largest_person' strategy.
```

**In face mode the reference photo must show the person's face.** If it does
not, enrollment fails — it is never quietly downgraded to body appearance,
because an identity registered from clothing would defeat the whole point of
the mode:

```
error: no face detected in the reference image for 'john_doe'
(data/persons/john_doe.jpg). recognition.mode is 'face', which identifies
people from facial appearance only, so the reference photo must show the
person's face. Use a clearer, more frontal photo, or lower
face.detection_confidence.
```

**Reference image quality** is assessed and reported. In face mode the checks
are about the face — resolution, inter-ocular distance, landmark availability,
detector confidence, sharpness and exposure — because that is what drives
accuracy. By default these are warnings; set
`enrollment.quality.fail_on_warnings: true` to make them fatal.

```
WARNING | Reference image quality | identity=person_b issue="inter-ocular
distance is 15px (recommended at least 22px); the face may be turned away"
```

A weak reference degrades every future comparison, so the reference gates are
deliberately stricter than the runtime ones (`enrollment.quality.min_face_size`
defaults to 60 px against a runtime `face.min_face_size` of 36 px).

---

## Face identity engine (passport photos)

The default enrollment path registers a person from any photograph. The **face
identity engine** is the path built specifically for the harder version of the
problem: the only thing you have is one passport-style portrait, and the person
you later see may be wearing glasses or a mask, half-turned, badly lit, further
away, partly hidden, or photographed years later.

It is the same architecture with more of it measured:

```
YOLO26 person detection → tracking → SCRFD face detection → alignment
→ quality + visibility → ArcFace embedding → identity gallery
→ calibrated matching → temporal evidence → identity decision
```

### Enrol, calibrate, measure

```bash
# One passport photo per person, from config.yaml's people: list ...
python main.py identity build

# ... or from a directory laid out as <person_id>/<photo>
python main.py identity build --from-dir data/evaluation/enrollment

python main.py identity list          # who is registered, and reference warnings

# Fit thresholds and the score→probability mapping on labelled data
python main.py calibrate --dataset data/evaluation

# Per-condition accuracy on the held-out split
python main.py evaluate-faces --dataset data/evaluation --output report.json

# Does each component pay for itself?
python main.py experiments --dataset data/evaluation
```

Once the face gallery has anyone in it, it becomes the identity path for
`run`, `video`, `webcam` and `image` automatically. Startup logs
`identity=face_identity` when it does.

### Enrollment is face-first

No person detection runs on a reference photograph. A passport photo is a head
and shoulders, and requiring a person box can only reject valid references.

Reference quality is checked and **reported rather than silently accepted** —
a weak reference degrades every future comparison against that person, whereas
a weak query degrades only that frame:

```
$ python main.py identity build --from-dir data/evaluation/enrollment
Face gallery: 3 identity(ies), 3 embedding(s)
  enrolled : bus_00, zidane_00, zidane_01

  Reference photographs with quality warnings:
    zidane_00: REFERENCE_POSE_TOO_EXTREME, WEAK_LANDMARKS, REFERENCE_OCCLUDED
```

Set `reference_quality.fail_on_warnings: true` to make that fatal instead.

### Seven numbers, never merged into one

| Field | Meaning |
| --- | --- |
| `detector_confidence` | YOLO26's score that this box is a person |
| `face_detection_confidence` | SCRFD's score that this is a face |
| `face_quality` | how much identity information the face carries |
| `face_similarity` | cosine between embeddings — **not a probability** |
| `track_id` | which trajectory this is — **not an identity** |
| `identity_id` | who the system says this is |
| `identity_confidence` | calibrated probability, or `null` when uncalibrated |

`identity_confidence` is `null` until `calibrate` has been run. The system
does not invent a probability from a cosine.

### Measured results

On the held-out test split (135 queries, 3 registered identities, thresholds
fitted on a disjoint split):

| Condition | n | accuracy | rank-1 | FAR |
| --- | --- | --- | --- | --- |
| normal / glasses / mask | 60 | 100 % | 100 % | 0 % |
| pose / light / distance / blur | 60 | 100 % | 100 % | 0 % |
| partial (heavy occlusion) | 15 | 73.3 % | 73.3 % | 0 % |
| **overall** | **135** | **97.0 %** | **96.7 %** | **0 %** |

Unknown rejection: 100 %. All four failures are face *detection* failures, not
matching failures.

**This is not a claim of 90 % real-world accuracy.** The repository contains
four distinct real faces, every condition is a synthetic transformation of
them, and there is no age-variation split because ageing cannot be simulated.
What the numbers establish is that the pipeline is correctly wired and that
genuine and impostor scores separate cleanly; what they cannot establish is a
field identification rate. [docs/identity.md](docs/identity.md) §10 explains
what a defensible evaluation would require.

---

## Gallery creation

```bash
python main.py gallery build    --config config.yaml       # enroll what's missing
python main.py gallery build    --rebuild                  # re-enroll everybody
python main.py gallery rebuild  --person person_001        # one person
python main.py gallery list                                # what is registered
python main.py gallery show     person_001                 # stored metadata
python main.py gallery remove   person_001                 # drop one entry
python main.py gallery clear    --yes                      # drop all embeddings
```

```
$ python main.py gallery list
ID        NAME                     TITLE                DIM  STATE
person_a  Person A                 Demo Subject         512  ready
person_b  Person B                                      512  ready  (1 quality warning(s))
```

Embeddings are cached on disk and **not** regenerated on every run:

```
data/gallery/
├── embeddings/person_a.npy            # L2-normalised float32 vector
├── metadata/person_a.json             # provenance + quality report
└── crops/person_a_person_a.jpg        # the crop that was actually embedded
```

A cached entry is automatically invalidated when the reference image changes on
disk, its path changes, or the *encoder fingerprint*
(`backend:model:HxW:dimension`) changes — so swapping the encoder, or switching
`recognition.mode` between `face` and `person_reid`, triggers a clean
re-enrollment instead of silently comparing incompatible vectors.
Embedding dimension is read from the model, never assumed.

---

## Webcam usage

```bash
python main.py webcam --device 0
python main.py webcam --device auto          # probe for the first working camera
python main.py webcam --device 1 --no-show   # headless
```

```yaml
source:
  type: "webcam"
  device: 0
  width: 1280      # requested; the driver may ignore it (a warning is logged)
  height: 720
  fps: 30
  buffer_size: 1   # keep latency low for live use
```

Press `q` or `Esc` in the preview window to stop; `Ctrl-C` finishes the current
frame and closes all output files cleanly.

---

## Video usage

```bash
python main.py video --input data/videos/test.mp4
python main.py video --input clip.mp4 --record event --snapshots recognized
```

Frame dimensions are preserved, the output FPS follows the source (override with
`output.video_fps`), and the codec is configurable (`output.codec`, default
`mp4v`). **Output files are never silently overwritten** — a numeric suffix is
appended unless `output.overwrite: true`.

---

## Image usage

```bash
python main.py image --input data/images/test.jpg
```

Produces an annotated image, a JSON document, and optional per-person crops.
Multiple people in one image are matched **independently**, so an image
containing *John, Jane, a stranger and John again* resolves all four correctly.

---

## Directory processing

```bash
python main.py images --input data/images/
python main.py images --input data/images/ --recursive
```

Supported extensions are centralised and configurable:

```yaml
source:
  extensions: [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
```

A single corrupt file is skipped with a warning; the batch still completes.

`python main.py run <anything>` infers the source kind from the argument
(camera index, file, directory or URL).

---

## Output structure

```
data/output/
├── images/      <name>_annotated.jpg          annotated still images
├── videos/      <source>_<timestamp>.mp4      annotated video / event clips
├── snapshots/   2026-09-21/10-30-01_john_doe_0.73.jpg  + matching .json
├── crops/       per-detection person crops (optional)
├── metadata/    <source>_<timestamp>.jsonl    one JSON object per frame
├── events/      events.jsonl                  the event stream
└── debug/       debug artefacts (only with --debug)
```

Snapshot sidecar:

```json
{
  "source": "webcam_0",
  "timestamp": 1789976624.03,
  "iso_time": "2026-09-21T11:13:44",
  "identity_id": "john_doe",
  "identity_name": "John Doe",
  "identity_title": "Manager",
  "reid_similarity": 0.9312,
  "recognition_status": "recognized",
  "recognition_detail": "",
  "bbox": [100.0, 50.0, 220.0, 400.0],
  "face": {
    "bbox": [138.0, 72.0, 186.0, 134.0],
    "score": 0.9712,
    "size": 48.0,
    "eye_distance": 20.6,
    "quality": "ok"
  },
  "track_id": 17,
  "detector_confidence": 0.9088
}
```

---

## Threshold configuration

```yaml
matching:
  metric: "cosine"
  recognition_threshold: 0.45        # below this -> Unknown
  high_confidence_threshold: 0.60    # above this -> "recognized" rather than "low_confidence"
  ambiguity_margin: 0.0              # reject when the top-2 identities are this close
  unknown_label: "Unknown"
  no_face_label: "No face"
```

**There is no universal threshold**, and face and whole-body embeddings live in
different similarity regimes — a cosine of 0.45 is a confident face match but
near-noise for body appearance. Leaving these unset picks the right default for
`recognition.mode`; an explicitly configured value always wins.

| Mode | Default threshold | Measured on the demo set |
| --- | --- | --- |
| `face` | 0.45 / 0.60 | impostors ≤ 0.15, genuine ≥ 0.91 (ArcFace) |
| `person_reid` | 0.70 / 0.80 | impostors ≤ 0.69, genuine ≥ 0.20 — overlapping |

The face distributions do not overlap at all on this data; the body ones do.
That gap is the quantitative reason face mode is the default.

Measure it on your own data:

```bash
python main.py evaluate --dataset data/demo/evaluation --output report.json
```

```
evaluation/
├── known/
│   ├── john_doe/       # directory name == a person id in your config
│   │   ├── img_01.jpg
│   │   └── img_02.jpg
│   └── jane_smith/
└── unknown/            # people who are NOT registered
    └── stranger_01.jpg
```

```
Threshold calibration (measured on the supplied dataset)

  genuine  (same person)   n=12   min=0.911 p05=0.913 mean=0.947 max=0.975
  impostor (other/unknown) n=24   min=-0.071 mean=0.031 p95=0.121 max=0.154

  threshold  TA   FA   TR   FR   IDerr  prec   recall  F1
       0.40  12   0    6    0    0      1.000  1.000   1.000
       0.45  12   0    6    0    0      1.000  1.000   1.000
       0.95  4    0    6    8    0      1.000  0.333   0.500

  Best F1 on THIS dataset: threshold = 0.53 (F1 1.000); configured = 0.45.
  Clean separation: impostors peak at 0.154, genuine matches bottom out at
  0.911 (gap 0.757).
  Suggested threshold for this data: 0.53
```

When the two distributions do not overlap, any threshold inside the gap
separates them perfectly, so the tool recommends the **midpoint** — the point
furthest from both a false accept and a false reject. Samples with no usable
face are excluded from the statistics rather than scored as misses, because
they are not evidence about any threshold.

Raise the threshold to reduce false accepts (a stranger matched to an employee);
lower it to reduce false rejects (an employee reported as Unknown). Which error
is worse is a deployment decision, not a technical one.

---

## Visualization

Each person gets a box coloured by outcome, with the face box drawn inside it in
face mode so a wrong match is immediately explainable:

```
recognized        [17] John Doe          low confidence    [17] John Doe
                  Manager                                  Manager
                  Face: 0.93                               Face: 0.51

unknown           [22] Unknown           face not visible  [22] No face
                  Face: 0.11                               face hidden
```

The score is labelled `Face:` in face mode and `ReID:` in `person_reid` mode, so
screenshots are never ambiguous about which modality produced them. Colours,
thickness, font scale and whether to draw the face box are all configurable
under `display:`.

---

## Snapshots

```yaml
output:
  save_snapshots: true
  snapshot_mode: "recognized"   # all | recognized | unknown | events_only | disabled
  snapshot_cooldown_seconds: 5.0   # per track/identity rate limit
  save_crops: false
  save_metadata: true
```

`events_only` saves one snapshot per track, taken once the identity has
*settled* — it waits for `identity_stability.minimum_recognized_frames` so a
person who is recognised two frames later is not filed under "unknown".

---

## Recording

```yaml
recording:
  mode: "event"                 # continuous | event | disabled
  save_when_identity_detected: true
  save_unknown: false
  pre_event_seconds: 3.0
  post_event_seconds: 5.0
  max_clip_seconds: 120.0
```

Event mode keeps a **ring buffer** of recent frames, so a clip genuinely starts
3 seconds *before* the person appeared:

```
[ ring buffer: last 3 s ] + [ trigger frame ] + [ 5 s tail ]
```

Without the buffer the interesting approach is always missing. Continuous mode
rolls over to a new file at `max_clip_seconds` instead of producing one huge
clip.

```bash
python main.py video --input clip.mp4 --record event
python main.py webcam --record continuous
```

---

## Events

A generic event layer keeps integrations out of the processing code. Subscribers
attach to the `EventManager`; a failing subscriber is logged and never stops
video processing.

`PERSON_DETECTED` · `PERSON_RECOGNIZED` · `UNKNOWN_PERSON_DETECTED` ·
`PERSON_LOST` · `IDENTITY_CHANGED` · `TRACK_STARTED` · `TRACK_ENDED` ·
`SNAPSHOT_SAVED` · `VIDEO_RECORDING_STARTED` · `VIDEO_RECORDING_STOPPED` ·
`SOURCE_STARTED` · `SOURCE_ENDED` · `GALLERY_UPDATED` · `ERROR`

```bash
python main.py events --limit 20
python main.py events --type person_recognized --json
```

---

## How recognition works

1. **Detect people.** YOLO26 returns person boxes. That is *all* it does here —
   it answers "where is a person", never "who".
2. **Find the face.** Inside each person box (the head region by default), YuNet
   locates the face and its five landmarks. No face, or a face below
   `face.min_face_size`, ends the chain here with `No face`.
3. **Align.** The five landmarks drive a *similarity* transform (rotation +
   uniform scale, no shear) onto the canonical 112×112 ArcFace template. This
   step is where most of face recognition's accuracy lives: it removes head
   roll and scale so the encoder sees a normalised face rather than an
   arbitrarily posed one.
4. **Embed.** ArcFace maps the aligned chip to a 512-D vector (SFace: 128-D).
   **Only pixels inside the aligned face reach the encoder** — nothing below
   the neck, which is exactly why clothing cannot influence the result.
5. **Normalise.** L2 normalisation, so cosine similarity is a dot product. A
   zero vector stays zero, so a degenerate crop cannot accidentally match.
6. **Search.** All query embeddings are compared against the whole gallery in
   **one matrix multiplication** (`[M, D] @ [D, N]`), so cost grows slowly with
   the number of registered people.
7. **Decide (open set).**

   ```
   no usable face                              -> NO_FACE
   best = argmax(similarity)
   if similarity[best] < recognition_threshold -> UNKNOWN
   elif margin to runner-up < ambiguity_margin -> REJECTED
   elif similarity[best] >= high_confidence    -> RECOGNIZED
   else                                        -> LOW_CONFIDENCE
   ```

Every detection carries both the instantaneous decision and the stabilized one,
the face box the decision was based on, and the top-k similarity scores — so an
uncertain or wrong match is explainable rather than opaque.

### Why clothing cannot affect the answer

The encoder only ever sees the aligned face chip. To show this is true in
practice rather than in principle, the demo includes an image where everything
below the chin is recoloured and the face is left bit-identical:

| Image | `face` mode | `person_reid` mode |
| --- | --- | --- |
| `01_person_a.jpg` (baseline) | Person A **0.96** | Person A **0.96** |
| `05_person_a_different_clothes.jpg` | Person A **0.98** | Unknown **0.43** |
| `06_person_a_face_hidden.jpg` | **No face** | Unknown 0.54 |

Whole-body appearance collapses from 0.96 to 0.43 and loses the person. The
face-based decision is unchanged. Reproduce it with:

```bash
python main.py images --input data/demo/test_images
```

## How tracking works

Tracking provides **temporal continuity**, not identity. BoT-SORT (with optional
appearance cues) assigns a track id; the identity still comes from the gallery
match every time.

### Temporal identity stabilization

A single frame is weak evidence — motion blur, a turned back or a passing
occluder can drop similarity for a frame or two. Displaying the raw per-frame
decision makes the label flicker `John → Unknown → John`.

```yaml
tracking:
  track_buffer: 30          # frames a lost track survives inside the tracker
  lost_track_timeout: 30    # frames before this app forgets the track's identity
  identity_stability:
    enabled: true
    strategy: "weighted_vote"        # weighted_vote | ema | majority
    history_size: 10
    minimum_recognized_frames: 3     # evidence needed before naming anybody
    switch_margin: 0.08              # hysteresis against identity swaps
    decay: 0.9
    identity_persistence_frames: 15  # how long an identity survives unsupported
```

**`weighted_vote`** (default): each observation contributes
`decay ** age * similarity` to the identity it voted for, so recent confident
frames dominate and old ones fade. A candidate becomes eligible only after
`minimum_recognized_frames` votes, which stops one lucky frame from naming
someone. An incumbent is replaced only when a challenger beats it by
`switch_margin`, which stops two look-alikes from swapping labels every frame.

### A hidden face is neutral evidence

Face mode adds a third kind of observation. Someone turning around produces
frames with **nothing to compare** — neither a match nor a rejection. Those
frames are excluded from the evidence window entirely, so a person walking away
cannot slowly erode their own identity. The track holds the identity it already
established for `face.identity_hold_frames`, and then releases it rather than
carrying it indefinitely on stale evidence.

Measured on the bundled demo clip, which obscures Person A's face from 3.5 s to
6.0 s (62 frames) while leaving their clothing untouched:

```
frames 88-132   instantaneous = no_face,  displayed = person_a   (45 frames held)
frames 133-149  instantaneous = no_face,  displayed = No face    (hold expired)
frames 214+     face visible again -> re-identified as person_a
```

Exactly `identity_hold_frames: 45` frames are held, then the label is dropped.
The bound matters: if the tracker has meanwhile swapped this track onto a
different person, an unbounded hold would attach the wrong name to them.

Critically, the pipeline does **not** reuse the track's last embedding to cover
a missing face. Doing so would keep asserting an identity from a face that is
no longer visible, with no limit and no audit trail. Carrying an identity across
a hidden face is the stabilizer's job, where it is explicit and bounded.

---

## Debugging

```bash
python main.py video --input clip.mp4 --debug
```

```yaml
debug:
  enabled: false        # off by default
  save_frames: false
  save_crops: true      # the exact crop the encoder saw
  save_matches: true    # full similarity vector behind each decision
  save_track_state: true
  max_frames: 200
```

```
data/output/debug/
├── frame_000001.jpg
├── crop_track_12_frame_000001.jpg
├── match_track_12_frame_000001.json
└── tracks_frame_000001.json
```

`--log-level DEBUG` additionally logs per-detection matches. Raw embedding
values are never logged.

---

## Benchmarking

```bash
python main.py benchmark --input data/demo/test_video/walkthrough.mp4 --frames 200
```

Only measured values are reported; the first `--warmup` frames are discarded.

Measured on **Apple M-series (MPS detector, CPU/CoreML face stage), 960×540,
yolo26n + YuNet + ArcFace, 2 registered identities, 150 frames**:

| Configuration | End-to-end FPS | Face stage | Recognitions |
| --- | --- | --- | --- |
| ArcFace, `reid_interval: 1` | 9.9 | 65.5 ms | 278 |
| **ArcFace, `reid_interval: 3` (default)** | **19.2** | 22.9 ms | 279 |
| SFace, `reid_interval: 1` | 17.7 | 26.6 ms | 278 |
| SFace, `reid_interval: 3` | 25.5 | 9.7 ms | 280 |
| `person_reid` mode (no face stage) | 26.7 | 5.1 ms | — |

Two things worth reading off that table:

* `reid_interval: 3` roughly doubles throughput **with no loss of
  recognitions**, which is why it is the shipped default.
* Face mode costs real time — roughly 1.4× slower than whole-body appearance
  even after tuning. That is the price of clothing invariance.

Counter-intuitively, `face.search_region: frame` (one detection pass over the
whole frame) measured *slower* here — 9.0 FPS against 9.9 — because YuNet at
960×540 costs more than two small crops. It only pays off with many people in
frame; measure before choosing it.

Your numbers will differ. Run the command on your own hardware.

### Benchmark matrix

Sweep configurations against one input, changing only the setting under test:

```bash
python main.py benchmark-suite scenarios              # list the groups
python main.py benchmark-suite matrix \
  --input data/demo/test_video/walkthrough.mp4 \
  -g recognition -g batch --frames 100 --output report.json
```

Groups: `recognition`, `batch`, `encoder`, `search`, `output`, `resolution`,
`pipeline`. Multi-person clips for the crowded scenarios are generated with:

```bash
python scripts/build_crowd_clips.py          # 1, 3, 5 and 10 people @ 720p
```

### Performance regression checking

Targets are per-profile, because a number that makes sense on an Orin Nano is
meaningless on a workstation. An unconfigured target reports `NOT_CHECKED`
rather than inventing a universal requirement.

```yaml
benchmark:
  targets:
    enabled: true
    minimum_fps: 25.0
    maximum_p95_latency_ms: 120.0
    maximum_drop_rate_percent: 5.0
    warn_margin_percent: 10.0      # inside a limit but close -> WARN
```

```bash
python main.py benchmark --input clip.mp4 --check-targets   # exit 1 on FAIL
```

### A measured result worth knowing: batching does nothing here

The shipped ArcFace `w600k_r50.onnx` is exported with a **fixed batch dimension
of 1**. Measured directly on the encoder:

| Call | Total | Per face |
| --- | --- | --- |
| 1 × batch-8 | 226.9 ms | 28.4 ms |
| 8 × batch-1 | 203.5 ms | 25.4 ms |

Batching it is not merely useless but slightly *harmful*, because ONNX Runtime
drops off its accelerated partition. The pipeline now detects the fixed batch
dimension at load time and falls back to single-image calls regardless of what
`recognition.batch` asks for, logging why. To actually benefit, re-export the
encoder with a dynamic batch axis.

This is also why batch size makes no measured difference even with ten people
in frame — and why the Jetson profile's `max_size: 8` is labelled an
engineering default that must be measured on the device, not a result.

---

## Testing

```bash
pytest                      # everything
pytest -m "not slow"        # unit tests only: no models, no GPU, no network
pytest -m integration       # acceptance tests against the real models
ruff check src tests scripts main.py
```

The unit suite mocks model inference, so it runs anywhere in seconds. The
integration suite exercises the real YOLO26 + YuNet + ArcFace models on the demo
dataset — including the clothing-change and hidden-face cases — and skips itself
automatically when the weights or demo data are missing.

```
296 passed, 26 deselected in 1.3s     # pytest -m "not slow"
322 passed in 24s                      # pytest
```

One harness note: a single pytest process loads torch, OpenCV DNN and ONNX
Runtime models many times over, and those libraries' static destructors race at
interpreter shutdown — intermittently aborting with `recursive_mutex lock
failed` *after* every test has passed. `tests/conftest.py` exits the process
once pytest has finished reporting, which preserves the summary and the real
exit status. The application is unaffected; the CLI exits 0 consistently. Set
`REID_TESTS_NO_FAST_EXIT=1` when running coverage, which needs a normal
shutdown to write its data file:

```bash
REID_TESTS_NO_FAST_EXIT=1 pytest --cov=src
```

---

## REST API

Optional, and the CLI does not depend on it:

```bash
pip install fastapi "uvicorn[standard]" python-multipart
python main.py serve --config config.yaml
```

| Method | Path |
| --- | --- |
| `GET` | `/health` |
| `GET` `POST` | `/people` |
| `GET` `PUT` `DELETE` | `/people/{id}` |
| `POST` | `/gallery/build`, `/gallery/rebuild/{id}` |
| `POST` | `/process/image` |
| `GET` | `/events` |

---

## Performance tuning

```yaml
performance:
  reid_interval: 3            # embed a stable track every Nth frame
  embedding_cache: true
  batch_size: 16
  max_detections: 50
  force_reid_on_new_track: true
  force_reid_when_unknown: true   # never skip a track that is still unidentified
  frame_stride: 1                 # process every Nth frame of a video
```

Running the encoder on every box of every frame is wasteful once a track is
stable. `reid_interval` reuses a track's cached embedding between runs — safe
because the identity decision is already smoothed over time. New tracks and
still-unknown tracks are always re-embedded, so skipping never costs a
recognition.

In face mode the recognition stage dominates, so the levers differ from
whole-body mode. Measured on the demo clip, in order of impact:

| Lever | Effect |
| --- | --- |
| `performance.reid_interval: 3` (default) | 9.9 → 19.2 FPS, no accuracy loss |
| SFace instead of ArcFace | a further ~1.3× at a narrower margin |
| smaller detector `imgsz` | cuts the YOLO26 stage |
| `performance.frame_stride` | processes every Nth frame of a video |
| `display.show_window: false` | removes the preview cost |

`face.search_region` trades one detection pass per *person* against one per
*frame*. `upper_body` (default) is faster with few people; `frame` only wins
when many people share a frame. Measure it — on the demo clip `frame` was
slower.

Other levers: a smaller detector model, a larger `batch_size` in crowded
scenes, and `output.save_video: false`.

---

## Edge deployment (NVIDIA Jetson)

A dedicated profile and deployment guide exist for the Jetson Orin Nano Super:

```bash
python main.py system info                       # what the platform provides
python main.py system backends --config configs/jetson_orin_nano_super.yaml
python main.py models build-tensorrt --config configs/jetson_orin_nano_super.yaml
python main.py webcam --config configs/jetson_orin_nano_super.yaml
```

See **[docs/jetson.md](docs/jetson.md)** for JetPack requirements, TensorRT
engine builds, CSI camera setup, hardware encoding and troubleshooting, and
**[requirements-jetson.txt](requirements-jetson.txt)** for the dependency set
(the main `requirements.txt` must *not* be installed as-is on aarch64).

> Every Jetson-specific path — TensorRT execution, CSI capture, NVMM, NVENC —
> is **implemented but not verified on hardware**: no Jetson was available
> during development. The performance figures in this README were measured on
> an Apple M4, and the Jetson profile's targets are engineering targets, not
> measurements. See [docs/jetson.md §1](docs/jetson.md).

### What is portable

Nothing about Jetson is mandatory, and nothing is hard-coded into the pipeline.
The same configuration runs anywhere, degrading with a logged reason:

| Missing | Result |
| --- | --- |
| TensorRT | ONNX Runtime / PyTorch, as before |
| CUDA | MPS on Apple Silicon, otherwise CPU |
| GStreamer in OpenCV | `jetson_camera` falls back to the V4L2 webcam source |
| NVENC encoders | software video writer, with the reason logged |

---

## Architecture: the asynchronous pipeline

Off by default; `pipeline.async: true` (or `--async`) moves capture and output
onto their own threads:

```
Camera ─▶ capture queue ─▶ Detection ─▶ Tracking ─▶ Recognition scheduler
                                                        │
                                                        ▼
                                              Recognition (batched)
                                                        │
       Rendering / Recording / Storage ◀── output queue ┘
```

Every queue is bounded. On a live source an overloaded pipeline discards the
*oldest* queued frames rather than accumulating latency — a preview three
seconds behind is worse than one that skipped frames. Shedding is ordered:
visualization first, then periodic metadata, **never** a recognition event or
an event-triggered snapshot, which are discrete facts rather than samples of a
continuous signal.

```yaml
pipeline:
  async: true
  capture_queue_size: 2
  output_queue_size: 4
  drop_stale_frames: true
  overload_policy: "drop_visualization"   # or drop_all_output | block
```

---

## Recognition scheduling

The single largest saving. The previous rule embedded every unidentified track
on *every* frame, which meant a passer-by who would never match pinned the
encoder at full rate — exactly the case that should cost least.

The scheduler decides per track from its identity state, and backs off
geometrically on tracks that keep failing to match:

| Track state | Behaviour |
| --- | --- |
| `NEW_TRACK` | recognise immediately |
| `RECOGNIZED` | every `stable_interval` frames |
| `LOW_CONFIDENCE` | every `low_confidence_interval` frames |
| `UNKNOWN` | `unknown_interval`, growing by `unknown_backoff_factor` |
| `NO_FACE` | face **detector** only; the encoder never runs |
| `RECOVERING` / `LOST` | forced — a reused track ID may be a different person |

It decides *when* to look, never *what the answer is*. A skipped frame asserts
nothing new: the track keeps the conclusion of its last pass, still bounded by
the existing identity-hold.

Measured on the demo clip (Apple M4, 960×540, ArcFace):

| Configuration | FPS | p50 latency | Recognitions |
| --- | --- | --- | --- |
| every frame | 9.7 | 126.5 ms | 513 |
| fixed interval 3 | 19.7 | 35.7 ms | — |
| **adaptive (default)** | **25.4** | **31.0 ms** | 486 |

2.6× throughput and 4× lower median latency, retaining 95% of recognitions.

---

## GPU setup

```yaml
device:
  device: "auto"     # CUDA -> MPS -> CPU
  fp16: true
  cuda_index: 0
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu          # replaces onnxruntime
python main.py config validate --check-models    # confirms the selected device
```

The selected device, both backends, FP16 availability and model load times are
logged at startup:

```
INFO | Compute device selected | device=cuda:0 detail="NVIDIA RTX 4070 (12.0 GiB)" fp16=True
INFO | YOLO26 detector loaded  | model=yolo26n device=cuda:0 imgsz=640 fp16=True load_ms=412.7
INFO | Face detector loaded    | model=face_detection_yunet_2023mar backend=opencv.yunet
INFO | Face encoder loaded     | model=w600k_r50 backend=onnxruntime.arcface dim=512
INFO | Engine ready            | mode=face detector=yolo26n.pt dim=512 device=cuda:0
```

If CUDA is requested but unavailable the system logs a warning and falls back to
CPU — it does not crash.

## CPU setup

```yaml
device:
  device: "cpu"
  fp16: false
detector:
  imgsz: 480
performance:
  reid_interval: 3
  frame_stride: 2
```

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `no face detected in the reference image` | Face mode needs a visible face in the reference photo. Use a clearer, more frontal shot, or lower `face.detection_confidence`. |
| `the face in the reference image is unusable` | The face is below `enrollment.quality.min_face_size`. Use a closer or higher-resolution photo. |
| Everything reports `No face` | Faces are smaller than `face.min_face_size` (lower it, or move the camera closer), or the face models are missing — run `python scripts/fetch_face_models.py`. |
| A person facing away is never identified | Expected: face recognition cannot identify someone from behind. Raise `face.identity_hold_frames` to carry the identity through, or use `recognition.mode: person_reid` if from-behind identification matters more than clothing invariance. |
| Identity changes when someone changes clothes | You are in `person_reid` mode. Set `recognition.mode: face`. |
| `no person detected in the reference image` | The photo is too small, dark or distant. Use a closer shot, or lower `enrollment.detector_confidence`. |
| `N people detected … refusing to guess` | Crop the reference to one person, or set `enrollment.require_single_person: false`. |
| Everyone is `Unknown` | The gallery is empty (`gallery build`), or the threshold is too high — run `main.py evaluate`. |
| Strangers are matched to registered people | The threshold is too low. Measure it with `main.py evaluate` and raise it. |
| `embedding dimension mismatch … Rebuild the gallery` | The encoder or `recognition.mode` changed. `python main.py gallery build --rebuild`. |
| `cannot open camera '0'` | Device in use, missing permission (macOS: System Settings → Privacy & Security → Camera), or wrong index — try `--device auto`. |
| `cannot open video … codec` | OpenCV lacks the codec. Re-encode to H.264/MP4, or change `output.codec`. |
| Preview window fails to open | Headless machine — the run continues and logs a warning. Set `display.show_window: false`. |
| Labels flicker between identities | Raise `identity_stability.history_size` / `minimum_recognized_frames` / `switch_margin`. |
| Identity lost during occlusion | Raise `tracking.track_buffer` and `identity_stability.identity_persistence_frames`; in face mode also `face.identity_hold_frames`. |
| Face mode is slow | Raise `performance.reid_interval`, or switch to the smaller SFace encoder. See [Performance tuning](#performance-tuning). |
| Low FPS | See [Performance tuning](#performance-tuning). |
| `corrupt embedding cache` warning | The cache entry was discarded and will be regenerated automatically. |

---

## Limitations

These are real and worth reading before deploying.

### Inherent to face-based identity

- **It needs a visible face.** A person facing away, wearing a full-face
  covering, or far enough that the face falls under `face.min_face_size`
  cannot be identified. The system reports `No face` rather than guessing.
  This is the direct trade for clothing invariance: whole-body ReID identifies
  people from behind, face recognition cannot.
- **Resolution is the binding constraint.** The default runtime floor is 36 px
  across the face. On a 1080p camera that is roughly a person at 10-15 m with a
  standard lens. Lowering it widens range at the cost of precision; the honest
  fix is a longer lens or a closer camera, not a lower threshold.
- **Pose and occlusion degrade it.** Strong profile views trip the
  `min_eye_distance` gate and are refused rather than matched badly.
- **Ageing, heavy makeup, and identical twins** remain hard for any face
  recogniser, this one included.

### Measurement and accuracy

- **Accuracy is not measured here.** The bundled demo verifies that the
  pipeline works end to end on real faces and that the clothing-invariance
  property holds. It is **not** a face-recognition benchmark: the test crops
  derive from the same photographs as the references, so they share lighting
  and session. No claim is made about LFW, IJB-C or your cameras. Measure it
  yourself with `main.py evaluate`.
- **The 97 % on the evaluation set is not a field accuracy.** The evaluation
  dataset holds four distinct real faces and every condition is a synthetic
  transformation of them. A drawn mask is not a mask and a warped frontal
  photograph is not a turned head. A defensible figure needs at least 30 real
  identities captured separately under each condition; see
  [docs/identity.md](docs/identity.md) §10.
- **Robustness to ageing is untested.** The evaluation set has no
  age-variation split, because ageing cannot be simulated. ArcFace is trained
  on data that includes it, so some robustness is inherited — but inherited is
  not demonstrated, and this system does not demonstrate it.
- **On this dataset no component of the identity stack can be shown to pay for
  itself.** The A/B ladder from `B_face_only` to `F_full_system` scores
  identically at 97.0 %. Calibration, quality weighting and per-visibility
  thresholds are justified by the failure modes they prevent, not by a
  measured gain here.
- **The shipped threshold is a starting point.** `0.45` was measured on a
  small, augmentation-derived demo set and sits inside the clean gap for both
  shipped encoders. Re-measure per deployment.
- **One reference image is a constraint, not a guarantee.** One-shot enrollment
  is the requested workflow and it works well for faces — far better than for
  whole-body appearance — but a single photo still captures one pose, one
  lighting condition and one point in time.

### Engineering

- **Face mode is slower.** ~19 FPS against ~27 for whole-body appearance on the
  reference hardware, after tuning. See [Performance tuning](#performance-tuning).
- **Multi-image enrollment is not exposed.** The aggregation path exists and is
  tested, but the supported workflow today is one image per person.
- **No vector database.** Gallery search is an exact matmul over a NumPy matrix
  — fast for hundreds to low thousands of identities. Beyond that, implement
  `EmbeddingStore` against FAISS/Qdrant/Milvus; the interface is already the seam.
- **Tracker ids are not stable across long occlusions.** That is expected; it is
  why identity comes from the face. But a track id in a snapshot is not a
  durable handle.
- **The identity hold is a deliberate trade-off.** While a face is hidden the
  label is carried for `face.identity_hold_frames` on the strength of the
  *track*. If the tracker swaps that track onto a different person within that
  window, the wrong name is shown until the budget expires. Shorten it where
  that risk matters more than label continuity; set it to 0 to require a
  visible face at every moment.
- **Benchmarks are hardware-specific.** The numbers above come from one Apple
  Silicon machine on a 960×540 clip.

### Regulatory

- **Face recognition is regulated far more heavily than person detection.**
  Several jurisdictions restrict or prohibit it outright, particularly in
  public spaces. See [Privacy considerations](#privacy-considerations) — and
  take actual legal advice before deploying.

## Privacy considerations

This system performs **face recognition**, which is biometric processing in the
strict legal sense — not merely "biometric-like". In the EU it is special
category data under GDPR Art. 9 and additionally constrained by the AI Act; in
Illinois it falls under BIPA, which carries a private right of action; several
US states and cities restrict or ban it outright, especially in public spaces
and for law enforcement.

It is built for legitimate, authorised use — access control, safety, attendance
— with the person's knowledge and an established lawful basis. Switching
`recognition.mode` to `face` materially raises your compliance obligations
compared with whole-body appearance matching. Get advice specific to your
jurisdiction and purpose before deploying, not after.

Privacy-conscious defaults, all under your control:

- **Local by default.** No cloud service, no telemetry, no upload. Images,
  embeddings and events stay on the machine you run it on. The only network
  access is the one-time model download.
- **Snapshots and recording are configurable and can be switched off entirely**
  (`output.save_snapshots: false`, `recording.mode: "disabled"`).
- **Raw embeddings are never logged** and are never returned by the API or the
  `gallery show` command.
- **Explicit storage locations** — everything written is under the configured
  `output.directory` and `gallery.directory`.
- **Retention policy** with a dedicated command:

  ```yaml
  retention:
    enabled: true
    snapshots_days: 30
    videos_days: 30
    events_days: 90
  ```

  ```bash
  python main.py retention --dry-run     # see what would be deleted
  python main.py retention               # apply
  ```

  A period of `0` days means "keep forever" and is never destructive.
- **Deletion is supported**: `gallery remove <id>` drops a person's face
  embedding and metadata; the original reference image is yours and is never
  modified.
- **Switching to non-biometric matching** is one line: set
  `recognition.mode: person_reid` to match on clothing/appearance instead of
  faces. Less accurate and not clothing-invariant, but it is not face
  recognition, which may matter for what you are allowed to deploy.

Consider: signage and notice wherever the system operates, an explicit consent
or other lawful basis per enrolled person, restricting who can read
`data/output/` and `data/gallery/`, keeping retention short, a documented
deletion path on request, and a DPIA (or local equivalent) before you switch it
on.

---

## Project structure

```
.
├── main.py                     CLI entry point
├── config.yaml                 working configuration
├── configs/                    default.yaml, production.yaml
├── requirements.txt            requirements-dev.txt, pyproject.toml, .env.example
├── models/                     model weights (git-ignored)
├── scripts/
│   ├── build_demo.py           builds the demo dataset
│   └── fetch_face_models.py    downloads the face models
├── data/
│   ├── persons/                your reference images
│   ├── demo/                   generated demo dataset (incl. the
│   │                            clothing-change and face-hidden cases)
│   ├── gallery/                embeddings/ metadata/ crops/
│   └── output/                 images/ videos/ snapshots/ crops/ metadata/ events/ debug/
├── src/
│   ├── config/     schema.py loader.py paths.py
│   ├── core/       types.py exceptions.py
│   ├── detection/  detector.py yolo26_detector.py
│   ├── face/       detector.py embedder.py align.py preprocess.py
│   │                factory.py types.py
│   ├── reid/       encoder.py yolo26_reid.py onnx_reid.py preprocess.py factory.py
│   ├── identity/   gallery.py matcher.py enrollment.py embedding_store.py quality.py
│   ├── tracking/   tracker.py botsort_tracker.py track_manager.py stabilizer.py
│   ├── sources/    base.py webcam.py video.py image.py directory.py stream.py factory.py
│   ├── pipeline/   engine.py processor.py runner.py image_runner.py metrics.py debug.py
│   ├── output/     renderer.py snapshot.py recorder.py metadata.py retention.py
│   ├── events/     types.py manager.py
│   ├── tools/      benchmark.py calibration.py
│   ├── cli/        main.py common.py gallery_cmd.py process_cmd.py tools_cmd.py
│   ├── api/        server.py schemas.py        (optional)
│   └── utils/      logging.py device.py image.py
└── tests/          322 tests
```

---

## Licence note

This project depends on **Ultralytics YOLO26**, which is distributed under
**AGPL-3.0** (a commercial licence is available from Ultralytics). Check that
this fits your intended use before deploying.
