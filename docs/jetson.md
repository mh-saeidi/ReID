# NVIDIA Jetson Orin Nano Super deployment

Reference for running this face-recognition ReID system on a Jetson Orin Nano
Super (8 GB), using the shipped deployment profile
[`configs/jetson_orin_nano_super.yaml`](../configs/jetson_orin_nano_super.yaml).

---

## 1. Scope and verification status

> **Every Jetson-specific code path in this repository is IMPLEMENTED but NOT
> VERIFIED on physical Jetson hardware.** No Jetson board was available at any
> point during development. TensorRT engine building and execution, CSI capture
> through `nvarguscamerasrc`, the NVMM zero-copy path and the NVENC hardware
> encoders have never been executed. They are written against the documented
> APIs and covered by unit tests that mock the platform; they have not been run
> against the real thing.

Nothing in this document reports a measured Jetson number, because none exists.
Where a value looks like a performance figure, it is an engineering default or a
target, and it is labelled as such — including inside the YAML profile itself.

### What each status means here

| Status | Meaning |
| --- | --- |
| **IMPLEMENTED** | The code exists, is linted and unit-tested, and is intended to work. It has not been executed on the hardware it targets. |
| **MEASURED** | Actually observed on the development machine (Apple M4, macOS `Darwin 25.6.0` arm64, 10 cores, 16 GiB, Python 3.13.9, torch 2.14.0, OpenCV 5.0.0 — **no CUDA, no TensorRT, no GStreamer support in OpenCV**). |
| **NOT VERIFIED** | Requires a Jetson (or at least a CUDA GPU) to exercise at all. Never run. |

### Component-by-component

| Area | Status | Notes |
| --- | --- | --- |
| Capability detection (`src/hardware/capabilities.py`) | IMPLEMENTED; **MEASURED** on the dev machine | `system info` runs and reports correctly on macOS/arm64. The Jetson branches (device-tree parsing, L4T→JetPack mapping, `nvarguscamerasrc` / `nvvidconv` / `nvv4l2h26xenc` probing) have never matched anything real. |
| Backend selection (`src/backends/selection.py`) | IMPLEMENTED; **MEASURED** for the non-TensorRT branches | On the dev machine every role resolves to `onnx`/`opencv` with the reason `Apple Silicon: MPS/CoreML`. The TensorRT branch is exercised only by tests with faked capabilities. |
| Engine cache and invalidation (`src/backends/engine_store.py`) | IMPLEMENTED; **MEASURED** | Metadata, fingerprinting and every invalidation rule are unit-tested with synthetic engine files on the dev machine. |
| TensorRT engine **building** (`src/backends/tensorrt_builder.py`) | **NOT VERIFIED** | Neither the Python-API path nor the `trtexec` path has ever produced an engine. Guard logic (INT8 refusal, unsupported source formats, `allow_build: false`) is unit-tested. |
| TensorRT engine **execution** (`src/backends/tensorrt_runtime.py`, `src/backends/trt_encoder.py`) | **NOT VERIFIED** | Requires TensorRT plus `pycuda`. Never loaded an engine. |
| CSI capture / NVMM (`src/sources/jetson_camera.py`) | **NOT VERIFIED** | Pipeline *strings* are unit-tested for shape and content; no pipeline has ever been instantiated. |
| Hardware encoding (`src/output/gst_recorder.py`) | **NOT VERIFIED** | The encoder-selection decision table is unit-tested; `nvv4l2h264enc`/`nvv4l2h265enc` have never encoded a frame. |
| Fallback to CPU/ONNX/software writer | IMPLEMENTED; **MEASURED** | This is the path the dev machine actually takes, and it is what the whole test suite runs on. |

On the development machine, `tests/test_jetson.py` (28 tests) and
`tests/test_backends.py` (36 tests) pass — 64 tests covering pipeline strings,
encoder selection, backend fallback and engine invalidation, all with the
platform mocked.

### What to do first on a real board

Treat the first deployment as a bring-up, not a rollout:

1. `python main.py system info` — confirm what is detected.
2. `python main.py system backends --config configs/jetson_orin_nano_super.yaml`
   — confirm which runtime each model would use.
3. `python main.py models build-tensorrt --config configs/jetson_orin_nano_super.yaml`
   — the first genuinely unverified step.
4. `python main.py benchmark --config configs/jetson_orin_nano_super.yaml --input <clip> --check-targets`
   — produce the first real numbers, then tune the profile from them.

---

## 2. Requirements

| Item | Requirement | Why |
| --- | --- | --- |
| JetPack | **6.x (L4T 36.x) recommended**; JetPack 5.1+ (L4T 35.2+) workable | The L4T→JetPack table in `src/hardware/capabilities.py` maps `35.2`–`35.5` and `36.2`–`36.5`. Anything else reports the raw L4T string; it is not an error, only a less informative report. |
| Python | **3.11 or newer** (`pyproject.toml`: `requires-python = ">=3.11"`) | JetPack 6 ships Python 3.10 as the system interpreter on some images; check with `python3 --version` and install a newer interpreter if needed. |
| OpenCV | **System OpenCV built with GStreamer** | JetPack's `python3-opencv` is built with `-DWITH_GSTREAMER=ON`. A PyPI `opencv-python` wheel is **not**, and installing one disables CSI capture and hardware encoding entirely. |
| TensorRT | JetPack's TensorRT packages (optional) | Needed only for the TensorRT backend. Everything runs without it. |
| `pycuda` | Required **to execute** TensorRT engines | `src/backends/tensorrt_runtime.py` imports `pycuda.driver`; without it, engine execution raises `TensorRTUnavailable` and the system falls back. |

### The OpenCV trap

This is the single most common way to end up with a technically working but
functionally degraded install. Capability detection checks
`cv2.getBuildInformation()` for a `GStreamer:` line that is not `NO`/`OFF`. If
that check fails:

- `source.type: "jetson_camera"` silently degrades to the portable V4L2 webcam
  source (with a logged warning) — no CSI, no NVMM;
- `recording.backend: "auto"` degrades to the OpenCV software writer — no NVENC;
- `system info` prints an explicit note saying so.

A pip `opencv-python` wheel installed inside the virtual environment **shadows**
the JetPack system package. See §3.

---

## 3. Installation

### 3.1 Create the environment with system site packages

JetPack installs OpenCV, TensorRT and CUDA into the system Python. A plain venv
hides all of it, so create the venv with `--system-site-packages`:

```bash
cd /path/to/ReID-project
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import cv2, sys; print(cv2.__version__)"
```

### 3.2 Install the project requirements without shadowing JetPack

`requirements.txt` lists `torch`, `torchvision`, `onnxruntime` and
`opencv-python`. On aarch64 the PyPI versions of all four are wrong for a
Jetson: the torch wheels have no CUDA, `onnxruntime` is CPU-only, and
`opencv-python` has no GStreamer. Exclude them and install the NVIDIA builds
instead:

```bash
grep -vE '^(torch|torchvision|onnxruntime|opencv-python)\b' requirements.txt \
  > /tmp/jetson-requirements.txt
pip install -r /tmp/jetson-requirements.txt
```

### 3.3 torch and torchvision from the NVIDIA Jetson wheel index

Install the Jetson wheels from NVIDIA's index rather than PyPI. The index URL is
specific to the JetPack/CUDA version on the board — take the exact URL from
NVIDIA's current Jetson PyTorch documentation for your JetPack release, then:

```bash
pip install --no-cache-dir --index-url <NVIDIA Jetson wheel index for your JetPack> \
  torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True`. If it prints `False`, the wheel
does not match the installed CUDA runtime and the whole CUDA/TensorRT path will
stay disabled.

### 3.4 onnxruntime-gpu for Jetson

The PyPI `onnxruntime-gpu` wheels are built for x86_64 desktop CUDA. Jetson
needs the aarch64 build published by NVIDIA (Jetson Zoo / the JetPack wheel
index for your release):

```bash
pip install <path or URL to the Jetson onnxruntime-gpu wheel>
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

`CUDAExecutionProvider` (and, where the build provides it,
`TensorrtExecutionProvider`) should appear in the list.

### 3.5 pycuda, only if you intend to run TensorRT engines

```bash
pip install pycuda
```

Building engines does not need it; executing them does.

### 3.6 Guard against a re-introduced opencv-python

`ultralytics` declares `opencv-python` as a dependency, so a later
`pip install ultralytics` or `pip install -r requirements.txt` can pull the
wheel back in and re-break GStreamer. After any install, re-check:

```bash
pip uninstall -y opencv-python opencv-python-headless   # if they reappear
python main.py system info | grep -A1 'OpenCV support'
```

### 3.7 Fetch the face models

```bash
python scripts/fetch_face_models.py
```

---

## 4. Verifying the platform

Two read-only commands. Neither changes `nvpmodel`, `jetson_clocks`, fan curves
or any other system-wide setting, and neither needs root.

```bash
python main.py system info
python main.py system info --json          # machine-readable
```

### 4.1 Field reference for `system info`

| Field | Meaning | If it reads `no` |
| --- | --- | --- |
| `jetson` | Board identified from `/proc/device-tree/model`, `/proc/device-tree/compatible` or `/etc/nv_tegra_release`. | You are not on a Jetson, or the device tree is unreadable. Everything still runs, on the CPU path. |
| `model` / `soc` / `L4T` / `JetPack` | Board identity and release. `JetPack` is derived from the L4T major.minor. | Unknown L4T releases show the raw string; harmless. |
| `CUDA / available` | `torch.cuda.is_available()`, or `nvidia-smi` as a secondary source. | Wrong torch wheel (§3.3), or CUDA not installed. Blocks TensorRT entirely. |
| `CUDA / torch.cuda` | Specifically whether torch sees CUDA. | Reinstall torch from the NVIDIA index. |
| `CUDA / compute capability` | Recorded in engine metadata and used for engine invalidation. | Only populated when torch sees the GPU. |
| `TensorRT / available` | `import tensorrt` succeeded, or a `trtexec` binary was found. | Install JetPack's TensorRT packages. Without it, the ONNX Runtime backend is used. |
| `TensorRT / python bindings` | `import tensorrt` worked. | Not fatal: the builder falls back to `trtexec`. Required to *execute* engines. |
| `TensorRT / trtexec` | Path found on `PATH`, or at `/usr/src/tensorrt/bin/trtexec` or `/usr/local/bin/trtexec`. | Only matters when the Python bindings are missing. |
| `TensorRT / can build engines` | `available` **and** (bindings **or** `trtexec`). | `models build-tensorrt` will refuse. |
| `GStreamer / available` | `gst-launch-1.0 --version` succeeded, or OpenCV reports GStreamer. | Install the GStreamer packages. |
| `GStreamer / OpenCV support` | **The gate for CSI capture and hardware encoding.** | You have a pip `opencv-python` wheel. See §2 and §3.6. |
| `nvarguscamerasrc` | CSI camera element present. | No CSI path. Check the camera ribbon, the sensor overlay and the `nvargus-daemon` service. |
| `nvvidconv` | NVMM conversion element — the gate for zero-copy. | CSI and hardware encode pipelines cannot be built. |
| `nvv4l2h264enc` / `nvv4l2h265enc` | NVENC elements. | Recording falls back to the software encoder with a logged reason. |
| `Derived / NVMM path` | `OpenCV support` **and** `nvvidconv`. | — |
| `Derived / hardware encoder` | `OpenCV support` **and** (h264 **or** h265 element). | — |
| `Derived / CSI camera` | `OpenCV support` **and** `nvarguscamerasrc`. | — |
| `Notes` | Actionable warnings, e.g. "Jetson detected but TensorRT was not found". | Read them; they name the fix. |

### 4.2 Backend selection

```bash
python main.py system backends --config configs/jetson_orin_nano_super.yaml
python main.py system backends -c configs/jetson_orin_nano_super.yaml --json
```

This resolves each configured model to the runtime that would actually execute
it, and prints the reason. It loads no models and builds nothing.

On a Jetson with TensorRT, the expected reason is
`TensorRT available on this Jetson`. Reasons that indicate the TensorRT path is
*not* being taken:

| Reason text | Cause |
| --- | --- |
| `TensorRT is not installed on this system` | `system info` → `TensorRT / available: no`. |
| `TensorRT needs a CUDA device and none was found` | `CUDA / available: no`. |
| `'.pt' cannot be converted to a TensorRT engine; export the model to ONNX first` | `models.detector` still points at a `.pt` file. See §5. |
| `no TensorRT builder (python bindings or trtexec) is available` | `can build engines: no`. |
| `backend.tensorrt.allow_build is false and no prebuilt engine was given` | Immutable-deployment mode with no pre-built engine. |

A `(fallback)` marker means an explicitly configured backend could not be
honoured; the reason states why. `face_detector: opencv` is **not** a fallback —
YuNet is an OpenCV DNN model and OpenCV is its runtime (§12).

> **Reporting caveat.** `select_backend` is generic over roles, so on a machine
> where TensorRT is usable it will report `face_detector: tensorrt` for a
> `.onnx` YuNet model. The face-detector factory ignores that decision and
> always constructs the OpenCV DNN detector, and no YuNet engine is ever
> requested by `build-tensorrt`. The row is informational only; see §12.1.

For reference, the same command on the development machine (no CUDA) prints:

```
  detector      : onnx
    reason : Apple Silicon: MPS/CoreML
  face_detector : opencv
    reason : Apple Silicon: MPS/CoreML
  face_encoder  : onnx
    reason : Apple Silicon: MPS/CoreML
```

---

## 5. Exporting models to ONNX

**TensorRT engines are built from ONNX only.** A `.pt` file cannot be converted
directly; `select_backend` refuses it with an explicit reason, and
`build-tensorrt` will report that there is nothing to build.

The Jetson profile therefore points `models.detector` at an `.onnx` file:

```yaml
models:
  detector: "../models/yolo26n.onnx"
  reid: "../models/yolo26n-reid.onnx"
```

Export the detector once (the `.pt` remains the source of truth; if the `.onnx`
is absent the system falls back to running the `.pt` through PyTorch):

```bash
yolo export model=yolo26n.pt format=onnx dynamic=True opset=17
```

The face models are **already ONNX** and need no export step:

| Model | File | Role |
| --- | --- | --- |
| YuNet | `models/face_detection_yunet_2023mar.onnx` | Face detection — runs on OpenCV DNN (§12) |
| SFace | `models/face_recognition_sface_2021dec.onnx` | Alternative face encoder — OpenCV `FaceRecognizerSF` |
| ArcFace w600k_r50 | `models/w600k_r50.onnx` | Face encoder used by the Jetson profile; this is the one with a TensorRT path |

Fetch them with `python scripts/fetch_face_models.py`, which pulls SCRFD and ArcFace from the InsightFace buffalo_l bundle and YuNet from the OpenCV zoo.

---

## 6. Building TensorRT engines

```bash
python main.py models build-tensorrt --config configs/jetson_orin_nano_super.yaml
```

### 6.1 Flags

| Flag | Effect |
| --- | --- |
| `--config PATH`, `-c PATH` | Configuration file (default `config.yaml`). |
| `--force` | Rebuild even when a valid cached engine exists. |
| `--precision {fp32\|fp16\|int8}` | Overrides `backend.tensorrt.precision` for this run. |
| `--max-batch N` | Overrides `backend.tensorrt.max_batch_size`; also sets `optimal_batch_size` to `min(N, 4)`. |
| `--role NAME` | Build only these roles. Repeatable. Valid values: `detector`, `face_encoder` (face mode), `body_encoder` (body mode). |
| `--json` | Machine-readable output. |

```bash
# Only the face encoder, at FP16, batch ceiling 4
python main.py models build-tensorrt -c configs/jetson_orin_nano_super.yaml \
  --role face_encoder --precision fp16 --max-batch 4

# Force a full rebuild after a JetPack upgrade
python main.py models build-tensorrt -c configs/jetson_orin_nano_super.yaml --force
```

Which engines are built is derived from the configuration, not hard-coded:

| Role | Source | Input shape | Optimisation profile |
| --- | --- | --- | --- |
| `detector` | `models.detector` (only if `.onnx`) | `(3, detector.imgsz, detector.imgsz)` | none — built with `dynamic_batch=False` |
| `face_encoder` | `face.recognition_model` (face mode, only if `.onnx`) | `(3, face.chip_size, face.chip_size)` | min/opt/max from `backend.tensorrt.*_batch_size` |
| `body_encoder` | `models.reid` (body mode, only if `.onnx`) | `(3, reid.size_hw)` | min/opt/max from `backend.tensorrt.*_batch_size` |

A role that fails to build is reported as `FAILED` with its reason, the command
exits non-zero, and **that role falls back to its ONNX/PyTorch backend** — the
system still runs.

### 6.2 Build paths

Two builders, in preference order:

1. **TensorRT Python API** — used when the bindings import. Gives explicit
   control over the optimisation profile, the workspace limit
   (`workspace_mb`), `builder_optimization_level` and the timing cache.
2. **`trtexec`** — used when the bindings are absent, a common state on stock
   JetPack images. Invoked with `--onnx`, `--saveEngine`, `--memPoolSize`,
   the `--minShapes/--optShapes/--maxShapes` triple where applicable, and
   `--fp16`/`--int8`.

If the Python build raises, the builder logs a warning and tries `trtexec`
before giving up.

### 6.3 Engines are derived artefacts

An engine is never the source of truth; the ONNX file is. Each engine is written
to `backend.tensorrt.engine_dir` as
`<stem>.<role>.<precision>.b<max_batch>.engine`, with a JSON sidecar
`<stem>.<role>.<precision>.b<max_batch>.engine.json` recording its provenance:
TensorRT version, CUDA version, JetPack version, GPU name, compute capability,
input shape, batch triple, workspace, optimisation level, source fingerprint and
build time.

Because the build parameters are encoded in the filename, several precisions or
batch ceilings coexist instead of overwriting one another.

**Engines are not portable.** Never copy an engine between machines, and never
commit one. An engine is tied to the exact GPU architecture, TensorRT version
and CUDA version it was built on.

An engine is automatically rebuilt when any of the following changes:

| Trigger | Detected by |
| --- | --- |
| Engine file missing or zero bytes | Direct check |
| Metadata sidecar missing, unreadable or a different schema version | Sidecar load |
| Source model replaced or edited | Source fingerprint (size + mtime + 1 MiB from head and tail) |
| Precision, role, input geometry, batch triple, `workspace_mb` or `builder_optimization_level` changed | Build fingerprint |
| TensorRT version, CUDA version or GPU compute capability changed | Build fingerprint, plus an explicit check when `strict_version_check: true` |

`strict_version_check: true` (the profile default) additionally refuses an
engine whose recorded TensorRT version or compute capability differs from the
running one, and states both values in the reason.

An engine supplied directly (a path ending `.engine`, `.plan` or `.trt` in the
config) is *adopted* rather than validated: its provenance cannot be verified,
it is marked `verified: false` in metadata, it will never be rebuilt
automatically, and a warning is logged.

### 6.4 Inspecting and clearing the cache

```bash
python main.py models list-engines -c configs/jetson_orin_nano_super.yaml
python main.py models list-engines -c configs/jetson_orin_nano_super.yaml --json
```

Prints one row per engine: role, precision, size, the TensorRT version it was
built with, and the filename.

```bash
python main.py models clean-engines -c configs/jetson_orin_nano_super.yaml
python main.py models clean-engines -c configs/jetson_orin_nano_super.yaml --yes
```

Deletes every engine and sidecar in `engine_dir`. Prompts unless `--yes`/`-y` is
given. **Source models are never touched.**

---

## 6b. The passport-photo identity engine on Jetson

The identity stack described in [identity.md](identity.md) becomes the active
path as soon as people are enrolled into the face gallery. Three things about
it are Jetson-specific.

**SCRFD is not accelerated by `models build-tensorrt`.** That command builds
engines through the project's own `TensorRTSession`, which returns a single
output tensor; SCRFD emits nine (three FPN strides × score/bbox/landmark).
It is accelerated instead by ONNX Runtime's TensorRT execution provider,
which handles multi-output graphs and compiles its own engine on first use.
Nothing to run — it happens automatically when `onnxruntime-gpu` exposes
`TensorrtExecutionProvider`. The engine is cached under
`backend.tensorrt.engine_dir/scrfd`.

> **Expect the first load after deployment to take minutes**, not seconds,
> while that engine compiles. Subsequent loads read the cache. Do a warm-up
> run before anything depends on start-up time, and keep the cache directory
> on persistent storage.

**The face encoder *is* built by `models build-tensorrt`.** Both
`face_identity.face_encoder_model` and `face.recognition_model` are covered,
so whichever path is active is accelerated. When they are the same file, one
engine is built.

**The orphan-face scan is the first thing to turn down.** Identifying a face
whose body the person detector missed costs one extra SCRFD pass on frames
where no tracked person is due for recognition. On the reference footage
`orphan_scan_interval: 1` gave 90.4% at 11.4 FPS and `3` gave 89.6% at
16.3 FPS — on desktop hardware, but the shape of the trade holds. The Jetson
profile ships `3`. Setting `recognize_orphan_faces: false` removes the cost
entirely at the price of never identifying a seated or doorway-framed person.

### Thresholds do not travel

Acceptance thresholds are fitted to a camera and a scene. A threshold
calibrated on other footage is worse than none, because it is confidently
wrong rather than visibly absent. The Jetson profile therefore ships **no**
pinned thresholds: until `calibrate` has been run on footage from the
deployed camera, the system uses `fallback_threshold` and says so in the
startup log and in `config validate`.

```
python main.py config validate --config configs/jetson_orin_nano_super.yaml
```

prints which identity path is active, how many people are enrolled, and
whether a calibration exists.

---

## 7. Why INT8 is opt-in

The profile ships `precision: "fp16"` and `allow_int8_for_face: false`, and this
is deliberate.

Quantising a *detector* costs a little accuracy in a bounded, visible way.
Quantising a *face encoder* is different in kind: INT8 shifts the embedding
distribution, which moves every similarity value the system produces — and
therefore invalidates `matching.recognition_threshold` and
`matching.high_confidence_threshold`, which were calibrated against the
FP16/FP32 embeddings. The system does not become obviously broken; it becomes
quietly miscalibrated, accepting or rejecting the wrong people at the old
numbers.

The builder therefore refuses INT8 for the `face_encoder` and `body_encoder`
roles unless it is enabled explicitly, and warns even then.

### 7.1 What enabling INT8 requires

| Requirement | Key |
| --- | --- |
| Precision set to INT8 | `backend.tensorrt.precision: "int8"` |
| Explicit face-encoder consent | `backend.tensorrt.allow_int8_for_face: true` |
| A calibration image directory | `backend.tensorrt.int8_calibration_dir: "<path>"` |
| Platform support | The build fails with `this platform has no fast INT8 support` if the GPU lacks it |

`int8_calibration_dir` is validated at configuration load: `precision: "int8"`
without it is a configuration error. The directory must contain representative
images — the same kind of faces, lighting and scale the deployment will see.

### 7.2 Re-calibration is mandatory, not advisory

After building INT8 engines, the thresholds must be re-derived on your own data:

```bash
python main.py evaluate --config configs/jetson_orin_nano_super.yaml \
  --dataset <evaluation dataset directory> --output data/output/int8-eval.json
```

`evaluate` reports the similarity distributions and candidate thresholds for the
model that is actually loaded. Write the resulting values back into
`matching.recognition_threshold` and `matching.high_confidence_threshold` before
deploying. Skipping this step is the failure mode the guard exists to prevent.

---

## 8. Camera setup

`source.type: "jetson_camera"` selects the accelerated capture source. Three
paths exist, chosen from what the board exposes:

| Path | Elements | When |
| --- | --- | --- |
| CSI/MIPI | `nvarguscamerasrc` → NVMM → `nvvidconv` → `appsink` | `csi: true`, or `csi: "auto"` with `nvarguscamerasrc` present |
| USB/V4L2, accelerated | `v4l2src` → `jpegdec` → `nvvidconv` → `appsink` | `csi: false`, or `auto` without `nvarguscamerasrc` |
| Plain V4L2 | OpenCV `VideoCapture` | OpenCV has no GStreamer support — the source factory substitutes the portable webcam source with a warning |

### 8.1 Configuration keys

All under `source:`:

| Key | Type / range | Meaning |
| --- | --- | --- |
| `type` | `jetson_camera` | Selects this source. |
| `csi` | `true` / `false` / `"auto"` | `auto` uses CSI when `nvarguscamerasrc` exists, else V4L2. |
| `sensor_id` | int ≥ 0 | CSI sensor index passed as `sensor-id=`. |
| `device` | int or path | V4L2 device; a bare integer becomes `/dev/video<N>`. |
| `width` / `height` | int | **Delivered** frame size — what the pipeline hands to Python. |
| `fps` | number | Requested framerate (`framerate=<fps>/1`). |
| `flip_method` | int 0–7 | `nvvidconv flip-method=` (rotation/mirroring). |
| `capture_width` / `capture_height` | int, optional | **Sensor** capture geometry when it differs from the delivered size. |
| `gst_pipeline` | string, optional | A complete manual pipeline. **Overrides everything else.** |

### 8.2 Why `capture_width`/`capture_height` exist

A sensor that natively delivers 1920×1080 can be scaled to 1280×720 *inside*
GStreamer, on the VIC, in the same pass as the colour conversion — instead of
handing Python a full-resolution frame that then gets resized again in NumPy for
inference. Set the sensor geometry once and let the hardware do the scaling:

```yaml
source:
  type: "jetson_camera"
  csi: "auto"
  sensor_id: 0
  width: 1280           # delivered to the pipeline
  height: 720
  fps: 30
  flip_method: 0
  capture_width: 1920   # sensor mode
  capture_height: 1080
```

### 8.3 Bounded buffers and low latency

Both generated pipelines end with:

```
appsink drop=true max-buffers=1 sync=false
```

This is a deliberate latency decision. With an unbounded sink, a pipeline whose
consumer is slower than the camera accumulates a queue, and the displayed frame
drifts further behind reality the longer the process runs. With
`drop=true max-buffers=1`, the camera discards frames the consumer could not
keep up with, and the frame you get is always the newest one. **A live view that
is three seconds late is worse than one that skipped frames.**

The same reasoning drives the shallow queues in the profile
(`pipeline.capture_queue_size: 2`, `drop_stale_frames: true`).

### 8.4 Testing a pipeline by hand

When capture fails, the error message includes the exact pipeline with the
`appsink` swapped for `fakesink`, so it can be pasted straight into
`gst-launch-1.0`. To test a CSI pipeline manually:

```bash
# Minimal sanity check
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12' ! \
  nvvidconv flip-method=0 ! 'video/x-raw,width=1280,height=720,format=BGRx' ! \
  videoconvert ! 'video/x-raw,format=BGR' ! fakesink

# Which elements exist at all
gst-inspect-1.0 nvarguscamerasrc
gst-inspect-1.0 nvvidconv

# USB camera formats
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

The V4L2 pipeline this system generates expects **MJPEG** from the camera
(`image/jpeg` → `jpegdec`). A camera that only offers raw YUYV at the requested
size will not negotiate; use `gst_pipeline` to write a pipeline that matches the
camera, or fall back to `source.type: "webcam"`.

### 8.5 Escape hatch

If the generated pipeline does not suit the sensor, supply a complete one. It is
used verbatim, and every other key above is ignored:

```yaml
source:
  type: "jetson_camera"
  gst_pipeline: >-
    nvarguscamerasrc sensor-id=0 !
    video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 !
    nvvidconv ! video/x-raw,format=BGRx !
    videoconvert ! video/x-raw,format=BGR !
    appsink drop=true max-buffers=1 sync=false
```

Keep the `appsink` bounded, or the latency argument in §8.3 no longer applies.

---

## 9. Hardware recording

On a Jetson the video encoders are fixed-function blocks. Using them frees both
the CPU and the GPU for inference, which is the point on a board where both are
scarce.

### 9.1 Configuration keys

| Key | Values | Meaning |
| --- | --- | --- |
| `recording.backend` | `auto` / `opencv` / `gstreamer` | `auto` uses NVENC when GStreamer exposes it, otherwise OpenCV's software writer. `opencv` pins the software writer. `gstreamer` requests hardware and **warns loudly** if it is unavailable — it still falls back. |
| `recording.codec` | `h264` / `h265` / `mp4v` | Used by the GStreamer backend (`nvv4l2h264enc` / `nvv4l2h265enc`). `mp4v` has no hardware encoder. The OpenCV backend uses `output.codec` instead. |
| `recording.hardware_acceleration` | bool | With `backend: auto`, this is what asks for hardware at all. |
| `recording.bitrate_kbps` | 100–100000 | Encoder bitrate; passed to the element as bits per second. |

Profile defaults: `backend: "auto"`, `codec: "h264"`,
`hardware_acceleration: true`, `bitrate_kbps: 4000`.

### 9.2 The pipeline

```
appsrc → videoconvert → I420 → nvvidconv → NVMM/NV12 →
  nvv4l2h264enc|nvv4l2h265enc → h264parse|h265parse → qtmux → filesink
```

### 9.3 Fallback is mandatory, not optional

Losing a recording entirely is a worse failure than encoding it on the CPU, so
the software writer is always available and the reason is always logged:

| Condition | Resulting backend | Logged reason |
| --- | --- | --- |
| `backend: opencv` | Software | `explicitly configured` |
| `backend: auto` and `hardware_acceleration: false` | Software | `hardware acceleration not requested` |
| OpenCV built without GStreamer | Software | `this OpenCV build has no GStreamer support` |
| `codec: h265`, element missing | Software | `nvv4l2h265enc is not available` |
| `codec: h264`, element missing | Software | `nvv4l2h264enc is not available` |
| `codec: mp4v` | Software | `mp4v has no hardware encoder; use h264 or h265` |
| GStreamer writer fails to open at runtime | Software, per clip | Warning naming the file |
| All elements present | **Hardware** | `NVIDIA hardware encoder available` |

When `backend: gstreamer` was requested explicitly and hardware is unavailable,
the fallback is logged at **warning** level rather than debug — an explicit
request that could not be honoured should be noisy.

Event clips, pre/post-roll ring buffering and continuous recording are unchanged
by any of this; only the encoder underneath differs.

---

## 10. Fallback behaviour

**Nothing about Jetson is mandatory, and nothing about it is hard-coded in the
pipeline.** `configs/jetson_orin_nano_super.yaml` is ordinary configuration: it
loads and runs on a laptop, falling back to whatever that machine has. That is
how the profile is tested today.

| Missing | Detector | Face detector | Face encoder | Capture | Recording |
| --- | --- | --- | --- | --- | --- |
| **TensorRT** | `.onnx` → ONNX Runtime; `.pt` → PyTorch | OpenCV DNN (unaffected) | ONNX Runtime (ArcFace) / OpenCV DNN (SFace) | unaffected | unaffected |
| **CUDA** | As above; `device: auto` resolves to CPU (or MPS on Apple Silicon) | OpenCV DNN | ONNX Runtime on CPU | unaffected | unaffected |
| **GStreamer in OpenCV** | unaffected | unaffected | unaffected | `jetson_camera` → portable V4L2 webcam source, with a warning | software encoder (`output.codec`) |
| **`nvarguscamerasrc`** | unaffected | unaffected | unaffected | `csi: auto` → V4L2 path; `csi: true` → actionable `SourceError` | unaffected |
| **`nvv4l2h26xenc`** | unaffected | unaffected | unaffected | unaffected | software encoder, reason logged |
| **`pycuda`** | Engine execution raises `TensorRTUnavailable` → ONNX Runtime | OpenCV DNN | ONNX Runtime | unaffected | unaffected |

Every fallback states its reason. A deployment that silently runs five times
slower is worse than one that says why, so backend fallbacks are logged at
warning level with the requested backend, the backend in use and the cause.

Two config switches make fallbacks deliberate rather than automatic:

- `backend.tensorrt.enabled: false` — never use TensorRT, regardless of
  hardware.
- `backend.tensorrt.allow_build: false` — never build an engine at runtime.
  Missing engines fall back instead. Use this for immutable deployments where
  engines are built ahead of time.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `TensorRT is not available on this system, so no engine can be built here` from `models build-tensorrt` | `system info` reports `TensorRT / available: no` — the JetPack TensorRT packages are missing, or the venv cannot see them. | Install JetPack's TensorRT packages. Recreate the venv with `--system-site-packages` (§3.1). Confirm with `python -c "import tensorrt; print(tensorrt.__version__)"`. Until then, the ONNX Runtime backend works. |
| `system info` shows `TensorRT available: yes` but `can build engines: no` | Neither the Python bindings nor `trtexec` was found. | `pip install` is not the route on Jetson — install `python3-libnvinfer` / `python3-libnvinfer-dev` from JetPack, or make `trtexec` reachable (JetPack puts it at `/usr/src/tensorrt/bin/trtexec`). |
| `GStreamer / OpenCV support: no`, and the `system info` note about `-DWITH_GSTREAMER=ON` | A pip `opencv-python` wheel is shadowing JetPack's `python3-opencv`. | `pip uninstall opencv-python opencv-python-headless`; recreate the venv with `--system-site-packages`; install requirements with `opencv-python` excluded (§3.2). Re-check `system info`. |
| `source.type='jetson_camera' needs an OpenCV built with GStreamer support` | Same as above, reached via the camera source directly. | As above, or set `source.type: "webcam"` to use the portable V4L2 path. |
| `cannot open the Jetson camera (csi mode)` | The pipeline failed to negotiate: wrong sensor mode, wrong `capture_width`/`capture_height`/`fps` combination, camera not detected, or Argus not running. | The error message embeds the exact pipeline with `fakesink` substituted — run it under `gst-launch-1.0` and read GStreamer's own error. Check `ls /dev/video*`, then try a documented sensor mode. |
| `a CSI camera was requested but nvarguscamerasrc is not available` | `csi: true` with the element missing. | Check the ribbon cable seating and the device-tree overlay for the sensor; verify `gst-inspect-1.0 nvarguscamerasrc`; restart `nvargus-daemon`. Or set `csi: false` for a USB camera. |
| `the Jetson camera opened but delivered no frames` | The pipeline negotiated but the sensor produced nothing — frequently a stuck Argus daemon. | `sudo systemctl restart nvargus-daemon`, then retry. If it recurs, check `journalctl -u nvargus-daemon`. Only one process may hold the CSI camera at a time — close any other consumer (including a stray `gst-launch-1.0`). |
| `cannot open the Jetson camera (v4l2 mode)` | The generated V4L2 pipeline expects MJPEG (`image/jpeg` → `jpegdec`); the camera may only offer raw YUYV at that size. | `v4l2-ctl -d /dev/video0 --list-formats-ext` to see what it offers, then write a matching `source.gst_pipeline`, or use `source.type: "webcam"`. |
| Engines rebuild on **every** start | Something in the fingerprint changes each run: the source model's mtime is rewritten by a sync/build step, `engine_dir` is not persistent, or the config differs from the one used to build (precision, batch triple, `workspace_mb`, `builder_optimization_level`). | `models list-engines` and compare the sidecar JSON against the running config. Ensure `engine_dir` is a persistent path (the profile uses `../models/engines`). Use the *same* config file for build and run. |
| Engine rebuilds after a JetPack or driver upgrade | Expected. The TensorRT version, CUDA version or compute capability changed, which invalidates the engine. | Rebuild once: `models build-tensorrt --force`. This is correct behaviour, not a bug — a stale engine is a correctness hazard. |
| `could not deserialize <name>.engine` at load | The engine was copied from another machine, or built against a different TensorRT version with `strict_version_check: false`. | Engines are never portable. `models clean-engines --yes`, then `models build-tensorrt --force` on this board. Keep `strict_version_check: true`. |
| `ONNX parse failed ...: The model may use an operator this TensorRT version does not implement` | An operator in the exported graph is unsupported by this TensorRT version. | Re-export with a lower `opset` (e.g. `opset=16`), or pin that model to ONNX Runtime: `backend.face_encoder: onnx` / `backend.detector: onnx`. The pipeline is unchanged; only the runtime differs. |
| `trtexec failed ... (exit N)` naming an unsupported operator | As above, via the `trtexec` path. | Same fix. The message itself suggests `backend.face_encoder: onnx`. |
| Out of memory during the engine build; the build is killed; the board freezes | The Orin Nano shares 8 GB between CPU and GPU, and the builder's workspace competes with everything else. | Reduce `backend.tensorrt.workspace_mb` (profile default 1024; try 512 or 256) and `backend.tensorrt.max_batch_size` (which also lowers `optimal_batch_size`). Build one role at a time with `--role`. Close the desktop session and other GPU consumers. Add swap if the build is being OOM-killed rather than failing inside TensorRT. |
| `TensorRT failed to build an engine for <name>` with no operator named | Frequently the same resource problem as above. | Same remedies. `models build-tensorrt` logs at INFO and the TensorRT builder writes its own warnings to the console; capture the full output when reporting the failure. |
| `pycuda is required to run TensorRT engines` | Engines were built, but `pycuda` is not installed. | `pip install pycuda`. Until then, the encoder falls back to ONNX Runtime. |
| `INT8 was requested ... allow_int8_for_face is false` | Deliberate guard. | Read §7 before enabling it; re-calibrate afterwards. |
| `tensorrt.precision='int8' requires tensorrt.int8_calibration_dir` at config load | INT8 without calibration data. | Point `int8_calibration_dir` at representative images, or return to `fp16`. |
| Recording files appear but the CPU is saturated | The software encoder is in use. | Check the logs for the reason line, and `system info` for `hardware encoder`. Usually `codec: mp4v`, a missing `nvv4l2h26xenc`, or OpenCV without GStreamer. |
| Throughput is far below expectation | Nothing in this repository has ever been profiled on a Jetson (§1). | Measure before tuning: `python main.py benchmark -c configs/jetson_orin_nano_super.yaml --input <clip> --check-targets`, then sweep with `python main.py benchmark-suite matrix --input <clip> -g batch`. Generate multi-person clips first with `python scripts/build_crowd_clips.py --people 1 3 5 10 --resolution 1280x720`, since a two-person clip tells you nothing about a batch size of 8. |

---

## 12. Unsupported and known gaps

### 12.1 YuNet and SFace have no TensorRT path

Face detection always executes on OpenCV DNN. YuNet is an OpenCV DNN model —
OpenCV *is* its runtime, there is no separate ONNX session, and no TensorRT
engine is ever requested for it (`default_build_requests` only ever emits
`detector`, `face_encoder` and `body_encoder`, and the face-detector factory
constructs `YuNetFaceDetector` unconditionally). This is not a fallback and not
a defect; it is what the model is.

The one rough edge is cosmetic: the generic backend selector will *report*
`tensorrt` for the `face_detector` role on a TensorRT-capable machine, because
the model file is an `.onnx`. Nothing acts on that decision. Treat the
`face_detector` row of `system backends` as informational.

SFace (`face_recognition_sface_2021dec.onnx`) is loaded through OpenCV's
`FaceRecognizerSF`, which performs its own preprocessing. The TensorRT encoder
implements ArcFace-style preprocessing specifically (same resize, same BGR→RGB,
same `(x - mean) / scale`) so that the engine swap is numerically neutral apart
from FP16 rounding. **Only the ArcFace encoder (`w600k_r50.onnx`, the profile
default) has a TensorRT path.** If you switch `face.recognition_model` to SFace,
expect OpenCV DNN, not TensorRT.

Consequence on a Jetson: face *detection* stays on the CPU even when everything
else is accelerated. Budget for it, and note that
`face.adaptive_search.enabled` is off in the profile precisely because
whole-frame YuNet at 1280×720 is not automatically cheaper than several small
ROI passes — that trade-off needs measuring on the target.

### 12.2 Engines are not portable, ever

A TensorRT engine is tied to the GPU architecture, TensorRT version and CUDA
version it was built on. Do not copy engines between boards, do not bake them
into an image built on a different machine, do not commit them. Build on the
target. `strict_version_check: true` exists to catch violations of this, and
should stay on.

For immutable deployments, build engines during image creation *on the same
board type and software stack*, then set `backend.tensorrt.allow_build: false`
so nothing is built at runtime.

### 12.3 `pycuda` is required to execute engines

Building an engine needs the TensorRT Python bindings or `trtexec`. *Running*
one additionally needs `pycuda`, which JetPack does not install. Without it,
engine execution raises `TensorRTUnavailable` and the encoder falls back to ONNX
Runtime — correct, but not what you built engines for.

### 12.4 Other open items

| Item | Note |
| --- | --- |
| Detector optimisation profile | The detector build request sets `dynamic_batch=False`, so no optimisation profile is added, while the recommended export uses `dynamic=True`. Whether a dynamic-batch detector ONNX builds cleanly without a profile on a real TensorRT has not been exercised. If the build fails on shape grounds, export the detector with a fixed batch dimension. |
| INT8 calibration | `ImageFolderCalibrator` has never been run against a real TensorRT calibrator interface. |
| Power modes | Nothing in this system touches `nvpmodel`, `jetson_clocks` or fan curves. Capability detection is strictly read-only and non-privileged. The deployment chooses its own power profile — set it yourself before benchmarking, and record which mode a measurement was taken in. |
| Profile values | Every number in `configs/jetson_orin_nano_super.yaml` — batch sizes, queue depths, scheduler intervals, `imgsz`, and the `benchmark.targets` block — is an engineering default or a target chosen for an 8 GB Orin Nano, not a measurement. Re-derive them from `benchmark-suite matrix` on the board. |
| Multi-camera | One camera per process. No multi-sensor Argus pipeline is implemented. |
