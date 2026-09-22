# Identity from a passport photograph

How this system answers "who is this?" when the only thing it was given about
a person is one portrait photograph, and the person it later sees may be
older, wearing glasses or a mask, half-turned, badly lit, further away, or
partly hidden.

---

## 1. Why the face, and not the body

The system enrols from a passport-style photograph. That photograph shows a
head and shoulders. Whole-body appearance — clothing, build, gait — is simply
not in it, and clothing changes daily even when it is.

This is not an assumption. It was measured on this project's own evaluation
set, enrolling from the passport photo and querying with the degraded
conditions:

| Identity signal | Rank-1 | Genuine min − impostor max |
|---|---|---|
| Body appearance (YOLO26-reid on the person box) | 98.3 % | **−0.445** |
| Face (SCRFD + ArcFace) | 100 % | **+0.392** |

Rank-1 says body appearance looks excellent. The separation column says it is
unusable. A negative separation means the *worst* genuine pair scores below
the *best* impostor pair: the two distributions overlap, so no threshold
exists that both accepts the registered person and rejects the stranger. Any
threshold you pick trades one error for the other.

That is the whole argument for the architecture. Rank-1 alone would have
hidden it, which is why every measurement in this system reports both.

> Body appearance is still used — by the tracker, for keeping a person's box
> attached to them between frames. It is never the identity.

---

## 2. The pipeline

```
frame
  → YOLO26 person detection          detector_confidence
  → person tracking (BoT-SORT)       track_id
  → face detection (SCRFD)           face_detection_confidence
  → face alignment (5-point, 112×112 ArcFace template)
  → face quality + visibility        face_quality, visibility
  → face embedding (ArcFace, 512-D, L2-normalised)
  → identity gallery search          face_similarity
  → identity matching                threshold by visibility class
  → temporal evidence                quality-weighted, per track
  → final identity decision          identity_id, identity_confidence
```

Face detection runs **once per frame** on the whole image, and each face is
then assigned to the person box containing its centre. With several people in
view that is one detector call instead of one per person. A face that belongs
to no detected person is dropped rather than promoted into a detection of its
own: the system identifies *people*, and a face with no person behind it has
no track to accumulate evidence on.

### Quantities that are never merged

Seven distinct numbers travel through the pipeline, and the code never
collapses them into a single "confidence":

| Name | What it is | What it is not |
|---|---|---|
| `detector_confidence` | YOLO26's score that this box is a person | not a statement about who it is |
| `face_detection_confidence` | SCRFD's score that this is a face | not a statement about whose face |
| `face_quality` | how much identity information this face carries, [0, 1] | not a similarity |
| `face_similarity` | cosine between two L2-normalised embeddings, [−1, 1] | **not a probability, and not a percentage** |
| `track_id` | which trajectory this is | not an identity |
| `identity_id` | who the system says this is | — |
| `identity_confidence` | calibrated P(the identity claim is correct) | `None` when no calibration has been fitted |

A cosine similarity of 0.62 does not mean "62 % sure". The only quantity in
this system that may be read as a probability is `identity_confidence`, and it
exists only after `calibrate` has been run on labelled data. With no
calibration the field is `None` — never a number invented from the similarity.

---

## 3. Face detection: why SCRFD

Two detectors were compared end to end. Recall differed by a single detection
across 32 images, which is inside the noise. What separated them was what
happened *downstream* of their landmarks:

| Detector | Genuine min | Impostor max | Separation |
|---|---|---|---|
| YuNet | −0.027 | 0.152 | **−0.179** |
| SCRFD | 0.481 | 0.072 | **+0.409** |

Both detectors find faces. SCRFD's landmarks align them well enough that the
embeddings of the same person stay together. YuNet's do not, and the
distributions overlap — the same failure mode as body appearance, from a
different cause.

The lesson is that a face detector must be judged by the identity separation
it produces, not by its detection recall. YuNet remains available
(`face_identity.face_detector_backend: yunet`) as a faster option where that
trade is acceptable and has been re-measured.

---

## 4. Alignment

Five landmarks (eyes, nose, mouth corners) are fitted onto the canonical
ArcFace 112×112 template with a **similarity** transform — rotation, uniform
scale, translation. Not a full affine: an affine fit would also shear, which
distorts facial geometry to match the template rather than correcting for pose.

Alignment is what makes a face at 10 m and the same face at 2 m produce the
same embedding. Without it, scale and head roll dominate the comparison.

A face with no landmarks is not aligned and not embedded. The fallback crop
exists for display only.

---

## 5. Quality and visibility

Quality is continuous, not a pass/fail flag, and is a weighted combination of:

| Component | Weight | Measured from |
|---|---|---|
| resolution | 0.34 | inter-ocular distance in pixels (floor 12 px, saturating at 55 px) |
| sharpness | 0.20 | Laplacian variance of the aligned chip |
| pose | 0.18 | yaw and pitch from landmark geometry |
| exposure | 0.12 | histogram spread, penalising crush and clipping |
| landmarks | 0.08 | fit residual against the template |
| visibility | 0.08 | the class below |

Visibility is classified as `full_face`, `masked`, `partial_face`,
`heavily_occluded` or `no_usable_face`, and drives which threshold family is
used.

**Occlusion is detected from structure, not from skin colour.** A skin-tone
test — the conventional YCrCb range — was implemented first and rejected:
it reported darker-skinned faces as occluded and was defeated by a beard. The
replacement measures local gradient energy and chroma dispersion, both of
which separate skin from fabric regardless of the skin's tone. This is
asserted by a test that runs the same face across five skin tones and requires
the same verdict for all of them.

A geometric cross-check catches the case where a scarf makes the detector box
only the visible upper face: a box shorter than 1.75× the inter-ocular
distance cannot be showing a whole face, whatever the pixels inside it say.

---

## 6. Matching and thresholds

Each identity holds one or more embeddings — the passport reference, plus any
adapted observations. A query is scored against **every** stored embedding and
the identity's score is the **best** of them, not the mean. Averaging would
blur exactly the variation the extra embeddings were stored to capture.

Acceptance requires, in order:

1. a usable face at all (otherwise `NO_FACE` — *not* `UNKNOWN`);
2. quality at or above the floor;
3. a non-empty gallery;
4. best similarity ≥ the threshold **for this visibility class**;
5. a margin over the runner-up (otherwise `AMBIGUOUS_IDENTITY`);
6. calibrated confidence above its floor, when one is configured;
7. enough accumulated temporal evidence to name anybody.

Thresholds default to `None`, meaning "use the calibrated value". Pinning one
in the configuration is only appropriate when it came from measurement on your
own data, and the schema warns when you do.

### Masked and partial faces get their own thresholds

A masked face carries less information, so its genuine scores sit lower. Using
the full-face threshold on it would reject the right person; lowering the
full-face threshold to compensate would accept strangers. The two are
calibrated separately and applied by visibility class.

---

## 7. Calibration

`python main.py calibrate --dataset <dir>` fits, per visibility class:

- a **logistic (Platt) mapping** from cosine similarity to probability, with
  target smoothing, standardised inputs and light L2 — so `identity_confidence`
  means something;
- an **isotonic (PAV) mapping** as an alternative when the relationship is not
  logistic;
- an **operating threshold**.

The threshold is chosen as follows, and the order matters:

1. If the genuine and impostor distributions do not overlap at all, use the
   **midpoint of the gap**. An equal-error threshold in that situation sits at
   the very bottom edge of the gap — mathematically optimal on the sample,
   and one unlucky impostor away from a false accept in the field.
2. Otherwise take the threshold meeting `max_far`, subject to a `min_tar`
   floor.
3. If no threshold meets both, fall back to the equal-error point and say so.

Step 3 exists because of a real failure: with only 144 impostor samples, a
`FAR ≤ 1 %` constraint permits about 1.4 false accepts, which drove the
threshold to 0.997 — a system with 96.7 % rank-1 accuracy and 67.5 % false
rejection. The constraint was satisfiable and useless. The `min_tar` guard
turns that into a visible fallback rather than a silent catastrophe.

Fewer than 12 samples in either class produces **no calibration at all** for
that condition, and `identity_confidence` stays `None`.

---

## 8. Temporal evidence

A single frame never names anybody. Each track accumulates quality-weighted
evidence per candidate identity, and the state machine moves it through:

```
NEW_TRACK → UNCONFIRMED → RECOGNIZED ⇄ TEMPORARILY_OCCLUDED
                 ↓                              ↓
              UNKNOWN  ←──────────────────── (hold expires)
```

- **Naming requires repetition.** `min_confirmation_frames` observations that
  each cleared the threshold, weighted by their quality.
- **A hidden face is not evidence against an identity.** Someone turning their
  head is not evidence that they are someone else; it only starts the
  occlusion clock.
- **An occluded, already-confirmed identity is held** for
  `occlusion_hold_frames` and then released. The hold is bounded and logged.
- **A track that never earned an identity can never acquire one this way.** A
  masked stranger who walks in and is tracked for a thousand frames is never
  named. This is asserted directly by a test.
- **Switching identity needs a clear margin**, or two similar-looking people
  swap labels frame to frame.

`UNCERTAIN` — a candidate leads but the evidence is not yet sufficient — is
reported as unknown downstream, because nobody has been named. The nuance
survives in the decision's reason text.

---

## 9. Online adaptation

Disabled by default. When enabled, a recognition may add its embedding to the
matched identity, so the gallery grows beyond the single passport photo.

Every gate below must pass, and each rejection is recorded with its reason:

| Gate | Why |
|---|---|
| track is confirmed, with N confirmations | one good frame is not evidence |
| track stability above a floor | a churning track is not trustworthy |
| similarity ≥ calibrated threshold **+ headroom** | a marginal accept must never become a stored truth |
| quality above a floor | a poor chip poisons every future comparison |
| margin over the runner-up | a contested match is the contamination route |
| full face only | never extend an identity from a masked or partial view |
| novelty inside [0.02, 0.45] | too similar adds nothing; too different is probably not them |
| cooldown, and a per-track cap | one track cannot flood an identity |

The passport reference is **never** evicted, whatever the capacity limit: the
one embedding whose provenance is certain is the one that stays.

---

## 10. What the numbers mean

There are two evaluation sets, and they answer different questions.

### 10a. Real footage — what the system actually does

Two people enrolled from **genuine passport photographs**, then identified in
footage of them walking, talking, sitting and working. One of them wears
glasses throughout the video and wears none in his passport photo, so the
glasses condition here is real rather than drawn on.

The dataset is built by `scripts/build_video_dataset.py` from 261 face
detections that a **human labelled by eye** — never the system being measured,
which would make the evaluation circular. Thresholds are fitted on
temporally disjoint blocks of the video, with a guard band so that no test
image is within 0.3 s of a calibration image.

Held-out test split, 99 queries, thresholds calibrated on the other blocks:

| Condition | n | accuracy | rank-1 | FAR |
|---|---|---|---|---|
| frontal | 45 | 100 % | 100 % | 0 % |
| distant (face < 85 px) | 12 | 100 % | 100 % | 0 % |
| turned (head rotated away) | 39 | 87.2 % | 94.9 % | 0 % |
| false detections (must be rejected) | 3 | 100 % | — | 0 % |
| **overall** | **99** | **94.9 %** | **97.9 %** | **0 %** |

Unknown rejection 100 %. Precision / recall / F1: 1.000 / 0.948 / 0.973.
**Zero identity errors**: the system never named one registered person as the
other. Every failure is a refusal on a near-profile face, and those score
0.04–0.11 against an impostor maximum of 0.14 — accepting them would mean
accepting false matches, so declining is correct.

### 10b. The same footage through the live pipeline

The table above scores cropped queries. Running the whole video through
`main.py video` and matching the output against the same labels, across 627
frames:

| Matching rule | correct | named wrong | unnamed |
|---|---|---|---|
| face box overlaps the labelled face (strict) | **90.4 %** | 1 | 9 |
| the labelled face lies in the detection's person box | **94.8 %** | 1 | 11 |

The two rules differ because between scheduled recognition passes a track
redraws its face box where the face was last seen, so the box is stale even
though the identity is current. The person box is recomputed every frame,
which makes the second rule the fairer test of "was this person named, and
named correctly". Both clear 90 %.

### What this establishes, and what it does not

**Established.** One passport photograph per person is enough to identify them
in real footage across pose, lighting, distance, partial occlusion and
glasses-not-in-the-reference. Genuine and impostor scores separate cleanly on
real data. Thresholds fitted on one part of the footage transfer to another.
The system refuses rather than guesses when the face carries no signal, and it
never confused the two registered people.

**Not established.**

- **Scale.** Two registered identities (four with the demo subjects). Impostor
  scores rise with gallery size, which pushes the threshold up, which costs
  recall. Nothing here predicts behaviour at fifty identities.
- **Unknown *people*.** The unknown split holds false face detections — hands,
  a dark doorway — not unregistered humans, because no third person's face is
  resolvable in this footage. Rejecting a hand is much easier than rejecting a
  stranger.
- **Ageing.** The passport photographs and the video are close in time. A
  reference from years earlier is **untested**.
- **Demographics.** Two men of similar background. Nothing here says anything
  about performance across skin tones, ages or sexes.

A deployment-grade figure needs at least 30 identities, genuine unregistered
people, and references separated from the queries by years.

### 10c. Synthetic footage — pipeline validation only

`data/evaluation` is built by `scripts/build_evaluation_dataset.py` from four
faces with drawn-on glasses and masks, warps, gamma and blur. It scores
**97.0 %** overall, and that number should not be quoted as accuracy: every
condition is a transformation of the same few photographs. It is useful for
exactly two things — checking the pipeline is wired correctly, and A/B
comparison between configurations.

On it, the A/B ladder from `B_face_only` to `F_full_system` scores
**identically**. On data that easy, calibration, quality weighting and
conditional thresholds cannot be shown to pay for themselves; they are
justified by the failure modes they prevent.

---

## 10d. What was measured and rejected

Changes that looked obviously right, were implemented, measured, and then
removed because the data disagreed. They are recorded here so nobody spends
the afternoon again.

| Change | Expected | Measured | Verdict |
|---|---|---|---|
| Box-framed crop instead of the warp when the five-point fit degenerates on a profile | recover turned faces | rank-1 **fell** 100 % → 94.8 %, genuine p5 0.244 → 0.040 | **rejected** — the encoder is trained on warped chips, so even a bad warp beats a plain crop |
| Mirror the reference embedding | cover the other profile | genuine p5 0.244 → 0.247, but impostor max 0.184 → **0.193**; end-to-end identical at 96.0 % | **rejected** — doubles the gallery and narrows the impostor margin for no gain at the operating point |
| Flip the query and take the best score | same | helps only at thresholds above the calibrated one; identical at 0.175 | **rejected** at this gallery size |

---

## 11. Commands

```bash
# Enrol from passport photographs listed in config.yaml
python main.py identity build

# ... or from a directory of <person_id>/<photo>
python main.py identity build --from-dir data/evaluation/enrollment

python main.py identity list                    # who is registered, with warnings
python main.py identity remove <id>

python main.py calibrate --dataset data/evaluation
python main.py evaluate-faces --dataset data/evaluation --output report.json
python main.py experiments --dataset data/evaluation
```

Once the face gallery has people in it, it becomes the identity path for
`run` / `video` / `webcam` / `image` automatically. Startup logs
`identity=face_identity` when it does, and `system info` reports which path is
active.

---

### Building an evaluation set from your own footage

```bash
# 1. Extract every face the detector finds, and label them by eye.
#    The labels file is a list of {frame, bbox, label}; label is a person id,
#    "not_a_face" for a false detection, or "ambiguous" to exclude.
#
# 2. Turn video + labels into an evaluation dataset.
python scripts/build_video_dataset.py \
    --video data/demo/test_video/test.mp4 \
    --labels data/demo/test_video/test_labels.json \
    --enrollment data/input \
    --output data/evaluation_video

# 3. Fit thresholds on part of it, measure on the rest.
python main.py calibrate --dataset data/evaluation_video
python main.py evaluate-faces --dataset data/evaluation_video
```

Query images are named `<block>__<frame>.jpg`, and the splitter keeps a block
whole. Two frames a tenth of a second apart are the same photograph for this
purpose; splitting them across calibration and test would measure
memorisation rather than accuracy.


## 12. Configuration

See `face_identity`, `temporal`, `online_adaptation` and `reference_quality`
in `config.yaml`. The settings most worth knowing:

| Setting | Default | Effect |
|---|---|---|
| `face_identity.face_detector_backend` | `scrfd` | `yunet` is faster and measurably worse at separation |
| `face_identity.threshold_*` | `None` | `None` means "use the calibrated value" |
| `face_identity.min_face_quality` | 0.30 | below this, `NO_FACE` rather than a guess |
| `temporal.min_confirmation_frames` | 4 | how much evidence before naming |
| `temporal.occlusion_hold_frames` | 45 | how long a confirmed identity survives a hidden face |
| `face_identity.recognize_orphan_faces` | `true` | identify a visible face whose body the person detector missed |
| `face_identity.orphan_scan_interval` | 3 | how often to look for those; 1 is more accurate and slower |
| `recognition_scheduler.low_confidence_interval` | 3 | passes while a track is confirming; 1 removes the ramp delay |
| `temporal.strong_evidence_weight` | 1.6 | accumulated weight that confirms in 2 frames instead of 4 |
| `online_adaptation.enabled` | `false` | gallery growth, off until deliberately enabled |
| `reference_quality.fail_on_warnings` | `false` | whether a weak passport photo is fatal |
