# Phase 1 audit: the existing system against passport-photo enrollment

An audit of the YOLO26 ReID system as it stood **before** the passport-photo
redesign, judged against one question: *given one passport-style photograph of
a person, can this system identify them later under glasses, a mask, a
different pose, poor light, greater distance, partial occlusion or the passage
of years?*

Findings are marked **KEPT** (correct, carried forward unchanged),
**CHANGED** (rebuilt in this redesign) or **GAP** (absent entirely).

---

## A. Person detection

**1. What detects people, and how is its confidence used?**
YOLO26n via Ultralytics, filtered to class 0. Its score reached the output as
`detector_confidence` and was never mixed into identity. **KEPT** — this was
already correct, and is the property most systems get wrong.

**2. Does person detection gate enrollment?**
It did. Enrollment ran person detection on the reference image and refused
images where no person box was found. **CHANGED** — a passport photograph is a
head and shoulders; a person detector is not built for it and can only reject
valid references. Enrollment is now face-first: the face detector locates the
subject directly, with no person detection involved.

**3. Is detection confidence ever read as identity confidence?**
No, in either version. **KEPT**.

---

## B. Tracking

**4. What provides track continuity?**
BoT-SORT through Ultralytics, optionally with appearance features. **KEPT**.

**5. Is a track ID ever treated as an identity?**
No. Track state and identity state were already separate objects.
**KEPT**, and now enforced by a test: a tracked person whose face never
appears stays unnamed for as long as the track lives.

**6. Can tracking *create* an identity?**
No — but the previous stabiliser could hold one for a bounded number of frames
after the face disappeared. **KEPT** in principle, **CHANGED** in mechanism:
the hold is now a state-machine transition (`TEMPORARILY_OCCLUDED`), applies
only to tracks that already reached `RECOGNIZED`, and is the single code path
by which a face-less frame can carry a name.

---

## C. Face detection and alignment

**7. Which face detector, and was it chosen by measurement?**
YuNet, chosen for speed and size. It had not been compared on the metric that
matters. **CHANGED** — SCRFD-10G is now the default. Measured end to end:

| Detector | Genuine min | Impostor max | Separation |
|---|---|---|---|
| YuNet | −0.027 | 0.152 | −0.179 |
| SCRFD | 0.481 | 0.072 | +0.409 |

Detection recall differed by one detection in 32 images. Identity separation
differed by 0.59. YuNet's landmarks were the problem, not its recall.

**8. Is alignment performed, and with what transform?**
Yes: five-point similarity transform onto the ArcFace 112×112 template.
**KEPT** — correct as it stood, including the choice of a similarity rather
than a full affine transform.

**9. What happens to a face with no landmarks?**
It was crop-fallback embedded. **CHANGED** — it is now refused for
recognition (`FACE_ALIGNMENT_FAILED`); the fallback crop remains for display.
An unaligned chip produces an embedding that is silently wrong, which is worse
than no answer.

---

## D. Face quality and visibility

**10. Was face quality assessed?**
Partly: a minimum face size and a blur check, used as a pass/fail gate.
**CHANGED** — quality is now a continuous weighted score over resolution,
sharpness, exposure, pose, landmark fit and visibility, and it also weights
temporal evidence. A quality *rejection* now names its weakest component, so
"why did this fail?" has an answer.

**11. Was occlusion detected?**
No. **GAP** — now a five-class visibility judgement
(`full_face`/`masked`/`partial_face`/`heavily_occluded`/`no_usable_face`).

**12. Was the occlusion test skin-tone dependent?**
The first implementation of it was — a YCrCb skin-colour range, which reported
darker-skinned faces as occluded and was defeated by a beard. **CHANGED
during this work**: replaced with a structure test (local gradient energy plus
chroma dispersion) which is tone-independent, and pinned by a test that runs
five skin tones through the same face and requires the same verdict.

**13. Was a masked face handled differently from a clear one?**
No — one threshold for everything. **GAP** — masked and partial faces now
have their own threshold families, calibrated separately.

---

## E. Embedding

**14. Which embedding model, and is the output normalised?**
ArcFace `w600k_r50` (512-D), L2-normalised, with SFace available.
**KEPT** — this was already the right choice for one-shot face recognition.

**15. Is the embedding ever computed from the body in face mode?**
No. Verified by inspection and by a test. **KEPT**.

---

## F. Gallery

**16. What did the gallery store per person?**
Exactly one embedding, in a flat `embeddings/` directory. **CHANGED** — one
directory per person holding the reference photo, one or more embeddings, and
metadata including the reference's quality warnings. Matching takes the best
score across a person's embeddings, never the mean.

**17. Was the reference image ever modified or the gallery ever auto-written?**
No, and no. **KEPT** — the operator's original is copied, never written.

**18. Could a false recognition contaminate the gallery?**
There was no adaptation at all, so no. **GAP → guarded**: adaptation now
exists, is **off by default**, and passes through eight gates before a single
embedding is stored. The passport reference can never be evicted.

---

## G. Matching and thresholds

**19. Where did the threshold come from?**
A number in `config.yaml` (0.70), chosen by hand. **CHANGED** — thresholds are
fitted from labelled data per visibility class, and the configured values
default to `None` meaning "use the calibrated value".

**20. Was similarity ever presented as a probability or a percentage?**
The CLI printed `0.96` beside a name, and the README described it as a score.
Nothing multiplied it by 100 or called it a percentage. **KEPT**, and now
made explicit: `identity_confidence` is a separate, calibrated field that is
`None` when no calibration exists.

**21. Was every detected person forced onto the nearest identity?**
No — open-set rejection was already implemented, along with a distinct
`NO_FACE` status. **KEPT**. This was the strongest part of the original
system.

**22. Was there an ambiguity rule?**
Yes, a margin check against the runner-up. **KEPT**, carried into the new
decision engine.

---

## H. Evidence and evaluation

**23. Was temporal evidence quality-weighted?**
No — the stabiliser counted frames. **CHANGED** — evidence is now weighted by
face quality, so four clear frames outweigh twenty poor ones, and a hidden
face is not counted as evidence against the identity.

**24. Could the system's accuracy be measured per condition?**
No. There was a threshold-sweep tool but no labelled multi-condition dataset
and no per-condition report. **GAP** — there is now a dataset builder, a
train/test split stratified by (condition, identity) and disjoint by
construction, per-condition accuracy / rank-1 / TAR / FAR / FRR, a failure
taxonomy by pipeline layer, and an A/B ladder over the components.

---

## Summary

| Verdict | Count | Areas |
|---|---|---|
| KEPT | 9 | detection confidence hygiene, alignment maths, ArcFace, open-set rejection, ambiguity margin, track/identity separation |
| CHANGED | 11 | enrollment path, face detector, landmark handling, quality model, gallery layout, thresholds, temporal evidence |
| GAP | 4 | occlusion detection, visibility-specific thresholds, gallery adaptation, per-condition evaluation |

The original system's identity **hygiene** was sound: it kept detection
confidence away from identity, it refused to name people it could not see, and
it never confused a track with a person. What it lacked was everything needed
to know whether it was *right* — no calibration, no per-condition measurement,
no occlusion awareness, and a threshold that was a guess.

The single most consequential finding is **#7**: the face detector had been
chosen on recall, and the one it replaced produced overlapping genuine and
impostor distributions. No amount of threshold tuning downstream could have
fixed that, and no metric the original system reported would have revealed it.
