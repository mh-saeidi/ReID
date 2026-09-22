#!/usr/bin/env python3
"""Build a condition-split evaluation dataset.

READ THIS FIRST
---------------
The conditions this script produces are SYNTHETIC. It takes the real faces
available in the repository and applies image transformations -- blur,
downscaling, illumination change, rotation, drawn spectacle frames, drawn
masks. That is not the same thing as photographing the same person wearing
real glasses, or a decade later.

What synthetic conditions are good for:

  * exercising the whole pipeline end to end under controlled degradation;
  * ranking configurations against each other (A/B), because every variant
    sees identical inputs;
  * finding the point where a stage breaks down.

What they are NOT good for:

  * claiming a real-world identification rate;
  * anything involving age, which cannot be simulated at all -- a face does
    not age by an affine transform, and no synthetic "age" condition is
    produced here for exactly that reason;
  * mask robustness, since a drawn rectangle is not a real mask and does not
    reproduce its shadows, folds, or the way it changes the detected box.

A real evaluation needs photographs of real people under real conditions. The
directory layout below is the one the evaluator expects, so real data can be
dropped in to replace the synthetic set without changing anything else.

    data/evaluation/
        enrollment/<person_id>/reference.jpg     one passport-style photo
        query/<condition>/<person_id>/*.jpg      labelled queries
        unknown/<condition>/*.jpg                people who are NOT registered

Usage::

    python scripts/build_evaluation_dataset.py
    python scripts/build_evaluation_dataset.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "data" / "evaluation"

# Conditions that are honest to synthesise. "age_variation" is deliberately
# absent: there is no transformation that makes a face look ten years older,
# and generating one would produce a number that means nothing.
SYNTHETIC_CONDITIONS = (
    "normal",
    "glasses",
    "mask",
    "partial",
    "different_pose",
    "low_light",
    "different_distance",
    "blur",
)


def _face_of(detector, image: np.ndarray):
    faces = detector.detect(image)
    return max(faces, key=lambda f: f.bbox.area) if faces else None


# --------------------------------------------------------------------------- #
# Condition generators
# --------------------------------------------------------------------------- #


def apply_normal(image: np.ndarray, _face, rng) -> np.ndarray:
    """Mild, realistic capture variation: slight exposure and JPEG loss."""
    out = cv2.convertScaleAbs(image, alpha=float(rng.uniform(0.92, 1.08)),
                              beta=float(rng.uniform(-10, 10)))
    ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 92))])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else out


def apply_glasses(image: np.ndarray, face, rng) -> np.ndarray:
    """Draw spectacle frames over the eyes.

    A crude approximation: real glasses add frames, lens distortion and
    reflections, and only the frames are reproduced here.
    """
    out = image.copy()
    if face is None or not face.has_landmarks:
        return out
    right_eye, left_eye = face.landmarks[0], face.landmarks[1]
    width = float(np.linalg.norm(left_eye - right_eye))
    radius = max(6, int(width * 0.42))
    thickness = max(2, int(width * 0.09))
    colour = (30, 30, 30) if rng.random() < 0.6 else (90, 70, 55)
    for centre in (right_eye, left_eye):
        cv2.circle(out, (int(centre[0]), int(centre[1])), radius, colour, thickness)
    cv2.line(out, (int(right_eye[0]) + radius, int(right_eye[1])),
             (int(left_eye[0]) - radius, int(left_eye[1])), colour, thickness)
    if rng.random() < 0.4:   # a lens reflection on one side
        gx, gy = int(right_eye[0] - radius * 0.3), int(right_eye[1] - radius * 0.3)
        cv2.ellipse(out, (gx, gy), (int(radius * 0.5), int(radius * 0.25)),
                    -30, 0, 360, (235, 235, 235), -1)
    return out


def apply_mask(image: np.ndarray, face, rng) -> np.ndarray:
    """Draw a surgical-style mask over the lower face."""
    out = image.copy()
    if face is None or not face.has_landmarks:
        return out
    nose = face.landmarks[2]
    mouth = (face.landmarks[3] + face.landmarks[4]) / 2.0
    width = float(np.linalg.norm(face.landmarks[1] - face.landmarks[0]))
    top = int(nose[1] + width * 0.10)
    bottom = int(mouth[1] + width * 1.15)
    left = int(face.bbox.x1 - width * 0.12)
    right = int(face.bbox.x2 + width * 0.12)
    colour = [(224, 226, 228), (150, 190, 205), (60, 60, 65)][int(rng.integers(0, 3))]
    points = np.array([
        [left, top], [right, top],
        [right, int(bottom * 0.92)], [int((left + right) / 2), bottom],
        [left, int(bottom * 0.92)],
    ], dtype=np.int32)
    cv2.fillPoly(out, [points], colour)
    for offset in range(1, 4):   # pleat lines
        y = top + int((bottom - top) * offset / 4)
        cv2.line(out, (left, y), (right, y),
                 tuple(max(0, c - 18) for c in colour), 1)
    return out


def apply_partial(image: np.ndarray, face, rng) -> np.ndarray:
    """Occlude one side of the face, as a hand or a frame edge would."""
    out = image.copy()
    if face is None:
        return out
    box = face.bbox
    if rng.random() < 0.5:
        x1, x2 = int(box.x1), int(box.x1 + box.width * 0.42)
    else:
        x1, x2 = int(box.x2 - box.width * 0.42), int(box.x2)
    out[int(box.y1):int(box.y2), max(0, x1):min(out.shape[1], x2)] = (
        int(rng.integers(40, 90)), int(rng.integers(40, 90)), int(rng.integers(45, 95))
    )
    return out


def apply_pose(image: np.ndarray, _face, rng) -> np.ndarray:
    """Approximate a turned head with a perspective warp.

    A projective warp of a flat image is not head rotation -- it cannot reveal
    the far cheek or hide the near one -- so this is a weak proxy for pose and
    is labelled as such.
    """
    height, width = image.shape[:2]
    shift = float(rng.uniform(0.10, 0.24)) * width
    direction = 1 if rng.random() < 0.5 else -1
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    if direction > 0:
        target = np.float32([[shift, 0], [width, shift * 0.35],
                             [width, height - shift * 0.35], [shift, height]])
    else:
        target = np.float32([[0, shift * 0.35], [width - shift, 0],
                             [width - shift, height], [0, height - shift * 0.35]])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (width, height),
                               borderValue=(110, 110, 110))


def apply_low_light(image: np.ndarray, _face, rng) -> np.ndarray:
    """Darken and add sensor noise, as a low-light camera would."""
    gain = float(rng.uniform(0.18, 0.38))
    out = image.astype(np.float32) * gain
    out += rng.normal(0.0, float(rng.uniform(4.0, 11.0)), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_distance(image: np.ndarray, _face, rng) -> np.ndarray:
    """Downsample hard, then restore the frame size.

    This is what a distant face really costs: detail that the sensor never
    captured cannot be recovered by upscaling.
    """
    height, width = image.shape[:2]
    factor = float(rng.uniform(0.16, 0.34))
    small = cv2.resize(image, (max(12, int(width * factor)), max(12, int(height * factor))),
                       interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        small = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_blur(image: np.ndarray, _face, rng) -> np.ndarray:
    """Motion or defocus blur."""
    if rng.random() < 0.5:
        size = int(rng.integers(9, 21)) | 1
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2, :] = 1.0 / size
        angle = float(rng.uniform(0, 180))
        matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        return cv2.filter2D(image, -1, kernel)
    size = int(rng.integers(7, 19)) | 1
    return cv2.GaussianBlur(image, (size, size), 0)


GENERATORS = {
    "normal": apply_normal,
    "glasses": apply_glasses,
    "mask": apply_mask,
    "partial": apply_partial,
    "different_pose": apply_pose,
    "low_light": apply_low_light,
    "different_distance": apply_distance,
    "blur": apply_blur,
}


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #


def discover_faces(detector, min_face_px: int = 30) -> list[tuple[str, np.ndarray]]:
    """Find every distinct usable face in the repository's sample images."""
    sources = [
        ("zidane", ROOT / "data" / "demo" / "_sources" / "zidane.jpg"),
        ("bus", ROOT / "data" / "demo" / "_sources" / "bus.jpg"),
    ]
    found: list[tuple[str, np.ndarray]] = []
    for tag, path in sources:
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        faces = sorted(detector.detect(image), key=lambda f: -f.bbox.area)
        for index, face in enumerate(faces):
            if face.size < min_face_px:
                continue
            box = face.bbox
            pad_x, pad_y = box.width * 0.75, box.height * 0.85
            x1 = max(0, int(box.x1 - pad_x))
            x2 = min(image.shape[1], int(box.x2 + pad_x))
            y1 = max(0, int(box.y1 - pad_y))
            y2 = min(image.shape[0], int(box.y2 + pad_y))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Upscale so every identity starts from a comparable resolution.
            scale = max(1.0, 420.0 / max(1, crop.shape[1]))
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
            found.append((f"{tag}_{index:02d}", crop))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--samples", type=int, default=6,
                        help="Query samples per person per condition.")
    parser.add_argument("--registered", type=int, default=0,
                        help="How many identities to register; 0 = all but one, "
                             "leaving at least one as a true unknown.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    from src.face.scrfd import SCRFDFaceDetector

    if OUTPUT.exists() and not args.force:
        print(f"{OUTPUT} already exists; pass --force to rebuild.")
        return 0
    if OUTPUT.exists():
        # A rebuild must start clean. Leaving stale files behind silently
        # corrupts the labels: an identity that was "unknown" in a previous run
        # and is registered in this one leaves images in unknown/ that the
        # evaluator will score as false accepts, which looks like a model
        # failure and is not.
        import shutil as _shutil

        _shutil.rmtree(OUTPUT)
        print(f"removed the previous dataset at {OUTPUT}")

    detector = SCRFDFaceDetector(str(ROOT / "models" / "scrfd_10g.onnx"), confidence=0.5)
    detector.load()

    people = discover_faces(detector)
    if len(people) < 2:
        print("error: fewer than two usable faces found in the sample images",
              file=sys.stderr)
        return 1

    registered_count = args.registered or max(1, len(people) - 1)
    registered = people[:registered_count]
    unknown = people[registered_count:]
    rng = np.random.default_rng(args.seed)

    for directory in ("enrollment", "query", "unknown"):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)

    written = 0
    for person_id, crop in registered:
        enrol_dir = OUTPUT / "enrollment" / person_id
        enrol_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(enrol_dir / "reference.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        face = _face_of(detector, crop)
        for condition in SYNTHETIC_CONDITIONS:
            target = OUTPUT / "query" / condition / person_id
            target.mkdir(parents=True, exist_ok=True)
            for sample in range(args.samples):
                variant = GENERATORS[condition](crop, face, rng)
                cv2.imwrite(str(target / f"{sample:03d}.jpg"), variant,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                written += 1

    for person_id, crop in unknown:
        face = _face_of(detector, crop)
        for condition in ("normal", "mask", "glasses"):
            target = OUTPUT / "unknown" / condition
            target.mkdir(parents=True, exist_ok=True)
            for sample in range(args.samples):
                variant = GENERATORS[condition](crop, face, rng)
                cv2.imwrite(str(target / f"{person_id}_{sample:03d}.jpg"), variant,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                written += 1

    manifest = {
        "generated_by": "scripts/build_evaluation_dataset.py",
        "synthetic": True,
        "warning": (
            "All query conditions are SYNTHETIC image transformations of a small "
            "number of real faces. They are suitable for pipeline validation and "
            "for A/B comparison between configurations, and are NOT suitable for "
            "claiming a real-world identification rate. No age-variation split is "
            "generated, because ageing cannot be simulated."
        ),
        "registered_identities": [p for p, _ in registered],
        "unknown_identities": [p for p, _ in unknown],
        "conditions": list(SYNTHETIC_CONDITIONS),
        "samples_per_condition": args.samples,
        "images_written": written,
        "seed": args.seed,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Registered identities : {len(registered)}  {[p for p, _ in registered]}")
    print(f"Unknown identities    : {len(unknown)}  {[p for p, _ in unknown]}")
    print(f"Conditions            : {len(SYNTHETIC_CONDITIONS)}")
    print(f"Images written        : {written}")
    print(f"Output                : {OUTPUT}")
    print()
    print("NOTE: these conditions are synthetic. See manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
