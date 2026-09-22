#!/usr/bin/env python3
"""Build the demo dataset used by the acceptance tests.

The system identifies people by **face**, so the demo is built to exercise
exactly that — including the two cases that decide whether the claim
"identity does not depend on clothing" is actually true.

Two public Ultralytics sample images provide the raw material:

* ``zidane.jpg`` — two people with large, clear faces. They become the
  registered identities "Person A" and "Person B".
* ``bus.jpg``    — four people who are deliberately never registered: two with
  small faces (open-set rejection) and two with no visible face at all
  (the ``no_face`` path).

Generated:

    data/demo/persons/person_a.jpg          one-shot reference photo (A)
    data/demo/persons/person_b.jpg          one-shot reference photo (B)
    data/demo/test_images/
        01_person_a.jpg                     A alone, rescaled and relit
        02_person_b.jpg                     B alone, rescaled and relit
        03_a_and_b.jpg                      both together
        04_unknown_only.jpg                 registered nobody
        05_person_a_different_clothes.jpg   A wearing a different colour
        06_person_a_face_hidden.jpg         A with the face obscured
    data/demo/test_video/walkthrough.mp4    A turns away and back; B is static
    data/demo/evaluation/                   labelled set for `main.py evaluate`

Cases 05 and 06 are the point of the whole dataset:

* **05** recolours everything below the chin and leaves the face untouched. A
  clothing-based system loses the identity here; a face-based one must not.
* **06** obscures the face and leaves the clothing untouched. The correct answer
  is "no face", *not* a confident match — a system that still names the person
  is reading the clothes.

Scope: this verifies the pipeline and the clothing-invariance property on real
faces. It is **not** a face-recognition accuracy benchmark — the test crops
derive from the same photographs as the references, so they share lighting and
pose. Measure accuracy on your own multi-session data with ``main.py evaluate``.

Usage::

    python scripts/build_demo.py
    python scripts/build_demo.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "data" / "demo"
SOURCES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}

# Person boxes measured with yolo26n.pt at conf=0.35 on the source images.
PERSON_A_BOX = (742, 30, 1160, 712)    # right-hand figure in zidane.jpg
PERSON_B_BOX = (118, 195, 706, 715)    # left-hand figure, cropped clear of A
STRANGER_BOXES = [
    (48, 397, 240, 902),    # bus.jpg, face visible but small
    (222, 404, 345, 860),   # bus.jpg, face visible but small
    (669, 394, 809, 879),   # bus.jpg, facing away -> no face
]

YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def fetch(name: str, directory: Path) -> Path:
    target = directory / name
    if target.exists():
        return target
    directory.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SOURCES[name]} -> {target}")
    urllib.request.urlretrieve(SOURCES[name], target)  # noqa: S310 - fixed https URLs
    return target


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    return image[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].copy()


def face_box_in(crop_image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the largest face inside a person crop, for the edit operations."""
    if not YUNET.exists():
        return None
    h, w = crop_image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.6, 0.3, 500)
    detector.setInputSize((w, h))
    _, faces = detector.detect(crop_image)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = (int(v) for v in best[:4])
    return x, y, x + fw, y + fh


# --------------------------------------------------------------------------- #
# Image edits
# --------------------------------------------------------------------------- #


def synthetic_background(width: int, height: int, *, seed: int = 7) -> np.ndarray:
    """A person-free backdrop.

    Blurring a real street photo leaves detectable people (and faces) in it,
    which would add unintended detections to the demo images. A generated
    gradient keeps each scene to exactly the people we place in it.
    """
    rng = np.random.default_rng(seed)
    vertical = np.linspace(40, 150, height, dtype=np.float32)[:, None]
    horizontal = np.linspace(-25, 25, width, dtype=np.float32)[None, :]
    base = np.clip(vertical + horizontal, 0, 255)
    canvas = np.dstack([base * 0.95, base * 1.0, base * 1.05])
    canvas += rng.normal(0.0, 6.0, canvas.shape)
    canvas = cv2.GaussianBlur(np.clip(canvas, 0, 255).astype(np.uint8), (15, 15), 0)
    cv2.line(canvas, (0, int(height * 0.72)), (width, int(height * 0.72)), (95, 95, 100), 3)
    return canvas


def recolour_clothing(person: np.ndarray, *, hue_shift: int = 80) -> np.ndarray:
    """Change the person's outfit while leaving the face completely untouched.

    Everything below the chin is hue-rotated and its brightness altered. The
    pixels above stay bit-identical, so any change in the identity decision can
    only have come from the clothing.
    """
    edited = person.copy()
    face = face_box_in(person)
    neck = face[3] if face else int(person.shape[0] * 0.22)
    neck = max(1, min(neck, person.shape[0] - 2))

    body = edited[neck:, :]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180          # different colour
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.8 + 40, 0, 255)  # more saturated
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.15 + 20, 0, 255)  # lighter
    edited[neck:, :] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return edited


def obscure_face(person: np.ndarray) -> np.ndarray:
    """Hide the face and leave the clothing untouched.

    The correct result for this image is "no face", not a confident match. A
    system that still names the person is identifying them by their clothes.
    """
    edited = person.copy()
    face = face_box_in(person)
    if face is None:
        height = int(person.shape[0] * 0.22)
        face = (0, 0, person.shape[1], height)
    x1, y1, x2, y2 = face
    pad_x = int((x2 - x1) * 0.35)
    pad_y = int((y2 - y1) * 0.45)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(person.shape[1], x2 + pad_x)
    y2 = min(person.shape[0], y2 + pad_y)
    region = edited[y1:y2, x1:x2]
    if region.size:
        # Heavy pixelation rather than a flat rectangle: the head shape stays,
        # so YOLO26 still detects a person and only the *face* is unusable.
        small = cv2.resize(region, (6, 6), interpolation=cv2.INTER_AREA)
        edited[y1:y2, x1:x2] = cv2.resize(
            small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return edited


def scene_from_person(
    person: np.ndarray,
    size: tuple[int, int],
    *,
    scale: float,
    offset: tuple[float, float],
    brightness: float,
    seed: int = 7,
) -> np.ndarray:
    """Composite one person onto a person-free background at a new scale."""
    width, height = size
    canvas = synthetic_background(width, height, seed=seed)

    target_h = int(height * scale)
    target_w = max(1, int(person.shape[1] * target_h / person.shape[0]))
    if target_w > width:
        target_h = max(1, int(target_h * width / target_w))
        target_w = width
    resized = cv2.resize(person, (target_w, target_h))
    resized = cv2.convertScaleAbs(resized, alpha=1.0, beta=brightness)

    x = int((width - target_w) * offset[0])
    y = int((height - target_h) * offset[1])
    x, y = max(0, min(x, width - target_w)), max(0, min(y, height - target_h))
    canvas[y : y + target_h, x : x + target_w] = resized
    return canvas


def augment(person: np.ndarray, variant: int) -> np.ndarray:
    """A different-looking view of the same person, for calibration data.

    Scale, exposure, blur and a small box jitter stand in for the variation a
    second camera introduces. A horizontal flip is deliberately *not* applied:
    it produces a mirror-image face, which is a different kind of change from
    the one a real second camera causes.
    """
    rng = np.random.default_rng(1000 + variant)
    image = person.copy()

    h, w = image.shape[:2]
    dx, dy = int(w * rng.uniform(0.0, 0.05)), int(h * rng.uniform(0.0, 0.04))
    image = image[dy : h - dy or h, dx : w - dx or w]

    scale = float(rng.uniform(0.7, 1.25))
    image = cv2.resize(
        image,
        (max(24, int(image.shape[1] * scale)), max(48, int(image.shape[0] * scale))),
    )
    image = cv2.convertScaleAbs(
        image, alpha=float(rng.uniform(0.82, 1.18)), beta=float(rng.uniform(-22, 22))
    )
    if variant % 3 == 1:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if variant % 4 == 2:
        image = recolour_clothing(image, hue_shift=40 + 25 * variant)
    return image


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #


def build_video(
    path: Path,
    person_a: np.ndarray,
    person_b: np.ndarray,
    *,
    width: int = 960,
    height: int = 540,
    fps: int = 25,
    seconds: int = 12,
) -> None:
    """Render a clip where A's face becomes unavailable and then returns.

    0.0-3.5 s : A and B both face the camera.
    3.5-6.0 s : A's face is obscured — the person is still tracked, but there is
                nothing to recognise. The identity must be *held*, not lost and
                not replaced.
    6.0-8.5 s : A leaves the frame entirely.
    8.5-12  s : A re-enters and must be re-identified from the face alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas_bg = synthetic_background(width, height, seed=11)

    a_h = int(height * 0.80)
    a_w = max(1, int(person_a.shape[1] * a_h / person_a.shape[0]))
    a_visible_img = cv2.resize(person_a, (a_w, a_h))
    a_hidden_img = cv2.resize(obscure_face(person_a), (a_w, a_h))

    b_h = int(height * 0.72)
    b_w = max(1, int(person_b.shape[1] * b_h / person_b.shape[0]))
    b_img = cv2.resize(person_b, (b_w, b_h))

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")

    b_x = max(0, min(int(width * 0.66), width - b_w))
    b_y = height - b_h - 10
    total = fps * seconds
    try:
        for index in range(total):
            t = index / fps
            frame = canvas_bg.copy()
            frame[b_y : b_y + b_h, b_x : b_x + b_w] = b_img

            present = t < 6.0 or t >= 8.5
            if present:
                face_hidden = 3.5 <= t < 6.0
                progress = (t / 6.0) if t < 6.0 else ((t - 8.5) / 3.5)
                a_x = max(0, min(int(progress * (width * 0.38)), width - a_w))
                a_y = height - a_h - 10
                source = a_hidden_img if face_hidden else a_visible_img
                frame[a_y : a_y + a_h, a_x : a_x + a_w] = source
            writer.write(frame)
    finally:
        writer.release()
    print(f"wrote {path} ({total} frames @ {fps} fps)")


# --------------------------------------------------------------------------- #
# Evaluation set
# --------------------------------------------------------------------------- #


def build_evaluation_set(
    root: Path,
    people: dict[str, np.ndarray],
    strangers: list[np.ndarray],
    *,
    variants: int = 6,
) -> None:
    """Write a labelled dataset for ``python main.py evaluate --dataset ...``."""
    for person_id, person in people.items():
        directory = root / "known" / person_id
        directory.mkdir(parents=True, exist_ok=True)
        for variant in range(variants):
            scene = scene_from_person(
                augment(person, variant),
                (640, 640),
                scale=float(0.70 + 0.05 * (variant % 4)),
                offset=(0.2 + 0.15 * (variant % 4), 1.0),
                brightness=float(-10 + 7 * (variant % 4)),
                seed=100 + variant,
            )
            cv2.imwrite(str(directory / f"{person_id}_{variant:02d}.jpg"), scene)

    unknown_dir = root / "unknown"
    unknown_dir.mkdir(parents=True, exist_ok=True)
    for index, stranger in enumerate(strangers):
        for variant in range(max(2, variants // 2)):
            scene = scene_from_person(
                augment(stranger, variant + 50),
                (640, 640),
                scale=float(0.72 + 0.04 * variant),
                offset=(0.3 + 0.1 * variant, 1.0),
                brightness=float(-8 + 6 * variant),
                seed=200 + index * 10 + variant,
            )
            cv2.imwrite(str(unknown_dir / f"stranger_{index:02d}_{variant:02d}.jpg"), scene)
    print(f"wrote the evaluation dataset under {root}")


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing artefacts.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEMO / "_sources",
        help="Where the downloaded source images are kept.",
    )
    args = parser.parse_args()

    persons_dir = DEMO / "persons"
    images_dir = DEMO / "test_images"
    video_dir = DEMO / "test_video"
    for directory in (persons_dir, images_dir, video_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not YUNET.exists():
        print(
            "warning: the YuNet model is missing, so the clothing-change and "
            "face-hidden images will use a rough head estimate.\n"
            "         Run: python scripts/fetch_face_models.py",
            file=sys.stderr,
        )

    bus = cv2.imread(str(fetch("bus.jpg", args.source_dir)))
    zidane = cv2.imread(str(fetch("zidane.jpg", args.source_dir)))
    if bus is None or zidane is None:
        print("error: could not read the source images", file=sys.stderr)
        return 1

    person_a = crop(zidane, PERSON_A_BOX)
    person_b = crop(zidane, PERSON_B_BOX)

    outputs: list[tuple[Path, np.ndarray]] = [
        (persons_dir / "person_a.jpg", person_a),
        (persons_dir / "person_b.jpg", person_b),
        (
            images_dir / "01_person_a.jpg",
            scene_from_person(
                person_a, (960, 720), scale=0.80, offset=(0.30, 1.0), brightness=12.0
            ),
        ),
        (
            images_dir / "02_person_b.jpg",
            scene_from_person(
                person_b, (960, 720), scale=0.78, offset=(0.62, 1.0),
                brightness=-10.0, seed=19,
            ),
        ),
        (images_dir / "03_a_and_b.jpg", zidane),
        (images_dir / "04_unknown_only.jpg", bus),
        (
            images_dir / "05_person_a_different_clothes.jpg",
            scene_from_person(
                recolour_clothing(person_a), (960, 720), scale=0.80,
                offset=(0.45, 1.0), brightness=5.0, seed=23,
            ),
        ),
        (
            images_dir / "06_person_a_face_hidden.jpg",
            scene_from_person(
                obscure_face(person_a), (960, 720), scale=0.80,
                offset=(0.45, 1.0), brightness=5.0, seed=29,
            ),
        ),
    ]

    for path, image in outputs:
        if path.exists() and not args.force:
            print(f"keeping existing {path}")
            continue
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"wrote {path} ({image.shape[1]}x{image.shape[0]})")

    evaluation_dir = DEMO / "evaluation"
    if evaluation_dir.exists() and not args.force:
        print(f"keeping existing {evaluation_dir}")
    else:
        strangers = [crop(bus, box) for box in STRANGER_BOXES]
        build_evaluation_set(
            evaluation_dir, {"person_a": person_a, "person_b": person_b}, strangers
        )

    video_path = video_dir / "walkthrough.mp4"
    if video_path.exists() and not args.force:
        print(f"keeping existing {video_path}")
    else:
        build_video(video_path, person_a, person_b)

    print("\nDemo dataset ready. Next:")
    print("  python main.py gallery build --config config.yaml")
    print("  python main.py images --input data/demo/test_images")
    print("  python main.py evaluate --dataset data/demo/evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
