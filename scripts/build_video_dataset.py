"""Turn a labelled video into an evaluation dataset.

The synthetic dataset built by ``build_evaluation_dataset.py`` validates the
pipeline; it cannot measure real accuracy, because every condition in it is a
transformation of a handful of photographs. This script builds the real thing
from footage of people who are registered from their passport photographs.

Input is the video plus a labels file, one entry per detected face::

    [{"frame": 180, "bbox": [x1, y1, x2, y2], "label": "Saeidi"}, ...]

Labels are assigned by a human looking at the crops -- never by the system
being measured, which would make the evaluation circular.

Each face becomes one query image: a region around it, wide enough that the
face survives re-detection and tight enough that a second person's face cannot
become the largest one in the frame. Every image is named
``<block>__<frame>.jpg``, and the block is a contiguous stretch of the video.
The splitter keeps a block whole, because two frames a tenth of a second apart
are the same photograph for this purpose, and splitting them across
calibration and test would measure memorisation.

Conditions are geometric, so they describe the capture rather than the
system's opinion of it:

    frontal   both eyes clearly separated -- interocular >= 22% of the box
    turned    the head is rotated away from the camera
    distant   the face spans fewer than 85 px, whatever its pose

Usage::

    python scripts/build_video_dataset.py \\
        --video data/demo/test_video/test.mp4 \\
        --labels data/evaluation_video/labels.json \\
        --enrollment data/input \\
        --output data/evaluation_video
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.types import BBox  # noqa: E402

BLOCK_FRAMES = 60
"""Two seconds at 30 fps. Long enough that separate blocks are genuinely
different moments, short enough to leave plenty of blocks to split."""

GUARD_FRAMES = 9
"""Frames dropped either side of a block boundary.

Without it the last frame of one block and the first of the next are a tenth
of a second apart, and if the split sends them to opposite halves the test set
contains a near-duplicate of a calibration image. Nine frames is 0.3 s, which
is enough for a person to have moved.
"""

CROP_MARGIN = 0.55
DISTANT_PX = 85.0
TURNED_IOD_RATIO = 0.22
NOT_A_FACE = "not_a_face"
AMBIGUOUS = "ambiguous"


def condition_for(bbox: BBox, iod: float) -> str:
    extent = max(bbox.width, bbox.height)
    if extent < DISTANT_PX:
        return "distant"
    return "turned" if iod / max(extent, 1e-6) < TURNED_IOD_RATIO else "frontal"


def crop_region(image: np.ndarray, bbox: BBox) -> np.ndarray:
    height, width = image.shape[:2]
    pad_x, pad_y = bbox.width * CROP_MARGIN, bbox.height * CROP_MARGIN
    x1 = int(max(0, bbox.x1 - pad_x))
    y1 = int(max(0, bbox.y1 - pad_y))
    x2 = int(min(width, bbox.x2 + pad_x))
    y2 = int(min(height, bbox.y2 + pad_y))
    return image[y1:y2, x1:x2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True,
                        help="Directory of <person_id>.jpg passport photographs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.force:
            print(f"{args.output} exists; pass --force to rebuild it", file=sys.stderr)
            return 1
        shutil.rmtree(args.output)

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict]] = {}
    for entry in labels:
        by_frame.setdefault(int(entry["frame"]), []).append(entry)

    # References: one passport photograph per person, copied unmodified.
    enrol_out = args.output / "enrollment"
    people = set()
    for photo in sorted(args.enrollment.iterdir()):
        if photo.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        person = photo.stem
        target = enrol_out / person
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(photo, target / f"reference{photo.suffix.lower()}")
        people.add(person)
    if not people:
        print(f"no enrollment photographs in {args.enrollment}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 1

    written = Counter()
    skipped = Counter()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for entry in by_frame.get(frame_index, ()):
            label = entry["label"]
            if label == AMBIGUOUS:
                skipped["ambiguous"] += 1
                continue
            bbox = BBox(*entry["bbox"])
            crop = crop_region(frame, bbox)
            if crop.size == 0 or min(crop.shape[:2]) < 24:
                skipped["too_small"] += 1
                continue

            offset = frame_index % BLOCK_FRAMES
            if offset < GUARD_FRAMES or offset >= BLOCK_FRAMES - GUARD_FRAMES:
                skipped["block_guard"] += 1
                continue

            block = f"block{frame_index // BLOCK_FRAMES:03d}"
            name = f"{block}__f{frame_index:05d}_{entry.get('index', 0):04d}.jpg"
            if label == NOT_A_FACE:
                # These are things the detector called a face and a human did
                # not: hands, a dark doorway. Nothing may ever be named from
                # one, so they live in the unknown split.
                target = args.output / "unknown" / "false_detection" / name
            else:
                iod = float(entry.get("iod", 0.0))
                target = (args.output / "query" / condition_for(bbox, iod)
                          / label / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), crop)
            written[label] += 1
        frame_index += 1
    capture.release()

    manifest = {
        "source_video": str(args.video),
        "frames": frame_index,
        "block_frames": BLOCK_FRAMES,
        "guard_frames": GUARD_FRAMES,
        "identities": sorted(people),
        "written": dict(written),
        "skipped": dict(skipped),
        "conditions": {
            "frontal": "both eyes separated: interocular >= "
                       f"{TURNED_IOD_RATIO:.0%} of the face box",
            "turned": "head rotated away from the camera",
            "distant": f"face spans fewer than {DISTANT_PX:.0f} px",
        },
        "synthetic": False,
        "notes": [
            "Real footage of the registered people, labelled by eye.",
            "Query images are regions of video frames, so images from one "
            "block are near-duplicates; the splitter keeps a block whole.",
            "The unknown split holds false face detections, not unregistered "
            "people: this footage contains no third person whose face the "
            "detector resolves, so the impostor evidence here is weaker than "
            "a deployment needs.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"wrote {sum(written.values())} query images to {args.output}")
    for label, count in sorted(written.items()):
        print(f"  {label:16s} {count}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
