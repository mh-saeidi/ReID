#!/usr/bin/env python3
"""Generate multi-person clips for the benchmark matrix.

Batch size, the recognition scheduler and the face-search strategy all behave
differently with one person in frame than with ten, and the bundled demo clip
has two. Measuring a batch size of 8 against a scene that never produces more
than two faces tells you nothing.

These clips composite the demo person crops at several scales and positions so
each frame contains a known number of detectable faces. They are synthetic --
repeated people on a flat background -- and exist to load the pipeline, not to
measure recognition accuracy.

Usage::

    python scripts/build_crowd_clips.py                 # 1, 3, 5 and 10 people
    python scripts/build_crowd_clips.py --people 5 --seconds 8
    python scripts/build_crowd_clips.py --resolution 1920x1080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "data" / "demo"
OUTPUT_DIR = DEMO / "benchmark"


def synthetic_background(width: int, height: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vertical = np.linspace(45, 140, height, dtype=np.float32)[:, None]
    horizontal = np.linspace(-20, 20, width, dtype=np.float32)[None, :]
    base = np.clip(vertical + horizontal, 0, 255)
    canvas = np.dstack([base * 0.96, base, base * 1.04])
    canvas += rng.normal(0.0, 5.0, canvas.shape)
    return cv2.GaussianBlur(np.clip(canvas, 0, 255).astype(np.uint8), (13, 13), 0)


def layout(count: int) -> list[tuple[float, float, float]]:
    """(x_fraction, y_fraction, scale) for ``count`` people, in rows.

    Positions are fractional so the same layout works at any resolution.

    People are placed on up to two rows with the back row smaller, which keeps
    the faces at a range of sizes -- the realistic case for the quality gate.
    """
    if count <= 5:
        rows = [(count, 0.98, 0.80)]
    else:
        front = (count + 1) // 2
        rows = [(front, 0.98, 0.78), (count - front, 0.72, 0.52)]

    placements: list[tuple[float, float, float]] = []
    for people, baseline, scale in rows:
        if people <= 0:
            continue
        for index in range(people):
            x = (index + 0.5) / people
            placements.append((x, baseline, scale))
    return placements


def composite(
    people: list[np.ndarray],
    placements: list[tuple[float, float, float]],
    background: np.ndarray,
    phase: float,
) -> np.ndarray:
    """Draw everybody onto the background, with a small horizontal sway."""
    frame = background.copy()
    height, width = frame.shape[:2]
    for index, (x_fraction, y_fraction, scale) in enumerate(placements):
        person = people[index % len(people)]
        target_h = max(32, int(height * scale))
        target_w = max(16, int(person.shape[1] * target_h / person.shape[0]))
        if target_w >= width:
            target_w = width - 2
            target_h = max(32, int(person.shape[0] * target_w / person.shape[1]))
        resized = cv2.resize(person, (target_w, target_h))

        sway = int(12 * np.sin(phase * 2 * np.pi + index))
        x = int(x_fraction * width - target_w / 2) + sway
        y = int(y_fraction * height - target_h)
        x = max(0, min(x, width - target_w))
        y = max(0, min(y, height - target_h))
        frame[y : y + target_h, x : x + target_w] = resized
    return frame


def build_clip(
    path: Path,
    people: list[np.ndarray],
    count: int,
    *,
    width: int,
    height: int,
    fps: int,
    seconds: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    background = synthetic_background(width, height)
    placements = layout(count)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    total = fps * seconds
    try:
        for index in range(total):
            writer.write(composite(people, placements, background, index / total))
    finally:
        writer.release()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--people", type=int, nargs="*", default=[1, 3, 5, 10],
        help="Person counts to generate (default: 1 3 5 10).",
    )
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--resolution", default="1280x720",
        help="Output resolution, e.g. 1280x720 or 1920x1080.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    width, height = (int(v) for v in args.resolution.lower().split("x"))

    sources = sorted((DEMO / "persons").glob("*.jpg"))
    if not sources:
        print(
            "error: no person crops found. Run: python scripts/build_demo.py",
            file=sys.stderr,
        )
        return 1
    people = [cv2.imread(str(p)) for p in sources]
    people = [p for p in people if p is not None]

    for count in args.people:
        name = f"crowd_{count:02d}p_{width}x{height}.mp4"
        path = OUTPUT_DIR / name
        if path.exists() and not args.force:
            print(f"keeping existing {path}")
            continue
        frames = build_clip(
            path, people, count,
            width=width, height=height, fps=args.fps, seconds=args.seconds,
        )
        print(f"wrote {path} ({count} people, {frames} frames @ {args.fps} fps)")

    print("\nBenchmark with:")
    print(f"  python main.py benchmark-suite matrix --input {OUTPUT_DIR}/crowd_05p_"
          f"{width}x{height}.mp4 -g batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
