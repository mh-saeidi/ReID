#!/usr/bin/env python3
"""Download the face detection and recognition models.

Two tiers are available:

**default** (~175 MB)
    YuNet face detector + InsightFace's ArcFace ``w600k_r50`` recogniser. This
    is what ``config.yaml`` points at, and it is the stronger of the two on
    pose, age and low light. Measured on the bundled demo set it separates
    genuine from impostor faces by 0.76 of cosine similarity.

**lite** (~39 MB, nothing but OpenCV needed to run it)
    YuNet + SFace, both from the OpenCV Model Zoo. Smaller and faster, with a
    narrower but still clean margin (0.59 on the same data). Use it on
    constrained hardware, then set::

        face:
          recognition_model: "models/face_recognition_sface_2021dec.onnx"

Usage::

    python scripts/fetch_face_models.py                # default (ArcFace)
    python scripts/fetch_face_models.py --tier lite
    python scripts/fetch_face_models.py --tier all --force
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
# The face detector is required by every tier.
DETECTOR = (
    "face_detection_yunet_2023mar.onnx",
    f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE = (
    "face_recognition_sface_2021dec.onnx",
    f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)

# ArcFace ships inside the InsightFace buffalo_l bundle; only the recognition
# model is extracted, so none of the unrelated auxiliary models are installed.
BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
ARCFACE_MEMBER = "w600k_r50.onnx"


def _progress(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    done = min(100, count * block * 100 // total)
    sys.stdout.write(f"\r    {done:3d}%")
    sys.stdout.flush()


def download(url: str, target: Path, *, force: bool) -> bool:
    if target.exists() and not force:
        print(f"  keeping existing {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {target.name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".part") as handle:
        temp = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, temp, _progress)  # noqa: S310 - fixed https URLs
        sys.stdout.write("\r")
        shutil.move(str(temp), target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def fetch_arcface(*, force: bool) -> bool:
    target = MODELS / ARCFACE_MEMBER
    if target.exists() and not force:
        print(f"  keeping existing {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        return False
    print(f"  downloading {ARCFACE_MEMBER} (from the buffalo_l bundle)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as handle:
        archive = Path(handle.name)
    try:
        urllib.request.urlretrieve(BUFFALO_L_URL, archive, _progress)  # noqa: S310
        sys.stdout.write("\r")
        with zipfile.ZipFile(archive) as bundle:
            member = next(
                (n for n in bundle.namelist() if n.endswith(ARCFACE_MEMBER)), None
            )
            if member is None:
                raise RuntimeError(
                    f"{ARCFACE_MEMBER} is not present in the downloaded bundle"
                )
            MODELS.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"the downloaded bundle is corrupt ({exc}); re-run this script"
        ) from exc
    finally:
        archive.unlink(missing_ok=True)
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=["default", "lite", "all"],
        default="default",
        help="Which recogniser to fetch (default: ArcFace; 'lite' uses SFace).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--checksums", action="store_true", help="Print SHA-256 digests.")
    args = parser.parse_args()

    print("Face detector:")
    download(DETECTOR[1], MODELS / DETECTOR[0], force=args.force)

    if args.tier in ("lite", "all"):
        print("\nSFace recogniser (lite tier):")
        download(SFACE[1], MODELS / SFACE[0], force=args.force)

    if args.tier in ("default", "all"):
        print("\nArcFace recogniser (default tier):")
        try:
            fetch_arcface(force=args.force)
        except (RuntimeError, OSError) as exc:
            print(f"  error: {exc}", file=sys.stderr)
            print(
                "  fall back to the smaller recogniser with: "
                "python scripts/fetch_face_models.py --tier lite",
                file=sys.stderr,
            )
            return 1

    if args.checksums:
        print("\nSHA-256:")
        for path in sorted(MODELS.glob("*.onnx")):
            print(f"  {sha256(path)}  {path.name}")

    print("\nDone. Next:")
    print("  python main.py config validate --config config.yaml")
    print("  python main.py gallery build   --config config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
