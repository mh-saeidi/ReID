#!/usr/bin/env python3
"""YOLO26 one-shot person re-identification -- command line entry point.

Usage examples::

    python main.py gallery build --config config.yaml
    python main.py webcam --device 0
    python main.py video --input data/videos/test.mp4
    python main.py image --input data/images/test.jpg
    python main.py images --input data/images/
    python main.py benchmark --input data/videos/test.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python main.py` from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli.main import run  # noqa: E402

if __name__ == "__main__":
    run()
