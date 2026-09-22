"""Retention policy enforcement.

This system stores biometric-like material (person snapshots, crops, clips),
so data is removed on a schedule rather than accumulating indefinitely. A
retention period of 0 days means "keep forever" and is never destructive.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

from src.config.paths import ProjectPaths
from src.config.schema import RetentionConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupReport:
    removed_files: int
    removed_bytes: int
    scanned: int
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "removed_files": self.removed_files,
            "removed_mb": round(self.removed_bytes / (1024 * 1024), 2),
            "scanned": self.scanned,
            "dry_run": self.dry_run,
        }


def _prune(directory: Path, days: int, cutoff: float, *, dry_run: bool) -> tuple[int, int, int]:
    if days <= 0 or not directory.exists():
        return 0, 0, 0
    removed = removed_bytes = scanned = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        try:
            stat = path.stat()
        except OSError:  # pragma: no cover - race with another process
            continue
        if stat.st_mtime >= cutoff:
            continue
        removed += 1
        removed_bytes += stat.st_size
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Cannot delete %s: %s", path, exc)
                removed -= 1
                removed_bytes -= stat.st_size
    return removed, removed_bytes, scanned


def apply_retention(
    config: RetentionConfig, paths: ProjectPaths, *, dry_run: bool = False
) -> CleanupReport:
    """Delete stored artefacts older than the configured retention periods."""
    if not config.enabled:
        logger.debug("Retention is disabled; nothing removed")
        return CleanupReport(0, 0, 0, dry_run)

    now = time.time()
    day = 86400.0
    targets = [
        (paths.snapshots_dir, config.snapshots_days),
        (paths.videos_dir, config.videos_days),
        (paths.crops_dir, config.crops_days),
        (paths.output_metadata_dir, config.metadata_days),
        (paths.events_dir, config.events_days),
    ]

    removed = removed_bytes = scanned = 0
    for directory, days in targets:
        r, b, s = _prune(directory, days, now - days * day, dry_run=dry_run)
        removed += r
        removed_bytes += b
        scanned += s

    if not dry_run:
        for directory, _ in targets:
            _remove_empty_dirs(directory)

    report = CleanupReport(removed, removed_bytes, scanned, dry_run)
    logger.info("Retention applied", extra=report.to_dict())
    return report


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            with contextlib.suppress(OSError):  # best effort
                path.rmdir()
