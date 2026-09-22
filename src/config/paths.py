"""Central resolution of every filesystem location used by the application.

Components never build paths from string literals; they ask :class:`ProjectPaths`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.loader import resolve_path
from src.config.schema import AppConfig
from src.core.exceptions import OutputError


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """All directories derived from the configuration, resolved to absolutes."""

    base_dir: Path
    gallery_dir: Path
    embeddings_dir: Path
    metadata_dir: Path
    enrollment_crop_dir: Path
    output_dir: Path
    snapshots_dir: Path
    videos_dir: Path
    crops_dir: Path
    output_metadata_dir: Path
    annotated_images_dir: Path
    events_dir: Path
    debug_dir: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> ProjectPaths:
        base = Path(config.base_dir) if config.base_dir else Path.cwd()
        gallery = resolve_path(config.gallery.directory, base)
        output = resolve_path(config.output.directory, base)
        return cls(
            base_dir=base,
            gallery_dir=gallery,
            embeddings_dir=gallery / config.gallery.embeddings_subdir,
            metadata_dir=gallery / config.gallery.metadata_subdir,
            enrollment_crop_dir=resolve_path(config.enrollment.crop_dir, base),
            output_dir=output,
            snapshots_dir=output / config.output.snapshot_dir,
            videos_dir=output / config.output.videos_dir,
            crops_dir=output / config.output.crops_dir,
            output_metadata_dir=output / config.output.metadata_dir,
            annotated_images_dir=output / config.output.annotated_image_dir,
            events_dir=output / config.events.directory,
            debug_dir=output / config.debug.directory,
        )

    def resolve(self, path: str | Path) -> Path:
        """Resolve any configured path relative to the configuration directory."""
        return resolve_path(path, self.base_dir)

    def ensure(self, *paths: Path) -> None:
        """Create directories, converting permission problems into clear errors."""
        for path in paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OutputError(
                    f"cannot create directory {path}: {exc}. "
                    "Check output.directory / gallery.directory and filesystem permissions."
                ) from exc
