"""``gallery`` subcommands: build, rebuild, list, remove, inspect."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.cli.common import (
    DEFAULT_CONFIG,
    ConfigOption,
    DeviceOption,
    JsonOption,
    LogLevelOption,
    build_overrides,
    emit,
    engine_for,
    fail,
    handle_errors,
    load,
)
from src.events.types import Event, EventType

app = typer.Typer(help="Manage the registered-identity gallery.", no_args_is_help=True)


@app.command("build")
@handle_errors
def build(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Re-enroll everyone, ignoring cached embeddings.")
    ] = False,
    person: Annotated[
        list[str] | None,
        typer.Option("--person", "-p", help="Limit the build to these person ids (repeatable)."),
    ] = None,
    log_level: LogLevelOption = None,
    device: DeviceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Enroll every configured person and store their reference embeddings."""
    app_config = load(config, build_overrides(log_level=log_level, device=device))
    engine = engine_for(app_config)
    try:
        report = engine.build_gallery(force=rebuild, only=person)
        engine.events.emit(
            Event(
                type=EventType.GALLERY_UPDATED,
                source_id="cli",
                payload=report.to_dict(),
            )
        )
        lines = [
            f"Gallery: {report.total} active identity(ies) in {engine.paths.gallery_dir}",
            f"  enrolled : {', '.join(report.enrolled) or '-'}",
            f"  reused   : {', '.join(report.reused) or '-'}",
            f"  disabled : {', '.join(report.skipped) or '-'}",
        ]
        for identity_id, message in report.failed.items():
            lines.append(f"  FAILED   : {identity_id}: {message}")
        emit(report.to_dict(), json_output, "\n".join(lines))
        if report.failed:
            raise typer.Exit(1)
    finally:
        engine.close()


@app.command("rebuild")
@handle_errors
def rebuild(
    person: Annotated[str, typer.Option("--person", "-p", help="Person id to re-enroll.")],
    config: ConfigOption = Path(DEFAULT_CONFIG),
    log_level: LogLevelOption = None,
    device: DeviceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Regenerate one person's embedding from their reference image."""
    app_config = load(config, build_overrides(log_level=log_level, device=device))
    engine = engine_for(app_config)
    try:
        report = engine.build_gallery(force=True, only=[person])
        if report.failed:
            fail(f"{person}: {report.failed[person]}")
        emit(report.to_dict(), json_output, f"Rebuilt embedding for '{person}'.")
    finally:
        engine.close()


@app.command("list")
@handle_errors
def list_people(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    log_level: LogLevelOption = None,
    json_output: JsonOption = False,
) -> None:
    """List registered people and the state of their gallery entries."""
    app_config = load(config, build_overrides(log_level=log_level or "WARNING"))
    # Model loading is unnecessary just to read the cache.
    engine = engine_for(app_config, load_models=False)
    try:
        engine.gallery.load()
        rows = engine.gallery.summary()
        if not rows:
            emit([], json_output, "No identities configured. Add entries under 'people:'.")
            return
        width = max(len(str(row["id"])) for row in rows)
        lines = [f"{'ID'.ljust(width)}  {'NAME':<24} {'TITLE':<18} {'DIM':>5}  STATE"]
        for row in rows:
            state = (
                "disabled"
                if not row["enabled"]
                else ("ready" if row["has_embedding"] else "NOT ENROLLED")
            )
            warnings = row["quality_warnings"]
            suffix = f"  ({len(warnings)} quality warning(s))" if warnings else ""
            lines.append(
                f"{str(row['id']).ljust(width)}  {str(row['name'])[:24]:<24} "
                f"{str(row['title'])[:18]:<18} {str(row['embedding_dimension'] or '-'):>5}  "
                f"{state}{suffix}"
            )
        emit(rows, json_output, "\n".join(lines))
    finally:
        engine.close()


@app.command("show")
@handle_errors
def show(
    person: Annotated[str, typer.Argument(help="Person id to inspect.")],
    config: ConfigOption = Path(DEFAULT_CONFIG),
    json_output: JsonOption = True,
) -> None:
    """Show the stored metadata for one identity (never the raw embedding)."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    engine = engine_for(app_config, load_models=False)
    try:
        stored = engine.gallery.store.get(person)
        if stored is None:
            fail(f"no gallery entry for '{person}'. Run: python main.py gallery build")
        emit(stored.metadata, json_output, str(stored.metadata))
    finally:
        engine.close()


@app.command("remove")
@handle_errors
def remove(
    person: Annotated[str, typer.Argument(help="Person id to remove from the gallery.")],
    config: ConfigOption = Path(DEFAULT_CONFIG),
    json_output: JsonOption = False,
) -> None:
    """Delete one identity's cached embedding and metadata."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    engine = engine_for(app_config, load_models=False)
    try:
        removed = engine.gallery.remove(person)
        emit(
            {"removed": removed, "id": person},
            json_output,
            f"Removed '{person}' from the gallery."
            if removed
            else f"No gallery entry for '{person}'.",
        )
    finally:
        engine.close()


@app.command("clear")
@handle_errors
def clear(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not prompt.")] = False,
    json_output: JsonOption = False,
) -> None:
    """Delete every cached embedding (reference images are untouched)."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    engine = engine_for(app_config, load_models=False)
    try:
        if not yes:
            typer.confirm(
                f"Delete all cached embeddings in {engine.paths.embeddings_dir}?", abort=True
            )
        count = engine.gallery.clear()
        emit({"removed": count}, json_output, f"Removed {count} cached embedding(s).")
    finally:
        engine.close()
