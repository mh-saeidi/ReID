"""CLI entry point."""

from __future__ import annotations

import typer

from src.cli import gallery_cmd, process_cmd, system_cmd, tools_cmd

app = typer.Typer(
    name="person-reid",
    help=(
        "One-shot person re-identification: YOLO26 person detection + ReID "
        "appearance embeddings matched against a gallery of registered people."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

app.add_typer(gallery_cmd.app, name="gallery")
process_cmd.register(app)
tools_cmd.register(app)
system_cmd.register(app)


@app.command("version")
def version() -> None:
    """Print the application version and the versions of key dependencies."""
    import importlib.metadata as metadata

    from src.config.schema import ApplicationConfig

    typer.echo(f"{ApplicationConfig().name} {ApplicationConfig().version}")
    for package in ("ultralytics", "torch", "opencv-python", "numpy", "onnxruntime", "pydantic"):
        try:
            typer.echo(f"  {package:<16} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            typer.echo(f"  {package:<16} (not installed)")


def run() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
