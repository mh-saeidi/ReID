"""CLI for the face identity engine: enrollment, calibration, evaluation, A/B."""

from __future__ import annotations

import json
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
    fail,
    handle_errors,
    load,
)

identity_app = typer.Typer(
    help="Face identity gallery: enrol, inspect and maintain registered people.",
    no_args_is_help=True,
)


def _system(app_config, load_models: bool = True):
    from src.config.paths import ProjectPaths
    from src.identity.factory import build_face_identity_system
    from src.utils.device import resolve_device

    paths = ProjectPaths.from_config(app_config)
    return build_face_identity_system(
        app_config, paths, resolve_device(app_config.device), load=load_models
    ), paths



_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _roster_from_directory(root: Path) -> list:
    """Build a roster from ``<root>/<person_id>/<photo>``.

    Used by ``--from-dir`` so an enrollment directory -- the evaluation
    dataset's layout -- can be registered without editing the configuration.
    """
    from src.config.schema import PersonConfig

    if not root.is_dir():
        fail(f"--from-dir path is not a directory: {root}")

    roster = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        photos = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file()
        )
        if not photos:
            continue
        roster.append(
            PersonConfig(
                id=directory.name,
                name=directory.name.replace("_", " ").title(),
                image_path=str(photos[0]),
            )
        )
    return roster


@identity_app.command("build")
@handle_errors
def build_gallery(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Re-enrol everyone from their reference.")
    ] = False,
    person: Annotated[
        list[str] | None,
        typer.Option("--person", "-p", help="Limit to these person ids (repeatable)."),
    ] = None,
    from_dir: Annotated[
        Path | None,
        typer.Option(
            "--from-dir",
            help=(
                "Enrol from a directory of <person_id>/<photo> instead of the "
                "people listed in the configuration."
            ),
        ),
    ] = None,
    log_level: LogLevelOption = None,
    device: DeviceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Enrol every configured person from their passport photograph.

    Enrollment is face-first: the face detector locates the subject directly.
    No person detection is involved, because a passport photo is not what a
    person detector is for and requiring one can only reject valid references.

    With ``--from-dir`` the people come from the filesystem instead: one
    subdirectory per person, named by id, holding the reference photograph.
    That is the layout the evaluation dataset uses, so a gallery can be built
    from it without editing the configuration.
    """
    from src.identity.face_gallery import FaceEmbeddingRecord, FaceIdentity

    app_config = load(config, build_overrides(log_level=log_level, device=device))
    system, paths = _system(app_config)
    try:
        wanted = set(person) if person else None
        enrolled, reused, skipped, failed = [], [], [], {}
        roster = (
            _roster_from_directory(from_dir) if from_dir is not None
            else list(app_config.people)
        )
        if not roster:
            fail(
                f"no people to enrol from {from_dir}" if from_dir is not None
                else "no people are configured; add entries under 'people:' in "
                     "the configuration, or pass --from-dir"
            )

        for entry in roster:
            if wanted is not None and entry.id not in wanted:
                continue
            if not entry.enabled:
                skipped.append(entry.id)
                continue

            existing = system.gallery.get(entry.id)
            if existing is not None and not rebuild:
                reused.append(entry.id)
                continue

            reference = paths.resolve(entry.all_image_paths[0])
            try:
                result = system.enroller.enroll(entry.id, reference)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
                failed[entry.id] = str(exc)
                typer.secho(f"  {entry.id}: {exc}", fg=typer.colors.RED, err=True)
                continue

            identity = FaceIdentity(
                id=entry.id,
                name=entry.name,
                title=entry.title,
                enabled=entry.enabled,
                encoder_fingerprint=system.encoder.info.fingerprint,
                quality={
                    "warnings": [w.value for w in result.report.warnings],
                    "messages": result.report.messages,
                    "score": result.report.quality.to_dict() if result.report.quality else None,
                },
            )
            quality_score = (
                result.report.quality.overall if result.report.quality else 0.0
            )
            identity.embeddings["reference"] = FaceEmbeddingRecord(
                key="reference",
                vector=result.embedding,
                source="reference",
                quality=quality_score,
            )

            # The operator's original is never modified; this is a copy.
            copied = system.gallery.copy_reference_image(entry.id, reference)
            identity.reference_image = str(copied)
            system.gallery.save(identity)
            enrolled.append(entry.id)

        payload = {
            "enrolled": enrolled, "reused": reused,
            "skipped": skipped, "failed": failed,
            "identities": len(system.gallery.active),
            "embeddings": system.gallery.embedding_count,
        }
        lines = [
            f"Face gallery: {len(system.gallery.active)} identity(ies), "
            f"{system.gallery.embedding_count} embedding(s) in {system.gallery.root}",
            f"  enrolled : {', '.join(enrolled) or '-'}",
            f"  reused   : {', '.join(reused) or '-'}",
            f"  disabled : {', '.join(skipped) or '-'}",
        ]
        for pid, message in failed.items():
            lines.append(f"  FAILED   : {pid}: {message}")

        warned = [
            i for i in system.gallery.identities if i.quality.get("warnings")
        ]
        if warned:
            lines.append("")
            lines.append("  Reference photographs with quality warnings:")
            for identity in warned:
                lines.append(
                    f"    {identity.id}: {', '.join(identity.quality['warnings'])}"
                )
        emit(payload, json_output, "\n".join(lines))
        if failed:
            raise typer.Exit(1)
    finally:
        system.close()


@identity_app.command("list")
@handle_errors
def list_identities(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    json_output: JsonOption = False,
) -> None:
    """List registered people and their stored embeddings."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    system, _ = _system(app_config, load_models=True)
    try:
        rows = system.gallery.summary()
        if not rows:
            emit([], json_output,
                 "No identities registered. Run: python main.py identity build")
            return
        width = max(len(r["id"]) for r in rows)
        lines = [
            f"{'ID'.ljust(width)}  {'NAME':<22} {'EMB':>4} {'LIVE':>5} {'DIM':>5}  WARNINGS"
        ]
        for row in rows:
            warnings = ", ".join(row["quality_warnings"]) or "-"
            lines.append(
                f"{row['id'].ljust(width)}  {str(row['name'])[:22]:<22} "
                f"{row['embeddings']:>4} {row['live_embeddings']:>5} "
                f"{str(row['dimension'] or '-'):>5}  {warnings}"
            )
        emit(rows, json_output, "\n".join(lines))
    finally:
        system.close()


@identity_app.command("remove")
@handle_errors
def remove_identity(
    person: Annotated[str, typer.Argument(help="Person id to remove.")],
    config: ConfigOption = Path(DEFAULT_CONFIG),
) -> None:
    """Delete a person's embeddings and gallery directory."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    system, _ = _system(app_config)
    try:
        removed = system.gallery.remove(person)
        typer.echo(
            f"Removed '{person}' from the face gallery."
            if removed
            else f"No gallery entry for '{person}'."
        )
    finally:
        system.close()


def register(app: typer.Typer) -> None:
    """Attach the identity, calibration, evaluation and experiment commands."""
    app.add_typer(identity_app, name="identity")

    @app.command("calibrate")
    @handle_errors
    def calibrate(
        dataset: Annotated[
            Path, typer.Option("--dataset", "-d", help="Evaluation dataset directory.")
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="Where to write it.")
        ] = None,
        method: Annotated[
            str, typer.Option("--method", help="logistic or isotonic.")
        ] = "logistic",
        max_far: Annotated[
            float, typer.Option("--max-far", help="False-accept ceiling.")
        ] = 0.01,
        fraction: Annotated[
            float, typer.Option("--fraction", help="Share of samples used to fit.")
        ] = 0.4,
        json_output: JsonOption = False,
    ) -> None:
        """Fit the score-to-probability mapping and the acceptance thresholds.

        Fitted on a calibration split only. The remaining samples are left
        untouched so ``evaluate`` reports against data the thresholds never saw
        -- tuning and reporting on the same samples produces a number that does
        not survive new data.
        """
        from src.identity.calibration import fit_calibration
        from src.tools.face_evaluation import (
            FacePipeline,
            collect_calibration_samples,
            load_dataset,
            split_samples,
        )

        app_config = load(config, build_overrides(log_level="WARNING"))
        system, paths = _system(app_config)
        try:
            if system.gallery.is_empty:
                fail("the face gallery is empty; run 'identity build' first")

            _, samples, manifest = load_dataset(dataset)
            if not samples:
                fail(f"no query samples found under {dataset}")

            calibration_split, _ = split_samples(samples, calibration_fraction=fraction)
            pipeline = FacePipeline(
                system.detector, system.encoder, system.gallery, system.engine,
                chip_size=system.chip_size,
            )
            collected = collect_calibration_samples(pipeline, calibration_split)
            model = fit_calibration(
                collected, method=method, max_far=max_far, dataset=str(dataset),
                encoder_fingerprint=system.encoder.info.fingerprint,
            )
            if manifest.get("synthetic"):
                model.notes.append(
                    "Fitted on SYNTHETIC data. The thresholds are valid for this "
                    "dataset only; refit on real captures before deployment."
                )

            target = output or paths.resolve(app_config.face_identity.calibration_file)
            model.save(target)

            lines = [f"Calibration written to {target}", ""]
            for name, condition in sorted(model.conditions.items()):
                state = "usable" if condition.usable else "TOO FEW SAMPLES"
                lines.append(
                    f"  {name:<10} {state:<16} threshold {condition.recommended_threshold:.3f}"
                    f"  (EER {condition.equal_error_rate:.3f} at "
                    f"{condition.equal_error_threshold:.3f})"
                )
                lines.append(
                    f"             genuine n={condition.genuine_count} "
                    f"mean={condition.genuine_stats.get('mean', 0):.3f} | "
                    f"impostor n={condition.impostor_count} "
                    f"max={condition.impostor_stats.get('max', 0):.3f}"
                )
            for note in model.notes:
                lines.append(f"  NOTE: {note}")
            emit(model.to_dict(), json_output, "\n".join(lines))
        finally:
            system.close()

    @app.command("evaluate-faces")
    @handle_errors
    def evaluate_faces(
        dataset: Annotated[
            Path, typer.Option("--dataset", "-d", help="Evaluation dataset directory.")
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="Write the JSON report.")
        ] = None,
        fraction: Annotated[
            float, typer.Option("--fraction", help="Calibration share to exclude.")
        ] = 0.4,
        all_samples: Annotated[
            bool,
            typer.Option("--all", help="Evaluate every sample, including the "
                                       "calibration split (not a clean measurement)."),
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Measure identification accuracy per condition on the test split."""
        from src.tools.face_evaluation import (
            FacePipeline,
            evaluate,
            load_dataset,
            split_samples,
        )

        app_config = load(config, build_overrides(log_level="WARNING"))
        system, _ = _system(app_config)
        try:
            if system.gallery.is_empty:
                fail("the face gallery is empty; run 'identity build' first")

            enrollment, samples, manifest = load_dataset(dataset)
            if not samples:
                fail(f"no query samples found under {dataset}")

            if all_samples:
                test = list(samples)
            else:
                _, test = split_samples(samples, calibration_fraction=fraction)

            pipeline = FacePipeline(
                system.detector, system.encoder, system.gallery, system.engine,
                chip_size=system.chip_size,
            )
            caveats = []
            if manifest.get("warning"):
                caveats.append(manifest["warning"])
            if all_samples:
                caveats.append(
                    "--all was used, so thresholds were fitted on some of these "
                    "same samples; this is not a clean held-out measurement."
                )
            if system.calibration is None:
                caveats.append(
                    "No calibration is loaded, so the fallback threshold was "
                    "used and no identity confidence was reported."
                )

            report = evaluate(
                pipeline, test, dataset=str(dataset),
                registered=sorted(i.id for i in system.gallery.active),
                synthetic=bool(manifest.get("synthetic")), caveats=caveats,
            )
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
                )
                typer.echo(f"Report written to {output}\n")
            emit(report.to_dict(), json_output, report.render())
        finally:
            system.close()

    @app.command("experiments")
    @handle_errors
    def experiments(
        dataset: Annotated[
            Path, typer.Option("--dataset", "-d", help="Evaluation dataset directory.")
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="Write the JSON report.")
        ] = None,
        json_output: JsonOption = False,
    ) -> None:
        """Run the A/B ladder: each variant adds one component to the last."""
        from src.tools.ab_experiments import run_ab, write_report
        from src.tools.face_evaluation import load_dataset, split_samples

        app_config = load(config, build_overrides(log_level="WARNING"))
        system, _ = _system(app_config)
        try:
            if system.gallery.is_empty:
                fail("the face gallery is empty; run 'identity build' first")
            _, samples, manifest = load_dataset(dataset)
            calibration_split, test = split_samples(samples)
            report = run_ab(
                system.detector, system.encoder, system.gallery,
                calibration_split, test,
                dataset=str(dataset),
                synthetic=bool(manifest.get("synthetic")),
                caveats=[manifest["warning"]] if manifest.get("warning") else [],
                registered=sorted(i.id for i in system.gallery.active),
                progress=lambda key: typer.echo(f"  running {key} ..."),
            )
            if output is not None:
                write_report(report, output)
                typer.echo(f"\nReport written to {output}")
            emit(report.to_dict(), json_output, "\n" + report.render())
        finally:
            system.close()
