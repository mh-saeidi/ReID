"""Optional FastAPI layer.

The CV engine does not depend on this module: the CLI works with FastAPI absent,
and this file only ever imports the engine's public surface. It exists to prove
the core is API-shaped, and to give a UI something to talk to.

Run it with::

    pip install fastapi "uvicorn[standard]" python-multipart
    python main.py serve --config config.yaml
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any

from src.api.schemas import (
    BuildResponse,
    DetectionOut,
    EventOut,
    HealthResponse,
    PersonIn,
    PersonOut,
    PersonUpdate,
    ProcessImageResponse,
)
from src.config.schema import AppConfig, PersonConfig
from src.core.exceptions import ReIDSystemError
from src.pipeline.engine import Engine, build_engine
from src.pipeline.image_runner import ImageRunner
from src.sources.image import ImageSource
from src.utils.logging import get_logger

logger = get_logger(__name__)

# FastAPI is imported at module scope rather than inside create_app: this file
# uses postponed annotation evaluation, and FastAPI resolves route type hints
# against the module globals, so `UploadFile` has to be visible there. The
# dependency stays optional -- nothing imports this module unless `serve` runs.
try:
    from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False

_MISSING_FASTAPI = (
    "fastapi is required for the API layer: "
    "pip install fastapi 'uvicorn[standard]' python-multipart"
)


def create_app(config: AppConfig, engine: Engine | None = None):
    """Build the FastAPI application around an engine instance."""
    if not FASTAPI_AVAILABLE:  # pragma: no cover - optional dependency
        raise ImportError(_MISSING_FASTAPI)

    state: dict[str, Any] = {"engine": engine}

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        """Release model handles when the server stops."""
        yield
        if state["engine"] is not None:
            state["engine"].close()

    app = FastAPI(
        title=config.application.name,
        version=config.application.version,
        description=(
            "One-shot person re-identification. Processing is local; no image "
            "or embedding leaves this host."
        ),
        lifespan=lifespan,
    )

    def get_engine() -> Engine:
        if state["engine"] is None:
            state["engine"] = build_engine(config)
            state["engine"].ensure_gallery()
        return state["engine"]

    def _person_out(row: dict[str, Any]) -> PersonOut:
        return PersonOut(**{k: row[k] for k in PersonOut.model_fields if k in row})

    # ------------------------------------------------------------------ health
    @app.get("/health", response_model=HealthResponse)
    def health(eng: Engine = Depends(get_engine)) -> HealthResponse:
        info = eng.describe()
        return HealthResponse(
            application=info["application"],
            version=info["version"],
            device=info["device"],
            detector=info["detector"],
            reid=info["reid"],
            gallery=info["gallery"],
        )

    # ------------------------------------------------------------------ people
    @app.get("/people", response_model=list[PersonOut])
    def list_people(eng: Engine = Depends(get_engine)) -> list[PersonOut]:
        return [_person_out(row) for row in eng.gallery.summary()]

    @app.get("/people/{person_id}", response_model=PersonOut)
    def get_person(person_id: str, eng: Engine = Depends(get_engine)) -> PersonOut:
        for row in eng.gallery.summary():
            if row["id"] == person_id:
                return _person_out(row)
        raise HTTPException(status_code=404, detail=f"unknown person '{person_id}'")

    @app.post("/people", response_model=PersonOut, status_code=201)
    def create_person(payload: PersonIn, eng: Engine = Depends(get_engine)) -> PersonOut:
        if any(p.id == payload.id for p in eng.config.people):
            raise HTTPException(status_code=409, detail=f"person '{payload.id}' already exists")
        eng.config.people = [*eng.config.people, PersonConfig(**payload.model_dump())]
        try:
            report = eng.build_gallery(only=[payload.id])
        except ReIDSystemError as exc:
            eng.config.people = [p for p in eng.config.people if p.id != payload.id]
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if report.failed:
            eng.config.people = [p for p in eng.config.people if p.id != payload.id]
            raise HTTPException(status_code=400, detail=report.failed[payload.id])
        return get_person(payload.id, eng)

    @app.put("/people/{person_id}", response_model=PersonOut)
    def update_person(
        person_id: str, payload: PersonUpdate, eng: Engine = Depends(get_engine)
    ) -> PersonOut:
        people = list(eng.config.people)
        for index, person in enumerate(people):
            if person.id != person_id:
                continue
            data = person.model_dump()
            data.update({k: v for k, v in payload.model_dump().items() if v is not None})
            people[index] = PersonConfig(**data)
            eng.config.people = people
            report = eng.build_gallery(force=True, only=[person_id])
            if report.failed:
                raise HTTPException(status_code=400, detail=report.failed[person_id])
            return get_person(person_id, eng)
        raise HTTPException(status_code=404, detail=f"unknown person '{person_id}'")

    @app.delete("/people/{person_id}", status_code=204)
    def delete_person(person_id: str, eng: Engine = Depends(get_engine)) -> None:
        eng.config.people = [p for p in eng.config.people if p.id != person_id]
        eng.gallery.remove(person_id)

    # ----------------------------------------------------------------- gallery
    @app.post("/gallery/build", response_model=BuildResponse)
    def build_gallery(force: bool = False, eng: Engine = Depends(get_engine)) -> BuildResponse:
        return BuildResponse(**eng.build_gallery(force=force).to_dict())

    @app.post("/gallery/rebuild/{person_id}", response_model=BuildResponse)
    def rebuild_person(person_id: str, eng: Engine = Depends(get_engine)) -> BuildResponse:
        report = eng.build_gallery(force=True, only=[person_id])
        if report.failed:
            raise HTTPException(status_code=400, detail=report.failed[person_id])
        return BuildResponse(**report.to_dict())

    # -------------------------------------------------------------- processing
    @app.post("/process/image", response_model=ProcessImageResponse)
    async def process_image(
        file: UploadFile = File(...), eng: Engine = Depends(get_engine)
    ) -> ProcessImageResponse:
        if not eng.config.api.allow_upload:
            raise HTTPException(status_code=403, detail="api.allow_upload is false")
        payload = await file.read()
        limit = eng.config.api.max_upload_mb * 1024 * 1024
        if len(payload) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds api.max_upload_mb ({eng.config.api.max_upload_mb} MB)",
            )
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            runner = ImageRunner(eng, eng.pipeline(use_tracking=False))
            summary = runner.run(ImageSource(temp_path))
            if not summary.outcomes:
                message = next(iter(summary.failures.values()), "no result produced")
                raise HTTPException(status_code=400, detail=message)
            outcome = summary.outcomes[0]
            return ProcessImageResponse(
                source=file.filename or temp_path.name,
                width=outcome.result.width,
                height=outcome.result.height,
                num_detections=len(outcome.result.detections),
                num_recognized=len(outcome.result.recognized),
                detections=[
                    DetectionOut(**{k: d.to_dict()[k] for k in DetectionOut.model_fields})
                    for d in outcome.result.detections
                ],
                annotated_path=outcome.annotated_path,
                timings_ms={
                    k: round(v * 1000, 3) for k, v in outcome.result.timings.items()
                },
            )
        finally:
            temp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ events
    @app.get("/events", response_model=list[EventOut])
    def events(limit: int = 100, eng: Engine = Depends(get_engine)) -> list[EventOut]:
        return [
            EventOut(**{k: v for k, v in event.to_dict().items() if k in EventOut.model_fields})
            for event in eng.events.history(limit=limit)
        ]

    return app


def serve(config: AppConfig, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "uvicorn is required to serve the API: pip install 'uvicorn[standard]'"
        ) from exc
    logger.info("Starting the REST API", extra={"host": host, "port": port})
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")
