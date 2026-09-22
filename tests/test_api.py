"""Optional REST API layer.

Skipped when FastAPI is not installed: the CV engine and the CLI must work
without it, and that independence is itself asserted below.
"""

from __future__ import annotations

import gc

import pytest
import yaml

from src.config.loader import load_config
from tests.conftest import (
    PROJECT_ROOT,
    requires_demo,
    requires_face_models,
    requires_models,
)

fastapi = pytest.importorskip("fastapi", reason="fastapi is an optional dependency")
pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.server import create_app  # noqa: E402

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    requires_models(),
    requires_face_models(),
    requires_demo(),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("api")
    base = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    base["models"] = {k: str(PROJECT_ROOT / v) for k, v in base["models"].items()}
    base["people"] = [
        {**p, "image_path": str(PROJECT_ROOT / p["image_path"])} for p in base["people"]
    ]
    base["gallery"] = {**base.get("gallery", {}), "directory": str(output_root / "gallery")}
    base["output"] = {
        **base.get("output", {}),
        "directory": str(output_root / "output"),
        "save_snapshots": False,
        "save_video": False,
    }
    base["display"] = {"show_window": False}
    path = output_root / "config.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")

    with TestClient(create_app(load_config(path))) as test_client:
        yield test_client
    gc.collect()   # release native model handles before interpreter teardown


def test_health_describes_the_loaded_system(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["reid"]["embedding_dimension"] > 0
    assert payload["gallery"]["identities"] == 2


def test_people_listing_and_lookup(client) -> None:
    listing = client.get("/people")
    assert listing.status_code == 200
    assert {row["id"] for row in listing.json()} == {"person_a", "person_b"}

    one = client.get("/people/person_a")
    assert one.status_code == 200
    assert one.json()["name"] == "Person A"
    assert "embedding" not in one.json()      # never expose raw biometrics

    assert client.get("/people/nobody").status_code == 404


def test_process_image_returns_open_set_results(client) -> None:
    image = PROJECT_ROOT / "data" / "demo" / "test_images" / "03_a_and_b.jpg"
    with image.open("rb") as handle:
        response = client.post(
            "/process/image", files={"file": (image.name, handle, "image/jpeg")}
        )
    assert response.status_code == 200
    payload = response.json()
    identities = [d["identity_id"] for d in payload["detections"]]
    assert "person_a" in identities
    assert "person_b" in identities


def test_gallery_build_endpoint(client) -> None:
    response = client.post("/gallery/build")
    assert response.status_code == 200
    assert response.json()["total_active"] == 2


def test_events_endpoint(client) -> None:
    assert client.get("/events?limit=5").status_code == 200


def test_the_engine_does_not_depend_on_the_api_layer() -> None:
    """Importing the CV engine must not pull in FastAPI."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import src.pipeline.engine, src.cli.main; "
            "assert 'fastapi' not in sys.modules, 'the engine imported fastapi'; "
            "print('ok')",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
