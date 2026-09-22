"""Identity gallery: persistence, cache invalidation and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.core.exceptions import EmbeddingStoreError, GalleryError
from src.core.types import BBox, Detection
from src.identity.embedding_store import (
    InMemoryEmbeddingStore,
    LocalNumpyStore,
    StoredEmbedding,
)
from src.identity.enrollment import Enroller
from src.identity.gallery import IdentityGallery, file_fingerprint
from src.reid.preprocess import PersonCropPreprocessor
from src.utils.image import imwrite
from tests.conftest import FakeDetector, FakeEncoder, person_scene


@pytest.fixture
def people_images(tmp_path: Path) -> dict[str, Path]:
    return {
        "alice": imwrite(tmp_path / "alice.jpg", person_scene((200, 60, 60))),
        "bob": imwrite(tmp_path / "bob.jpg", person_scene((60, 200, 60))),
    }


@pytest.fixture
def gallery_config(base_config_dict: dict, tmp_path: Path, people_images: dict[str, Path]):
    base_config_dict["people"] = [
        {"id": "alice", "name": "Alice A", "title": "Manager",
         "image_path": str(people_images["alice"])},
        {"id": "bob", "name": "Bob B", "title": "", "image_path": str(people_images["bob"])},
    ]
    return config_from_dict(base_config_dict, base_dir=tmp_path)


def build_parts(config):
    detector = FakeDetector([[Detection(bbox=BBox(200, 80, 320, 440), confidence=0.9)]])
    encoder = FakeEncoder()
    paths = ProjectPaths.from_config(config)
    paths.ensure(paths.gallery_dir, paths.embeddings_dir, paths.metadata_dir)
    gallery = IdentityGallery(config, paths)
    enroller = Enroller(config, detector, encoder, PersonCropPreprocessor())
    return gallery, enroller, encoder, paths


class TestEmbeddingStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        vector = np.arange(8, dtype=np.float32)
        store.put(StoredEmbedding("x", vector, {"name": "X"}))

        fresh = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        loaded = fresh.get("x")
        assert loaded is not None
        assert np.allclose(loaded.embedding, vector)
        assert loaded.metadata["name"] == "X"
        assert loaded.metadata["embedding_dimension"] == 8
        assert loaded.metadata["created_at"]

    def test_files_land_where_the_documented_layout_says(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        store.put(StoredEmbedding("john_doe", np.ones(4, dtype=np.float32)))
        assert (tmp_path / "emb" / "john_doe.npy").exists()
        assert (tmp_path / "meta" / "john_doe.json").exists()

    def test_delete_and_listing(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        store.put(StoredEmbedding("a", np.ones(4, dtype=np.float32)))
        store.put(StoredEmbedding("b", np.ones(4, dtype=np.float32)))
        assert store.ids() == ["a", "b"]
        assert store.delete("a") is True
        assert store.delete("a") is False
        assert store.ids() == ["b"]

    def test_corrupt_embedding_is_discarded_not_fatal(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        store.put(StoredEmbedding("x", np.ones(4, dtype=np.float32)))
        (tmp_path / "emb" / "x.npy").write_bytes(b"not a numpy file")

        fresh = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        assert fresh.get("x") is None                       # dropped, not raised
        assert not (tmp_path / "emb" / "x.npy").exists()     # cleared for rebuild

    def test_corrupt_metadata_is_discarded(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        store.put(StoredEmbedding("x", np.ones(4, dtype=np.float32)))
        (tmp_path / "meta" / "x.json").write_text("{ broken", encoding="utf-8")
        assert LocalNumpyStore(tmp_path / "emb", tmp_path / "meta").get("x") is None

    def test_empty_embedding_is_refused(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        with pytest.raises(EmbeddingStoreError, match="empty embedding"):
            store.put(StoredEmbedding("x", np.zeros((0,), dtype=np.float32)))

    def test_matrix_is_row_aligned_with_ids(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        store.put(StoredEmbedding("b", np.array([0, 1, 0, 0], dtype=np.float32)))
        store.put(StoredEmbedding("a", np.array([1, 0, 0, 0], dtype=np.float32)))
        matrix, ids = store.matrix()
        assert ids == ["a", "b"]
        assert matrix.shape == (2, 4)
        assert matrix[ids.index("a"), 0] == 1.0

    def test_mixed_dimensions_are_actionable(self, tmp_path: Path) -> None:
        store = LocalNumpyStore(tmp_path / "emb", tmp_path / "meta")
        store.put(StoredEmbedding("a", np.ones(4, dtype=np.float32)))
        store.put(StoredEmbedding("b", np.ones(8, dtype=np.float32)))
        with pytest.raises(EmbeddingStoreError, match="rebuild the gallery"):
            store.matrix()

    def test_in_memory_store_satisfies_the_same_contract(self) -> None:
        store = InMemoryEmbeddingStore()
        store.put(StoredEmbedding("x", np.ones(4, dtype=np.float32)))
        assert "x" in store
        assert len(store) == 1
        assert store.matrix()[1] == ["x"]


class TestGalleryBuild:
    def test_builds_every_configured_person(self, gallery_config) -> None:
        gallery, enroller, encoder, paths = build_parts(gallery_config)
        report = gallery.build(enroller, encoder.info)

        assert report.ok
        assert sorted(report.enrolled) == ["alice", "bob"]
        assert report.reused == []
        assert len(gallery.active_identities) == 2
        assert gallery.view.matrix.shape == (2, 32)
        assert (paths.embeddings_dir / "alice.npy").exists()

    def test_metadata_records_the_documented_fields(self, gallery_config) -> None:
        gallery, enroller, encoder, paths = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)
        metadata = json.loads((paths.metadata_dir / "alice.json").read_text())

        for key in (
            "id", "name", "title", "image_path", "embedding_file",
            "embedding_dimension", "created_at", "updated_at",
        ):
            assert key in metadata, key
        assert metadata["name"] == "Alice A"
        assert metadata["embedding_dimension"] == 32
        assert metadata["encoder_fingerprint"] == encoder.info.fingerprint

    def test_second_build_reuses_the_cache(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)
        calls_after_first = encoder.calls

        gallery2, enroller2, encoder2, _ = build_parts(gallery_config)
        report = gallery2.build(enroller2, encoder2.info)
        assert sorted(report.reused) == ["alice", "bob"]
        assert encoder2.calls == 0            # no embeddings recomputed
        assert calls_after_first > 0

    def test_force_rebuild_recomputes(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)

        gallery2, enroller2, encoder2, _ = build_parts(gallery_config)
        report = gallery2.build(enroller2, encoder2.info, force=True)
        assert sorted(report.enrolled) == ["alice", "bob"]
        assert encoder2.calls > 0

    def test_changed_reference_image_invalidates_the_cache(
        self, gallery_config, people_images
    ) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)

        imwrite(people_images["alice"], person_scene((10, 10, 250)))
        gallery2, enroller2, encoder2, _ = build_parts(gallery_config)
        report = gallery2.build(enroller2, encoder2.info)
        assert "alice" in report.enrolled     # re-enrolled
        assert "bob" in report.reused         # untouched

    def test_changed_reid_model_invalidates_every_entry(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)

        gallery2, enroller2, encoder2, _ = build_parts(gallery_config)
        different = FakeEncoder(dimension=64)
        enroller2._encoder = different  # noqa: SLF001 - simulating a model swap
        report = gallery2.build(enroller2, different.info)
        assert sorted(report.enrolled) == ["alice", "bob"]
        assert gallery2.view.matrix.shape == (2, 64)

    def test_disabled_people_are_skipped(self, base_config_dict, tmp_path, people_images) -> None:
        base_config_dict["people"] = [
            {"id": "alice", "name": "Alice", "image_path": str(people_images["alice"])},
            {"id": "bob", "name": "Bob", "image_path": str(people_images["bob"]),
             "enabled": False},
        ]
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        gallery, enroller, encoder, _ = build_parts(config)
        report = gallery.build(enroller, encoder.info)
        assert report.skipped == ["bob"]
        assert [i.id for i in gallery.active_identities] == ["alice"]

    def test_a_failed_person_does_not_stop_the_others(
        self, base_config_dict, tmp_path, people_images
    ) -> None:
        base_config_dict["people"] = [
            {"id": "alice", "name": "Alice", "image_path": str(people_images["alice"])},
            {"id": "ghost", "name": "Ghost", "image_path": str(tmp_path / "missing.jpg")},
        ]
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        gallery, enroller, encoder, _ = build_parts(config)
        report = gallery.build(enroller, encoder.info)
        assert report.enrolled == ["alice"]
        assert "ghost" in report.failed
        assert not report.ok

    def test_building_an_unknown_person_id_is_actionable(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        with pytest.raises(GalleryError, match="unknown person id"):
            gallery.build(enroller, encoder.info, only=["nobody"])

    def test_removing_a_person_from_config_prunes_the_cache(
        self, gallery_config, base_config_dict, tmp_path, people_images
    ) -> None:
        gallery, enroller, encoder, paths = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)
        assert (paths.embeddings_dir / "bob.npy").exists()

        base_config_dict["people"] = [
            {"id": "alice", "name": "Alice", "image_path": str(people_images["alice"])}
        ]
        trimmed = config_from_dict(base_config_dict, base_dir=tmp_path)
        gallery2, enroller2, encoder2, paths2 = build_parts(trimmed)
        gallery2.build(enroller2, encoder2.info)
        assert not (paths2.embeddings_dir / "bob.npy").exists()


class TestGalleryLifecycle:
    def test_load_without_enrolling(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)

        gallery2, _, encoder2, _ = build_parts(gallery_config)
        assert gallery2.load(encoder2.info) == 2
        assert gallery2.view.size == 2

    def test_enable_and_disable_affects_the_matching_view(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)
        gallery.set_enabled("bob", False)
        assert gallery.view.ids == ["alice"]
        gallery.set_enabled("bob", True)
        assert gallery.view.ids == ["alice", "bob"]

    def test_remove_and_clear(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)
        assert gallery.remove("alice") is True
        assert gallery.view.ids == ["bob"]
        assert gallery.clear() == 1
        assert gallery.view.is_empty

    def test_summary_never_exposes_raw_embeddings(self, gallery_config) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)
        rows = gallery.summary()
        assert {row["id"] for row in rows} == {"alice", "bob"}
        for row in rows:
            assert "embedding" not in row
            assert row["embedding_dimension"] == 32

    def test_display_fields_refresh_from_configuration(
        self, gallery_config, base_config_dict, tmp_path, people_images
    ) -> None:
        gallery, enroller, encoder, _ = build_parts(gallery_config)
        gallery.build(enroller, encoder.info)

        base_config_dict["people"] = [
            {"id": "alice", "name": "Alice Renamed", "title": "Director",
             "image_path": str(people_images["alice"])},
            {"id": "bob", "name": "Bob B", "image_path": str(people_images["bob"])},
        ]
        renamed = config_from_dict(base_config_dict, base_dir=tmp_path)
        gallery2, enroller2, encoder2, _ = build_parts(renamed)
        report = gallery2.build(enroller2, encoder2.info)
        assert "alice" in report.reused          # no re-embedding for a rename
        assert gallery2.get("alice").name == "Alice Renamed"
        assert gallery2.get("alice").title == "Director"


def test_file_fingerprint_changes_with_content(tmp_path: Path) -> None:
    path = imwrite(tmp_path / "f.jpg", person_scene((10, 20, 30)))
    first = file_fingerprint(path)
    imwrite(path, person_scene((200, 20, 30)))
    assert file_fingerprint(path) != first
    assert file_fingerprint(tmp_path / "absent.jpg") == ""
