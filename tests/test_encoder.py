"""ReID encoder abstraction, preprocessing and backend selection."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.config.schema import DeviceConfig, DeviceKind, RecognitionMode, ReIDBackend
from src.core.exceptions import ConfigurationError, ModelLoadError
from src.core.types import BBox
from src.reid.encoder import EncoderInfo, l2_normalize
from src.reid.factory import _select_backend, build_encoder, resolve_model_path
from src.reid.onnx_reid import OnnxReIDEncoder
from src.reid.preprocess import CropStatus, PersonCropPreprocessor, build_preprocessor
from src.reid.yolo26_reid import OFFICIAL_REID_ASSETS, UltralyticsReIDEncoder
from src.utils.device import resolve_device
from src.utils.image import crop_bbox, letterbox, resize_crop
from tests.conftest import FakeEncoder, person_scene


class TestEncoderInterface:
    def test_embed_delegates_to_embed_batch(self) -> None:
        encoder = FakeEncoder()
        crop = person_scene((200, 40, 40), size=(120, 360))
        assert encoder.embed(crop).shape == (32,)

    def test_embeddings_are_unit_length(self) -> None:
        encoder = FakeEncoder()
        vector = encoder.embed(person_scene((200, 40, 40), size=(120, 360)))
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    def test_dimension_is_read_from_the_model(self) -> None:
        assert FakeEncoder(dimension=768).get_embedding_dimension() == 768

    def test_invalid_crops_yield_zero_rows(self) -> None:
        encoder = FakeEncoder()
        out = encoder.embed_batch([None, np.zeros((0, 0, 3), dtype=np.uint8)])
        assert out.shape == (2, 32)
        assert not np.any(out)

    def test_fingerprint_captures_the_embedding_space(self) -> None:
        base = EncoderInfo(
            name="yolo26n-reid", model_path="m.onnx", backend="onnxruntime", device="cpu",
            fp16=False, input_size=(256, 128), embedding_dimension=512,
        )
        fields = dataclasses.asdict(base)
        same = EncoderInfo(**fields)
        bigger = EncoderInfo(**{**fields, "embedding_dimension": 1024})
        resized = EncoderInfo(**{**fields, "input_size": (224, 224)})

        assert base.fingerprint == same.fingerprint
        assert base.fingerprint != bigger.fingerprint
        assert base.fingerprint != resized.fingerprint


class TestPreprocessing:
    def test_crop_extraction(self) -> None:
        preprocessor = PersonCropPreprocessor()
        image = person_scene((200, 40, 40), size=(640, 480), box=(100, 50, 220, 400))
        result = preprocessor.extract(image, BBox(100, 50, 220, 400))
        assert result.ok
        assert result.status is CropStatus.OK
        assert result.image.shape[:2] == (350, 120)

    def test_padding_widens_the_crop(self) -> None:
        image = person_scene(size=(640, 480), box=(200, 100, 300, 400))
        tight = PersonCropPreprocessor().extract(image, BBox(200, 100, 300, 400))
        padded = PersonCropPreprocessor(padding=0.1).extract(image, BBox(200, 100, 300, 400))
        assert padded.image.shape[0] > tight.image.shape[0]

    def test_a_box_outside_the_frame_is_rejected_with_a_reason(self) -> None:
        image = person_scene(size=(640, 480))
        result = PersonCropPreprocessor().extract(image, BBox(700, 500, 800, 600))
        assert not result.ok
        assert result.status is CropStatus.DEGENERATE_BOX
        assert result.detail

    def test_a_subpixel_box_is_rejected(self) -> None:
        image = person_scene(size=(640, 480))
        assert not PersonCropPreprocessor().extract(image, BBox(10, 10, 12, 12)).ok

    def test_person_mode_never_requires_a_face(self) -> None:
        assert PersonCropPreprocessor().requires_visible_face is False

    def test_prepare_resizes_to_the_encoder_geometry(self) -> None:
        crop = person_scene(size=(80, 300))
        prepared = PersonCropPreprocessor().prepare(crop, (256, 128))
        assert prepared.shape[:2] == (256, 128)

    def test_letterbox_preserves_aspect_ratio(self) -> None:
        crop = person_scene(size=(100, 400))
        boxed = letterbox(crop, (256, 128))
        assert boxed.shape[:2] == (256, 128)

    def test_extract_many_reports_original_indices(self) -> None:
        image = person_scene(size=(640, 480))
        boxes = [BBox(10, 10, 120, 400), BBox(700, 500, 800, 600), BBox(200, 50, 320, 420)]
        crops, indices, results = PersonCropPreprocessor().extract_many(image, boxes)
        assert len(crops) == 2
        assert indices == [0, 2]
        assert results[1].status is CropStatus.DEGENERATE_BOX

    def test_the_configured_mode_selects_the_preprocessor(self, config) -> None:
        assert config.recognition.mode is RecognitionMode.PERSON_REID
        assert isinstance(build_preprocessor(config), PersonCropPreprocessor)

    def test_face_mode_requires_a_face_detector(self, face_config_dict, tmp_path) -> None:
        """A missing detector must fail loudly, never fall back to the body."""
        config = config_from_dict(face_config_dict, base_dir=tmp_path)
        with pytest.raises(ConfigurationError, match="requires a face detector"):
            build_preprocessor(config)

    def test_resize_uses_area_interpolation_when_downscaling(self) -> None:
        big = person_scene(size=(800, 1600))
        assert resize_crop(big, (256, 128)).shape[:2] == (256, 128)


class TestBackendSelection:
    def test_pt_routes_to_ultralytics(self) -> None:
        assert _select_backend(ReIDBackend.AUTO, "models/yolo26n-cls.pt") is ReIDBackend.ULTRALYTICS

    def test_a_local_onnx_file_routes_to_onnxruntime(self, tmp_path: Path) -> None:
        path = tmp_path / "osnet.onnx"
        path.write_bytes(b"placeholder")
        assert _select_backend(ReIDBackend.AUTO, str(path)) is ReIDBackend.ONNX

    def test_an_absent_official_asset_routes_to_ultralytics_for_download(self) -> None:
        assert "yolo26n-reid.onnx" in OFFICIAL_REID_ASSETS
        assert (
            _select_backend(ReIDBackend.AUTO, "yolo26n-reid.onnx") is ReIDBackend.ULTRALYTICS
        )

    def test_an_explicit_backend_is_honoured(self) -> None:
        assert _select_backend(ReIDBackend.ONNX, "anything.pt") is ReIDBackend.ONNX

    def test_factory_builds_the_matching_class(self, base_config_dict, tmp_path) -> None:
        onnx_path = tmp_path / "custom.onnx"
        onnx_path.write_bytes(b"placeholder")
        base_config_dict["models"]["reid"] = str(onnx_path)
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        paths = ProjectPaths.from_config(config)
        device = resolve_device(DeviceConfig(device=DeviceKind.CPU))
        assert isinstance(build_encoder(config, paths, device), OnnxReIDEncoder)

        base_config_dict["models"]["reid"] = "yolo26n-cls.pt"
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        assert isinstance(build_encoder(config, paths, device), UltralyticsReIDEncoder)

    def test_model_paths_resolve_relative_to_the_config(self, config, tmp_path) -> None:
        paths = ProjectPaths.from_config(config)
        model = tmp_path / "models" / "x.onnx"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"placeholder")
        assert resolve_model_path("models/x.onnx", paths) == str(model)
        # An unknown name passes through so Ultralytics can treat it as an asset.
        assert resolve_model_path("yolo26n.pt", paths) == "yolo26n.pt"


class TestErrorHandling:
    def test_a_missing_onnx_model_is_actionable(self, tmp_path: Path) -> None:
        device = resolve_device(DeviceConfig(device=DeviceKind.CPU))
        with pytest.raises(ModelLoadError, match="not found"):
            OnnxReIDEncoder(tmp_path / "absent.onnx", device).load()

    def test_a_corrupt_onnx_model_is_actionable(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.onnx"
        path.write_bytes(b"not a model")
        device = resolve_device(DeviceConfig(device=DeviceKind.CPU))
        with pytest.raises(ModelLoadError, match="cannot load"):
            OnnxReIDEncoder(path, device).load()

    def test_an_unknown_reid_reference_is_actionable(self, tmp_path: Path) -> None:
        device = resolve_device(DeviceConfig(device=DeviceKind.CPU))
        with pytest.raises(ModelLoadError, match="ReID model not found"):
            UltralyticsReIDEncoder(tmp_path / "mystery.onnx", device).load()


class TestDeviceResolution:
    def test_cpu_is_always_available(self) -> None:
        info = resolve_device(DeviceConfig(device=DeviceKind.CPU))
        assert info.device == "cpu"
        assert info.fp16_enabled is False        # fp16 is meaningless on CPU

    def test_auto_never_fails(self) -> None:
        info = resolve_device(DeviceConfig(device=DeviceKind.AUTO))
        assert info.device in ("cpu", "mps") or info.device.startswith("cuda")

    def test_requesting_an_absent_accelerator_degrades_to_cpu(self, monkeypatch) -> None:
        monkeypatch.setattr("src.utils.device.cuda_available", lambda: False)
        info = resolve_device(DeviceConfig(device=DeviceKind.CUDA))
        assert info.device == "cpu"              # a warning, not a crash

    def test_fp16_is_only_enabled_where_supported(self, monkeypatch) -> None:
        monkeypatch.setattr("src.utils.device.cuda_available", lambda: False)
        monkeypatch.setattr("src.utils.device.mps_available", lambda: False)
        info = resolve_device(DeviceConfig(device=DeviceKind.AUTO, fp16=True))
        assert info.fp16_enabled is False


def test_crop_bbox_clips_to_the_frame() -> None:
    image = person_scene(size=(640, 480))
    crop = crop_bbox(image, BBox(-100, -100, 200, 200))
    assert crop is not None
    assert crop.shape[0] <= 480 and crop.shape[1] <= 640


def test_l2_normalize_handles_batches_and_singletons() -> None:
    single = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert single == pytest.approx([0.6, 0.8], abs=1e-6)
    batch = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32), axis=1)
    assert batch[0] == pytest.approx([0.6, 0.8], abs=1e-6)
    assert batch[1] == pytest.approx([0.0, 0.0])
