"""Backend selection, TensorRT engine metadata and fallback behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backends.engine_store import (
    EngineMetadata,
    EngineStore,
    build_expected_metadata,
    source_fingerprint,
)
from src.backends.selection import (
    ModelRole,
    select_backend,
    tensorrt_enabled,
)
from src.backends.tensorrt_builder import (
    BuildRequest,
    TensorRTBuilder,
    TensorRTUnavailable,
    _precision_flags,
    default_build_requests,
)
from src.config.loader import config_from_dict
from src.config.schema import InferenceBackend
from src.hardware.capabilities import (
    CudaInfo,
    GStreamerInfo,
    HostInfo,
    JetsonInfo,
    SystemCapabilities,
    TensorRTInfo,
)


def caps(
    *,
    jetson: bool = False,
    cuda: bool = False,
    tensorrt: bool = False,
    bindings: bool = True,
    trtexec: str | None = "/usr/src/tensorrt/bin/trtexec",
    gstreamer: bool = False,
    compute: str | None = "8.7",
    trt_version: str | None = "10.3.0",
) -> SystemCapabilities:
    """A synthetic capability set, so platform logic is testable anywhere."""
    return SystemCapabilities(
        host=HostInfo(system="Linux", machine="aarch64", cpu_count=6),
        jetson=JetsonInfo(
            is_jetson=jetson,
            model="NVIDIA Jetson Orin Nano Super" if jetson else None,
            jetpack_version="6.2" if jetson else None,
        ),
        cuda=CudaInfo(
            available=cuda, torch_cuda=cuda, runtime_version="12.6" if cuda else None,
            compute_capability=compute if cuda else None,
        ),
        tensorrt=TensorRTInfo(
            available=tensorrt,
            version=trt_version if tensorrt else None,
            python_bindings=tensorrt and bindings,
            trtexec=trtexec if tensorrt else None,
        ),
        gstreamer=GStreamerInfo(
            available=gstreamer, opencv_gstreamer=gstreamer,
            nvarguscamerasrc=gstreamer, nvvidconv=gstreamer,
            nvv4l2h264enc=gstreamer, nvv4l2h265enc=gstreamer,
        ),
    )


@pytest.fixture
def config(base_config_dict: dict, tmp_path: Path):
    base_config_dict["recognition"] = {"mode": "face"}
    base_config_dict["face"] = {
        "detector_model": "yunet.onnx",
        "recognition_model": "w600k_r50.onnx",
    }
    return config_from_dict(base_config_dict, base_dir=tmp_path)


class TestBackendSelection:
    def test_jetson_with_tensorrt_prefers_tensorrt(self, config) -> None:
        decision = select_backend(
            ModelRole.FACE_ENCODER, "model.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.TENSORRT
        assert "Jetson" in decision.reason

    def test_cuda_without_tensorrt_falls_back_to_onnx(self, config) -> None:
        decision = select_backend(
            ModelRole.FACE_ENCODER, "model.onnx", config, caps(cuda=True, tensorrt=False)
        )
        assert decision.backend is InferenceBackend.ONNX
        assert "CUDA" in decision.reason

    def test_cpu_only_uses_onnx(self, config) -> None:
        decision = select_backend(ModelRole.FACE_ENCODER, "model.onnx", config, caps())
        assert decision.backend is InferenceBackend.ONNX

    def test_a_pt_model_uses_pytorch(self, config) -> None:
        decision = select_backend(ModelRole.DETECTOR, "yolo26n.pt", config, caps())
        assert decision.backend is InferenceBackend.PYTORCH

    def test_yunet_uses_opencv(self, config) -> None:
        decision = select_backend(ModelRole.FACE_DETECTOR, "yunet.onnx", config, caps())
        assert decision.backend is InferenceBackend.OPENCV

    def test_a_pt_model_cannot_use_tensorrt(self, config) -> None:
        """TensorRT builds from ONNX; a .pt must be exported first."""
        decision = select_backend(
            ModelRole.DETECTOR, "yolo26n.pt", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.PYTORCH

    def test_explicit_tensorrt_falls_back_with_a_reason(self, config) -> None:
        config.backend.face_encoder = InferenceBackend.TENSORRT
        decision = select_backend(
            ModelRole.FACE_ENCODER, "model.onnx", config, caps(tensorrt=False)
        )
        assert decision.backend is InferenceBackend.ONNX
        assert decision.is_fallback
        assert decision.fallback_from is InferenceBackend.TENSORRT
        assert "not installed" in decision.reason

    def test_an_explicit_backend_is_honoured(self, config) -> None:
        config.backend.face_encoder = InferenceBackend.ONNX
        decision = select_backend(
            ModelRole.FACE_ENCODER, "model.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.ONNX

    def test_tensorrt_can_be_disabled_outright(self, config) -> None:
        config.backend.tensorrt.enabled = False
        platform = caps(jetson=True, cuda=True, tensorrt=True)
        assert tensorrt_enabled(config, platform) is False
        decision = select_backend(ModelRole.FACE_ENCODER, "m.onnx", config, platform)
        assert decision.backend is InferenceBackend.ONNX

    def test_a_prebuilt_engine_is_accepted(self, config) -> None:
        decision = select_backend(
            ModelRole.FACE_ENCODER, "model.engine", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.TENSORRT

    def test_no_builder_blocks_building_from_onnx(self, config) -> None:
        platform = caps(jetson=True, cuda=True, tensorrt=True, bindings=False, trtexec=None)
        decision = select_backend(ModelRole.FACE_ENCODER, "m.onnx", config, platform)
        assert decision.backend is InferenceBackend.ONNX

    def test_allow_build_false_without_an_engine_falls_back(self, config) -> None:
        config.backend.tensorrt.allow_build = False
        decision = select_backend(
            ModelRole.FACE_ENCODER, "m.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.ONNX

    def test_an_explicit_request_reports_why_it_could_not_be_honoured(
        self, config
    ) -> None:
        """An auto decision logs at DEBUG; an explicit one must say so out loud."""
        config.backend.tensorrt.allow_build = False
        config.backend.face_encoder = InferenceBackend.TENSORRT
        decision = select_backend(
            ModelRole.FACE_ENCODER, "m.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.ONNX
        assert decision.is_fallback
        assert "allow_build" in decision.reason


class TestEngineMetadata:
    @pytest.fixture
    def onnx(self, tmp_path: Path) -> Path:
        path = tmp_path / "model.onnx"
        path.write_bytes(b"fake onnx payload" * 100)
        return path

    def _expected(self, store: EngineStore, onnx: Path, config) -> EngineMetadata:
        return build_expected_metadata(
            role="face_encoder",
            source=onnx,
            engine=store.engine_path(onnx, "face_encoder", "fp16", 8),
            precision="fp16",
            input_shape=[3, 112, 112],
            trt_config=config.backend.tensorrt,
            caps=caps(jetson=True, cuda=True, tensorrt=True),
        )

    def test_metadata_records_the_required_provenance(self, tmp_path, onnx, config) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        data = metadata.to_dict()
        for key in (
            "source_model", "source_fingerprint", "engine_path", "precision",
            "tensorrt_version", "cuda_version", "jetpack_version", "input_shape",
            "max_batch_size", "built_at", "build_fingerprint",
        ):
            assert key in data, key

    def test_a_valid_engine_is_reused(self, tmp_path, onnx, config) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)

        verdict = store.validate(
            engine, onnx, metadata, caps(jetson=True, cuda=True, tensorrt=True)
        )
        assert verdict.valid, verdict.reason

    def test_a_changed_source_model_invalidates_the_engine(
        self, tmp_path, onnx, config
    ) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)

        onnx.write_bytes(b"a completely different model" * 100)
        verdict = store.validate(
            engine, onnx, self._expected(store, onnx, config),
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert not verdict.valid
        assert "source model changed" in verdict.reason

    def test_a_changed_build_configuration_invalidates_the_engine(
        self, tmp_path, onnx, config
    ) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)

        config.backend.tensorrt.max_batch_size = 16
        verdict = store.validate(
            engine, onnx, self._expected(store, onnx, config),
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert not verdict.valid
        assert "build configuration changed" in verdict.reason

    def test_a_different_tensorrt_version_invalidates_the_engine(
        self, tmp_path, onnx, config
    ) -> None:
        """Engines are not portable across TensorRT versions."""
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)

        newer = caps(jetson=True, cuda=True, tensorrt=True, trt_version="10.7.0")
        verdict = store.validate(engine, onnx, self._expected(store, onnx, config), newer)
        assert not verdict.valid
        assert "TensorRT" in verdict.reason

    def test_a_different_gpu_invalidates_the_engine(self, tmp_path, onnx, config) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)

        other_gpu = caps(jetson=True, cuda=True, tensorrt=True, compute="9.0")
        verdict = store.validate(engine, onnx, self._expected(store, onnx, config), other_gpu)
        assert not verdict.valid
        assert "compute capability" in verdict.reason

    def test_missing_metadata_invalidates_the_engine(self, tmp_path, onnx, config) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        verdict = store.validate(engine, onnx, metadata, caps(cuda=True, tensorrt=True))
        assert not verdict.valid
        assert "metadata" in verdict.reason

    def test_corrupt_metadata_is_tolerated(self, tmp_path, onnx, config) -> None:
        store = EngineStore(tmp_path / "engines")
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)
        store.metadata_path(engine).write_text("{ not json", encoding="utf-8")
        assert store.load_metadata(engine) is None

    def test_strict_version_check_can_be_disabled(self, tmp_path, onnx, config) -> None:
        store = EngineStore(tmp_path / "engines", strict_version_check=False)
        metadata = self._expected(store, onnx, config)
        engine = Path(metadata.engine_path)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        store.save_metadata(metadata)
        newer = caps(jetson=True, cuda=True, tensorrt=True, trt_version="99.0.0")
        assert store.validate(engine, onnx, self._expected(store, onnx, config), newer).valid

    def test_engine_names_encode_the_build_parameters(self, tmp_path, onnx) -> None:
        """Several precisions must coexist rather than overwrite each other."""
        store = EngineStore(tmp_path / "engines")
        fp16 = store.engine_path(onnx, "face_encoder", "fp16", 8)
        fp32 = store.engine_path(onnx, "face_encoder", "fp32", 8)
        batch16 = store.engine_path(onnx, "face_encoder", "fp16", 16)
        assert len({fp16, fp32, batch16}) == 3

    def test_fingerprint_changes_with_content(self, tmp_path: Path) -> None:
        path = tmp_path / "m.onnx"
        path.write_bytes(b"a" * 5000)
        first = source_fingerprint(path)
        path.write_bytes(b"b" * 5000)
        assert source_fingerprint(path) != first


class TestTensorRTGuards:
    def test_int8_for_a_face_encoder_is_refused_by_default(self, config) -> None:
        """Quantisation moves the embedding distribution and every threshold."""
        request = BuildRequest("face_encoder", Path("m.onnx"), (3, 112, 112))
        assert config.backend.tensorrt.allow_int8_for_face is False
        with pytest.raises(TensorRTUnavailable, match="allow_int8_for_face"):
            _precision_flags("int8", request, config.backend.tensorrt)

    def test_int8_is_allowed_for_the_person_detector(self, config) -> None:
        """Only the recognition encoders carry the threshold-calibration risk."""
        request = BuildRequest("detector", Path("m.onnx"), (3, 640, 640))
        fp16, int8 = _precision_flags("int8", request, config.backend.tensorrt)
        assert fp16 and int8

    def test_int8_is_allowed_once_explicitly_enabled(self, config) -> None:
        request = BuildRequest("face_encoder", Path("m.onnx"), (3, 112, 112))
        config.backend.tensorrt.allow_int8_for_face = True
        fp16, int8 = _precision_flags("int8", request, config.backend.tensorrt)
        assert fp16 and int8

    def test_fp16_needs_no_special_permission(self, config) -> None:
        request = BuildRequest("face_encoder", Path("m.onnx"), (3, 112, 112))
        fp16, int8 = _precision_flags("fp16", request, config.backend.tensorrt)
        assert fp16 and not int8

    def test_int8_without_calibration_data_is_rejected_at_load(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.core.exceptions import ConfigurationError

        base_config_dict["backend"] = {"tensorrt": {"precision": "int8"}}
        with pytest.raises(ConfigurationError, match="int8_calibration_dir"):
            config_from_dict(base_config_dict, base_dir=tmp_path)

    def test_a_non_onnx_source_is_refused_with_advice(self, tmp_path, config) -> None:
        store = EngineStore(tmp_path / "engines")
        builder = TensorRTBuilder(
            config.backend.tensorrt, store, caps(cuda=True, tensorrt=True)
        )
        model = tmp_path / "model.pt"
        model.write_bytes(b"torch")
        request = BuildRequest("detector", model, (3, 640, 640))
        with pytest.raises(TensorRTUnavailable, match="ONNX"):
            builder.ensure_engine(request)

    def test_a_missing_source_is_actionable(self, tmp_path, config) -> None:
        store = EngineStore(tmp_path / "engines")
        builder = TensorRTBuilder(config.backend.tensorrt, store, caps(cuda=True, tensorrt=True))
        request = BuildRequest("detector", tmp_path / "absent.onnx", (3, 640, 640))
        with pytest.raises(TensorRTUnavailable, match="not found"):
            builder.ensure_engine(request)

    def test_allow_build_false_is_actionable(self, tmp_path, config) -> None:
        config.backend.tensorrt.allow_build = False
        store = EngineStore(tmp_path / "engines")
        builder = TensorRTBuilder(config.backend.tensorrt, store, caps(cuda=True, tensorrt=True))
        model = tmp_path / "m.onnx"
        model.write_bytes(b"onnx")
        with pytest.raises(TensorRTUnavailable, match="build-tensorrt"):
            builder.ensure_engine(BuildRequest("face_encoder", model, (3, 112, 112)))

    def test_batch_bounds_are_validated(self, base_config_dict, tmp_path) -> None:
        from src.core.exceptions import ConfigurationError

        base_config_dict["backend"] = {
            "tensorrt": {"min_batch_size": 4, "optimal_batch_size": 2, "max_batch_size": 8}
        }
        with pytest.raises(ConfigurationError, match="batch sizes"):
            config_from_dict(base_config_dict, base_dir=tmp_path)


class TestBuildRequests:
    def test_requests_are_derived_from_the_configuration(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.config.paths import ProjectPaths

        base_config_dict["recognition"] = {"mode": "face"}
        base_config_dict["models"] = {"detector": "yolo26n.onnx", "reid": "reid.onnx"}
        base_config_dict["face"] = {"recognition_model": "w600k_r50.onnx"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        requests = default_build_requests(config, ProjectPaths.from_config(config))
        roles = {r.role for r in requests}
        assert roles == {"detector", "face_encoder"}

    def test_a_pt_detector_produces_no_request(self, base_config_dict, tmp_path) -> None:
        from src.config.paths import ProjectPaths

        base_config_dict["models"] = {"detector": "yolo26n.pt"}
        base_config_dict["recognition"] = {"mode": "face"}
        base_config_dict["face"] = {"recognition_model": "w600k_r50.onnx"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        requests = default_build_requests(config, ProjectPaths.from_config(config))
        assert {r.role for r in requests} == {"face_encoder"}

    def test_the_detector_uses_a_fixed_batch(self, base_config_dict, tmp_path) -> None:
        """Live frames arrive one at a time; a dynamic profile buys nothing."""
        from src.config.paths import ProjectPaths

        base_config_dict["models"] = {"detector": "yolo26n.onnx"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        requests = default_build_requests(config, ProjectPaths.from_config(config))
        detector = next(r for r in requests if r.role == "detector")
        assert detector.dynamic_batch is False


class TestFaceDetectorNeverUsesTensorRT:
    """YuNet runs on OpenCV DNN; reporting a TensorRT backend would mislead."""

    def test_auto_reports_opencv_even_on_a_tensorrt_jetson(self, config) -> None:
        decision = select_backend(
            ModelRole.FACE_DETECTOR, "yunet.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.OPENCV

    def test_an_explicit_request_falls_back_with_the_reason(self, config) -> None:
        config.backend.face_detector = InferenceBackend.TENSORRT
        decision = select_backend(
            ModelRole.FACE_DETECTOR, "yunet.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.OPENCV
        assert decision.is_fallback
        assert "OpenCV DNN" in decision.reason

    def test_the_encoder_is_unaffected(self, config) -> None:
        decision = select_backend(
            ModelRole.FACE_ENCODER, "w600k_r50.onnx", config,
            caps(jetson=True, cuda=True, tensorrt=True),
        )
        assert decision.backend is InferenceBackend.TENSORRT
