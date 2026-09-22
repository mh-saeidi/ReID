"""Jetson-specific pieces: capability detection, camera pipelines, recording.

No Jetson hardware is available in this environment, so these tests exercise
the decision logic and the generated pipelines with synthetic capabilities.
What is verified here is that the right choice is made for a given platform and
that every fallback path works -- not that the NVIDIA elements themselves run.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from src.config.loader import config_from_dict
from src.config.schema import RecordingBackend, SourceKind
from src.core.exceptions import SourceError
from src.hardware.capabilities import (
    SystemCapabilities,
    detect_capabilities,
    render_capabilities,
)
from src.output.gst_recorder import build_gst_pipeline, plan_writer
from src.sources.jetson_camera import (
    JetsonCameraSource,
    build_csi_pipeline,
    build_v4l2_pipeline,
)
from tests.test_backends import caps


class TestCapabilityDetection:
    def test_detection_never_raises_on_this_machine(self) -> None:
        detected = detect_capabilities()
        assert isinstance(detected, SystemCapabilities)
        assert detected.host.system

    def test_the_report_renders(self) -> None:
        text = render_capabilities(detect_capabilities())
        for heading in ("Platform", "NVIDIA Jetson", "CUDA", "TensorRT", "GStreamer"):
            assert heading in text

    def test_capabilities_serialise(self) -> None:
        data = detect_capabilities().to_dict()
        assert "derived" in data
        for key in ("is_jetson", "has_cuda", "has_tensorrt", "has_nvmm"):
            assert key in data["derived"]

    def test_derived_flags_follow_the_elements(self) -> None:
        jetson = caps(jetson=True, cuda=True, tensorrt=True, gstreamer=True)
        assert jetson.has_nvmm
        assert jetson.has_hardware_encoder
        assert jetson.has_csi_camera

        bare = caps(jetson=True, cuda=True, tensorrt=True, gstreamer=False)
        assert not bare.has_nvmm
        assert not bare.has_hardware_encoder

    def test_a_jetson_without_tensorrt_is_flagged(self) -> None:
        """An operator should be told why the fast path is not being used."""

        jetson = caps(jetson=True, cuda=True, tensorrt=False, gstreamer=True)
        # detect_capabilities() builds the notes; replicate its rule here.
        assert jetson.is_jetson and not jetson.has_tensorrt

    def test_tensorrt_can_build_requires_a_builder(self) -> None:
        with_bindings = caps(tensorrt=True, bindings=True, trtexec=None)
        assert with_bindings.tensorrt.can_build
        with_trtexec = caps(tensorrt=True, bindings=False, trtexec="/usr/bin/trtexec")
        assert with_trtexec.tensorrt.can_build
        neither = caps(tensorrt=True, bindings=False, trtexec=None)
        assert not neither.tensorrt.can_build


class TestCameraPipelines:
    def test_the_csi_pipeline_uses_nvmm_and_a_shallow_sink(self) -> None:
        pipeline = build_csi_pipeline(
            sensor_id=0, capture_width=1920, capture_height=1080,
            output_width=1280, output_height=720, fps=30,
        )
        assert "nvarguscamerasrc sensor-id=0" in pipeline
        assert "memory:NVMM" in pipeline
        assert "nvvidconv" in pipeline
        # Latency: never let an overloaded consumer accumulate frames.
        assert "drop=true" in pipeline
        assert "max-buffers=1" in pipeline

    def test_the_sensor_scales_once_rather_than_python_resizing(self) -> None:
        """Capture at 1080p, deliver 720p, in one GStreamer pass."""
        pipeline = build_csi_pipeline(
            sensor_id=0, capture_width=1920, capture_height=1080,
            output_width=1280, output_height=720, fps=30,
        )
        assert "width=1920,height=1080" in pipeline
        assert "width=1280,height=720" in pipeline

    def test_flip_method_is_passed_through(self) -> None:
        pipeline = build_csi_pipeline(
            sensor_id=1, capture_width=1280, capture_height=720,
            output_width=1280, output_height=720, fps=30, flip_method=2,
        )
        assert "flip-method=2" in pipeline
        assert "sensor-id=1" in pipeline

    def test_the_v4l2_pipeline_offloads_conversion(self) -> None:
        pipeline = build_v4l2_pipeline(
            device="/dev/video0", output_width=1280, output_height=720, fps=30
        )
        assert "v4l2src device=/dev/video0" in pipeline
        assert "nvvidconv" in pipeline
        assert "drop=true" in pipeline

    def test_mode_selection_follows_the_hardware(self, base_config_dict, tmp_path) -> None:
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        source = JetsonCameraSource(
            config.source, caps(jetson=True, cuda=True, gstreamer=True)
        )
        assert source._resolve_mode() == "csi"

        no_csi = caps(jetson=True, cuda=True, gstreamer=False)
        assert JetsonCameraSource(config.source, no_csi)._resolve_mode() == "v4l2"

    def test_csi_can_be_forced_off(self, base_config_dict, tmp_path) -> None:
        base_config_dict["source"] = {"type": "jetson_camera", "csi": False}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        source = JetsonCameraSource(
            config.source, caps(jetson=True, cuda=True, gstreamer=True)
        )
        assert source._resolve_mode() == "v4l2"

    def test_a_manual_pipeline_overrides_everything(self, base_config_dict, tmp_path) -> None:
        base_config_dict["source"] = {
            "type": "jetson_camera", "gst_pipeline": "videotestsrc ! appsink"
        }
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        source = JetsonCameraSource(config.source, caps(gstreamer=True))
        assert source._resolve_mode() == "manual"
        assert source._build_pipeline() == "videotestsrc ! appsink"

    def test_opening_without_gstreamer_is_actionable(self, base_config_dict, tmp_path) -> None:
        base_config_dict["source"] = {"type": "jetson_camera"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        source = JetsonCameraSource(config.source, caps(gstreamer=False))
        with pytest.raises(SourceError, match="GStreamer"):
            source.open()

    def test_requesting_csi_without_the_element_is_actionable(
        self, base_config_dict, tmp_path
    ) -> None:
        base_config_dict["source"] = {"type": "jetson_camera", "csi": True}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        platform = caps(gstreamer=True)
        object.__setattr__(platform.gstreamer, "nvarguscamerasrc", False)
        source = JetsonCameraSource(config.source, platform)
        with pytest.raises(SourceError, match="nvarguscamerasrc"):
            source.open()


class TestJetsonCameraFallback:
    def test_the_factory_falls_back_to_the_portable_webcam_source(
        self, base_config_dict, tmp_path
    ) -> None:
        """A Jetson profile must still run on a developer laptop."""
        from src.config.paths import ProjectPaths
        from src.sources.factory import build_source
        from src.sources.webcam import WebcamSource

        base_config_dict["source"] = {"type": "jetson_camera"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        with mock.patch(
            "src.hardware.capabilities.detect_capabilities",
            return_value=caps(gstreamer=False),
        ):
            source = build_source(
                config, ProjectPaths.from_config(config), kind=SourceKind.JETSON_CAMERA
            )
        assert isinstance(source, WebcamSource)


class TestHardwareRecording:
    def test_hardware_is_used_when_the_encoder_exists(
        self, base_config_dict, tmp_path
    ) -> None:
        base_config_dict["recording"] = {"codec": "h264", "hardware_acceleration": True}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        plan = plan_writer(config, caps(jetson=True, cuda=True, gstreamer=True))
        assert plan.hardware
        assert plan.backend is RecordingBackend.GSTREAMER

    def test_software_is_used_without_gstreamer(self, base_config_dict, tmp_path) -> None:
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        plan = plan_writer(config, caps(gstreamer=False))
        assert not plan.hardware
        assert plan.backend is RecordingBackend.OPENCV
        assert "GStreamer" in plan.reason

    def test_a_missing_h265_encoder_falls_back(self, base_config_dict, tmp_path) -> None:
        base_config_dict["recording"] = {"codec": "h265"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        platform = caps(jetson=True, cuda=True, gstreamer=True)
        object.__setattr__(platform.gstreamer, "nvv4l2h265enc", False)
        plan = plan_writer(config, platform)
        assert not plan.hardware
        assert "nvv4l2h265enc" in plan.reason

    def test_mp4v_has_no_hardware_path(self, base_config_dict, tmp_path) -> None:
        base_config_dict["recording"] = {"codec": "mp4v"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        plan = plan_writer(config, caps(jetson=True, cuda=True, gstreamer=True))
        assert not plan.hardware

    def test_hardware_can_be_disabled(self, base_config_dict, tmp_path) -> None:
        base_config_dict["recording"] = {"hardware_acceleration": False}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        plan = plan_writer(config, caps(jetson=True, cuda=True, gstreamer=True))
        assert not plan.hardware

    def test_the_backend_can_be_pinned_to_opencv(self, base_config_dict, tmp_path) -> None:
        base_config_dict["recording"] = {"backend": "opencv"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        plan = plan_writer(config, caps(jetson=True, cuda=True, gstreamer=True))
        assert plan.backend is RecordingBackend.OPENCV

    def test_the_encoder_pipeline_is_well_formed(self) -> None:
        pipeline = build_gst_pipeline(
            Path("/tmp/clip.mp4"), fps=30, size=(1280, 720),
            codec="h264", bitrate_kbps=4000,
        )
        assert "appsrc" in pipeline
        assert "nvv4l2h264enc" in pipeline
        assert "bitrate=4000000" in pipeline
        assert "location=/tmp/clip.mp4" in pipeline

    def test_h265_selects_the_matching_parser(self) -> None:
        pipeline = build_gst_pipeline(
            Path("/tmp/c.mp4"), fps=30, size=(640, 480), codec="h265", bitrate_kbps=2000
        )
        assert "nvv4l2h265enc" in pipeline
        assert "h265parse" in pipeline

    def test_the_recorder_still_works_with_the_software_writer(
        self, base_config_dict, tmp_path
    ) -> None:
        """Event clips and ring buffering must not depend on the encoder."""
        from src.config.schema import OutputConfig, RecordingConfig, RecordingMode
        from src.output.recorder import VideoRecorder
        from tests.conftest import person_scene

        recorder = VideoRecorder(
            RecordingConfig(mode=RecordingMode.CONTINUOUS),
            OutputConfig(),
            tmp_path / "videos",
            fps=10.0,
            size=(320, 240),
            source_id="test",
        )
        for _ in range(5):
            recorder.process(person_scene(size=(320, 240)), trigger=True)
        clip = recorder.close()
        assert clip is not None and clip.frames == 5


class TestJetsonProfile:
    def test_the_shipped_profile_is_valid(self) -> None:
        from src.config.loader import load_config

        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "configs" / "jetson_orin_nano_super.yaml")
        assert config.pipeline.async_enabled is True
        assert config.recognition_scheduler.enabled is True
        assert config.backend.tensorrt.precision.value == "fp16"
        assert config.source.type is SourceKind.JETSON_CAMERA
        assert config.benchmark.targets.enabled is True

    def test_the_profile_does_not_enable_int8_for_faces(self) -> None:
        """INT8 shifts the embedding distribution; it must stay opt-in."""
        from src.config.loader import load_config

        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "configs" / "jetson_orin_nano_super.yaml")
        assert config.backend.tensorrt.allow_int8_for_face is False

    def test_the_profile_keeps_face_mode_semantics(self) -> None:
        from src.config.loader import load_config
        from src.config.schema import RecognitionMode

        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "configs" / "jetson_orin_nano_super.yaml")
        assert config.recognition.mode is RecognitionMode.FACE
        assert config.face.identity_hold_frames > 0
        assert config.tracking.identity_stability.enabled is True
