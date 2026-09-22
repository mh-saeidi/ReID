"""Input source abstraction: images, directories, video, webcam, streams."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.config.schema import SourceKind
from src.core.exceptions import ConfigurationError, SourceError
from src.sources.base import BaseSource
from src.sources.directory import ImageDirectorySource
from src.sources.factory import build_source, infer_kind
from src.sources.image import ImageSource
from src.sources.stream import NetworkStreamSource
from src.sources.video import VideoFileSource
from src.sources.webcam import WebcamSource
from src.utils.image import imwrite
from tests.conftest import person_scene, write_video


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    for index, color in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200)]):
        imwrite(directory / f"img_{index}.jpg", person_scene(color))
    (directory / "notes.txt").write_text("not an image", encoding="utf-8")
    return directory


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    frames = [person_scene((20 * i, 100, 200), size=(320, 240)) for i in range(12)]
    return write_video(tmp_path / "clip.mp4", frames, fps=10)


class TestImageSource:
    def test_yields_exactly_one_frame(self, tmp_path: Path) -> None:
        path = imwrite(tmp_path / "one.jpg", person_scene((10, 20, 30), size=(320, 240)))
        source = ImageSource(path)
        info = source.open()

        assert info.kind == "image"
        assert (info.width, info.height) == (320, 240)
        assert info.frame_count == 1
        assert info.is_stream is False

        frame = source.read()
        assert frame is not None
        assert frame.index == 0
        assert frame.path == str(path)
        assert source.read() is None
        source.close()

    def test_missing_file_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(SourceError, match="image not found"):
            ImageSource(tmp_path / "gone.jpg").open()

    def test_corrupt_file_is_actionable(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.jpg"
        path.write_bytes(b"\xff\xd8\xff garbage")
        with pytest.raises(SourceError, match="cannot decode"):
            ImageSource(path).open()

    def test_iteration_protocol(self, tmp_path: Path) -> None:
        path = imwrite(tmp_path / "one.jpg", person_scene((10, 20, 30)))
        assert len(list(ImageSource(path))) == 1


class TestImageDirectorySource:
    def test_reads_every_supported_image_in_sorted_order(self, image_dir: Path) -> None:
        source = ImageDirectorySource(image_dir)
        info = source.open()
        assert info.frame_count == 3            # the .txt file is ignored

        names = [Path(frame.path).name for frame in iter(lambda: source.read(), None)]
        assert names == ["img_0.jpg", "img_1.jpg", "img_2.jpg"]
        source.close()

    def test_extension_filter_is_configurable(self, image_dir: Path) -> None:
        imwrite(image_dir / "extra.png", person_scene((1, 2, 3)))
        assert ImageDirectorySource(image_dir, extensions=[".png"]).open().frame_count == 1

    def test_recursive_descent(self, image_dir: Path) -> None:
        nested = image_dir / "nested"
        nested.mkdir()
        imwrite(nested / "deep.jpg", person_scene((9, 9, 9)))
        assert ImageDirectorySource(image_dir, recursive=False).open().frame_count == 3
        assert ImageDirectorySource(image_dir, recursive=True).open().frame_count == 4

    def test_empty_directory_is_actionable(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SourceError, match="no images found"):
            ImageDirectorySource(empty).open()

    def test_missing_directory_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(SourceError, match="directory not found"):
            ImageDirectorySource(tmp_path / "nope").open()

    def test_a_corrupt_file_is_skipped_not_fatal(self, image_dir: Path) -> None:
        (image_dir / "img_1.jpg").write_bytes(b"garbage")
        source = ImageDirectorySource(image_dir)
        source.open()
        frames = list(iter(lambda: source.read(), None))
        assert len(frames) == 2          # the batch survives one bad file


class TestVideoFileSource:
    def test_reads_all_frames_with_metadata(self, video_file: Path) -> None:
        source = VideoFileSource(video_file)
        info = source.open()
        assert info.kind == "video"
        assert (info.width, info.height) == (320, 240)
        assert info.fps == pytest.approx(10.0, abs=0.1)
        assert info.is_stream is False

        frames = list(iter(lambda: source.read(), None))
        assert len(frames) == 12
        assert [f.index for f in frames] == list(range(12))
        source.close()

    def test_stride_subsamples(self, video_file: Path) -> None:
        source = VideoFileSource(video_file, stride=3)
        source.open()
        frames = list(iter(lambda: source.read(), None))
        assert len(frames) == 4
        assert [f.index for f in frames] == [0, 1, 2, 3]   # indices stay contiguous

    def test_missing_file_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(SourceError, match="video file not found"):
            VideoFileSource(tmp_path / "gone.mp4").open()

    def test_corrupt_file_is_actionable(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.mp4"
        path.write_bytes(b"definitely not a video")
        with pytest.raises(SourceError, match="cannot open video"):
            VideoFileSource(path).open()


class TestWebcamSource:
    """The interface is tested with a mocked capture: CI has no camera."""

    def test_opens_and_reports_actual_geometry(self) -> None:
        frame = person_scene((10, 10, 10), size=(640, 480))
        capture = mock.MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
        capture.get.return_value = 30.0

        with mock.patch("cv2.VideoCapture", return_value=capture):
            source = WebcamSource(0, width=640, height=480)
            info = source.open()

        assert info.kind == "webcam"
        assert info.source_id == "webcam_0"
        assert info.is_stream is True
        assert info.frame_count is None
        assert (info.width, info.height) == (640, 480)
        assert source.read().index == 0
        assert source.read().index == 1
        source.close()
        capture.release.assert_called()

    def test_unopenable_device_is_actionable(self) -> None:
        capture = mock.MagicMock()
        capture.isOpened.return_value = False
        with mock.patch("cv2.VideoCapture", return_value=capture), pytest.raises(SourceError, match="cannot open camera"):
                WebcamSource(3).open()

    def test_device_that_delivers_no_frames_is_actionable(self) -> None:
        capture = mock.MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with mock.patch("cv2.VideoCapture", return_value=capture), pytest.raises(SourceError, match="no frames"):
                WebcamSource(0).open()

    def test_auto_probes_for_a_device(self) -> None:
        frame = person_scene((0, 0, 0), size=(320, 240))

        def factory(index, *args, **kwargs):
            capture = mock.MagicMock()
            capture.isOpened.return_value = index == 1
            capture.read.return_value = (index == 1, frame if index == 1 else None)
            capture.get.return_value = 30.0
            return capture

        with mock.patch("cv2.VideoCapture", side_effect=factory):
            source = WebcamSource("auto")
            assert source.open().source_id == "webcam_1"

    def test_auto_with_no_camera_is_actionable(self) -> None:
        capture = mock.MagicMock()
        capture.isOpened.return_value = False
        with mock.patch("cv2.VideoCapture", return_value=capture), pytest.raises(SourceError, match="no usable camera"):
                WebcamSource("auto").open()

    def test_read_after_disconnect_ends_the_stream(self) -> None:
        frame = person_scene((0, 0, 0), size=(320, 240))
        capture = mock.MagicMock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, frame), (True, frame), (False, None)]
        capture.get.return_value = 30.0
        with mock.patch("cv2.VideoCapture", return_value=capture):
            source = WebcamSource(0)
            source.open()
            assert source.read() is not None
            assert source.read() is None


class TestStreamSource:
    def test_opens_a_url(self) -> None:
        frame = person_scene((5, 5, 5), size=(640, 360))
        capture = mock.MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
        capture.get.return_value = 25.0
        with mock.patch("cv2.VideoCapture", return_value=capture):
            info = NetworkStreamSource("rtsp://example/stream").open()
        assert info.kind == "stream"
        assert info.is_stream is True

    def test_unreachable_stream_is_actionable(self) -> None:
        capture = mock.MagicMock()
        capture.isOpened.return_value = False
        with mock.patch("cv2.VideoCapture", return_value=capture), pytest.raises(SourceError, match="cannot open stream"):
                NetworkStreamSource("rtsp://nope").open()


class TestFactoryAndInterface:
    def test_every_source_implements_the_common_interface(self, tmp_path: Path) -> None:
        path = imwrite(tmp_path / "x.jpg", person_scene((1, 1, 1), size=(64, 128)))
        source: BaseSource = ImageSource(path)
        source.open()
        assert source.width() == 64
        assert source.height() == 128
        assert source.fps() == 0.0
        assert source.source_name().startswith("image:")
        assert source.frame_count() == 1

    def test_factory_builds_each_kind(self, base_config_dict, tmp_path, image_dir, video_file):
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        paths = ProjectPaths.from_config(config)
        assert isinstance(
            build_source(config, paths, kind=SourceKind.DIRECTORY, target=str(image_dir)),
            ImageDirectorySource,
        )
        assert isinstance(
            build_source(config, paths, kind=SourceKind.VIDEO, target=str(video_file)),
            VideoFileSource,
        )
        assert isinstance(
            build_source(config, paths, kind=SourceKind.WEBCAM, target="0"), WebcamSource
        )
        assert isinstance(
            build_source(config, paths, kind=SourceKind.STREAM, target="rtsp://x"),
            NetworkStreamSource,
        )

    def test_factory_requires_a_target_for_file_sources(self, base_config_dict, tmp_path):
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        paths = ProjectPaths.from_config(config)
        with pytest.raises(ConfigurationError, match="requires source.path"):
            build_source(config, paths, kind=SourceKind.VIDEO)

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("0", SourceKind.WEBCAM),
            ("rtsp://cam/stream", SourceKind.STREAM),
            ("http://cam/stream", SourceKind.STREAM),
            ("clip.mp4", SourceKind.VIDEO),
            ("photo.jpg", SourceKind.IMAGE),
        ],
    )
    def test_kind_inference(self, base_config_dict, tmp_path, target, expected) -> None:
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        assert infer_kind(target, config) is expected

    def test_directory_is_inferred(self, base_config_dict, tmp_path, image_dir) -> None:
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        assert infer_kind(str(image_dir), config) is SourceKind.DIRECTORY
