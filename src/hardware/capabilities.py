"""System capability detection.

Everything here is read-only and non-privileged. Nothing in this module changes
``nvpmodel``, jetson_clocks, fan curves or any other system-wide setting: the
deployment decides its own power profile, and a vision application has no
business overriding it.

Detection is cached for the process lifetime because it involves filesystem and
subprocess probes that would otherwise be repeated on every engine build.
"""

from __future__ import annotations

import functools
import os
import platform as _platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Device-tree entries that identify a Jetson board. /proc/device-tree/model is
# the most reliable source on JetPack 5/6.
_DEVICE_TREE_MODEL = Path("/proc/device-tree/model")
_JETSON_RELEASE = Path("/etc/nv_tegra_release")
_L4T_APT_SOURCES = Path("/etc/apt/sources.list.d/nvidia-l4t-apt-source.list")

# L4T (Linux for Tegra) release -> JetPack version. Only the entries relevant to
# Orin-class boards are listed; anything else reports the raw L4T string.
_L4T_TO_JETPACK: dict[str, str] = {
    "35.1": "5.0.2",
    "35.2": "5.1",
    "35.3": "5.1.1",
    "35.4": "5.1.2",
    "35.5": "5.1.3",
    "36.2": "6.0 DP",
    "36.3": "6.0",
    "36.4": "6.1",
    "36.5": "6.2",
}


def _run(command: list[str], timeout: float = 4.0) -> str | None:
    """Run a probe command, returning ``None`` on any failure."""
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed, non-shell probe commands
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip("\x00\n ")
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class JetsonInfo:
    """NVIDIA Jetson board identification."""

    is_jetson: bool = False
    model: str | None = None
    l4t_version: str | None = None
    jetpack_version: str | None = None
    soc: str | None = None

    @property
    def is_orin(self) -> bool:
        return bool(self.model and "orin" in self.model.lower())

    @property
    def is_orin_nano(self) -> bool:
        model = (self.model or "").lower()
        return "orin" in model and "nano" in model


@dataclass(frozen=True, slots=True)
class CudaInfo:
    available: bool = False
    torch_cuda: bool = False
    runtime_version: str | None = None
    driver_version: str | None = None
    device_count: int = 0
    device_name: str | None = None
    compute_capability: str | None = None
    total_memory_gb: float | None = None


@dataclass(frozen=True, slots=True)
class TensorRTInfo:
    available: bool = False
    version: str | None = None
    python_bindings: bool = False
    trtexec: str | None = None
    error: str | None = None

    @property
    def can_build(self) -> bool:
        """Can this machine actually build an engine?"""
        return self.available and (self.python_bindings or self.trtexec is not None)


@dataclass(frozen=True, slots=True)
class GStreamerInfo:
    available: bool = False
    opencv_gstreamer: bool = False
    version: str | None = None
    nvarguscamerasrc: bool = False
    nvvidconv: bool = False
    nvv4l2h264enc: bool = False
    nvv4l2h265enc: bool = False

    @property
    def has_nvmm_elements(self) -> bool:
        """The NVMM conversion element is the gate for zero-copy pipelines."""
        return self.available and self.nvvidconv


@dataclass(frozen=True, slots=True)
class HostInfo:
    system: str = ""
    machine: str = ""
    release: str = ""
    python_version: str = ""
    cpu_model: str | None = None
    cpu_count: int = 0
    total_memory_gb: float | None = None


@dataclass(frozen=True, slots=True)
class SystemCapabilities:
    """Everything the backend selector and the operator need to know."""

    host: HostInfo
    jetson: JetsonInfo
    cuda: CudaInfo
    tensorrt: TensorRTInfo
    gstreamer: GStreamerInfo
    torch_available: bool = False
    torch_version: str | None = None
    onnxruntime_providers: list[str] = field(default_factory=list)
    opencv_version: str | None = None
    mps_available: bool = False
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- summaries
    @property
    def is_jetson(self) -> bool:
        return self.jetson.is_jetson

    @property
    def has_cuda(self) -> bool:
        return self.cuda.available

    @property
    def has_tensorrt(self) -> bool:
        return self.tensorrt.available

    @property
    def has_nvmm(self) -> bool:
        """NVMM (NVIDIA zero-copy memory) capture/encode path available."""
        return self.gstreamer.opencv_gstreamer and self.gstreamer.nvvidconv

    @property
    def has_hardware_encoder(self) -> bool:
        return self.gstreamer.opencv_gstreamer and (
            self.gstreamer.nvv4l2h264enc or self.gstreamer.nvv4l2h265enc
        )

    @property
    def has_csi_camera(self) -> bool:
        return self.gstreamer.opencv_gstreamer and self.gstreamer.nvarguscamerasrc

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": asdict(self.host),
            "jetson": asdict(self.jetson),
            "cuda": asdict(self.cuda),
            "tensorrt": asdict(self.tensorrt),
            "gstreamer": asdict(self.gstreamer),
            "torch": {"available": self.torch_available, "version": self.torch_version},
            "onnxruntime_providers": list(self.onnxruntime_providers),
            "opencv_version": self.opencv_version,
            "mps_available": self.mps_available,
            "derived": {
                "is_jetson": self.is_jetson,
                "has_cuda": self.has_cuda,
                "has_tensorrt": self.has_tensorrt,
                "has_nvmm": self.has_nvmm,
                "has_hardware_encoder": self.has_hardware_encoder,
                "has_csi_camera": self.has_csi_camera,
            },
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def detect_jetson() -> JetsonInfo:
    """Identify a Jetson board from the device tree and L4T release file."""
    if _platform.system() != "Linux":
        return JetsonInfo()

    model = _read_text(_DEVICE_TREE_MODEL)
    compatible = _read_text(Path("/proc/device-tree/compatible")) or ""
    is_tegra = bool(
        (model and "jetson" in model.lower())
        or "nvidia,tegra" in compatible
        or _JETSON_RELEASE.exists()
    )
    if not is_tegra:
        return JetsonInfo()

    l4t = None
    release = _read_text(_JETSON_RELEASE)
    if release:
        # "# R36 (release), REVISION: 4.0, GCID: ..., BOARD: generic, ..."
        match = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", release)
        if match:
            l4t = f"{match.group(1)}.{match.group(2).rstrip('.')}"
    if l4t is None:
        apt = _read_text(_L4T_APT_SOURCES)
        if apt:
            match = re.search(r"r(\d+\.\d+)", apt)
            if match:
                l4t = match.group(1)

    jetpack = None
    if l4t:
        major_minor = ".".join(l4t.split(".")[:2])
        jetpack = _L4T_TO_JETPACK.get(major_minor)

    soc = None
    if compatible:
        soc_match = re.search(r"nvidia,(tegra\w+)", compatible)
        if soc_match:
            soc = soc_match.group(1)

    return JetsonInfo(
        is_jetson=True,
        model=model,
        l4t_version=l4t,
        jetpack_version=jetpack,
        soc=soc,
    )


def detect_cuda() -> CudaInfo:
    """CUDA availability via torch, with nvidia-smi as a secondary source."""
    torch_cuda = False
    runtime = device_count = None
    name = capability = memory = None
    try:
        import torch  # noqa: PLC0415

        torch_cuda = bool(torch.cuda.is_available())
        runtime = getattr(torch.version, "cuda", None)
        if torch_cuda:
            device_count = torch.cuda.device_count()
            try:
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                capability = f"{props.major}.{props.minor}"
                memory = round(props.total_memory / (1024**3), 2)
            except Exception:  # noqa: BLE001 - cosmetic detail only
                pass
    except ImportError:
        pass

    driver = None
    smi = _run(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]
    )
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        if parts:
            driver = parts[0]
        if name is None and len(parts) > 1:
            name = parts[1]

    return CudaInfo(
        available=torch_cuda or driver is not None,
        torch_cuda=torch_cuda,
        runtime_version=runtime,
        driver_version=driver,
        device_count=device_count or 0,
        device_name=name,
        compute_capability=capability,
        total_memory_gb=memory,
    )


def detect_tensorrt() -> TensorRTInfo:
    """TensorRT Python bindings and/or the trtexec builder."""
    version = None
    bindings = False
    error = None
    try:
        import tensorrt as trt  # noqa: PLC0415

        version = str(trt.__version__)
        bindings = True
    except ImportError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a broken install must not crash us
        error = f"tensorrt import failed: {exc}"

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        # JetPack installs it outside PATH by default.
        for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"):
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                trtexec = candidate
                break

    if version is None and trtexec is not None:
        output = _run([trtexec, "--help"]) or ""
        match = re.search(r"TensorRT\.trtexec\s*\[TensorRT v(\d+)\]", output)
        if match:
            raw = match.group(1)
            # trtexec reports 100300 for 10.3.0
            if len(raw) >= 5:
                version = f"{int(raw[:-4])}.{int(raw[-4:-2])}.{int(raw[-2:])}"
            else:
                version = raw

    return TensorRTInfo(
        available=bindings or trtexec is not None,
        version=version,
        python_bindings=bindings,
        trtexec=trtexec,
        error=error,
    )


@functools.lru_cache(maxsize=1)
def _gst_elements() -> frozenset[str]:
    """Names of the GStreamer elements this system can instantiate."""
    output = _run(["gst-inspect-1.0"], timeout=12.0)
    if not output:
        return frozenset()
    names: set[str] = set()
    for line in output.splitlines():
        # "nvarguscamerasrc:  nvarguscamerasrc: NvArgusCameraSrc"
        if ":" in line and not line.startswith(" "):
            parts = line.split(":")
            if len(parts) >= 2:
                names.add(parts[1].strip())
    return frozenset(names)


def detect_gstreamer() -> GStreamerInfo:
    """GStreamer availability, including the NVIDIA-specific elements."""
    opencv_gstreamer = False
    try:
        import cv2  # noqa: PLC0415

        build = cv2.getBuildInformation()
        match = re.search(r"GStreamer:\s*(\S+)", build)
        opencv_gstreamer = bool(match and match.group(1).upper() not in ("NO", "OFF"))
    except Exception:  # noqa: BLE001
        pass

    version = None
    raw = _run(["gst-launch-1.0", "--version"])
    if raw:
        match = re.search(r"([\d.]+)", raw.splitlines()[-1])
        if match:
            version = match.group(1)

    elements = _gst_elements() if (version or opencv_gstreamer) else frozenset()
    return GStreamerInfo(
        available=bool(version) or opencv_gstreamer,
        opencv_gstreamer=opencv_gstreamer,
        version=version,
        nvarguscamerasrc="nvarguscamerasrc" in elements,
        nvvidconv="nvvidconv" in elements,
        nvv4l2h264enc="nvv4l2h264enc" in elements,
        nvv4l2h265enc="nvv4l2h265enc" in elements,
    )


def detect_host() -> HostInfo:

    cpu_model = None
    if _platform.system() == "Linux":
        info = _read_text(Path("/proc/cpuinfo")) or ""
        match = re.search(r"model name\s*:\s*(.+)", info)
        if match:
            cpu_model = match.group(1).strip()
        else:  # ARM kernels expose a different key
            match = re.search(r"Model\s*:\s*(.+)", info)
            cpu_model = match.group(1).strip() if match else None
    elif _platform.system() == "Darwin":
        cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])

    memory_gb = None
    try:
        if _platform.system() == "Linux":
            meminfo = _read_text(Path("/proc/meminfo")) or ""
            match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
            if match:
                memory_gb = round(int(match.group(1)) / (1024**2), 2)
        elif _platform.system() == "Darwin":
            raw = _run(["sysctl", "-n", "hw.memsize"])
            if raw:
                memory_gb = round(int(raw) / (1024**3), 2)
    except (ValueError, TypeError):
        pass

    return HostInfo(
        system=_platform.system(),
        machine=_platform.machine(),
        release=_platform.release(),
        python_version=_platform.python_version(),
        cpu_model=cpu_model,
        cpu_count=os.cpu_count() or 0,
        total_memory_gb=memory_gb,
    )


@functools.lru_cache(maxsize=1)
def detect_capabilities() -> SystemCapabilities:
    """Probe the machine once and cache the result for the process lifetime."""
    host = detect_host()
    jetson = detect_jetson()
    cuda = detect_cuda()
    tensorrt = detect_tensorrt()
    gstreamer = detect_gstreamer()

    torch_available = False
    torch_version = None
    mps = False
    try:
        import torch  # noqa: PLC0415

        torch_available = True
        torch_version = torch.__version__
        mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    providers: list[str] = []
    try:
        import onnxruntime as ort  # noqa: PLC0415

        providers = list(ort.get_available_providers())
    except ImportError:
        pass

    opencv_version = None
    try:
        import cv2  # noqa: PLC0415

        opencv_version = cv2.__version__
    except ImportError:
        pass

    notes: list[str] = []
    if jetson.is_jetson and not tensorrt.available:
        notes.append(
            "Jetson detected but TensorRT was not found. Install the JetPack "
            "TensorRT packages, or run with the ONNX Runtime backend."
        )
    if jetson.is_jetson and not gstreamer.opencv_gstreamer:
        notes.append(
            "This OpenCV build has no GStreamer support, so CSI camera capture "
            "and hardware encoding are unavailable. Install an OpenCV built "
            "with -DWITH_GSTREAMER=ON (JetPack's system OpenCV has it)."
        )
    if cuda.available and not tensorrt.available:
        notes.append("CUDA is available but TensorRT is not; CUDA/ONNX will be used.")

    return SystemCapabilities(
        host=host,
        jetson=jetson,
        cuda=cuda,
        tensorrt=tensorrt,
        gstreamer=gstreamer,
        torch_available=torch_available,
        torch_version=torch_version,
        onnxruntime_providers=providers,
        opencv_version=opencv_version,
        mps_available=mps,
        notes=notes,
    )


def render_capabilities(caps: SystemCapabilities) -> str:
    """Human-readable report for ``main.py system info``."""
    host, jetson, cuda, trt, gst = caps.host, caps.jetson, caps.cuda, caps.tensorrt, caps.gstreamer

    def yes_no(value: bool) -> str:
        return "yes" if value else "no"

    lines = [
        "System capabilities",
        "",
        "  Platform",
        f"    system            : {host.system} {host.release} ({host.machine})",
        f"    python            : {host.python_version}",
        f"    cpu               : {host.cpu_model or 'unknown'} ({host.cpu_count} cores)",
        "    memory            : "
        + (f"{host.total_memory_gb:.1f} GiB" if host.total_memory_gb else "unknown"),
        "",
        "  NVIDIA Jetson",
        f"    jetson            : {yes_no(jetson.is_jetson)}",
    ]
    if jetson.is_jetson:
        lines += [
            f"    model             : {jetson.model or 'unknown'}",
            f"    soc               : {jetson.soc or 'unknown'}",
            f"    L4T               : {jetson.l4t_version or 'unknown'}",
            f"    JetPack           : {jetson.jetpack_version or 'unknown'}",
        ]
    lines += [
        "",
        "  CUDA",
        f"    available         : {yes_no(cuda.available)}",
        f"    torch.cuda        : {yes_no(cuda.torch_cuda)}",
        f"    runtime           : {cuda.runtime_version or '-'}",
        f"    driver            : {cuda.driver_version or '-'}",
        f"    device            : {cuda.device_name or '-'}"
        + (f"  ({cuda.total_memory_gb:.1f} GiB)" if cuda.total_memory_gb else ""),
        f"    compute capability: {cuda.compute_capability or '-'}",
        "",
        "  TensorRT",
        f"    available         : {yes_no(trt.available)}",
        f"    version           : {trt.version or '-'}",
        f"    python bindings   : {yes_no(trt.python_bindings)}",
        f"    trtexec           : {trt.trtexec or '-'}",
        f"    can build engines : {yes_no(trt.can_build)}",
        "",
        "  GStreamer",
        f"    available         : {yes_no(gst.available)}  (version {gst.version or '-'})",
        f"    OpenCV support    : {yes_no(gst.opencv_gstreamer)}",
        f"    nvarguscamerasrc  : {yes_no(gst.nvarguscamerasrc)}   (CSI camera)",
        f"    nvvidconv         : {yes_no(gst.nvvidconv)}   (NVMM conversion)",
        f"    nvv4l2h264enc     : {yes_no(gst.nvv4l2h264enc)}",
        f"    nvv4l2h265enc     : {yes_no(gst.nvv4l2h265enc)}",
        "",
        "  Runtimes",
        f"    torch             : {caps.torch_version or 'not installed'}",
        f"    opencv            : {caps.opencv_version or 'not installed'}",
        f"    mps (Apple)       : {yes_no(caps.mps_available)}",
        f"    onnxruntime       : {', '.join(caps.onnxruntime_providers) or 'not installed'}",
        "",
        "  Derived",
        f"    NVMM path         : {yes_no(caps.has_nvmm)}",
        f"    hardware encoder  : {yes_no(caps.has_hardware_encoder)}",
        f"    CSI camera        : {yes_no(caps.has_csi_camera)}",
    ]
    if caps.notes:
        lines += ["", "  Notes"]
        lines += [f"    - {note}" for note in caps.notes]
    return "\n".join(lines)
