"""Runtime system telemetry for benchmarking.

Everything is read-only and best-effort: a missing sysfs node or an absent
``tegrastats`` yields ``None`` rather than an error, because a benchmark that
refuses to run because it cannot read a thermal zone is useless.

On a Jetson the interesting numbers (GPU load, power rails, SoC temperatures)
live in sysfs and are readable without root. Nothing here changes a power
profile, clock setting or fan curve.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Jetson exposes GPU busy-percent here across JetPack 5 and 6.
_JETSON_GPU_LOAD = (
    Path("/sys/devices/platform/gpu.0/load"),
    Path("/sys/devices/gpu.0/load"),
)
_THERMAL_ROOT = Path("/sys/devices/virtual/thermal")
_POWER_RAIL_GLOBS = (
    "/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/",
    "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/",
)


@dataclass(slots=True)
class TelemetrySample:
    """One instantaneous reading."""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float | None = None
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    gpu_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    temperatures_c: dict[str, float] = field(default_factory=dict)
    power_mw: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, [])}


class TelemetryCollector:
    """Samples system counters during a benchmark run."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._samples: list[TelemetrySample] = []
        self._prev_cpu: tuple[int, int] | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._is_jetson = any(p.exists() for p in _JETSON_GPU_LOAD)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def sample(self) -> TelemetrySample | None:
        if not self._enabled:
            return None
        reading = TelemetrySample(
            cpu_percent=self._cpu_percent(),
            gpu_percent=self._gpu_percent(),
            gpu_memory_used_mb=self._gpu_memory(),
            temperatures_c=self._temperatures(),
            power_mw=self._power(),
        )
        used, total = self._memory()
        reading.ram_used_mb, reading.ram_total_mb = used, total
        self._samples.append(reading)
        return reading

    # ------------------------------------------------------------------ CPU
    def _cpu_percent(self) -> float | None:
        stat = Path("/proc/stat")
        if stat.exists():
            try:
                fields = stat.read_text().splitlines()[0].split()[1:]
                values = [int(v) for v in fields]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                total = sum(values)
                if self._prev_cpu is not None:
                    prev_idle, prev_total = self._prev_cpu
                    delta_total = total - prev_total
                    delta_idle = idle - prev_idle
                    self._prev_cpu = (idle, total)
                    if delta_total > 0:
                        return round(100.0 * (delta_total - delta_idle) / delta_total, 1)
                self._prev_cpu = (idle, total)
                return None
            except (OSError, ValueError, IndexError):
                return None
        # macOS and anything else: load average scaled by core count.
        try:
            load = os.getloadavg()[0]
            cores = os.cpu_count() or 1
            return round(min(100.0, 100.0 * load / cores), 1)
        except (OSError, AttributeError):
            return None

    def _memory(self) -> tuple[float | None, float | None]:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                text = meminfo.read_text()
                total = _kb(text, "MemTotal")
                available = _kb(text, "MemAvailable")
                if total is not None and available is not None:
                    return round((total - available) / 1024, 1), round(total / 1024, 1)
            except OSError:
                pass
        try:
            import resource  # noqa: PLC0415

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS reports bytes.
            scale = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
            return round(usage / scale, 1), None
        except Exception:  # noqa: BLE001
            return None, None

    # ------------------------------------------------------------------ GPU
    def _gpu_percent(self) -> float | None:
        for path in _JETSON_GPU_LOAD:
            if path.exists():
                try:
                    # Reported in tenths of a percent.
                    return round(int(path.read_text().strip()) / 10.0, 1)
                except (OSError, ValueError):
                    continue
        if self._nvidia_smi:
            output = _run(
                [self._nvidia_smi, "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"]
            )
            if output:
                try:
                    return float(output.splitlines()[0].strip())
                except ValueError:
                    return None
        return None

    def _gpu_memory(self) -> float | None:
        if self._is_jetson:
            # Jetson GPU memory is the shared system RAM, already reported above.
            return None
        if self._nvidia_smi:
            output = _run(
                [self._nvidia_smi, "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"]
            )
            if output:
                try:
                    return float(output.splitlines()[0].strip())
                except ValueError:
                    return None
        return None

    # ------------------------------------------------------- thermal / power
    def _temperatures(self) -> dict[str, float]:
        readings: dict[str, float] = {}
        if not _THERMAL_ROOT.exists():
            return readings
        try:
            zones = sorted(_THERMAL_ROOT.glob("thermal_zone*"))
        except OSError:
            return readings
        for zone in zones:
            try:
                name = (zone / "type").read_text().strip()
                milli = int((zone / "temp").read_text().strip())
            except (OSError, ValueError):
                continue
            readings[name] = round(milli / 1000.0, 1)
        return readings

    def _power(self) -> dict[str, float]:
        readings: dict[str, float] = {}
        for pattern in _POWER_RAIL_GLOBS:
            # These live under different roots across JetPack releases, so the
            # pattern includes its own base directory.
            root = Path(pattern).parts[0]
            relative = str(Path(*Path(pattern).parts[1:]))
            for base in Path(root).glob(relative):
                for power_file in sorted(base.glob("in_power*_input")) + sorted(
                    base.glob("power*_input")
                ):
                    label_file = Path(str(power_file).replace("_input", "_label"))
                    try:
                        value = float(power_file.read_text().strip())
                        label = (
                            label_file.read_text().strip()
                            if label_file.exists()
                            else power_file.stem
                        )
                    except (OSError, ValueError):
                        continue
                    readings[label] = round(value, 1)
        return readings

    # -------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        """Aggregate the samples taken during the run."""
        if not self._samples:
            return {"available": False}

        def average(attribute: str) -> float | None:
            values = [
                getattr(s, attribute) for s in self._samples
                if getattr(s, attribute) is not None
            ]
            return round(sum(values) / len(values), 2) if values else None

        def peak(attribute: str) -> float | None:
            values = [
                getattr(s, attribute) for s in self._samples
                if getattr(s, attribute) is not None
            ]
            return round(max(values), 2) if values else None

        temperatures: dict[str, float] = {}
        for sample in self._samples:
            for name, value in sample.temperatures_c.items():
                temperatures[name] = max(temperatures.get(name, value), value)
        power: dict[str, float] = {}
        for sample in self._samples:
            for name, value in sample.power_mw.items():
                power[name] = max(power.get(name, value), value)

        return {
            "available": True,
            "samples": len(self._samples),
            "cpu_percent_avg": average("cpu_percent"),
            "cpu_percent_peak": peak("cpu_percent"),
            "gpu_percent_avg": average("gpu_percent"),
            "gpu_percent_peak": peak("gpu_percent"),
            "ram_used_mb_avg": average("ram_used_mb"),
            "ram_used_mb_peak": peak("ram_used_mb"),
            "ram_total_mb": self._samples[-1].ram_total_mb,
            "gpu_memory_used_mb_peak": peak("gpu_memory_used_mb"),
            "temperature_peak_c": temperatures or None,
            "power_peak_mw": power or None,
        }


def _kb(text: str, key: str) -> float | None:
    match = re.search(rf"{key}:\s*(\d+)\s*kB", text)
    return float(match.group(1)) if match else None


def _run(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603 - fixed probe command
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
