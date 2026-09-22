"""Processing pipeline and composition root."""

from src.pipeline.engine import Engine, build_engine, require_identities
from src.pipeline.metrics import MetricsCollector
from src.pipeline.processor import ProcessOutcome, ReIDPipeline
from src.pipeline.runner import RunSummary, StreamRunner

__all__ = [
    "Engine",
    "build_engine",
    "require_identities",
    "ReIDPipeline",
    "ProcessOutcome",
    "StreamRunner",
    "RunSummary",
    "MetricsCollector",
]
