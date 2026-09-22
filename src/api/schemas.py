"""Pydantic request/response models for the optional REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    application: str
    version: str
    device: dict[str, Any]
    detector: dict[str, Any]
    reid: dict[str, Any]
    gallery: dict[str, Any]


class PersonIn(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    title: str = ""
    image_path: str
    enabled: bool = True


class PersonUpdate(BaseModel):
    name: str | None = None
    title: str | None = None
    image_path: str | None = None
    enabled: bool | None = None


class PersonOut(BaseModel):
    id: str
    name: str
    title: str
    enabled: bool
    has_embedding: bool
    embedding_dimension: int | None = None
    image_path: str = ""
    created_at: str = ""
    updated_at: str = ""
    quality_warnings: list[str] = Field(default_factory=list)


class DetectionOut(BaseModel):
    detection_id: str
    track_id: int | None = None
    bbox: list[float]
    detector_confidence: float
    identity_id: str | None = None
    identity_name: str | None = None
    identity_title: str | None = None
    reid_similarity: float
    recognition_status: str


class ProcessImageResponse(BaseModel):
    source: str
    width: int
    height: int
    num_detections: int
    num_recognized: int
    detections: list[DetectionOut]
    annotated_path: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class BuildResponse(BaseModel):
    enrolled: list[str]
    reused: list[str]
    skipped: list[str]
    failed: dict[str, str]
    total_active: int


class EventOut(BaseModel):
    event_id: str
    type: str
    timestamp: float
    source_id: str
    frame_index: int
    track_id: int | None = None
    identity_id: str | None = None
    identity_name: str | None = None
    similarity: float | None = None
