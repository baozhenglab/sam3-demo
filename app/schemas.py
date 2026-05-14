from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PersonDetection(BaseModel):
    id: int
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixels
    pose: str          # "standing" | "sitting" | "lying"
    is_moving: Optional[bool] = None  # None for images, bool for video/stream


class FrameAnalysis(BaseModel):
    frame_idx: int
    timestamp_sec: float
    persons: list[PersonDetection]


class PersonSummary(BaseModel):
    id: int
    dominant_pose: str
    was_moving: Optional[bool]
    frames_detected: int


class AnalysisResult(BaseModel):
    source: str
    type: str  # "image" | "video" | "stream"
    has_humans: bool
    analyzed_at: datetime
    output_file: Optional[str] = None  # URL to download the annotated result

    # Image response
    persons: Optional[list[PersonDetection]] = None

    # Video / stream response
    duration_sec: Optional[float] = None
    source_fps: Optional[float] = None
    total_frames: Optional[int] = None
    analyzed_frames: Optional[int] = None
    frames: Optional[list[FrameAnalysis]] = None
    person_summary: Optional[list[PersonSummary]] = None
