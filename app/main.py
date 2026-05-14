import io
import os
import tempfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app.detector import SAM3HumanDetector
from app.pose import classify_pose
from app.tracker import PersonTracker
from app.video_processor import process_video
from app.visualizer import draw_persons
from app.schemas import AnalysisResult, PersonDetection

_OUTPUTS_DIR = Path("outputs")
_detector: SAM3HumanDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    _OUTPUTS_DIR.mkdir(exist_ok=True)
    _detector = SAM3HumanDetector()
    yield
    _detector = None


app = FastAPI(
    title="SAM3 Human Analysis API",
    description=(
        "Detect humans in images/videos/streams, classify pose "
        "(standing/sitting/lying), and detect movement using SAM3."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _output_url(filename: str) -> str:
    return f"/files/{filename}"


# ---------------------------------------------------------------------------
# Download endpoint for result files
# ---------------------------------------------------------------------------

@app.get("/files/{filename}", summary="Download an annotated result file")
async def download_file(filename: str):
    path = _OUTPUTS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), filename=filename)


# ---------------------------------------------------------------------------
# Image endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze/image", response_model=AnalysisResult, summary="Analyze a single image")
async def analyze_image(file: UploadFile = File(...)):
    """Upload a JPG or PNG image. Returns detected persons with pose labels
    and a URL to download the annotated image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

    raw = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    detections = _detector.detect(pil_image)

    persons: list[PersonDetection] = []
    for idx, det in enumerate(detections, start=1):
        pose = classify_pose(det)
        persons.append(
            PersonDetection(
                id=idx,
                confidence=round(det.score, 4),
                bbox=[round(v, 1) for v in det.bbox],
                pose=pose,
                is_moving=None,
            )
        )

    # Save annotated image
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    annotated = draw_persons(bgr, persons)
    stem = Path(file.filename or "image").stem
    out_filename = f"result_{_timestamp_str()}_{stem}.jpg"
    cv2.imwrite(str(_OUTPUTS_DIR / out_filename), annotated)

    return AnalysisResult(
        source=file.filename or "upload",
        type="image",
        has_humans=len(persons) > 0,
        analyzed_at=_now(),
        output_file=_output_url(out_filename),
        persons=persons,
    )


# ---------------------------------------------------------------------------
# Video endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze/video", response_model=AnalysisResult, summary="Analyze a video file")
async def analyze_video(file: UploadFile = File(...)):
    """Upload an MP4/AVI video. Returns per-frame detections, a person summary,
    and a URL to download the annotated video."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video (mp4/avi)")

    raw = await file.read()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"

    stem = Path(file.filename or "video").stem
    out_filename = f"result_{_timestamp_str()}_{stem}.mp4"
    out_path = str(_OUTPUTS_DIR / out_filename)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        result = process_video(tmp_path, _detector, output_path=out_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return AnalysisResult(
        source=file.filename or "upload",
        type="video",
        has_humans=result["has_humans"],
        analyzed_at=_now(),
        output_file=_output_url(out_filename),
        source_fps=result["source_fps"],
        total_frames=result["total_frames"],
        duration_sec=result["duration_sec"],
        analyzed_frames=result["analyzed_frames"],
        frames=result["frames"],
        person_summary=result["person_summary"],
    )


# ---------------------------------------------------------------------------
# Camera / RTSP stream endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/analyze/stream",
    response_model=AnalysisResult,
    summary="Analyze a live camera or RTSP stream",
)
async def analyze_stream(
    url: str = Form(..., description="RTSP URL or device index (e.g. '0' for webcam)"),
    duration_sec: int = Form(
        default=int(os.getenv("STREAM_CAPTURE_SECONDS", "10")),
        description="How many seconds to capture",
    ),
):
    """Capture `duration_sec` seconds from an RTSP URL or webcam, then analyze.
    Returns JSON results and a URL to download the annotated video."""
    source: str | int = int(url) if url.isdigit() else url

    out_filename = f"result_{_timestamp_str()}_stream.mp4"
    out_path = str(_OUTPUTS_DIR / out_filename)

    try:
        result = process_video(source, _detector, capture_seconds=duration_sec, output_path=out_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AnalysisResult(
        source=str(url),
        type="stream",
        has_humans=result["has_humans"],
        analyzed_at=_now(),
        output_file=_output_url(out_filename),
        source_fps=result["source_fps"],
        total_frames=result["total_frames"],
        duration_sec=result["duration_sec"],
        analyzed_frames=result["analyzed_frames"],
        frames=result["frames"],
        person_summary=result["person_summary"],
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _detector is not None}
