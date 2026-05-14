import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter

from app.detector import SAM3HumanDetector, Detection
from app.pose import classify_pose
from app.tracker import PersonTracker
from app.visualizer import draw_persons
from app.schemas import FrameAnalysis, PersonDetection, PersonSummary

_ANALYZE_FPS = float(os.getenv("ANALYZE_FPS", "2"))


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _build_person_summary(all_frame_results: list[FrameAnalysis]) -> list[PersonSummary]:
    pose_log: dict[int, list[str]] = {}
    move_log: dict[int, list[bool]] = {}
    count_log: dict[int, int] = {}

    for frame in all_frame_results:
        for p in frame.persons:
            pose_log.setdefault(p.id, []).append(p.pose)
            count_log[p.id] = count_log.get(p.id, 0) + 1
            if p.is_moving is not None:
                move_log.setdefault(p.id, []).append(p.is_moving)

    summaries: list[PersonSummary] = []
    for pid in sorted(pose_log):
        dominant_pose = Counter(pose_log[pid]).most_common(1)[0][0]
        was_moving: bool | None = None
        if pid in move_log:
            was_moving = any(move_log[pid])
        summaries.append(
            PersonSummary(
                id=pid,
                dominant_pose=dominant_pose,
                was_moving=was_moving,
                frames_detected=count_log[pid],
            )
        )
    return summaries


def process_video(
    source: str | int,
    detector: SAM3HumanDetector,
    capture_seconds: int | None = None,
    output_path: str | None = None,
) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames: int | None = None
    if capture_seconds is not None:
        max_frames = int(src_fps * capture_seconds)

    step = max(1, int(src_fps / _ANALYZE_FPS))

    writer: cv2.VideoWriter | None = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

    tracker = PersonTracker()
    frame_results: list[FrameAnalysis] = []
    # Holds the most recent person list so intermediate frames still get annotations
    last_persons: list[PersonDetection] = []
    frame_idx = 0
    analyzed = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % step == 0:
                pil_img = _bgr_to_pil(bgr)
                detections: list[Detection] = detector.detect(pil_img)
                track_info = tracker.update(detections)

                persons: list[PersonDetection] = []
                for det_idx, (pid, is_moving) in enumerate(track_info):
                    det = detections[det_idx]
                    pose = classify_pose(det)
                    persons.append(
                        PersonDetection(
                            id=pid,
                            confidence=round(det.score, 4),
                            bbox=[round(v, 1) for v in det.bbox],
                            pose=pose,
                            is_moving=is_moving,
                        )
                    )

                last_persons = persons
                frame_results.append(
                    FrameAnalysis(
                        frame_idx=frame_idx,
                        timestamp_sec=round(frame_idx / src_fps, 3),
                        persons=persons,
                    )
                )
                analyzed += 1

            if writer is not None:
                annotated = draw_persons(bgr, last_persons)
                writer.write(annotated)

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    has_humans = any(len(f.persons) > 0 for f in frame_results)
    summary = _build_person_summary(frame_results) if has_humans else []

    return {
        "has_humans": has_humans,
        "source_fps": round(src_fps, 2),
        "total_frames": frame_idx,
        "duration_sec": round(frame_idx / src_fps, 3),
        "analyzed_frames": analyzed,
        "frames": frame_results,
        "person_summary": summary,
    }
