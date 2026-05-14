import os
import numpy as np
from app.detector import Detection


_MOVEMENT_THRESHOLD = float(os.getenv("MOVEMENT_THRESHOLD_PX", "15"))


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class PersonTracker:
    """Greedy IoU tracker that assigns consistent IDs across frames
    and detects whether each person is moving."""

    def __init__(self):
        # {person_id: {"bbox": [...], "center": (cx, cy)}}
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[tuple[int, bool | None]]:
        """Match detections to existing tracks.

        Returns a list of (person_id, is_moving) for each detection in the
        same order as `detections`. `is_moving` is None when no previous
        track exists for that detection.
        """
        if not self._tracks:
            results = []
            for det in detections:
                pid = self._next_id
                self._next_id += 1
                self._tracks[pid] = {"bbox": det.bbox, "center": _center(det.bbox)}
                results.append((pid, None))
            return results

        current_boxes = [d.bbox for d in detections]
        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid]["bbox"] for tid in track_ids]

        # IoU matrix: rows = tracks, cols = detections
        iou_matrix = np.array(
            [[_iou(tb, cb) for cb in current_boxes] for tb in track_boxes],
            dtype=np.float32,
        )

        matched_track: dict[int, int] = {}   # det_idx → track_id
        used_tracks: set[int] = set()

        # Greedy matching: highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            row, col = int(idx[0]), int(idx[1])
            if iou_matrix[row, col] < 0.3:
                break
            tid = track_ids[row]
            if tid not in used_tracks and col not in matched_track:
                matched_track[col] = tid
                used_tracks.add(tid)
            iou_matrix[row, :] = -1
            iou_matrix[:, col] = -1

        results: list[tuple[int, bool | None]] = []
        for det_idx, det in enumerate(detections):
            new_center = _center(det.bbox)

            if det_idx in matched_track:
                tid = matched_track[det_idx]
                prev_center = self._tracks[tid]["center"]
                dist = float(
                    np.sqrt(
                        (new_center[0] - prev_center[0]) ** 2
                        + (new_center[1] - prev_center[1]) ** 2
                    )
                )
                is_moving = dist >= _MOVEMENT_THRESHOLD
                self._tracks[tid] = {"bbox": det.bbox, "center": new_center}
                results.append((tid, is_moving))
            else:
                pid = self._next_id
                self._next_id += 1
                self._tracks[pid] = {"bbox": det.bbox, "center": new_center}
                results.append((pid, None))

        return results
