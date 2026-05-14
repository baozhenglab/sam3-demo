import cv2
import numpy as np
from app.schemas import PersonDetection

# BGR colors per pose
_POSE_COLORS: dict[str, tuple[int, int, int]] = {
    "standing": (50, 205, 50),   # green
    "sitting":  (0, 165, 255),   # orange
    "lying":    (60, 60, 220),   # red
}
_DEFAULT_COLOR = (180, 180, 180)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1


def draw_persons(frame: np.ndarray, persons: list[PersonDetection]) -> np.ndarray:
    out = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p.bbox]
        color = _POSE_COLORS.get(p.pose, _DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"ID:{p.id} {p.pose} {p.confidence:.2f}"
        if p.is_moving is True:
            label += " [MOVING]"
        elif p.is_moving is False:
            label += " [STATIC]"

        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        tag_y1 = max(y1 - th - baseline - 4, 0)
        tag_y2 = y1

        # Filled label background
        cv2.rectangle(out, (x1, tag_y1), (x1 + tw + 4, tag_y2), color, cv2.FILLED)
        cv2.putText(
            out, label,
            (x1 + 2, tag_y2 - baseline),
            _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return out
