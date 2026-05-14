import numpy as np
from app.detector import Detection


# Heuristic thresholds for bounding-box aspect ratio (height / width)
_STANDING_RATIO = 1.4  # taller than wide  → standing
_SITTING_RATIO = 0.7   # roughly square    → sitting (between thresholds)
#                        ratio < 0.7       → lying


def _aspect_ratio(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return height / width


def _mask_vertical_bias(mask: np.ndarray) -> float:
    """Return fraction of mask pixels in the top half vs full mask.

    A standing person has pixels more spread vertically; a lying person
    has most pixels in a narrow vertical band near the centre.
    We use this as a secondary signal only when the ratio is ambiguous.
    """
    if mask is None or mask.sum() == 0:
        return 0.5
    h = mask.shape[0]
    top_half = mask[: h // 2].sum()
    total = mask.sum()
    return float(top_half) / float(total)


def classify_pose(detection: Detection) -> str:
    ratio = _aspect_ratio(detection.bbox)

    if ratio >= _STANDING_RATIO:
        return "standing"

    if ratio <= _SITTING_RATIO:
        return "lying"

    # Ambiguous range: use mask vertical distribution as a tiebreaker.
    # A sitting person usually has the torso centred; we look at how much
    # of the mask sits in the top half of the bounding crop.
    if detection.mask is not None:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        crop = detection.mask[y1:y2, x1:x2]
        top_frac = _mask_vertical_bias(crop)
        # Sitting: upper body and head visible → more mass in the top half
        if top_frac > 0.45:
            return "sitting"
        return "lying"

    return "sitting"
