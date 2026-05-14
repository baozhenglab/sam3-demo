import os
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: list[float]   # [x1, y1, x2, y2]
    score: float
    mask: np.ndarray    # H x W binary mask


class SAM3HumanDetector:
    def __init__(self):
        model_id = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
        device_str = os.getenv("DEVICE", "cpu")

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            device_str = "cpu"

        self.device = torch.device(device_str)
        self.threshold = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
        self.mask_threshold = float(os.getenv("MASK_THRESHOLD", "0.5"))

        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[Detection]:
        inputs = self.processor(
            images=image,
            text="person",
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        detections: list[Detection] = []
        for i in range(len(results["scores"])):
            score = float(results["scores"][i])
            bbox = results["boxes"][i].tolist()
            mask = results["masks"][i].cpu().numpy().astype(np.uint8)
            detections.append(Detection(bbox=bbox, score=score, mask=mask))

        return detections
