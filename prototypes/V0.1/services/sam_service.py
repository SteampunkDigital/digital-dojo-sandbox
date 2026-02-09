"""
SAM (Segment Anything) service for interactive object masking.

Lazy-loads SAM vit_h, caches image encodings per item, and provides
mask prediction from point prompts + white-background compositing.
"""
import os
import gc
import time
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SAM_CHECKPOINT = os.getenv(
    "SAM_CHECKPOINT",
    "/mnt/c/Users/david/Documents/GitHub/sam3/sam_vit_h_4b8939.pth"
)
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")
SAM_AUTO_UNLOAD_SECONDS = int(os.getenv("SAM_AUTO_UNLOAD_SECONDS", "60"))


class SAMService:
    def __init__(self):
        self.predictor = None
        self.sam_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cached_item_id = None
        self._last_use_time = 0

    @property
    def loaded(self):
        return self.predictor is not None

    def load(self):
        """Lazy-load SAM model (~2.5GB VRAM for vit_h)."""
        if self.predictor is not None:
            self._last_use_time = time.time()
            return

        logger.info(f"Loading SAM ({SAM_MODEL_TYPE}) from {SAM_CHECKPOINT}")
        from segment_anything import sam_model_registry, SamPredictor

        self.sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        self.sam_model.to(device=self.device)
        self.predictor = SamPredictor(self.sam_model)
        self._last_use_time = time.time()
        logger.info("SAM loaded")

    def unload(self):
        """Free GPU memory."""
        if self.predictor is None:
            return
        logger.info("Unloading SAM from GPU")
        del self.predictor
        del self.sam_model
        self.predictor = None
        self.sam_model = None
        self._cached_item_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_auto_unload(self):
        """Unload if idle for too long. Call at start of each request."""
        if self.predictor is None:
            return
        if SAM_AUTO_UNLOAD_SECONDS > 0:
            idle = time.time() - self._last_use_time
            if idle > SAM_AUTO_UNLOAD_SECONDS:
                logger.info(f"SAM idle for {idle:.0f}s, auto-unloading")
                self.unload()

    def _set_image(self, item_id, image_path):
        """Encode image for SAM. Cached per item_id."""
        if self._cached_item_id == item_id:
            return
        logger.info(f"Encoding image for SAM: {image_path}")
        image_np = np.array(Image.open(image_path).convert("RGB"))
        self.predictor.set_image(image_np)
        self._cached_item_id = item_id

    def predict(self, item_id, image_path, points):
        """
        Run SAM prediction from point prompts.

        Args:
            item_id: Item ID (for image encoding cache)
            image_path: Path to the original image
            points: List of dicts with x, y, label (1=foreground, 0=background)

        Returns:
            (mask_array, score) where mask_array is boolean [H, W]
        """
        self.load()
        self._set_image(item_id, image_path)
        self._last_use_time = time.time()

        coords = np.array([[p["x"], p["y"]] for p in points])
        labels = np.array([p["label"] for p in points])

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        return masks[best_idx], float(scores[best_idx])

    def composite_on_white(self, image_path, mask, output_size=1024):
        """
        Extract masked object and composite onto white background.

        Args:
            image_path: Path to the original image
            mask: Boolean mask array [H, W]
            output_size: Output image size (square)

        Returns:
            PIL Image (RGB, output_size x output_size)
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Build RGBA with mask as alpha
        h, w = image_np.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image_np
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)

        # Crop to mask bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            # Empty mask, return white image
            return Image.new("RGB", (output_size, output_size), (255, 255, 255))

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = Image.fromarray(rgba[rmin:rmax + 1, cmin:cmax + 1])

        # Scale to fill ~80% of output, centered
        max_dim = max(cropped.size)
        target = int(output_size * 0.8)
        scale = target / max_dim
        new_w = int(cropped.width * scale)
        new_h = int(cropped.height * scale)
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

        # Paste onto white background
        white = Image.new("RGB", (output_size, output_size), (255, 255, 255))
        offset_x = (output_size - new_w) // 2
        offset_y = (output_size - new_h) // 2
        white.paste(cropped, (offset_x, offset_y), mask=cropped.split()[3])

        return white


# Singleton
_sam_service = None


def get_sam_service():
    global _sam_service
    if _sam_service is None:
        _sam_service = SAMService()
    return _sam_service
