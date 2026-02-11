"""
SAM3 Service - Text-prompted object segmentation using Meta SAM 3.

Lazy-loads SAM3, caches image encodings per item, and provides
text-prompted mask prediction + white-background compositing.
"""
import os
import gc
import sys
import time
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SAM3_REPO_PATH = os.getenv(
    "SAM3_REPO_PATH",
    "c:/Users/david/Documents/GitHub/sam3"
)
SAM3_AUTO_UNLOAD_SECONDS = int(os.getenv("SAM3_AUTO_UNLOAD_SECONDS", "120"))


class SAM3Service:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cached_item_id = None
        self._cached_state = None
        self._last_use_time = 0

    @property
    def loaded(self):
        return self.processor is not None

    def load(self):
        """Lazy-load SAM3 model."""
        if self.processor is not None:
            self._last_use_time = time.time()
            return

        # Add SAM3 to path
        if SAM3_REPO_PATH not in sys.path:
            sys.path.insert(0, SAM3_REPO_PATH)

        logger.info("Loading SAM3 model...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(device=self.device)
        self.processor = Sam3Processor(self.model, device=self.device)
        self._last_use_time = time.time()
        logger.info("SAM3 loaded")

    def unload(self):
        """Free GPU memory."""
        if self.processor is None:
            return
        logger.info("Unloading SAM3 from GPU")
        del self.processor
        del self.model
        self.processor = None
        self.model = None
        self._cached_item_id = None
        self._cached_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_auto_unload(self):
        """Unload if idle for too long."""
        if self.processor is None:
            return
        if SAM3_AUTO_UNLOAD_SECONDS > 0:
            idle = time.time() - self._last_use_time
            if idle > SAM3_AUTO_UNLOAD_SECONDS:
                logger.info(f"SAM3 idle for {idle:.0f}s, auto-unloading")
                self.unload()

    def _set_image(self, item_id, image_path):
        """Encode image for SAM3. Cached per item_id."""
        if self._cached_item_id == item_id and self._cached_state is not None:
            return
        logger.info(f"Encoding image for SAM3: {image_path}")
        image = Image.open(image_path).convert("RGB")
        self._cached_state = self.processor.set_image(image)
        self._cached_item_id = item_id

    def predict_text(self, item_id, image_path, text_prompt):
        """
        Run SAM3 text-prompted segmentation.

        Args:
            item_id: Item ID (for image encoding cache)
            image_path: Path to the original image
            text_prompt: Object name to segment (e.g. "coffee mug")

        Returns:
            (mask_array, score) where mask_array is boolean [H, W]
        """
        self.load()
        self._set_image(item_id, image_path)
        self._last_use_time = time.time()

        # Reset any previous prompts (keeps image encoding)
        self.processor.reset_all_prompts(self._cached_state)

        # Run text-prompted segmentation
        state = self.processor.set_text_prompt(
            prompt=text_prompt,
            state=self._cached_state
        )
        self._cached_state = state

        if 'masks' not in state or state['masks'].numel() == 0:
            img = Image.open(image_path)
            h, w = img.height, img.width
            return np.zeros((h, w), dtype=bool), 0.0

        scores = state['scores']
        masks = state['masks']  # [N, 1, H, W]

        best_idx = scores.argmax().item()
        best_mask = masks[best_idx, 0].cpu().numpy()  # [H, W] boolean
        best_score = float(scores[best_idx].item())

        logger.info(f"SAM3 text prompt '{text_prompt}': score={best_score:.3f}, "
                     f"{masks.shape[0]} candidates")
        return best_mask, best_score

    def predict_point(self, item_id, image_path, points, text_prompt=None):
        """
        Run SAM3 with geometric (box) prompts from user click points.
        Converts click points to small boxes for SAM3's geometric prompt API.

        Args:
            item_id: Item ID (for image encoding cache)
            image_path: Path to the original image
            points: List of dicts with x, y (pixel coords), label (1=fg, 0=bg)
            text_prompt: Optional text prompt to combine with geometric prompts

        Returns:
            (mask_array, score) where mask_array is boolean [H, W]
        """
        self.load()
        self._set_image(item_id, image_path)
        self._last_use_time = time.time()

        img = Image.open(image_path)
        img_w, img_h = img.size

        # Reset prompts (keeps image encoding)
        self.processor.reset_all_prompts(self._cached_state)

        # Re-encode image since reset clears backbone features we need
        image = Image.open(image_path).convert("RGB")
        self._cached_state = self.processor.set_image(image)

        # Set text prompt if provided
        if text_prompt:
            self._cached_state = self.processor.set_text_prompt(
                prompt=text_prompt,
                state=self._cached_state
            )

        # Add geometric prompts from points (convert to small boxes)
        box_size = 0.02  # 2% of image dimension
        for p in points:
            cx = p['x'] / img_w  # normalize to [0, 1]
            cy = p['y'] / img_h
            box = [cx, cy, box_size, box_size]
            label = bool(p['label'])
            self._cached_state = self.processor.add_geometric_prompt(
                box=box,
                label=label,
                state=self._cached_state
            )

        state = self._cached_state

        if 'masks' not in state or state['masks'].numel() == 0:
            return np.zeros((img_h, img_w), dtype=bool), 0.0

        scores = state['scores']
        masks = state['masks']

        best_idx = scores.argmax().item()
        best_mask = masks[best_idx, 0].cpu().numpy()
        best_score = float(scores[best_idx].item())

        return best_mask, best_score

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
_sam3_service = None


def get_sam_service():
    global _sam3_service
    if _sam3_service is None:
        _sam3_service = SAM3Service()
    return _sam3_service
