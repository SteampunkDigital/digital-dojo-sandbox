"""
GPU Model Orchestrator

Manages loading/unloading of ML models on a single GPU to avoid VRAM conflicts.
Models are loaded on-demand and unloaded after use to free memory.

Pipeline: Text → Ollama (parse) → SD3.5 (image) → SAM (mask) → SAM3D (splat)

External repos:
- SD3.5: c:/Users/david/Documents/GitHub/sd3.5
- SAM3D: c:/Users/david/Documents/GitHub/sam-3d-objects

SAM (Segment Anything Model) is used for precise object segmentation with center-point prompting.
"""

import os
import sys
import torch
import gc
import logging
import uuid
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    NONE = "none"
    STABLE_DIFFUSION = "stable_diffusion"
    SAM = "sam"  # Segment Anything Model for masking
    SAM3D = "sam3d"
    # Ollama runs as separate process, not managed here


@dataclass
class GenerationJob:
    """A job in the generation pipeline"""
    id: str
    prompt: str
    scene_data: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None
    splat_path: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None


class GPUOrchestrator:
    """
    Manages GPU resources and model lifecycle.
    Only one heavy model loaded at a time.
    """

    def __init__(self, output_dir: str = "media/generated"):
        self.current_model: ModelType = ModelType.NONE
        self.model_instance: Optional[Any] = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # External repo paths (configurable via env)
        # Use /mnt/c/ paths for WSL compatibility
        self.sd35_repo = Path(os.getenv(
            "SD35_REPO_PATH",
            "/mnt/c/Users/david/Documents/GitHub/sd3.5"
        ))
        self.sam3d_repo = Path(os.getenv(
            "SAM3D_REPO_PATH",
            "/mnt/c/Users/david/Documents/GitHub/sam-3d-objects"
        ))

        # SD3.5 model config
        self.sd35_model = os.getenv("SD35_MODEL", "sd3.5_medium.safetensors")

        # SAM model config (for object segmentation before SAM3D)
        self.sam_checkpoint = os.getenv(
            "SAM_CHECKPOINT",
            "/mnt/c/Users/david/Documents/GitHub/sam3/sam_vit_h_4b8939.pth"
        )
        self.sam_model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
        self._sam_predictor = None

        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, running on CPU")

    def clear_gpu_memory(self):
        """Aggressively free GPU memory"""
        if self.model_instance is not None:
            # Try to cleanup model-specific resources
            if hasattr(self.model_instance, 'cleanup'):
                self.model_instance.cleanup()
            del self.model_instance
            self.model_instance = None

        # Also unload SAM if loaded
        if self._sam_predictor is not None:
            del self._sam_predictor
            self._sam_predictor = None

        self.current_model = ModelType.NONE
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(f"GPU memory cleared. VRAM: {self.get_vram_usage()}")

    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in GB"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "total": 0}

        return {
            "allocated": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved": round(torch.cuda.memory_reserved() / 1e9, 2),
            "total": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        }

    def _add_to_path(self, repo_path: Path):
        """Add repo to Python path if not already there"""
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    # ==================== SD3.5 Integration ====================

    def load_stable_diffusion(self):
        """Load SD3.5 model from local repo"""
        if self.current_model == ModelType.STABLE_DIFFUSION:
            return  # Already loaded

        # Clear any existing model first
        self.clear_gpu_memory()

        logger.info(f"Loading SD3.5 from {self.sd35_repo}...")

        # Add SD3.5 repo to path
        self._add_to_path(self.sd35_repo)

        try:
            from sd3_infer import SD3Inferencer

            # Wrap in no_grad to avoid in-place operation errors with newer PyTorch
            with torch.no_grad():
                self.model_instance = SD3Inferencer()

                # Load models - keep text encoders on CPU to save VRAM
                model_path = self.sd35_repo / "models" / self.sd35_model
                self.model_instance.load(
                    model=str(model_path),
                    shift=3.0,
                    model_folder=str(self.sd35_repo / "models"),
                    text_encoder_device="cpu",
                    verbose=False
                )

            self.current_model = ModelType.STABLE_DIFFUSION
            logger.info(f"SD3.5 loaded. VRAM: {self.get_vram_usage()}")

        except ImportError as e:
            raise RuntimeError(f"Failed to import SD3.5: {e}. Check SD35_REPO_PATH.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SD3.5: {e}")

    def _enhance_prompt_for_isolation(self, prompt: str) -> str:
        """
        Enhance prompt to generate isolated objects on clean backgrounds.
        This improves background removal and splat generation quality.
        """
        # Keywords that indicate we should add isolation instructions
        isolation_suffix = (
            ", centered in frame, isolated object, plain white background, "
            "studio lighting, product photography style, no shadows on background, "
            "object does not touch edges of image, clean simple backdrop"
        )
        return prompt + isolation_suffix

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024,
                       steps: int = 40, cfg_scale: float = 4.5,
                       seed: int = None, enhance_for_isolation: bool = True) -> str:
        """
        Generate image using SD3.5.
        Returns path to generated image.

        Args:
            prompt: Text description of the object
            enhance_for_isolation: If True, adds instructions for clean background
                                   and centered object (recommended for splat pipeline)
        """
        self.load_stable_diffusion()

        # Enhance prompt for better isolation (helps with bg removal + splat)
        if enhance_for_isolation:
            prompt = self._enhance_prompt_for_isolation(prompt)
            logger.info(f"Enhanced prompt: {prompt[:80]}...")
        else:
            logger.info(f"Generating image: {prompt[:50]}...")

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        # Generate unique output path
        output_id = uuid.uuid4().hex[:8]
        output_subdir = self.output_dir / f"sd_{output_id}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Generate image
        # Note: SD3.5 doesn't use negative prompts the same way as SDXL
        # The isolation is handled entirely through the positive prompt enhancement
        self.model_instance.gen_image(
            prompts=[prompt],
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            seed_type="fixed",
            out_dir=str(output_subdir),
            sampler="dpmpp_2m"
        )

        # Find the generated image
        generated_images = list(output_subdir.rglob("*.png"))
        if not generated_images:
            raise RuntimeError("No image was generated")

        output_path = generated_images[0]
        logger.info(f"Image saved: {output_path}")
        return str(output_path)

    # ==================== SAM Integration ====================

    def load_sam(self):
        """
        Load SAM (Segment Anything Model) for object segmentation.
        SAM runs separately from the main model pipeline - it's loaded
        alongside other models when needed for masking.
        """
        if self._sam_predictor is not None:
            return  # Already loaded

        logger.info(f"Loading SAM from {self.sam_checkpoint}...")

        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self._sam_predictor = SamPredictor(sam)

            logger.info(f"SAM loaded. VRAM: {self.get_vram_usage()}")

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SAM: {e}. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"SAM checkpoint not found: {self.sam_checkpoint}. "
                "Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )

    def segment_with_sam(self, image: Image.Image) -> np.ndarray:
        """
        Segment the main object in an image using SAM with center-point prompting.

        Assumes the object is centered in the image (which our SD3.5 prompts ensure).
        Returns a binary mask where 255 = object, 0 = background.
        """
        self.load_sam()

        # Convert PIL image to numpy array
        image_np = np.array(image.convert('RGB'))

        # Set image for SAM
        self._sam_predictor.set_image(image_np)

        # Use center point as prompt (object should be centered)
        h, w = image_np.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])  # 1 = foreground

        # Get mask prediction
        masks, scores, _ = self._sam_predictor.predict(
            point_coords=center_point,
            point_labels=point_labels,
            multimask_output=True
        )

        # Select the mask with highest score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]

        # Convert to uint8 (0 or 255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        fg_pixels = np.sum(mask)
        logger.info(f"SAM mask: {fg_pixels} foreground pixels ({100*fg_pixels/mask.size:.1f}%)")

        return mask_uint8

    def generate_mask(self, image_path: str) -> str:
        """
        Generate a segmentation mask for an image using SAM.
        Saves the mask to disk and returns the path.

        Uses center-point prompting (assumes object is centered,
        which our SD3.5 prompts ensure).

        Args:
            image_path: Path to input image

        Returns:
            Path to saved mask image (PNG, grayscale)
        """
        self.load_sam()

        logger.info(f"Generating mask for: {image_path}")

        # Load and segment image
        image = Image.open(image_path)
        mask = self.segment_with_sam(image)

        # Save mask next to the image
        image_dir = Path(image_path).parent
        mask_filename = Path(image_path).stem + "_mask.png"
        mask_path = image_dir / mask_filename

        Image.fromarray(mask).save(str(mask_path))
        logger.info(f"Mask saved: {mask_path}")

        return str(mask_path)

    def unload_sam(self):
        """Unload SAM to free VRAM for other models"""
        if self._sam_predictor is not None:
            del self._sam_predictor
            self._sam_predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SAM unloaded")

    # ==================== SAM3D Integration ====================

    def load_sam3d(self):
        """Load SAM3D model from local repo"""
        if self.current_model == ModelType.SAM3D:
            return  # Already loaded

        # Clear any existing model first
        self.clear_gpu_memory()

        logger.info(f"Loading SAM3D from {self.sam3d_repo}...")

        # Add SAM3D repo and notebook to path
        self._add_to_path(self.sam3d_repo)
        self._add_to_path(self.sam3d_repo / "notebook")

        try:
            from inference import Inference

            config_path = self.sam3d_repo / "checkpoints" / "hf" / "pipeline.yaml"
            self.model_instance = Inference(str(config_path), compile=False)

            self.current_model = ModelType.SAM3D
            logger.info(f"SAM3D loaded. VRAM: {self.get_vram_usage()}")

        except ImportError as e:
            raise RuntimeError(f"Failed to import SAM3D: {e}. Check SAM3D_REPO_PATH.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3D: {e}")

    def generate_splat(self, image_path: str, mask_path: str,
                       seed: int = 42) -> str:
        """
        Generate Gaussian splat from image + mask using SAM3D.
        Returns path to generated .ply file.

        IMPORTANT: mask_path is required. Use generate_mask() first to create it.
        This separation allows efficient GPU batching (all masks, then all splats).

        Args:
            image_path: Path to input image (PNG)
            mask_path: Path to mask image (required)
            seed: Random seed for reproducibility
        """
        if not mask_path:
            raise ValueError("mask_path is required. Use generate_mask() first.")

        logger.info(f"Generating splat from: {image_path} with mask: {mask_path}")

        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # Load SAM3D for splat generation
        self.load_sam3d()

        # Convert image to numpy array (RGB)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_np = np.array(image)

        # Run SAM3D inference
        output = self.model_instance(image_np, mask, seed=seed)

        # Save the gaussian splat
        output_path = self.output_dir / f"splat_{uuid.uuid4().hex[:8]}.ply"
        output["gs"].save_ply(str(output_path))

        logger.info(f"Splat saved: {output_path}")
        return str(output_path)

    # ==================== Pipeline ====================

    def run_pipeline(self, job: GenerationJob) -> GenerationJob:
        """
        Run the full generation pipeline for a job.

        1. Generate image from prompt (SD3.5)
        2. Segment object with SAM (center-point) → mask
        3. Generate splat from image + mask (SAM3D)

        Args:
            job: The generation job with prompt
        """
        try:
            job.status = "generating_image"

            # Get prompt from scene data or job
            if job.scene_data and "generator" in job.scene_data:
                prompt = job.scene_data["generator"].get("prompt", job.prompt)
            else:
                prompt = job.prompt

            # Stage 1: Generate image with SD3.5
            logger.info(f"Pipeline Stage 1: Generating image for '{prompt[:50]}...'")
            job.image_path = self.generate_image(prompt)

            # Clear SD3.5 from GPU
            self.clear_gpu_memory()

            # Stage 2: Generate splat with SAM3D (uses brightness auto-threshold for masking)
            job.status = "generating_splat"
            logger.info(f"Pipeline Stage 2: Generating splat from {job.image_path}")
            job.splat_path = self.generate_splat(job.image_path)

            job.status = "completed"
            logger.info(f"Pipeline complete: {job.splat_path}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Pipeline failed: {e}", exc_info=True)

        finally:
            # Free GPU memory after job completes
            self.clear_gpu_memory()

        return job

    def run_image_only(self, prompt: str, **kwargs) -> str:
        """Run only the image generation stage"""
        try:
            result = self.generate_image(prompt, **kwargs)
            return result
        finally:
            self.clear_gpu_memory()

    def run_splat_only(self, image_path: str, **kwargs) -> str:
        """
        Run only the splat generation stage.
        Uses brightness auto-threshold for masking white backgrounds.

        Args:
            image_path: Path to input image
        """
        try:
            result = self.generate_splat(image_path, **kwargs)
            return result
        finally:
            self.clear_gpu_memory()
