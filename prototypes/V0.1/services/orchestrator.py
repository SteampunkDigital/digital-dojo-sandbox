"""
GPU Model Orchestrator

Manages loading/unloading of SD3.5 on the GPU for image generation.
3D generation is handled by Trellis2 in a separate conda environment.

Pipeline: Text → Ollama (parse) → SD3.5 (image) → Trellis2 (3D mesh)

External repos:
- SD3.5: c:/Users/david/Documents/GitHub/sd3.5
- Trellis2: G:/GitHub/TRELLIS.2 (runs in separate conda env)
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
    # Trellis2 runs in a separate process/conda environment
    # Ollama runs as separate process, not managed here


@dataclass
class GenerationJob:
    """A job in the generation pipeline"""
    id: str
    prompt: str
    scene_data: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None
    output_path: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None


class GPUOrchestrator:
    """
    Manages GPU resources for SD3.5 image generation.
    3D generation is handled by the Trellis2 worker in a separate environment.
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

        # SD3.5 model config
        self.sd35_model = os.getenv("SD35_MODEL", "sd3.5_medium.safetensors")

        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, running on CPU")

    def clear_gpu_memory(self, force_reset: bool = False):
        """
        Aggressively free GPU memory.

        Args:
            force_reset: If True, attempt to reset CUDA context after OOM errors
        """
        if self.model_instance is not None:
            try:
                if hasattr(self.model_instance, 'cleanup'):
                    self.model_instance.cleanup()
            except Exception as e:
                logger.warning(f"Model cleanup failed: {e}")
            try:
                del self.model_instance
            except Exception:
                pass
            self.model_instance = None

        self.current_model = ModelType.NONE

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA synchronize failed: {e}")

            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"CUDA empty_cache failed: {e}")
                force_reset = True

            if force_reset:
                logger.warning("Attempting CUDA context reset after OOM...")
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except Exception:
                    pass
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"CUDA recovery failed: {e}. May need worker restart.")

        try:
            logger.info(f"GPU memory cleared. VRAM: {self.get_vram_usage()}")
        except Exception:
            logger.info("GPU memory cleared (unable to query VRAM)")

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

        self.clear_gpu_memory()

        logger.info(f"Loading SD3.5 from {self.sd35_repo}...")

        self._add_to_path(self.sd35_repo)

        try:
            from sd3_infer import SD3Inferencer

            with torch.no_grad():
                self.model_instance = SD3Inferencer()

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
        This improves Trellis2's background removal and 3D generation quality.
        """
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
                                   and centered object (recommended for 3D pipeline)
        """
        self.load_stable_diffusion()

        if enhance_for_isolation:
            prompt = self._enhance_prompt_for_isolation(prompt)
            logger.info(f"Enhanced prompt: {prompt[:80]}...")
        else:
            logger.info(f"Generating image: {prompt[:50]}...")

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        output_id = uuid.uuid4().hex[:8]
        output_subdir = self.output_dir / f"sd_{output_id}"
        output_subdir.mkdir(parents=True, exist_ok=True)

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

        generated_images = list(output_subdir.rglob("*.png"))
        if not generated_images:
            raise RuntimeError("No image was generated")

        output_path = generated_images[0]
        logger.info(f"Image saved: {output_path}")
        return str(output_path)

    @staticmethod
    def get_output_format() -> str:
        """Get the configured output format. Trellis2 always produces GLB mesh."""
        return "mesh"

    # ==================== Pipeline ====================

    def run_image_only(self, prompt: str, **kwargs) -> str:
        """Run only the image generation stage"""
        try:
            result = self.generate_image(prompt, **kwargs)
            return result
        finally:
            self.clear_gpu_memory()
