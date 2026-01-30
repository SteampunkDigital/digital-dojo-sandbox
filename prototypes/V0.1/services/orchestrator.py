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

    def clear_gpu_memory(self, force_reset: bool = False):
        """
        Aggressively free GPU memory.

        Args:
            force_reset: If True, attempt to reset CUDA context after OOM errors
        """
        # Delete model references first
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

        # Also unload SAM if loaded
        if self._sam_predictor is not None:
            try:
                del self._sam_predictor
            except Exception:
                pass
            self._sam_predictor = None

        self.current_model = ModelType.NONE

        # Force Python garbage collection
        gc.collect()

        # CUDA cleanup with error handling for corrupted state
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

            # If CUDA is in a bad state, try more aggressive reset
            if force_reset:
                logger.warning("Attempting CUDA context reset after OOM...")
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except Exception:
                    pass
                # Final attempt - may need process restart if this fails
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

    def _find_foreground_points(self, image_np: np.ndarray, max_points: int = 5,
                                  num_samples: int = 200, min_separation: float = 0.15) -> list:
        """
        Find multiple spread-out points likely to be on foreground objects.

        Strategy:
        1. Sample edge pixels to estimate background color
        2. Sample random points across the image
        3. Find points that differ significantly from background
        4. Use greedy selection to pick spatially spread points (for multiple objects)

        Args:
            image_np: RGB image as numpy array (H, W, 3)
            max_points: Maximum number of points to return (up to 5)
            num_samples: Number of random points to sample
            min_separation: Minimum distance between selected points as fraction of image size

        Returns:
            List of (x, y) tuples for foreground points
        """
        h, w = image_np.shape[:2]
        min_dist_pixels = min(h, w) * min_separation

        # Sample edge pixels to estimate background color
        edge_pixels = []
        border = 20  # pixels from edge
        # Top edge
        edge_pixels.extend(image_np[:border, :, :].reshape(-1, 3).tolist())
        # Bottom edge
        edge_pixels.extend(image_np[-border:, :, :].reshape(-1, 3).tolist())
        # Left edge
        edge_pixels.extend(image_np[:, :border, :].reshape(-1, 3).tolist())
        # Right edge
        edge_pixels.extend(image_np[:, -border:, :].reshape(-1, 3).tolist())

        edge_pixels = np.array(edge_pixels)
        bg_color = np.median(edge_pixels, axis=0)  # median is robust to outliers
        logger.info(f"Estimated background color: RGB{tuple(bg_color.astype(int))}")

        # Sample random points across the central region of the image
        margin = int(min(h, w) * 0.1)  # 10% margin
        np.random.seed(42)  # reproducible
        sample_y = np.random.randint(margin, h - margin, num_samples)
        sample_x = np.random.randint(margin, w - margin, num_samples)

        # Calculate color distance from background for each sample
        candidates = []
        for x, y in zip(sample_x, sample_y):
            pixel_color = image_np[y, x, :]
            # Euclidean distance in RGB space
            color_dist = np.sqrt(np.sum((pixel_color.astype(float) - bg_color) ** 2))
            candidates.append((x, y, color_dist))

        # Filter to points that are significantly different from background
        bg_threshold = 50
        foreground_candidates = [(x, y, cd) for x, y, cd in candidates if cd > bg_threshold]

        if not foreground_candidates:
            logger.warning("No clear foreground points found, using center")
            return [(w // 2, h // 2)]

        # Sort by color distance (strongest foreground signal first)
        foreground_candidates.sort(key=lambda p: p[2], reverse=True)

        # Greedy selection: pick points that are far apart from already selected points
        selected_points = []
        for x, y, cd in foreground_candidates:
            if len(selected_points) >= max_points:
                break

            # Check distance from all already selected points
            is_far_enough = True
            for sx, sy in selected_points:
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < min_dist_pixels:
                    is_far_enough = False
                    break

            if is_far_enough:
                selected_points.append((x, y))
                logger.info(f"Selected foreground point {len(selected_points)}: ({x}, {y}) color_dist={cd:.1f}")

        logger.info(f"Found {len(selected_points)} spread foreground points")
        return selected_points

    def segment_with_sam(self, image: Image.Image) -> np.ndarray:
        """
        Segment all foreground objects in an image using SAM with multi-point prompting.

        Uses background color estimation to find multiple spread-out foreground points,
        capturing multiple objects if present in the scene.

        Returns a binary mask where 255 = object, 0 = background.
        """
        self.load_sam()

        # Convert PIL image to numpy array
        image_np = np.array(image.convert('RGB'))
        h, w = image_np.shape[:2]

        # Set image for SAM
        self._sam_predictor.set_image(image_np)

        # Find multiple spread-out foreground points
        fg_points = self._find_foreground_points(image_np, max_points=5)

        # Convert to arrays for SAM
        prompt_points = np.array(fg_points)
        point_labels = np.ones(len(fg_points), dtype=np.int32)  # All foreground

        logger.info(f"SAM prompt: {len(fg_points)} points in {w}x{h} image")

        # Get mask prediction with multiple points
        masks, scores, _ = self._sam_predictor.predict(
            point_coords=prompt_points,
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

        # Debug: log mask stats before conversion
        fg_pixels_before = np.sum(mask > 0)
        logger.info(f"Mask loaded: shape={mask.shape}, dtype={mask.dtype}, "
                    f"min={mask.min()}, max={mask.max()}, "
                    f"fg_pixels={fg_pixels_before} ({100*fg_pixels_before/mask.size:.1f}%)")

        # Convert to boolean mask - SAM3D expects boolean, not uint8!
        # (see sam-3d-objects/notebook/inference.py load_mask function)
        mask = mask > 127  # Returns boolean array (True/False)
        fg_pixels = np.sum(mask)
        logger.info(f"Boolean mask: dtype={mask.dtype}, fg_pixels={fg_pixels} ({100*fg_pixels/mask.size:.1f}%)")

        # Sanity check: mask should have some foreground
        if fg_pixels == 0:
            raise ValueError("Mask has no foreground pixels - SAM may have failed to segment the object")

        if fg_pixels < 100:
            logger.warning(f"Mask has very few foreground pixels ({fg_pixels}), splat quality may be poor")

        # Load SAM3D for splat generation
        self.load_sam3d()

        # Convert image to numpy array (RGB)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_np = np.array(image)

        # Debug: log image stats
        logger.info(f"Image: shape={image_np.shape}, dtype={image_np.dtype}")

        # Verify dimensions match
        if mask.shape[:2] != image_np.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {image_np.shape[:2]}")

        # Run SAM3D inference
        logger.info(f"Running SAM3D inference (seed={seed})...")
        logger.info(f"VRAM before inference: {self.get_vram_usage()}")
        try:
            output = self.model_instance(image_np, mask, seed=seed)
        except Exception as e:
            logger.error(f"SAM3D inference failed: {e}")
            logger.error(f"  Image shape: {image_np.shape}, dtype: {image_np.dtype}")
            logger.error(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}, fg: {fg_pixels}")
            raise

        # NOTE: SAM3D outputs Z-up coordinates, but we do NOT transform here.
        # Rotating positions without rotating spherical harmonics causes artifacts.
        # Instead, apply rotation as a parent transform in the viewer/scene.
        gs = output["gs"]

        # Center the splat at origin so rotations work correctly
        # Without centering, rotating around origin shifts the object off-center
        try:
            xyz = gs.get_xyz  # Shape: (N, 3)
            center = xyz.mean(dim=0)  # Centroid of all Gaussians
            centered_xyz = xyz - center
            gs.from_xyz(centered_xyz)
            logger.info(f"Centered splat: moved by ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        except Exception as e:
            logger.warning(f"Could not center splat: {e}")

        # Save the gaussian splat
        output_path = self.output_dir / f"splat_{uuid.uuid4().hex[:8]}.ply"
        gs.save_ply(str(output_path))

        # Aggressive cleanup to prevent memory accumulation across jobs
        del output
        del image_np
        del mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(f"Splat saved: {output_path}")
        logger.info(f"VRAM after cleanup: {self.get_vram_usage()}")
        return str(output_path)

    def generate_mesh(self, image_path: str, mask_path: str,
                      seed: int = 42) -> str:
        """
        Generate GLB mesh from image + mask using SAM3D with mesh postprocessing.
        Returns path to generated .glb file.

        Args:
            image_path: Path to input image (PNG)
            mask_path: Path to mask image (required)
            seed: Random seed for reproducibility
        """
        if not mask_path:
            raise ValueError("mask_path is required. Use generate_mask() first.")

        logger.info(f"Generating mesh from: {image_path} with mask: {mask_path}")

        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # Convert to boolean mask
        mask = mask > 127
        fg_pixels = np.sum(mask)
        logger.info(f"Boolean mask: dtype={mask.dtype}, fg_pixels={fg_pixels} ({100*fg_pixels/mask.size:.1f}%)")

        if fg_pixels == 0:
            raise ValueError("Mask has no foreground pixels - SAM may have failed to segment the object")

        # Load SAM3D
        self.load_sam3d()

        # Convert image to numpy array (RGB)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_np = np.array(image)

        # Merge mask to RGBA for SAM3D
        rgba_image = self.model_instance.merge_mask_to_rgba(image_np, mask)

        # Run SAM3D pipeline with mesh postprocessing enabled
        logger.info(f"Running SAM3D inference with mesh generation (seed={seed})...")
        logger.info(f"VRAM before inference: {self.get_vram_usage()}")

        # Use vertex colors directly (texture baking requires diff-gaussian-rasterization
        # which has complex build requirements). SAM3D's to_glb only applies vertex colors
        # when with_mesh_postprocess=False and with_texture_baking=False.
        logger.info("Running SAM3D with vertex colors (no texture baking)")
        try:
            output = self.model_instance._pipeline.run(
                rgba_image,
                None,
                seed,
                stage1_only=False,
                with_mesh_postprocess=False,  # Must be False for vertex colors to work
                with_texture_baking=False,
                with_layout_postprocess=False,
                use_vertex_color=True,  # Enable vertex colors
                stage1_inference_steps=None,
                pointmap=None,
            )
        except Exception as e:
            logger.error(f"SAM3D mesh inference failed: {e}")
            raise

        # Save the GLB mesh
        glb = output.get("glb")
        if glb is None:
            raise RuntimeError("SAM3D did not return mesh output - check with_mesh_postprocess")

        # Debug: Check if mesh has vertex colors
        if hasattr(glb, 'visual') and hasattr(glb.visual, 'vertex_colors'):
            vc = glb.visual.vertex_colors
            if vc is not None:
                logger.info(f"Mesh has vertex colors: shape={vc.shape}, dtype={vc.dtype}, "
                           f"min={vc.min()}, max={vc.max()}")
            else:
                logger.warning("Mesh visual.vertex_colors is None")
        else:
            logger.warning("Mesh does not have vertex_colors attribute")

        # Ensure vertex colors are properly formatted for GLB export
        # trimesh expects RGBA uint8 [0-255] for best GLB compatibility
        if hasattr(glb, 'visual') and glb.visual is not None:
            if hasattr(glb.visual, 'vertex_colors') and glb.visual.vertex_colors is not None:
                vc = glb.visual.vertex_colors
                # Check if colors are in float [0-1] range and need conversion
                if vc.dtype in [np.float32, np.float64]:
                    if vc.max() <= 1.0:
                        # Convert from [0, 1] float to [0, 255] uint8
                        vc_uint8 = (vc * 255).astype(np.uint8)
                        # Ensure RGBA (add alpha if needed)
                        if vc_uint8.shape[1] == 3:
                            alpha = np.full((vc_uint8.shape[0], 1), 255, dtype=np.uint8)
                            vc_uint8 = np.concatenate([vc_uint8, alpha], axis=1)
                        glb.visual.vertex_colors = vc_uint8
                        logger.info(f"Converted vertex colors to uint8 RGBA: shape={vc_uint8.shape}")

        output_path = self.output_dir / f"mesh_{uuid.uuid4().hex[:8]}.glb"
        glb.export(str(output_path))

        # Cleanup
        del output
        del image_np
        del mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(f"Mesh saved: {output_path}")
        logger.info(f"VRAM after cleanup: {self.get_vram_usage()}")
        return str(output_path)

    @staticmethod
    def get_output_format() -> str:
        """Get the configured output format (splat or mesh)"""
        return os.getenv("OUTPUT_FORMAT", "splat").lower()

    def generate_3d(self, image_path: str, mask_path: str, seed: int = 42) -> str:
        """
        Generate 3D output (splat or mesh) based on OUTPUT_FORMAT setting.

        Returns path to generated file (.ply for splat, .glb for mesh).
        """
        output_format = self.get_output_format()
        logger.info(f"Output format: {output_format}")

        if output_format == "mesh":
            return self.generate_mesh(image_path, mask_path, seed)
        else:
            return self.generate_splat(image_path, mask_path, seed)

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
