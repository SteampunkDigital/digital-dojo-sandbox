"""
Embedding Service for 3D Asset Library Search

Supports multiple embedding backends:
- OpenCLIP ViT-L-14 (default): General-purpose CLIP embeddings (768 dims)
- FLAIR: Apple's fine-grained language-image alignment (768 dims)

Configure via environment variable:
    EMBEDDING_MODEL=openclip  (default)
    EMBEDDING_MODEL=flair
"""

import os
import logging
import torch
import gc
from typing import List, Optional
from abc import ABC, abstractmethod
from PIL import Image

logger = logging.getLogger(__name__)


class BaseEmbeddingBackend(ABC):
    """Abstract base class for embedding backends"""

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = 768  # Both OpenCLIP and FLAIR use 768

    @abstractmethod
    def load(self):
        """Load the model"""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model to free memory"""
        pass

    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """Encode text to embedding"""
        pass

    @abstractmethod
    def encode_image(self, image_path: str) -> List[float]:
        """Encode image to embedding"""
        pass

    def is_loaded(self) -> bool:
        return self.model is not None


class OpenCLIPBackend(BaseEmbeddingBackend):
    """OpenCLIP ViT-L-14 embedding backend"""

    def __init__(self):
        super().__init__()
        self.preprocess = None
        self.tokenizer = None
        self.model_name = os.getenv("CLIP_MODEL", "ViT-L-14")
        self.pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s32b_b82k")

    def load(self):
        if self.is_loaded():
            return

        logger.info(f"Loading OpenCLIP {self.model_name} ({self.pretrained})...")

        try:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained
            )
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            logger.info(f"OpenCLIP loaded on {self.device}")
            self._log_vram()

        except ImportError:
            raise RuntimeError(
                "open_clip_torch not installed. "
                "Install with: pip install open_clip_torch"
            )

    def unload(self):
        if not self.is_loaded():
            return

        logger.info("Unloading OpenCLIP...")

        del self.model
        del self.preprocess
        del self.tokenizer

        self.model = None
        self.preprocess = None
        self.tokenizer = None

        self._cleanup_gpu()
        logger.info("OpenCLIP unloaded")

    def encode_text(self, text: str) -> List[float]:
        self.load()

        tokens = self.tokenizer([text]).to(self.device)

        # Keep text encoding in fp32 to avoid intermittent fp16/cuBLAS failures
        # seen with some Torch/OpenCLIP/CUDA combinations.
        # This is really quick.
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features[0].cpu().numpy().tolist()

    def encode_image(self, image_path: str) -> List[float]:
        self.load()

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast(self.device):
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features[0].cpu().numpy().tolist()

    def _log_vram(self):
        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM usage: {vram:.2f} GB")

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class FLAIRBackend(BaseEmbeddingBackend):
    """
    FLAIR embedding backend for fine-grained image retrieval.

    FLAIR (Fine-grained Language-Aligned Image Representation) from ExplainableML
    uses attention pooling for better fine-grained matching.

    Repo: https://github.com/ExplainableML/flair
    Model: xiaorui638/flair on HuggingFace

    Based on ViT-B-16, produces 512-dim embeddings.
    """

    def __init__(self):
        super().__init__()
        self.processor = None
        self.tokenizer = None
        self.flair_repo = os.getenv("FLAIR_REPO_PATH")
        self.model_name = os.getenv("FLAIR_MODEL", "xiaorui638/flair")
        self.embedding_dim = 512  # ViT-B-16 based

    def load(self):
        if self.is_loaded():
            return

        logger.info(f"Loading FLAIR ({self.model_name})...")

        # Try to load FLAIR from local repo if available
        if self.flair_repo:
            self._load_from_repo()
        else:
            self._load_from_huggingface()

    def _load_from_repo(self):
        """Load FLAIR from local repository clone"""
        import sys
        flair_src = os.path.join(self.flair_repo, "src")
        if flair_src not in sys.path:
            sys.path.insert(0, flair_src)

        try:
            from model import FLAIR
            from transformers import CLIPProcessor

            # Load model from HuggingFace cache or download
            self.model = FLAIR.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Use CLIP processor for preprocessing
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            logger.info(f"FLAIR loaded from repo on {self.device}")
            self._log_vram()

        except Exception as e:
            raise RuntimeError(f"Failed to load FLAIR from repo: {e}")

    def _load_from_huggingface(self):
        """Load FLAIR using transformers with trust_remote_code"""
        try:
            from transformers import AutoModel, AutoProcessor, CLIPProcessor

            # Try loading with trust_remote_code
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except Exception:
                # Fallback: FLAIR is based on OpenCLIP ViT-B-16
                # Use that as a compatible alternative
                logger.warning("FLAIR model not directly loadable, using OpenCLIP ViT-B-16")
                import open_clip
                self.model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-16",
                    pretrained="laion2b_s34b_b88k"
                )
                self.processor = preprocess
                self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
                self.model.to(self.device)
                self.model.eval()
                self._using_openclip_fallback = True
                logger.info(f"OpenCLIP ViT-B-16 loaded as FLAIR fallback on {self.device}")
                self._log_vram()
                return

            self.model.to(self.device)
            self.model.eval()

            # Load processor
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            except Exception:
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            self._using_openclip_fallback = False
            logger.info(f"FLAIR loaded on {self.device}")
            self._log_vram()

        except ImportError:
            raise RuntimeError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FLAIR: {e}")

    def unload(self):
        if not self.is_loaded():
            return

        logger.info("Unloading FLAIR...")

        del self.model
        if self.processor:
            del self.processor
        if self.tokenizer:
            del self.tokenizer

        self.model = None
        self.processor = None
        self.tokenizer = None

        self._cleanup_gpu()
        logger.info("FLAIR unloaded")

    def encode_text(self, text: str) -> List[float]:
        self.load()

        # OpenCLIP fallback path
        if getattr(self, '_using_openclip_fallback', False):
            tokens = self.tokenizer([text]).to(self.device)
            # Match the OpenCLIP backend: text encoding is more stable in fp32.
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features[0].cpu().numpy().tolist()

        # Standard FLAIR/transformers path
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'get_text_features'):
                outputs = self.model.get_text_features(**inputs)
            else:
                outputs = self.model.encode_text(inputs['input_ids'])
            text_features = outputs / outputs.norm(dim=-1, keepdim=True)

        return text_features[0].cpu().numpy().tolist()

    def encode_image(self, image_path: str) -> List[float]:
        self.load()

        image = Image.open(image_path).convert("RGB")

        # OpenCLIP fallback path
        if getattr(self, '_using_openclip_fallback', False):
            image_tensor = self.processor(image).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.amp.autocast(self.device):
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features[0].cpu().numpy().tolist()

        # Standard FLAIR/transformers path
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'get_image_features'):
                outputs = self.model.get_image_features(**inputs)
            else:
                outputs = self.model.encode_image(inputs['pixel_values'])
            image_features = outputs / outputs.norm(dim=-1, keepdim=True)

        return image_features[0].cpu().numpy().tolist()

    def _log_vram(self):
        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM usage: {vram:.2f} GB")

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class EmbeddingService:
    """
    Unified embedding service supporting multiple backends.

    Produces 768-dimensional embeddings suitable for cosine similarity search.
    Model is loaded on-demand and can be unloaded to free GPU memory.

    Configure backend via EMBEDDING_MODEL env var:
        - "openclip" (default): OpenCLIP ViT-L-14
        - "flair": Apple FLAIR
    """

    def __init__(self, backend: str = None):
        """
        Initialize the embedding service.

        Args:
            backend: Override backend selection ("openclip" or "flair").
                     Defaults to EMBEDDING_MODEL env var or "openclip".
        """
        backend = backend or os.getenv("EMBEDDING_MODEL", "openclip").lower()

        if backend == "flair":
            self._backend = FLAIRBackend()
            logger.info("EmbeddingService using FLAIR backend (512 dims)")
        else:
            self._backend = OpenCLIPBackend()
            logger.info("EmbeddingService using OpenCLIP backend")

        self.backend_name = backend
        self.embedding_dim = self._backend.embedding_dim

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded"""
        return self._backend.is_loaded()

    def load(self):
        """Load the embedding model"""
        self._backend.load()

    def unload(self):
        """Unload the model to free GPU memory"""
        self._backend.unload()

    def encode_text(self, text: str) -> List[float]:
        """
        Encode text to a normalized embedding vector.

        Args:
            text: Text string to encode

        Returns:
            List of floats (768 dimensions)
        """
        return self._backend.encode_text(text)

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts to normalized embedding vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        # Process one at a time (backends can override for batching)
        return [self._backend.encode_text(t) for t in texts]

    def encode_image(self, image_path: str) -> List[float]:
        """
        Encode an image to a normalized embedding vector.

        Args:
            image_path: Path to the image file

        Returns:
            List of floats (768 dimensions)
        """
        return self._backend.encode_image(image_path)

    def encode_images(self, image_paths: List[str]) -> List[List[float]]:
        """
        Encode multiple images to normalized embedding vectors.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of embedding vectors
        """
        return [self._backend.encode_image(p) for p in image_paths]

    def encode_image_with_text(
        self,
        image_path: str,
        text: str,
        image_weight: float = 0.5
    ) -> List[float]:
        """
        Encode an image combined with text description.

        Creates a weighted average of image and text embeddings,
        useful for multi-modal search where both visual and semantic
        information should be captured.

        Args:
            image_path: Path to the image file
            text: Text description of the image
            image_weight: Weight for image embedding (0-1). Text weight = 1 - image_weight.
                          Default 0.5 gives equal weight to both.

        Returns:
            Normalized combined embedding vector
        """
        import numpy as np

        # Get both embeddings
        image_emb = np.array(self._backend.encode_image(image_path))
        text_emb = np.array(self._backend.encode_text(text))

        # Weighted average
        text_weight = 1.0 - image_weight
        combined = (image_weight * image_emb) + (text_weight * text_emb)

        # Re-normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined.tolist()

    def encode_query(
        self,
        text: str = None,
        image_path: str = None,
        image_weight: float = 0.5
    ) -> List[float]:
        """
        Encode a search query that can include text, image, or both.

        Args:
            text: Text query (optional)
            image_path: Path to query image (optional)
            image_weight: When both provided, weight for image (0-1)

        Returns:
            Normalized query embedding vector
        """
        import numpy as np

        if text and image_path:
            # Combined query
            return self.encode_image_with_text(image_path, text, image_weight)
        elif image_path:
            return self._backend.encode_image(image_path)
        elif text:
            return self._backend.encode_text(text)
        else:
            raise ValueError("At least one of text or image_path must be provided")

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1 for normalized vectors)
        """
        import numpy as np
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        return float(np.dot(e1, e2))


# Singleton instance for lazy loading
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(backend: str = None) -> EmbeddingService:
    """
    Get the singleton embedding service instance.

    Args:
        backend: Override backend on first call ("openclip" or "flair")
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(backend=backend)
    return _embedding_service


def reset_embedding_service():
    """Reset the singleton to allow backend change"""
    global _embedding_service
    if _embedding_service is not None:
        _embedding_service.unload()
        _embedding_service = None
