"""
FAISS Index Service for Efficient Similarity Search

Provides fast approximate nearest neighbor search using FAISS indices.
Supports both CPU and GPU backends.

Usage:
    from services.faiss_service import get_faiss_service

    faiss_svc = get_faiss_service()
    faiss_svc.build_index("my_library_id")  # Build from database
    results = faiss_svc.search(query_embedding, k=10)
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# FAISS index directory
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "media/faiss_indices"))


class FAISSService:
    """
    FAISS-based vector similarity search service.

    Maintains separate indices for text and image embeddings.
    Supports incremental updates and persistence.
    """

    def __init__(self, embedding_dim: int = 768):
        """
        Initialize the FAISS service.

        Args:
            embedding_dim: Dimension of embedding vectors (768 for OpenCLIP, 512 for FLAIR)
        """
        self.embedding_dim = embedding_dim
        self.faiss = None
        self.use_gpu = False

        # Index storage: library_id -> (index, id_mapping)
        self._text_indices: Dict[str, Tuple[Any, List[str]]] = {}
        self._image_indices: Dict[str, Tuple[Any, List[str]]] = {}

        # Combined indices for cross-library search
        self._combined_text_index = None
        self._combined_text_ids: List[str] = []
        self._combined_image_index = None
        self._combined_image_ids: List[str] = []

        self._load_faiss()

    def _load_faiss(self):
        """Load FAISS library, preferring GPU if available"""
        try:
            import faiss
            self.faiss = faiss

            # Check for GPU support
            try:
                num_gpus = faiss.get_num_gpus()
                if num_gpus > 0:
                    self.use_gpu = True
                    logger.info(f"FAISS loaded with GPU support ({num_gpus} GPUs)")
                else:
                    logger.info("FAISS loaded (CPU only)")
            except Exception:
                # faiss-cpu doesn't have get_num_gpus
                logger.info("FAISS loaded (CPU version)")

        except ImportError as e:
            logger.warning(
                f"FAISS not installed: {e}. Install with: "
                "pip install faiss-gpu (CUDA) or pip install faiss-cpu"
            )
            self.faiss = None  # Allow fallback to brute-force

    def is_available(self) -> bool:
        """Check if FAISS is available"""
        return self.faiss is not None

    def _create_index(self, dim: int) -> Any:
        """Create a new FAISS index"""
        if not self.is_available():
            raise RuntimeError("FAISS not available")

        # Use IndexFlatIP for cosine similarity (vectors should be normalized)
        index = self.faiss.IndexFlatIP(dim)

        if self.use_gpu:
            # Move to GPU for faster search
            res = self.faiss.StandardGpuResources()
            index = self.faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def build_index(self, library_id: str = None, embedding_field: str = "text_embedding"):
        """
        Build FAISS index from database items.

        Args:
            library_id: Build index for specific library, or None for all libraries
            embedding_field: "text_embedding" or "image_embedding"
        """
        if not self.is_available():
            raise RuntimeError("FAISS not available")

        from services.database import DatabaseService

        db = DatabaseService()
        db.connect()

        # Query items with embeddings
        query = {
            "status": "ready",
            embedding_field: {"$exists": True, "$ne": None}
        }
        if library_id:
            query["library_id"] = library_id

        items = list(db.library_items.find(query))
        db.close()

        if not items:
            logger.warning(f"No items found with {embedding_field}")
            return

        # Extract embeddings and IDs
        embeddings = []
        item_ids = []

        for item in items:
            emb = item.get(embedding_field)
            if emb and len(emb) > 0:
                embeddings.append(emb)
                item_ids.append(str(item["_id"]))

        if not embeddings:
            logger.warning("No valid embeddings found")
            return

        # Update dimension if needed
        actual_dim = len(embeddings[0])
        if actual_dim != self.embedding_dim:
            logger.info(f"Updating embedding dimension: {self.embedding_dim} -> {actual_dim}")
            self.embedding_dim = actual_dim

        # Create and populate index
        index = self._create_index(actual_dim)

        # Convert to numpy array and normalize
        emb_array = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        emb_array = emb_array / np.maximum(norms, 1e-8)

        index.add(emb_array)

        # Store index
        if embedding_field == "text_embedding":
            if library_id:
                self._text_indices[library_id] = (index, item_ids)
            else:
                self._combined_text_index = index
                self._combined_text_ids = item_ids
        else:
            if library_id:
                self._image_indices[library_id] = (index, item_ids)
            else:
                self._combined_image_index = index
                self._combined_image_ids = item_ids

        logger.info(f"Built {embedding_field} index: {len(item_ids)} items")

        # Save to disk
        self._save_index(library_id, embedding_field)

    def _get_index_path(self, library_id: str, embedding_field: str) -> Path:
        """Get path for index file"""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        prefix = "text" if embedding_field == "text_embedding" else "image"
        lib_suffix = library_id if library_id else "combined"
        return INDEX_DIR / f"{prefix}_{lib_suffix}.faiss"

    def _save_index(self, library_id: str, embedding_field: str):
        """Save index to disk"""
        if not self.is_available():
            return

        index_path = self._get_index_path(library_id, embedding_field)

        if embedding_field == "text_embedding":
            if library_id and library_id in self._text_indices:
                index, ids = self._text_indices[library_id]
            elif self._combined_text_index is not None:
                index, ids = self._combined_text_index, self._combined_text_ids
            else:
                return
        else:
            if library_id and library_id in self._image_indices:
                index, ids = self._image_indices[library_id]
            elif self._combined_image_index is not None:
                index, ids = self._combined_image_index, self._combined_image_ids
            else:
                return

        # Convert GPU index to CPU for saving
        if self.use_gpu:
            index = self.faiss.index_gpu_to_cpu(index)

        self.faiss.write_index(index, str(index_path))

        # Save ID mapping
        ids_path = index_path.with_suffix(".ids")
        with open(ids_path, "w") as f:
            f.write("\n".join(ids))

        logger.info(f"Saved index to {index_path}")

    def _load_index(self, library_id: str, embedding_field: str) -> bool:
        """Load index from disk if available"""
        if not self.is_available():
            return False

        index_path = self._get_index_path(library_id, embedding_field)
        ids_path = index_path.with_suffix(".ids")

        if not index_path.exists() or not ids_path.exists():
            return False

        try:
            index = self.faiss.read_index(str(index_path))

            if self.use_gpu:
                res = self.faiss.StandardGpuResources()
                index = self.faiss.index_cpu_to_gpu(res, 0, index)

            with open(ids_path) as f:
                ids = f.read().strip().split("\n")

            if embedding_field == "text_embedding":
                if library_id:
                    self._text_indices[library_id] = (index, ids)
                else:
                    self._combined_text_index = index
                    self._combined_text_ids = ids
            else:
                if library_id:
                    self._image_indices[library_id] = (index, ids)
                else:
                    self._combined_image_index = index
                    self._combined_image_ids = ids

            logger.info(f"Loaded index from {index_path}: {len(ids)} items")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        library_id: str = None,
        embedding_field: str = "text_embedding"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items using FAISS.

        Args:
            query_embedding: Query vector (will be normalized)
            k: Number of results to return
            library_id: Search specific library, or None for all
            embedding_field: "text_embedding" or "image_embedding"

        Returns:
            List of dicts with "id" and "similarity" keys

        Raises:
            RuntimeError: If FAISS is not available or no index could be built
        """
        if not self.is_available():
            raise RuntimeError(
                "FAISS not available. Install with: pip install faiss-gpu or faiss-cpu"
            )

        # Get appropriate index
        index = None
        ids = []

        if embedding_field == "text_embedding":
            if library_id and library_id in self._text_indices:
                index, ids = self._text_indices[library_id]
            elif self._combined_text_index is not None:
                index, ids = self._combined_text_index, self._combined_text_ids
            else:
                # Try loading from disk
                if self._load_index(library_id, embedding_field):
                    return self.search(query_embedding, k, library_id, embedding_field)
                # Build index on demand
                self.build_index(library_id, embedding_field)
                # Check if build succeeded
                if self._combined_text_index is not None:
                    return self.search(query_embedding, k, library_id, embedding_field)
                else:
                    raise RuntimeError(
                        f"No items with {embedding_field} found in database. "
                        "Ensure library items have embeddings."
                    )
        else:
            if library_id and library_id in self._image_indices:
                index, ids = self._image_indices[library_id]
            elif self._combined_image_index is not None:
                index, ids = self._combined_image_index, self._combined_image_ids
            else:
                if self._load_index(library_id, embedding_field):
                    return self.search(query_embedding, k, library_id, embedding_field)
                self.build_index(library_id, embedding_field)
                if self._combined_image_index is not None:
                    return self.search(query_embedding, k, library_id, embedding_field)
                else:
                    raise RuntimeError(
                        f"No items with {embedding_field} found in database. "
                        "Ensure library items have embeddings."
                    )

        if index is None or len(ids) == 0:
            raise RuntimeError(f"No index available for {embedding_field}")

        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Search
        k = min(k, len(ids))
        distances, indices = index.search(query, k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(ids):
                results.append({
                    "id": ids[idx],
                    "similarity": float(dist)  # Inner product = cosine sim for normalized vectors
                })

        return results

    def add_item(
        self,
        item_id: str,
        text_embedding: List[float] = None,
        image_embedding: List[float] = None,
        library_id: str = None
    ):
        """
        Add a single item to the index (for incremental updates).

        Args:
            item_id: Unique item identifier
            text_embedding: Text embedding vector
            image_embedding: Image embedding vector
            library_id: Library the item belongs to
        """
        if text_embedding:
            self._add_to_index(item_id, text_embedding, library_id, "text_embedding")
        if image_embedding:
            self._add_to_index(item_id, image_embedding, library_id, "image_embedding")

    def _add_to_index(
        self,
        item_id: str,
        embedding: List[float],
        library_id: str,
        embedding_field: str
    ):
        """Add embedding to appropriate index"""
        # Get or create index
        if embedding_field == "text_embedding":
            indices_dict = self._text_indices
            combined_index = self._combined_text_index
            combined_ids = self._combined_text_ids
        else:
            indices_dict = self._image_indices
            combined_index = self._combined_image_index
            combined_ids = self._combined_image_ids

        # Normalize embedding
        emb = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Add to library-specific index
        if library_id:
            if library_id not in indices_dict:
                indices_dict[library_id] = (self._create_index(len(embedding)), [])
            index, ids = indices_dict[library_id]
            index.add(emb)
            ids.append(item_id)

        # Add to combined index
        if combined_index is not None:
            combined_index.add(emb)
            combined_ids.append(item_id)

    def rebuild_all(self):
        """Rebuild all indices from database"""
        logger.info("Rebuilding all FAISS indices...")

        # Clear existing indices
        self._text_indices.clear()
        self._image_indices.clear()
        self._combined_text_index = None
        self._combined_text_ids = []
        self._combined_image_index = None
        self._combined_image_ids = []

        # Build combined indices
        self.build_index(None, "text_embedding")
        self.build_index(None, "image_embedding")

        logger.info("FAISS indices rebuilt")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            "embedding_dim": self.embedding_dim,
            "use_gpu": self.use_gpu,
            "text_indices": {},
            "image_indices": {}
        }

        for lib_id, (index, ids) in self._text_indices.items():
            stats["text_indices"][lib_id] = len(ids)

        for lib_id, (index, ids) in self._image_indices.items():
            stats["image_indices"][lib_id] = len(ids)

        if self._combined_text_index is not None:
            stats["combined_text_count"] = len(self._combined_text_ids)
        if self._combined_image_index is not None:
            stats["combined_image_count"] = len(self._combined_image_ids)

        return stats


# Singleton instance
_faiss_service: Optional[FAISSService] = None


def get_faiss_service(embedding_dim: int = None) -> FAISSService:
    """
    Get the singleton FAISS service instance.

    Args:
        embedding_dim: Override embedding dimension on first call
    """
    global _faiss_service

    if _faiss_service is None:
        # Determine dimension from embedding service
        if embedding_dim is None:
            try:
                from services.embedding_service import get_embedding_service
                emb_svc = get_embedding_service()
                embedding_dim = emb_svc.embedding_dim
            except Exception:
                embedding_dim = 768  # Default to OpenCLIP

        _faiss_service = FAISSService(embedding_dim=embedding_dim)

    return _faiss_service


def reset_faiss_service():
    """Reset the singleton to rebuild indices"""
    global _faiss_service
    _faiss_service = None
