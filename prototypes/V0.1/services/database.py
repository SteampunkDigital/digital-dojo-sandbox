"""
MongoDB Database Service

Handles connection to MongoDB and provides collections for:
- Scenes: Stored scene graphs
- Jobs: Generation job queue
- Assets: Generated asset metadata
- Libraries: Object library metadata
- Library Items: Individual items with embeddings for vector search
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class DatabaseService:
    """MongoDB database service"""

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._connected = False

    def connect(self, uri: Optional[str] = None, database: Optional[str] = None):
        """
        Connect to MongoDB.
        Uses environment variables if parameters not provided.
        """
        uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        database = database or os.getenv("MONGODB_DATABASE", "digital_dojo")

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[database]
            self._connected = True
            logger.info(f"Connected to MongoDB: {database}")

            # Create indexes
            self._ensure_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            raise

    def _ensure_indexes(self):
        """Create necessary indexes"""
        if self.db is None:
            return

        # Scenes collection
        self.db.scenes.create_index("created_at")
        self.db.scenes.create_index("updated_at")

        # Jobs collection
        self.db.jobs.create_index("status")
        self.db.jobs.create_index("created_at")
        self.db.jobs.create_index([("status", 1), ("created_at", 1)])

        # Libraries collection
        self.db.libraries.create_index("created_at")
        self.db.libraries.create_index("status")

        # Library items collection
        self.db.library_items.create_index("library_id")
        self.db.library_items.create_index("status")
        self.db.library_items.create_index([("status", 1), ("created_at", 1)])
        self.db.library_items.create_index("description")
        self.db.library_items.create_index("variant_group_id")
        self.db.library_items.create_index([("library_id", 1), ("variant_group_id", 1)])

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def scenes(self) -> Collection:
        """Scenes collection"""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db.scenes

    @property
    def jobs(self) -> Collection:
        """Jobs collection"""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db.jobs

    @property
    def assets(self) -> Collection:
        """Assets collection"""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db.assets

    @property
    def libraries(self) -> Collection:
        """Libraries collection"""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db.libraries

    @property
    def library_items(self) -> Collection:
        """Library items collection"""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db.library_items

    @property
    def gpu_lock(self) -> Collection:
        """GPU lock collection"""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db.gpu_lock

    # Migration

    def migrate_to_trellis2(self) -> Dict[str, int]:
        """
        Migrate existing data from 3-stage (SAM3D) to 2-stage (Trellis2) pipeline.
        Updates: needs_mask/needs_splat → needs_3d, removes mask_path field.
        """
        results = {}

        # Scene jobs: needs_mask or needs_splat → needs_3d
        r = self.jobs.update_many(
            {"status": {"$in": ["needs_mask", "needs_splat"]}},
            {"$set": {"status": "needs_3d"}, "$unset": {"mask_path": ""}}
        )
        results["jobs_migrated"] = r.modified_count

        # Library items: needs_mask → needs_3d
        r = self.library_items.update_many(
            {"status": "needs_mask"},
            {"$set": {"status": "needs_3d"}}
        )
        results["items_needs_mask_migrated"] = r.modified_count

        # Library items: approved → needs_3d (they had images, skip to 3D)
        r = self.library_items.update_many(
            {"status": "approved"},
            {"$set": {"status": "needs_3d"}}
        )
        results["items_approved_migrated"] = r.modified_count

        # Clean up mask_path field from all library items
        r = self.library_items.update_many(
            {"mask_path": {"$exists": True}},
            {"$unset": {"mask_path": ""}}
        )
        results["mask_path_removed"] = r.modified_count

        logger.info(f"Migration to Trellis2 complete: {results}")
        return results

    # Scene operations

    def save_scene(self, scene_data: Dict[str, Any]) -> str:
        """Save or update a scene"""
        scene_data["updated_at"] = datetime.utcnow()
        if "created_at" not in scene_data:
            scene_data["created_at"] = datetime.utcnow()

        result = self.scenes.replace_one(
            {"_id": scene_data["_id"]},
            scene_data,
            upsert=True
        )
        return scene_data["_id"]

    def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Get a scene by ID"""
        return self.scenes.find_one({"_id": scene_id})

    def list_scenes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent scenes"""
        return list(self.scenes.find().sort("updated_at", -1).limit(limit))

    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene"""
        result = self.scenes.delete_one({"_id": scene_id})
        return result.deleted_count > 0

    # Job operations

    def create_job(self, scene_id: str, node_id: str, prompt: str,
                   job_type: str = "splat") -> str:
        """Create a new generation job"""
        import uuid
        job = {
            "_id": uuid.uuid4().hex,
            "scene_id": scene_id,
            "node_id": node_id,
            "prompt": prompt,
            "type": job_type,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "output_path": None
        }
        self.jobs.insert_one(job)
        return job["_id"]

    def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending jobs, oldest first"""
        return list(
            self.jobs.find({"status": "pending"})
            .sort("created_at", 1)
            .limit(limit)
        )

    def get_jobs_by_stage(self, stage: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get jobs at a specific processing stage.

        Stages:
          - "pending" = needs image generation (SD3.5)
          - "needs_3d" = image done, needs 3D generation (Trellis2)
          - "processing" = currently being worked on
          - "completed" = finished
          - "failed" = error occurred
        """
        return list(
            self.jobs.find({"status": stage})
            .sort("created_at", 1)
            .limit(limit)
        )

    def count_jobs_by_stage(self, stage: str) -> int:
        """Count jobs at a specific stage"""
        return self.jobs.count_documents({"status": stage})

    def update_job_status(self, job_id: str, status: str,
                          output_path: Optional[str] = None,
                          error: Optional[str] = None):
        """Update job status"""
        update = {"status": status}

        if status == "processing":
            update["started_at"] = datetime.utcnow()
        elif status in ("completed", "failed"):
            update["completed_at"] = datetime.utcnow()

        if output_path:
            update["output_path"] = output_path
        if error:
            update["error"] = error

        self.jobs.update_one({"_id": job_id}, {"$set": update})

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID"""
        return self.jobs.find_one({"_id": job_id})

    def reset_failed_jobs(self, reset_to_stage: str = "needs_3d") -> int:
        """
        Reset all failed jobs to retry them.

        Args:
            reset_to_stage: Stage to reset to. Options:
              - "pending" = restart from image generation
              - "needs_3d" = restart from 3D generation (keep image)

        Returns:
            Number of jobs reset
        """
        # Find all failed jobs
        failed_jobs = list(self.jobs.find({"status": "failed"}))

        reset_count = 0
        for job in failed_jobs:
            if reset_to_stage == "pending":
                new_status = "pending"
            else:  # needs_3d
                if job.get("image_path"):
                    new_status = "needs_3d"
                else:
                    new_status = "pending"

            self.jobs.update_one(
                {"_id": job["_id"]},
                {"$set": {
                    "status": new_status,
                    "error": None,
                    "completed_at": None
                }}
            )
            reset_count += 1
            logger.info(f"Reset job {job['_id']} from failed to {new_status}")

        return reset_count

    def get_failed_jobs(self) -> List[Dict[str, Any]]:
        """Get all failed jobs"""
        return list(self.jobs.find({"status": "failed"}))

    def reset_all_jobs(self, reset_to_stage: str = "needs_3d") -> int:
        """
        Reset ALL jobs (including completed) to retry them.

        Args:
            reset_to_stage: Stage to reset to. Options:
              - "pending" = restart from image generation
              - "needs_3d" = restart from 3D generation (keep image)

        Returns:
            Number of jobs reset
        """
        all_jobs = list(self.jobs.find({
            "status": {"$nin": ["pending", "needs_3d"]}
        }))

        reset_count = 0
        for job in all_jobs:
            if reset_to_stage == "pending":
                new_status = "pending"
            else:  # needs_3d
                if job.get("image_path"):
                    new_status = "needs_3d"
                else:
                    new_status = "pending"

            self.jobs.update_one(
                {"_id": job["_id"]},
                {"$set": {
                    "status": new_status,
                    "error": None,
                    "completed_at": None,
                    "output_path": None
                }}
            )
            reset_count += 1
            logger.info(f"Reset job {job['_id']} to {new_status}")

        return reset_count

    def clear_all_jobs(self) -> int:
        """Delete all jobs (use with caution)"""
        result = self.jobs.delete_many({})
        return result.deleted_count

    # Asset operations

    def register_asset(self, asset_type: str, path: str,
                       scene_id: str, node_id: str,
                       metadata: Optional[Dict] = None) -> str:
        """Register a generated asset"""
        import uuid
        asset = {
            "_id": uuid.uuid4().hex,
            "type": asset_type,
            "path": path,
            "scene_id": scene_id,
            "node_id": node_id,
            "metadata": metadata or {},
            "created_at": datetime.utcnow()
        }
        self.assets.insert_one(asset)
        return asset["_id"]

    def get_assets_for_scene(self, scene_id: str) -> List[Dict[str, Any]]:
        """Get all assets for a scene"""
        return list(self.assets.find({"scene_id": scene_id}))

    # Library operations

    def create_library(self, name: str, source_text: str,
                       variants_per_item: int = 3) -> str:
        """
        Create a new object library from a text list.

        Args:
            name: Library name
            source_text: Newline-separated object descriptions
            variants_per_item: Number of SD3.5 variants per object

        Returns:
            Library ID
        """
        import uuid

        # Parse descriptions (one per line, skip empty)
        descriptions = [
            line.strip()
            for line in source_text.strip().split('\n')
            if line.strip()
        ]

        library = {
            "_id": uuid.uuid4().hex,
            "name": name,
            "source_text": source_text,
            "item_count": len(descriptions),
            "variants_per_item": variants_per_item,
            "total_variants": len(descriptions) * variants_per_item,
            "status": "generating",
            "created_at": datetime.utcnow(),
            "completed_at": None
        }
        self.libraries.insert_one(library)

        # Create library items for each description + variant
        import hashlib
        for desc in descriptions:
            # Generate variant group ID from description (same for all variants)
            variant_group_id = hashlib.md5(desc.encode()).hexdigest()[:12]

            for variant_idx in range(variants_per_item):
                # Use different seeds for each variant
                seed = hash(f"{desc}_{variant_idx}") % (2**32)
                item = {
                    "_id": uuid.uuid4().hex,
                    "library_id": library["_id"],
                    "description": desc,
                    "variant_group_id": variant_group_id,
                    "variant_index": variant_idx,
                    "seed": seed,
                    "image_path": None,
                    "asset_path": None,
                    "asset_type": None,
                    "text_embedding": None,
                    "image_embedding": None,
                    "status": "pending",
                    "error": None,
                    "created_at": datetime.utcnow()
                }
                self.library_items.insert_one(item)

        logger.info(f"Created library '{name}' with {len(descriptions)} items "
                   f"x {variants_per_item} variants = {len(descriptions) * variants_per_item} total")

        return library["_id"]

    def add_uploaded_item(self, library_id: str, description: str,
                          original_image_path: str) -> str:
        """
        Create a library item from an uploaded image (needs interactive masking).

        Returns:
            Item ID
        """
        import uuid
        import hashlib

        variant_group_id = hashlib.md5(description.encode()).hexdigest()[:12]

        item = {
            "_id": uuid.uuid4().hex,
            "library_id": library_id,
            "description": description,
            "variant_group_id": variant_group_id,
            "variant_index": 0,
            "seed": 0,
            "source": "upload",
            "original_image_path": original_image_path,
            "image_path": original_image_path,  # Show original in review UI
            "asset_path": None,
            "asset_type": None,
            "text_embedding": None,
            "image_embedding": None,
            "status": "needs_masking",
            "error": None,
            "created_at": datetime.utcnow()
        }
        self.library_items.insert_one(item)

        # Update library counts
        self.libraries.update_one(
            {"_id": library_id},
            {"$inc": {"total_variants": 1, "item_count": 1}}
        )

        logger.info(f"Added uploaded item to library {library_id}: {description[:50]}")
        return item["_id"]

    def add_capture_item(self, description: str, original_image_path: str) -> str:
        """
        Create a library item from a phone capture (no library_id needed).

        Returns:
            Item ID
        """
        import uuid

        item = {
            "_id": uuid.uuid4().hex,
            "library_id": None,
            "description": description,
            "variant_group_id": None,
            "variant_index": 0,
            "seed": 0,
            "source": "capture",
            "original_image_path": original_image_path,
            "image_path": original_image_path,
            "asset_path": None,
            "asset_type": None,
            "text_embedding": None,
            "image_embedding": None,
            "status": "review",
            "error": None,
            "created_at": datetime.utcnow()
        }
        self.library_items.insert_one(item)

        logger.info(f"Added capture item: {description[:50]}")
        return item["_id"]

    def get_library(self, library_id: str) -> Optional[Dict[str, Any]]:
        """Get a library by ID"""
        return self.libraries.find_one({"_id": library_id})

    def list_libraries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all libraries"""
        return list(self.libraries.find().sort("created_at", -1).limit(limit))

    def get_library_items(self, library_id: str,
                          status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get items for a library, optionally filtered by status"""
        query = {"library_id": library_id}
        if status:
            query["status"] = status
        return list(self.library_items.find(query).sort("created_at", 1))

    def get_pending_library_items(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pending library items for processing"""
        return list(
            self.library_items.find({"status": "pending"})
            .sort("created_at", 1)
            .limit(limit)
        )

    def get_library_items_by_stage(self, stage: str,
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get library items at a specific processing stage"""
        return list(
            self.library_items.find({"status": stage})
            .sort("created_at", 1)
            .limit(limit)
        )

    def count_library_items_by_stage(self, stage: str) -> int:
        """Count library items at a specific stage"""
        return self.library_items.count_documents({"status": stage})

    def update_library_item(self, item_id: str, **updates):
        """Update a library item"""
        self.library_items.update_one(
            {"_id": item_id},
            {"$set": updates}
        )

    # Review workflow operations

    def get_items_for_review(self, library_id: str = None,
                              status: str = "needs_review",
                              limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get items for review, grouped by variant_group_id.

        Returns list of groups, each containing variants of the same description.

        Special case: status="processing" returns all items in the approval pipeline
        (approved, needs_mask, needs_3d, needs_embedding).
        """
        # Handle "processing" as a special multi-status query
        if status == "processing":
            query = {"status": {"$in": ["approved", "needs_3d", "needs_embedding"]}}
        else:
            query = {"status": status}

        if library_id:
            query["library_id"] = library_id

        items = list(
            self.library_items.find(query)
            .sort([("variant_group_id", 1), ("variant_index", 1)])
            .limit(limit)
        )

        # Group by variant_group_id
        groups = {}
        for item in items:
            group_id = item.get("variant_group_id", item["_id"])
            if group_id not in groups:
                groups[group_id] = {
                    "variant_group_id": group_id,
                    "description": item["description"],
                    "library_id": item["library_id"],
                    "variants": []
                }
            groups[group_id]["variants"].append(item)

        return list(groups.values())

    def approve_item(self, item_id: str, reviewer: str = None) -> bool:
        """Approve an item - moves from needs_review to approved"""
        result = self.library_items.update_one(
            {"_id": item_id, "status": "needs_review"},
            {"$set": {
                "status": "approved",
                "reviewed_at": datetime.utcnow(),
                "reviewed_by": reviewer
            }}
        )
        return result.modified_count > 0

    def reject_item(self, item_id: str, reviewer: str = None,
                    notes: str = None) -> bool:
        """Reject an item - moves to rejected status"""
        update = {
            "status": "rejected",
            "reviewed_at": datetime.utcnow(),
            "reviewed_by": reviewer
        }
        if notes:
            update["review_notes"] = notes

        result = self.library_items.update_one(
            {"_id": item_id},
            {"$set": update}
        )
        return result.modified_count > 0

    def reset_item_for_regeneration(self, item_id: str, new_seed: bool = True) -> bool:
        """
        Reset a rejected or failed item back to pending for regeneration.
        Clears image/mask/asset paths and optionally assigns a new seed.
        """
        item = self.library_items.find_one({"_id": item_id})
        if not item:
            return False

        update = {
            "status": "pending",
            "image_path": None,
            "asset_path": None,
            "asset_type": None,
            "error": None,
            "reviewed_at": None,
            "reviewed_by": None,
            "review_notes": None
        }

        if new_seed:
            # Generate a new seed based on current time
            update["seed"] = hash(f"{item['description']}_{item_id}_{datetime.utcnow().timestamp()}") % (2**32)

        result = self.library_items.update_one(
            {"_id": item_id},
            {"$set": update}
        )
        return result.modified_count > 0

    def retry_stuck_item(self, item_id: str) -> bool:
        """
        Retry a stuck processing item or ready item.
        Keeps the image but resets to needs_3d to retry 3D generation.
        Works for: approved, needs_3d, needs_embedding, ready
        """
        item = self.library_items.find_one({"_id": item_id})
        if not item:
            return False

        allowed_states = ["approved", "needs_3d", "needs_embedding", "ready"]
        if item.get("status") not in allowed_states:
            return False

        # Keep the image, reset everything else
        update = {
            "status": "needs_3d",
            "asset_path": None,
            "asset_type": None,
            "error": None
        }

        result = self.library_items.update_one(
            {"_id": item_id},
            {"$set": update}
        )
        return result.modified_count > 0

    def reset_group_for_regeneration(self, variant_group_id: str, new_seeds: bool = True) -> int:
        """Reset all rejected/failed items in a group for regeneration."""
        items = list(self.library_items.find({
            "variant_group_id": variant_group_id,
            "status": {"$in": ["rejected", "failed"]}
        }))

        count = 0
        for item in items:
            if self.reset_item_for_regeneration(item["_id"], new_seed=new_seeds):
                count += 1

        return count

    def update_item_description(self, item_id: str,
                                 new_description: str) -> Dict[str, Any]:
        """
        Update item description (saves original).
        Item stays in current status.
        """
        item = self.library_items.find_one({"_id": item_id})
        if not item:
            return {"updated": False, "error": "Item not found"}

        # Save original description if not already saved
        original = item.get("original_description") or item["description"]

        self.library_items.update_one(
            {"_id": item_id},
            {"$set": {
                "description": new_description,
                "original_description": original
            }}
        )
        return {"updated": True, "original_description": original}

    def update_group_description(self, variant_group_id: str,
                                  new_description: str) -> Dict[str, Any]:
        """
        Update description for ALL items in a variant group.
        Saves the original description for reference.
        """
        # Find any item to get the original description
        sample = self.library_items.find_one({"variant_group_id": variant_group_id})
        if not sample:
            return {"updated": False, "error": "Group not found"}

        original = sample.get("original_description") or sample["description"]

        result = self.library_items.update_many(
            {"variant_group_id": variant_group_id},
            {"$set": {
                "description": new_description,
                "original_description": original
            }}
        )
        return {
            "updated": True,
            "updated_count": result.modified_count,
            "original_description": original
        }

    def create_variant(self, item_id: str, seed: int = None) -> str:
        """
        Create a new variant for the same description.
        Returns new item ID.
        """
        import uuid

        # Get source item
        source = self.library_items.find_one({"_id": item_id})
        if not source:
            raise ValueError(f"Item not found: {item_id}")

        # Count existing variants to determine new index
        existing = self.library_items.count_documents({
            "variant_group_id": source["variant_group_id"]
        })

        # Generate new seed if not provided
        if seed is None:
            seed = hash(f"{source['description']}_{existing}_{datetime.utcnow().timestamp()}") % (2**32)

        new_item = {
            "_id": uuid.uuid4().hex,
            "library_id": source["library_id"],
            "description": source["description"],
            "variant_group_id": source["variant_group_id"],
            "variant_index": existing,
            "seed": seed,
            "image_path": None,
            "asset_path": None,
            "asset_type": None,
            "text_embedding": None,
            "image_embedding": None,
            "status": "pending",
            "error": None,
            "created_at": datetime.utcnow()
        }
        self.library_items.insert_one(new_item)

        # Update library total_variants count
        self.libraries.update_one(
            {"_id": source["library_id"]},
            {"$inc": {"total_variants": 1}}
        )

        return new_item["_id"]

    def approve_group(self, variant_group_id: str, reviewer: str = None) -> int:
        """Approve all variants in a group. Returns count approved."""
        result = self.library_items.update_many(
            {"variant_group_id": variant_group_id, "status": "needs_review"},
            {"$set": {
                "status": "approved",
                "reviewed_at": datetime.utcnow(),
                "reviewed_by": reviewer
            }}
        )
        return result.modified_count

    def reject_group(self, variant_group_id: str, reviewer: str = None,
                     notes: str = None) -> int:
        """Reject all variants in a group. Returns count rejected."""
        update = {
            "status": "rejected",
            "reviewed_at": datetime.utcnow(),
            "reviewed_by": reviewer
        }
        if notes:
            update["review_notes"] = notes

        result = self.library_items.update_many(
            {"variant_group_id": variant_group_id, "status": "needs_review"},
            {"$set": update}
        )
        return result.modified_count

    def delete_item(self, item_id: str) -> bool:
        """Delete an item from the library."""
        item = self.library_items.find_one({"_id": item_id})
        if not item:
            return False

        # Delete from database
        result = self.library_items.delete_one({"_id": item_id})

        # Update library total_variants count
        if result.deleted_count > 0:
            self.libraries.update_one(
                {"_id": item["library_id"]},
                {"$inc": {"total_variants": -1}}
            )

        return result.deleted_count > 0

    def delete_group(self, variant_group_id: str) -> int:
        """Delete all variants in a group. Returns count deleted."""
        # Find items to get library_id
        items = list(self.library_items.find({"variant_group_id": variant_group_id}))
        if not items:
            return 0

        library_id = items[0]["library_id"]

        # Delete items
        result = self.library_items.delete_many({"variant_group_id": variant_group_id})

        # Update library total_variants count
        if result.deleted_count > 0:
            self.libraries.update_one(
                {"_id": library_id},
                {"$inc": {"total_variants": -result.deleted_count}}
            )

        return result.deleted_count

    def get_library_stats(self, library_id: str = None) -> Dict[str, int]:
        """Get count of items at each status. If library_id is None, counts all items."""
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        if library_id:
            pipeline.insert(0, {"$match": {"library_id": library_id}})
        results = list(self.library_items.aggregate(pipeline))
        stats = {r["_id"]: r["count"] for r in results}

        # Ensure all statuses have a value
        all_statuses = ["pending", "needs_masking", "needs_review", "approved", "rejected",
                       "needs_3d", "needs_embedding", "ready", "failed"]
        for status in all_statuses:
            if status not in stats:
                stats[status] = 0

        # Add combined "processing" count (approved through needs_embedding)
        stats["processing"] = (stats.get("approved", 0) +
                               stats.get("needs_3d", 0) + stats.get("needs_embedding", 0))

        return stats

    def update_library_status(self, library_id: str):
        """
        Update library status based on item statuses.
        Sets to 'embedding' when all items have 3D assets,
        'ready' when all items have embeddings.
        """
        items = list(self.library_items.find({"library_id": library_id}))

        if not items:
            return

        all_have_assets = all(item.get("asset_path") for item in items)
        all_have_embeddings = all(
            item.get("text_embedding") and item.get("image_embedding")
            for item in items
        )
        any_failed = any(item.get("status") == "failed" for item in items)

        if any_failed:
            status = "partial"
        elif all_have_embeddings:
            status = "ready"
            self.libraries.update_one(
                {"_id": library_id},
                {"$set": {"status": status, "completed_at": datetime.utcnow()}}
            )
            return
        elif all_have_assets:
            status = "embedding"
        else:
            status = "generating"

        self.libraries.update_one(
            {"_id": library_id},
            {"$set": {"status": status}}
        )

    def search_library_text(self, query: str, limit: int = 10,
                             library_id: str = None) -> List[Dict[str, Any]]:
        """
        Search library items by text (simple substring match).
        For semantic search, use search_library_vector.

        Args:
            query: Text to search for in descriptions
            limit: Max results
            library_id: Optional - limit search to specific library
        """
        search_query = {
            "description": {"$regex": query, "$options": "i"},
            "status": "ready"
        }
        if library_id:
            search_query["library_id"] = library_id

        return list(self.library_items.find(search_query).limit(limit))

    def search_library_vector(self, query_embedding: List[float],
                              embedding_field: str = "text_embedding",
                              limit: int = 10,
                              library_id: str = None) -> List[Dict[str, Any]]:
        """
        Vector similarity search using MongoDB Atlas vector search.

        Args:
            query_embedding: Query embedding vector (768 dims)
            embedding_field: Which embedding to search (text_embedding or image_embedding)
            limit: Max results to return
            library_id: Optional - limit search to specific library

        Returns:
            List of items with similarity scores
        """
        # MongoDB Atlas vector search aggregation
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "library_vector_index",
                    "path": embedding_field,
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "library_id": 1,
                    "description": 1,
                    "variant_index": 1,
                    "variant_group_id": 1,
                    "seed": 1,
                    "status": 1,
                    "image_path": 1,
                    "asset_path": 1,
                    "asset_type": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Add library filter if specified
        if library_id:
            pipeline.insert(1, {"$match": {"library_id": library_id}})

        try:
            results = list(self.library_items.aggregate(pipeline))
            # Normalize score field to similarity for consistency
            for r in results:
                r["similarity"] = r.pop("score", 0)
            return results
        except Exception as e:
            logger.warning(f"Vector search failed (Atlas index may not exist): {e}")
            # Fallback to brute-force cosine similarity for local MongoDB
            return self._brute_force_vector_search(
                query_embedding, embedding_field, limit, library_id
            )

    def _brute_force_vector_search(self, query_embedding: List[float],
                                   embedding_field: str,
                                   limit: int,
                                   library_id: str = None) -> List[Dict[str, Any]]:
        """
        Fallback vector search for local MongoDB without Atlas.
        Computes cosine similarity in Python (slower but works locally).
        """
        import numpy as np

        query = np.array(query_embedding)

        search_query = {
            embedding_field: {"$exists": True, "$ne": None},
            "status": "ready"
        }
        if library_id:
            search_query["library_id"] = library_id

        items = list(self.library_items.find(search_query))

        results = []
        for item in items:
            embedding = np.array(item[embedding_field])
            # Cosine similarity (vectors are already normalized)
            similarity = float(np.dot(query, embedding))
            results.append({
                **item,
                "similarity": similarity
            })

        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")


# Global database instance
db = DatabaseService()
