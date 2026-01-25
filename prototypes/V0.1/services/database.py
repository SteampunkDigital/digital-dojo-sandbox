"""
MongoDB Database Service

Handles connection to MongoDB and provides collections for:
- Scenes: Stored scene graphs
- Jobs: Generation job queue
- Assets: Generated asset metadata
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
          - "needs_mask" = image done, needs mask generation (SAM)
          - "needs_splat" = mask done, needs splat generation (SAM3D)
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

    def reset_failed_jobs(self, reset_to_stage: str = "needs_splat") -> int:
        """
        Reset all failed jobs to retry them.

        Args:
            reset_to_stage: Stage to reset to. Options:
              - "pending" = restart from image generation
              - "needs_mask" = restart from mask generation (keep image)
              - "needs_splat" = restart from splat generation (keep image + mask)

        Returns:
            Number of jobs reset
        """
        # Find all failed jobs
        failed_jobs = list(self.jobs.find({"status": "failed"}))

        reset_count = 0
        for job in failed_jobs:
            # Determine what stage to reset to based on what we have
            if reset_to_stage == "pending":
                # Full restart
                new_status = "pending"
            elif reset_to_stage == "needs_mask":
                # Restart from mask generation if we have an image
                if job.get("image_path"):
                    new_status = "needs_mask"
                else:
                    new_status = "pending"
            else:  # needs_splat
                # Restart from splat generation if we have image + mask
                if job.get("image_path") and job.get("mask_path"):
                    new_status = "needs_splat"
                elif job.get("image_path"):
                    new_status = "needs_mask"
                else:
                    new_status = "pending"

            # Reset the job
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

    def reset_all_jobs(self, reset_to_stage: str = "needs_splat") -> int:
        """
        Reset ALL jobs (including completed) to retry them.

        Args:
            reset_to_stage: Stage to reset to. Options:
              - "pending" = restart from image generation
              - "needs_mask" = restart from mask generation (keep image)
              - "needs_splat" = restart from splat generation (keep image + mask)

        Returns:
            Number of jobs reset
        """
        # Find all jobs that aren't already at the target stage
        all_jobs = list(self.jobs.find({
            "status": {"$nin": ["pending", "needs_mask", "needs_splat"]}
        }))

        reset_count = 0
        for job in all_jobs:
            # Determine what stage to reset to based on what we have
            if reset_to_stage == "pending":
                new_status = "pending"
            elif reset_to_stage == "needs_mask":
                if job.get("image_path"):
                    new_status = "needs_mask"
                else:
                    new_status = "pending"
            else:  # needs_splat
                if job.get("image_path") and job.get("mask_path"):
                    new_status = "needs_splat"
                elif job.get("image_path"):
                    new_status = "needs_mask"
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

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")


# Global database instance
db = DatabaseService()
