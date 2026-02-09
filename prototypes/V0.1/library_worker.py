#!/usr/bin/env python3
"""
Library Worker - Processes object library items through the generation pipeline.

Pipeline stages for library items:
1. pending → needs_review: SD3.5 generates image
2. approved → needs_3d: Status transition (no GPU needed)
3. needs_3d → needs_embedding: Trellis2 worker (separate process/env) generates mesh
4. needs_embedding → ready: CLIP generates embeddings

Usage:
    python library_worker.py          # Run continuously
    python library_worker.py --once   # Process available items and exit
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system environment variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_library_items(once: bool = False, batch_size: int = 100):
    """
    Main processing loop for library items.

    Handles SD3.5 image generation, approved→needs_3d transitions,
    and CLIP embedding generation. 3D mesh generation is handled
    by the Trellis2 worker in a separate conda environment.
    """
    from services.database import DatabaseService
    from services.orchestrator import GPUOrchestrator
    from services.embedding_service import get_embedding_service
    from gpu_lock import GPULock

    # Connect to database
    db = DatabaseService()
    try:
        db.connect()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return

    # Initialize GPU orchestrator
    orchestrator = GPUOrchestrator()

    logger.info("=" * 60)
    logger.info("Library Worker Started")
    logger.info(f"SD3.5 repo: {orchestrator.sd35_repo}")
    logger.info("3D generation: handled by Trellis2 worker (separate env)")
    logger.info("=" * 60)

    gpu_lock = GPULock("library_worker")
    poll_interval = 5  # seconds

    while True:
        try:
            # ========== Stage 1: SD3.5 Image Generation ==========
            pending_items = db.get_library_items_by_stage("pending", limit=batch_size)

            if pending_items:
                logger.info(f"=== SD3.5 STAGE: {len(pending_items)} pending items ===")

                logger.info("Acquiring GPU lock for SD3.5...")
                if not gpu_lock.acquire(timeout=-1):
                    logger.error("Failed to acquire GPU lock")
                    continue

                try:
                    orchestrator.load_stable_diffusion()

                    for item in pending_items:
                        item_id = item["_id"]
                        description = item["description"]
                        seed = item.get("seed", 42)
                        library_id = item["library_id"]

                        logger.info(f"  [{item_id}] Generating image for: {description[:50]}...")
                        gpu_lock.heartbeat()

                        try:
                            db.update_library_item(item_id, status="processing")

                            # Generate image with specific seed for this variant
                            image_path = orchestrator.generate_image(
                                description,
                                seed=seed,
                                enhance_for_isolation=True
                            )

                            # Update item with image path and move to review stage
                            db.update_library_item(
                                item_id,
                                status="needs_review",
                                image_path=image_path
                            )
                            logger.info(f"  [{item_id}] Image saved: {image_path} (awaiting review)")

                        except Exception as e:
                            logger.error(f"  [{item_id}] Image generation failed: {e}")
                            db.update_library_item(item_id, status="failed", error=str(e))

                    # Check for more pending items before switching models
                    more_pending = db.count_library_items_by_stage("pending")
                    if more_pending > 0:
                        logger.info(f"  {more_pending} more pending items, continuing with SD3.5...")
                        continue

                    # Unload SD3.5 before next stage
                    orchestrator.clear_gpu_memory()

                finally:
                    gpu_lock.release()
                    logger.info("GPU lock released (SD3.5)")

            # ========== Stage 2: Approved → needs_3d (no GPU needed) ==========
            approved_items = db.get_library_items_by_stage("approved", limit=batch_size)

            if approved_items:
                logger.info(f"=== TRANSITION: {len(approved_items)} approved items → needs_3d ===")

                for item in approved_items:
                    item_id = item["_id"]
                    image_path = item.get("image_path")

                    if not image_path:
                        logger.error(f"  [{item_id}] Missing image_path")
                        db.update_library_item(item_id, status="failed", error="Missing image_path")
                        continue

                    # Simply transition to needs_3d for Trellis2 worker to pick up
                    db.update_library_item(item_id, status="needs_3d")
                    logger.info(f"  [{item_id}] → needs_3d (awaiting Trellis2 worker)")

                # Check for new pending items
                new_pending = db.count_library_items_by_stage("pending")
                if new_pending > 0:
                    logger.info(f"  {new_pending} new pending items, going back to SD3.5...")
                    continue

            # ========== Stage 3: Trellis2 handles needs_3d → needs_embedding ==========
            # (handled by trellis2_worker.py in separate conda env)

            # ========== Stage 4: CLIP Embedding Generation ==========
            needs_embedding_items = db.get_library_items_by_stage("needs_embedding", limit=batch_size)

            if needs_embedding_items:
                logger.info(f"=== EMBEDDING STAGE: {len(needs_embedding_items)} items ===")

                logger.info("Acquiring GPU lock for embeddings...")
                if not gpu_lock.acquire(timeout=-1):
                    logger.error("Failed to acquire GPU lock")
                    continue

                try:
                    embedding_service = get_embedding_service()
                    embedding_service.load()

                    for item in needs_embedding_items:
                        item_id = item["_id"]
                        description = item["description"]
                        image_path = item.get("image_path")
                        library_id = item["library_id"]

                        logger.info(f"  [{item_id}] Generating embeddings...")
                        gpu_lock.heartbeat()

                        try:
                            db.update_library_item(item_id, status="processing")

                            # Generate text embedding (pure text for text-based search)
                            text_embedding = embedding_service.encode_text(description)

                            # Generate image embedding with text description
                            # Combines visual features with semantic text for better retrieval
                            image_embedding = None
                            if image_path and os.path.exists(image_path):
                                # Use 70% image, 30% text for image embeddings
                                # This captures visual appearance while keeping semantic context
                                image_embedding = embedding_service.encode_image_with_text(
                                    image_path, description, image_weight=0.7
                                )

                            db.update_library_item(
                                item_id,
                                status="ready",
                                text_embedding=text_embedding,
                                image_embedding=image_embedding
                            )

                            logger.info(f"  [{item_id}] Embeddings generated (text: {len(text_embedding)} dims)")

                            # Update library status
                            db.update_library_status(library_id)

                        except Exception as e:
                            logger.error(f"  [{item_id}] Embedding generation failed: {e}")
                            db.update_library_item(item_id, status="failed", error=str(e))

                    # Unload embedding model to free memory
                    embedding_service.unload()

                    more_embeddings = db.count_library_items_by_stage("needs_embedding")
                    if more_embeddings > 0:
                        logger.info(f"  {more_embeddings} more items need embeddings...")

                finally:
                    gpu_lock.release()
                    logger.info("GPU lock released (embeddings)")

                continue

            # ========== No items at any stage ==========
            if once:
                logger.info("No items to process, exiting (--once mode)")
                break

            # Log status summary
            pending = db.count_library_items_by_stage("pending")
            needs_review = db.count_library_items_by_stage("needs_review")
            approved = db.count_library_items_by_stage("approved")
            needs_3d = db.count_library_items_by_stage("needs_3d")
            needs_embedding = db.count_library_items_by_stage("needs_embedding")
            ready = db.count_library_items_by_stage("ready")

            if pending + approved + needs_3d + needs_embedding > 0:
                logger.info(f"Queue: pending={pending}, review={needs_review}, "
                           f"approved={approved}, 3d={needs_3d}, "
                           f"embed={needs_embedding}, ready={ready}")
            else:
                logger.debug(f"No items in queue. Ready items: {ready}")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(poll_interval)

    # Cleanup
    orchestrator.clear_gpu_memory()
    db.close()
    logger.info("Library worker stopped")


def main():
    parser = argparse.ArgumentParser(description="Library item processing worker")
    parser.add_argument('--once', action='store_true',
                        help='Process available items and exit')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Max items to process per batch (default: 100)')
    args = parser.parse_args()

    process_library_items(once=args.once, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
