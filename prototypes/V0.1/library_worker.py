#!/usr/bin/env python3
"""
Library Worker - Processes object library items through the embedding pipeline.

Pipeline stages:
1. approved → needs_3d: Status transition for legacy items (no GPU needed)
2. needs_3d → needs_embedding: Trellis2 worker (separate process/env) generates mesh
3. needs_embedding → ready: Embedding model generates text + image embeddings

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
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_library_items(once: bool = False, batch_size: int = 100):
    """
    Main processing loop for library items.

    Handles approved→needs_3d transitions and embedding generation.
    3D mesh generation is handled by trellis2_worker.py in a separate env.
    """
    from services.database import DatabaseService
    from services.embedding_service import get_embedding_service
    from gpu_lock import GPULock

    db = DatabaseService()
    try:
        db.connect()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return

    logger.info("=" * 60)
    logger.info("Library Worker Started")
    logger.info("3D generation: handled by Trellis2 worker (separate env)")
    logger.info("This worker handles: status transitions + embeddings")
    logger.info("=" * 60)

    gpu_lock = GPULock("library_worker")
    poll_interval = 5

    while True:
        try:
            # ========== Stage 1: Approved → needs_3d (no GPU needed) ==========
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

                    db.update_library_item(item_id, status="needs_3d")
                    logger.info(f"  [{item_id}] → needs_3d (awaiting Trellis2 worker)")

            # ========== Stage 2: Trellis2 handles needs_3d → needs_embedding ==========
            # (handled by trellis2_worker.py in separate conda env)

            # ========== Stage 3: Embedding Generation ==========
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
                        library_id = item.get("library_id")

                        logger.info(f"  [{item_id}] Generating embeddings...")
                        gpu_lock.heartbeat()

                        try:
                            db.update_library_item(item_id, status="processing")

                            text_embedding = embedding_service.encode_text(description)

                            image_embedding = None
                            if image_path and os.path.exists(image_path):
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

                            if library_id:
                                db.update_library_status(library_id)

                        except Exception as e:
                            logger.error(f"  [{item_id}] Embedding generation failed: {e}")
                            db.update_library_item(item_id, status="failed", error=str(e))

                    embedding_service.unload()

                finally:
                    gpu_lock.release()
                    logger.info("GPU lock released (embeddings)")

                continue

            # ========== No items at any stage ==========
            if once:
                logger.info("No items to process, exiting (--once mode)")
                break

            # Log status summary
            approved = db.count_library_items_by_stage("approved")
            needs_3d = db.count_library_items_by_stage("needs_3d")
            needs_embedding = db.count_library_items_by_stage("needs_embedding")
            ready = db.count_library_items_by_stage("ready")

            if approved + needs_3d + needs_embedding > 0:
                logger.info(f"Queue: approved={approved}, 3d={needs_3d}, "
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
