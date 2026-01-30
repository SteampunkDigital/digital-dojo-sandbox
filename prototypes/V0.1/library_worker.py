#!/usr/bin/env python3
"""
Library Worker - Processes object library items through the generation pipeline.

Pipeline stages for library items:
1. pending → needs_mask: SD3.5 generates image
2. needs_mask → needs_3d: SAM generates mask
3. needs_3d → needs_embedding: SAM3D generates mesh/splat
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

    Uses the same GPU orchestrator as the scene worker but processes
    library_items collection instead of jobs collection.
    """
    from services.database import DatabaseService
    from services.orchestrator import GPUOrchestrator
    from services.embedding_service import get_embedding_service

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
    logger.info(f"SAM3D repo: {orchestrator.sam3d_repo}")
    logger.info(f"Output format: {orchestrator.get_output_format()}")
    logger.info("=" * 60)

    poll_interval = 5  # seconds

    while True:
        try:
            # ========== Stage 1: SD3.5 Image Generation ==========
            pending_items = db.get_library_items_by_stage("pending", limit=batch_size)

            if pending_items:
                logger.info(f"=== SD3.5 STAGE: {len(pending_items)} pending items ===")
                orchestrator.load_stable_diffusion()

                for item in pending_items:
                    item_id = item["_id"]
                    description = item["description"]
                    seed = item.get("seed", 42)
                    library_id = item["library_id"]

                    logger.info(f"  [{item_id}] Generating image for: {description[:50]}...")

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

            # ========== Stage 2: SAM Mask Generation (approved items only) ==========
            # Only process items that have been approved by human review
            approved_items = db.get_library_items_by_stage("approved", limit=batch_size)

            if approved_items:
                logger.info(f"=== SAM STAGE: {len(approved_items)} approved items need masks ===")

                for item in approved_items:
                    item_id = item["_id"]
                    image_path = item.get("image_path")

                    if not image_path:
                        logger.error(f"  [{item_id}] Missing image_path")
                        db.update_library_item(item_id, status="failed", error="Missing image_path")
                        continue

                    logger.info(f"  [{item_id}] Generating mask for {image_path}...")

                    try:
                        db.update_library_item(item_id, status="processing")

                        mask_path = orchestrator.generate_mask(image_path)

                        db.update_library_item(
                            item_id,
                            status="needs_3d",
                            mask_path=mask_path
                        )
                        logger.info(f"  [{item_id}] Mask saved: {mask_path}")

                    except Exception as e:
                        logger.error(f"  [{item_id}] Mask generation failed: {e}")
                        db.update_library_item(item_id, status="failed", error=str(e))

                # Check for more items or new pending
                new_pending = db.count_library_items_by_stage("pending")
                if new_pending > 0:
                    logger.info(f"  {new_pending} new pending items, going back to SD3.5...")
                    continue

                more_approved = db.count_library_items_by_stage("approved")
                if more_approved > 0:
                    logger.info(f"  {more_approved} more approved items need masks...")
                    continue

                # Unload SAM
                orchestrator.unload_sam()

            # ========== Stage 3: SAM3D 3D Generation (subprocess) ==========
            # Run SAM3D in a subprocess to guarantee GPU memory cleanup
            needs_3d_items = db.get_library_items_by_stage("needs_3d", limit=1)

            if needs_3d_items:
                import subprocess
                import uuid

                item = needs_3d_items[0]
                item_id = item["_id"]
                image_path = item.get("image_path")
                mask_path = item.get("mask_path")
                seed = item.get("seed", 42)

                logger.info(f"=== 3D GENERATION STAGE (subprocess): Processing item {item_id} ===")

                if not image_path or not mask_path:
                    logger.error(f"  [{item_id}] Missing image or mask path")
                    db.update_library_item(item_id, status="failed", error="Missing paths")
                    continue

                output_format = orchestrator.get_output_format()
                logger.info(f"  [{item_id}] Generating {output_format} via subprocess...")

                db.update_library_item(item_id, status="processing")

                # Generate output path
                ext = ".glb" if output_format == "mesh" else ".ply"
                output_path = str(Path(orchestrator.output_dir) / f"{output_format}_{uuid.uuid4().hex[:8]}{ext}")

                # Run SAM3D in subprocess
                script_path = Path(__file__).parent / "sam3d_subprocess.py"
                cmd = [
                    sys.executable, str(script_path),
                    image_path, mask_path, output_path,
                    "--format", output_format,
                    "--seed", str(seed)
                ]

                try:
                    # Stream output in real-time (no capture)
                    logger.info(f"  [{item_id}] Running: {' '.join(cmd[-4:])}")
                    process = subprocess.Popen(
                        cmd,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                    )

                    # Wait with timeout
                    try:
                        returncode = process.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        logger.error(f"  [{item_id}] 3D generation timed out (5 min)")
                        db.update_library_item(item_id, status="failed", error="Timeout after 5 minutes")
                        continue

                    if returncode == 0 and os.path.exists(output_path):
                        db.update_library_item(
                            item_id,
                            status="needs_embedding",
                            asset_path=output_path,
                            asset_type=output_format
                        )
                        logger.info(f"  [{item_id}] {output_format.capitalize()} saved: {output_path}")
                    else:
                        error_msg = f"Subprocess exited with code {returncode}"
                        logger.error(f"  [{item_id}] 3D generation failed: {error_msg}")
                        db.update_library_item(item_id, status="failed", error=error_msg)

                except Exception as e:
                    logger.error(f"  [{item_id}] 3D generation failed: {e}")
                    db.update_library_item(item_id, status="failed", error=str(e)[:500])

                # Check for more items
                more_3d = db.count_library_items_by_stage("needs_3d")
                if more_3d > 0:
                    logger.info(f"  {more_3d} more items need 3D generation...")
                continue

            # ========== Stage 4: CLIP Embedding Generation ==========
            needs_embedding_items = db.get_library_items_by_stage("needs_embedding", limit=batch_size)

            if needs_embedding_items:
                logger.info(f"=== EMBEDDING STAGE: {len(needs_embedding_items)} items ===")

                embedding_service = get_embedding_service()
                embedding_service.load()

                for item in needs_embedding_items:
                    item_id = item["_id"]
                    description = item["description"]
                    image_path = item.get("image_path")
                    library_id = item["library_id"]

                    logger.info(f"  [{item_id}] Generating embeddings...")

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
                continue

            # ========== No items at any stage ==========
            if once:
                logger.info("No items to process, exiting (--once mode)")
                break

            # Log status summary
            pending = db.count_library_items_by_stage("pending")
            needs_mask = db.count_library_items_by_stage("needs_mask")
            needs_3d = db.count_library_items_by_stage("needs_3d")
            needs_embedding = db.count_library_items_by_stage("needs_embedding")
            ready = db.count_library_items_by_stage("ready")

            if pending + needs_mask + needs_3d + needs_embedding > 0:
                logger.info(f"Queue: pending={pending}, mask={needs_mask}, "
                           f"3d={needs_3d}, embed={needs_embedding}, ready={ready}")
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
