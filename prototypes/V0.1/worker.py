"""
Job Processing Worker

Polls MongoDB for pending generation jobs and processes them
through the GPU orchestrator pipeline.

Usage:
    python worker.py              # Process jobs continuously
    python worker.py --once       # Process one batch and exit
"""

import os
import sys
import time
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_jobs(once: bool = False, batch_size: int = 100):
    """
    Main worker loop with DYNAMIC MODEL-AWARE processing.

    Instead of processing fixed batches, this checks for jobs at each stage
    and only switches models when no jobs remain for the current model.

    Flow (3 stages, 3 GPU models):
    1. Check for "pending" jobs (need SD3.5)
       - If any: load SD3.5, process ALL, mark as "needs_mask"
    2. Check for "needs_mask" jobs (need SAM)
       - If any: load SAM, process ALL, mark as "needs_splat"
    3. Check for "needs_splat" jobs (need SAM3D)
       - If any: load SAM3D, process ALL, mark as "completed"
    4. If no jobs at any stage, sleep and poll again

    Args:
        once: If True, process available jobs once and exit
        batch_size: Max jobs to process per stage (default 100)
    """
    from services import db, GPUOrchestrator

    # Connect to database
    try:
        db.connect()
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Make sure MongoDB is running: mongod")
        return

    # Initialize orchestrator
    orchestrator = GPUOrchestrator(
        output_dir=os.path.join(os.path.dirname(__file__), 'media', 'generated')
    )
    logger.info(f"GPU Orchestrator initialized. Device: {orchestrator.device}")
    logger.info(f"VRAM: {orchestrator.get_vram_usage()}")

    # Check external repos
    if not orchestrator.sd35_repo.exists():
        logger.error(f"SD3.5 repo not found at: {orchestrator.sd35_repo}")
        logger.error("Set SD35_REPO_PATH environment variable or clone the repo")
        return

    if not orchestrator.sam3d_repo.exists():
        logger.error(f"SAM3D repo not found at: {orchestrator.sam3d_repo}")
        logger.error("Set SAM3D_REPO_PATH environment variable or clone the repo")
        return

    logger.info(f"SD3.5 repo: {orchestrator.sd35_repo}")
    logger.info(f"SAM3D repo: {orchestrator.sam3d_repo}")
    logger.info("DYNAMIC MODE: Processing all jobs for current model before switching")

    poll_interval = 5  # seconds

    while True:
        try:
            # ========== Check for jobs needing SD3.5 (image generation) ==========
            pending_jobs = db.get_jobs_by_stage("pending", limit=batch_size)

            if pending_jobs:
                logger.info(f"=== SD3.5 STAGE: {len(pending_jobs)} jobs need images ===")
                orchestrator.load_stable_diffusion()

                for job_data in pending_jobs:
                    job_id = job_data['_id']
                    prompt = job_data.get('prompt', '')
                    logger.info(f"  [{job_id}] Generating image: {prompt[:40]}...")

                    try:
                        # Generate image (with white background prompt)
                        image_path = orchestrator.generate_image(prompt)
                        logger.info(f"  [{job_id}] Image saved: {image_path}")

                        # Update job: ready for mask generation
                        db.jobs.update_one(
                            {"_id": job_id},
                            {"$set": {
                                "status": "needs_mask",
                                "image_path": image_path
                            }}
                        )

                    except Exception as e:
                        logger.error(f"  [{job_id}] Image generation failed: {e}")
                        db.update_job_status(job_id, "failed", error=str(e))

                # Check if more pending jobs arrived while we were processing
                more_pending = db.count_jobs_by_stage("pending")
                if more_pending > 0:
                    logger.info(f"  {more_pending} more pending jobs arrived, continuing SD3.5...")
                    continue  # Stay on SD3.5

                # No more pending jobs, unload SD3.5
                orchestrator.clear_gpu_memory()
                logger.info("SD3.5 unloaded - no more pending jobs")

            # ========== Check for jobs needing SAM (mask generation) ==========
            mask_jobs = db.get_jobs_by_stage("needs_mask", limit=batch_size)

            if mask_jobs:
                logger.info(f"=== SAM STAGE: {len(mask_jobs)} jobs need masks ===")
                orchestrator.load_sam()

                for job_data in mask_jobs:
                    job_id = job_data['_id']
                    image_path = job_data.get('image_path')

                    if not image_path:
                        logger.error(f"  [{job_id}] No image_path, marking failed")
                        db.update_job_status(job_id, "failed", error="No image path")
                        continue

                    logger.info(f"  [{job_id}] Generating mask for {image_path}...")

                    try:
                        mask_path = orchestrator.generate_mask(image_path)
                        logger.info(f"  [{job_id}] Mask saved: {mask_path}")

                        # Update job: ready for splat generation
                        db.jobs.update_one(
                            {"_id": job_id},
                            {"$set": {
                                "status": "needs_splat",
                                "mask_path": mask_path
                            }}
                        )

                    except Exception as e:
                        logger.error(f"  [{job_id}] Mask generation failed: {e}")
                        db.update_job_status(job_id, "failed", error=str(e))

                # Check if more mask jobs or higher-priority jobs arrived
                more_masks = db.count_jobs_by_stage("needs_mask")
                new_pending = db.count_jobs_by_stage("pending")

                if more_masks > 0 and new_pending == 0:
                    logger.info(f"  {more_masks} more mask jobs, continuing SAM...")
                    continue  # Stay on SAM
                elif new_pending > 0:
                    logger.info(f"  {new_pending} new pending jobs, switching to SD3.5...")
                    orchestrator.clear_gpu_memory()
                    continue  # Switch to SD3.5

                # No more mask jobs, unload SAM
                orchestrator.unload_sam()
                logger.info("SAM unloaded - no more mask jobs")

            # ========== Check for jobs needing SAM3D (splat generation) ==========
            # Process ONE job at a time to prevent memory accumulation
            splat_jobs = db.get_jobs_by_stage("needs_splat", limit=1)

            if splat_jobs:
                job_data = splat_jobs[0]
                job_id = job_data['_id']
                image_path = job_data.get('image_path')
                mask_path = job_data.get('mask_path')
                prompt = job_data.get('prompt', '')
                scene_id = job_data.get('scene_id')
                node_id = job_data.get('node_id')

                logger.info(f"=== SAM3D STAGE: Processing job {job_id} ===")

                if not image_path or not mask_path:
                    logger.error(f"  [{job_id}] Missing image_path or mask_path, marking failed")
                    db.update_job_status(job_id, "failed", error="Missing image or mask path")
                    continue

                logger.info(f"  [{job_id}] Generating splat from {image_path}...")

                try:
                    # Load SAM3D fresh for each job to prevent memory accumulation
                    orchestrator.load_sam3d()
                    splat_path = orchestrator.generate_splat(image_path, mask_path)
                    logger.info(f"  [{job_id}] Splat saved: {splat_path}")

                    # Mark completed
                    db.update_job_status(job_id, "completed", output_path=splat_path)

                    # Register asset
                    db.register_asset(
                        asset_type="splat",
                        path=splat_path,
                        scene_id=scene_id,
                        node_id=node_id,
                        metadata={
                            "prompt": prompt,
                            "image_path": image_path,
                            "mask_path": mask_path
                        }
                    )

                except Exception as e:
                    logger.error(f"  [{job_id}] Splat generation failed: {e}")
                    db.update_job_status(job_id, "failed", error=str(e))

                finally:
                    # ALWAYS unload SAM3D after each job to free memory
                    orchestrator.clear_gpu_memory()
                    logger.info("SAM3D unloaded after job")

                # Check for more jobs (will reload SAM3D on next iteration)
                more_splats = db.count_jobs_by_stage("needs_splat")
                if more_splats > 0:
                    logger.info(f"  {more_splats} more splat jobs remaining...")
                continue

            # ========== No jobs at any stage ==========
            if not pending_jobs and not mask_jobs and not splat_jobs:
                # Log stats at INFO level so user can see them
                completed = db.count_jobs_by_stage("completed")
                failed = db.count_jobs_by_stage("failed")

                if failed > 0:
                    logger.warning(f"No active jobs. {failed} jobs failed, {completed} completed.")
                    logger.warning("Failed jobs will not be retried. Check logs above for errors.")

                if once:
                    logger.info("No jobs to process. Exiting (--once mode)")
                    break

                logger.debug(f"Waiting for jobs... (completed: {completed}, failed: {failed})")
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            orchestrator.clear_gpu_memory()
            time.sleep(poll_interval)

    # Cleanup
    orchestrator.clear_gpu_memory()
    db.close()
    logger.info("Worker shutdown complete")


def check_dependencies():
    """Check if required dependencies are available"""
    issues = []

    # Check torch/CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - will run on CPU (very slow)")
        else:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check rembg
    try:
        import rembg
    except ImportError:
        issues.append("rembg not installed - run: pip install rembg")

    # Check PIL
    try:
        from PIL import Image
    except ImportError:
        issues.append("Pillow not installed - run: pip install Pillow")

    # Check pymongo
    try:
        import pymongo
    except ImportError:
        issues.append("pymongo not installed - run: pip install pymongo")

    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Dojo Job Worker")
    parser.add_argument("--once", action="store_true",
                        help="Process one batch and exit")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Jobs per batch (default: 1)")
    parser.add_argument("--check", action="store_true",
                        help="Check dependencies and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("  Digital Dojo - Job Processing Worker")
    print("=" * 60)

    # Check dependencies
    issues = check_dependencies()
    if issues:
        print("\nDependency issues:")
        for issue in issues:
            print(f"  - {issue}")
        if args.check:
            sys.exit(1 if issues else 0)
        print("\nContinuing anyway...\n")
    elif args.check:
        print("All dependencies OK")
        sys.exit(0)

    print(f"\nMode: {'Single batch' if args.once else 'Continuous'}")
    print(f"Batch size: {args.batch_size}")
    print("")

    process_jobs(once=args.once, batch_size=args.batch_size)
