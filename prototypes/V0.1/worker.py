"""
SD3.5 Image Generation Worker

Polls MongoDB for pending generation jobs and generates images via SD3.5.
3D generation is handled by the Trellis2 worker in a separate conda environment.

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
    Main worker loop - SD3.5 image generation only.

    Flow (2 stages):
    1. Check for "pending" jobs (need SD3.5)
       - If any: load SD3.5, process ALL, mark as "needs_3d"
    2. Trellis2 worker (separate process) handles needs_3d → completed

    Args:
        once: If True, process available jobs once and exit
        batch_size: Max jobs to process per stage (default 100)
    """
    from services import db, GPUOrchestrator
    from gpu_lock import GPULock

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

    logger.info(f"SD3.5 repo: {orchestrator.sd35_repo}")
    logger.info("SD3.5-only worker: 3D generation handled by Trellis2 worker")

    gpu_lock = GPULock("sd35_worker")
    poll_interval = 5  # seconds

    while True:
        try:
            # ========== Check for jobs needing SD3.5 (image generation) ==========
            pending_jobs = db.get_jobs_by_stage("pending", limit=batch_size)

            if pending_jobs:
                logger.info(f"=== SD3.5 STAGE: {len(pending_jobs)} jobs need images ===")

                logger.info("Acquiring GPU lock...")
                if not gpu_lock.acquire(timeout=-1):
                    logger.error("Failed to acquire GPU lock")
                    continue

                try:
                    orchestrator.load_stable_diffusion()

                    for job_data in pending_jobs:
                        job_id = job_data['_id']
                        prompt = job_data.get('prompt', '')
                        logger.info(f"  [{job_id}] Generating image: {prompt[:40]}...")
                        gpu_lock.heartbeat()

                        try:
                            image_path = orchestrator.generate_image(prompt)
                            logger.info(f"  [{job_id}] Image saved: {image_path}")

                            # Update job: ready for 3D generation (Trellis2 worker)
                            db.jobs.update_one(
                                {"_id": job_id},
                                {"$set": {
                                    "status": "needs_3d",
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
                        continue

                    # No more pending jobs, unload SD3.5
                    orchestrator.clear_gpu_memory()
                    logger.info("SD3.5 unloaded - no more pending jobs")

                finally:
                    gpu_lock.release()
                    logger.info("GPU lock released")

            # ========== No jobs to process ==========
            if not pending_jobs:
                completed = db.count_jobs_by_stage("completed")
                failed = db.count_jobs_by_stage("failed")
                needs_3d = db.count_jobs_by_stage("needs_3d")

                if failed > 0:
                    logger.warning(f"No active jobs. {failed} failed, {completed} completed, {needs_3d} awaiting Trellis2.")

                if once:
                    logger.info("No jobs to process. Exiting (--once mode)")
                    break

                logger.debug(f"Waiting for jobs... (completed: {completed}, needs_3d: {needs_3d}, failed: {failed})")
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

    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - will run on CPU (very slow)")
        else:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        issues.append("PyTorch not installed")

    try:
        from PIL import Image
    except ImportError:
        issues.append("Pillow not installed - run: pip install Pillow")

    try:
        import pymongo
    except ImportError:
        issues.append("pymongo not installed - run: pip install pymongo")

    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Dojo SD3.5 Worker")
    parser.add_argument("--once", action="store_true",
                        help="Process one batch and exit")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Jobs per batch (default: 100)")
    parser.add_argument("--check", action="store_true",
                        help="Check dependencies and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("  Digital Dojo - SD3.5 Image Generation Worker")
    print("  (3D generation handled by Trellis2 worker)")
    print("=" * 60)

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
