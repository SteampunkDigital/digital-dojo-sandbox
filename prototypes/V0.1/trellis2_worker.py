"""
Trellis2 worker - runs in the trellis2 conda environment.
Polls MongoDB for needs_3d items and generates GLB meshes.

Trellis2 has a slow memory leak, so use --max-items to restart periodically.
Wrap in a loop for continuous processing:
    while true; do python trellis2_worker.py --max-items 10; sleep 2; done

Usage:
    conda activate trellis2
    python trellis2_worker.py                  # Process 10 items and exit (default)
    python trellis2_worker.py --max-items 20   # Process 20 items and exit
    python trellis2_worker.py --max-items 0    # No limit (not recommended)
    python trellis2_worker.py --once           # Process available items (up to limit) and exit
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import uuid
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'digital_dojo')
TRELLIS2_REPO_PATH = os.getenv('TRELLIS2_REPO_PATH', '/mnt/g/GitHub/TRELLIS.2')
TRELLIS2_MODEL = os.getenv('TRELLIS2_MODEL', 'microsoft/TRELLIS.2-4B')
TRELLIS2_PIPELINE_TYPE = os.getenv('TRELLIS2_PIPELINE_TYPE', '1024_cascade')
TRELLIS2_DECIMATION = int(os.getenv('TRELLIS2_DECIMATION', '200000'))
TRELLIS2_TEXTURE_SIZE = int(os.getenv('TRELLIS2_TEXTURE_SIZE', '2048'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.join(os.path.dirname(__file__), 'media', 'generated'))
POLL_INTERVAL = int(os.getenv('TRELLIS2_POLL_INTERVAL', '10'))
MAX_ITEMS = int(os.getenv('TRELLIS2_MAX_ITEMS', '10'))

# Add Trellis2 to path
sys.path.insert(0, TRELLIS2_REPO_PATH)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Trellis2Worker] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Import GPU lock (standalone module, works in any conda env)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu_lock import GPULock


def get_db():
    """Get MongoDB database connection."""
    client = MongoClient(MONGODB_URI)
    return client[MONGODB_DATABASE]


def get_pending_items(db, batch_size: int):
    """Get all items needing 3D generation from both collections."""
    items = []

    # Library items with status 'needs_3d'
    for item in db.library_items.find(
        {'status': 'needs_3d'},
        limit=batch_size
    ).sort('created_at', 1):
        items.append({
            'collection': 'library_items',
            'id': item['_id'],
            'image_path': item.get('image_path'),
            'seed': item.get('seed', 42),
            'description': item.get('description', ''),
            'next_status': 'needs_embedding',
        })

    # Scene jobs with status 'needs_3d'
    remaining = batch_size - len(items)
    if remaining > 0:
        for job in db.jobs.find(
            {'status': 'needs_3d'},
            limit=remaining
        ).sort('created_at', 1):
            items.append({
                'collection': 'jobs',
                'id': job['_id'],
                'image_path': job.get('image_path'),
                'seed': job.get('seed', 42),
                'description': job.get('prompt', ''),
                'next_status': 'completed',
            })

    return items


def mark_processing(db, item):
    """Mark an item as currently being processed."""
    db[item['collection']].update_one(
        {'_id': item['id']},
        {'$set': {
            'status': 'processing_3d',
            'started_3d_at': datetime.now(timezone.utc),
        }}
    )


def mark_completed(db, item, output_path: str):
    """Mark an item as completed with output path."""
    update = {
        'status': item['next_status'],
        'asset_path': output_path,
        'asset_type': 'mesh',
        'completed_3d_at': datetime.now(timezone.utc),
    }
    # For scene jobs, also set output_path and completed_at
    if item['collection'] == 'jobs':
        update['output_path'] = output_path
        update['completed_at'] = datetime.now(timezone.utc)

    db[item['collection']].update_one(
        {'_id': item['id']},
        {'$set': update}
    )


def mark_failed(db, item, error: str):
    """Mark an item as failed."""
    db[item['collection']].update_one(
        {'_id': item['id']},
        {'$set': {
            'status': 'failed',
            'error': error,
            'failed_at': datetime.now(timezone.utc),
        }}
    )


def load_pipeline():
    """Load the Trellis2 pipeline."""
    logger.info(f"Loading Trellis2 pipeline: {TRELLIS2_MODEL}")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS2_MODEL)
    pipeline.cuda()
    logger.info("Trellis2 pipeline loaded")
    return pipeline


def generate_glb(pipeline, image_path: str, seed: int) -> str:
    """Generate a GLB mesh from an image. Returns output path."""
    import o_voxel

    # Load image
    image = Image.open(image_path)
    logger.info(f"Generating mesh from {image_path} (seed={seed})")

    # Generate mesh
    mesh = pipeline.run(
        image,
        seed=seed,
        pipeline_type=TRELLIS2_PIPELINE_TYPE,
    )[0]

    # Export to GLB
    output_filename = f"mesh_{uuid.uuid4().hex[:8]}.glb"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=TRELLIS2_DECIMATION,
        texture_size=TRELLIS2_TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(output_path)
    logger.info(f"Exported: {output_path}")

    return output_path


def process_batch(pipeline, db, gpu_lock: GPULock, batch_size: int, max_items: int, total_processed: int) -> int:
    """Process a batch of needs_3d items. Returns count processed."""
    # Limit batch to remaining items allowed
    if max_items > 0:
        batch_size = min(batch_size, max_items - total_processed)
        if batch_size <= 0:
            return 0

    items = get_pending_items(db, batch_size)
    if not items:
        return 0

    logger.info(f"Processing batch of {len(items)} items")
    processed = 0

    for item in items:
        # Check max items limit
        if max_items > 0 and (total_processed + processed) >= max_items:
            logger.info(f"Reached max items limit ({max_items}), stopping")
            break

        if not item.get('image_path') or not os.path.exists(item['image_path']):
            mark_failed(db, item, f"Image not found: {item.get('image_path')}")
            logger.warning(f"Skipping {item['id']}: image not found")
            continue

        mark_processing(db, item)
        gpu_lock.heartbeat()

        try:
            output_path = generate_glb(pipeline, item['image_path'], item['seed'])
            mark_completed(db, item, output_path)
            processed += 1
            logger.info(f"Completed {item['id']}: {item['description'][:50]} ({total_processed + processed} total)")
        except Exception as e:
            mark_failed(db, item, str(e))
            logger.error(f"Failed {item['id']}: {e}")
            # On CUDA OOM, stop immediately
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.error("CUDA error detected, stopping batch to prevent corruption")
                break

        # Cleanup between items
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return processed


def run_worker(once: bool = False, batch_size: int = 100, max_items: int = 10):
    """Main worker loop."""
    db = get_db()
    gpu_lock = GPULock("trellis2_worker")
    pipeline = None
    total_processed = 0

    logger.info(f"Trellis2 worker started (once={once}, batch_size={batch_size}, max_items={max_items})")
    logger.info(f"MongoDB: {MONGODB_URI}/{MONGODB_DATABASE}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    if max_items > 0:
        logger.info(f"Will exit after {max_items} items (memory leak workaround)")

    try:
        while True:
            # Check max items limit
            if max_items > 0 and total_processed >= max_items:
                logger.info(f"Processed {total_processed} items, exiting (max_items={max_items})")
                break

            # Check for pending work
            items = get_pending_items(db, 1)
            if not items:
                if once:
                    logger.info("No items to process, exiting (--once mode)")
                    break
                time.sleep(POLL_INTERVAL)
                continue

            # Acquire GPU lock
            logger.info("Acquiring GPU lock...")
            if not gpu_lock.acquire(timeout=-1):
                logger.error("Failed to acquire GPU lock")
                continue

            try:
                # Load pipeline if not loaded
                if pipeline is None:
                    pipeline = load_pipeline()

                # Process available items
                processed = process_batch(pipeline, db, gpu_lock, batch_size, max_items, total_processed)
                total_processed += processed
                logger.info(f"Batch complete: {processed} items ({total_processed} total)")

            finally:
                # Release GPU lock
                gpu_lock.release()
                logger.info("GPU lock released")

            if once:
                break

            # Brief pause before checking for more work
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    finally:
        # Cleanup
        if pipeline is not None:
            del pipeline
            torch.cuda.empty_cache()
        logger.info(f"Worker stopped. Processed {total_processed} items total.")
        remaining = db.library_items.count_documents({'status': 'needs_3d'})
        remaining += db.jobs.count_documents({'status': 'needs_3d'})
        if remaining > 0:
            logger.info(f"{remaining} items still need 3D generation. Restart worker to continue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trellis2 3D generation worker")
    parser.add_argument("--once", action="store_true",
                        help="Process available items and exit")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Max items per batch")
    parser.add_argument("--max-items", type=int, default=MAX_ITEMS,
                        help=f"Exit after processing this many items (default: {MAX_ITEMS}, 0=unlimited)")
    args = parser.parse_args()

    run_worker(once=args.once, batch_size=args.batch_size, max_items=args.max_items)
