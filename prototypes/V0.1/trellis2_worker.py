"""
Trellis2 worker - runs in the trellis2 conda environment.
Polls MongoDB for needs_3d items and generates GLB meshes.

Architecture: lightweight supervisor spawns a fresh subprocess for each batch.
This gives true memory isolation — when the child exits, ALL GPU/CPU memory
is reclaimed by the OS. No more OOM kills from accumulated PyTorch state.

Usage:
    conda activate trellis2
    python trellis2_worker.py              # Supervisor: polls and spawns children
    python trellis2_worker.py --once       # Process one batch and exit
    python trellis2_worker.py --batch      # (internal) Child process: do GPU work
"""
import os
import sys
import time
import argparse
import logging
import subprocess as sp
from datetime import datetime, timezone

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
TRELLIS2_ALPHA_CUTOFF = float(os.getenv('TRELLIS2_ALPHA_CUTOFF', '0.5'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.join(os.path.dirname(__file__), 'media', 'generated'))
POLL_INTERVAL = int(os.getenv('TRELLIS2_POLL_INTERVAL', '10'))
BATCH_SIZE = int(os.getenv('TRELLIS2_BATCH_SIZE', '5'))

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


def has_pending_items(db) -> bool:
    """Check if there are items needing 3D generation (lightweight query)."""
    if db.library_items.find_one({'status': 'needs_3d'}, {'_id': 1}):
        return True
    if db.jobs.find_one({'status': 'needs_3d'}, {'_id': 1}):
        return True
    return False


# ---------------------------------------------------------------------------
# Supervisor: lightweight loop that spawns child processes
# ---------------------------------------------------------------------------

def run_supervisor(once: bool = False):
    """
    Supervisor loop. Never imports torch/trellis2 — stays tiny in memory.
    Spawns a fresh child process for each batch of GPU work.
    """
    db = get_db()

    logger.info("Trellis2 supervisor started")
    logger.info(f"MongoDB: {MONGODB_URI}/{MONGODB_DATABASE}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    try:
        while True:
            if not has_pending_items(db):
                if once:
                    logger.info("No items to process, exiting (--once mode)")
                    break
                time.sleep(POLL_INTERVAL)
                continue

            # Spawn a child process to handle the batch
            logger.info("Work found, spawning batch subprocess...")
            child = sp.Popen(
                [sys.executable, __file__, '--batch', '--batch-size', str(BATCH_SIZE)],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            child.wait()

            if child.returncode != 0:
                logger.warning(f"Batch subprocess exited with code {child.returncode}")
                # Back off on failure to avoid tight crash loops
                time.sleep(POLL_INTERVAL)
            else:
                logger.info("Batch subprocess completed successfully")

            if once:
                break

            # Brief pause before checking for more work
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Supervisor interrupted")


# ---------------------------------------------------------------------------
# Batch child: does the actual GPU work, then exits
# ---------------------------------------------------------------------------

def run_batch(batch_size: int):
    """
    Child process: acquire GPU lock, load pipeline, process items, exit.
    When this process exits, ALL GPU/CPU memory is reclaimed by the OS.
    """
    # Heavy imports only in the child process
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    import uuid
    import gc
    from PIL import Image

    sys.path.insert(0, TRELLIS2_REPO_PATH)

    db = get_db()
    gpu_lock = GPULock("trellis2_worker")

    # Gather pending items
    items = _get_pending_items(db, batch_size)
    if not items:
        logger.info("No items to process in batch")
        return

    logger.info(f"Batch: {len(items)} items to process")

    # Acquire GPU lock
    logger.info("Acquiring GPU lock...")
    if not gpu_lock.acquire(timeout=-1):
        logger.error("Failed to acquire GPU lock")
        return

    pipeline = None
    processed = 0

    try:
        # Load pipeline
        logger.info(f"Loading Trellis2 pipeline: {TRELLIS2_MODEL}")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS2_MODEL)
        pipeline.cuda()
        logger.info("Trellis2 pipeline loaded")

        for item in items:
            if not item.get('image_path') or not os.path.exists(item['image_path']):
                _mark_failed(db, item, f"Image not found: {item.get('image_path')}")
                logger.warning(f"Skipping {item['id']}: image not found")
                continue

            _mark_processing(db, item)
            gpu_lock.heartbeat()

            try:
                output_path = _generate_glb(pipeline, item['image_path'], item['seed'])
                _mark_completed(db, item, output_path)
                processed += 1
                logger.info(f"Completed {item['id']}: {item['description'][:50]} ({processed} total)")
            except Exception as e:
                _mark_failed(db, item, str(e))
                logger.error(f"Failed {item['id']}: {e}")
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.error("CUDA error, stopping batch")
                    break

            # Cleanup between items
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Batch complete: {processed} items processed")

    except Exception as e:
        logger.error(f"Batch failed: {e}")
        sys.exit(1)

    finally:
        # Release GPU lock. Pipeline cleanup happens automatically when process exits.
        gpu_lock.release()
        logger.info("GPU lock released, child process exiting")


# ---------------------------------------------------------------------------
# Shared helpers (used by batch child)
# ---------------------------------------------------------------------------

def _get_pending_items(db, batch_size: int):
    """Get items needing 3D generation from both collections."""
    items = []

    for item in db.library_items.find(
        {'status': 'needs_3d'},
        limit=batch_size
    ).sort('created_at', 1):
        is_capture = item.get('source') == 'capture'
        items.append({
            'collection': 'library_items',
            'id': item['_id'],
            'image_path': item.get('image_path'),
            'seed': item.get('seed', 42),
            'description': item.get('description', ''),
            'next_status': 'reconstructed' if is_capture else 'needs_embedding',
        })

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


def _mark_processing(db, item):
    db[item['collection']].update_one(
        {'_id': item['id']},
        {'$set': {
            'status': 'processing_3d',
            'started_3d_at': datetime.now(timezone.utc),
        }}
    )


def _mark_completed(db, item, output_path: str):
    update = {
        'status': item['next_status'],
        'asset_path': output_path,
        'asset_type': 'mesh',
        'completed_3d_at': datetime.now(timezone.utc),
    }
    if item['collection'] == 'jobs':
        update['output_path'] = output_path
        update['completed_at'] = datetime.now(timezone.utc)

    db[item['collection']].update_one(
        {'_id': item['id']},
        {'$set': update}
    )


def _mark_failed(db, item, error: str):
    db[item['collection']].update_one(
        {'_id': item['id']},
        {'$set': {
            'status': 'failed',
            'error': error,
            'failed_at': datetime.now(timezone.utc),
        }}
    )


def _generate_glb(pipeline, image_path: str, seed: int) -> str:
    """Generate a GLB mesh from an image. Returns output path."""
    import uuid
    import o_voxel
    from PIL import Image

    image = Image.open(image_path)
    logger.info(f"Generating mesh from {image_path} (seed={seed})")

    mesh = pipeline.run(
        image,
        seed=seed,
        pipeline_type=TRELLIS2_PIPELINE_TYPE,
    )[0]

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

    _patch_glb_alpha_mode(output_path, TRELLIS2_ALPHA_CUTOFF)
    return output_path


def _patch_glb_alpha_mode(path: str, alpha_cutoff: float = 0.5):
    """Patch GLB BLEND → MASK for correct depth buffer behavior."""
    import struct, json

    with open(path, 'rb') as f:
        header = f.read(12)
        magic, version, total_len = struct.unpack('<4sII', header)
        chunk_len = struct.unpack('<I', f.read(4))[0]
        chunk_type = f.read(4)
        json_bytes = f.read(chunk_len)
        rest = f.read()

    gltf = json.loads(json_bytes.decode('utf-8'))
    patched = False

    for mat in gltf.get('materials', []):
        if mat.get('alphaMode') == 'BLEND':
            mat['alphaMode'] = 'MASK'
            mat['alphaCutoff'] = alpha_cutoff
            patched = True

    if not patched:
        return

    new_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    padding = (4 - len(new_json) % 4) % 4
    new_json += b' ' * padding

    with open(path, 'wb') as f:
        new_total = 12 + 8 + len(new_json) + len(rest)
        f.write(struct.pack('<4sII', magic, version, new_total))
        f.write(struct.pack('<I', len(new_json)))
        f.write(chunk_type)
        f.write(new_json)
        f.write(rest)

    logger.info(f"Patched alpha mode: BLEND → MASK (cutoff={alpha_cutoff})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trellis2 3D generation worker")
    parser.add_argument("--once", action="store_true",
                        help="Process one batch and exit")
    parser.add_argument("--batch", action="store_true",
                        help="(internal) Run as batch child process")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Max items per batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    if args.batch:
        run_batch(batch_size=args.batch_size)
    else:
        run_supervisor(once=args.once)
