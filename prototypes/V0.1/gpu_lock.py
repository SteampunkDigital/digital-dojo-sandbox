"""
MongoDB-based distributed GPU lock for cross-environment coordination.
Zero ML dependencies - only needs pymongo and python-dotenv.
"""
import os
import time
import socket
from datetime import datetime, timezone

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'digital_dojo')
LOCK_TIMEOUT_SECONDS = int(os.getenv('GPU_LOCK_TIMEOUT', '300'))  # 5 minutes
POLL_INTERVAL = 5  # seconds between acquire attempts


class GPULock:
    """
    Distributed GPU mutex using MongoDB.

    Usage:
        lock = GPULock("sd35_worker")
        with lock:
            # GPU is exclusively yours
            do_gpu_work()

    Or manually:
        lock = GPULock("trellis2_worker")
        if lock.acquire(timeout=60):
            try:
                do_gpu_work()
            finally:
                lock.release()
    """

    def __init__(self, holder_name: str, db_uri: str = None, db_name: str = None):
        self.holder_name = f"{holder_name}_{socket.gethostname()}_{os.getpid()}"
        uri = db_uri or MONGODB_URI
        name = db_name or MONGODB_DATABASE
        self._client = MongoClient(uri)
        self._db = self._client[name]
        self._collection = self._db['gpu_lock']
        self._held = False

    def acquire(self, timeout: int = 300) -> bool:
        """
        Try to acquire the GPU lock. Blocks until acquired or timeout.

        Args:
            timeout: Max seconds to wait. 0 = try once, -1 = wait forever.

        Returns:
            True if lock acquired, False if timed out.
        """
        start = time.time()
        while True:
            now = datetime.now(timezone.utc)

            # Try to acquire: either no lock exists, or lock is stale
            result = self._collection.find_one_and_update(
                {
                    '_id': 'gpu',
                    '$or': [
                        {'holder': None},
                        {'heartbeat': {'$lt': datetime.fromtimestamp(
                            time.time() - LOCK_TIMEOUT_SECONDS, tz=timezone.utc
                        )}}
                    ]
                },
                {
                    '$set': {
                        'holder': self.holder_name,
                        'acquired_at': now,
                        'heartbeat': now,
                    }
                },
                upsert=False,
                return_document=True
            )

            if result and result.get('holder') == self.holder_name:
                self._held = True
                return True

            # If document doesn't exist yet, create it
            try:
                self._collection.insert_one({
                    '_id': 'gpu',
                    'holder': self.holder_name,
                    'acquired_at': now,
                    'heartbeat': now,
                })
                self._held = True
                return True
            except Exception:
                pass  # Document already exists, another worker got it

            # Check timeout
            if timeout == 0:
                return False
            if timeout > 0 and (time.time() - start) >= timeout:
                return False

            # Wait and retry
            holder = self.get_holder()
            elapsed = int(time.time() - start)
            print(f"[GPULock] Waiting for GPU (held by {holder}, {elapsed}s elapsed)...")
            time.sleep(POLL_INTERVAL)

    def release(self):
        """Release the GPU lock."""
        self._collection.find_one_and_update(
            {'_id': 'gpu', 'holder': self.holder_name},
            {'$set': {'holder': None, 'heartbeat': datetime.now(timezone.utc)}}
        )
        self._held = False

    def heartbeat(self):
        """Update heartbeat timestamp. Call periodically during long operations."""
        self._collection.update_one(
            {'_id': 'gpu', 'holder': self.holder_name},
            {'$set': {'heartbeat': datetime.now(timezone.utc)}}
        )

    def get_holder(self) -> str:
        """Get the current lock holder name, or None."""
        doc = self._collection.find_one({'_id': 'gpu'})
        if doc:
            return doc.get('holder')
        return None

    def is_held(self) -> bool:
        """Check if GPU is currently locked by anyone."""
        holder = self.get_holder()
        return holder is not None

    def force_release(self):
        """Force-release the lock regardless of holder. Use with caution."""
        self._collection.update_one(
            {'_id': 'gpu'},
            {'$set': {'holder': None}}
        )
        self._held = False

    def __enter__(self):
        if not self.acquire(timeout=-1):
            raise RuntimeError("Failed to acquire GPU lock")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __del__(self):
        if self._held:
            self.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Lock utility")
    parser.add_argument("action", choices=["status", "release", "test"],
                        help="status: show lock state, release: force release, test: acquire and release")
    args = parser.parse_args()

    if args.action == "status":
        lock = GPULock("cli_check")
        doc = lock._collection.find_one({'_id': 'gpu'})
        if doc and doc.get('holder'):
            print(f"GPU locked by: {doc['holder']}")
            print(f"Acquired at:  {doc.get('acquired_at')}")
            print(f"Heartbeat:    {doc.get('heartbeat')}")
        else:
            print("GPU lock is free")

    elif args.action == "release":
        lock = GPULock("cli_release")
        lock.force_release()
        print("GPU lock force-released")

    elif args.action == "test":
        lock = GPULock("cli_test")
        print("Acquiring GPU lock...")
        if lock.acquire(timeout=10):
            print(f"Acquired by: {lock.holder_name}")
            time.sleep(2)
            lock.release()
            print("Released")
        else:
            print("Failed to acquire (timeout)")
