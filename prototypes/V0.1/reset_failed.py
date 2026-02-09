#!/usr/bin/env python3
"""Reset failed library items back to their previous stage for retry."""

import argparse
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services.database import DatabaseService


def main():
    parser = argparse.ArgumentParser(description="Reset failed library items")
    parser.add_argument('--stage', default='needs_3d',
                        choices=['pending', 'needs_3d', 'needs_embedding'],
                        help='Stage to reset to (default: needs_3d)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be reset without changing')
    args = parser.parse_args()

    db = DatabaseService()
    db.connect()

    # Find failed items
    failed = list(db.library_items.find({'status': 'failed'}))
    print(f"Found {len(failed)} failed items")

    if not failed:
        print("Nothing to reset")
        return

    # Group by error type
    error_counts = {}
    for item in failed:
        err = item.get('error', 'Unknown')[:50]
        error_counts[err] = error_counts.get(err, 0) + 1

    print("\nFailure reasons:")
    for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4} - {err}")

    if args.dry_run:
        print(f"\nWould reset {len(failed)} items to '{args.stage}'")
        return

    # Reset items
    result = db.library_items.update_many(
        {'status': 'failed'},
        {'$set': {'status': args.stage, 'error': None}}
    )

    print(f"\nReset {result.modified_count} items to '{args.stage}'")
    print("Run 'python library_worker.py' to retry")


if __name__ == '__main__':
    main()
