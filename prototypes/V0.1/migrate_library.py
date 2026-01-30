#!/usr/bin/env python3
"""
Migrate existing library items to the new review workflow structure.

Adds variant_group_id to items that don't have it, and optionally
resets items to needs_review status for human review.

Usage:
    python migrate_library.py                    # Add variant_group_id only
    python migrate_library.py --reset-to-review  # Also reset completed items to needs_review
    python migrate_library.py --dry-run          # Show what would change
"""

import argparse
import hashlib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services.database import DatabaseService


def main():
    parser = argparse.ArgumentParser(description="Migrate library items to review workflow")
    parser.add_argument('--reset-to-review', action='store_true',
                        help='Reset ALL items with images back to needs_review for re-review')
    parser.add_argument('--reset-failed', action='store_true',
                        help='Reset only FAILED items back to needs_review')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without making changes')
    parser.add_argument('--library-id', '-l',
                        help='Only migrate specific library (default: all)')
    args = parser.parse_args()

    db = DatabaseService()
    db.connect()

    # Find items without variant_group_id
    query = {"variant_group_id": {"$exists": False}}
    if args.library_id:
        query["library_id"] = args.library_id

    items = list(db.library_items.find(query))
    print(f"Found {len(items)} items without variant_group_id")

    if not items:
        print("Nothing to migrate!")
        return

    # Group by description to show summary
    descriptions = {}
    for item in items:
        desc = item["description"]
        if desc not in descriptions:
            descriptions[desc] = []
        descriptions[desc].append(item)

    print(f"\nUnique descriptions: {len(descriptions)}")

    # Migrate items
    migrated = 0
    reset_count = 0

    for desc, desc_items in descriptions.items():
        # Generate variant_group_id from description (same logic as create_library)
        variant_group_id = hashlib.md5(desc.encode()).hexdigest()[:12]

        print(f"\n{desc[:60]}...")
        print(f"  Group ID: {variant_group_id}")
        print(f"  Variants: {len(desc_items)}")

        for item in desc_items:
            item_id = item["_id"]
            current_status = item.get("status", "unknown")
            has_image = bool(item.get("image_path"))

            update = {"variant_group_id": variant_group_id}

            # Optionally reset to needs_review
            should_reset = False
            if args.reset_to_review and has_image and current_status not in ["pending", "needs_review"]:
                should_reset = True
            elif args.reset_failed and has_image and current_status == "failed":
                should_reset = True

            if should_reset:
                update["status"] = "needs_review"
                reset_count += 1
                print(f"    [{item_id[:8]}] {current_status} → needs_review")
            else:
                print(f"    [{item_id[:8]}] {current_status} (adding group_id only)")

            if not args.dry_run:
                db.library_items.update_one(
                    {"_id": item_id},
                    {"$set": update}
                )

            migrated += 1

    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {migrated} items")
    if args.reset_to_review:
        print(f"{'Would reset' if args.dry_run else 'Reset'} {reset_count} items to needs_review")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")
    else:
        print("\nMigration complete! Open the review UI to see grouped items:")
        libraries = db.list_libraries()
        for lib in libraries[:5]:
            print(f"  http://localhost:5000/library-review/{lib['_id']}")


if __name__ == '__main__':
    main()
