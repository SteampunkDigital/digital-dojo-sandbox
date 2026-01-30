#!/usr/bin/env python3
"""
Create a library from a text file of object descriptions.

Usage:
    python create_library.py --name "Household Items" --file categories.txt
    python create_library.py -n "Furniture" -f furniture.txt -v 5  # 5 variants
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Create a library from a text file of object descriptions"
    )
    parser.add_argument(
        '--name', '-n',
        required=True,
        help='Library name'
    )
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Text file with object descriptions (one per line)'
    )
    parser.add_argument(
        '--variants', '-v',
        type=int,
        default=int(os.getenv('LIBRARY_VARIANTS_PER_ITEM', 3)),
        help='Number of variants per item (default: 3)'
    )
    args = parser.parse_args()

    # Read descriptions from file
    text_file = Path(args.file)
    if not text_file.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    objects_text = text_file.read_text().strip()

    # Count non-empty lines
    lines = [line.strip() for line in objects_text.split('\n') if line.strip()]
    num_objects = len(lines)

    if num_objects == 0:
        print("Error: No object descriptions found in file")
        sys.exit(1)

    print(f"Found {num_objects} object descriptions")
    print(f"Will create {num_objects * args.variants} total variants ({args.variants} per item)")
    print()

    # Create library via database service
    from services.database import DatabaseService

    db = DatabaseService()
    try:
        db.connect()
    except Exception as e:
        print(f"Error: Failed to connect to database: {e}")
        sys.exit(1)

    library_id = db.create_library(
        name=args.name,
        source_text=objects_text,
        variants_per_item=args.variants
    )

    print(f"Library created successfully!")
    print(f"  Library ID: {library_id}")
    print(f"  Items created: {num_objects * args.variants}")
    print()
    print("Run the library worker to process items:")
    print("  python library_worker.py")
    print()
    print("Or process all available items and exit:")
    print("  python library_worker.py --once")


if __name__ == '__main__':
    main()
