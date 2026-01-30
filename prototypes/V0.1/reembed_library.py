#!/usr/bin/env python3
"""
Re-embed Library Items

Regenerates embeddings for existing library items using the configured
embedding backend (OpenCLIP or FLAIR).

Use this when:
- Switching embedding backends (e.g., OpenCLIP → FLAIR)
- Updating to a new model version
- Fixing corrupted embeddings

Usage:
    # Re-embed all ready items using FLAIR
    EMBEDDING_MODEL=flair python reembed_library.py

    # Re-embed specific library
    python reembed_library.py --library-id abc123

    # Dry run (show what would be updated)
    python reembed_library.py --dry-run

    # Force re-embed even if embeddings exist
    python reembed_library.py --force
"""

import argparse
import os
import sys
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed library items with current embedding model"
    )
    parser.add_argument(
        '--library-id', '-l',
        help='Only re-embed items from specific library'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-embed even if embeddings already exist'
    )
    parser.add_argument(
        '--status',
        default='ready',
        help='Only re-embed items with this status (default: ready)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of items to process before reporting progress'
    )
    parser.add_argument(
        '--backend',
        choices=['openclip', 'flair'],
        help='Override embedding backend (default: from EMBEDDING_MODEL env)'
    )
    args = parser.parse_args()

    # Import services
    from services.database import DatabaseService
    from services.embedding_service import EmbeddingService, reset_embedding_service

    # Connect to database
    db = DatabaseService()
    try:
        db.connect()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    # Initialize embedding service with specified backend
    if args.backend:
        os.environ["EMBEDDING_MODEL"] = args.backend

    reset_embedding_service()  # Clear any cached service
    embedding_service = EmbeddingService(backend=args.backend)

    logger.info("=" * 60)
    logger.info("Re-embedding Library Items")
    logger.info(f"Backend: {embedding_service.backend_name}")
    logger.info(f"Status filter: {args.status}")
    logger.info(f"Force: {args.force}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Build query
    query = {"status": args.status}
    if args.library_id:
        query["library_id"] = args.library_id
        logger.info(f"Filtering to library: {args.library_id}")

    # If not forcing, only get items missing embeddings
    if not args.force:
        query["$or"] = [
            {"text_embedding": None},
            {"text_embedding": {"$exists": False}},
            {"image_embedding": None},
            {"image_embedding": {"$exists": False}}
        ]

    # Count items
    total_count = db.library_items.count_documents(query)
    logger.info(f"Found {total_count} items to process")

    if total_count == 0:
        logger.info("Nothing to re-embed!")
        if not args.force:
            logger.info("Use --force to re-embed items that already have embeddings")
        return

    if args.dry_run:
        # Show sample of items
        sample = list(db.library_items.find(query).limit(10))
        logger.info("\nSample of items to re-embed:")
        for item in sample:
            has_text = bool(item.get("text_embedding"))
            has_image = bool(item.get("image_embedding"))
            logger.info(f"  [{item['_id'][:8]}] {item['description'][:50]}... "
                       f"(text: {'✓' if has_text else '✗'}, image: {'✓' if has_image else '✗'})")
        if total_count > 10:
            logger.info(f"  ... and {total_count - 10} more")
        logger.info("\nRun without --dry-run to apply changes")
        return

    # Load embedding model
    logger.info("Loading embedding model...")
    embedding_service.load()
    logger.info(f"Model loaded on {embedding_service._backend.device}")

    # Process items
    processed = 0
    errors = 0
    cursor = db.library_items.find(query)

    for item in cursor:
        item_id = item["_id"]
        description = item["description"]
        image_path = item.get("image_path")

        try:
            # Generate text embedding (pure text for text-based search)
            text_embedding = embedding_service.encode_text(description)

            # Generate image embedding with text description
            # Combines visual features with semantic text for better retrieval
            image_embedding = None
            if image_path and os.path.exists(image_path):
                # Use 70% image, 30% text for image embeddings
                image_embedding = embedding_service.encode_image_with_text(
                    image_path, description, image_weight=0.7
                )

            # Update database
            update = {"text_embedding": text_embedding}
            if image_embedding:
                update["image_embedding"] = image_embedding

            db.library_items.update_one(
                {"_id": item_id},
                {"$set": update}
            )

            processed += 1

            # Progress report
            if processed % args.batch_size == 0:
                logger.info(f"Progress: {processed}/{total_count} "
                           f"({100*processed/total_count:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to embed [{item_id[:8]}]: {e}")
            errors += 1

    # Final report
    logger.info("=" * 60)
    logger.info(f"Re-embedding complete!")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Backend: {embedding_service.backend_name}")
    logger.info("=" * 60)

    # Cleanup
    embedding_service.unload()
    db.close()


if __name__ == '__main__':
    main()
