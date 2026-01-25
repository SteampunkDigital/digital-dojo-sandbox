"""
Utility to reset jobs for retry.

Usage:
    python reset_jobs.py              # Reset failed jobs to retry from splat stage
    python reset_jobs.py --all        # Reset ALL jobs (including completed) to splat stage
    python reset_jobs.py --from mask  # Reset to retry from mask stage
    python reset_jobs.py --from image # Reset to retry from image stage (full restart)
    python reset_jobs.py --status     # Show job status counts
    python reset_jobs.py --clear      # Delete all jobs (use with caution)
"""

import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Reset jobs for retry")
    parser.add_argument("--from", dest="from_stage", default="splat",
                        choices=["image", "mask", "splat"],
                        help="Stage to restart from (default: splat)")
    parser.add_argument("--all", action="store_true",
                        help="Reset ALL jobs (including completed), not just failed")
    parser.add_argument("--status", action="store_true",
                        help="Show job status counts")
    parser.add_argument("--clear", action="store_true",
                        help="Delete ALL jobs (use with caution)")
    parser.add_argument("--list-failed", action="store_true",
                        help="List failed jobs with errors")
    args = parser.parse_args()

    from services import db

    # Connect to database
    try:
        db.connect()
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return

    if args.status:
        # Show status counts
        print("\nJob Status Counts:")
        print("-" * 30)
        for stage in ["pending", "needs_mask", "needs_splat", "completed", "failed"]:
            count = db.count_jobs_by_stage(stage)
            print(f"  {stage:15} : {count}")
        print("-" * 30)
        total = sum(db.count_jobs_by_stage(s) for s in ["pending", "needs_mask", "needs_splat", "completed", "failed"])
        print(f"  {'Total':15} : {total}")
        return

    if args.list_failed:
        # List failed jobs
        failed = db.get_failed_jobs()
        if not failed:
            print("\nNo failed jobs.")
            return

        print(f"\nFailed Jobs ({len(failed)}):")
        print("-" * 60)
        for job in failed:
            print(f"  ID: {job['_id']}")
            print(f"    Prompt: {job.get('prompt', 'N/A')[:50]}...")
            print(f"    Error: {job.get('error', 'No error message')}")
            print(f"    Image: {job.get('image_path', 'None')}")
            print(f"    Mask: {job.get('mask_path', 'None')}")
            print()
        return

    if args.clear:
        # Delete all jobs
        confirm = input("Are you sure you want to delete ALL jobs? (yes/no): ")
        if confirm.lower() == "yes":
            count = db.clear_all_jobs()
            print(f"Deleted {count} jobs")
        else:
            print("Cancelled")
        return

    # Reset jobs
    stage_map = {
        "image": "pending",
        "mask": "needs_mask",
        "splat": "needs_splat"
    }
    reset_to = stage_map[args.from_stage]

    if args.all:
        # Reset ALL jobs (completed + failed)
        count = db.reset_all_jobs(reset_to_stage=reset_to)
        print(f"Reset {count} jobs to '{reset_to}'. Run the worker to process them.")
    else:
        # Reset only failed jobs
        failed_count = db.count_jobs_by_stage("failed")
        if failed_count == 0:
            print("No failed jobs to reset. Use --all to reset completed jobs too.")
            return

        print(f"\nResetting {failed_count} failed jobs to '{reset_to}'...")
        count = db.reset_failed_jobs(reset_to_stage=reset_to)
        print(f"Reset {count} jobs. Run the worker to process them.")

    db.close()


if __name__ == "__main__":
    main()
