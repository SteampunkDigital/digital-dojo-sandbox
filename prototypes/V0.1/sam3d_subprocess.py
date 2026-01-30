#!/usr/bin/env python3
"""
SAM3D subprocess runner - runs a single 3D generation in isolation.

This script is called by the library worker to generate a single mesh/splat.
Running in a subprocess ensures complete GPU memory cleanup after each job.

Usage:
    python sam3d_subprocess.py <image_path> <mask_path> <output_path> [--format mesh|splat] [--seed 42]
"""

import argparse
import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Generate 3D from image+mask")
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('mask_path', help='Path to mask image')
    parser.add_argument('output_path', help='Path for output file')
    parser.add_argument('--format', choices=['mesh', 'splat'], default='mesh',
                        help='Output format (default: mesh)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Verify inputs exist
    if not os.path.exists(args.image_path):
        print(f"ERROR: Image not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.mask_path):
        print(f"ERROR: Mask not found: {args.mask_path}", file=sys.stderr)
        sys.exit(1)

    # Import and run SAM3D
    from services.orchestrator import GPUOrchestrator

    orchestrator = GPUOrchestrator()

    try:
        orchestrator.load_sam3d()

        if args.format == 'mesh':
            output = orchestrator.generate_mesh(
                args.image_path,
                args.mask_path,
                seed=args.seed
            )
        else:
            output = orchestrator.generate_splat(
                args.image_path,
                args.mask_path,
                seed=args.seed
            )

        # Move/rename output to requested path
        import shutil
        if output != args.output_path:
            shutil.move(output, args.output_path)

        print(f"SUCCESS: {args.output_path}")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        orchestrator.clear_gpu_memory()


if __name__ == '__main__':
    main()
