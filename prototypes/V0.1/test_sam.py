"""Quick test for SAM (Segment Anything Model) integration"""

import sys
import os

# Add SAM repo to path
SAM_REPO = "/mnt/c/Users/david/Documents/GitHub/sam3"
sys.path.insert(0, SAM_REPO)

print("=" * 50)
print("SAM Integration Test")
print("=" * 50)

# Check checkpoint exists
checkpoint_path = os.path.join(SAM_REPO, "sam_vit_h_4b8939.pth")
print(f"\n1. Checking checkpoint: {checkpoint_path}")
if os.path.exists(checkpoint_path):
    size_gb = os.path.getsize(checkpoint_path) / (1024**3)
    print(f"   [OK] Found ({size_gb:.2f} GB)")
else:
    print("   [FAIL] NOT FOUND")
    sys.exit(1)

# Try importing SAM
print("\n2. Importing segment_anything...")
try:
    from segment_anything import sam_model_registry, SamPredictor
    print("   [OK] Import successful")
except ImportError as e:
    print(f"   [FAIL] Import failed: {e}")
    print("\n   Try: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

# Check CUDA
print("\n3. Checking PyTorch/CUDA...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("   [WARN] CUDA not available, using CPU (will be slow)")
        device = "cpu"
except ImportError:
    print("   [FAIL] PyTorch not installed")
    sys.exit(1)

# Load SAM model
print("\n4. Loading SAM model (vit_h)...")
try:
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("   [OK] SAM loaded successfully")
except Exception as e:
    print(f"   [FAIL] Failed to load SAM: {e}")
    sys.exit(1)

# Test with a simple image
print("\n5. Testing inference...")
try:
    import numpy as np
    from PIL import Image

    # Create a simple test image (red square on white background)
    test_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    test_img[128:384, 128:384] = [255, 0, 0]  # Red square in center

    # Set image
    predictor.set_image(test_img)

    # Predict with center point
    center_point = np.array([[256, 256]])
    point_labels = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=point_labels,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    fg_pixels = np.sum(best_mask)
    total_pixels = best_mask.size

    print(f"   [OK] Inference successful")
    print(f"   Mask shape: {best_mask.shape}")
    print(f"   Foreground: {fg_pixels} pixels ({100*fg_pixels/total_pixels:.1f}%)")
    print(f"   Best score: {scores.max():.3f}")

    # The red square is 256x256 = 65536 pixels
    expected_fg = 256 * 256
    if 0.5 * expected_fg < fg_pixels < 1.5 * expected_fg:
        print("   [OK] Mask size looks reasonable for the test square")

except Exception as e:
    print(f"   [FAIL] Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
print("\n6. Cleaning up GPU memory...")
del predictor
del sam
if device == "cuda":
    torch.cuda.empty_cache()
print("   [OK] Done")

print("\n" + "=" * 50)
print("[OK] SAM is working correctly!")
print("=" * 50)
