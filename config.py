"""
Configuration constants for the Symmetry-based Brain Tumor Detection system.
"""

# DINOv3 Model Configuration
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RESIZE_SIZE = 518  # Input image size for DINOv3

# Feature Extraction Parameters
PATCH_SIZE = 16  # DINOv3 ViT-B/16 patch size

# Symmetry Analysis Parameters
SEARCH_RADIUS = 5  # Neighborhood radius in patch units for mirror correspondence search

# Visualization Parameters
OVERLAY_ALPHA = 0.45  # Alpha transparency for asymmetry map overlay
PATCH_GRID_ALPHA = 0.15  # Alpha transparency for patch grid lines

# Device Configuration
DEVICE = "cuda"  # Use "cuda" for GPU, "cpu" for CPU (auto-selected in code)

# Path Configuration
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"
