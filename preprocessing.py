"""
Image preprocessing utilities for DINOv3 feature extraction.
Handles image loading, normalization, and transformation.
"""

from torchvision.transforms import v2
import torch


def make_transform(resize_size: int = 518):
    """
    Create a torchvision transform pipeline for image preprocessing.
    
    Follows the Meta DINOv3 preprocessing standards:
    - Resize to resize_size x resize_size
    - Convert to float32
    - Normalize with ImageNet statistics
    
    Args:
        resize_size (int): Target image size (default: 518 for DINOv3)
    
    Returns:
        v2.Compose: Transform pipeline
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
