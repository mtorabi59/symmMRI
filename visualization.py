"""
Visualization module for brain MRI asymmetry analysis results.
Creates publication-quality figures for asymmetry maps and overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _upsample_asymmetry_map(asymmetry_map: np.ndarray, resize_size: int) -> np.ndarray:
    asym_img = Image.fromarray(
        (asymmetry_map / (asymmetry_map.max() + 1e-6) * 255).astype(np.uint8)
    )
    asym_up = asym_img.resize((resize_size, resize_size), resample=Image.BILINEAR)
    return np.array(asym_up).astype(np.float32) / 255.0


def plot_input_image(image: Image.Image, title: str = "Input image"):
    """
    Display the input image.
    
    Args:
        image (Image.Image): PIL Image object
        title (str): Plot title
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.close()


def plot_asymmetry_map(asymmetry_map: np.ndarray, grid_dims: tuple, search_radius: int):
    """
    Display the computed asymmetry map as a heatmap.
    
    Higher values (red) indicate more asymmetric regions.
    Lower values (purple) indicate symmetric regions.
    
    Args:
        asymmetry_map (np.ndarray): 2D asymmetry values, shape (gh, gw)
        grid_dims (tuple): Patch grid dimensions (gh, gw)
        search_radius (int): Search radius used in analysis
    """
    gh, gw = grid_dims
    plt.figure(figsize=(5, 4))
    plt.imshow(asymmetry_map, cmap="plasma", interpolation="nearest")
    plt.title(f"Asymmetry (patch grid {gh}×{gw}), r={search_radius}")
    plt.axis("off")
    plt.colorbar(label="Asymmetry Score")
    plt.tight_layout()
    plt.close()


def plot_asymmetry_overlay(
    image: Image.Image,
    asymmetry_map: np.ndarray,
    resize_size: int = 518,
    alpha: float = 0.45,
    title: str = "Asymmetry Overlay"
):
    """
    Display the asymmetry map overlaid on the original image.
    
    This visualization helps identify asymmetric regions in the context
    of the original anatomy.
    
    Args:
        image (Image.Image): Original PIL Image
        asymmetry_map (np.ndarray): 2D asymmetry values
        resize_size (int): Size to resize image for display
        alpha (float): Transparency of overlay (0-1)
        title (str): Plot title
    """
    img_resized = image.resize((resize_size, resize_size), Image.BILINEAR)
    asym_up_np = _upsample_asymmetry_map(asymmetry_map, resize_size)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_resized)
    plt.imshow(asym_up_np, cmap="plasma", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.close()
    
    return asym_up_np


def plot_asymmetry_with_grid(
    image: Image.Image,
    asymmetry_map: np.ndarray,
    grid_dims: tuple,
    resize_size: int = 518,
    alpha: float = 0.45,
    grid_alpha: float = 0.15
):
    """
    Display asymmetry map overlay with patch grid lines.
    
    The grid lines show the DINOv3 patch boundaries for reference.
    
    Args:
        image (Image.Image): Original PIL Image
        asymmetry_map (np.ndarray): 2D asymmetry values
        grid_dims (tuple): Patch grid dimensions (gh, gw)
        resize_size (int): Size to resize image for display
        alpha (float): Transparency of asymmetry overlay
        grid_alpha (float): Transparency of grid lines
    """
    gh, gw = grid_dims
    
    img_resized = image.resize((resize_size, resize_size), Image.BILINEAR)
    asym_up_np = _upsample_asymmetry_map(asymmetry_map, resize_size)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_resized)
    plt.imshow(asym_up_np, cmap="plasma", alpha=alpha)
    
    # Draw patch grid
    step_h = resize_size // gh
    step_w = resize_size // gw
    for i in range(0, resize_size, step_h):
        plt.axhline(i, color="white", alpha=grid_alpha, linewidth=1)
    for j in range(0, resize_size, step_w):
        plt.axvline(j, color="white", alpha=grid_alpha, linewidth=1)
    
    plt.title("Asymmetry Overlay + Patch Grid")
    plt.axis("off")
    plt.tight_layout()
    plt.close()


def save_asymmetry_map(asymmetry_map: np.ndarray, grid_dims: tuple, search_radius: int, output_path: str):
    gh, gw = grid_dims
    plt.figure(figsize=(5, 4))
    plt.imshow(asymmetry_map, cmap="plasma", interpolation="nearest")
    plt.title(f"Asymmetry (patch grid {gh}×{gw}), r={search_radius}")
    plt.axis("off")
    plt.colorbar(label="Asymmetry Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_asymmetry_overlay(
    image: Image.Image,
    asymmetry_map: np.ndarray,
    output_path: str,
    resize_size: int = 518,
    alpha: float = 0.45,
    title: str = "Asymmetry Map Overlay",
):
    img_resized = image.resize((resize_size, resize_size), Image.BILINEAR)
    asym_up_np = _upsample_asymmetry_map(asymmetry_map, resize_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_resized)
    plt.imshow(asym_up_np, cmap="plasma", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_asymmetry_with_grid(
    image: Image.Image,
    asymmetry_map: np.ndarray,
    grid_dims: tuple,
    output_path: str,
    resize_size: int = 518,
    alpha: float = 0.45,
    grid_alpha: float = 0.15,
):
    gh, gw = grid_dims
    img_resized = image.resize((resize_size, resize_size), Image.BILINEAR)
    asym_up_np = _upsample_asymmetry_map(asymmetry_map, resize_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_resized)
    plt.imshow(asym_up_np, cmap="plasma", alpha=alpha)

    step_h = resize_size // gh
    step_w = resize_size // gw
    for i in range(0, resize_size, step_h):
        plt.axhline(i, color="white", alpha=grid_alpha, linewidth=1)
    for j in range(0, resize_size, step_w):
        plt.axvline(j, color="white", alpha=grid_alpha, linewidth=1)

    plt.title("Asymmetry Overlay + Patch Grid")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
