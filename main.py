"""
Brain Tumor Detection using Symmetry Rules and DINOv3 Features

This script detects potential brain tumors by analyzing the symmetry of brain MRI images
using features extracted from the DINOv3 foundation model. The analysis is based on the
principle that healthy brain tissue exhibits mirror symmetry, while tumors create asymmetric
patterns.

Usage:
    Place input MRI images in the 'input/' folder, then run:
    python main.py

Output:
    Results (asymmetry maps and visualizations) are saved to the 'output/' folder.

Main Workflow:
    1. Load MRI image from input folder
    2. Extract DINOv3 features from original and horizontally flipped images
    3. Compute mirror correspondence and asymmetry patterns
    4. Generate visualizations and save results
"""

import torch
from pathlib import Path
from PIL import Image

# Import modular components
from config import (
    MODEL_NAME, RESIZE_SIZE, SEARCH_RADIUS, 
    INPUT_FOLDER, OUTPUT_FOLDER
)
from preprocessing import make_transform
from feature_extraction import load_model, encode_tokens
from symmetry_analysis import max_local_similarity_in_flipped_grid, compute_asymmetry_map
from visualization import (
    save_asymmetry_map,
    save_asymmetry_overlay,
    save_asymmetry_with_grid,
)


def setup_folders():
    """Create input and output folders if they don't exist."""
    Path(INPUT_FOLDER).mkdir(exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)


def find_input_images(input_folder: str = INPUT_FOLDER):
    """
    Find all image files in the input folder.
    
    Args:
        input_folder (str): Path to input folder
    
    Returns:
        list[str]: Sorted list of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    input_path = Path(input_folder)
    
    image_paths = [
        str(file) for file in sorted(input_path.iterdir())
        if file.suffix.lower() in image_extensions
    ]

    return image_paths


def analyze_brain_image(image_path: str, model, transform, device: str):
    """
    Perform complete symmetry-based brain tumor analysis on an MRI image.
    
    Args:
        image_path (str): Path to input MRI image
    
    Returns:
        dict: Analysis results including asymmetry map and metadata
    """
    # Load image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Extract features
    print("Extracting features from original image...")
    tokens, grid_hw, n_special, pixel_values = encode_tokens(
        model, image, transform, device=device
    )
    
    print("Extracting features from flipped image...")
    tokens_flipped, _, _, _ = encode_tokens(
        model, image_flipped, transform, device=device
    )
    
    gh, gw = grid_hw
    print(f"Feature grid: {gh}×{gw}, Special tokens: {n_special}")
    
    # Compute symmetry analysis
    print(f"Computing mirror correspondence (search radius: {SEARCH_RADIUS})...")
    max_sim, _ = max_local_similarity_in_flipped_grid(
        tokens=tokens,
        tokens_flipped=tokens_flipped,
        grid_hw=grid_hw,
        r=SEARCH_RADIUS,
        n_special=n_special,
        normalize=True,
        return_best_idx=True,
    )
    
    # Extract batch and compute asymmetry
    max_sim = max_sim[0].detach().cpu()  # Remove batch dimension
    asymmetry_map = compute_asymmetry_map(max_sim)
    
    # Convert to numpy for visualization
    max_sim_np = max_sim.numpy()
    asymmetry_np = asymmetry_map.numpy()
    
    print(f"Asymmetry range: [{asymmetry_np.min():.4f}, {asymmetry_np.max():.4f}]")
    
    return {
        "image": image,
        "max_similarity": max_sim_np,
        "asymmetry_map": asymmetry_np,
        "grid_hw": grid_hw,
        "image_path": image_path,
    }


def save_results(results: dict, output_folder: str = OUTPUT_FOLDER):
    """
    Save analysis visual outputs.
    
    Args:
        results (dict): Dictionary with analysis results
        output_folder (str): Path where to save results
    """
    output_root = Path(output_folder)
    output_root.mkdir(exist_ok=True)

    input_name = Path(results["image_path"]).stem
    output_path = output_root / input_name
    output_path.mkdir(exist_ok=True)
    
    # Save exactly the three visualization outputs (no interactive display)
    asymmetry_np = results["asymmetry_map"]
    image = results["image"]
    grid_hw = results["grid_hw"]

    heatmap_path = output_path / "heatmap.png"
    overlay_path = output_path / "overlay.png"
    overlay_grid_path = output_path / "overlay_with_grid.png"

    save_asymmetry_map(
        asymmetry_map=asymmetry_np,
        grid_dims=grid_hw,
        search_radius=SEARCH_RADIUS,
        output_path=str(heatmap_path),
    )
    save_asymmetry_overlay(
        image=image,
        asymmetry_map=asymmetry_np,
        output_path=str(overlay_path),
        resize_size=RESIZE_SIZE,
        title="Asymmetry Map Overlay",
    )
    save_asymmetry_with_grid(
        image=image,
        asymmetry_map=asymmetry_np,
        grid_dims=grid_hw,
        output_path=str(overlay_grid_path),
        resize_size=RESIZE_SIZE,
    )

    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved overlay with grid: {overlay_grid_path}")


def main():
    """Main entry point for the symmetry-based brain tumor detection."""
    print("=" * 60)
    print("Symmetry-based Brain Tumor Detection using DINOv3")
    print("=" * 60)
    
    # Setup folders
    setup_folders()
    
    image_paths = find_input_images()
    if not image_paths:
        print(f"\nError: No images found in {INPUT_FOLDER}/")
        print("Please place an MRI image in the 'input' folder and try again.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME, device=device)
    transform = make_transform(resize_size=RESIZE_SIZE)

    print(f"Found {len(image_paths)} image(s) in {INPUT_FOLDER}/")
    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] Processing {Path(image_path).name}")
        results = analyze_brain_image(image_path, model=model, transform=transform, device=device)
        save_results(results)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the 'output' folder for per-image result folders.")
    print("=" * 60)


if __name__ == "__main__":
    main()
