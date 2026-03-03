# Symmetry-based Brain Tumor Detection using DINOv3

A novel approach for detecting brain tumors in MRI images by leveraging symmetry rules applied to features extracted from the DINOv3 foundation model.

## Overview

This project proposes a symmetry-aware analysis pipeline for identifying abnormal regions (potential tumors) in brain MRI scans. The core principle is that **healthy brain tissue exhibits mirror symmetry**, while tumors create **asymmetric patterns** that can be detected through feature-level comparison.

### Key Features

- **Foundation Model Features**: Uses DINOv3 (a self-supervised vision transformer) to extract robust, semantic features from MRI images
- **Symmetry Analysis**: Compares features from original and horizontally flipped images to compute asymmetry maps
- **Modular Architecture**: Clean, well-organized code with separate modules for each functionality
- **Publication-Ready Visualization**: Multiple visualization options for asymmetry maps and overlays
- **Easy to Use**: Simple command-line interface with automatic input/output folder management

## Architecture & Design

```
symmMRI/
├── main.py                 # Main entry point - orchestrates the analysis pipeline
├── config.py              # Configuration constants and hyperparameters
├── preprocessing.py       # Image preprocessing and normalization
├── feature_extraction.py  # DINOv3 model loading and token extraction
├── symmetry_analysis.py   # Mirror correspondence and asymmetry computation
├── visualization.py       # Result visualization and plotting
├── requirements.txt       # Python dependencies
├── input/                 # Place your MRI images here
└── output/                # Analysis results are saved here
```

### Module Descriptions

**[config.py](config.py)** - Configuration Management
- Centralized configuration for model, parameters, and paths
- Easy to customize without modifying core code

**[preprocessing.py](preprocessing.py)** - Image Preprocessing
- Implements DINOv3-standard preprocessing (resizing, normalization)
- Uses torchvision transforms for pipeline composition

**[feature_extraction.py](feature_extraction.py)** - Feature Extraction
- Model loading from Hugging Face
- Efficient token extraction from DINOv3
- Automatic patch grid inference from model config

**[symmetry_analysis.py](symmetry_analysis.py)** - Symmetry Analysis
- Core algorithm for computing mirror correspondence
- Efficient neighborhood search using padded tensors
- Asymmetry map computation

**[visualization.py](visualization.py)** - Result Visualization
- Multiple visualization options (heatmaps, overlays, grid visualizations)
- Publication-quality figures with proper colormaps and transparency

**[main.py](main.py)** - Main Pipeline
- Orchestrates the complete analysis workflow
- Handles input image discovery and output saving
- Provides informative console output

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU acceleration, optional)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd symmMRI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

1. **Prepare your input**
   - Place your brain MRI image(s) in the `input/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

2. **Run the analysis**
   ```bash
   python main.py
   ```

3. **View results**
   - Results are automatically saved to the `output/` folder
   - Check visualization plots that appear on screen
   - View saved PNG images and analysis statistics

### Configuration

Edit `config.py` to customize:

```python
# Model Configuration
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # DINOv3 model variant
RESIZE_SIZE = 518                                         # Input image size
SEARCH_RADIUS = 5                                         # Neighborhood search radius

# Visualization
OVERLAY_ALPHA = 0.45      # Transparency of asymmetry overlay
PATCH_GRID_ALPHA = 0.15   # Transparency of patch grid lines

# Paths
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"
```

### Output Files

For each processed image, the following files are generated:

- **`{image_name}_asymmetry_map.png`** - Asymmetry heatmap at patch grid resolution
- **`{image_name}_asymmetry_overlay.png`** - Upsampled asymmetry map overlaid on original image
- **`{image_name}_analysis_results.txt`** - Statistical summary of asymmetry analysis

## Method Details

### Symmetry Analysis Algorithm

1. **Feature Extraction**
   - Load brain MRI image (RGB conversion if needed)
   - Horizontally flip the image
   - Extract DINOv3 tokens from both original and flipped images
   - DINOv3 produces patch-level features in a grid layout (typically 32×32 patches for 518×518 input)

2. **Mirror Correspondence Search**
   - For each patch position (y, x) in the original image
   - Search a local neighborhood (radius r) in the flipped image
   - Compute cosine similarity between feature vectors
   - Track the maximum similarity found

3. **Asymmetry Computation**
   - Asymmetry score = 1 - max_similarity (range: [0, 1])
   - Higher values indicate more asymmetric regions
   - Potential tumor locations show high asymmetry

4. **Visualization**
   - Generate asymmetry heatmaps (plasma colormap)
   - Create overlays on original images for anatomical context
   - Visualize patch grid boundaries for reference

### Mathematical Foundation

For patch position $(y, x)$ in the original image, the asymmetry is computed as:

$$\text{Asymmetry}(y,x) = 1 - \max_{(dy,dx) \in N(r)} \text{CosSim}(\mathbf{f}_o(y,x), \mathbf{f}_f(y+dy, x+dx))$$

where:
- $\mathbf{f}_o$ = original image features from DINOv3
- $\mathbf{f}_f$ = flipped image features from DINOv3
- $N(r)$ = neighborhood of radius $r$ around mirrored location
- $\text{CosSim}$ = cosine similarity between L2-normalized features

## Advantages of This Approach

1. **Self-supervised Foundation Model**: DINOv3 requires no fine-tuning on labeled brain tumor data
2. **Symmetry Principle**: Leverages well-established biological principle (brain symmetry)
3. **Feature-level Analysis**: Works at semantic feature level, not pixel level
4. **Interpretable**: Asymmetry maps directly correspond to image regions
5. **Modular Design**: Easy to extend or modify individual components

## Future Work

- [ ] Integration with segmentation networks for tumor boundary delineation
- [ ] Quantitative evaluation on benchmark datasets (BraTS, etc.)
- [ ] Support for 3D volumetric analysis
- [ ] Multi-modal MRI support (T1, T2, FLAIR)
- [ ] Clinical validation and comparison with radiologist annotations
- [ ] Real-time inference optimization
- [ ] Web interface for clinical deployment

## Citation

If you use this work in your research, please cite:

```bibtex
@software{symmMRI2024,
  title={Symmetry-based Brain Tumor Detection using DINOv3},
  author={Author Name},
  year={2024},
  url={https://github.com/mtorabi59/symmMRI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- **DINOv3**: [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2304.07193)
- **Vision Transformers**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **Self-Supervised Learning**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or inquiries, please open an issue on the GitHub repository.

---

**Note**: This is a research prototype. For clinical applications, proper validation and regulatory approval are required.
