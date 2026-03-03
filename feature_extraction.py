"""
DINOv3 feature extraction module.
Handles model loading and token encoding from images.
"""

import torch
from transformers import AutoModel
from typing import Tuple

# -------------------------------------------------
# Compiler shim (for torch versions where torch.compiler.is_compiling is missing)
# -------------------------------------------------
if not hasattr(torch, "compiler"):
    class _CompilerShim:
        pass
    torch.compiler = _CompilerShim()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
# -------------------------------------------------


def load_model(model_name: str, device: str = "cpu"):
    """
    Load a DINOv3 model from Hugging Face.
    
    Args:
        model_name (str): Hugging Face model identifier
        device (str): Device to load model on ("cuda" or "cpu")
    
    Returns:
        PreTrainedModel: Loaded and evaluated model
    """
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    return model


@torch.no_grad()
def encode_tokens(model, image_pil, transform, device: str = "cpu") -> Tuple[torch.Tensor, Tuple[int, int], int, torch.Tensor]:
    """
    Extract DINOv3 tokens from an image.
    
    This function:
    1. Preprocesses the image using the provided transform
    2. Passes the image through the DINOv3 model
    3. Extracts the last hidden state (tokens)
    4. Computes the patch grid dimensions from model config
    
    Args:
        model: Loaded DINOv3 model
        image_pil: PIL Image object
        transform: Preprocessing transform pipeline
        device (str): Device to run inference on
    
    Returns:
        Tuple containing:
            - tokens (torch.Tensor): Shape (1, T, C) where T=num_tokens, C=embedding_dim
            - grid_hw (Tuple[int, int]): Patch grid dimensions (height, width)
            - n_special (int): Number of special tokens (typically 1)
            - pixel_values (torch.Tensor): Preprocessed image (1, 3, H, W)
    
    Raises:
        ValueError: If token count doesn't match expected patch grid
    """
    pixel_values = transform(image_pil).unsqueeze(0).to(device)

    outputs = model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state  # (1, T, C)

    # Infer patch grid from model input size & patch size
    H, W = pixel_values.shape[-2], pixel_values.shape[-1]
    patch = model.config.patch_size  # e.g., 16 for ViT-B/16
    gh, gw = H // patch, W // patch
    n_patches = gh * gw

    T = tokens.shape[1]
    n_special = T - n_patches
    if n_special < 1:
        raise ValueError(f"Unexpected token count: T={T}, inferred patches={n_patches}")

    return tokens, (gh, gw), n_special, pixel_values
