"""
Brain symmetry analysis module.
Computes asymmetry maps using mirror correspondence of DINOv3 features.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


@torch.no_grad()
def max_local_similarity_in_flipped_grid(
    tokens: torch.Tensor,
    tokens_flipped: torch.Tensor,
    grid_hw: Tuple[int, int],
    r: int = 1,
    n_special: int = 1,
    normalize: bool = True,
    return_best_idx: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute maximum cosine similarity between features and their mirrored counterparts.
    
    For each patch (y,x) in the original image, this function searches a neighborhood
    of patches in the horizontally flipped image to find the best matching features.
    Higher similarity indicates brain symmetry at that location.
    
    Algorithm:
    1. Extract patch embeddings from original and flipped image tokens
    2. For each patch in the original image, search a (2r+1)×(2r+1) neighborhood
       in the flipped image grid centered at the mirrored location
    3. Compute cosine similarity for each candidate and track the maximum
    
    Key Mapping:
        flipped_grid[y, x] ↔ original content at (y, gw-1-x)
        So comparing O[y,x] with F[y,x] is a true mirror correspondence
    
    Args:
        tokens (torch.Tensor): Original image tokens, shape (B, T, D)
        tokens_flipped (torch.Tensor): Flipped image tokens, shape (B, T, D)
        grid_hw (Tuple[int, int]): Patch grid dimensions (height, width)
        r (int): Search radius in patch units (default: 1)
        n_special (int): Number of special tokens to skip (default: 1)
        normalize (bool): Whether to L2-normalize features (default: True)
        return_best_idx (bool): Whether to return best match indices (default: True)
    
    Returns:
        If return_best_idx is True:
            Tuple of:
                - max_sim (torch.Tensor): Max similarity map, shape (B, gh, gw), range [0, 1]
                - best_yx_flipped (torch.Tensor): Best match indices in flipped grid, shape (B, gh, gw, 2)
        Else:
            - max_sim (torch.Tensor): Max similarity map only
    
    Raises:
        ValueError: If token count is incompatible with grid size
    """
    B, T, D = tokens.shape
    gh, gw = grid_hw
    N = gh * gw
    if T < n_special + N:
        raise ValueError("Token count smaller than expected given grid_hw and n_special.")

    # Extract patch tokens -> spatial grids
    O = tokens[:, n_special:n_special + N, :].view(B, gh, gw, D)
    Fg = tokens_flipped[:, n_special:n_special + N, :].view(B, gh, gw, D)

    if normalize:
        O = F.normalize(O, dim=-1)
        Fg = F.normalize(Fg, dim=-1)

    # Rearrange to (B, D, gh, gw) for efficient conv-like operations
    O_chw = O.permute(0, 3, 1, 2)    # (B, D, gh, gw)
    F_chw = Fg.permute(0, 3, 1, 2)   # (B, D, gh, gw)

    # Pad flipped grid for boundary handling during neighborhood search
    F_pad = F.pad(F_chw, (r, r, r, r), mode="replicate")  # (B, D, gh+2r, gw+2r)

    max_sim = None
    if return_best_idx:
        best_y = torch.zeros((B, gh, gw), device=tokens.device, dtype=torch.int64)
        best_x = torch.zeros((B, gh, gw), device=tokens.device, dtype=torch.int64)

    # Search neighborhood offsets
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y0 = dy + r
            x0 = dx + r
            cand = F_pad[:, :, y0:y0 + gh, x0:x0 + gw]   # aligned with center (y,x) in flipped grid
            sim = (O_chw * cand).sum(dim=1)              # (B, gh, gw) - cosine similarity

            if max_sim is None:
                max_sim = sim
                if return_best_idx:
                    # Store absolute indices in flipped grid
                    yy = torch.arange(gh, device=sim.device).view(1, gh, 1).expand(B, gh, gw)
                    xx = torch.arange(gw, device=sim.device).view(1, 1, gw).expand(B, gh, gw)
                    best_y = (yy + dy).clamp(0, gh - 1)
                    best_x = (xx + dx).clamp(0, gw - 1)
            else:
                better = sim > max_sim
                max_sim = torch.where(better, sim, max_sim)
                if return_best_idx:
                    yy = torch.arange(gh, device=sim.device).view(1, gh, 1).expand(B, gh, gw)
                    xx = torch.arange(gw, device=sim.device).view(1, 1, gw).expand(B, gh, gw)
                    cand_y = (yy + dy).clamp(0, gh - 1)
                    cand_x = (xx + dx).clamp(0, gw - 1)
                    best_y = torch.where(better, cand_y, best_y)
                    best_x = torch.where(better, cand_x, best_x)

    if return_best_idx:
        best_yx = torch.stack([best_y, best_x], dim=-1)  # (B, gh, gw, 2)
        return max_sim, best_yx
    return max_sim


def compute_asymmetry_map(max_sim: torch.Tensor) -> torch.Tensor:
    """
    Convert similarity scores to asymmetry map.
    
    Higher values indicate more asymmetric regions (potential tumor locations).
    
    Args:
        max_sim (torch.Tensor): Maximum similarity map from symmetry analysis
    
    Returns:
        torch.Tensor: Asymmetry map (1 - similarity), clipped to [0, 1]
    """
    asym = (1.0 - max_sim).clamp(min=0.0)
    return asym
