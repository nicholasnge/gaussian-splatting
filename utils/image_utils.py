#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def segmented_psnr(img1, img2, mask):
    """
    Compute PSNR separately for pixels where the mask is nonzero and where the mask is zero.
    
    Args:
        img1 (torch.Tensor): Predicted image tensor.
        img2 (torch.Tensor): Ground truth image tensor.
        mask (torch.Tensor): Binary mask tensor (1 for relevant regions, 0 for background).
    
    Returns:
        Tuple (psnr_foreground, psnr_background)
    """
    # Ensure mask is binary (0 or 1)
    # Ensure mask is strictly binary (0 or 1)
    mask = (mask > 0).float()

    # Expand mask to match image channels
    if mask.dim() == 2:  # Convert (H, W) → (1, 1, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:  # Convert (1, H, W) → (1, 1, H, W)
        mask = mask.unsqueeze(1)

    mask = mask.to(img1.device).expand_as(img1)  # Match img1 shape

    # Get number of pixels in foreground and background
    num_fg = mask.sum()
    num_bg = (1 - mask).sum()

    # Compute MSE only over the relevant pixels
    mse_fg = ((img1 - img2) ** 2 * mask).sum() / num_fg if num_fg > 0 else torch.tensor(float("nan"), device=img1.device)
    mse_bg = ((img1 - img2) ** 2 * (1 - mask)).sum() / num_bg if num_bg > 0 else torch.tensor(float("nan"), device=img1.device)

    # Convert to PSNR
    psnr_fg = 20 * torch.log10(1.0 / torch.sqrt(mse_fg)) if num_fg > 0 else torch.tensor(float("nan"), device=img1.device)
    psnr_bg = 20 * torch.log10(1.0 / torch.sqrt(mse_bg)) if num_bg > 0 else torch.tensor(float("nan"), device=img1.device)

    return psnr_fg, psnr_bg