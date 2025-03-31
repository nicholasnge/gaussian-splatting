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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def compute_ssim_map(img1, img2, window, window_size, channel, size_average=True):
    """Computes the SSIM map without averaging."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map

def segmented_ssim(img1, img2, mask, window_size=11, size_average=True):
    """
    Computes SSIM separately for masked (nonzero) and non-masked (zero) regions.
    :param img1: First image (B, C, H, W)
    :param img2: Second image (B, C, H, W)
    :param mask: Binary mask (H, W), where nonzero pixels define segmented area
    :return: SSIM for masked and non-masked regions
    """

    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Compute SSIM map separately
    ssim_map = compute_ssim_map(img1, img2, window, window_size, channel, size_average=False)
    # print("ssim map shape" + str(ssim_map.shape))
    # print("mask shape" + str(mask.shape))

    # Ensure mask shape matches (B, 1, H, W)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Convert (H, W) -> (1, 1, H, W)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)  # Convert (1, H, W) -> (1, 1, H, W)

    mask = mask.to(img1.device)
    # print("mask shape" + str(mask.shape))

    # Expand mask to match SSIM map shape
    mask_binary = (mask > 0).float()
    mask_binary = mask_binary.expand_as(ssim_map)
    # print("mask shape" + str(mask_binary.shape))

    # DEBUG: Print statistics
    # print(f"Mask min/max: {mask_binary.min()} / {mask_binary.max()}")
    # print(f"Masked region count: {mask_binary.sum()}")
    # print(f"Non-masked region count: {(1 - mask_binary).sum()}")


    # Select pixels for masked and non-masked areas
    masked_pixels = ssim_map[mask_binary == 1]
    non_masked_pixels = ssim_map[mask_binary == 0]
    # print(masked_pixels.mean())
    # print(non_masked_pixels.mean())
    # Compute mean SSIM for both regions
    masked_ssim = masked_pixels.mean() if masked_pixels.numel() > 0 else torch.tensor(float("nan"), device=img1.device)
    non_masked_ssim = non_masked_pixels.mean() if non_masked_pixels.numel() > 0 else torch.tensor(float("nan"), device=img1.device)
    return masked_ssim, non_masked_ssim