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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import segmented_ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import segmented_psnr
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []

    mask_dir = Path(str(gt_dir) + "M")  # Mask directory (same as GT but with 'M')

    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)

        # Load the corresponding mask
        mask_path = mask_dir / fname  # Mask has the same filename
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")  # Ensure grayscale
            mask = tf.to_tensor(mask).squeeze(0)  # Convert to tensor & remove extra dim
        else:
            print(f"Warning: Mask not found for {fname}, defaulting to empty mask.")
            mask = torch.zeros_like(tf.to_tensor(gt)[0])  # Create empty mask


        mask = (mask > 0).float()

        # Store data
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(mask.cuda())  # Ensure the mask is moved to the correct device
        image_names.append(fname)

        # print(f"Mask min/max: {mask.min()} / {mask.max()}")
        # print(f"Mask shape: {mask.shape}")

        # plt.figure(figsize=(6, 3))
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask.cpu().numpy(), cmap="gray")
        # plt.title("Mask")
        
        # plt.subplot(1, 2, 2)
        # plt.imshow(gt)
        # plt.title("Ground Truth Image")

        # plt.show()

    return renders, gts, masks, image_names


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            if method != "ours_30000":
                continue
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, masks, image_names = readImages(renders_dir, gt_dir)

            maskedssims = []
            nonmaskedssims = []
            maskedpsnrs = []
            nonmaskedpsnrs = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                maskedssim, nonmaskedssim = segmented_ssim(renders[idx], gts[idx], masks[idx])
                maskedssims.append(maskedssim)
                nonmaskedssims.append(nonmaskedssim)

                maskedpsnr, nonmaskedpsnr = segmented_psnr(renders[idx], gts[idx], masks[idx])
                maskedpsnrs.append(maskedpsnr)
                nonmaskedpsnrs.append(nonmaskedpsnr)
                
            print(" SEGMENTED MASKED SSIM : {:>12.7f}".format(torch.tensor(maskedssims).nanmean(), ".5"))
            print(" SEGMENTED NONMASKED SSIM : {:>12.7f}".format(torch.tensor(nonmaskedssims).nanmean(), ".5"))
            print(" SEGMENTED MASKED PSNR : {:>12.7f}".format(torch.tensor(maskedpsnrs).nanmean(), ".5"))
            print(" SEGMENTED NONMASKED PSNR : {:>12.7f}".format(torch.tensor(nonmaskedpsnrs).nanmean(), ".5"))
            
            print("")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
