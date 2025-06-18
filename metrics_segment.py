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

def evaluate(model_paths):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

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

        test_dir = Path(scene_dir) / "train"
        mask_dir = Path(test_dir) / "masks" 

        for method in os.listdir(test_dir):
            if method == "masks":
                continue
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"

            maskedssims = []
            nonmaskedssims = []
            maskedpsnrs = []
            nonmaskedpsnrs = []

            with torch.no_grad():
                for fname in tqdm(os.listdir(renders_dir), desc="Metric evaluation progress"):
                    render_path = renders_dir / fname
                    gt_path = gt_dir / fname
                    mask_path = mask_dir / fname

                    render = tf.to_tensor(Image.open(render_path).convert("RGB")).unsqueeze(0).to(device)
                    gt = tf.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)

                    if mask_path.exists():
                        mask = tf.to_tensor(Image.open(mask_path).convert("L")).squeeze(0).to(device)
                        mask = (mask > 0).float()
                    else:
                        print(f"Warning: Mask not found for {fname}, defaulting to empty mask.")
                        mask = torch.zeros_like(gt[0, 0])

                    maskedssim, nonmaskedssim = segmented_ssim(render, gt, mask)
                    maskedssims.append(maskedssim)
                    nonmaskedssims.append(nonmaskedssim)

                    maskedpsnr, nonmaskedpsnr = segmented_psnr(render, gt, mask)
                    maskedpsnrs.append(maskedpsnr)
                    nonmaskedpsnrs.append(nonmaskedpsnr)

                    del render, gt, mask
                    torch.cuda.empty_cache()

            print(" SEGMENTED MASKED SSIM : {:>12.7f}".format(torch.tensor(maskedssims).nanmean(), ".5"))
            print(" SEGMENTED NONMASKED SSIM : {:>12.7f}".format(torch.tensor(nonmaskedssims).nanmean(), ".5"))
            print(" SEGMENTED MASKED PSNR : {:>12.7f}".format(torch.tensor(maskedpsnrs).nanmean(), ".5"))
            print(" SEGMENTED NONMASKED PSNR : {:>12.7f}".format(torch.tensor(nonmaskedpsnrs).nanmean(), ".5"))
            print("")

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate segmented image metrics")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
