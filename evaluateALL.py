from pathlib import Path
import os
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from utils.loss_utils import segmented_ssim
from utils.image_utils import segmented_psnr
import re

def evaluate_all_models(parent_folder, output_csv, mask_root=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    rows = ["Focus PSNR", "Nonfocus PSNR", "Focus SSIM", "Nonfocus SSIM", "Size (MB)"]
    results = {}

    parent_path = Path(parent_folder)
    subfolders = [f for f in parent_path.iterdir() if f.is_dir()]

    variants = {
        "": ("point_cloud.ply", ""),
        "quantised": ("point_cloud_quantised.ply", "Q"),
        "quantised_half": ("point_cloud_quantised_half.ply", "QH")
    }

    for subfolder in subfolders:
        model_name = subfolder.name
        base_model_name = re.split(r'[_\d]', model_name)[0]

        for variant_key, (ply_file, suffix) in variants.items():
            method_dir_name = f"{ply_file}_30000"
            method_dir = subfolder / "test" / method_dir_name
            renders_dir = method_dir / "renders"
            gt_dir = method_dir / "gt"
            ply_path = subfolder / "point_cloud" / "iteration_30000" / ply_file

            if not ply_path.exists() or not renders_dir.exists() or not gt_dir.exists():
                print(f"Skipping {model_name}{suffix if suffix != 'PLAIN' else ''}: missing PLY or render/gt folders.")
                continue

            mask_dir = Path(mask_root) / base_model_name if mask_root else method_dir / "masks"
            col_name = f"{model_name}{'' if suffix == 'PLAIN' else suffix}"

            print(f"\nEvaluating {col_name}")
            metric_values = {
                "Focus PSNR": [],
                "Nonfocus PSNR": [],
                "Focus SSIM": [],
                "Nonfocus SSIM": [],
            }

            with torch.no_grad():
                for fname in tqdm(os.listdir(renders_dir), desc=f"Eval {col_name}"):
                    render_path = renders_dir / fname
                    gt_path = gt_dir / fname
                    mask_path = mask_dir / fname

                    if not render_path.exists() or not gt_path.exists():
                        continue

                    render = tf.to_tensor(Image.open(render_path).convert("RGB")).unsqueeze(0).to(device)
                    gt = tf.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)

                    if mask_path.exists():
                        mask = tf.to_tensor(Image.open(mask_path).convert("L")).squeeze(0).to(device)
                        mask = (mask > 0).float()
                    else:
                        print(f"Warning: Missing mask for {fname}, using empty mask.")
                        mask = torch.zeros_like(gt[0, 0])

                    ssim_focus, ssim_nonfocus = segmented_ssim(render, gt, mask)
                    psnr_focus, psnr_nonfocus = segmented_psnr(render, gt, mask)

                    metric_values["Focus PSNR"].append(psnr_focus)
                    metric_values["Nonfocus PSNR"].append(psnr_nonfocus)
                    metric_values["Focus SSIM"].append(ssim_focus)
                    metric_values["Nonfocus SSIM"].append(ssim_nonfocus)

                    del render, gt, mask
                    torch.cuda.empty_cache()

            if not metric_values["Focus PSNR"]:
                print(f"Skipping {col_name}: no valid image pairs.")
                continue

            results[col_name] = [
                torch.tensor(metric_values["Focus PSNR"]).nanmean().item(),
                torch.tensor(metric_values["Nonfocus PSNR"]).nanmean().item(),
                torch.tensor(metric_values["Focus SSIM"]).nanmean().item(),
                torch.tensor(metric_values["Nonfocus SSIM"]).nanmean().item(),
                os.path.getsize(ply_path) / (1024 * 1024)
            ]

    df = pd.DataFrame(results, index=rows)
    df.to_csv(output_csv)
    print(f"\n✅ Saved results to {output_csv}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-folder", "-i", required=True, type=str, help="Folder with model subfolders")
    parser.add_argument("--output-csv", "-o", required=True, type=str, help="Output CSV path")
    parser.add_argument("--mask-root", "-m", required=False, type=str, help="Folder containing masks (optional)")
    args = parser.parse_args()

    evaluate_all_models(args.input_folder, args.output_csv, args.mask_root)
