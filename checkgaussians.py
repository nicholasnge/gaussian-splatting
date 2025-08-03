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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import pandas as pd

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

models_configuration = {
    'baseline': {
        'quantised': False,
        'half_float': False,
        'name': 'point_cloud.ply'
        },
    'quantised': {
        'quantised': True,
        'half_float': False,
        'name': 'point_cloud_quantised.ply'
        },
    'quantised_half': {
        'quantised': True,
        'half_float': True,
        'name': 'point_cloud_quantised_half.ply'
        },
}

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        for model in args.models:
            name = models_configuration[model]['name']
            quantised = models_configuration[model]['quantised']
            half_float = models_configuration[model]['half_float']
            try:
                scene.gaussians.load_ply(os.path.join(scene.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(scene.loaded_iter),
                                                            name), quantised=quantised, half_float=half_float)
            except:
                raise RuntimeError(f"Configuration {model} with name {name} not found!")
            


            # Count sparsity in SH3 Gaussians
            f_rest = scene.gaussians._features_rest
            sh_degrees = scene.gaussians._sh_degree

            sh3_mask = (sh_degrees == 3)
            sh3_coeffs = f_rest[sh3_mask].transpose(1,2)  # [N, 3, 45] if RGB and SH3
            #print(f_rest[sh3_mask][0])

            # Group indices per SH degree
            deg1 = slice(0, 3)
            deg2 = slice(3, 8)
            deg3 = slice(8, 15)

            # Compute norms
            deg3_norm = torch.norm(sh3_coeffs[:, :, deg3], dim=(1, 2))
            deg2_norm = torch.norm(sh3_coeffs[:, :, deg2], dim=(1, 2))
            deg1_norm = torch.norm(sh3_coeffs[:, :, deg1], dim=(1, 2))

            threshold = 1e-2

            deg3_zero = (deg3_norm < threshold).sum().item()
            deg2_3_zero = (deg2_norm + deg3_norm < threshold).sum().item()
            deg1_2_3_zero = (deg1_norm + deg2_norm + deg3_norm < threshold).sum().item()

            print(f"[{model}] SH3 Gaussians: {sh3_mask.sum().item()}")
            print(f"  Degree 3 ≈ 0:           {deg3_zero}")
            print(f"  Degree 2+3 ≈ 0:        {deg2_3_zero}")
            print(f"  Degree 1+2+3 ≈ 0:      {deg1_2_3_zero}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--models",
                    help="Types of models to test",
                    choices=models_configuration.keys(),
                    default=['baseline', 'quantised', 'quantised_half'],
                    nargs="+")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), SPARSE_ADAM_AVAILABLE)