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

import os
import torch
from random import randint
# import new loss which considers weighted mask
from utils.loss_utils import l1_loss, ssim
import torch
import torchvision.transforms as transforms

from gaussian_renderer import render, original_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import GaussianScoreTracker
from GaussianGrowthTracker import GaussianGrowthTracker, RatioScalingGaussianGrowthTracker
import faiss
import numpy as np
from scene.cameras import CameraManager

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, memMB):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # initialise score tracker
    gaussianScoreTracker = GaussianScoreTracker.GaussianScoreTracker()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    cameraManager = CameraManager(scene.getTrainCameras(), 40)
    #growthTracker = GaussianGrowthTracker([30, 60], densify_until=opt.iterations)
    growthTracker = RatioScalingGaussianGrowthTracker(ratios=[2.0, 1.0], total_memory_MB=memMB, densify_until=opt.densify_until_iter)

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = cameraManager.getNextCamera()

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration <= args.segIter:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            gaussianScoreTracker.update(render_pkg["gaussian_scores"])
        else:
            render_pkg = original_render(viewpoint_cam, gaussians, pipe, bg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end))

            if iteration == args.segIter:
                cameraManager.toggleMaskLoadingOff()
                gaussian_scores = gaussianScoreTracker.get_scores()

                # 1. Set objectid = 1 if score > ..., else remain 0
                objectid = torch.zeros_like(gaussian_scores, dtype=torch.int8)
                objectid[gaussian_scores > args.segReq] = 1

                xyz_all = gaussians.get_xyz.detach().cpu().numpy().astype('float32')
                objid_np = objectid.cpu().numpy()

                # 2. Expand objectid=1 via FAISS
                print("expanding objectid 1 via radius search")
                radius = args.segSpread * scene.cameras_extent
                src_mask = (objid_np == 1)
                tgt_mask = (objid_np == 0)

                src_xyz = xyz_all[src_mask]
                tgt_xyz = xyz_all[tgt_mask]
                tgt_indices = np.where(tgt_mask)[0]

                index = faiss.IndexFlatL2(3)
                index.add(tgt_xyz)
                LIMS, D, I = index.range_search(src_xyz, radius**2)
                global_ids = tgt_indices[I.astype(np.int64)]
                unique_ids = np.unique(global_ids)
                objectid[unique_ids] = 1  # promote these to objectid 1

                # 3. Assign SH degree based on objectid
                degrees = torch.where(objectid == 1, torch.tensor(3, dtype=torch.int32, device='cuda'),
                                                torch.tensor(0, dtype=torch.int32, device='cuda'))

                # 4. Zero unused SH coefficients
                f_rest = gaussians._features_rest
                def num_sh_coeffs(deg): return (deg + 1)**2
                for deg in range(gaussians.max_sh_degree + 1):
                    cutoff = num_sh_coeffs(deg) - 1
                    mask = (degrees == deg)
                    f_rest[mask, cutoff:, :] = 0.0

                # 5. Update model
                gaussians.objectid = objectid.cuda()
                gaussians._sh_degree = degrees
                gaussians._features_rest = f_rest

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Track growth and get thresholds
                    with torch.no_grad():
                        dens_thresh_multi = growthTracker.update(iteration, gaussians)
                        print(f"dens thresh multi: {dens_thresh_multi}")

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(
                        dens_thresh_multi,                       # tensor [num_objectids]
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii
                    )
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            elif iteration % opt.densification_interval == 0:
                # Just record memory without applying threshold
                growthTracker.updateOnly(iteration, gaussians)
                gaussians.prune_only(1/255)

            if iteration >= 5000 and iteration % 500 == 0:
                gaussians.cull_sh_bands(1e-2)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        

    with torch.no_grad():
        # _features_rest: (N, F, SH-1)
        sh_rest = gaussians._features_rest  # shape: (N, 3, num_SH-1)

        # Check which Gaussians have *all* SH>0 coefficients = 0
        zeroed_mask = (sh_rest == 0).all(dim=1).all(dim=1)  # shape: (N,)

        count = zeroed_mask.sum().item()
        total = sh_rest.shape[0]

        print(f"\n[Training Complete] {count}/{total} Gaussians have SH>0 features zeroed out (degree 0 only).")

    growthTracker.print_history()

    scene.save(iteration)
    #scene.gaussians.produce_clusters(store_dict_path=scene.model_path)
    #scene.save_separate(iteration)        
    #scene.save(iteration, quantise=True)
    #scene.save(iteration, quantise=True, half_float=True)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, total_loss, elapsed):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/Ll1', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--memMB", type=int, default = None)
    parser.add_argument("--segIter", type=int, default = 500)
    parser.add_argument("--segReq", type=float, default = 0.1)
    parser.add_argument("--segSpread", type=float, default = 0.1)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.start_checkpoint, args.debug_from, args.memMB)

    # All done
    print("\nTraining complete.")
