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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import os

WARNED = False

def modify_path(file_path):
    dir_path, filename = os.path.split(file_path)  # Split into directory and filename
    parent_dir = os.path.dirname(dir_path)  # Get parent directory
    new_dir = os.path.join(parent_dir, "masks")  # Target directory

    # Change file extension from .JPG to .png
    base_name, _ = os.path.splitext(filename)  # Remove original extension
    new_filename = base_name + ".png"  # Set new extension

    return os.path.join(new_dir, new_filename)  # Return new full path

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    mask_path = modify_path(cam_info.image_path)
    image = Image.open(cam_info.image_path)
    orig_w, orig_h = image.size

    # Compute resolution
    if args.resolution in [1, 2, 4, 8]:
        resolution = (round(orig_w / (resolution_scale * args.resolution)),
                      round(orig_h / (resolution_scale * args.resolution)))
    else:
        global_down = orig_w / (1600 if args.resolution == -1 and orig_w > 1600 else args.resolution)
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(image_path=cam_info.image_path,
                  mask_path=mask_path,
                  resolution=resolution,
                  colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry