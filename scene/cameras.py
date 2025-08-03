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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
from PIL import Image
from random import randint
import cv2
from collections import deque
import threading
import time

class CameraManager:
    def __init__(self, all_cameras, buffer_size=30):
        self.buffer_size = buffer_size
        self.all_cameras = all_cameras
        self.viewpoint_indices = list(range(len(self.all_cameras)))
        self.queue = deque()
        self.toUnload = deque()
        self.lock = threading.Lock()
        self.queue_not_empty = threading.Condition()
        
        for _ in range(buffer_size):
            #rand_idx = randint(0, len(self.viewpoint_indices) - 1)
            vind = self.viewpoint_indices.pop()
            self.all_cameras[vind].load()
            self.queue.append(vind)

    def getNextCamera(self):
        # Spawn loader thread
        threading.Thread(target=self.loadNext, daemon=True).start()

        with self.queue_not_empty:
            while not self.queue:
                print("[CameraManager] Waiting for camera to be loaded...")
                self.queue_not_empty.wait()

            vind = self.queue.pop()
            self.toUnload.append(vind)
        
        cam = self.all_cameras[vind]
        while not cam.loaded:
            print(f"[CameraManager] Waiting for camera {cam.image_path} to finish loading...")
            time.sleep(0.01)
            print(f"[CameraManager] Waiting for camera {cam.image_path} to finish loading...")
            time.sleep(0.01)
            print(f"[CameraManager] Waiting for camera {cam.image_path} to finish loading...")
            time.sleep(0.01)
            cam.load()
            print(f"[CameraManager] Sent load again to camera {cam.image_path}")

        return cam

    def loadNext(self):
        with self.lock:
            # Refill viewpoint indices if empty
            if not self.viewpoint_indices:
                self.viewpoint_indices = list(range(len(self.all_cameras)))

            # Unload camera if available
            if len(self.toUnload) > 5:
                unloadIdx = self.toUnload.popleft()
                self.all_cameras[unloadIdx].unload()

            # Load a new camera
            #rand_idx = randint(0, len(self.viewpoint_indices) - 1)
            vind = self.viewpoint_indices.pop()
            self.all_cameras[vind].load()

            with self.queue_not_empty:
                self.queue.append(vind)
                self.queue_not_empty.notify()

    def toggleMaskLoadingOff(self):
        for camera in self.all_cameras:
            camera.loadmaskbool = False
            del camera.cpu_mask
            camera.cpu_mask = None


class Camera(nn.Module):
    def __init__(self, image_path, mask_path, resolution, colmap_id, R, T, FoVx, FoVy, 
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.mask_path = mask_path
        self.resolution = resolution
        self.data_device = torch.device(data_device)
        self.trans = trans
        self.scale = scale
        self.image_width = self.resolution[0]
        self.image_height = self.resolution[1]

        self.loadmaskbool = True
        self.mask = None
        self.original_image = None
        
        self.cpu_original_image = None
        self.cpu_mask = None

        self.loaded = False

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load(self):
        if self.loaded:
            print(f"Already loaded: {self.image_path}")
            return

        if self.cpu_original_image is None:
            # Load from disk and keep on CPU
            image = Image.open(self.image_path)
            resized_image_rgb = PILtoTorch(image, self.resolution)
            gt_image = resized_image_rgb[:3, ...]
            self.cpu_alpha_mask = resized_image_rgb[3:4, ...] if resized_image_rgb.shape[0] == 4 \
                                else torch.ones_like(resized_image_rgb[0:1, ...])
            self.cpu_original_image = gt_image.clamp(0.0, 1.0)
            if self.loadmaskbool:
                self.cpu_mask = PILtoTorch(Image.open(self.mask_path), self.resolution)[0:1, ...] \
                                if self.mask_path else torch.ones_like(self.cpu_alpha_mask)

        # Transfer from CPU to GPU
        self.original_image = self.cpu_original_image.to(self.data_device)
        self.alpha_mask = self.cpu_alpha_mask.to(self.data_device)
        if self.loadmaskbool:
            self.mask = self.cpu_mask.to(self.data_device)
        self.loaded = True


    def unload(self):
        if not self.loaded:
            print(f"Already unloaded: {self.image_path}")
            return

        del self.original_image
        del self.alpha_mask
        del self.mask
        if hasattr(self, "depth_mask"):
            del self.depth_mask
        torch.cuda.empty_cache()
        self.original_image = None
        self.mask = None
        self.loaded = False