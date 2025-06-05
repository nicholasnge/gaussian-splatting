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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        args.depths = ""
        args.train_test_exp = False
        print("depths: " + str(args.depths))
        print("train test: " + str(args.train_test_exp))
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    #     exposure_dict = {
    #         image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
    #         for image_name in self.gaussians.exposure_mapping
    #     }

    #     with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
    #         json.dump(exposure_dict, f, indent=2)

    # NEW

    def save_separate(self, iteration):
        point_cloud_dir = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(point_cloud_dir, exist_ok=True)

        # Group Gaussians by SH degree
        unique_degrees = torch.unique(self.gaussians._sh_degree).tolist()
        for degree in unique_degrees:
            mask = self.gaussians._sh_degree == degree
            if not torch.any(mask):
                continue
            path = os.path.join(point_cloud_dir, f"point_cloud_SH{degree}.ply")
            self.gaussians.save_ply_separate(path, mask, degree)
        # mask = self.gaussians._sh_degree == 3
        # path = os.path.join(point_cloud_dir, f"point_cloud.ply")
        # self.gaussians.save_ply_separate(path, mask, 3)

    def save(self, iteration, quantise=False, half_float=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        ply_name = "point_cloud"
        if quantise:
            ply_name += "_quantised"
        if half_float:
            ply_name += "_half"
        ply_name_to_view = ply_name + ".ply"
        ply_name += "_ours.ply"
        self.gaussians.save_ply(os.path.join(point_cloud_path, ply_name), quantise, half_float)

        #self.gaussians.save_ply_to_view(os.path.join(point_cloud_path, ply_name_to_view), quantise, half_float)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
