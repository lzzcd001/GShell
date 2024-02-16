# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json

import torch
import numpy as np

from render import util

from .dataset import Dataset

import cv2 as cv

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K


    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def _load_img(path):
    img = util.load_image_raw(path)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img



class DatasetDeepFashion(Dataset):
    def __init__(self, base_dir, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = base_dir

        # Load config / transforms
        self.n_images = 72 ### hardcoded

        self.fovy               = np.deg2rad(60)
        self.proj_mtx = util.perspective(
            self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1]
        )



        camera_dict = np.load(os.path.join(self.base_dir, 'cameras_sphere.npz'))

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scale_mats_np = []


        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.join(self.base_dir, '{:03d}.png'.format(0))).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

    def _parse_frame(self, idx):
        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, '{:03d}.png'.format(idx)))
        img[:,:,:3] = img[:,:,:3] * img[:,:,3:]
        img[:,:,3] = torch.sign(img[:,:,3])
        assert img.size(-1) == 4

        flip_mat = torch.tensor([
            [ 1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1]
        ], dtype=torch.float)

        mv = flip_mat @ torch.linalg.inv(self.pose_all[idx])
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp = self.proj_mtx @ mv

        return img[None, ...].cuda(), mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda() # Add batch dimension

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img      = []

        img, mv, mvp, campos = self._parse_frame(itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
