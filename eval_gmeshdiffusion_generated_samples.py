# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import argparse
import json

import numpy as np
import torch

# Import topology / geometry trainers
from geometry.gshell_tets_geometry import GShellTetsGeometry

from render import texture

import pymeshlab
from pytorch3d.io import save_obj

import tqdm

RADIUS = 4.0
# RADIUS = 2.5

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--grid_root', type=str)
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                      # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.3                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.use_tanh_deform     = False
    FLAGS.use_sdf_mlp         = False
    FLAGS.force_default_mtl   = True
    FLAGS.twosided_texture    = True
    FLAGS.random_lgt          = False
    FLAGS.sphere_init         = False
    FLAGS.num_smooth_steps    = 3
    FLAGS.use_msdf_mlp        = False

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]


    os.makedirs(FLAGS.out_dir, exist_ok=True)

    mtl_default_diffuse = {
        'name' : '_default_mat',
        'bsdf': 'diffuse',
        'uniform': True,
        'kd'   : texture.Texture2D(torch.tensor([0.75, 0.3, 0.6], dtype=torch.float32, device='cuda')),
        'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
    }

    if FLAGS.force_default_mtl:
        mtl_default = mtl_default_diffuse
    else:
        mtl_default = None


    tet_path = './data/tets/64_tets_cropped_reordered.npz'
    tet = np.load(tet_path)
    vertices = torch.tensor(tet['vertices'])
    edges = torch.tensor(tet['edges']).long()
    vertices_unique = vertices[:].unique()
    dx = (vertices_unique[1] - vertices_unique[0]) / 2.0

    vertices_discretized = (torch.round(
        (vertices - vertices.min()) / dx)
    ).long()


    midpoints = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2.0
    midpoints_dicretized = (torch.round(
        (midpoints - vertices.min()) / dx)
    ).long()

    aabb = torch.tensor(FLAGS.aabb, dtype=torch.float).cuda().view(2, 3)
    center = aabb.mean(0, keepdim=True) / 2.0

    mesh_scale = 3.8
    mesh_scale = mesh_scale / torch.max(aabb[1] - aabb[0]).item()

    count = 0
    grid_root = FLAGS.grid_root
    geometry = GShellTetsGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS, tet_init_file=tet_path, extract_from_generative=True)
    with torch.no_grad():
        for grid_name in tqdm.tqdm(sorted(list(os.listdir(grid_root)))):
            if '_occ' in grid_name:
                continue


            grid_all = torch.load(
                os.path.join(grid_root, grid_name), map_location='cuda'
            )
            occgrid_all = torch.load(
                os.path.join(grid_root, grid_name).replace('.pt', '_occ.pt'), map_location='cuda'
            )[:, 0]
            for i in tqdm.trange(grid_all.size(0), leave=False):
                mesh_path = FLAGS.out_dir
                os.makedirs(mesh_path, exist_ok=True)
                mesh_savepath = os.path.join(mesh_path, '{:06d}.obj'.format(count))

                if os.path.exists(mesh_savepath):
                    count += 1
                    continue
                grid = grid_all[i]
                occgrid = occgrid_all[i]

                sdf_sign = (
                        grid[0, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]]
                    ).cuda().float()
                geometry.deform.data[:] = (
                        grid[1:4, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]]
                    ).cuda().transpose(0, 1).float().clamp(-1, 1)
                

                sdf_coeff = torch.ones(128, 128, 128).float().cuda() * 0.5

                msdf_sign = torch.zeros(128, 128, 128).float().cuda()
                msdf_sign[midpoints_dicretized[:, 0], midpoints_dicretized[:, 1], midpoints_dicretized[:, 2]] = torch.sign(
                    grid[0, midpoints_dicretized[:, 0], midpoints_dicretized[:, 1], midpoints_dicretized[:, 2]].cuda()
                ).float()
                geometry.deform.data[:] = geometry.deform.data[:].clip(-1.0, 1.0)
                geometry.deform_scale = 2.0

                base_mesh = geometry.getMesh_from_augmented_grid_withocc(mtl_default, torch.sign(sdf_sign), sdf_coeff, msdf_sign, occgrid=occgrid)['imesh']

                ### rescale and translate back to align with the dataset
                base_mesh.v_pos = (base_mesh.v_pos / mesh_scale) + center

                ### save post-processed mesh
                save_obj(
                    verts=base_mesh.v_pos,
                    faces=base_mesh.t_pos_idx,
                    f=mesh_savepath
                )

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(mesh_savepath)
                ms.meshing_remove_unreferenced_vertices()
                ms.meshing_isotropic_explicit_remeshing()
                ms.apply_coord_laplacian_smoothing(stepsmoothnum=FLAGS.num_smooth_steps, cotangentweight=True)
                # ms.apply_coord_hc_laplacian_smoothing()
                # ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, cotangentweight=True) ## for smoother surface
                ms.meshing_isotropic_explicit_remeshing()
                ms.apply_filter_script()
                ms.save_current_mesh(mesh_savepath)

                count += 1
