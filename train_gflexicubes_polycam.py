# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import sys
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from dataset.dataset_nerf_colmap import DatasetNERF

# Import topology / geometry trainers
from geometry.gshell_flexicubes_geometry import GShellFlexiCubesGeometry

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render


from denoiser.denoiser import BilateralDenoiser

import tqdm

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background

    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    try:
        eval_mesh = eval_mesh['imesh']
    except:
        pass
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks'])

    # Dilate all textures & use average color for background
    kd_avg = torch.sum(torch.sum(torch.sum(kd * mask, dim=0), dim=0), dim=0) / torch.sum(torch.sum(torch.sum(mask, dim=0), dim=0), dim=0)
    kd = util.dilate(kd, kd_avg[None, None, None, :], mask, 7)

    ks_avg = torch.sum(torch.sum(torch.sum(ks * mask, dim=0), dim=0), dim=0) / torch.sum(torch.sum(torch.sum(mask, dim=0), dim=0), dim=0)
    ks = util.dilate(ks, ks_avg[None, None, None, :], mask, 7)

    nrm_avg = torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda")
    normal = nrm_avg[None, None, None, :].repeat(kd.shape[0], kd.shape[1], kd.shape[2], 1)
    
    new_mesh.material = mat.copy()
    del new_mesh.material['kd_ks']

    if FLAGS.transparency:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)
        print("kd shape", kd.shape)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    new_mesh.material.update({
        'kd'     : texture.Texture2D(kd.clone().detach().requires_grad_(True), min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks.clone().detach().requires_grad_(True), min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal.clone().detach().requires_grad_(True), min_max=[nrm_min, nrm_max]),
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=6, min_max=[mlp_min, mlp_max], use_float16=FLAGS.use_float16)
        mat =  {'kd_ks' : mlp_map_opt}
    else:
        raise NotImplementedError

    mat['bsdf'] = FLAGS.bsdf

    mat['no_perturbed_nrm'] = FLAGS.no_perturbed_nrm

    return mat

def initial_guess_material_knownkskd(geometry, mlp, FLAGS, init_mat=None):
    mat =  {
        'kd'     : init_mat['kd'],
        'ks'     : init_mat['ks']
    }

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

###############################################################################
# Validation & testing
###############################################################################

@torch.no_grad()
def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, denoiser=None):
    result_dict = {}
    with torch.no_grad():
        buffers = geometry.render(glctx, target, lgt, opt_material, use_uv=False, denoiser=denoiser)['buffers']

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_dict['mask_opt'] = buffers['shaded'][...,3:][0].expand(-1, -1, 3)
        result_dict['mask_ref'] = target['img'][...,3:][0].expand(-1, -1, 3)
        result_dict['msdf_image'] = buffers['msdf_image'][...,:][0].expand(-1, -1, 3).clamp(min=0, max=1)
        result_image = torch.cat([result_dict['opt'], result_dict['ref'], result_dict['mask_opt'], result_dict['mask_ref'], result_dict['msdf_image']], axis=1)

        result_dict = {}
        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]

        return result_image, result_dict

@torch.no_grad()
def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS, denoiser=None, save_viz=False):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(tqdm.tqdm(dataloader_validate)):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, denoiser=denoiser)

            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            if save_viz:
                for k in result_dict.keys():
                    np_img = result_dict[k].detach().cpu().numpy()
                    util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def optimize_mesh(
        denoiser,
        glctx,
        geometry,
        opt_material,
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        pass_idx=0,
        pass_name="",
        optimize_light=True,
        optimize_geometry=True,
        visualize=True,
        save_path=None
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate * 6.0

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)



    params = list(material.get_parameters(opt_material))

    if optimize_light:
        optimizer_light = torch.optim.Adam((lgt.parameters() if lgt is not None else []), lr=learning_rate_lgt)
        scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    if optimize_geometry:
        if FLAGS.use_sdf_mlp:
            deform_params = list(v[1] for v in geometry.named_parameters() if 'deform' in v[0]) if optimize_geometry else []
            msdf_params = list(v[1] for v in geometry.named_parameters() if 'msdf' in v[0]) if optimize_geometry else []
            sdf_params = list(v[1] for v in geometry.named_parameters() if 'sdf' in v[0] and 'msdf' not in v[0]) if optimize_geometry else []
            other_params = list(v[1] for v in geometry.named_parameters() if 'sdf' not in v[0] and 'msdf' not in v[0] and 'deform' not in v[0]) if optimize_geometry else []
            optimizer_mesh = torch.optim.Adam([
                    {'params': deform_params, 'lr': learning_rate_pos},
                    {'params': msdf_params, 'lr': learning_rate_pos},
                    {'params': sdf_params, 'lr': learning_rate_pos * 1e-2},
                    {'params': other_params, 'lr': learning_rate_pos * 1e-2},
                ], eps=1e-8)
        else:
            optimizer_mesh = torch.optim.Adam(geometry.parameters(), lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    optimizer = torch.optim.Adam(params, lr=learning_rate_mat)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    depth_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    if visualize:
        dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if visualize and FLAGS.local_rank == 0 and it != 0:
            with torch.no_grad():
                display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
                save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
                if display_image or save_image:
                    save_mesh = True
                    if save_mesh:
                        os.makedirs(os.path.join(save_path, pass_name), exist_ok=True)
                        obj.write_obj(os.path.join(save_path, pass_name), geometry.getMesh(opt_material)['imesh'], save_material=False)
                    result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background), geometry, opt_material, lgt, FLAGS, denoiser=denoiser)
            
                    np_result_image = result_image.detach().cpu().numpy()
                    if display_image:
                        util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                    if save_image:
                        util.save_image(os.path.join(save_path, ('img_%s_%06d.png' % (pass_name, img_cnt))), np_result_image)
                        img_cnt = img_cnt + 1

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()
        if optimize_light:
            optimizer_light.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================

        xfm_lgt = None
        if optimize_light:
            if False and FLAGS.camera_space_light:
                lgt.xfm(target['mv'])
            elif False and ('envlight_transform' in target and target['envlight_transform'] is not None):
                xfm_lgt = target['envlight_transform']
                lgt.xfm(xfm_lgt)
            lgt.update_pdf()
            

        img_loss, depth_loss, reg_loss = geometry.tick(
            glctx, target, lgt, opt_material, image_loss_fn, it, 
            denoiser=denoiser)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())
        depth_loss_vec.append(depth_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if 'kd_ks' in opt_material:
            opt_material['kd_ks'].encoder.params.grad /= 8.0
        if 'kd_ks_back' in opt_material:
            opt_material['kd_ks_back'].encoder.params.grad /= 8.0

        # Optionally clip gradients
        if FLAGS.clip_max_norm > 0.0:
            if optimize_geometry:
                torch.nn.utils.clip_grad_norm_(geometry.parameters() + params, FLAGS.clip_max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(params, FLAGS.clip_max_norm)

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        if optimize_light:
            optimizer_light.step()
            scheduler_light.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'kd_back' in opt_material:
                opt_material['kd_back'].clamp_()
            if 'ks_back' in opt_material:
                opt_material['ks_back'].clamp_()
            if 'normal' in opt_material and not FLAGS.normal_only:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                # lgt.clamp_(min=0.01) # For some reason gradient dissapears if light becomes 0
                lgt.clamp_(min=1e-4) # For some reason gradient dissapears if light becomes 0

            geometry.clamp_deform()
        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            depth_loss_avg = np.mean(np.asarray(depth_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, img_loss=%.6f, depth_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, depth_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
            sys.stdout.flush()

        if it == FLAGS.iter:
            break

    return geometry, opt_material

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
    # Render specific arguments
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--bsdf', type=str, default='pbr', choices=['pbr', 'diffuse', 'white'])
    # Denoiser specific arguments
    parser.add_argument('--denoiser', default='bilateral', choices=['none', 'bilateral'])
    parser.add_argument('--denoiser_demodulate', type=bool, default=True)
    parser.add_argument('--msdf_reg_open_scale', type=float, default=1e-6)
    parser.add_argument('--msdf_reg_close_scale', type=float, default=3e-4)
    parser.add_argument('--eikonal_scale', type=float, default=5e-2)
    parser.add_argument('--trainset_path', type=str)
    parser.add_argument('--testset_path', type=str, default='')

    FLAGS = parser.parse_args()
    FLAGS.mtl_override        = None        # Override material of model
    FLAGS.gshell_grid          = 64          # Resolution of initial tet grid. We provide 64 and 128 resolution grids. 
                                            #    Other resolutions can be generated with https://github.com/crawforddoran/quartet
                                            #    We include examples in data/tets/generate_tets.py
    FLAGS.mesh_scale          = 3.6         # Scale of tet grid box. Adjust to cover the model
    FLAGS.envlight            = None        # HDR environment probe
    FLAGS.env_scale           = 1.0         # Env map intensity multiplier
    FLAGS.probe_res           = 256         # Env map probe resolution
    FLAGS.learn_lighting      = True        # Enable optimization of env lighting
    FLAGS.display             = None        # Configure validation window/display. E.g. [{"bsdf" : "kd"}, {"bsdf" : "ks"}]
    FLAGS.transparency        = False       # Enabled transparency through depth peeling
    FLAGS.lock_light          = False       # Disable light optimization in the second pass
    FLAGS.lock_pos            = False       # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2         # Weight for sdf regularizer.
    FLAGS.laplace             = "relative"  # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 3000.0      # Weight for Laplace regularizer. Default is relative with large weight
    FLAGS.pre_load            = True        # Pre-load entire dataset into memory for faster training
    FLAGS.no_perturbed_nrm    = False       # Disable normal map
    FLAGS.decorrelated        = False       # Use decorrelated sampling in forward and backward passes
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0]
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    # FLAGS.ks_min              = [ 0.0,  0.08, 0.0]
    FLAGS.ks_min              = [ 0.0,  0.001, 0.0]
    FLAGS.ks_max              = [ 0.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.clip_max_norm       = 0.0
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.lambda_kd           = 0.1
    FLAGS.lambda_ks           = 0.05
    FLAGS.lambda_nrm          = 0.025
    FLAGS.lambda_nrm2         = 0.25
    FLAGS.lambda_chroma       = 0.0
    FLAGS.lambda_diffuse      = 0.15
    FLAGS.lambda_specular     = 0.0025

    FLAGS.random_lgt                  = False
    FLAGS.normal_only                 = False
    FLAGS.use_img_2nd_layer           = False
    FLAGS.use_depth                   = False
    FLAGS.use_depth_2nd_layer         = False
    FLAGS.use_tanh_deform             = False
    FLAGS.use_sdf_mlp                 = True
    FLAGS.use_msdf_mlp                = False
    FLAGS.use_eikonal                 = True
    FLAGS.sdf_mlp_pretrain_steps      = 10000
    FLAGS.use_mesh_msdf_reg           = True
    FLAGS.sphere_init                 = False
    FLAGS.sphere_init_norm            = 1.5
    FLAGS.pretrained_sdf_mlp_path     = f'./data/pretrained_mlp_{FLAGS.gshell_grid}_polycam.pt'
    FLAGS.n_hidden                    = 6
    FLAGS.d_hidden                    = 256
    FLAGS.n_freq                      = 6
    FLAGS.skip_in                     = [3]
    FLAGS.use_float16                 = False
    FLAGS.visualize_watertight        = False

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()
    glctx_display = glctx if FLAGS.batch < 16 else dr.RasterizeGLContext() # Context for display

    mtl_default = None

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    data_root = FLAGS.trainset_path


    dataset_train    = DatasetNERF(os.path.join(data_root, 'transforms.json'), FLAGS, examples=int(1e6))
    dataset_validate = DatasetNERF(os.path.join(data_root, 'transforms.json'), FLAGS)



    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    lgt = None
    if FLAGS.learn_lighting:
        lgt = light.create_trainable_env_rnd(FLAGS.probe_res, scale=0.0, bias=0.5)
        # lgt = light.create_trainable_env_rnd(FLAGS.probe_res, scale=0.0, bias=0.1)
    else:
        lgt = light.load_env(FLAGS.envlight, scale=FLAGS.env_scale, res=[FLAGS.probe_res, FLAGS.probe_res])

    # ==============================================================================================
    #  Setup denoiser
    # ==============================================================================================

    denoiser = None
    if FLAGS.denoiser == 'bilateral':
        denoiser = BilateralDenoiser().cuda()
    else:
        assert FLAGS.denoiser == 'none', "Invalid denoiser %s" % FLAGS.denoiser

    # Setup geometry for optimization
    geometry = GShellFlexiCubesGeometry(FLAGS.gshell_grid, FLAGS.mesh_scale, FLAGS)

    # Setup textures, make initial guess from reference if possible
    if not FLAGS.normal_only:
        mat = initial_guess_material(geometry, True, FLAGS, mtl_default)
    else:
        mat = initial_guess_material_knownkskd(geometry, True, FLAGS, mtl_default)
    mat['no_perturbed_nrm'] = True

    # Run optimization
    geometry, mat = optimize_mesh(denoiser, glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                    FLAGS, pass_idx=0, pass_name="pass1", optimize_light=FLAGS.learn_lighting, save_path=FLAGS.out_dir)

    validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "validate"), FLAGS, denoiser=denoiser, save_viz=True)

    with torch.no_grad():
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        torch.save(geometry.state_dict(), os.path.join(FLAGS.out_dir, "mesh/model.pt"))
        torch.save(mat['kd_ks'].state_dict(), os.path.join(FLAGS.out_dir, "mesh/mtl.pt"))
        light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

        # Create textured mesh from result
        base_mesh = geometry.getMesh(mat)['imesh']


        # Dump mesh for debugging.
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), base_mesh, save_material=False)

        # Free temporaries / cached memory
        torch.cuda.empty_cache()
        mat['kd_ks'].cleanup()
        del mat['kd_ks']
        if 'kd_ks_back' in mat:
            mat['kd_ks_back'].cleanup()
            del mat['kd_ks_back']

        # Free temporaries / cached memory
        torch.cuda.empty_cache()
        del mat