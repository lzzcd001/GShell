import os
import sys
import numpy as np
import tqdm

import logging
from . import losses
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from . import sde_lib
import torch
from .utils import restore_checkpoint
from . import sampling

def uncond_gen(
        config
    ):
    """
        Unconditional Generation
    """
    with torch.no_grad():
        eval_dir, ckpt_path = config.eval.eval_dir, config.eval.ckpt_path
        idx = config.eval.idx
        bin_size = config.eval.bin_size
        print(f"idx to save: {idx * bin_size} to {idx * bin_size + bin_size - 1}")
        # Create directory to eval_folder
        os.makedirs(eval_dir, exist_ok=True)

        scaler, inverse_scaler = lambda x: x, lambda x: x

        # Initialize model
        score_model = mutils.create_model(config, use_parallel=False)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Setup SDEs
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

        sampling_eps = 1e-3
        sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.grid_size, config.data.grid_size, config.data.grid_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

        assert os.path.exists(ckpt_path)
        print('ckpt path:', ckpt_path)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            raise
        ema.copy_to(score_model.parameters())

        print(f"loaded model is trained till iter {state['step'] // config.training.num_grad_acc_steps}")


        for k in range(bin_size):
            save_file_path = os.path.join(eval_dir, f"{idx * bin_size + k}")
            print(f'check: {save_file_path}')
            if os.path.exists(save_file_path + '.pt'):
                # continue
                pass
            print(f'will save to: {save_file_path}')
            samples, n = sampling_fn(score_model)
            if type(samples) != tuple:
                print(samples[:, 0].unique())
                torch.save(samples, save_file_path + '.pt')
                samples = samples.cpu().numpy()
                # np.save(save_file_path, samples)
            else:
                print(samples[0][:, 0].unique())
                torch.save(samples[0], save_file_path + '.pt')
                torch.save(samples[1], save_file_path + '_occ.pt')
                # samples, occ = samples[0].cpu().numpy(), samples[1].cpu().numpy()
            # np.save(save_file_path + '.npy, samples)


def slerp(z1, z2, alpha):
    '''
        Spherical Linear Interpolation
    '''
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
            torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
            + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )

def uncond_gen_interp(
        config,
        idx=0,
    ):
    """
        Generation with interpolation between initial noises
        Used for DDIM
    """
    with torch.no_grad():
        eval_dir, ckpt_path = config.eval.eval_dir, config.eval.ckpt_path
        # Create directory to eval_folder
        os.makedirs(eval_dir, exist_ok=True)

        scaler, inverse_scaler = lambda x: x, lambda x: x

        # Initialize model
        score_model = mutils.create_model(config, use_parallel=False)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Setup SDEs
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

        sampling_eps = 1e-3
        sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.grid_size, config.data.grid_size, config.data.grid_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

        assert os.path.exists(ckpt_path)
        print('ckpt path:', ckpt_path)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            raise
        ema.copy_to(score_model.parameters())

        print(f"loaded model is trained till iter {state['step'] // config.training.num_grad_acc_steps}")


        idx = config.eval.idx
        bin_size = config.eval.bin_size
        config.eval.interp_batch_size = 32
        print(f"idx to save: {idx * bin_size} to {idx * bin_size + bin_size - 1}")

        for k in range(bin_size):
            save_file_path = os.path.join(eval_dir, f"{idx * bin_size + k}")

            noise = sde.prior_sampling(
                (2, config.data.num_channels, config.data.grid_size, config.data.grid_size, config.data.grid_size)
            ).to(config.device)
        
            interp_sampling_shape = (config.eval.interp_batch_size,
                            config.data.num_channels,
                            config.data.grid_size, config.data.grid_size, config.data.grid_size)
            x0 = torch.zeros(interp_sampling_shape, device=config.device)
            x0[0] = noise[0]
            x0[-1] = noise[1]
            for i in range(1, config.eval.interp_batch_size - 1):
                x0[i] = slerp(x0[0], x0[-1], i / float(config.eval.interp_batch_size - 1))

            if config.model.use_occ_grid:
                noise_occ = sde.prior_sampling(
                    (2, 1, config.data.grid_size * 2, config.data.grid_size * 2, config.data.grid_size * 2)
                ).to(config.device)
                interp_sampling_shape = (config.eval.interp_batch_size,
                                1,
                                config.data.grid_size * 2, config.data.grid_size * 2, config.data.grid_size * 2)
                x0_occ = torch.zeros(interp_sampling_shape, device=config.device)
                x0_occ[0] = noise_occ[0]
                x0_occ[-1] = noise_occ[1]
                for i in range(1, config.eval.interp_batch_size - 1):
                    x0_occ[i] = slerp(x0_occ[0], x0_occ[-1], i / float(config.eval.interp_batch_size - 1))
            else:
                x0_occ = None

            sample_list = []
            sample_occ_list = []
            for i in tqdm.trange(config.eval.interp_batch_size):
                samples, n = sampling_fn(score_model, x0=x0[i:i+1], x0_occ=x0_occ[i:i+1])
                if type(samples) != tuple:
                    # samples = samples.cpu()
                    sample_list.append(samples.cpu())
                else:
                    # samples = samples.cpu()
                    sample_list.append(samples[0].cpu())
                    sample_occ_list.append(samples[1].cpu())

            # np.save(save_file_path, np.concatenate(sample_list, axis=0))
            torch.save(torch.cat(sample_list, dim=0), save_file_path + '.pt')
            if config.model.use_occ_grid:
                torch.save(torch.cat(sample_occ_list, dim=0), save_file_path + '_occ.pt')


def cond_gen(
        config,
        save_fname='0',
    ):
    """
        Conditional Generation with partially completed dmtet from a 2.5D view (converted into a cubic grid)
    """
    with torch.no_grad():
        eval_dir, ckpt_path = config.eval.eval_dir, config.eval.ckpt_path
        # Create directory to eval_folder
        os.makedirs(eval_dir, exist_ok=True)

        scaler, inverse_scaler = lambda x: x, lambda x: x

        # Initialize model
        score_model = mutils.create_model(config)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Setup SDEs
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

        resolution = config.data.image_size
        grid_mask = torch.load(f'./data/grid_mask_{resolution}.pt').view(1, 1, resolution, resolution, resolution).to("cuda")
        grid_mask = grid_mask[:, :, :config.data.input_size, :config.data.input_size, :config.data.input_size]

        sampling_eps = 1e-3
        sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        # resolution, resolution, resolution)
                        config.data.input_size, config.data.input_size, config.data.input_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, grid_mask=grid_mask)

        assert os.path.exists(ckpt_path)
        print('ckpt path:', ckpt_path)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            raise
        ema.copy_to(score_model.parameters())

        print(f"loaded model is trained till iter {state['step'] // config.training.iter_size}")

        
        save_file_path = os.path.join(eval_dir, f"{save_fname}.npy")

        ### Conditional but free gradients; start from small t

        partial_dict = torch.load(config.eval.partial_dmtet_path)
        partial_sdf = partial_dict['sdf']
        partial_mask = partial_dict['vis']


        ### compute the mapping from tet indices to 3D cubic grid vertex indices
        tet_path = config.eval.tet_path
        tet = np.load(tet_path)
        vertices = torch.tensor(tet['vertices'])
        vertices_unique = vertices[:].unique()
        dx = vertices_unique[1] - vertices_unique[0]

        ind_to_coord = (torch.round(
            (vertices - vertices.min()) / dx)
        ).long()

        
        partial_sdf_grid = torch.zeros((1, 1, resolution, resolution, resolution))
        partial_sdf_grid[0, 0, ind_to_coord[:, 0], ind_to_coord[:, 1], ind_to_coord[:, 2]] = partial_sdf
        partial_mask_grid = torch.zeros((1, 1, resolution, resolution, resolution))
        partial_mask_grid[0, 0, ind_to_coord[:, 0], ind_to_coord[:, 1], ind_to_coord[:, 2]] = partial_mask.float()

        samples, n = sampling_fn(
            score_model, 
            partial=partial_sdf_grid.cuda(), 
            partial_mask=partial_mask_grid.cuda(), 
            freeze_iters=config.eval.freeze_iters
        )

        samples = samples.cpu().numpy()
        np.save(save_file_path, samples)

