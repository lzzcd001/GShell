import os
import sys
import numpy as np

import logging
# Keep the import below for registering all model definitions
from .models import unet3d, unet3d_occgrid, unet3d_tet_aware, unet3d_occgrid_v2, unet3d_meshdiffusion

from . import losses
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from . import sde_lib
import torch
from torch.utils import tensorboard
from .utils import save_checkpoint, restore_checkpoint
from ..dataset.gshell_dataset import GShellDataset
from ..dataset.gshell_dataset_aug import GShellAugDataset

from .lion.lion import Lion
import torch.distributed as dist

def train(config):
    """Runs the training pipeline.

    Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    world_size = torch.cuda.device_count()
    device_id = rank % torch.cuda.device_count()

    workdir = config.training.train_dir
    # Create directories for experimental logs
    logging.info("working dir: {:s}".format(workdir))


    tb_dir = os.path.join(workdir, "tensorboard")
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config, ddp=True, rank=rank)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    gradscaler = torch.cuda.amp.GradScaler(growth_interval=config.training.gradscaler_growth_interval)

    state = dict(optimizer=optimizer, model=score_model, ema=ema, gradscaler=gradscaler, step=0)


    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device, rank=rank)
    initial_step = int(state['step'])

    print(f"work dir: {workdir}")

    try:
        use_occ_grid = config.data.use_occ_grid
    except:
        use_occ_grid = False
    if use_occ_grid:
        train_dataset = GShellAugDataset(config)
    else:
        train_dataset = GShellDataset(config.data.dataset_metapath)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=world_size,
    	rank=rank
    )

    try:
        collate_fn = train_dataset.collate
    except:
        collate_fn = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        num_workers=config.data.num_workers,
        # pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn
    )

    data_iter = iter(train_loader)

    print("data loader set")

    # Setup SDEs
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    try:
        use_vis_mask = config.model.use_vis_mask
    except:
        use_vis_mask = False
    print('use_vis_mask', use_vis_mask)
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        loss_type=config.training.loss_type,
                                        pred_type=config.model.pred_type,
                                        use_vis_mask=use_vis_mask,
                                        use_occ=use_occ_grid,
                                        use_aux=config.training.use_aux_loss)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step // config.training.num_grad_acc_steps,))

    iter_size = config.training.num_grad_acc_steps
    epoch = 0
    train_sampler.set_epoch(epoch)
    for step in range(initial_step // iter_size, num_train_steps + 1):
        tmp_loss_dict = {
            'loss_total': 0.0,
            'loss_score': 0.0,
            'loss_reg': 0.0,
        }
        for step_inner in range(iter_size):
            try:
                # batch, batch_mask = next(data_iter)
                batch = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader 
                epoch += 1
                train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if type(batch) == dict:
                for k in batch:
                    batch[k] = batch[k].to(rank, non_blocking=False)
            else:
                batch = batch.to(rank, non_blocking=False)

            # Execute one training step
            clear_grad_flag = (step_inner == 0)
            update_param_flag = (step_inner == iter_size - 1)
            if not update_param_flag:
                with score_model.no_sync():
                    loss_dict = train_step_fn(state, batch, clear_grad=clear_grad_flag, update_param=update_param_flag, gradscaler=gradscaler)
            else:
                loss_dict = train_step_fn(state, batch, clear_grad=clear_grad_flag, update_param=update_param_flag, gradscaler=gradscaler)
            for key in loss_dict:
                tmp_loss_dict[key] += loss_dict[key].item() / iter_size

            # print(torch.cuda.memory_summary())

        if step % config.training.log_freq == 0:
            loss = tmp_loss_dict['loss_total']
            loss = torch.tensor(loss / world_size).to(rank)

            # logging.info("step: %d, training_loss: %.5e" % (step, tmp_loss))
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                loss = loss.item()
                logging.info("step: %d, loss: %.5e, scale: %.5e" % (step, loss, gradscaler.get_scale()))
                sys.stdout.flush()
                writer.add_scalar("loss", loss, step)

        if rank == 0:
            # Save a temporary checkpoint to resume training after pre-emption periodically
            if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
                logging.info(f"save meta at iter {step}")
                save_checkpoint(checkpoint_meta_dir, state)

            # Save a checkpoint periodically and generate samples if needed
            if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
                logging.info(f"save model: {step}-th")
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'), state)

    dist.destroy_process_group()