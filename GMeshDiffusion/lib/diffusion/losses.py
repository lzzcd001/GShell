# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from .models import utils as mutils


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  elif config.optim.optimizer == 'AdamW':
    optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip,
                  gradscaler=None):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      gradscaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    gradscaler.step(optimizer)
    gradscaler.update()
    # optimizer.step()

  return optimize_fn

def get_ddpm_loss_fn(vpsde, train, loss_type='l2', pred_type='noise', use_vis_mask=False, use_occ=False, use_aux=False):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""


  if use_occ:
    def loss_fn(model, batch, use_mesh_reg=False, verts_discretiezd=None, midpoints_discretiezd=None, edges=None):
      batch, batch_occ = batch['grid'], batch['occgrid']
      model_fn = mutils.get_model_fn(model, train=train)
      labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
      sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
      sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
      with torch.no_grad():
        noise = torch.randn_like(batch, device=batch.device)
        noise_occ = torch.randn_like(batch_occ, device=batch.device)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None, None] * batch + \
                        sqrt_1m_alphas_cumprod[labels, None, None, None, None] * noise
        perturbed_data = perturbed_data.type(batch.dtype)
        perturbed_data_occ = sqrt_alphas_cumprod[labels, None, None, None, None] * batch_occ + \
                        sqrt_1m_alphas_cumprod[labels, None, None, None, None] * noise_occ
        perturbed_data_occ = perturbed_data_occ.type(batch_occ.dtype)


      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred, pred_occ = model_fn((perturbed_data, perturbed_data_occ), labels)
    
      pred, pred_occ = pred.float(), pred_occ.float()
      alphas1 = sqrt_alphas_cumprod[labels, None, None, None, None]
      alphas2 = sqrt_1m_alphas_cumprod[labels, None, None, None, None]
      if pred_type == 'noise':
        score = pred
        score_occ = pred_occ
        x0 = (perturbed_data - score * alphas2) / alphas1
        x0_occ = (perturbed_data_occ - score_occ * alphas2) / alphas1
      elif pred_type == 'x0':
        x0 = pred
        x0_occ = pred_occ
        score = (perturbed_data - x0 * alphas1) / alphas2
        score_occ = (perturbed_data_occ - pred_occ * alphas1) / alphas2
      
      # noise = noise[:, :, :score.size(2), :score.size(3), :score.size(4)] ### to accommodate change of size due to arch
      if loss_type == 'l2':
        losses = torch.square(score - noise)
        losses_occ = torch.square(score_occ - noise_occ)
        assert losses_occ.size(1) == 1
      elif loss_type == 'l1':
        raise NotImplementedError
        losses = torch.abs(score - noise)
      else:
        raise NotImplementedError

      mask = model.module.feature_mask
      occ_mask = model.module.occ_mask
      assert len(mask.size()) == 5
      assert mask.size(1) == losses.size(1)
      assert occ_mask.size(1) == losses_occ.size(1)
      assert losses.size(0) == losses_occ.size(0)
      if mask is not None:
        losses = losses * mask
        losses_occ = losses_occ * occ_mask
        occ_loss_scale = 1.0 if not use_aux else 1.0
        loss = (torch.sum(losses) + torch.sum(losses_occ)) / (mask.sum() + occ_mask.sum()) / losses.size(0)
      else:
        raise NotImplementedError
        
      if use_aux:
        pred_vis = model.module.extract_vis_from_cubicgrid(x0, x0_occ)
        with torch.no_grad():
          gt_vis = model.module.extract_vis_from_cubicgrid(batch, batch_occ.view(*x0_occ.size()))
        reg_loss = (
          (pred_vis - gt_vis).pow(2).view(x0.size(0), -1).mean(dim=-1) * sqrt_alphas_cumprod[labels]
        ).mean()
      else:
        reg_loss = torch.zeros_like(loss)
      total_loss = loss + reg_loss

      return total_loss, loss, reg_loss
  else:
    def loss_fn(model, batch, use_mesh_reg=False, verts_discretiezd=None, midpoints_discretiezd=None, edges=None):
      model_fn = mutils.get_model_fn(model, train=train)
      labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
      sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
      sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
      noise = torch.randn_like(batch, device=batch.device)
      perturbed_data = sqrt_alphas_cumprod[labels, None, None, None, None] * batch + \
                      sqrt_1m_alphas_cumprod[labels, None, None, None, None] * noise
      perturbed_data = perturbed_data.type(batch.dtype)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model_fn(perturbed_data, labels)
      pred = pred.float()
      alphas1 = sqrt_alphas_cumprod[labels, None, None, None, None]
      alphas2 = sqrt_1m_alphas_cumprod[labels, None, None, None, None]
      if pred_type == 'noise':
        score = pred
        x0 = (perturbed_data - score * alphas2) / alphas1
      elif pred_type == 'x0':
        x0 = pred
        score = (perturbed_data - x0 * alphas1) / alphas2
      
      if use_vis_mask:
        assert x0.size(0) == 1
        vis_mask = model.extract_vismask_from_cubicgrid(x0)
        # noise = noise[:, :, :score.size(2), :score.size(3), :score.size(4)] ### to accommodate change of size due to arch
        if loss_type == 'l2':
          losses = torch.square((score - noise) * vis_mask)
        elif loss_type == 'l1':
          losses = torch.abs((score - noise) * vis_mask)
        else:
          raise NotImplementedError
      else:
        if loss_type == 'l2':
          losses = torch.square(score - noise)
        elif loss_type == 'l1':
          losses = torch.abs(score - noise)
        else:
          raise NotImplementedError

      mask = model.module.feature_mask
      assert len(mask.size()) == 5
      assert mask.size(1) == losses.size(1)
      if mask is not None:
        losses = losses * mask
        loss = torch.sum(losses) / mask.sum() / losses.size(0)
      else:
        raise NotImplementedError


      reg_loss = torch.zeros_like(loss)
      total_loss = loss

      return total_loss, loss, reg_loss

  return loss_fn

def get_step_fn(sde, train, optimize_fn=None, loss_type='l2', pred_type='noise', use_vis_mask=False, use_occ=False, use_aux=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  
  loss_fn = get_ddpm_loss_fn(sde, train, loss_type=loss_type, pred_type=pred_type, use_vis_mask=use_vis_mask, use_occ=use_occ, use_aux=use_aux)

  def step_fn(state, batch, clear_grad=True, update_param=True, gradscaler=None):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      if clear_grad:
        optimizer.zero_grad()
      loss_total, loss_score, loss_reg = loss_fn(model, batch)
      gradscaler.scale(loss_total).backward()
      if update_param:
        optimize_fn(optimizer, model.parameters(), step=state['step'], gradscaler=gradscaler)
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss_total, loss_score, loss_reg = loss_fn(model, batch)
        ema.restore(model.parameters())

    return {
      'loss_total': loss_total,
      'loss_score': loss_score,
      'loss_reg': loss_reg,
    }

  return step_fn