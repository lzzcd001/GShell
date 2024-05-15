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

# pylint: skip-file
"""Common layers for defining score networks.
"""
import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .normalization import ConditionalInstanceNorm3dPlus

class GroupNormFloat32(nn.GroupNorm):
    def forward(self, input):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
            return F.group_norm(
                input.float(), self.num_groups, self.weight, self.bias, self.eps)


def get_act_fn(act_name):
    """Get activation functions from the config file."""

    if act_name.lower() == 'elu':
        return nn.ELU()
    elif act_name.lower() == 'relu':
        return nn.ReLU()
    elif act_name.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif act_name.lower() == 'swish' or act_name.lower() == 'silu':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class Dense(nn.Module):
    """Linear layer with `default_init`."""
    def __init__(self):
        super().__init__()


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                    dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

def conv5x5(in_planes, out_planes, stride=2, bias=True, dilation=1, init_scale=1., padding=2):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding,
                    dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def conv3x3_transposed(in_planes, out_planes, stride=2, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                    dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

def conv5x5_transposed(in_planes, out_planes, stride=2, bias=True, dilation=1, init_scale=1., padding=2):
    """3x3 convolution with DDPM initialization."""
    conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding,
                    dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    with torch.no_grad():
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
        half_dim = embedding_dim // 2
        # magic number 10000 is from transformers
        emb = math.log(max_positions) / (half_dim - 1)
        # emb = math.log(2.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
        # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb

class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.GroupNorm_0 = GroupNormFloat32(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.NIN_0 = conv1x1(channels, channels)
        self.NIN_1 = conv1x1(channels, channels)
        self.NIN_2 = conv1x1(channels, channels)
        self.NIN_3 = conv1x1(channels, channels, init_scale=0.)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        # q = q.view(B, C, -1).permute(0, 2, 1)
        # k = k.view(B, C, -1).permute(0, 2, 1)
        # v = v.view(B, C, -1).permute(0, 2, 1)
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
        #     h = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        # h = h.permute(0, 2, 1).view(B, C, D, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
            w = torch.einsum('bcdhw,bckij->bdhwkij', q.float(), k.float()) * (int(C) ** (-0.5))
            w = torch.reshape(w, (B, D, H, W, D * H * W))
            w = F.softmax(w, dim=-1)
            w = torch.reshape(w, (B, D, H, W, D, H, W))
        h = torch.einsum('bdhwkij,bckij->bcdhw', w, v)

        h = self.NIN_3(h)
        return x + h

class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x, temb=None):
        B, C, D, H, W = x.shape
        h = F.interpolate(x.float(), (D * 2, H * 2, W * 2), mode='nearest')
        if self.with_conv:
            h = self.Conv_0(h)
        return h


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = conv3x3(channels, channels, stride=2, padding=0)
            self.with_conv = with_conv

    def forward(self, x, temb=None):
        B, C, D, H, W = x.shape
        # Emulate 'SAME' padding
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)

        assert x.shape == (B, C, D // 2, H // 2, W // 2)
        return x


class ResBlock(nn.Module):
    """The ResNet Blocks used in DDPM."""
    def __init__(self, act_fn, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1, num_groups=32):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = GroupNormFloat32(num_groups=num_groups, num_channels=in_ch, eps=1e-6)
        self.act = act_fn
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = GroupNormFloat32(num_groups=num_groups, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = conv1x1(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        B, C, D, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h

class AttnResBlock(ResBlock):
    """The ResNet Blocks used in DDPM."""
    def __init__(self, act_fn, in_ch, out_ch, temb_dim=None, conv_shortcut=False, dropout=0.1, num_groups=32):
        super().__init__(act_fn, in_ch, out_ch, temb_dim, conv_shortcut, dropout, num_groups=num_groups)
        self.attn_block = AttnBlock(out_ch, num_groups=num_groups)

    def forward(self, x, temb=None):
        h = super().forward(x, temb)
        return self.attn_block(h)
