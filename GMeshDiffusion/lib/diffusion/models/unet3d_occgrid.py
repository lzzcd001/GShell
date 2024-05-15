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
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools
import numpy as np

from . import utils
from .layers import ResBlock, AttnResBlock, Upsample, Downsample, conv1x1, conv3x3, conv5x5, get_act_fn, default_init, get_timestep_embedding, GroupNormFloat32

import sys


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


@utils.register_model(name='unet3d_occgrid')
class UNet3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn = get_act_fn(config.model.act_fn)
        self.nf = nf = config.model.base_channels
        data_ch = config.data.num_channels
        ch_mult = config.model.ch_mult
        feature_mask = torch.load(config.model.feature_mask_path, map_location='cpu').view(1, data_ch, 128, 128, 128)
        pixcat_mask = torch.load(config.model.pixcat_mask_path, map_location='cpu').view(1, 1, 128, 128, 128)

        occ_mask_path = config.data.occ_mask_path
        occ_mask = torch.load(occ_mask_path, map_location='cpu').view(1, 1, 256, 256, 256)


        tet_info = torch.load(config.data.tet_info_path)
        self.tet_edge_vpos = tet_info['tet_edge_vpos'].cuda()
        self.tet_edge_pix_loc = tet_info['tet_edge_pix_loc'].cuda().view(-1, 2, 3)
        self.tet_edge_pix_loc = self.tet_edge_pix_loc.view(-1, 2, 3)
        # self.tet_center_loc = tet_info['tet_center_loc'].cuda()
        self.vis_edges = tet_info['vis_edges'].cuda()
        self.occ_edge_cano_order = tet_info['occ_edge_cano_order'].cuda()
        self.tet_edgenode_loc = self.tet_edge_pix_loc.float().mean(dim=1).long()
        self.occ_edge_loc = self.tet_edgenode_loc.view(-1, 6, 3)[:, self.vis_edges.view(-1)].view(-1, 2, 3)
        self.occ_node_loc = (self.occ_edge_loc.view(-1, 12, 2, 3).float().mean(dim=-2) * 2.0).long().view(-1, 3)
        print(self.tet_edgenode_loc.size(), self.vis_edges.size(), self.occ_edge_loc.size(), self.occ_node_loc.size())
        self.tet_edge_pix_loc = self.tet_edge_pix_loc.view(-1, 3)
        
        
        self.feature_mask = torch.nn.Parameter(feature_mask, requires_grad=False)
        self.pixcat_mask = torch.nn.Parameter(pixcat_mask, requires_grad=False)
        self.occ_mask = torch.nn.Parameter(occ_mask, requires_grad=False)
        self.down_block_types = config.model.down_block_types
        self.up_block_types = config.model.up_block_types
        self.num_res_blocks = config.model.num_res_blocks
        self.num_res_blocks_1st_layer = config.model.num_res_blocks_1st_layer
        resamp_with_conv = config.model.resamp_with_conv
        dropout = config.model.dropout
        assert len(self.down_block_types) == len(self.up_block_types)


        module_dict = {
            module: functools.partial(str_to_class(module), act_fn=self.act_fn, temb_dim=4 * nf, dropout=dropout)
            for module in ["ResBlock", "AttnResBlock"]
        }


        # Condition on noise levels.
        noise_temb_layers = [nn.Linear(nf, nf * 4), nn.SiLU(), nn.Linear(nf * 4, nf * 4)]
        noise_temb_layers[0].weight.data = default_init()(noise_temb_layers[0].weight.data.shape)
        nn.init.zeros_(noise_temb_layers[0].bias)
        noise_temb_layers[2].weight.data = default_init()(noise_temb_layers[2].weight.data.shape)
        nn.init.zeros_(noise_temb_layers[2].bias)
        self.noise_temb_layers = nn.Sequential(*noise_temb_layers)

        self.occ_conv = conv3x3(1, nf, stride=2, padding=1)
        self.occ_mask_conv = conv3x3(1, nf, stride=2, padding=1)

        # Downsampling block
        self.mask_layer = conv5x5(1, nf, stride=1, padding=2)
        self.input_layer = conv5x5(data_ch, nf, stride=1, padding=2)
        hs_c = [nf]
        in_ch = nf
        
        modules = []
        for i_level, down_block_type in enumerate(self.down_block_types):
            curr_block = module_dict[down_block_type]
            # Residual blocks for this resolution
            num_res_blocks = self.num_res_blocks_1st_layer if i_level == 0 else self.num_res_blocks
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(curr_block(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                hs_c.append(in_ch)
        
            if i_level != len(self.down_block_types) - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(module_dict["AttnResBlock"](in_ch=in_ch, out_ch=in_ch))
        modules.append(module_dict["ResBlock"](in_ch=in_ch))

        # Upsampling block
        for i_level, up_block_type in enumerate(self.up_block_types):
            curr_block = module_dict[up_block_type]
            num_res_blocks = self.num_res_blocks_1st_layer if i_level == len(self.up_block_types) - 1 else self.num_res_blocks
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[len(self.up_block_types) - i_level - 1]
                modules.append(curr_block(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if i_level != len(self.up_block_types) - 1:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        self.all_modules = nn.ModuleList(modules)

        self.output_norm_layer = nn.Sequential(
            GroupNormFloat32(num_channels=in_ch, num_groups=32, eps=1e-6),
            nn.SiLU(),
        )
        self.output_layer = conv5x5(in_ch, data_ch, init_scale=0., stride=1, padding=2)


        self.occ_output_layer = nn.ConvTranspose3d(in_ch, 1, 4, stride=2, padding=1)

    def sequentially_call_module(self, idx, x, temb=None):
        return idx + 1, self.all_modules[idx](x, temb)

    def forward(self, x, labels):
        modules = self.all_modules

        with torch.no_grad():
            x, occ_grid = x[0], x[1]
            if True or self.centered:
                # Input is in [-1, 1]
                x = x
            else:
                # Input is in [0, 1]
                x = 2 * x - 1.

            # Mask out unused values
            x = x * self.feature_mask

            occ_grid = occ_grid * self.occ_mask
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
            # timestep/scale embedding
            timesteps = labels
            temb = get_timestep_embedding(timesteps.float(), self.nf)
            temb = self.noise_temb_layers(temb)

        # Downsampling block
        hs = [self.input_layer(x) + self.mask_layer(self.pixcat_mask) + self.occ_conv(occ_grid) + self.occ_mask_conv(self.occ_mask)]

        m_idx = 0
        for i_level in range(len(self.down_block_types)):
            num_res_blocks = self.num_res_blocks_1st_layer if i_level == 0 else self.num_res_blocks
            for i_block in range(num_res_blocks):
                m_idx, h = self.sequentially_call_module(m_idx, hs[-1], temb)
                hs.append(h)
            if i_level != len(self.down_block_types) - 1:
                m_idx, h = self.sequentially_call_module(m_idx, hs[-1])
                hs.append(h)

        h = hs[-1]
        m_idx, h = self.sequentially_call_module(m_idx, h, temb)
        m_idx, h = self.sequentially_call_module(m_idx, h, temb)

        # Upsampling block
        for i_level in range(len(self.up_block_types)):
            num_res_blocks = self.num_res_blocks_1st_layer if i_level == len(self.up_block_types) - 1 else self.num_res_blocks
            for i_block in range(num_res_blocks + 1):
                hspop = hs.pop()
                h = torch.cat([h, hspop], dim=1)
                m_idx, h = self.sequentially_call_module(m_idx, h, temb)
            if i_level != len(self.up_block_types) - 1:
                m_idx, h = self.sequentially_call_module(m_idx, h, temb)

        assert not hs
        h = self.output_norm_layer(h)
        grid = self.output_layer(h)
        grid_occ = self.occ_output_layer(h)

        # Mask out unused values
        grid = grid * self.feature_mask
        grid_occ = grid_occ * self.occ_mask

        return grid, grid_occ
