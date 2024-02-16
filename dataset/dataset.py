# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0) if 'img' in batch[0] else None,
            'img_second' : torch.cat(list([item['img_second'] for item in batch]), dim=0) if 'img_second' in batch[0] else None,
            'invdepth' : torch.cat(list([item['invdepth'] for item in batch]), dim=0)if 'invdepth' in batch[0] else None,
            'invdepth_second' : torch.cat(list([item['invdepth_second'] for item in batch]), dim=0) if 'invdepth_second' in batch[0] else None,
            'envlight_transform': torch.cat(list([item['envlight_transform'] for item in batch]), dim=0) if 'envlight_transform' in batch and batch[0]['envlight_transform'] is not None else None,
        }