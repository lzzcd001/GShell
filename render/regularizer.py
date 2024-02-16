# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from render import util
from . import mesh

def luma(x):
    return ((x[..., 0:1] + x[..., 1:2] + x[..., 2:3]) / 3).repeat(1, 1, 1, 3)
def value(x):
    return torch.max(x[..., 0:3], dim=-1, keepdim=True)[0].repeat(1, 1, 1, 3)

def chroma_loss(kd, color_ref, lambda_chroma):
    eps = 0.001
    ref_chroma = color_ref[..., 0:3] / torch.clip(value(color_ref), min=eps)
    opt_chroma = kd[..., 0:3] / torch.clip(value(kd), min=eps)
    return torch.mean(torch.abs((opt_chroma - ref_chroma) * color_ref[..., 3:])) * lambda_chroma

# Diffuse luma regularizer + specular 
def shading_loss(diffuse_light, specular_light, color_ref, lambda_diffuse, lambda_specular):
    diffuse_luma  = luma(diffuse_light)
    specular_luma = luma(specular_light)
    ref_luma      = value(color_ref)
    
    eps = 0.001
    img    = util.rgb_to_srgb(torch.log(torch.clamp((diffuse_luma + specular_luma) * color_ref[..., 3:], min=0, max=65535) + 1))
    target = util.rgb_to_srgb(torch.log(torch.clamp(ref_luma * color_ref[..., 3:], min=0, max=65535) + 1))
    # error  = torch.abs(img - target) * diffuse_luma / torch.clamp(diffuse_luma + specular_luma, min=eps) ### encourage specular component to take control
    error  = torch.abs(img - target) ### the original version in the paper
    loss   = torch.mean(error) * lambda_diffuse
    loss  += torch.mean(specular_luma) / torch.clamp(torch.mean(diffuse_luma), min=eps) * lambda_specular
    return loss

######################################################################################
# Material smoothness loss
######################################################################################

def material_smoothness_grad(kd_grad, ks_grad, nrm_grad, lambda_kd=0.25, lambda_ks=0.1, lambda_nrm=0.0):
    kd_luma_grad = (kd_grad[..., 0] + kd_grad[..., 1] + kd_grad[..., 2]) / 3
    loss  = torch.mean(kd_luma_grad * kd_grad[..., -1]) * lambda_kd
    loss += torch.mean(ks_grad[..., :-1] * ks_grad[..., -1:]) * lambda_ks
    loss += torch.mean(nrm_grad[..., :-1] * nrm_grad[..., -1:]) * lambda_nrm
    return loss

######################################################################################
# Computes the image gradient, useful for kd/ks smoothness losses
######################################################################################
def image_grad(buf, std=0.01):
    t, s = torch.meshgrid(torch.linspace(-1.0 + 1.0 / buf.shape[1], 1.0 - 1.0 / buf.shape[1], buf.shape[1], device="cuda"), 
                          torch.linspace(-1.0 + 1.0 / buf.shape[2], 1.0 - 1.0 / buf.shape[2], buf.shape[2], device="cuda"),
                          indexing='ij')
    tc   = torch.normal(mean=0, std=std, size=(buf.shape[0], buf.shape[1], buf.shape[2], 2), device="cuda") + torch.stack((s, t), dim=-1)[None, ...]
    tap  = dr.texture(buf, tc, filter_mode='linear', boundary_mode='clamp')
    return torch.abs(tap[..., :-1] - buf[..., :-1]) * tap[..., -1:] * buf[..., -1:]

######################################################################################
# Computes the avergage edge length of a mesh. 
# Rough estimate of the tessellation of a mesh. Can be used e.g. to clamp gradients
######################################################################################
def avg_edge_length(v_pos, t_pos_idx):
    e_pos_idx = mesh.compute_edges(t_pos_idx)
    edge_len  = util.length(v_pos[e_pos_idx[:, 0]] - v_pos[e_pos_idx[:, 1]])
    return torch.mean(edge_len)

######################################################################################
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
######################################################################################
def laplace_regularizer_const(v_pos, t_pos_idx):
    term = torch.zeros_like(v_pos)
    norm = torch.zeros_like(v_pos[..., 0:1])

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, t_pos_idx[:, 0:1], two)
    norm.scatter_add_(0, t_pos_idx[:, 1:2], two)
    norm.scatter_add_(0, t_pos_idx[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

######################################################################################
# Smooth vertex normals
######################################################################################
def normal_consistency(v_pos, t_pos_idx):
    # Compute face normals
    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))

    tris_per_edge = mesh.compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(util.dot(n0, n1), min=-1.0, max=1.0)
    term = (1.0 - term) * 0.5

    return torch.mean(torch.abs(term))
