'''
    Storing tet-grid related meta-info into a single file
'''

import numpy as np
import torch
import os
import tqdm
import argparse

from itertools import combinations


def tet_to_grids(vertices, values_list, grid_size):
    grid = torch.zeros(12, grid_size, grid_size, grid_size, device=vertices.device)
    with torch.no_grad():
        for k, values in enumerate(values_list):
            if k == 0:
                grid[k, vertices[:, 0], vertices[:, 1], vertices[:, 2]] = values.squeeze()
            else:
                grid[1:4, vertices[:, 0], vertices[:, 1], vertices[:, 2]] = values.transpose(0, 1)
    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('-res', '--resolution', type=int)
    parser.add_argument('-r', '--root', type=str)
    parser.add_argument('-s', '--source', type=str)
    parser.add_argument('-t', '--target', type=str)
    FLAGS = parser.parse_args()

    tet_path = f'./tets/{FLAGS.resolution}_tets_cropped_reordered.npz'
    tet = np.load(tet_path)
    vertices = torch.tensor(tet['vertices']).cuda()
    indices = torch.tensor(tet['indices']).long().cuda()

    edges = torch.tensor(tet['edges']).long().cuda()
    tet_edges = torch.tensor(tet['tet_edges']).long().view(-1, 2).cuda()

    vertices_unique = vertices[:].unique()
    dx = vertices_unique[1] - vertices_unique[0]
    dx = dx / 2.0 ### denser grid
    vertices_discretized = (
        ((vertices - vertices.min()) / dx)
    ).long()

    midpoints = (vertices_discretized[edges[:, 0]] + vertices_discretized[edges[:, 1]]) / 2.0
    midpoints_dicretized = midpoints.long()

    tet_verts = vertices_discretized[indices.view(-1)].view(-1, 4, 3)
    tet_center = tet_verts.float().mean(dim=1)
    tet_center_discretized = tet_center.long()


    edge_ind_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    msdf_tetedges = []
    msdf_from_tetverts = []
    for i in range(5):
        for j in range(i+1, 6):
            if (edge_ind_list[i][0] == edge_ind_list[j][0]
                or edge_ind_list[i][0] == edge_ind_list[j][1]
                or edge_ind_list[i][1] == edge_ind_list[j][0]
                or edge_ind_list[i][1] == edge_ind_list[j][1]
            ):
                msdf_tetedges.append(i)
                msdf_tetedges.append(j)
                msdf_from_tetverts.extend([edge_ind_list[i][0], edge_ind_list[i][1], edge_ind_list[j][0], edge_ind_list[j][1]])
    msdf_tetedges = torch.tensor(msdf_tetedges)
    msdf_from_tetverts = torch.tensor(msdf_from_tetverts)
    print(msdf_tetedges)
    print(msdf_tetedges.size())



    tet_edges = tet_edges.view(-1, 2)
    msdf_tetedges = msdf_tetedges.view(-1)
    tet_edgenodes_pos = (vertices_discretized[tet_edges[:, 0]] + vertices_discretized[tet_edges[:, 1]]) / 2.0
    tet_edgenodes_pos = tet_edgenodes_pos.view(-1, 6, 2)
    occ_edge_pos = tet_edgenodes_pos[:, msdf_tetedges].view(-1, 12, 2, 3)
    

    edge_twopoint_order = torch.sign(occ_edge_pos[:, :, 0, :] - occ_edge_pos[:, :, 1, :])
    edge_twopoint_order_binary_code = (edge_twopoint_order * torch.tensor([16, 4, 1], device=edge_twopoint_order.device).view(1, 1, -1)).sum(dim=-1)
    edge_twopoint_order_binary_code = torch.stack([edge_twopoint_order_binary_code, -edge_twopoint_order_binary_code], dim=-1)
    _, edge_twopoint_order = edge_twopoint_order_binary_code.sort(dim=-1)

    occ_edge_cano_order = torch.arange(2).view(1, 1, 2).expand(occ_edge_pos.size(0), 12, 2).cuda()
    occ_edge_cano_order = torch.gather(
        input=occ_edge_cano_order,
        dim=-1,
        index=edge_twopoint_order
    )

    tet_edges = tet_edges.view(-1)

    torch.save({
        'tet_v_pos': vertices,
        'tet_edge_vpos': vertices[tet_edges].view(-1, 2, 3),
        'tet_edge_pix_loc': vertices_discretized[tet_edges].view(-1, 2, 3),
        'tet_center_loc': tet_center_discretized,
        'msdf_edges': msdf_tetedges.view(12, 2),
        'occ_edge_cano_order': occ_edge_cano_order
    }, 'tet_info.pt')
