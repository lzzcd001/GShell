import numpy as np
import torch
import os
import tqdm
import argparse

def tet_to_grids(vertices, values_list, grid_size):
    grid = torch.zeros(4, grid_size, grid_size, grid_size, device=vertices.device)
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
    parser.add_argument('-ss', '--split-size', type=int, default=int(1e8))
    parser.add_argument('-ind', '--index', type=int)
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

    print(vertices_discretized.size())
    midpoints = (vertices_discretized[edges[:, 0]] + vertices_discretized[edges[:, 1]]) / 2.0
    midpoints_dicretized = midpoints.long()

    tet_verts = vertices_discretized[indices.view(-1)].view(-1, 4, 3)
    tet_center = tet_verts.float().mean(dim=1)
    tet_center_discretized = tet_center.long()


    global_mask = torch.zeros(4, FLAGS.resolution * 2, FLAGS.resolution * 2, FLAGS.resolution * 2).cuda()
    cat_mask = torch.zeros(FLAGS.resolution * 2, FLAGS.resolution * 2, FLAGS.resolution * 2).cuda()
    global_mask[:4, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]] += 1.0
    cat_mask[vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]] = 1
    global_mask[0, midpoints_dicretized[:, 0], midpoints_dicretized[:, 1], midpoints_dicretized[:, 2]] += 1.0
    cat_mask[midpoints_dicretized[:, 0], midpoints_dicretized[:, 1], midpoints_dicretized[:, 2]] = -1


    torch.save(global_mask, f'global_mask_res{FLAGS.resolution}.pt')
    torch.save(cat_mask, f'cat_mask_res{FLAGS.resolution}.pt')

    save_folder = FLAGS.root

    grid_folder_base = os.path.join(save_folder, FLAGS.target)
    os.makedirs(grid_folder_base, exist_ok=True)

    print(grid_folder_base)

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



    occgrid_mask_already_saved = False
    tets_folder = os.path.join(save_folder, FLAGS.source)

    with torch.no_grad():
        for k in tqdm.trange(FLAGS.split_size):
            global_index = k + FLAGS.index * FLAGS.split_size
            tet_path = os.path.join(tets_folder, 'dmt_dict_{:05d}.pt'.format(global_index))
            if os.path.exists(os.path.join(grid_folder_base, 'grid_{:05d}.pt'.format(global_index))):
                # continue
                pass
            try:
                if os.path.exists(tet_path):
                    tet = torch.load(tet_path, map_location="cuda")

                    sdf = tet['sdf'].view(-1, 1)
                    msdf = tet['msdf'].view(-1, 1)
                    deform = tet['deform']


                    ### resetting sdfs and offsets of all non-mesh-generating tet vertices
                    tet_edges = tet_edges.view(-1, 2)
                    tet_edge_mask = ((torch.sign(sdf[tet_edges[:, 0]]) - torch.sign(sdf[tet_edges[:, 1]])) != 0).bool().squeeze(-1).view(-1, 6)
                    tet_sdf_coeff = (
                        torch.abs(sdf[tet_edges[:, 0]]) 
                        / (torch.abs(sdf[tet_edges[:, 0]] - sdf[tet_edges[:, 1]]) + 1e-10)
                    ).squeeze(-1)
                    tet_sdf_coeff = tet_sdf_coeff.view(-1, 1)
                    midpoint_msdf_tet = msdf[tet_edges[:, 0]] * (1 - tet_sdf_coeff) + msdf[tet_edges[:, 1]] * tet_sdf_coeff
                    midpoint_msdf_tet = midpoint_msdf_tet.view(-1, 6)
                    tet_mask = ((midpoint_msdf_tet > 0) & tet_edge_mask).sum(dim=-1).bool()
                    vert_mask = torch.zeros_like(sdf.squeeze())
                    vert_mask[indices[tet_mask].view(-1)] = 1.0
                    vert_mask = ~vert_mask.bool()
                    msdf[vert_mask] = -1.0
                    deform[vert_mask] = 0.0

                    tet_nonallnegmsdf = (torch.sign(msdf[indices.view(-1)].view(-1, 4)).sum(dim=-1) != -4)
                    vert_mask_nonallnegmsdf = torch.zeros_like(sdf.squeeze())
                    vert_mask_nonallnegmsdf[indices[tet_nonallnegmsdf].view(-1)] = 1.0
                    vert_mask_nonallnegmsdf = ~vert_mask_nonallnegmsdf.bool()
                    sdf[vert_mask_nonallnegmsdf] = 1.0
                    

                    

                    mask = (
                        (torch.sign(sdf[edges[:, 0]]) - torch.sign(sdf[edges[:, 1]]) != 0).bool()
                    )

                    nan_mask = (
                            ((torch.sign(sdf[edges[:, 0]]) + torch.sign(sdf[edges[:, 1]])) == 2)
                            | ((torch.sign(sdf[edges[:, 0]]) + torch.sign(sdf[edges[:, 1]])) == -2) 
                        ).bool().squeeze(-1)

                    original_sdf_coeff = torch.abs(sdf[edges[:, 0]]) / (torch.abs(sdf[edges[:, 0]] - sdf[edges[:, 1]]) + 1e-10)


                    original_sdf_coeff[nan_mask] = torch.nan

                    normalized_sdf_coeff = ((original_sdf_coeff - 0.5) * 2.0)
                    normalized_sdf_coeff = torch.nan_to_num(normalized_sdf_coeff)
                    assert torch.all(normalized_sdf_coeff.abs() <= 1.0)


                    sdf_sign = torch.sign(sdf)
                    sdf_sign[sdf_sign == 0] = 1

                    midpoint_msdf = msdf[edges[:, 0]] * (1 - original_sdf_coeff.view(-1, 1)) + msdf[edges[:, 1]] * original_sdf_coeff.view(-1, 1)
                    midpoint_msdf_sign = torch.sign(midpoint_msdf)
                    midpoint_msdf_sign[midpoint_msdf_sign == 0] = -1
                    midpoint_msdf_sign = midpoint_msdf_sign * mask - (1.0 - mask.float())

                    ############################ Occ Grid ############################


                    tet_edges = tet_edges.view(-1, 2)
                    tet_edge_mask = ((torch.sign(sdf[tet_edges[:, 0]]) - torch.sign(sdf[tet_edges[:, 1]])) != 0).bool().squeeze(-1).view(-1, 6)
                    tet_sdf_coeff = (
                        torch.abs(sdf[tet_edges[:, 0]]) 
                        / (torch.abs(sdf[tet_edges[:, 0]] - sdf[tet_edges[:, 1]]) + 1e-10)
                    ).squeeze(-1)
                    tet_sdf_coeff = tet_sdf_coeff * tet_edge_mask.view(-1)
                    tet_sdf_coeff = tet_sdf_coeff.view(-1, 1)
                    nan_mask = (
                            ((torch.sign(sdf[tet_edges[:, 0]]) + torch.sign(sdf[tet_edges[:, 1]])) == 2)
                            | ((torch.sign(sdf[tet_edges[:, 0]]) + torch.sign(sdf[tet_edges[:, 1]])) == -2) 
                        ).bool().squeeze(-1)
                    tet_sdf_coeff[nan_mask] = torch.nan
                    midpoint_msdf_tet = msdf[tet_edges[:, 0]] * (1 - tet_sdf_coeff) + msdf[tet_edges[:, 1]] * tet_sdf_coeff
                    midpoint_msdf_tet = midpoint_msdf_tet.view(-1, 6)
                    inscribed_edge_twopoint_msdf = midpoint_msdf_tet[:, msdf_tetedges.view(-1)].view(-1, 12, 2)

                    assert ((
                        (tet_edges.view(-1, 6, 2)[:, msdf_tetedges.view(-1), :].view(-1, 24, 2).sum(dim=-1)) - indices[:, msdf_from_tetverts].view(-1, 24, 2).sum(dim=-1)
                    ).sum().item() == 0)

                    assert msdf_tetedges.view(-1).size(0) == 24
                    inscribed_tet_fourpoint_pos = vertices_discretized[indices[:, msdf_from_tetverts].view(-1)].view(-1, 12, 4, 3).to(torch.float64)
                    inscribed_edge_twopoint_pos = inscribed_tet_fourpoint_pos.view(-1, 12, 2, 2, 3).mean(dim=-2)
                    occgrid_loc = inscribed_edge_twopoint_pos.mean(dim=-2)
                    occgrid_loc = (occgrid_loc * 2).to(torch.int64).view(-1, 3)


                    edge_twopoint_order = torch.sign(inscribed_edge_twopoint_pos[:, :, 0, :] - inscribed_edge_twopoint_pos[:, :, 1, :])
                    edge_twopoint_order_binary_code = (edge_twopoint_order * torch.tensor([16, 4, 1], device=edge_twopoint_order.device).view(1, 1, -1)).sum(dim=-1)
                    edge_twopoint_order_binary_code = torch.stack([edge_twopoint_order_binary_code, -edge_twopoint_order_binary_code], dim=-1)
                    _, edge_twopoint_order = edge_twopoint_order_binary_code.sort(dim=-1)

                    inscribed_edge_twopoint_msdf = torch.gather(
                        input=inscribed_edge_twopoint_msdf,
                        dim=-1,
                        index=edge_twopoint_order
                    )

                    mask_msdf = (
                        ((inscribed_edge_twopoint_msdf[:, :, 0] > 0) & (inscribed_edge_twopoint_msdf[:, :, 1] <= 0)) |
                        ((inscribed_edge_twopoint_msdf[:, :, 0] <= 0) & (inscribed_edge_twopoint_msdf[:, :, 1] > 0)) 
                    )
                    msdf_coeff_12 = (
                        torch.abs(inscribed_edge_twopoint_msdf[:, :, 0]) 
                        / (
                            torch.abs(inscribed_edge_twopoint_msdf[:, :, 0] - inscribed_edge_twopoint_msdf[:, :, 1])
                            + 1e-10
                        )
                    )

                    msdf_coeff_12 = (msdf_coeff_12 - 0.5) * 2.0 * mask_msdf
                    msdf_coeff_12 = torch.nan_to_num(msdf_coeff_12)

                    occ_grid = torch.zeros(256, 256, 256, dtype=torch.float, device=msdf_coeff_12.device)
                    occ_grid[occgrid_loc[:, 0], occgrid_loc[:, 1], occgrid_loc[:, 2]] = msdf_coeff_12.view(-1).to(torch.float)

                    if not occgrid_mask_already_saved:
                        occ_grid_mask = torch.zeros(256, 256, 256, dtype=torch.float, device=msdf_coeff_12.device)
                        occ_grid_mask[occgrid_loc[:, 0], occgrid_loc[:, 1], occgrid_loc[:, 2]] = 1
                        torch.save(occ_grid_mask, f'occ_mask_res{FLAGS.resolution}.pt')
                        occgrid_mask_already_saved = True



                    # #################

                    torch.cuda.empty_cache()
                    grid = torch.zeros(4, FLAGS.resolution * 2, FLAGS.resolution * 2, FLAGS.resolution * 2).cuda()
                    grid[0, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]] = sdf_sign.squeeze()
                    grid[1:4, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]] = deform.transpose(0, 1)
                    grid[0, midpoints_dicretized[:, 0], midpoints_dicretized[:, 1], midpoints_dicretized[:, 2]] = midpoint_msdf_sign.squeeze()

                    assert grid.abs().max() <= 1

                    save_path = os.path.join(grid_folder_base, 'grid_{:05d}.pt'.format(global_index))
                    torch.save(grid, save_path)

                    save_path = os.path.join(grid_folder_base, 'occgrid_{:05d}.pt'.format(global_index))
                    torch.save(occ_grid, save_path)
                
            except:
                raise