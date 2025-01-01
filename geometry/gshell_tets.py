import numpy as np
import torch

from render import util

######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(v_pos, t_pos_idx):

    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm, t_pos_idx

######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(v_pos, v_tex, v_nrm, t_pos_idx, t_tex_idx, t_nrm_idx):
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = v_pos[t_pos_idx[:, i]]
        tex[i] = v_tex[t_tex_idx[:, i]]
        vn_idx[i] = t_nrm_idx[:, i]

    tangents = torch.zeros_like(v_nrm)
    tansum   = torch.zeros_like(v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # Avoid dimsdfion by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i][:, None].repeat(1,3)
        tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, v_nrm) * v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return tangents, t_nrm_idx

class GShell_Tets:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.mesh_edge_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2,  1, -1, -1],
                [ 4,  0,  3,  4, -1, -1],
                [ 1,  3,  4,  2,  1, -1],
                [ 3,  1,  5,  3, -1, -1],
                [ 2,  5,  3,  0,  2, -1],
                [ 1,  5,  4,  0,  1, -1],
                [ 4,  2,  5,  4, -1, -1],
                [ 4,  5,  2,  4, -1, -1],
                [ 4,  5,  1,  0,  4, -1],
                [ 3,  5,  2,  0,  3, -1],
                [ 1,  3,  5,  1, -1, -1],
                [ 4,  3,  1,  2,  4, -1],
                [ 3,  0,  4,  3, -1, -1],
                [ 2,  0,  1,  2, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')


        self.triangle_table_tri = torch.tensor([
            ## 000
                [-1, -1, -1, -1, -1, -1],
            ## 001
                [ 4,  2,  5, -1, -1, -1],
            ## 010
                [ 3,  1,  4, -1, -1, -1],
            ## 011
                [ 3,  1,  2,  3,  2,  5],
            ## 100
                [ 0,  3,  5, -1, -1, -1],
            ## 101
                [ 0,  3,  4,  0,  4,  2],
            ## 110
                [ 0,  1,  4,  0,  4,  5],
            ## 111
                [ 0,  1,  2, -1, -1, -1],
        ], dtype=torch.long, device='cuda')

        self.triangle_table_quad = torch.tensor([
            ### in the order of [0, 1, 2, 3]
            ### so 1000 corresponds to single positive mSDF vertex of index 0
            ## 0000
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ## 0001
                [ 6,  3,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ## 0010
                [ 5,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ## 0011
                [ 5,  2,  7,  3,  7,  2, -1, -1, -1, -1, -1, -1],
            ## 0100
                [ 4,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ## 0101
                [ 4,  1,  5,  4,  5,  7,  5,  6,  7,  7,  6,  3],
            ## 0110
                [ 4,  1,  2,  6,  4,  2, -1, -1, -1, -1, -1, -1],
            ## 0111
                [ 4,  1,  2,  7,  4,  2,  7,  2,  3, -1, -1, -1],
            ## 1000
                [ 0,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ## 1001
                [ 0,  4,  6,  3,  0,  6, -1, -1, -1, -1, -1, -1],
            ## 1010
                [ 0,  4,  5,  0,  5,  2,  0,  2,  6,  0,  6,  7],
            ## 1011
                [ 0,  4,  5,  0,  5,  2,  0,  2,  3, -1, -1, -1],
            ## 1100
                [ 0,  1,  5,  7,  0,  5, -1, -1, -1, -1, -1, -1],
            ## 1101
                [ 0,  1,  5,  0,  5,  6,  0,  6,  3, -1, -1, -1],
            ## 1110
                [ 0,  1,  2,  0,  2,  6,  0,  6,  7, -1, -1, -1],
            ## 1111
                [ 0,  1,  2,  0,  2,  3, -1, -1, -1, -1, -1, -1],
        ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

        self.num_triangles_tri_table = torch.tensor([0,1,1,2,1,2,2,1], dtype=torch.long, device='cuda')
        self.num_triangles_quad_table = torch.tensor([0,1,1,2,1,4,2,3,1,2,4,3,2,3,3,2], dtype=torch.long, device='cuda')

        edge_ind_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        msdf_from_tetverts = []
        for i in range(5):
            for j in range(i+1, 6):
                if (edge_ind_list[i][0] == edge_ind_list[j][0]
                    or edge_ind_list[i][0] == edge_ind_list[j][1]
                    or edge_ind_list[i][1] == edge_ind_list[j][0]
                    or edge_ind_list[i][1] == edge_ind_list[j][1]
                ):
                    msdf_from_tetverts.extend([edge_ind_list[i][0], edge_ind_list[i][1], edge_ind_list[j][0], edge_ind_list[j][1]])

        self.msdf_from_tetverts = torch.tensor(msdf_from_tetverts)

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, msdf_n, tet_fx4, output_watertight_template=True):
        sdf_n = sdf_n.float()
        with torch.no_grad():
            ### To determine if tets are valid
            ### Step 1: SDF criteria
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)


            ### Step 2: pre-filtering with mSDF - mSDF cannot be all non-negative
            msdf_fx4 = msdf_n[tet_fx4.reshape(-1)].reshape(-1,4)
            msdf_sign_fx4 = msdf_fx4 > 0
            msdf_sign_sum = torch.sum(msdf_sign_fx4, -1)

            if output_watertight_template:
                valid_tets = (occ_sum>0) & (occ_sum<4) 
            else:
                valid_tets = (occ_sum>0) & (occ_sum<4) & (msdf_sign_sum > 0)

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)  

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim = True)
        denominator = torch.sign(denominator) * (denominator.abs() + 1e-12)
        denominator[denominator == 0] = 1e-12

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        msdf_to_interp = msdf_n[interp_v.reshape(-1)].reshape(-1,2)
        msdf_vert = (msdf_to_interp * edges_to_interp_sdf.squeeze(-1)).sum(1)
        msdf_vert_stopvgd = (msdf_to_interp * edges_to_interp_sdf.squeeze(-1).detach()).sum(1)


        # (M, 6), M: num of pre-filtered tets, storing indices (besides -1) from 0 to num_mask_edges
        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        # triangle count
        num_triangles = self.num_triangles_table[tetindex]

        # Get global face index (static, does not depend on topology), before mSDF processing
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx_pre = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        # Get uv before mSDF processing
        uvs_pre, uv_idx_pre = self.map_uv(face_gidx_pre, num_tets*2)

        # Generate triangle indices before msdf processing
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        v_nrm, t_nrm_idx = auto_normals(verts, faces)
        v_tng, _ = compute_tangents(verts, uvs_pre, v_nrm, faces, faces, faces)
        
        ###### Triangulation with mSDF
        ### Note: we allow area-0 triangular faces for convenience. Can always remove them during post-processing
        with torch.no_grad():
            mesh_edge_tri = torch.gather(input=idx_map[num_triangles == 1], dim=1, 
                    index=self.mesh_edge_table[tetindex[num_triangles == 1]][:, [0, 1, 1, 2, 2, 0]]
                ).view(-1, 3, 2)
            mesh_edge_quad = torch.gather(input=idx_map[num_triangles == 2], dim=1, 
                    index=self.mesh_edge_table[tetindex[num_triangles == 2]][:, [0, 1, 1, 2, 2, 3, 3, 0]]
                ).view(-1, 4, 2)
            mocc_fx3 = (msdf_vert[mesh_edge_tri[:, :, 0].reshape(-1)].reshape(-1, 3) > 0).long()
            mocc_fx4 = (msdf_vert[mesh_edge_quad[:, :, 0].reshape(-1)].reshape(-1, 4) > 0).long()


        ### Attributes to be interpolated for (non-watertight) mesh vertices on the boundary
        edges_to_interp_vpos_tri = verts[mesh_edge_tri.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_vpos_quad = verts[mesh_edge_quad.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_tng_tri = v_tng[mesh_edge_tri.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_tng_quad = v_tng[mesh_edge_quad.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_msdf_tri = msdf_vert[mesh_edge_tri.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_msdf_quad = msdf_vert[mesh_edge_quad.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_msdf_tri_stopvgd = msdf_vert_stopvgd[mesh_edge_tri.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_msdf_quad_stopvgd = msdf_vert_stopvgd[mesh_edge_quad.reshape(-1)].reshape(-1,2,1)


        ### Linear interpolation on mesh edges (triangle / quad faces)
        denominator_tri_nonzero = torch.sign(edges_to_interp_msdf_tri[:,:,0]).sum(dim=1).abs() != 2
        denominator_quad_nonzero = torch.sign(edges_to_interp_msdf_quad[:,:,0]).sum(dim=1).abs() != 2

        edges_to_interp_msdf_tri[:,-1] *= -1
        edges_to_interp_msdf_quad[:,-1] *= -1
        denominator_tri = edges_to_interp_msdf_tri.sum(1, keepdim=True)
        denominator_quad = edges_to_interp_msdf_quad.sum(1, keepdim=True)

        denominator_tri_nonzero = (denominator_tri[:,0,0].abs() > 1e-12) & denominator_tri_nonzero
        denominator_quad_nonzero = (denominator_quad[:,0,0].abs() > 1e-12) & denominator_quad_nonzero



        edges_to_interp_msdf_tri_new = torch.zeros_like(edges_to_interp_msdf_tri)
        edges_to_interp_msdf_quad_new = torch.zeros_like(edges_to_interp_msdf_quad)
        edges_to_interp_msdf_tri_new[denominator_tri_nonzero] = torch.flip(edges_to_interp_msdf_tri[denominator_tri_nonzero], [1]) / denominator_tri[denominator_tri_nonzero]
        edges_to_interp_msdf_quad_new[denominator_quad_nonzero] = torch.flip(edges_to_interp_msdf_quad[denominator_quad_nonzero], [1]) / denominator_quad[denominator_quad_nonzero]

        edges_to_interp_msdf_tri = edges_to_interp_msdf_tri_new
        edges_to_interp_msdf_quad = edges_to_interp_msdf_quad_new

        ### Append additional boundary vertices (with negligible corner cases). Notice that unused vertices are included for efficiency reasons.
        verts_aug = torch.cat([
                    verts,
                    (edges_to_interp_vpos_tri * edges_to_interp_msdf_tri).sum(1), 
                    (edges_to_interp_vpos_quad * edges_to_interp_msdf_quad).sum(1)
                ],
            dim=0)

        v_tng_aug = torch.cat([
                    v_tng,
                    (edges_to_interp_tng_tri * edges_to_interp_msdf_tri).sum(1), 
                    (edges_to_interp_tng_quad * edges_to_interp_msdf_quad).sum(1)
                ],
            dim=0)

        ### NOTE: important to stop gradients from passing through the 'interpolation coefficients' (basically the 'coordinates' of boundary vertices)
        msdf_vert_tri_stopvgd = (edges_to_interp_msdf_tri_stopvgd * edges_to_interp_msdf_tri.detach()).sum(1).squeeze(dim=-1)
        msdf_vert_quad_stopvgd = (edges_to_interp_msdf_quad_stopvgd * edges_to_interp_msdf_quad.detach()).sum(1).squeeze(dim=-1)

        msdf_vert_aug_stopvgd = torch.cat([
            msdf_vert_stopvgd,
            msdf_vert_tri_stopvgd,
            msdf_vert_quad_stopvgd,
        ])

        msdf_vert_boundary_stopvgd = msdf_vert_aug_stopvgd[msdf_vert.size(0):] ## not all boundary vertices but good enough

        ### Determine how to cut polygon faces by checking the look-up tables
        with torch.no_grad():
            v_id_msdf_tri = torch.flip(torch.pow(2, torch.arange(3, dtype=torch.long, device="cuda")), dims=[0])
            v_id_msdf_quad = torch.flip(torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda")), dims=[0])
            mesh_index_tri = (mocc_fx3 * v_id_msdf_tri.unsqueeze(0)).sum(-1)
            mesh_index_quad = (mocc_fx4 * v_id_msdf_quad.unsqueeze(0)).sum(-1)


        idx_map_tri = torch.cat([mesh_edge_tri[:, :, 0], verts.size(0) + torch.arange(mesh_edge_tri.size(0) * 3, device='cuda').view(-1, 3)], dim=-1)
        idx_map_quad = torch.cat([mesh_edge_quad[:, :, 0], verts.size(0) + mesh_edge_tri.size(0) * 3 + torch.arange(mesh_edge_quad.size(0) * 4, device='cuda').view(-1, 4)], dim=-1)

        num_triangles_tri = self.num_triangles_tri_table[mesh_index_tri]
        num_triangles_quad = self.num_triangles_quad_table[mesh_index_quad]

        ### Cut the polygon faces (case-by-case)
        faces_aug = torch.cat((
            torch.gather(input=idx_map_tri[num_triangles_tri == 1], dim=1, index=self.triangle_table_tri[mesh_index_tri[num_triangles_tri == 1]][:, :3]).view(-1, 3),
            torch.gather(input=idx_map_tri[num_triangles_tri == 2], dim=1, index=self.triangle_table_tri[mesh_index_tri[num_triangles_tri == 2]][:, :6]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 1], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 1]][:, :3]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 2], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 2]][:, :6]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 3], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 3]][:, :9]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 4], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 4]][:, :12]).view(-1, 3),
        ), dim=0)

        ### Mark all unused vertices (only for convenience in visualization; not necessary)
        with torch.no_grad():
            referenced_vert_idx = faces_aug.unique()
            mask = torch.ones(verts_aug.size(0))
            mask[referenced_vert_idx] = 0
        verts_aug[mask.bool()] = 0


        if output_watertight_template:
            extra = {
                'n_verts_watertight': verts.size(0),
                'vertices_watertight': verts,
                'faces_watertight': faces, 
                'v_tng_watertight': v_tng,
                'msdf': msdf_vert_aug_stopvgd,
                'msdf_watertight': msdf_vert_stopvgd,
                'msdf_boundary': msdf_vert_boundary_stopvgd,
            }
        else:
            extra = {
                'msdf': msdf_vert_aug_stopvgd,
                'msdf_watertight': msdf_vert_stopvgd,
                'msdf_boundary': msdf_vert_boundary_stopvgd,
            }

        return verts_aug, faces_aug, None, None, v_tng_aug, extra
    

    @torch.no_grad()
    def marching_from_auggrid(self, pos_nx3, sdf_n, tet_fx4, 
                          sorted_tet_edges_fx6x2, coeff_sdf_interp, verts_discretized, 
                          midpoint_msdf_sign_n, occgrid
                          ):
        sdf_n = sdf_n.float()
        ### To determine if tets are valid
        ### Step 1: SDF criteria
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
        occ_sum = torch.sum(occ_fx4, -1)

        valid_tets = (occ_sum>0) & (occ_sum<4)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)


        # find all vertices
        all_edges = sorted_tet_edges_fx6x2.reshape(-1, 6, 2)[valid_tets].reshape(-1, 2)
        all_edges = all_edges.view(-1, 1, 2)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)


        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device="cuda")
        idx_map = mapping[idx_map] # map edges to verts

        interp_v = unique_edges[mask_edges]


        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_canonical = verts_discretized[interp_v.reshape(-1)].reshape(-1,2,3).float()
        verts_canonical = (edges_to_interp_canonical[:, 0] + edges_to_interp_canonical[:, 1]) / 2.0

        tetedge_cano_midpts = verts_discretized[interp_v.reshape(-1)].float().reshape(-1,2,3).mean(dim=1).long()

        coeff_sdf_interp = coeff_sdf_interp[tetedge_cano_midpts[:, 0], tetedge_cano_midpts[:, 1], tetedge_cano_midpts[:, 2]].view(-1, 1).clamp(0, 1)
        verts = edges_to_interp[:, 1] * coeff_sdf_interp + edges_to_interp[:, 0] * (1 - coeff_sdf_interp)

        msdf_vert = midpoint_msdf_sign_n[tetedge_cano_midpts[:, 0], tetedge_cano_midpts[:, 1], tetedge_cano_midpts[:, 2]]

        # (M, 6), M: num of pre-filtered tets, storing indices (besides -1) from 0 to num_mask_edges
        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        # triangle count
        num_triangles = self.num_triangles_table[tetindex]

        # Get global face index (static, does not depend on topology), before mSDF processing
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx_pre = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        valid_tet_gidx = torch.cat([tet_gidx[num_triangles == 1], tet_gidx[num_triangles == 2]], dim=0)

        # Get uv before mSDF processing
        uvs_pre, uv_idx_pre = self.map_uv(face_gidx_pre, num_tets*2)

        # Generate triangle indices before vis processing
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        v_nrm, t_nrm_idx = auto_normals(verts, faces)
        v_tng, _ = compute_tangents(verts, uvs_pre, v_nrm, faces, faces, faces)

        ###### Triangulation with mSDF
        # edge_indices_tri = self.pre_mesh_edge_table[tetindex[num_triangles == 1]][:, [0, 1, 1, 2, 2, 0]]
        # edge_indices_quad = self.pre_mesh_edge_table[tetindex[num_triangles == 2]][:, [0, 1, 1, 2, 2, 3, 3, 0]]
        # pre_mesh_edge_tri = torch.gather(input=idx_map[num_triangles == 1], dim=1, 
        #         index=edge_indices_tri
        #     ).view(-1, 3, 2)
        # pre_mesh_edge_quad = torch.gather(input=idx_map[num_triangles == 2], dim=1, 
        #         index=edge_indices_quad
        #     ).view(-1, 4, 2)
                              
        pre_mesh_edge_tri = torch.gather(input=idx_map[num_triangles == 1], dim=1, 
                index=self.mesh_edge_table[tetindex[num_triangles == 1]][:, [0, 1, 1, 2, 2, 0]]
            ).view(-1, 3, 2)
        pre_mesh_edge_quad = torch.gather(input=idx_map[num_triangles == 2], dim=1, 
                index=self.mesh_edge_table[tetindex[num_triangles == 2]][:, [0, 1, 1, 2, 2, 3, 3, 0]]
            ).view(-1, 4, 2)

        msdf_positive_fx3 = (msdf_vert[pre_mesh_edge_tri[:, :, 0].reshape(-1)].reshape(-1, 3) > 0).long()
        msdf_positive_fx4 = (msdf_vert[pre_mesh_edge_quad[:, :, 0].reshape(-1)].reshape(-1, 4) > 0).long()

                              


        edges_to_interp_prevert_tri = verts[pre_mesh_edge_tri.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_prevert_quad = verts[pre_mesh_edge_quad.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_pretng_tri = v_tng[pre_mesh_edge_tri.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_pretng_quad = v_tng[pre_mesh_edge_quad.reshape(-1)].reshape(-1,2,3)


        edges_to_interp_sort_tri = verts_canonical[pre_mesh_edge_tri.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sort_quad = verts_canonical[pre_mesh_edge_quad.reshape(-1)].reshape(-1,2,3)


        meshocc_loc_tri = (edges_to_interp_sort_tri.mean(dim=1) * 2.0).long()
        meshocc_loc_quad = (edges_to_interp_sort_quad.mean(dim=1) * 2.0).long()


        msdf_coeff_tri = occgrid[meshocc_loc_tri[:, 0], meshocc_loc_tri[:, 1], meshocc_loc_tri[:, 2]] * 0.5 + 0.5
        msdf_coeff_quad = occgrid[meshocc_loc_quad[:, 0], meshocc_loc_quad[:, 1], meshocc_loc_quad[:, 2]] * 0.5 + 0.5


        msdf_coeff_tri = torch.stack([msdf_coeff_tri, 1 - msdf_coeff_tri], dim=-1)
        msdf_coeff_quad = torch.stack([msdf_coeff_quad, 1 - msdf_coeff_quad], dim=-1)


        inscribed_edge_twopoint_order_tri = torch.sign(edges_to_interp_sort_tri[:, 0, :] - edges_to_interp_sort_tri[:, 1, :])
        inscribed_edge_twopoint_order_tri = (inscribed_edge_twopoint_order_tri * torch.tensor([16, 4, 1], device=inscribed_edge_twopoint_order_tri.device).view(1, -1)).sum(dim=-1)
        inscribed_edge_twopoint_order_tri = torch.stack([inscribed_edge_twopoint_order_tri, -inscribed_edge_twopoint_order_tri], dim=-1)
        _, inscribed_edge_twopoint_order_tri = inscribed_edge_twopoint_order_tri.sort(dim=-1, descending=True)

        inscribed_edge_twopoint_order_quad = torch.sign(edges_to_interp_sort_quad[:, 0, :] - edges_to_interp_sort_quad[:, 1, :])
        inscribed_edge_twopoint_order_quad = (inscribed_edge_twopoint_order_quad * torch.tensor([16, 4, 1], device=inscribed_edge_twopoint_order_quad.device).view(1, -1)).sum(dim=-1)
        inscribed_edge_twopoint_order_quad = torch.stack([inscribed_edge_twopoint_order_quad, -inscribed_edge_twopoint_order_quad], dim=-1)
        _, inscribed_edge_twopoint_order_quad = inscribed_edge_twopoint_order_quad.sort(dim=-1, descending=True)

        msdf_coeff_tri = torch.gather(
            input=msdf_coeff_tri, 
            dim=-1, 
            index=inscribed_edge_twopoint_order_tri.view(-1, 2)
        ).view(-1, 2, 1)

        msdf_coeff_quad = torch.gather(
            input=msdf_coeff_quad, 
            dim=-1, 
            index=inscribed_edge_twopoint_order_quad.view(-1, 2)
        ).view(-1, 2, 1)

        msdf_coeff_tri = msdf_coeff_tri.view(-1, 2, 1)
        msdf_coeff_quad = msdf_coeff_quad.view(-1, 2, 1)


        verts_aug = torch.cat([
                    verts,
                    (edges_to_interp_prevert_tri * msdf_coeff_tri).sum(1), 
                    (edges_to_interp_prevert_quad * msdf_coeff_quad).sum(1),
                ],
            dim=0)

        v_tng_aug = torch.cat([
                    v_tng,
                    (edges_to_interp_pretng_tri * msdf_coeff_tri).sum(1), 
                    (edges_to_interp_pretng_quad * msdf_coeff_quad).sum(1),
                ],
            dim=0)

        msdf_vert_aug = torch.cat([
            msdf_vert,
            torch.zeros(v_tng_aug.size(0) - v_tng.size(0)).cuda()
        ])

        v_id_msdf_tri = torch.flip(torch.pow(2, torch.arange(3, dtype=torch.long, device="cuda")), dims=[0]) ## do this flip because the triangle table uses a different assumption by mistake..
        v_id_msdf_quad = torch.flip(torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda")), dims=[0])
        premesh_index_tri = (msdf_positive_fx3 * v_id_msdf_tri.unsqueeze(0)).sum(-1)
        premesh_index_quad = (msdf_positive_fx4 * v_id_msdf_quad.unsqueeze(0)).sum(-1)

        idx_map_tri = torch.cat([pre_mesh_edge_tri[:, :, 0], verts.size(0) + torch.arange(pre_mesh_edge_tri.size(0) * 3, device='cuda').view(-1, 3)], dim=-1)
        idx_map_quad = torch.cat([pre_mesh_edge_quad[:, :, 0], verts.size(0) + pre_mesh_edge_tri.size(0) * 3 + torch.arange(pre_mesh_edge_quad.size(0) * 4, device='cuda').view(-1, 4)], dim=-1)

        num_triangles_tri = self.num_triangles_tri_table[premesh_index_tri]
        num_triangles_quad = self.num_triangles_quad_table[premesh_index_quad]

        faces_aug = torch.cat((
            torch.gather(input=idx_map_tri[num_triangles_tri == 1], dim=1, index=self.triangle_table_tri[premesh_index_tri[num_triangles_tri == 1]][:, :3]).view(-1, 3),
            torch.gather(input=idx_map_tri[num_triangles_tri == 2], dim=1, index=self.triangle_table_tri[premesh_index_tri[num_triangles_tri == 2]][:, :6]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 1], dim=1, index=self.triangle_table_quad[premesh_index_quad[num_triangles_quad == 1]][:, :3]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 2], dim=1, index=self.triangle_table_quad[premesh_index_quad[num_triangles_quad == 2]][:, :6]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 3], dim=1, index=self.triangle_table_quad[premesh_index_quad[num_triangles_quad == 3]][:, :9]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 4], dim=1, index=self.triangle_table_quad[premesh_index_quad[num_triangles_quad == 4]][:, :12]).view(-1, 3),
        ), dim=0)

        return verts_aug, faces_aug, None, None, v_tng_aug, verts, valid_tet_gidx, msdf_vert_aug, msdf_vert
