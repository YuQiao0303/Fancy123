# Qiao modified from 
# https://github.com/TencentARC/InstantMesh/blob/main/src/models/geometry/camera/perspective_camera.py
# https://github.com/TencentARC/InstantMesh/blob/main/src/models/geometry/render/neural_render.py

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import sys
sys.path.append('...')
sys.path.append('..')
sys.path.append('.')
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from src.utils.camera_util import center_looking_at_camera_pose, elevation_azimuth_radius_to_xyz

_FG_LUT = None
glctx = dr.RasterizeCudaContext( device="cuda")#glctx = dr.RasterizeGLContext(output_db=False, device="cuda")


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_normals(normals, matrix, use_python=True):
    '''
    Transform normals.
    Args:
    normals: Tensor containing 3D normals with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
    matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
    use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
    Transformed normals with shape [minibatch_size, num_vertices, 3].
    '''
    # matrix[:,:,-1]=0 # ignore translate, only rotate
    # # matrix = torch.inverse(matrix)
    # out = torch.matmul(torch.nn.functional.pad(normals, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    # if torch.is_anomaly_enabled():
    #     assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    # return out
    
    #####################################################
    # Extract the upper-left 3x3 part of the  matrix
    matrix = matrix[:, :3, :3] # rotateion matrix
    
    # matrix = torch.inverse(matrix)
    matrix = torch.transpose(matrix, 1, 2)
    # Transform the normals
    transformed_normals = torch.matmul(normals, matrix)

    # Normalize the transformed normals
    transformed_normals = torch.nn.functional.normalize(transformed_normals, p=2, dim=-1)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(transformed_normals)), "Output of xfm_normals contains inf or NaN"
    return transformed_normals


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def compute_vertex_normal(v_pos, t_pos_idx):
    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)
    assert torch.all(torch.isfinite(v_nrm))

    return v_nrm

def compute_face_normal(v_pos, t_pos_idx):
    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)
    
    # Normalize, replace zero (degenerated) normals with some default value
    face_normals = torch.where(
        dot(face_normals, face_normals) > 1e-20, face_normals, torch.as_tensor([0.0, 0.0, 1.0]).to(face_normals)
    )
    face_normals = F.normalize(face_normals, dim=1)

    return face_normals

# perspective projection
def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array(
        [[n / x, 0, 0, 0],
         [0, n / -x, 0, 0],
         [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
         [0, 0, -1, 0]]).astype(np.float32)


def project(points_bxnx4, fovy=49.0 ,ortho=False):
    if ortho:
        return project_ortho(points_bxnx4)
    else:
        device = points_bxnx4.device
        focal = np.tan(fovy / 180.0 * np.pi * 0.5)
        proj_mtx = torch.from_numpy(projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)).to(device).unsqueeze(dim=0)
        out = torch.matmul(
                points_bxnx4,
                torch.transpose(proj_mtx, 1, 2))
        out_NDC = out/out[:, :, -1].unsqueeze(dim=-1) # so that returned coordinates are in [-1,1]
        return out_NDC


# orthographic projection
def _orthographic(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    # Unique3D/mesh_reconstruction/func.py
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    o = torch.zeros([4,4],device=device)
    o[0,0] = 2/(r-l)
    o[0,3] = -(r+l)/(r-l)
    o[1,1] = 2/(t-b) * (-1 if flip_y else 1)
    o[1,3] = -(t+b)/(t-b)
    o[2,2] = -2/(f-n)
    o[2,3] = -(f+n)/(f-n)
    o[3,3] = 1
    return o #4,4


def project_ortho(points_bxnx4,  right=1):
    
    device = points_bxnx4.device
    proj_mtx = _orthographic(r=right,device=device).unsqueeze(dim=0)
    out = torch.matmul(
        points_bxnx4,
        torch.transpose(proj_mtx, 1, 2))
    # 注意：正交投影后不需要除以 w 分量，因为正交投影没有透视除法
    out_NDC = out
    return out_NDC

class NeuralRender():
    def __init__(self, device='cuda'):
        super(NeuralRender, self).__init__()
        self.ctx = dr.RasterizeCudaContext(device=device)
   
    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            fov_in_degrees=30,
            device='cuda'
    ):
        mesh_t_pos_idx_fx3 = mesh_t_pos_idx_fx3.int()
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = project(v_pos, fovy=fov_in_degrees)  # Projection in the camera # [B,V,4]

        # v_nrm = compute_vertex_normal(mesh_v_pos_bxnx3[0], mesh_t_pos_idx_fx3.long())  # vertex normals in world coordinates

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd.repeat(v_pos.shape[0], 1, 1), v_pos], dim=-1)  # Concatenate the pos

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer() # [B,res,res,4],_
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)  # [B,res,res,1]
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3) # [B,res,res,1]

        depth = gb_feat[..., -2:-1] # [B,res,res,1]
        ori_mesh_feature = gb_feat[..., :-4] # [B,res,res,d]
        # ori_mesh_feature = dr.antialias(ori_mesh_feature.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3) # added by Qiao

        # normal, _ = interpolate(v_nrm[None, ...], rast, mesh_t_pos_idx_fx3)
        # normal = dr.antialias(normal.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        # normal = F.normalize(normal, dim=-1)
        # normal = torch.lerp(torch.zeros_like(normal), (normal + 1.0) / 2.0, hard_mask.float())      # black background # [B,res,res,3]
        normal = None
        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal

    # def get_rendered_rgba_by_vert_col(self, vert_col, rast_out, vertices_clip, faces):
    #     rgb,_ = dr.interpolate(vert_col, rast_out, faces) #B,H,W,3
    #     alpha = torch.clamp(rast_out[..., -1:], max=1) #B,H,W,1
    #     rgba = torch.concat((rgb,alpha),dim=-1) #B,H,W,4
    #     rgba = dr.antialias(rgba, rast_out, vertices_clip, faces) #B,H,W,4
    #     return rgba
    
    def get_rendered_rgba_by_vert_col(
        self,
        mesh_v_pos_bxnx3,
        mesh_t_pos_idx_fx3,
        camera_mv_bx4x4,
        mesh_v_feat_bxnxd, # colors
        resolution=256,
        spp=1,
        fov_in_degrees=30,
        device='cuda',return_rgba_only=False):
        mesh_t_pos_idx_fx3 = mesh_t_pos_idx_fx3.int()
        ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal = self.render_mesh(
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=resolution,
            spp=spp,
            fov_in_degrees=fov_in_degrees,
            device=device
        )
        
        rgb = ori_mesh_feature
        rgba = torch.concat((rgb,hard_mask),dim=-1)
        rgba = dr.antialias(rgba.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        if return_rgba_only:
            return rgba
        return rgba,ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal

########################################################
# The following are modified from Unique3D/mesh_reconstruction/render.py
def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

def crop_vertices0(vertices_clip,crop_views):
    '''
    vertices_clip: [B,V,4]
    crop_views: list of views to be croped
    '''
    if crop_views is not None:
        device = vertices_clip.device
        crop_views = torch.tensor(crop_views).int().to(device)
        vertice_uvs = vertices_clip[crop_views, :, :2]
        
        ori_vertice_uvs_min = vertice_uvs.min(1)[0] # cam_num,2
        ori_vertice_uvs_max = vertice_uvs.max(1)[0] # cam_num,2
        ori_vertice_uvs_min = ori_vertice_uvs_min.unsqueeze(1) # cam_num,1,2
        ori_vertice_uvs_max = ori_vertice_uvs_max.unsqueeze(1) # cam_num,1,2
        uv_centers = (ori_vertice_uvs_min + ori_vertice_uvs_max) / 2  # cam_num,1,2
        uv_scales = (ori_vertice_uvs_max-ori_vertice_uvs_min).max(2)[0].unsqueeze(2)  # cam_num,1,2
        vertice_uvs = (vertice_uvs - uv_centers) / uv_scales # now all between -0.5, 0.5
        vertice_uvs = vertice_uvs*2 # now all between -1 and 1

        vertices_clip[crop_views, :, :2] = vertice_uvs 
    return vertices_clip
  
  
def crop_vertices_by_scale(vertices_clip,crop_views,scale=None):
    '''
    vertices_clip: [B,V,4]
    crop_views: list of views to be croped
    
    '''
    if crop_views is not None:
        device = vertices_clip.device
        crop_views = torch.tensor(crop_views).int().to(device)
        vertice_uvs = vertices_clip[crop_views, :, :2]
        
        if scale is None:
            scale = vertice_uvs[0].abs().max() # by default, use the first view's max value as scale
        
        # ori_vertice_uvs_min = vertice_uvs.min(1)[0] # cam_num,2
        # ori_vertice_uvs_max = vertice_uvs.max(1)[0] # cam_num,2
        # ori_vertice_uvs_min = ori_vertice_uvs_min.unsqueeze(1) # cam_num,1,2
        # ori_vertice_uvs_max = ori_vertice_uvs_max.unsqueeze(1) # cam_num,1,2
        # uv_centers = (ori_vertice_uvs_min + ori_vertice_uvs_max) / 2  # cam_num,1,2
        # uv_scales = (ori_vertice_uvs_max-ori_vertice_uvs_min).max(2)[0].unsqueeze(2)  # cam_num,1,2
        # vertice_uvs = (vertice_uvs - uv_centers) / uv_scales # now all between -0.5, 0.5
        vertice_uvs = vertice_uvs / scale #  # now all between -1 and 1

        vertices_clip[crop_views, :, :2] = vertice_uvs 
    return vertices_clip     

def crop_vertices(vertices_clip,crop_views,by_scale=False):
    if by_scale:
        return crop_vertices_by_scale(vertices_clip,crop_views) # top or bottom or left or right at the border of the image
    else:
        return crop_vertices0(vertices_clip,crop_views) # horizontally or vertically occupy the whole image

class GeneralRenderer:

    _glctx:dr.RasterizeCudaContext = None #_glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            device=None,
            fov_in_degrees = 30, # only used when ortho is False
            default_res = 320,
            ortho=False
            ):
        self.device=device
        self._glctx = glctx
        self.fov_in_degrees = fov_in_degrees
        self.default_res = default_res
        self.ortho = ortho
        _warmup(self._glctx, device)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float   in [-1, 1]
            faces: torch.Tensor, #F,3 int32
            fov_in_degrees = None, camera_positions_np =None,res=None,
            rotate_normal=True,
            camera_positions=None,
            background_color = None,
            crop_views=None,
            return_rast_out=False
            ) ->torch.Tensor: #C,H,W,4
        if fov_in_degrees is None:
            fov_in_degrees = self.fov_in_degrees
        if res is None:
            res = self.default_res
        # print('fov_in_degrees',fov_in_degrees)
        # print('orth',self.ortho)
        # print('camera_positions_np',camera_positions_np)
        # print('camera_positions',camera_positions)
        # print('------------------------------')
     
        
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        
        # init cameras
        if camera_positions is None:
            if camera_positions_np is None:
                camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=4) 
            camera_positions = torch.tensor(camera_positions_np).float().to(device).requires_grad_()
  
        # if torch.all(camera_positions[:, [0, 2]] == 0, dim=1): # 
        #     up_world = torch.tensor([0,0,1]).float().to(device)
        # else:
        #     up_world = torch.tensor([0,1,0]).float().to(device)
        up_world = torch.tensor([0,1,0]).float().to(device)
        c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
        camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
        
        # # trans points
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        vertices_clip = project(v_pos, fovy=fov_in_degrees,ortho=self.ortho)  # Projection in the camera # [B,V,4]
        # vertices_clip = camera_trans_points(vertices,camera_positions,fov_in_degrees=fov_in_degrees,orthogonal=self.ortho) # [C,V,3] # we don't use this here since mtx_in is used in rotate_normal
        vertices_clip = crop_vertices(vertices_clip,crop_views)
        
        
        
        
        # rasterize
        
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4
        if rotate_normal:
            used_normals = xfm_normals(normals.unsqueeze(0), mtx_in)[:,:,:3]  # Rotate it to camera coordinates
        else:
            used_normals = normals
        vert_col = (used_normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        if background_color is not None:
            col = col * alpha + (1-alpha) * torch.tensor(background_color).float().to(device)
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        if return_rast_out:
            return col, rast_out
        return col #C,H,W,4

    def get_per_pixel_face_ids(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 int32
            fov_in_degrees = None, camera_positions_np =None,res=None,crop_views=None):
        if fov_in_degrees is None:
            fov_in_degrees = self.fov_in_degrees
        if res is None:
            res = self.default_res
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        
        # init cameras
        camera_positions = camera_positions_np
        if camera_positions is None:
            camera_positions = elevation_azimuth_radius_to_xyz(elevation=[0],azimuth=[0],radius=4) 
        camera_positions = torch.tensor(camera_positions).float().to(device).requires_grad_()
        
        vertices_clip = camera_trans_points(vertices,camera_positions,fov_in_degrees=fov_in_degrees,orthogonal=self.ortho) # [C,V,3]
        vertices_clip = crop_vertices(vertices_clip,crop_views)
        # up_world = torch.tensor([0,1,0]).float().to(device)
        # c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        # w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
        # camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
        
        # # trans points
        # mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        # v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        # vertices_clip = project(v_pos, fovy=fov_in_degrees,ortho=self.ortho)  # Projection in the camera # [B,V,4]
        
        # rasterize
        
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4
        per_pixel_face_ids = rast_out[...,-1] -1 # face id starts from 0; -1 means background
        # per_pixel_face_ids = per_pixel_face_ids.int()
        # visible_face_ids = torch.unique(per_pixel_face_ids)
        return per_pixel_face_ids #C,H,W,4
    
    
    def get_per_pixel_face_ids_and_rgba(self,
            vertices: torch.Tensor, #V,3 float
            vertex_colors:torch.Tensor,
            faces: torch.Tensor, #F,3 int32
            fov_in_degrees = None, camera_positions_np =None,res=None,crop_views=None,background_color = None):
        
        if fov_in_degrees is None:
            fov_in_degrees = self.fov_in_degrees
        if res is None:
            res = self.default_res
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        
        # init cameras
        camera_positions = camera_positions_np
        if camera_positions is None:
            camera_positions = elevation_azimuth_radius_to_xyz(elevation=[0],azimuth=[0],radius=4) 
        camera_positions = torch.tensor(camera_positions).float().to(device).requires_grad_()
        
        vertices_clip = camera_trans_points(vertices,camera_positions,fov_in_degrees=fov_in_degrees,orthogonal=self.ortho) # [C,V,3]
        vertices_clip = crop_vertices(vertices_clip,crop_views)

        
        # rasterize
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4
        per_pixel_face_ids = rast_out[...,-1] -1 # face id starts from 0; -1 means background
        # per_pixel_face_ids = per_pixel_face_ids.int()
        # visible_face_ids = torch.unique(per_pixel_face_ids)
        
        vert_col = vertex_colors #V,3
        
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
       
        if background_color is not None:
            col = col * alpha + (1-alpha) * torch.tensor(background_color).float().to(device)
       
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return per_pixel_face_ids,col #C,H,W,4
    def get_per_pixel_face_ids_and_vertex_NDC(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 int32
            fov_in_degrees = None, camera_positions_np =None,res=None,crop_views=None):
        if fov_in_degrees is None:
            fov_in_degrees = self.fov_in_degrees
        if res is None:
            res = self.default_res
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        
        # init cameras
        camera_positions = camera_positions_np
        if camera_positions is None:
            camera_positions = elevation_azimuth_radius_to_xyz(elevation=[0],azimuth=[0],radius=4) 
        camera_positions = torch.tensor(camera_positions).float().to(device).requires_grad_()
        
        vertices_clip = camera_trans_points(vertices,camera_positions,fov_in_degrees=fov_in_degrees,orthogonal=self.ortho) # [C,V,3]
        vertices_clip = crop_vertices(vertices_clip,crop_views)

       
        # rasterize
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4
        per_pixel_face_ids = rast_out[...,-1] -1 # face id starts from 0; -1 means background
        # per_pixel_face_ids = per_pixel_face_ids.int()
        # visible_face_ids = torch.unique(per_pixel_face_ids)
        return per_pixel_face_ids,vertices_clip #[C,H,W,4], [C,V,4]

    def render_depth(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 int32
            fov_in_degrees = None, camera_positions_np =None,res=None
            ) ->torch.Tensor: #C,H,W,4
        if fov_in_degrees is None:
            fov_in_degrees = self.fov_in_degrees
        if res is None:
            res = self.default_res
        # V = vertices.shape[0]
        # faces = faces.type(torch.int32)
        # vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        # vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        
        # init cameras
        camera_positions = camera_positions_np
        if camera_positions is None:
            camera_positions = elevation_azimuth_radius_to_xyz(elevation=[0],azimuth=[0],radius=4) 
        camera_positions = torch.tensor(camera_positions).float().to(device).requires_grad_()
        # up_world = torch.tensor([0,1,0]).float().to(device)
        # c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        # w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
        # camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
        
        # # # trans points
        # mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        # v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        # vertices_clip = project(v_pos, fovy=fov_in_degrees,ortho=self.ortho)  # Projection in the camera # [B,V,4]
        
        vertices_clip = camera_trans_points(vertices,camera_positions,fov_in_degrees=fov_in_degrees,orthogonal=self.ortho) # [C,V,4]
        import kiui

        # rasterize
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4
        vertice_depths = vertices_clip[...,-2] # [C,V]
        vert_col = vertice_depths.unsqueeze(-1).repeat(1,1,3) # [C,V,3]
        # vertice_depths_min = vertice_depths.min(1)[0].unsqueeze(1).repeat(1,vertice_depths.shape[1]) # [C,V]
        # vertice_depths_max = vertice_depths.max(1)[0].unsqueeze(1).repeat(1,vertice_depths.shape[1]) # [C,V]
        # kiui.lo(vertice_depths)
        # kiui.lo(vertice_depths_min)
        # kiui.lo(vertice_depths_max)
        # vert_col = (vertice_depths - vertice_depths_min) / (vertice_depths_max - vertice_depths_min) # [C,V]
        # vert_col = vert_col.unsqueeze(-1).repeat(1,1,3) # [C,V,3]
        
        # kiui.lo(vert_col)
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        
        
        depth_CHW = col[:,:,:,0] #C,H,W
        alpha_CHW = alpha.squeeze(-1)
        # normalize
        for i in range(col.shape[0]):
            C, H, W = depth_CHW.shape
            mask = (alpha_CHW[i] == 1)

            max_val = torch.max(depth_CHW[i][mask])
            min_val = torch.min(depth_CHW[i][mask])

            if max_val == min_val:
                continue

            depth_CHW[i] -= min_val
            depth_CHW[i] /= (max_val - min_val)
            depth_CHW[i][~mask] *= 0
        
        normalized_depth_CHW3 = depth_CHW.unsqueeze(-1).repeat(1,1,1,3)
        # depth_CHW = col[:,:,:,0] #C,H,W
        # depth_C_HxW = depth_CHW.reshape(depth_CHW.shape[0],-1) #C,HxW
        # normalized_depth_C_HxW = (depth_C_HxW - depth_C_HxW[alpha].min()[0]) / 
        # (depth_C_HxW.max(1,keepdim=True)[0]- depth_C_HxW.min(1,keepdim=True)[0])
        # normalized_depth_CHW = normalized_depth_C_HxW.reshape(depth_CHW.shape)#C,H,W
        # normalized_depth_CHW3 = normalized_depth_CHW.unsqueeze(-1).repeat(1,1,1,3)
        # import kiui
        # kiui.lo(depth_CHW)
        # kiui.lo(depth_C_HxW)
        # kiui.lo(normalized_depth_C_HxW)
        # kiui.lo(normalized_depth_CHW)
        # kiui.lo(normalized_depth_CHW3)
        
        col = torch.concat((normalized_depth_CHW3,alpha),dim=-1) #C,H,W,4
        
        # col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

    
    def render_with_texture(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 int32
            uvs, # [uv_num, 2] float
            face_uvs_idx, # [F,3] int
            material_list_of_13HW, # [HW3] float, atlas_img_HW3,
            face_material_idx=None, # [F], int
            face_uvs_F32 = None, #[F,3,2]
            fov_in_degrees = None, camera_positions_np =None,res=None,
            camera_positions=None,
            background_color = None,
            crop_views=None,
            return_rast_out=False
            ) ->torch.Tensor: #C,H,W,4

        if fov_in_degrees is None:
            fov_in_degrees = self.fov_in_degrees
        if res is None:
            res = self.default_res
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        if face_material_idx is None:
            face_material_idx = torch.zeros(faces.shape[0],device=faces.device).long()

        # init cameras
        if camera_positions is None:
            if camera_positions_np is None:
                camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=4) 
            camera_positions = torch.tensor(camera_positions_np).float().to(device).requires_grad_()

        
        vertices_clip = camera_trans_points(vertices,camera_positions,fov_in_degrees=fov_in_degrees,orthogonal=self.ortho) # [C,V,3]
        vertices_clip = crop_vertices(vertices_clip, crop_views)
        
        # rasterize
        
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4


        # deal with texture
        if uvs is None:
            flat_face_uvs = face_uvs_F32.reshape(-1, 2)
            unique_uvs, inverse_indices = torch.unique(flat_face_uvs, dim=0, return_inverse=True)
           
            num_faces = face_uvs_F32.size(0)
            face_uvs_idx = inverse_indices.view(num_faces, -1)
            uvs = unique_uvs

        uvs[...,1] =  - uvs[...,1]
        uvs = torch.nn.functional.pad(uvs.unsqueeze(0).to(device), (0, 0, 0, 1)) #% 1. # don't %1 here. %1 after interpolate
        face_uvs_idx[face_uvs_idx == -1] = 0 # face_uvs_idx[face_uvs_idx == -1] = uvs.shape[1] - 1
        # materials = [atlas_img_HW3.permute(2,0,1).unsqueeze(0)]


        uv_map = dr.interpolate(uvs, rast_out, face_uvs_idx.int())[0] # cam_num,res,res,2, right here

        imgs = torch.zeros((len(camera_positions), res, res, 3), dtype=torch.float, device=device) # this is our final rendered img
    
        # Obj meshes can be composed of multiple materials (though in this function we assume it's only one)
        # so at rendering we need to interpolate from corresponding materials
        hard_mask = rast_out[:, :, :, -1:] != 0 # # cam_num,res,res,1
        face_idx = (rast_out[..., -1].long() - 1).contiguous() # cam_num,res,res
        
        im_material_idx = face_material_idx[face_idx]
        im_material_idx[face_idx == -1] = -1 # cam_num, res,res
        
       
        # the following is relatively slow

        # each camera has an image
        # im_material_idx # [cam_num, res,res], material_idx of each pixel of each image
        # imgs # [cam_num, res, res, 3],
        # materials: list of material images, each element is of size [1,3,H,W]. Materials is with different H and W.
        # uv_map # [cam_num,res,res,2], uv coordiantes of each pixel of each image


        for cam_i in range(len(camera_positions)):
            for i, material in enumerate(material_list_of_13HW):
                mask = im_material_idx[cam_i] == i # res,res # only foreground pixels
                # mask = torch.ones_like(im_material_idx[cam_i]).bool()  # res,res # all pixels
                _texcoords = (uv_map[cam_i].unsqueeze(0) %1)* 2. - 1.  # cam_num,res,res,2 # %1->(0,1); *2-1 -> (-1,1)
                _texcoords[..., 1] = -_texcoords[..., 1]  # cam_num,res,res,2 # necessary for torch.nn.functional.grid_sample
                pixel_val = torch.nn.functional.grid_sample(
                    material_list_of_13HW[i], _texcoords.reshape(1, 1, -1, 2),
                    mode='bilinear', align_corners=False,
                    padding_mode='border')

                imgs[cam_i][mask] = pixel_val[0, :, 0].permute(1, 0).reshape(res,res,3)[mask]#[0]

        # now deal with background, and antialias
        col = imgs
        alpha = hard_mask
        
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        if background_color is not None:
            col = col * alpha + (1-alpha) * torch.tensor(background_color).float().to(device)
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        if return_rast_out:
            return col, rast_out
        return col #C,H,W,4

    def render_orthographic(self,vertices,faces,vertex_colors,camera_positions,background_color=None,crop_views=None,res=256,return_rast_out=False):
        device = vertices.device
        mesh_v_pos_bxnx3 = vertices.unsqueeze(0)
        faces = faces.int()
        
        # init cameras
        if camera_positions is None:
            camera_positions = elevation_azimuth_radius_to_xyz(elevation=[0],azimuth=[0],radius=4) 
            camera_positions = torch.tensor(camera_positions).float().to(device).requires_grad_()
        # up_world = torch.tensor([0,1,0]).float().to(device)
        # c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        # w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
        # camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
        
        # # # trans points
        # mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        # v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        # vertices_clip = project(v_pos, fovy=fov_in_degrees,ortho=self.ortho)  # Projection in the camera # [B,V,4]
        
        vertices_clip = camera_trans_points(vertices,camera_positions,orthogonal=True) # [C,V,4]
        vertices_clip = crop_vertices(vertices_clip,crop_views)
        
        
        
        
        # rasterize
        
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=(res,res), grad_db=False) #C,H,W,4
      
        
        vert_col = vertex_colors #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        if background_color is not None:
            col = col * alpha + (1-alpha) * torch.tensor(background_color).float().to(device)
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        if return_rast_out:
            return col, rast_out
        return col
        
    def add_light_to_rendered_image(self,albedos_NHW4,face_idx_NHW,face_normals_F3,camera_positions_N3, 
                                    light_dirs=None,gamma=1.5,double_side=False,rotate_normals = True ):
        # face_idx_NHW = rast_out_NHW4[...,-1] -1
        hard_mask = face_idx_NHW > 0
        device = albedos_NHW4.device
        imgs = torch.zeros_like(albedos_NHW4).to(device)
        res = albedos_NHW4.shape[1]
        
        if light_dirs is None:
            light_dirs = torch.tensor(np.array([
                # [0.5, 0, 0],  # left
                # [0.0, 0.5, 0.0],  # top
                [0.0, 0.0, 1.2],  # front
            ])).float().to(device)
            
        
        up_world = torch.tensor([0,1,0]).float().to(device)
        # if torch.all(camera_positions[:, [0, 2]] == 0, dim=1): # 
        #     up_world = torch.tensor([0,0,1]).float().to(device)
        # else:
        #     up_world = torch.tensor([0,1,0]).float().to(device)
        c2ws = center_looking_at_camera_pose(camera_positions_N3,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
        camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4

        # convert light_dirs to camera coordinate system
        # matrix = mtx_in[:, :3, :3] # rotateion matrix: Extract the upper-left 3x3 part of the  matrix
        # matrix = torch.transpose(matrix, 1, 2)
        # used_light_dirs = torch.matmul(light_dirs.unsqueeze(0), matrix)
        # print('used_light_dirs.shape',used_light_dirs.shape)
        # print('used_light_dirs[0]',used_light_dirs[0])
        # print('used_light_dirs[2]',used_light_dirs[2]) # 90 degree
        
        used_light_dirs = light_dirs.unsqueeze(0).repeat(len(camera_positions_N3), 1, 1)

        
        
        # rotate normals
        # rotate_normals = True # so that the rendered img is always bright
        if rotate_normals:
            face_normals_BF3 = xfm_normals(face_normals_F3.unsqueeze(0), mtx_in)[:,:,:3]
        
        
        for cam_i in range(len(camera_positions_N3)):
            face_idx_valid = face_idx_NHW[cam_i].clone().long()
            face_idx_valid[face_idx_valid < 0] = 0

            
            if rotate_normals:
                view_pixel_normals = face_normals_BF3[cam_i][face_idx_valid]  # res,res,3
                # fix normal to point outwards
                camera_forward = torch.tensor([0,0,1]).float().to(device) # camera_positions_N3[cam_i]
                temp = torch.matmul(face_normals_BF3[cam_i], camera_forward.unsqueeze(1))  # F
                temp_face_normals_F3 = face_normals_BF3[cam_i].clone()
                temp_face_normals_F3[temp.squeeze(-1)<0] *=-1
                view_pixel_normals = temp_face_normals_F3[face_idx_valid]  # res,res,3
            else:
                # fix normal to point outwards
                camera_forward = camera_positions_N3[cam_i]
                temp = torch.matmul(face_normals_F3, camera_forward.unsqueeze(1))  # F
                temp_face_normals_F3 = face_normals_F3.clone()
                temp_face_normals_F3[temp.squeeze(-1)<0] *=-1
                view_pixel_normals = temp_face_normals_F3[face_idx_valid]  # res,res,3

   

            for light_dir in used_light_dirs[cam_i]:
                
                ##
                VN_dot_Light_dir = torch.matmul(view_pixel_normals.reshape(-1, 3), light_dir.unsqueeze(1))
                VN_dot_Light_dir = VN_dot_Light_dir.reshape(res, res, 1)  # .squeeze(-1)

                if double_side:
                    VN_dot_Light_dir = torch.abs(VN_dot_Light_dir) # double sided
                imgs[cam_i] += albedos_NHW4[cam_i] * VN_dot_Light_dir.clip(0, 1)

            # imgs[cam_i][~hard_mask[cam_i].squeeze(-1)] = 0

            imgs[cam_i] = imgs[cam_i].clip(0, 1)
            if gamma is not None:
                imgs[cam_i] = imgs[cam_i] ** (1.0 / gamma)  # gamma
            imgs[cam_i][...,3] = albedos_NHW4[cam_i][...,3]
        return imgs
        
        
def camera_trans_points(points,camera_positions,fov_in_degrees=30,orthogonal=False):
    '''
    camera_positions: torch.tensor, [cam_num,3], float
    '''
    device = points.device
    up_world = torch.tensor([0,1,0]).float().to(device)
    # if torch.all(camera_positions[:, [0, 2]] == 0, dim=1): # 
    #         up_world = torch.tensor([0,0,1]).float().to(device)
    # else:
    #     up_world = torch.tensor([0,1,0]).float().to(device)
    try:
        c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
    except:
        up_world = torch.tensor([0,0,1]).float().to(device)
        c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
    camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
    
    # trans points
    mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
    v_pos = xfm_points(points.unsqueeze(0), mtx_in)  # Rotate it to camera coordinates
  
    vertices_clip = project(v_pos, fovy=fov_in_degrees,ortho=orthogonal)  # Projection in the camera # [B,V,4]
    
    transformed_points = vertices_clip#[0]
    return transformed_points



if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    from src.utils.mesh_util import loadobj_color
    import torch
    import kiui
    from PIL import Image

    device = 'cuda'
    fov_in_degrees = 30
    resolution=256

    # # Load mesh
    name = 'cute_horse'
    obj_path = os.path.join('outputs/instant-mesh-large/meshes', f'{name}.obj')
    vertices, faces, vertex_colors  = loadobj_color(obj_path,device=device)

    # # Load renderer and camera positions
    use_my_renderer = False
    # camera_positions = elevation_azimuth_radius_to_xyz(elevation=[0,0],azimuth=[0,20],radius=4) # camera_positions = np.array([[0,0,5],[3,0,4]]) # cam_num,3
    
    
    azim_list=[30.0, 90.0, 150.0, 210.0, 270.0, 330.0] 
    elevations_v12 = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]
    elevations_v11 = [30.0, -20.0, 30.0, -20.0, 30.0, -20.0]
    # camera_positions_np_v11 = elevation_azimuth_radius_to_xyz(elevation_in_degrees=elevations_v11,azimuth_in_degrees=azim_list,radius=4) 
    camera_positions_np_v12 = elevation_azimuth_radius_to_xyz(elevation_in_degrees=elevations_v12,azimuth_in_degrees=azim_list,radius=4) 
    
    camera_positions_np = camera_positions_np_v12
    camera_positions = torch.tensor(camera_positions_np).float()
    if use_my_renderer:
        # if camera_positions[:, [0, 2]] == 0, dim=1): # 
        #     up_world = torch.tensor([0,0,1]).float().to(device)
        # else:
        #     up_world = torch.tensor([0,1,0]).float().to(device)
        up_world = torch.tensor([0,1,0]).float().to(device)    
        c2ws = center_looking_at_camera_pose(camera_positions,up_world=up_world) # c2ws, extrincs # cam_num,4,4
        w2cs = torch.linalg.inv(c2ws) # w2cs, mv # cam_num,4,4
        camera_mv_bx4x4 = w2cs.to(device) # cam_num,4,4
        renderer = NeuralRender(device=device)
        rendered_rgba = renderer.get_rendered_rgba_by_vert_col(vertices.unsqueeze(0),faces.int(),camera_mv_bx4x4,vertex_colors.unsqueeze(0), 
                                  resolution=resolution,fov_in_degrees=fov_in_degrees,device=device)  # [B,res,res, 4]
        for i in range(len(camera_positions)):
            kiui.vis.plot_image(rendered_rgba[i])
    else:
        renderer = GeneralRenderer(device=device)
        rgba_NHW4 = renderer.render(vertices,
            vertex_colors*2-1,
            faces,
            fov_in_degrees = 30, camera_positions_np =camera_positions_np,
            res=320,
            rotate_normal=False)
        for i in range(len(camera_positions)):
            pil = Image.fromarray((rgba_NHW4[i]*255).detach().cpu().numpy().astype(np.uint8))
            pil.save(f'{name}_{i}.png')

    
    
    # result = renderer.render_mesh(vertices.unsqueeze(0),faces.int(),camera_mv_bx4x4,vertex_colors.unsqueeze(0), 
    #                               resolution=resolution,fov_in_degrees=fov_in_degrees,device=device)
    
    # ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal = result # most ly [B,res,res, X]
    # rendered_rgb = ori_mesh_feature
    # rendered_rgba = renderer.get_rendered_rgba_by_vert_col(vertex_colors.unsqueeze(0), rast, v_pos_clip, faces.int())  # [B,res,res, 4]


