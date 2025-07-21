# NBF: Non-Border-First Unprojection
# This script implements another variants of NBF: Non-Border-First Unprojection.
# For more introduction of NBF, please refer to: 
# PointDreamer: Zero-shot 3D Textured Mesh Reconstruction from Colored Point Cloud by 2D Inpainting
# https://github.com/YuQiao0303/PointDreamer
import torch
import numpy as np
import pymeshlab
import trimesh

import torch.nn.functional as F

import os

import torch
import numpy as np
import cv2
import math
from fancy123.render.general_renderer import GeneralRenderer, camera_trans_points, crop_vertices
from fancy123.utils.temp_utils import get_view_visible_V_F
from fancy123.utils.utils2d import display_CHW_RGB_img_np_matplotlib,cat_images,save_CHW_RGB_img,detect_edges_in_gray_by_scharr,detect_edges_in_gray_by_scharr_torch_batch,dilate_torch_batch


import kaolin as kal
import time
from fancy123.utils.mesh_utils import simplify_mesh, subdivide_with_uv, xatlas_uvmap_w_face_id

import nvdiffrast.torch as dr

from src.utils.mesh_util import save_obj
from PIL import Image




def get_point_validation_by_depth(cam_res,point_uvs,point_depths,mesh_depths,offset = 0,vis=False):
    cam_num, point_num,_ = point_uvs.shape
    device = point_uvs.device
    point_visibility = torch.zeros((cam_num, point_num), device=point_uvs.device).bool()
    point_pixels = point_uvs * cam_res
    point_pixels = point_pixels.clip(0, cam_res - 1)
    point_pixels = point_pixels.long()
    point_pixels = torch.cat((point_pixels[:, :, 1].unsqueeze(-1), point_pixels[:, :, 0].unsqueeze(-1)),
                             dim=-1)  # switch x and y if you ever need to query pixel coordiantes

    reference_depth = mesh_depths[
        torch.arange(cam_num).view(-1, 1),  # cam_num, 1
        point_pixels[:, :, 0].long(),  # cam_num, point_num
        point_pixels[:, :, 1].long()  # cam_num, point_num
    ]

    if vis:
        for i in range(cam_num):
            # reference depth img
            depth_img = mesh_depths[i].unsqueeze(0).repeat(3, 1, 1)
            depth_img = depth_img
            mask = depth_img != 0
            depth_min, depth_max = depth_img[mask].min(), depth_img[mask].max()
            # print('depth_min', depth_min)
            # print('depth_max', depth_max)
            depth_img = (depth_img - depth_min) / (depth_max - depth_min)
            depth_img[~mask] = 0
            img1 = depth_img.clone()[0]
            img2 = depth_img.clone()[0]
            depth_img = depth_img.detach().cpu().numpy()
  

            img1[point_pixels[i, :, 0], point_pixels[i, :, 1]] = (point_depths[i] - depth_min) / (depth_max - depth_min)
            img1 = img1.unsqueeze(0).repeat(3, 1, 1)
            # img1 = (img1 - depth_min) / (depth_max - depth_min)
            img1[~mask] = 0
            img1 = img1.detach().cpu().numpy()


            img2 = img2.unsqueeze(0).repeat(3, 1, 1)
            img2[:,point_pixels[i, :, 0], point_pixels[i, :, 1]] = torch.tensor([1.0,0,0]).float().to(device).unsqueeze(1)

            img2 = img2.detach().cpu().numpy()

            cat = cat_images(img1,img2)
            display_CHW_RGB_img_np_matplotlib(cat)

    point_visibility[point_depths - reference_depth <= offset] = True  # # [num_cameras,point_num]

    return point_visibility,point_pixels.long()


## Deal with invisible areas
def compute_vertex_only_uv_mask(face_vertex_idx, face_uvs_idx):
    '''Compute a mask that indicates whether each vertex has only one corresponding uv index.
    This function is used in 'paint_invisible_areas_by_neighbors'
    '''
    device = face_vertex_idx.device
    # Step 1: Flatten the face_vertex_idx and face_uvs_idx arrays
    face_vertex_idx_flat = face_vertex_idx.flatten()
    face_uvs_idx_flat = face_uvs_idx.flatten()

    # Step 2: Create counts for each vertex and uv index
    V = face_vertex_idx.max().item() + 1  # number of vertices
    vertex_uv_counts = torch.zeros(V, dtype=torch.int64).to(device)

    # Step 3: Count the occurrences of vertex-uv pairs
    vertex_uv_pairs = torch.stack((face_vertex_idx_flat, face_uvs_idx_flat), dim=1)
    unique_vertex_uv_pairs, counts = torch.unique(vertex_uv_pairs, dim=0, return_counts=True)

    # Step 4: Update the vertex_uv_counts based on unique vertex-uv pairs
    vertex_uv_counts.scatter_add_(0, unique_vertex_uv_pairs[:, 0], torch.ones_like(counts).to(device))

    # Step 5: Determine if each vertex has only one corresponding uv index
    vertex_only_uv_mask = vertex_uv_counts == 1

    return vertex_only_uv_mask,unique_vertex_uv_pairs



## Unproject and Non-Border-First

def unproject_uv(mv_imgs,vertices,f_normals,
              view_img_res,
              cam_positions,cam_res,mesh_normalized_depths,
              gb_pos,mask,per_atlas_pixel_face_id,
              edge_dilate_kernels,save_img_path,complete_unseen_by_projection=False):
    '''
    uvs:                per mesh vertex uv:                     [vert_num, 2]
    mesh_tex_idx:       per face uv coord id (to index uvs):    [face_num,3]
    gb_pos:             per pixel 3D coordinate:                [1,res,res,3] 
    mask:               per pixel validation:                   [1,res,res,1]
    per_pixel_face_id:  per pixel face id:                      [1,res,res]
    
    '''
    
    # view_img_res = res
    # res = xatlas_texture_res
    res = mask.shape[1]
    view_num = len(cam_positions)
    device = vertices.device
    renderer = GeneralRenderer(device=device)
    base_dirs = cam_positions

    per_pixel_mask = mask[0,:,:,0] # [res,res]
    per_pixel_point_coord = gb_pos[0] # [res,res,3]
    per_atlas_pixel_face_id = per_atlas_pixel_face_id[0] #[1,res,res]

    per_pixel_pixel_coord = torch.zeros((res,res,2),device=device).long()
    xx, yy = torch.meshgrid(torch.arange(res).to(device), torch.arange(res).to(device))
    per_pixel_pixel_coord[:, :, 0] = xx
    per_pixel_pixel_coord[:, :, 1] = yy

    points = per_pixel_point_coord[per_pixel_mask] # [?,3] ??
    points_atlas_pixel_coord = per_pixel_pixel_coord[per_pixel_mask].long() # [?,2] ??


    # get per-atlas-pixel's corresponding depth and uv in multiview images 
    # (depth used for calculating visibility, uv used for query correspondign color)
    transformed_points = camera_trans_points(points,cam_positions)
    per_view_per_point_depths = transformed_points[ ..., 2]
    per_view_per_point_uvs = transformed_points[..., :2]
    


    # Get per-atls-pixel  visibility by depth (so that we have per-view visible atlas)
    per_view_per_point_visibility,_ = get_point_validation_by_depth(cam_res,per_view_per_point_uvs,
                                    per_view_per_point_depths,mesh_normalized_depths,offset = 0.0001,
                                                                    vis=False)# [cam_num, point_num]

    start_shrink_visibility = time.time()



    per_atlas_pixel_per_view_visibility = torch.zeros((res,res,view_num),device=device).bool()

    per_atlas_pixel_per_view_visibility[per_pixel_mask] = per_view_per_point_visibility.permute(1,0)#.clone() # (res,res,view_num)

    # shrink per-view visible atlas (remove border areas, only keep non-border areas for later use)
    per_kernel_per_view_shrinked_per_pixel_visibility = get_shrinked_per_view_per_pixel_visibility_torch(
        per_pixel_mask,per_atlas_pixel_per_view_visibility,
        kernel_sizes= edge_dilate_kernels*(res//256),
        save_path = os.path.join(save_img_path,'shrink_per_view_edge')) # [kernel_num,view_num,res,res]

    # try:
    #     logger.info(f'shrink visibility: {time.time() - start_shrink_visibility} s')
    # except:
    #     pass

    # Get direction priority (similarity between point normal and view_dir)
    per_atlas_pixel_face_normal = f_normals[per_atlas_pixel_face_id] #res,res,3
    # print('f_normals.shape',f_normals.shape) #[face_num,3]
    # print('per_atlas_pixel_face_id.shape',per_atlas_pixel_face_id.shape) #[res,res]
    # print('per_atlas_pixel_face_normal.shape', per_atlas_pixel_face_normal.shape)
    per_point_face_normal = per_atlas_pixel_face_normal[per_pixel_mask] # [?,3]


    similarity_between_point_normal_and_view_dir = per_point_face_normal @ base_dirs.t()  # [ point_num,view_num]

    # Get per view per point pixel (for each point, its corresponding pixel coordnate in each view image)
    per_view_per_point_pixel = per_view_per_point_uvs * view_img_res
    per_view_per_point_pixel = per_view_per_point_pixel.clip(0, view_img_res - 1)
    per_view_per_point_pixel = per_view_per_point_pixel.long()
    per_view_per_point_pixel = torch.cat((per_view_per_point_pixel[:, :, 1].unsqueeze(-1),
                                            per_view_per_point_pixel[:, :, 0].unsqueeze(-1)),
                                dim=-1)  # switch x and y if you ever need to query pixel coordiantes



    # per_point_view_weight =  similarity_between_point_normal_and_view_dir
    # per_point_view_weight[~(per_view_per_point_visibility.permute(1,0).bool())] -=100
    '''Non-Border-First Unprojection (UBF)'''
    point_num = per_point_face_normal.shape[0]
    # candidate_per_point_per_view_mask = torch.ones((point_num,view_num)).bool().to(device) # [point_num,view_num]

    # first use shrinked visibility (only contains non-border areas) 
    shrinked_per_view_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[0]
    shrinked_per_view_per_point_visibility = \
        shrinked_per_view_per_pixel_visibility.permute(1, 2, 0)[per_pixel_mask].permute(1, 0)

    candidate_per_point_per_view_mask = \
        shrinked_per_view_per_point_visibility.permute(1, 0)  # [point_num,view_num]

    # multi-level NBF: the size of border areas can be controled by the dilation kernels
    for i in range(1,len(edge_dilate_kernels)):
        # if a point is not visible in any view, try less tight mask 
        # (we have multiple shrinked visibility mask with different dilation kernels. 
        # a smaller kernel means a smaller area is regarded as border area, which enables more areas to be considered by projection (less tight mask)
        per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)

        shrinked_per_view_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[i]
        shrinked_per_view_per_point_visibility = \
            shrinked_per_view_per_pixel_visibility.permute(1, 2, 0)[per_pixel_mask].permute(1, 0)

        candidate_per_point_per_view_mask[per_point_left_view_num < 1, :] = \
            torch.logical_or(
                candidate_per_point_per_view_mask[per_point_left_view_num < 1, :],
                shrinked_per_view_per_point_visibility.permute(1, 0)[per_point_left_view_num < 1, :]
            )


    if complete_unseen_by_projection:
        # if a point is not visible in any view's non-border area, now we consider all areas, no matter border or not, by using the unshrinked visibility
        per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)
        candidate_per_point_per_view_mask[per_point_left_view_num < 1, :] = \
            torch.logical_or(
                candidate_per_point_per_view_mask[per_point_left_view_num < 1, :],
                per_view_per_point_visibility.permute(1, 0)[per_point_left_view_num < 1, :]
            )


    # now choose the ones with best normal similarity
    per_point_per_view_weight = torch.softmax(similarity_between_point_normal_and_view_dir,1) # [pointnum, view_num]
    per_point_per_view_weight[~candidate_per_point_per_view_mask] = -100
    point_view_ids = torch.argmax(per_point_per_view_weight, dim=1)
    
    if not complete_unseen_by_projection:
        point_view_ids[candidate_per_point_per_view_mask.sum(1)<1] = -100 #view_num # mark unseen areas

    single_view_atlas_imgs = torch.zeros((view_num,res, res, 3), device=device)
    single_view_atlas_masks = torch.zeros((view_num,res, res, 3), device=device).bool()
    atlas_img = torch.zeros((res, res, 3), device=device)
    atlas_painted_mask = torch.zeros((res,res),device=device).bool()
    per_pixel_view_id = -torch.ones((res,res),device=device).long()



    # paint each pixel in the atlas by the query color in each view img
    for i in range(view_num):

        point_this_view_mask = point_view_ids == i

        view_img = mv_imgs[i]
        view_img = torch.flip(view_img,[1]) # flip upside down
        view_img = view_img.permute(1,2,0) # HWC

        atlas_img[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                    points_atlas_pixel_coord[point_this_view_mask][:, 1]] = \
            view_img[per_view_per_point_pixel[i][point_this_view_mask][:,0],
                        per_view_per_point_pixel[i][point_this_view_mask][:,1]]

        per_pixel_view_id[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                    points_atlas_pixel_coord[point_this_view_mask][:, 1]] = i

        atlas_painted_mask[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                    points_atlas_pixel_coord[point_this_view_mask][:, 1]] = True




    
    return atlas_img,shrinked_per_view_per_pixel_visibility,point_view_ids,points_atlas_pixel_coord,points,atlas_painted_mask




def find_border_edge_vertexs(faces,visible_vertex_mask):
    '''
    if a vertex is visible, but has invisible neighbor, it's regarded as a border-edge vertex
    faces: [F,3]
    visible_vertex_mask: [N,V]
    '''
    device= faces.device
    N = visible_vertex_mask.size(0)
    V = faces.max().item()+1
    # Step 1: Create a mask for each face indicating if the vertex is visible for each batch.
    per_view_per_face_per_vertex_visibility = visible_vertex_mask[:, faces]  # Shape: [N, F, 3]

    # Step 2: Identify faces that have both visible and non-visible vertices in each batch.
    per_view_per_face_mix_mask = (per_view_per_face_per_vertex_visibility.sum(dim=2) > 0) & (per_view_per_face_per_vertex_visibility.sum(dim=2) < 3)  # Shape: [N, F]

    
    border_edge_vertex_mask_NV = torch.zeros(N,V).bool().to(device)
    for i in range(N):
        face_mix_mask = per_view_per_face_mix_mask[i] # [F]

        mix_vertices = faces[face_mix_mask].unique() # [M] , M is the total number of mixed visibility vertices in this view
        border_edge_vertex_mask_NV[i,mix_vertices.long()] = True
    return border_edge_vertex_mask_NV
    


def NBF_prepare_per_kernel_per_view_mv_img_validation(vertices,faces,
              cam_positions,mv_img_res,crop_views=None,
              edge_dilate_kernel=9, # 21 too big, 15 seems ok
              renderer=None,
              debug=False,vertex_colors=None,mv_imgs_NHWC=None,save_img_path=None): 
    
    '''
    Prepare per-view per-kernel multiview images for NeRF-Borders, given a mesh and camera positions.
    
    Args:
    - vertices: [V,3]
    - faces: [F,3]
    - cam_positions: [Camera_num,3]
    - mv_img_res: int, the resolution of the output multiview images
    - crop_views: None or a list of the crop views for each camera
    - edge_dilate_kernel: int, the kernel size for dilating the border edges
    - debug: bool, whether to save the visualization results
    - vertex_colors: [V,3] or None, the colors of the vertices
    - mv_imgs_NHWC: [N,H,W,3] or None, the multiview images
    - save_img_path: str or None, the path to save the visualization results
    
    Returns:
    - border_area_Nrr: [N,mv_img_res,mv_img_res], the border areas in each view
    - border_edge_Nrr: [N,mv_img_res,mv_img_res], the border edges in each view
    '''
    device = vertices.device
    N = len(cam_positions)
    initial_cam_res = 256
    # get per-atlas-pixel's corresponding depth and uv in multiview images 
    # (depth used for calculating visibility, uv used for query correspondign color)
    
    # renderer = GeneralRenderer(device=device)
    camera_positions = cam_positions
    
    # 1. Find vertices that are visible but have invisible neighbors
    visible_face_mask,visible_vertex_mask_NV = \
        get_view_visible_V_F(vertices,faces,camera_positions,renderer,eps = 0,return_other=False,crop_views=crop_views)
        

    border_edge_vertex_mask_NV = find_border_edge_vertexs(faces,visible_vertex_mask_NV).float()
    
    # vis = False
    # if vis:
    #     import kiui
    #     # kiui.lo(visible_vertex_mask_NV)
    #     # kiui.lo(border_edge_vertex_mask_NV)
    #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk,get_pc_actor_vtk
    #     vis_actors_vtk(
    #         [
    #             get_colorful_pc_actor_vtk(vertices[visible_vertex_mask_NV[0].bool()].detach().cpu().numpy(),
    #                                         point_colors=vertex_colors[visible_vertex_mask_NV[0].bool()].detach().cpu().numpy(),opacity=0.5),
    #             get_pc_actor_vtk(vertices[border_edge_vertex_mask_NV[0].bool()].detach().cpu().numpy(),
    #                              color=[1.0,0.0,0.0],opacity=1),
    #         ]
    #     )
   
    
    # 2. Find their corresponding 2D pixels in rendered mv imgs (we fixed the initial mv img resolution and kernel size to avoid adjusting hyper parameters given different resolution of mesh)
    border_edge_masks_NHW = torch.zeros((N, initial_cam_res, initial_cam_res), dtype=torch.float).to(device)
    if debug:
        border_area_masks_NHW = torch.zeros((N, initial_cam_res, initial_cam_res), dtype=torch.float).to(device)
    use_vertex_pixels = False
    if use_vertex_pixels:

        transformed_vertices = camera_trans_points(vertices,camera_positions)
        transformed_vertices = crop_vertices(transformed_vertices,crop_views)
    
        per_view_per_vertex_uvs_NV2 = transformed_vertices[..., :2] # from -1 to 1
        per_view_per_vertex_uvs_NV2 = (per_view_per_vertex_uvs_NV2 + 1)*0.5
    
        per_view_per_vertex_pixels_NV2 = per_view_per_vertex_uvs_NV2 * initial_cam_res
        per_view_per_vertex_pixels_NV2 = per_view_per_vertex_pixels_NV2.clip(0, initial_cam_res - 1)
        per_view_per_vertex_pixels_NV2 = per_view_per_vertex_pixels_NV2.long()
        per_view_per_vertex_pixels_NV2 = torch.cat((per_view_per_vertex_pixels_NV2[:, :, 1].unsqueeze(-1), per_view_per_vertex_pixels_NV2[:, :, 0].unsqueeze(-1)),
                                dim=-1)  # switch x and y if you ever need to query pixel coordiantes

        for i in range(N):
            border_edge_masks_NHW[
                                    i,  
                                    per_view_per_vertex_pixels_NV2[i,..., 0],
                                    per_view_per_vertex_pixels_NV2[i,..., 1],     
                                ] = border_edge_vertex_mask_NV[i]
    else:
        
        for i in range(N):
            crop = False
            temp_crop_views = None
            if crop_views is not None:
                if i in crop_views:
                    crop = True
                    temp_crop_views = [0]
     
            colors = torch.zeros_like(vertices).to(device)
            colors[border_edge_vertex_mask_NV[i].bool()] = torch.tensor([1,1,1],dtype=torch.float).to(device)
            temp_1HW3 = renderer.render(vertices,colors*2-1,faces,camera_positions_np=camera_positions[i:i+1].detach().cpu().numpy(),
                                        rotate_normal=False,crop_views=temp_crop_views,res=initial_cam_res)
            # Image.fromarray((temp_1HW3[0,:,:,:]*255).detach().cpu().numpy().astype(np.uint8)).save(os.path.join( f'{i}.png'))
            border_edge_masks_NHW[i] = temp_1HW3[0,:,:,0]
    
    # 3. Now we find the foreground edge for each view, which will not be considered as a border
    if debug:
        per_pixel_face_ids,rgba_NHW4 = renderer.get_per_pixel_face_ids_and_rgba(vertices,vertex_colors,
                                                                                faces,camera_positions_np=camera_positions.detach().cpu().numpy(),
                                                            res = initial_cam_res,crop_views=crop_views)
        # import kiui
        # kiui.vis.plot_image(rgba_NHW4[0].detach().cpu().numpy())
    else:
        per_pixel_face_ids = renderer.get_per_pixel_face_ids(vertices,faces,camera_positions_np=camera_positions.detach().cpu().numpy(),
                                                            res = initial_cam_res,crop_views=crop_views)
    foreground_mask_NHW = per_pixel_face_ids >= 0
    foreground_edges_NHW = detect_edges_in_gray_by_scharr_torch_batch(foreground_mask_NHW.unsqueeze(1).float()).squeeze(1).bool()
    for i in range(N):
        border_edge_masks_NHW[foreground_edges_NHW] = False
    
     
    # 4. Dilate the corresponding 2D areas as the final border-areas
    for i in range(N):
        if debug:
            border_area_masks_NHW[i] = dilate_torch_batch(border_edge_masks_NHW[i].unsqueeze(0).float(), edge_dilate_kernel)[0]
        else:
            border_edge_masks_NHW[i] = dilate_torch_batch(border_edge_masks_NHW[i].unsqueeze(0).float(), edge_dilate_kernel)[0]
    if not debug:
        border_area_masks_NHW = border_edge_masks_NHW
        
    # 5. resize the mask if necessary

    if mv_img_res != initial_cam_res:
        border_edge_masks_N1HW = border_edge_masks_NHW.unsqueeze(1)
        border_edge_N1rr = F.interpolate(border_edge_masks_N1HW, size=(mv_img_res, mv_img_res), mode='nearest') 
        border_edge_Nrr = border_edge_N1rr.squeeze(1)
        border_area_Nrr = border_edge_Nrr
        if debug:
            border_area_masks_N1HW = border_area_masks_NHW.unsqueeze(1)
            border_area_N1rr = F.interpolate(border_area_masks_N1HW, size=(mv_img_res, mv_img_res), mode='nearest') 
            border_area_Nrr = border_area_N1rr.squeeze(1)
            
            foreground_edges_N1HW = foreground_edges_NHW.unsqueeze(1)
            foreground_edges_N1rr = F.interpolate(foreground_edges_N1HW.float(), size=(mv_img_res, mv_img_res), mode='nearest').bool() 
            foreground_edges_Nrr = foreground_edges_N1rr.squeeze(1)
            
            rgba_Nrr4 = F.interpolate(rgba_NHW4.permute(0,3,1,2), size=(mv_img_res, mv_img_res)).permute(0,2,3,1)

    else:
        border_area_Nrr = border_edge_masks_NHW
        
    # 5. Save visualization results if debug == True
    if debug:
        # mv_imgs_with_foregournd_edge_NHW3 = mv_imgs_NHWC[...,:3].clone()
        # mv_imgs_with_foregournd_edge_NHW3[foreground_edges_Nrr.bool()] = torch.tensor([1,0,0],dtype=torch.float32).to(device)

        mv_imgs_with_edge_NHW3 = mv_imgs_NHWC[...,:3].clone()
        mv_imgs_with_edge_NHW3[border_edge_Nrr.bool()] = torch.tensor([0,0,1],dtype=torch.float32).to(device)
        mv_imgs_with_edge_NHW3[foreground_edges_Nrr.bool()] = torch.tensor([1,0,0],dtype=torch.float32).to(device)
        
        mvimgs_with_border_NHW3 = mv_imgs_NHWC[...,:3].clone()
        mvimgs_with_border_NHW3[border_area_Nrr.bool()] = torch.tensor([0,0,1],dtype=torch.float32).to(device)
        
        rendered_imgs_with_edge_NHW3 = rgba_Nrr4[...,:3].clone()
        rendered_imgs_with_edge_NHW3[border_edge_Nrr.bool()] = torch.tensor([0,0,1],dtype=torch.float32).to(device)
        rendered_imgs_with_edge_NHW3[foreground_edges_Nrr.bool()] = torch.tensor([1,0,0],dtype=torch.float32).to(device)
        
        renderedimgs_with_border_NHW3 = rgba_Nrr4[...,:3].clone()
        renderedimgs_with_border_NHW3[border_area_Nrr.bool()] = torch.tensor([0,0,1],dtype=torch.float32).to(device)
        
        cat = torch.cat([mvimgs_with_border_NHW3,mv_imgs_with_edge_NHW3],dim=2) # [N,res,res*x,3]
        cat2 = torch.cat([renderedimgs_with_border_NHW3,rendered_imgs_with_edge_NHW3],dim=2) # [N,res,res*x,3]
        cat = torch.cat([cat,cat2],dim=2)
        cat = cat.reshape(N*mv_img_res,-1,3)
        Image.fromarray((cat.detach().cpu().numpy()*255).astype(np.uint8)).save(save_img_path)
    
    return border_area_Nrr.bool(), border_edge_Nrr.bool()
    
    
    
    
    

    
    
    

def NBF_prepare_per_kernel_per_view_shrinked_per_vert_visibility(vertices,faces,
              cam_positions,cam_res,atlas_res = 1024,
              edge_dilate_kernels=[21],save_img_path=None):
    '''
    uvs:                per mesh vertex uv:                     [vert_num, 2]
    mesh_tex_idx:       per face uv coord id (to index uvs):    [face_num,3]
    gb_pos:             per pixel 3D coordinate:                [1,res,res,3] 
    mask:               per pixel validation:                   [1,res,res,1]
    per_pixel_face_id:  per pixel face id:                      [1,res,res]
    
    '''
    
    # view_img_res = res
    # res = xatlas_texture_res
    
    res = atlas_res
    view_num = len(cam_positions)
    device = cam_positions.device
    renderer = GeneralRenderer(device=device)
    glctx = renderer._glctx
    base_dirs = cam_positions
    
    original_vertices = vertices.clone()
    original_faces = faces.clone()
    
    # simplify mesh
    simp_vertices,simp_faces = simplify_mesh(vertices,faces)
    
    # uv unwrapping
    
    uvs, mesh_tex_idx, gb_pos, mask,per_atlas_pixel_face_id = xatlas_uvmap_w_face_id(
            glctx, simp_vertices, simp_faces, resolution=atlas_res)

    xatlas_dict = {'uvs': uvs, 'mesh_tex_idx': mesh_tex_idx, 'gb_pos': gb_pos, 
                    'mask': mask,'per_atlas_pixel_face_id':per_atlas_pixel_face_id}

    # get points_atlas_pixel_coord
    per_pixel_mask = mask[0,:,:,0] # [res,res]
    per_pixel_point_coord = gb_pos[0] # [res,res,3]
    per_atlas_pixel_face_id = per_atlas_pixel_face_id[0] #[1,res,res]

    per_pixel_pixel_coord = torch.zeros((res,res,2),device=device).long()
    xx, yy = torch.meshgrid(torch.arange(res).to(device), torch.arange(res).to(device))
    per_pixel_pixel_coord[:, :, 0] = xx
    per_pixel_pixel_coord[:, :, 1] = yy

    points = per_pixel_point_coord[per_pixel_mask] # [?,3] ??
    points_atlas_pixel_coord = per_pixel_pixel_coord[per_pixel_mask].long() # [?,2] ??


    # get per-atlas-pixel's corresponding depth and uv in multiview images 
    # (depth used for calculating visibility, uv used for query correspondign color)
    transformed_points = camera_trans_points(points,cam_positions)
    per_view_per_point_depths = transformed_points[ ..., 2]
    per_view_per_point_uvs = transformed_points[..., :2] # from -1 to 1
    per_view_per_point_uvs = (per_view_per_point_uvs + 1)*0.5

    # Get mesh normalized depth (used for visibility calculation)
    simp_vertices_clip = camera_trans_points(simp_vertices,cam_positions) # [C,V,4]
    
    # save_obj(simp_vertices.detach().cpu().numpy(),simp_faces.detach().cpu().numpy(),
    #          simp_vertices.detach().cpu().numpy(),'simp_mesh.obj')
    simp_rast_out,_ = dr.rasterize(glctx, simp_vertices_clip, simp_faces.int(), resolution=(cam_res,cam_res), grad_db=False) #C,H,W,4
    simp_vertice_depths = simp_vertices_clip[...,-2] # [C,V]
   
   
    
    vert_col = simp_vertice_depths.unsqueeze(-1).repeat(1,1,3) # [C,V,3]
    # kiui.lo(vert_col)
    col,_ = dr.interpolate(vert_col, simp_rast_out, simp_faces.int()) #C,H,W,3
    # alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
    mesh_normalized_depths = depth_CHW = col[:,:,:,0] #C,H,W
    # alpha_CHW = alpha.squeeze(-1)
    vis_depth_maps = False
    if vis_depth_maps:
        import kiui
        for i in range(1):
            mask = mesh_normalized_depths[i] > 0
            img = mesh_normalized_depths
            img[i][mask] =  (mesh_normalized_depths[i][mask] - mesh_normalized_depths[i][mask].min()) / (mesh_normalized_depths[i][mask].max() - mesh_normalized_depths[i][mask].min())
            
            kiui.vis.plot_image(img[i])

    # Get per-atls-pixel  visibility by depth (so that we have per-view visible atlas)
    
    per_view_per_point_visibility,_ = get_point_validation_by_depth(cam_res,per_view_per_point_uvs,
                                    per_view_per_point_depths,mesh_normalized_depths,offset = 0.0001,
                                                                    vis=False)# [cam_num, point_num]
    # import kiui
    # kiui.lo(mesh_normalized_depths)
    # kiui.lo(per_view_per_point_depths)

    vis_per_view_per_point_visibility=False
    if vis_per_view_per_point_visibility:
        print('vis_per_view_per_point_visibility')
        from fancy123.utils.vtk_basic import vis_actors_vtk,get_pc_actor_vtk,get_colorful_pc_actor_vtk
        for i in range(1):
            print('get point validation by depth')
            vis_actors_vtk(
                [
                    get_pc_actor_vtk(points[per_view_per_point_visibility[i]].detach().cpu().numpy(),color=(0,0,1),opacity=0.1,point_size=2),
                    get_pc_actor_vtk(points[~per_view_per_point_visibility[i]].detach().cpu().numpy(),color=(1,0,0),opacity=0.4,point_size=5),
                ]
            )

    per_atlas_pixel_per_view_visibility = torch.zeros((res,res,view_num),device=device).bool()

    per_atlas_pixel_per_view_visibility[per_pixel_mask] = per_view_per_point_visibility.permute(1,0)#.clone() # (res,res,view_num)
    vis_per_atlas_pixel_per_view_visibility=False
    if vis_per_atlas_pixel_per_view_visibility:
        print('vis_per_atlas_pixel_per_view_visibility')
        import kiui
        for i in range(1):
            img = per_atlas_pixel_per_view_visibility[...,i]
            kiui.vis.plot_image(img)
    # shrink per-view visible atlas (remove border areas, only keep non-border areas for later use)
    per_kernel_per_view_shrinked_per_pixel_visibility = get_shrinked_per_view_per_pixel_visibility_torch(
        per_pixel_mask,per_atlas_pixel_per_view_visibility,
        kernel_sizes= edge_dilate_kernels*(atlas_res//256),
        save_path = os.path.join(save_img_path,'shrink_per_view_edge')) # [kernel_num,view_num,res,res]

    vis_shrinked_visibility_per_point=False
    if vis_shrinked_visibility_per_point:
        print('vis_shrinked_visibility_per_point')
        per_view_shrinked_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[0]
        per_view_shrinked_per_point_visibility = per_view_shrinked_per_pixel_visibility[...,
                                                                                           points_atlas_pixel_coord[:, 0], 
                                                                                           points_atlas_pixel_coord[:, 1]]
        from fancy123.utils.vtk_basic import vis_actors_vtk,get_pc_actor_vtk,get_colorful_pc_actor_vtk
        for i in range(1):
            import kiui
            kiui.lo(per_view_shrinked_per_point_visibility[i]) # all false here
            vis_actors_vtk(
                [
                    get_pc_actor_vtk(points[per_view_shrinked_per_point_visibility[i]].detach().cpu().numpy(),color=(0,0,1)),
                    get_pc_actor_vtk(points[~per_view_shrinked_per_point_visibility[i]].detach().cpu().numpy(),color=(1,0,0)),
                ]
            )
    # so far we've get the per-atlas-pixel's visibility with NBF areas
    ###############################################################################
    
    ## Subdivide the mesh, so that even with per vertex color, the color resolution wouldn't be too low
    subdivided_vertices = simp_vertices.cpu().numpy()
    subdivided_faces = simp_faces.cpu().numpy()
    subdivided_uvs = uvs.cpu().numpy()
    subdivided_face_uv_idx = mesh_tex_idx.cpu().numpy()
    iterations = 2
    for i in range(iterations):
        subdivided_vertices, subdivided_faces, subdivided_uvs, subdivided_face_uv_idx = subdivide_with_uv(
            subdivided_vertices, subdivided_faces, subdivided_face_uv_idx,subdivided_uvs,face_index=None)
    subdivided_vertices = torch.tensor(subdivided_vertices).to(device)
    subdivided_faces = torch.tensor(subdivided_faces).to(device).long()
    subdivided_uvs = torch.tensor(subdivided_uvs).to(device)
    subdivided_face_uv_idx = torch.tensor(subdivided_face_uv_idx).to(device).long()
    
    
    ## Get per vertex's atlas pixel coordinate of the subdivided mesh
    vertex_only_uv_mask,unique_vertex_uv_pairs = compute_vertex_only_uv_mask(subdivided_faces,subdivided_face_uv_idx)
    subdivided_vert_uvs = torch.zeros(len(subdivided_vertices), 2).to(device)

    unique_vertex_uvs = subdivided_uvs[unique_vertex_uv_pairs[:, 1]]
    subdivided_vert_uvs[unique_vertex_uv_pairs[:, 0]] = unique_vertex_uvs

    
    subdivided_vert_uv_pixel_coords = (subdivided_vert_uvs*atlas_res).clip(0,atlas_res-1).long().to(device)
    subdivided_vert_uv_pixel_coords = torch.cat((subdivided_vert_uv_pixel_coords[..., 1].unsqueeze(-1),
                                        subdivided_vert_uv_pixel_coords[..., 0].unsqueeze(-1)),
                            dim=-1)  # switch x and y if you ever need to query pixel coordiantes
    per_kernel_per_view_shrinked_per_vert_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[...,
                                                                                           subdivided_vert_uv_pixel_coords[:, 0], 
                                                                                           subdivided_vert_uv_pixel_coords[:, 1]]
    
    
    return subdivided_vertices, subdivided_faces, subdivided_uvs, subdivided_face_uv_idx, \
            xatlas_dict, per_kernel_per_view_shrinked_per_vert_visibility
    

def get_shrinked_per_view_per_pixel_visibility_torch(per_pixel_mask,per_atlas_pixel_per_view_visibility,
                                                     kernel_sizes = [21],save_path=None):
    '''
    :param per_pixel_mask: res,res
    :param per_atlas_pixel_per_view_visibility:
    :return:
    '''
    if kernel_sizes[0] == 0:
        return per_atlas_pixel_per_view_visibility.permute(2,0,1).unsqueeze(0)
    device = per_atlas_pixel_per_view_visibility.device
    view_num = per_atlas_pixel_per_view_visibility.shape[-1]
    atlas_background_edges = detect_edges_in_gray_by_scharr_torch_batch(
        per_pixel_mask.unsqueeze(0).unsqueeze(0).float() * 255.0)  # [1,1,res,res]
    atlas_background_edges_mask = atlas_background_edges > 125  # [1,1,res,res]
    atlas_background_edges_mask = atlas_background_edges_mask[0] # [1,res,res]

    per_view_atlas_edges = detect_edges_in_gray_by_scharr_torch_batch(
        per_atlas_pixel_per_view_visibility.permute(2, 0, 1).unsqueeze(1).float() * 255.0)  # [view_num,1,res,res]
    per_view_atlas_edges_mask = per_view_atlas_edges > 255.0 / 2 - 1  # [view_num,1,res,res]
    per_view_atlas_edges_mask = per_view_atlas_edges_mask.squeeze(1)  # [view_num,res,res]
    per_view_atlas_edges_mask = per_view_atlas_edges_mask * ~atlas_background_edges_mask # [view_num,res,res]
    per_kernel_per_view_shrinked_per_pixel_visibility = []
    for kernel_size in kernel_sizes:
        per_view_atlas_border_mask = dilate_torch_batch(per_view_atlas_edges_mask.float() * 255.0,
                                                        kernel_size=kernel_size)  # [view_num,res,res]
        per_view_atlas_border_mask = per_view_atlas_border_mask>255.0/2
        shrinked_per_view_per_pixel_visibility = (per_atlas_pixel_per_view_visibility.permute(2, 0, 1) * \
                                                 (~per_view_atlas_border_mask)) # [view_num,res,res]
        per_kernel_per_view_shrinked_per_pixel_visibility.append(shrinked_per_view_per_pixel_visibility)
    per_kernel_per_view_shrinked_per_pixel_visibility = torch.stack(per_kernel_per_view_shrinked_per_pixel_visibility,0)  # [kernel_num,view_num,res,res]
    if save_path is not None:
        os.makedirs(save_path,exist_ok=True)


        src_img_color_with_edges = per_atlas_pixel_per_view_visibility.permute(2,0,1).clone().float().unsqueeze(-1).repeat(1,1,1,3)  # [view_num,res,res,3,]

        src_img_color_with_edges[atlas_background_edges_mask.repeat(view_num,1,1)]=  torch.tensor([[1.0,0,0]],device=device) # background edges painted in red
        src_img_color_with_edges[per_view_atlas_edges_mask] = torch.tensor([[0,0,1.0]],device=device)


        src_img_color_with_edges = src_img_color_with_edges.permute(0,3,1,2)  # [view_num,3,res,res,]
        for i in range(view_num):
            cat = cat_images(src_img_color_with_edges[i].detach().cpu().numpy(),
                             per_view_atlas_edges_mask[i].float().unsqueeze(0).repeat(3,1,1).detach().cpu().numpy())
            cat = cat_images(cat,per_view_atlas_border_mask[i].float().unsqueeze(0).repeat(3,1,1).detach().cpu().numpy())
            save_CHW_RGB_img(cat[:,::-1,:],os.path.join(save_path,f'{i}.png'))
    return per_kernel_per_view_shrinked_per_pixel_visibility



