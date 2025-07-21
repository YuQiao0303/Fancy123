import os
from PIL import Image
from einops import rearrange
import numpy as np
import torch
from fancy123.utils.utils2d import dilate_torch_batch,detect_edges_in_gray_by_scharr_torch_batch, save_CHW_RGB_img
from src.models.geometry.render.neural_render import compute_vertex_normal
from src.utils.infer_util import resize_foreground
from src.utils.mesh_util import loadobj_color
from unique3d.app.utils import make_image_grid, rgba_to_rgb, split_image
from unique3d.mesh_reconstruction.remesh import calc_face_normals
from fancy123.render.general_renderer import GeneralRenderer
import kaolin as kal
import torch.nn.functional as F

from unique3d.scripts.refine_lr_to_sr import refine_lr_with_sd


def sr_6_view_imgs_by_SD(input_image_pil,mv_imgs_pil,model_zoo):
    prompt = "4views, multiview"
    NEG_PROMPT="sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"
    neg_prompt = NEG_PROMPT
    
    
    mv_imgs_pil = rgba_to_rgb(mv_imgs_pil)
    divided_6_mv_imgs_pil = split_image(mv_imgs_pil, rows=3,cols=2)
    first_4_mv_imgs_pil = [divided_6_mv_imgs_pil[i] for i in [0,1,4,3]]
    last_4_mv_imgs_pil = [divided_6_mv_imgs_pil[i] for i in [5,1,4,2]]
    
    rgb_pil = make_image_grid(first_4_mv_imgs_pil, rows=2)
    control_image = rgb_pil.resize((1024, 1024))
    refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(input_image_pil)], [control_image], prompt_list=[prompt], neg_prompt_list=[neg_prompt], pipe=model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i, strength=0.2, output_size=(1024, 1024))[0]
    refined_first_4_image_pil = split_image(refined_rgb, rows=2,cols=2)
    
    rgb_pil = make_image_grid(last_4_mv_imgs_pil, rows=2)
    rgb_pil.save('sr_input2.png')
    control_image = rgb_pil.resize((1024, 1024))
    refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(input_image_pil)], [control_image], prompt_list=[prompt], neg_prompt_list=[neg_prompt], pipe=model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i, strength=0.2, output_size=(1024, 1024))[0]
    refined_last_4_image_pil = split_image(refined_rgb, rows=2,cols=2)
    
    refined_deformed_imgs_pil_list = [
        refined_first_4_image_pil[0] ,
        refined_first_4_image_pil[1] ,
        refined_last_4_image_pil[3] ,
        refined_first_4_image_pil[3] ,
        refined_first_4_image_pil[2] ,
        refined_last_4_image_pil[0] ,
        ]
    refined_deformed_imgs_pil = make_image_grid(refined_deformed_imgs_pil_list, rows=3,cols=2)
    return refined_deformed_imgs_pil,refined_deformed_imgs_pil_list

def sr_4_view_imgs_by_SD(input_image_pil,mv_img_pils, model_zoo):
    prompt = "4views, multiview"
    NEG_PROMPT="sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"
    neg_prompt = NEG_PROMPT
    
    
    rgb_pil = make_image_grid(mv_img_pils, rows=2)
    control_image = rgb_pil.resize((1024, 1024))
    refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(input_image_pil)], [control_image], prompt_list=[prompt], neg_prompt_list=[neg_prompt], pipe=model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i, strength=0.2, output_size=(1024, 1024))[0]
    refined_first_4_image_pils = split_image(refined_rgb, rows=2,cols=2)
 
  
    return refined_first_4_image_pils
    
    
def load_test_data_instantmesh(name='cute_horse',device='cuda',input_path = 'outputs/instant-mesh-large',load_normals=False):
    # Load input image
    input_image_path = os.path.join(input_path,'images', f'{name}_input.png')
    ori_input_image_pil = Image.open(input_image_path)  # ( H,W,3)
    ori_input_image_pil = resize_foreground(ori_input_image_pil, 1.0) 
    
    input_image_pil = ori_input_image_pil.resize((320, 320))#.convert('RGB') # (H,W,C)
    input_image = np.asarray(input_image_pil, dtype=np.float32) / 255.0
    input_image_HW4 = torch.from_numpy(input_image).contiguous().float().to(device)    
    
    # Load multi-view images
    images_path = os.path.join(input_path,'images', f'{name}_rmbg.png')
    if not os.path.exists(images_path):
        images_path = os.path.join(input_path,'images', f'{name}.png')
    
    mv_images_pil = Image.open(images_path)  # ( 960, 640,C)

    mv_images_whole = np.asarray(mv_images_pil, dtype=np.float32) / 255.0
    mv_images_whole = torch.from_numpy(mv_images_whole).permute(2, 0, 1).contiguous().float().to(device)      # (C, 960, 640)
    mv_images = rearrange(mv_images_whole, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)       # (6, C, 320, 320)
    mv_images_BHW4 = mv_images.permute(0,2,3,1)# 6,H,W,4

    # for i in range(6):
    #     img_pil = Image.fromarray((mv_images[i].cpu().numpy()*255).astype(np.uint8))
    #     img_pil.save(f'outputs/instant-mesh-large/images/{name}_view{i+1}.png')
    mv_images_whole = mv_images_whole.permute(1,2,0) # 960,640,4
    
    # if mv_images_pil.mode != 'RGBA':
    #     mv_imgs_pil_list = [Image.fromarray((mv_images_BHW4[i]*255.0).cpu().numpy().astype(np.uint8)) for i in range(6)]
        
    

    # Load mesh
    # mesh_path_idx = f'outputs/instant-mesh-large/meshes/{name}_zero.obj'
    mesh_path_idx = os.path.join(input_path,'meshes',f'{name}.obj')
    # mesh_path_idx = f'temp/sphere.obj'
    vertices, faces, vertex_colors  = loadobj_color(mesh_path_idx)
    vertices = vertices.astype(np.float32)
    vertex_colors = (vertex_colors*255.0).clip(0,255).astype(np.uint8)
    # vertices_old = vertices.copy()
    # vertices[...,0]=vertices_old[...,2]
    # vertices[...,1]=vertices_old[...,0]
    # vertices[...,2]=vertices_old[...,1]

    vertices = torch.tensor(vertices).float().to(device)
    faces = torch.tensor(faces).long().to(device)
    vertex_colors = torch.tensor(vertex_colors).float().to(device)/255.0
    
    
    # Load multi-view normals
    if load_normals:
        normals_path = os.path.join(input_path,'images', f'{name}_normal_rmbg.png')
        if not os.path.exists(normals_path):
            normals_path = os.path.join(input_path,'images', f'{name}_normal.png')
        if os.path.exists(normals_path):
            mv_normals_pil = Image.open(normals_path)  # ( 960, 640,C)
            # mv_normals_pil = mv_normals_pil.convert('RGBA')
            # alpha_channel = mv_images_pil.resize(mv_normals_pil.size).split()[-1]
            # mv_normals_pil.putalpha(alpha_channel)
            # output_path = f'outputs/instant-mesh-large/images/{name}_normal_with_alpha.png'
            # mv_normals_pil.save(output_path)
            
            mv_normals = np.asarray(mv_normals_pil, dtype=np.float32) / 255.0
            mv_normals = torch.from_numpy(mv_normals).permute(2, 0, 1).contiguous().float().to(device)      # (C, 960, 640)
            mv_normals = rearrange(mv_normals, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)       # (6, C, 320, 320)
            mv_normals = mv_normals.permute(0,2,3,1)# 6,H,W,4
            return vertices,faces,vertex_colors,mv_images_BHW4,mv_images_pil,input_image_HW4,input_image_pil,ori_input_image_pil,mv_normals
        else:
            print(f'No normals found at {normals_path}')
            raise ValueError('No normals found')
    return vertices,faces,vertex_colors,mv_images_BHW4,mv_images_pil,input_image_HW4,input_image_pil,ori_input_image_pil,None





def pt3d_laplacian(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    # https://github.com/facebookresearch/pytorch3d/blob/44702fdb4ba0f80e96bee724766c545d4d93509c/pytorch3d/ops/laplacian_matrices.py#L23
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L
    

def pt3d_compute_edge(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # https://github.com/facebookresearch/pytorch3d/blob/44702fdb4ba0f80e96bee724766c545d4d93509c/pytorch3d/structures/meshes.py#L1024
    """
    Computes edges in packed form from the packed version of faces and verts.
    """


    F = faces.shape[0]
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    # All edges including duplicates.
    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)


    # Sort the edges in increasing vertex order to remove duplicates as
    # the same edge may appear in different orientations in different faces.
    # i.e. rows in edges after sorting will be of the form (v0, v1) where v1 > v0.
    # This sorting does not change the order in dim=0.
    edges, _ = edges.sort(dim=1)

    # Remove duplicate edges: convert each edge (v0, v1) into an
    # integer hash = V * v0 + v1; this allows us to use the scalar version of
    # unique which is much faster than edges.unique(dim=1) which is very slow.
    # After finding the unique elements reconstruct the vertex indices as:
    # (v0, v1) = (hash / V, hash % V)
    # The inverse maps from unique_edges back to edges:
    # unique_edges[inverse_idxs] == edges
    # i.e. inverse_idxs[i] == j means that edges[i] == unique_edges[j]

    V = verts.shape[0]
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

    # Find indices of unique elements.
    # TODO (nikhilar) remove following 4 lines when torch.unique has support
    # for returning unique indices
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(
        edges_hash.shape[0], dtype=torch.bool, device=verts.device
    )
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    unique_idx = sort_idx[unique_mask]

    edges = torch.stack([u // V, u % V], dim=1)
    return edges



def complete_vert_colors_by_neighbors(vertices, faces,vertex_colors, valid_vert_id):
    device = vertex_colors.device
    V = vertex_colors.shape[0]
    vertex_colors.clip(0.0, 1.0)
    invalid_index = torch.ones_like(vertex_colors[:, 0]).bool()    # [V]
    invalid_index[valid_vert_id] = False
    invalid_index = torch.arange(V).to(device)[invalid_index]

    # L = kal.ops.mesh.uniform_laplacian(V, faces) # OOM

    edges = pt3d_compute_edge(vertices,faces)
    L = pt3d_laplacian(vertices,edges)
    # kiui.lo(L)
    E = torch.sparse_coo_tensor(torch.tensor([list(range(V))] * 2), torch.ones((V,)), size=(V, V)).to(device) # eye
    L = L + E

    colored_count = torch.ones_like(vertex_colors[:, 0])   # [V]
    colored_count[invalid_index] = 0
    L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]
    
    total_colored = colored_count.sum()
    coloring_round = 0
    stage = "uncolored"
    from tqdm import tqdm
    pbar = tqdm(miniters=100)
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, vertex_colors * colored_count[:, None])    # [IV, 3]
        new_count = torch.matmul(L_invalid, colored_count)[:, None]             # [IV, 1]
        vertex_colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, vertex_colors[invalid_index])
        colored_count[invalid_index] = (new_count[:, 0] > 0).float()
        
        new_total_colored = colored_count.sum()
        if new_total_colored > total_colored:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        pbar.update(1)
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break
    assert not torch.isnan(vertex_colors).any()
    vertex_colors = vertex_colors.clip(0.0, 1.0)
    return vertex_colors

def get_fixed_area(vertices,faces, 
                   face_fixed_mask,camera_position_np,renderer,res):
    ## Get fixed_area , dilated_fixed_area
    if res==320:
        kernel_size = 25
    else:
        print("!!!! res should be 320")
        raise NotImplementedError
    per_pixel_face_ids_1HW = renderer.get_per_pixel_face_ids(vertices,faces,
                        camera_positions_np =camera_position_np,res=res) # Camera_num,H,W (camera_num ==1)
    per_pixel_face_ids_HW = per_pixel_face_ids_1HW[0] # HW
    pixel_fixed_mask_HW = face_fixed_mask[per_pixel_face_ids_HW.long()] # H,W
    pixel_fixed_mask_HW[per_pixel_face_ids_HW.long()==-1] = False
    
    pixel_fixed_dilated_mask_1HW = dilate_torch_batch(pixel_fixed_mask_HW.unsqueeze(0).float()*255.0,kernel_size=kernel_size)# 1,H,W
    pixel_fixed_dilated_mask_HW = pixel_fixed_dilated_mask_1HW[0].bool() # H,W
    return pixel_fixed_mask_HW, pixel_fixed_dilated_mask_HW,per_pixel_face_ids_HW


def apply_view_color2mesh(vertices,faces,vertex_colors,camera_positions,imgs_BHW4,eps = 0.05,
                          res=2048,
                          renderer=None,mode='select',view_weights_B=None,crop_views=None,
                          per_view_shrinked_per_vert_visibility=None):
    '''
    camera_positions: B,3
    input_images: B,H,W,C
    
    mode: 'select': select best view;  'fuse' weighted sum of all views
    
    
    return
    visible_vert_mask: B,V
    '''
    B = imgs_BHW4.shape[0]
    # assert B  ==1 # so far only test with one single view
    # img_pil_list = [Image.fromarray((input_image*255).detach().cpu().numpy().astype(np.uint8)) for input_image in input_images]
    # img_pil_list = erode_alpha_and_dilate_foreground(img_pil_list,kernel_size=5)
    # img_pil_list[0].save('img_pil_list.png')
    device = vertices.device
    if res != imgs_BHW4.shape[1]:
        imgs_BHW4 = F.interpolate(imgs_BHW4.permute(0,3,1,2).float(),size=(res,res),mode='bilinear').float().permute(0,2,3,1)
    if vertex_colors is not None:
        new_vertex_colors = vertex_colors.clone() # [V,3]
    else:
        new_vertex_colors = vertices.clone()
    # if renderer is None:
    #     renderer = GeneralRenderer(device=device)
    visible_face_mask,visible_vertex_mask,per_pixel_face_ids,vertices_NDC,cos_angles = get_view_visible_V_F(
        vertices,faces,camera_positions,renderer,eps,return_other=True,crop_views=crop_views) 
   
    # visible_vertex_mask: [Camera_num,V]
    vertices_uv = vertices_NDC[...,:2] # from -1 to 1
    
    vertices_pixel_coord = (vertices_uv +1)*0.5*res # [camera_num, V,2]
    vertices_pixel_coord = vertices_pixel_coord.long().clip(0,res-1)
    vertices_pixel_coord = torch.cat((vertices_pixel_coord[..., 1].unsqueeze(-1), 
                                      vertices_pixel_coord[..., 0].unsqueeze(-1)), dim=-1)  # switch x and y
    
    # see if each point falls within the foreground of target images
    input_alphas = (imgs_BHW4[...,-1]*255.0).int() > 245 #== 255 # [camera_num,H,W]
    view_point_in_foreground_mask = torch.zeros((B,vertices.shape[0])).to(device).bool() # [Camera_num,V]
    for i in range(B):
        vis_alpha = False
        if vis_alpha:
            alpha = input_alphas[i].float().unsqueeze(0)#.repeat(3,1,1)
            alpha = imgs_BHW4[i,...,-1]
            import kiui
            kiui.vis.plot_image(alpha)
        # vis = True
        # if vis:
        #     temp_img = input_images[i].float()
        #     temp_img[...,-1] = input_alphas.float()[i]
        #     temp_img[input_alphas[i]==0] = torch.tensor([1.0,0.0,0.0,1.0]).float().to(device)
        #     import kiui
        #     kiui.vis.plot_image(temp_img)
        # view_img = torch.flip(view_img,[1]) # flip upside down, only use this when using kaolin's camera for camera transform
        view_point_in_foreground_mask[i] = input_alphas[i][
            vertices_pixel_coord[i][:,0],
            vertices_pixel_coord[i][:,1] ] # 
    if per_view_shrinked_per_vert_visibility is not None:
        view_point_in_foreground_mask = view_point_in_foreground_mask & per_view_shrinked_per_vert_visibility
    visible_vertex_mask = visible_vertex_mask & view_point_in_foreground_mask # debug # TODO
    
    if B==1:
        vert_normals = compute_vertex_normal(vertices,faces)  # [V,3]
        view_direction = camera_positions # [1,3]
        vert_view_cos_angles  = vert_normals  @ view_direction.t() # [V,1]
        vert_view_cos_angles  = vert_view_cos_angles.permute(1,0) # [1,V]
        vert_view_cos_angles = vert_view_cos_angles.squeeze(0) # [V]
        
        point_this_view_mask = visible_vertex_mask[0] & (vert_view_cos_angles > eps)  #  [V]
        final_visible_vertex_mask_V = torch.zeros(vertices.shape[0]).to(device).bool()
        final_visible_vertex_mask_V[point_this_view_mask] = True
        # point_this_view_mask = visible_vertex_mask[i] #& (point_view_ids == i) #  [V] # debug
        view_img = imgs_BHW4[0].float() # HW4
        # view_img = torch.flip(view_img,[1]) # flip upside down, only use this when using kaolin's camera for camera transform
        new_vertex_colors[point_this_view_mask] = view_img[
            vertices_pixel_coord[0,point_this_view_mask][:,0],
            vertices_pixel_coord[0,point_this_view_mask][:,1] ][...,:3] # [V,3]
 
    else:
        vert_normals = compute_vertex_normal(vertices,faces)  # [V,3]
        view_direction = camera_positions # [camera_num,3]
        vert_view_cos_angles  = vert_normals  @ view_direction.t() # [V,camera_num]
        vert_view_cos_angles  = vert_view_cos_angles.permute(1,0) # [Camera_num,V]
        # per_point_per_view_weight = torch.softmax(cos_angles,1) # no
        point_view_weight_VB = vert_view_cos_angles.permute(1,0) # [V, Camera_num]
        if view_weights_B is not None:
            point_view_weight_VB = point_view_weight_VB * torch.tensor(view_weights_B).float().to(device).unsqueeze(0)
        
        if mode == 'select':
            point_view_weight_VB[~visible_vertex_mask.permute(1,0)] -= 100
            # choose the best view for each point
            point_view_ids = torch.argmax(point_view_weight_VB, dim=1)
           
            final_visible_vertex_mask_V = torch.zeros(vertices.shape[0]).to(device).bool()
            for i in range(B):
                point_this_view_mask = visible_vertex_mask[i] & (point_view_ids == i) #  [V]
                final_visible_vertex_mask_V[point_this_view_mask] = True
                
                # point_this_view_mask = visible_vertex_mask[i] #& (point_view_ids == i) #  [V] # debug
                view_img = imgs_BHW4[i].float() # HW4
                # view_img = torch.flip(view_img,[1]) # flip upside down, only use this when using kaolin's camera for camera transform
                new_vertex_colors[point_this_view_mask] = view_img[
                    vertices_pixel_coord[i,point_this_view_mask][:,0],
                    vertices_pixel_coord[i,point_this_view_mask][:,1] ][...,:3] # [V,3]
                temp = new_vertex_colors[point_this_view_mask]
                
        
        elif mode == 'fuse':
            new_vertex_colors = new_vertex_colors*0.0
            point_view_weight_VB[~visible_vertex_mask.permute(1,0)] = 0
            point_view_weight_VB[point_view_weight_VB<eps] = 0
            point_view_weight_VB = point_view_weight_VB * point_view_weight_VB
            final_visible_vertex_mask_V = torch.zeros(vertices.shape[0]).to(device).bool()
            for i in range(B):
                view_img = imgs_BHW4[i].float() # HW4
                point_this_view_mask = point_view_weight_VB[:,i]>0 # [V]
                final_visible_vertex_mask_V[point_this_view_mask] = True
                this_view_color = view_img[
                    vertices_pixel_coord[i,point_this_view_mask][:,0],
                    vertices_pixel_coord[i,point_this_view_mask][:,1] ][...,:3] # [this_view_V,3]
                this_view_color_weight = point_view_weight_VB[point_this_view_mask,i].unsqueeze(-1) # [this_view_V,1] 
                new_vertex_colors[point_this_view_mask] += this_view_color * this_view_color_weight # [V,3]

            point_weight_V = point_view_weight_VB.sum(1)
            
            new_vertex_colors[point_weight_V>eps*eps] /= (point_weight_V[point_weight_V>eps*eps].view(-1,1))
    #         import kiui
    #         kiui.lo(new_vertex_colors)

    # print('final_visible_vertex_mask_V',final_visible_vertex_mask_V.sum())
    # import kiui
    # kiui.lo(new_vertex_colors)
    new_vertex_colors = new_vertex_colors.clip(0,1)
    

    debug = False
    if debug:
        new_vertex_colors[~final_visible_vertex_mask_V] = torch.tensor([1.0,0.0,0.0]).to(device)
    
    vis=False
    if vis:
        from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
        vis_actors_vtk(
            [
                get_colorful_pc_actor_vtk(vertices.detach().cpu().numpy(),
                                            point_colors=new_vertex_colors.detach().cpu().numpy(),opacity=1.0)
            ]
        )
    return new_vertex_colors,final_visible_vertex_mask_V
    
    
def get_view_visible_V_F(vertices,faces,camera_positions,renderer,eps = 0.05,return_other=False,crop_views=None):
    '''
    vertices: [V,3]
    faces: [F,3]
    camera_positions: [Camera_num,3]
    
    return:
    visible_face_mask: [Camera_num,F]
    visible_vertex_mask: [Camera_num,V]
    
    if return_other:
        per_pixel_face_ids: [Camera_num,H,W]
        vertices_NDC: [Camera_num,V,4]
    '''
    device = vertices.device
    camera_num = camera_positions.shape[0]
    # 1. render to see visible faces
   
    per_pixel_face_ids,vertices_NDC = renderer.get_per_pixel_face_ids_and_vertex_NDC(
                        vertices,faces,
                    camera_positions_np =camera_positions.detach().cpu().numpy(),res=2048,
                    crop_views=crop_views) # [Camera_num,H,W],[Camera_num,V,4]
  
        
    visible_face_mask_NF = torch.zeros(camera_num,len(faces)).to(device).bool()
    visible_vertex_mask_NV = torch.zeros(camera_num,len(vertices)).to(device).bool()

   

    for i in range(camera_num):
        visible_face_ids = torch.unique(per_pixel_face_ids[i])
        visible_face_mask_NF[i,visible_face_ids.long()] = True # [F]
    

    cos_angles=None
    # 2. further remove faces with bad normals
    faces_normals = calc_face_normals(vertices,faces,normalize=True) # [F,3]
    view_direction = camera_positions.float() # [Camera_num,3]
    view_direction = view_direction / view_direction.norm(dim=1, keepdim=True)


    cos_angles = faces_normals @ view_direction.t()  # [F,Camera_num]
    cos_angles = cos_angles.permute(1,0) # [Camera_num,F]
    

    visible_face_mask_NF[i,cos_angles[i] < eps] = False
    vis_face_normals = False
    if vis_face_normals:
        import kiui
        kiui.lo(faces_normals)
        kiui.lo(view_direction)
        kiui.lo(cos_angles)
        face_normal_img = faces_normals[per_pixel_face_ids[0].long()]
        face_normal_img = (face_normal_img+1)*0.5
        
        face_cos_angle_img = cos_angles[0][per_pixel_face_ids[0].long()].unsqueeze(-1).repeat(1,1,3)
        face_cos_angle_img = (face_cos_angle_img+1)*0.5
        cat = torch.cat([face_normal_img,face_cos_angle_img],dim=1)
        import kiui
        kiui.vis.plot_image(cat)


    # update vertex mask based on visible faces
    for i in range(camera_num):
        visible_vertex_ids = torch.unique(faces[visible_face_mask_NF[i]].long())
        visible_vertex_mask_NV[i,visible_vertex_ids.long()] = True
    

    # 3. further remove vertices considering NDC space (out of image)
    transformed_vertices = vertices_NDC
    pt_tensor = transformed_vertices[..., :2] # [Camera_num,V,2]

    
    # pt_tensor = cameras.transform_points(verts_coordinates)[..., :2] # NDC space points
    vertex_in_view = ~((pt_tensor.isnan()|(pt_tensor<-1)|(1<pt_tensor)).any(dim=-1))  # [Camera_num,V]
    visible_vertex_mask_NV = visible_vertex_mask_NV & vertex_in_view
    
    # 4. further update face visibility based on vertex visibility (only if all 3 vertices visible)
    for i in range(camera_num):
        visible_face_mask_NF[i] = visible_vertex_mask_NV[i][faces].min(-1)[0].bool()
    
    if return_other:
        return visible_face_mask_NF,visible_vertex_mask_NV,per_pixel_face_ids,vertices_NDC,cos_angles
    return visible_face_mask_NF,visible_vertex_mask_NV
  