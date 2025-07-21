# ====================================================================================================================
# Given a 3D mesh of an object and an image of the object from known camera parameters, the code tries to get the mesh parameters
# ====================================================================================================================

import logging
import sys

from fancy123.utils.temp_utils import get_view_visible_V_F
from fancy123.utils.utils2d import scharr_edge_RGB_torch, sobel_edge_torch
from src.models.geometry.render.neural_render import compute_vertex_normal
sys.path.append('...')
sys.path.append('..')
sys.path.append('.')
import os
import torch
import torch.optim as optim
from kaolin.render.camera import Camera,perspective_camera, generate_perspective_projection,generate_transformation_matrix
import numpy as np
import kaolin as kal
import nvdiffrast
import nvdiffrast.torch as dr
import imageio
from PIL import Image #
import kiui
from src.utils.infer_util import remove_background, resize_foreground, resize_foreground_torch,save_video
# import rembg
from torchvision import transforms
from skimage import img_as_ubyte
from kiui.lpips import LPIPS
# from fancy123.optimize.my_renderer import NeuralRender
from unique3d.mesh_reconstruction.opt import MeshOptimizer
import torch.nn.functional as tfunc
from unique3d.mesh_reconstruction.remesh import calc_face_normals, calc_vertex_normals

from tqdm import tqdm
from typing import List
import pymeshlab


from fancy123.render.general_renderer import GeneralRenderer
# from unique3d.scripts.project_mesh import get_cameras_list_azim_ele, multiview_color_projection, get_cameras_list
from unique3d.scripts.utils import to_py3d_mesh, from_py3d_mesh, init_target
from src.utils.camera_util import elevation_azimuth_radius_to_xyz, center_looking_at_camera_pose, xyz_to_elevation_azimuth_radius

# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
def laplace_regularizer_const(mesh_verts, mesh_faces):
    '''
    # https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dmtet_tutorial.ipynb
    # https://mgarland.org/class/geom04/material/smoothing.pdf
    This function assumes batch size is 1.
    :param mesh_verts: [N,3]
    :param mesh_faces: [N,3]
    :return:
    '''
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

# class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces):
        # Input:
        #  faces: B x F x 3
        from .laplacian import Laplacian
        # V x V
        self.laplacian = Laplacian(faces)
        self.Lx = None

    def __call__(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = torch.norm(Lx, p=2, dim=1).mean()
        return loss





def calculate_first_derivative(image):
    '''
    image: NHWC
    '''
    # R
    grad_x = torch.gradient(image[..., 0], dim=2)
    grad_y = torch.gradient(image[..., 0], dim=1)
    grad_r = torch.sqrt(grad_x[0]**2 + grad_y[0]**2)

    # G
    grad_x = torch.gradient(image[..., 1], dim=2)
    grad_y = torch.gradient(image[..., 1], dim=1)
    grad_g = torch.sqrt(grad_x[0]**2 + grad_y[0]**2)

    # B
    grad_x = torch.gradient(image[..., 2], dim=2)
    grad_y = torch.gradient(image[..., 2], dim=1)
    grad_b = torch.sqrt(grad_x[0]**2 + grad_y[0]**2)

    # combine
    derivative = torch.stack([grad_r, grad_g, grad_b], dim=-1)
    return derivative





def run_mesh_refine(vertices, faces, pils: List[Image.Image],camera_positions_np, 
                    fixed_vertex_ids=None,per_pixel_fixed_mask=None,
                    res=2048,
                    view_weights=None,
                    steps=100, start_edge_len=0.02, end_edge_len=0.02, decay=0.99, 
                    update_normal_interval=20, update_warmup=5, 
                    lr=0.3,
                    return_mesh=True, process_inputs=True, process_outputs=True,debug=False,remesh=True,
                    expand=False,background_color = None,use_debug_img = False,renderer=None):
    '''
    modified from Unique3D/mesh_reconstruction/refine.py: use perspective camera instead of orthographic ones
    '''
    logger = logging.getLogger('logger')
    device = vertices.device
    if process_inputs:
        vertices = vertices * 2 / 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]
    
    poission_steps = []

    # # assert len(pils) == 4
    # mv,proj = make_star_cameras_orthographic(4, 1)   
    # renderer = NormalsRenderer(mv,proj,list(pils[0].size))
    # # cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
    # # renderer = Pytorch3DNormalsRenderer(cameras, list(pils[0].size), device="cuda")
    
         


    pils = [pil.resize((res,res)) for pil in pils]

    target_images = init_target(pils, new_bkgd=(0., 0., 0.)) # 4s
    # import kiui
    # kiui.lo(target_images)
    # for i in range(len(target_images)):
    #     Image.fromarray((target_images[i].detach().cpu().numpy()*255).astype(np.uint8)).save(os.path.join(f'target_images_{i}.png'))
    #     Image.fromarray((target_images[i].detach().cpu().numpy()*255).astype(np.uint8)[...,:3]).save(f'target_images_{i}_RGB.png')

    if debug:
        ## Prepare for saving gif
        filename_output = "./optimize_mesh_first_gradient.gif"
        normal_file_output = './optimize_mesh_normal.gif'
        first_gradiant_writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
        normal_writer = imageio.get_writer(normal_file_output, mode='I', duration=0.3)

    opt = MeshOptimizer(vertices,faces, ramp=5,
                        lr=lr,
                        edge_len_lims=(end_edge_len, start_edge_len),
                        local_edgelen=False, laplacian_weight=0.02)

    vertices = opt.vertices
    alpha_init = None
    if target_images.shape[-1] == 4:
        mask = target_images[..., -1] < 0.5
        use_alpha = True
    else:
        use_alpha = False

    # if fixed_vertex_ids is not None:
    #     # old_vertices = vertices.clone()
    #     remesh = False
    #     # vertices[fixed_vertex_ids].requires_grad = False
    #     # opt._vertices.requires_grad_(False)
    if per_pixel_fixed_mask is not None: # B,H,W
        remesh = False #
        pixel_fixed_mask_resized = torch.nn.functional.interpolate(
            per_pixel_fixed_mask.unsqueeze(0).float(),size=(res,res),mode='nearest').squeeze(0).bool()
        per_pixel_fixed_mask = pixel_fixed_mask_resized
        # mask[pixel_fixed_mask_resized] = False
        
        rgba = renderer.render(vertices,vertices,faces,res=res,camera_positions_np=camera_positions_np)
        alpha = rgba[...,-1].bool()
        new_mask = alpha.clone()
        new_mask[~pixel_fixed_mask_resized] = mask[~pixel_fixed_mask_resized]
    
    
    # if view_weights is None:
    #     view_weights = torch.ones(len(camera_positions_np), device=device)
    if view_weights is not None:
        view_weights = torch.tensor(view_weights, device=device)
        view_weights_N11 = view_weights.unsqueeze(-1).unsqueeze(-1) # [N] -> [N,1,1]
        view_weights_NHW = view_weights_N11.repeat(1,res,res) # [N,1,1] -> [N,H,W]
    # for i in tqdm(range(steps)):
    for i in tqdm(range(steps), desc='Geo_Optimization', total=steps):
        opt.zero_grad()
        opt._lr *= decay
        # face_normals = calc_face_normals(vertices,faces,normalize=True)
        # normals = calc_vertex_normals(vertices,faces)
        

        normals = compute_vertex_normal(vertices, faces)
        # lengths = torch.norm(normals, dim=1)
        # kiui.lo(lengths)
        images = renderer.render(vertices,normals,faces,res=res,camera_positions_np=camera_positions_np,background_color=background_color) # N,H,W,4
        # kiui.lo(images)
        # if i==0:
        #     for j in range(len(images)):
        #         Image.fromarray((images[j].detach().cpu().numpy()*255).astype(np.uint8)).save(f'images{j}_0.png')
        #         Image.fromarray((images[j].detach().cpu().numpy()*255).astype(np.uint8)[...,:3]).save(f'images{j}_0_RGB.png')
        # if i==steps-1:
        #     for j in range(len(images)):
        #         Image.fromarray((images[j].detach().cpu().numpy()*255).astype(np.uint8)).save(f'images{j}_final.png')
        #         Image.fromarray((images[j].detach().cpu().numpy()*255).astype(np.uint8)[...,:3]).save(f'images{j}_1_final.png')

        
        # images = renderer.render(vertices,face_normals,faces,res=res,camera_positions_np=camera_positions_np) # N,H,W,4
        if alpha_init is None:
            alpha_init = images.detach()
        if False:
            pass
        # if use_debug_img:
        #     if i < update_warmup or i % update_normal_interval == 0:
        #         with torch.no_grad():
        #             py3d_mesh = to_py3d_mesh(vertices, faces, normals)
        #             # cameras = get_cameras_list(azim_list = [0, 90, 180, 270], device=vertices.device, focal=1.)
        #             elevations, azim_list, radius = xyz_to_elevation_azimuth_radius(camera_positions=camera_positions_np)
        #             cameras_list = get_cameras_list_azim_ele(azim_list=azim_list,  elevations = elevations,
        #                                     cam_type="fov",  dist=radius, device=device, )
        #             _, _, target_normal = from_py3d_mesh(multiview_color_projection(py3d_mesh, pils, cameras_list=cameras_list, 
        #                                                                             confidence_threshold=0.1, complete_unseen=False, 
        #                                                                             below_confidence_strategy='original', reweight_with_cosangle='linear'))
        #             target_normal = target_normal * 2 - 1
        #             target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
        #             debug_images = renderer.render(vertices,target_normal,faces,res=res,camera_positions_np=camera_positions_np,
        #                                            rotate_normal=False)
                
        else:
            debug_images = target_images
        # if i == 0:
        #     for j in range(len(debug_images)):
        #         Image.fromarray((debug_images[j].detach().cpu().numpy()*255).astype(np.uint8)).save(f'debug_images_{j}_0.png')
        #         Image.fromarray((debug_images[j].detach().cpu().numpy()*255).astype(np.uint8)[...,:3]).save(f'debug_images_{j}_0_RGB.png')

        # if i == steps-1:
        #     for j in range(len(debug_images)):
        #         Image.fromarray((debug_images[j].detach().cpu().numpy()*255).astype(np.uint8)).save(f'debug_images_{j}_final.png')
        #         Image.fromarray((debug_images[j].detach().cpu().numpy()*255).astype(np.uint8)[...,:3]).save(f'debug_images_{j}_final_RGB.png')

        
        d_mask = images[..., -1] > 0.5 # N,H,W
        if per_pixel_fixed_mask is not None:
            d_mask[per_pixel_fixed_mask] = False
        # import kiui
        # kiui.vis.plot_image(per_pixel_fixed_mask[0])
        # kiui.vis.plot_image(d_mask[0])
        # kiui.vis.plot_image(mask[0])
        # if i==steps-1:
        #     import kiui
            
        #     kiui.vis.plot_image(images[..., :3][0])
        #     kiui.vis.plot_image(debug_images[..., :3][0])
        #     kiui.vis.plot_image(target_images[..., :3][0])
        
        # NHW4,   NHW

        # temp1 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2)
        # temp2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2)
        # import kiui
        # kiui.lo(temp1)
        # kiui.lo(temp2)
        vis = False
        # if i == steps-1 or i==0:
        #     vis = True
        if vis:
            import kiui
            
            # kiui.vis.plot_image(d_mask[0])
            # kiui.vis.plot_image(mask[0])
            merge_normal =(debug_images + images)/2
            cat_normal = torch.cat([images,target_images],dim=1)
            
            kiui.lo(cat_normal)
            cat_normal = torch.cat([cat_normal,merge_normal],dim=1)
            kiui.lo(cat_normal)
            # now cat images of different cameras
            temp = torch.unbind(cat_normal, dim=0)
        
            cat_normal = torch.cat(temp, dim=1)
            kiui.lo(cat_normal)
            cat_normal = cat_normal.clip(0,1).detach().squeeze().cpu().numpy()
            import kiui
            # kiui.vis.plot_image(cat_normal[...,:3])
        


        if view_weights is None:
            loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean() # the core is to use debug images instead of target images?
            # loss_debug_l2 = (images[..., :3][d_mask] - target_images[..., :3][d_mask]).pow(2).mean() # what would happen if we use this instead like everyone else?
            # loss_alpha_target_mask_l2 = (images[..., -1] - target_images[..., -1]).pow(2).mean()# if not use mask for alpha
            if use_alpha:
                loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
               

        
        else:
            loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2) # [M,3]
            loss_debug_l2 = loss_debug_l2 * view_weights_NHW[d_mask].unsqueeze(-1) # [M,3] * [M,1] -> [M,3]

            loss_debug_l2 = loss_debug_l2.sum() / view_weights_NHW[d_mask].unsqueeze(-1).repeat(1,3).sum()
            if use_alpha:
                loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2) # [M]
                loss_alpha_target_mask_l2 = loss_alpha_target_mask_l2 * view_weights_NHW[mask]  # [M] * [M] -> [M]
                loss_alpha_target_mask_l2 = loss_alpha_target_mask_l2.sum() / view_weights_NHW[mask].sum()
        if use_alpha:
            loss = loss_debug_l2 + loss_alpha_target_mask_l2 
            # loss = loss_alpha_target_mask_l2 #+  
        else:
            loss = loss_debug_l2
            
        
        # expand
        expand_weight = 0.1
        if end_edge_len == 0.005:
            expand_weight = 0.1
        if expand:
            # visible_face_mask,visible_vertex_mask = get_view_visible_V_F(vertices,faces,torch.tensor(camera_positions_np).to(vertices.device),renderer,eps=0)
            # visible_vertex_mask = visible_vertex_mask.max(0)[0].bool()
            # loss_expand = 0.5 * ((vertices+normals).detach() - vertices)[~visible_vertex_mask].pow(2).mean() # expand invisible vertices
            loss_expand = 0.5 * ((vertices+normals).detach() - vertices).pow(2).mean() # expand invisible vertices
            loss +=   expand_weight * loss_expand
        
        # # out of box
        # loss_oob = (vertices.abs() > 0.99).float().mean() * 10
        # loss = loss + loss_oob
        
        # laplas smooth
        lap_weight = 0
        if end_edge_len == 0.005:
            lap_weight = 1e5
        lap_smooth_Loss = laplace_regularizer_const(vertices,faces).mean()
        loss = loss + lap_weight*lap_smooth_Loss
        
        # # normal gradient
        # derivative_images = calculate_first_derivative(images[..., :3]) # N,H,W,3
        
        # derivative_debug_images = calculate_first_derivative(debug_images[..., :3])# N,H,W,3

        # loss_derivative = (derivative_images - derivative_debug_images).pow(2).mean()
        # loss = loss + 100* loss_derivative
        
        # vis = False
        # if vis:
        #     merge_normal =(derivative_debug_images + derivative_images)/2
        #     cat_normal = torch.cat([derivative_images,derivative_debug_images],dim=1)
        #     import kiui
        #     kiui.lo(cat_normal)
        #     cat_normal = torch.cat([cat_normal,merge_normal],dim=1)
        #     kiui.lo(cat_normal)
        #     # now cat images of different cameras
        #     temp = torch.unbind(cat_normal, dim=0)
        #     kiui.lo(temp)
        #     cat_normal = torch.cat(temp, dim=1)
        #     kiui.lo(cat_normal)
        #     cat_normal = cat_normal.clip(0,1).detach().squeeze().cpu().numpy()
        #     import kiui
        #     kiui.vis.plot_image(cat_normal)
        
        
        # ## edge
        # edge_debug_img_BHWC = scharr_edge_RGB_torch(debug_images.permute(0,3,1,2)).permute(0,2,3,1)
        # edge_img_BHWC = scharr_edge_RGB_torch(images.permute(0,3,1,2)).permute(0,2,3,1)
        # vis = False
        # if i == steps-1:
        #     vis=True
        # if vis:
        #     for j in range(edge_debug_img_BHWC.shape[0]):
        #         temp = torch.cat([edge_debug_img_BHWC[j],edge_img_BHWC[j]],dim=1) # edge_target_img_BHWC[i]
        #         import kiui
        #         kiui.vis.plot_image(temp)
        # loss_edge = (edge_img_BHWC - edge_debug_img_BHWC).pow(2).mean()
        # loss = loss +  100*loss_edge
        
        if logger is not None and i % 10 == 0:
            
            logger.info(f'Iteration {i}: loss = {loss.item():.6f}, \
                        loss_debug_l2 = {loss_debug_l2.item():.6f}, \
                        loss_alpha_target_mask_l2 = {loss_alpha_target_mask_l2.item():.6f}, \
                        loss_expand = {loss_expand.item():.6f}, lap_smooth_Loss = {lap_smooth_Loss.item():.6f}')
        if debug and i in [0,1,3,7,10,20,30,40,50,70,90,100] :
            # print(f'Iteration {i}:')

            
            merge_normal =(target_images + images)/2
            cat_normal = torch.cat([images,target_images],dim=1)
            cat_normal = torch.cat([cat_normal,merge_normal],dim=1)
            # now cat images of different cameras
            cat_normal = torch.cat(torch.unbind(cat_normal, dim=0), dim=1)
            cat_normal = cat_normal.clip(0,1).detach().squeeze().cpu().numpy()
            cat_normal = img_as_ubyte(cat_normal)[...,:3]
            normal_writer.append_data(cat_normal)
            
            # merge_normal =(derivative_debug_images + derivative_images)/2
            # cat_normal = torch.cat([derivative_images,derivative_debug_images],dim=1)
            # cat_normal = torch.cat([cat_normal,merge_normal],dim=1)
            # # now cat images of different cameras
            # cat_normal = torch.cat(torch.unbind(cat_normal, dim=0), dim=1)
            # cat_normal = cat_normal.clip(0,1).detach().squeeze().cpu().numpy()
            # cat_normal = img_as_ubyte(cat_normal)
            # first_gradiant_writer.append_data(cat_normal)
        # tqdm.set_postfix({'loss': loss.item()})
        
        loss.backward()
        opt.step()

        if remesh:
            vertices,faces = opt.remesh(poisson=(i in poission_steps)) # what happens if we comment this
       
    
    vertices, faces = vertices.detach(), faces.detach()
    
    if debug:
        normal_writer.close()
        first_gradiant_writer.close()
    
    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces


def taubin_smooth(vertices,faces,selected_face_mask=None,iterations=10):
    '''
    vertices: [V,3]
    faces: [F,3]
    selected_face_mask: [F]
    '''
    device = vertices.device
    if selected_face_mask is None:
        selected_face_mask = torch.ones(len(faces), dtype=torch.bool).to(device)
    pyml_mesh = pymeshlab.Mesh(
    vertex_matrix=vertices.cpu().float().numpy().astype(np.float64),
    face_matrix=faces.cpu().long().numpy().astype(np.int32),
    f_scalar_array= selected_face_mask.cpu().numpy().astype(np.int32),
        )
    
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")
    ms.compute_selection_by_condition_per_face(condselect="fq==1")  
    # https://pymeshlab.readthedocs.io/en/latest/filter_list.html#compute_selection_by_condition_per_face
    ms.apply_filter('apply_coord_taubin_smoothing',stepsmoothnum =iterations,selected=True) 

    mesh = ms.current_mesh()
    
    vertices = np.array(mesh.vertex_matrix())
    vertices=torch.tensor(vertices).float().to(device).contiguous()
    
    return vertices