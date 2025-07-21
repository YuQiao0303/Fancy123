import datetime
import logging
import os
import imageio
import numpy as np
import pytz
import torch
import torch.nn.functional as F
from PIL import Image
import pymeshlab
import trimesh
from tqdm import tqdm
from fancy123.mesh_deform.math_utils import normalize_vs, quat_to_mat_torch
from fancy123.mesh_deform.poisson_system import PoissonSystemClass
from fancy123.optimize.mesh_optimize import laplace_regularizer_const
from fancy123.render.general_renderer import GeneralRenderer
from src.utils.mesh_util import save_obj
from torchvision.transforms import transforms

def cleanup_mesh(vertices,faces,vertex_colors):
    """
    Applies a series of filters to the input mesh.
    Code borrowed from: https://github.com/KAIST-Visual-AI-Group/APAP/blob/06f3d728e7da6ea2bc039981b5ad614e25f0180b/src/utils/geometry_utils.py#L555

    For instance,
    - Duplicate vertex removal
    - Unreference vertex removal
    - Remove isolated pieces
    """
    device = vertices.device
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    vertex_colors = torch.cat([vertex_colors,torch.zeros(vertices.shape[0],1).to(device)],dim=1) # Pymeshlab requires 4-dim colors as RGBA
    vertex_colors = vertex_colors.detach().cpu().numpy()
    
    # create PyMeshLab MeshSet
    mesh = pymeshlab.Mesh(
        vertex_matrix=vertices.astype(np.float64),
        face_matrix=faces.astype(int),
        v_color_matrix = vertex_colors.astype(np.float64)
    )
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)
    for i in range(1):
        # remove duplicate vertices
        meshset.meshing_remove_duplicate_vertices()

        # remove unreferenced vertices
        meshset.meshing_remove_unreferenced_vertices()

        # remove isolated pieces
        meshset.meshing_remove_connected_component_by_diameter()
        
        # # added by Qiao Yu
        # meshset.meshing_remove_duplicate_faces()
        # meshset.meshing_remove_folded_faces()
        # meshset.meshing_remove_null_faces()
        # meshset.meshing_repair_non_manifold_edges()
        # meshset.meshing_repair_non_manifold_vertices()
        
    
    

    # extract the processed mesh
    mesh_new = meshset.current_mesh()
    vertices_proc = mesh_new.vertex_matrix().astype(np.float32)
    faces_proc = mesh_new.face_matrix()
    vertex_colors_proc = mesh_new.vertex_color_matrix().astype(np.float32)
    
    # to device
    vertices_proc = torch.from_numpy(vertices_proc).to(device).contiguous().float()
    faces_proc = torch.from_numpy(faces_proc).to(device).contiguous().long()
    vertex_colors_proc = torch.from_numpy(vertex_colors_proc).to(device)[...,:3].contiguous().float()

    return vertices_proc, faces_proc,vertex_colors_proc


def split_mesh_by_component(vertices,faces,vertex_colors,max_component_num=20,by_trimesh=False):
    device = vertices.device
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    
    if by_trimesh: # may not work with a few wrongly-oriented faces
        vertex_colors_np = torch.cat([vertex_colors,torch.ones(vertices_np.shape[0],1).to(device)],dim=1)*255 #  4-dim colors as RGBA
        vertex_colors_np = vertex_colors_np.detach().cpu().numpy().astype(np.uint8)
        mesh = trimesh.Trimesh(vertices=vertices_np,faces=faces_np,vertex_colors=vertex_colors_np)
        sub_meshes = mesh.split(only_watertight=False)
        if len(sub_meshes)>max_component_num:
            return None,len(sub_meshes),None
        vertices_list = []
        faces_list = []
        vertex_colors_list = []
        for i in range(len(sub_meshes)):
            vertices_list.append(torch.tensor(np.array(sub_meshes[i].vertices)).float().contiguous().to(device))
            faces_list.append(torch.tensor(np.array(sub_meshes[i].faces)).long().contiguous().to(device))
            vertex_colors_list.append(torch.tensor(np.array(sub_meshes[i].visual.vertex_colors)[...,:3]/255).float().contiguous().to(device))
    
    else:
        vertex_colors_np = torch.cat([vertex_colors,torch.ones(vertices_np.shape[0],1).to(device)],dim=1) #  4-dim colors as RGBA
        vertex_colors_np = vertex_colors_np.detach().cpu().numpy().astype(np.float64)
        pyml_mesh = pymeshlab.Mesh(vertex_matrix=vertices_np,face_matrix=faces_np,v_color_matrix=vertex_colors_np)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pyml_mesh, "cube_mesh")
        # split
        ms.generate_splitting_by_connected_components(delete_source_mesh=True)
        num_components = ms.mesh_number()
        if num_components > max_component_num:
            return None,num_components,None
        vertices_list = []
        faces_list = []
        vertex_colors_list = []
        for i in range(num_components):
            # ms.set_current_mesh(i)
            # mesh = ms.current_mesh()
            mesh = ms.mesh(i+1) # the original mesh has the id of 0 and has been deleted, so now the id starts from 1
            vertices = np.array(mesh.vertex_matrix())
            faces = np.array(mesh.face_matrix()) 
            vertex_colors = np.array(mesh.vertex_color_matrix())
            vertices=torch.tensor(vertices).float().to(device).contiguous()
            faces=torch.tensor(faces).to(device).long().contiguous()
            vertex_colors=torch.tensor(vertex_colors)[...,:3].float().to(device).contiguous()
            
       

            vertices_list.append(vertices)
            faces_list.append(faces)
            vertex_colors_list.append(vertex_colors)
    return vertices_list,faces_list,vertex_colors_list
    
    
def jacob_mesh_deformation(vertices,faces,vertex_colors,fixed_imgs,camera_position_np,crop=True,
                           train_J = True,train_quat = False,
                           lr=1e-2, epochs = 200,save_path=None,lap_weight = 1e5,renderer=None,save_gif=False):
    '''
    vertices: [V,3]
    faces: [F,3]
    train_J = True # whether to train source Jacobian field
    train_quat = False # whether to train quaternion field
    '''
    if save_gif:
        timestr = datetime.datetime.now(pytz.timezone('Etc/GMT-8')).strftime('%Y.%m.%d.%H.%M')

        filename_output = f"./{timestr}_3D_deform.mp4"
        writer = imageio.get_writer(filename_output, mode='I' ,fps=18) # duration=0.15
        margin = 16
    
    fixed_imgs = transforms.Resize((320,320))(fixed_imgs.permute(0,3,1,2)).permute(0,2,3,1)
    if crop:
        crop_views = np.arange(len(camera_position_np)) # all views are to be cropped
    else:
        crop_views = None
    logger = logging.getLogger('logger')
    vertices,faces,vertex_colors = cleanup_mesh(vertices,faces,vertex_colors)
    # save_obj(vertices.detach().cpu().numpy(),
    #         faces.detach().cpu().numpy(),
    #         vertex_colors.detach().cpu().numpy(),
    #         os.path.join(f'jacobtest_density/cleaned_up.obj'),
    #         flip_normal=False)
    vertices_list, faces_list,vertex_colors_list = split_mesh_by_component(vertices,faces,vertex_colors)
    # vertices_list0 = vertices_list[0]
    # vertices_list1 = vertices_list[1]
    # vertices_list2 = vertices_list[2]
    
    # import kiui
    # kiui.lo(vertices_list0)
    # kiui.lo(vertices_list1)
    # kiui.lo(vertices_list2)
    
    if vertices_list is None:
        component_num = faces_list
        print('Error: Too many mesh components to deform!!!',component_num)
        return None,None,None
    else:
        print('Number of mesh components: ',len(vertices_list))
        logger.info(f'Number of mesh components: {len(vertices_list)}')
    
    device = vertices.device
    
    B, H, W, C = fixed_imgs.size()
  
    fixed_imgs[...,:3] = fixed_imgs[..., :3] * fixed_imgs[..., 3:] + (1 - fixed_imgs[..., 3:]) # background to white


    ori_vertices = []
    optim_vars = []
    
    poissons = []
    quat_fields = []
    component_centers = []
    # vis=True
    # if vis:
    #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk,get_pc_actor_vtk
    #     vis_actors_vtk([
    #         get_colorful_pc_actor_vtk(vertices_list[0].detach().cpu().numpy(), vertex_colors_list[0].detach().cpu().numpy(), opacity=0.5,point_size=1),
    #         get_pc_actor_vtk(vertices_list[1].detach().cpu().numpy(),color = (0,1,0),opacity=1),
    #         get_pc_actor_vtk(vertices_list[2].detach().cpu().numpy(),color = (0,0,1),opacity=1),
    #     ])
            
    for i in range(len(vertices_list)):
        
        # vis=True
        # if vis:
        #     if i==0:
        #         continue
        #     print(i)
        #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
        #     vis_actors_vtk([
        #         get_colorful_pc_actor_vtk(vertices_list[i].detach().cpu().numpy(), vertex_colors_list[i].detach().cpu().numpy(), opacity=1),
        #     ])
            
        # get component centers: the vertices calculated from jacobian fields are recentered to the origin, so we need to know the original center
        # center = (torch.max(vertices_list[i], axis=0)[0] + torch.min(vertices_list[i], axis=0)[0])/2
        center = torch.mean(vertices_list[i], axis=0)
        component_centers.append(center.detach())
        ori_vertices.append(vertices_list[i])
        
        # Initialize Poisson System
        
        poissons.append(PoissonSystemClass(
            vertices_list[i], faces_list[i],
            device,
            train_J=train_J,
            anchor_inds=None,
        ))
        if train_J:
            assert poissons[i].J.requires_grad, "Jacobian field must be trainable"
            optim_vars.append(poissons[i].J)

        # Initialize auxiliary learnable parameters
        quat_fields.append( torch.zeros(
            (faces_list[i].shape[0], 4),
            dtype=torch.float64,
            device=device,
        )) 
        quat_fields[i][:, 0] = 1.0  # identity rotation
        if train_quat:
            quat_fields[i].requires_grad_(True)
            
    ori_vertices = torch.cat(ori_vertices, axis=0)
        
    
    # =================================================================================
    # Optimization loop begins

    best_vertices = ori_vertices.detach().clone()
    best_loss = 100.0
    best_rendered_img = None
    
    assert len(optim_vars) > 0, "No trainable variables found"

    optim = torch.optim.Adam(optim_vars, lr=lr)
    for epoch in tqdm(range(epochs), desc="deform 3D"):
        
        # loop over all sub components
        all_vertices = []
        vertex_counts = []
        all_faces = []
        all_vertex_colors = []
        offset = 0
        # print('len(vertices_list)',len(vertices_list))
        for i in range(len(vertices_list)):
            curr_v, curr_f = poissons[i].get_current_mesh(
                constraints = None, #anchor_pos,
                trans_mats=quat_to_mat_torch(normalize_vs(quat_fields[i])),
            )
            curr_v += component_centers[i]
            curr_color = vertex_colors_list[i]
            # vis=True
            # if vis:
            #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
            #     vis_actors_vtk([
            #         get_colorful_pc_actor_vtk(curr_v.detach().cpu().numpy(), curr_color.detach().cpu().numpy(), opacity=1),
            #     ])
            
            all_vertices.append(curr_v)
            all_vertex_colors.append(curr_color)
            vertex_counts.append(curr_v.shape[0])
            
            adjusted_faces = curr_f + offset
            all_faces.append(adjusted_faces)
            offset += curr_v.shape[0]

        vertices = torch.cat(all_vertices, dim=0).contiguous()
        faces = torch.cat(all_faces, dim=0).contiguous()
        vertex_colors = torch.cat(all_vertex_colors, dim=0).contiguous()

        # vis=False
        # if vis:
        #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
        #     vis_actors_vtk([
        #         get_colorful_pc_actor_vtk(vertices.detach().cpu().numpy(), vertex_colors.detach().cpu().numpy(), opacity=1),
        #     ])
        
        ## get loss
        rendered_imgs = renderer.render(vertices,vertex_colors.detach()*2-1,
                                        faces.detach(),camera_positions_np=camera_position_np,rotate_normal=False,
                                        crop_views=crop_views,background_color=[1,1,1,0])
        rendered_imgs_white_bg = rendered_imgs
        vis=False
        if vis:
            kiui.vis.plot_image(rendered_imgs_white_bg[0])
        # rendered_imgs_white_bg = rendered_imgs.clone()
        # rendered_imgs_white_bg[...,:3] = rendered_imgs[..., :3] * rendered_imgs[..., 3:] + (1 - rendered_imgs[..., 3:]) # background to white
        
        mse_loss = F.mse_loss(rendered_imgs_white_bg[...,:3], fixed_imgs[...,:3])
        mask_loss = F.mse_loss(rendered_imgs_white_bg[...,3], fixed_imgs[...,3])
        loss = mask_loss + 0.1 * mse_loss
        
        # laplas smooth
        
        lap_smooth_Loss = laplace_regularizer_const(vertices,faces).mean()
        loss = loss + lap_weight*lap_smooth_Loss
        
        # # small offset loss
        # small_offset_loss = (vertices-ori_vertices).abs().mean()
        # loss = loss + 1e-2 * small_offset_loss
        
        
        getting_better=False
        if loss < best_loss:
            best_loss = loss.detach().clone()
            best_vertices = vertices.detach().clone()
            best_rendered_img = rendered_imgs_white_bg
            getting_better=True
        ## visualize
        if save_path is not None:
            if epoch == 0:
                initial_rendered_imgs_white_bg = rendered_imgs_white_bg.clone()
            if epoch == epochs-1:
                import kiui
                cat = torch.cat([fixed_imgs[0],best_rendered_img[0],
                                (fixed_imgs[0]+best_rendered_img[0])/2,
                                (fixed_imgs[0]-best_rendered_img[0]).abs(),
                                ],dim=1)[...,:3]
                cat2 = torch.cat([fixed_imgs[0],initial_rendered_imgs_white_bg[0],
                                (fixed_imgs[0]+initial_rendered_imgs_white_bg[0])/2,
                                (fixed_imgs[0]-initial_rendered_imgs_white_bg[0]).abs(),
                                ],dim=1)[...,:3]
                cat = torch.cat([cat2,cat],dim=0)
                # kiui.vis.plot_image(cat)
                Image.fromarray((cat*255.0).detach().cpu().numpy().astype(np.uint8)).save(save_path)

        ## print
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, mse_loss: {mse_loss.item()}")
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if save_gif and (epoch<30 or (epoch<60 and epoch%5==0) or epoch%10==0):
            from skimage import img_as_ubyte
            if not getting_better:
                temp_img = best_rendered_img
            else:
                temp_img = rendered_imgs_white_bg
            merge_img =(fixed_imgs + temp_img)/2
            diff_img = (fixed_imgs-temp_img).abs()
            margin_img = torch.full((H, margin, C), 1.0).float().to(device)
           
            cat_img = torch.cat([temp_img[0],margin_img,merge_img[0],margin_img,diff_img[0]],dim=1)
          

            cat_img = cat_img.clip(0,1).detach().squeeze().cpu().numpy()
            
            cat_img = img_as_ubyte(cat_img[...,:3])
            writer.append_data(cat_img)
    if save_gif:
        merge_img =(fixed_imgs+best_rendered_img)/2
        diff_img = (fixed_imgs-best_rendered_img).abs()
        margin_img = torch.full((H, margin, C), 1.0).float().to(device)
        cat_img = torch.cat([best_rendered_img[0],margin_img,merge_img[0],margin_img,diff_img[0]],dim=1)
        
     
        cat_img = cat_img.clip(0,1).detach().squeeze().cpu().numpy()
        cat_img = img_as_ubyte(cat_img[...,:3])
        writer.append_data(cat_img)
        writer.close()
            
    return best_vertices,faces,vertex_colors    
        
        
def no_jacob_mesh_deformation(vertices,faces,vertex_colors,fixed_imgs,camera_position_np,crop=True,
                           lr=1e-3, epochs = 800,save_path=None,lap_weight = 1e5,renderer=None):
    '''
    vertices: [V,3]
    faces: [F,3]
    train_J = True # whether to train source Jacobian field
    train_quat = False # whether to train quaternion field
    '''
    fixed_imgs = transforms.Resize((320,320))(fixed_imgs.permute(0,3,1,2)).permute(0,2,3,1)
    if crop:
        crop_views = np.arange(len(camera_position_np)) # all views are to be cropped
    else:
        crop_views = None
    logger = logging.getLogger('logger')
    vertices,faces,vertex_colors = cleanup_mesh(vertices,faces,vertex_colors)
   
    device = vertices.device
    
    B, C, H, W = fixed_imgs.size()

    fixed_imgs[...,:3] = fixed_imgs[..., :3] * fixed_imgs[..., 3:] + (1 - fixed_imgs[..., 3:]) # background to white




        
    # vis=True
    # if vis:
    #     if i==0:
    #         continue
    #     print(i)
    #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
    #     vis_actors_vtk([
    #         get_colorful_pc_actor_vtk(vertices_list[i].detach().cpu().numpy(), vertex_colors_list[i].detach().cpu().numpy(), opacity=1),
    #     ])
        
    # get component centers: the vertices calculated from jacobian fields are recentered to the origin, so we need to know the original center
    # center = (torch.max(vertices_list[i], axis=0)[0] + torch.min(vertices_list[i], axis=0)[0])/2

    ori_vertices = vertices.clone()
    
    # =================================================================================
    # Optimization loop begins
    
    best_vertices = ori_vertices.detach().clone()
    best_loss = 100.0
    best_rendered_img = None
    

    offsets = torch.zeros_like(vertices).to(device).requires_grad_()
    optim = torch.optim.Adam([offsets], lr=lr)
    for epoch in tqdm(range(epochs), desc="deform 3D by optimizint vertex positions"):
   

        # vis=False
        # if vis:
        #     from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
        #     vis_actors_vtk([
        #         get_colorful_pc_actor_vtk(vertices.detach().cpu().numpy(), vertex_colors.detach().cpu().numpy(), opacity=1),
        #     ])
        
        ## get loss
        rendered_imgs = renderer.render(vertices+ offsets,vertex_colors.detach()*2-1,
                                        faces.detach(),camera_positions_np=camera_position_np,rotate_normal=False,
                                        crop_views=crop_views,background_color=[1,1,1,0])
        rendered_imgs_white_bg = rendered_imgs
        vis=False
        if vis:
            kiui.vis.plot_image(rendered_imgs_white_bg[0])
        # rendered_imgs_white_bg = rendered_imgs.clone()
        # rendered_imgs_white_bg[...,:3] = rendered_imgs[..., :3] * rendered_imgs[..., 3:] + (1 - rendered_imgs[..., 3:]) # background to white
        
        mse_loss = F.mse_loss(rendered_imgs_white_bg[...,:3], fixed_imgs[...,:3])
        mask_loss = F.mse_loss(rendered_imgs_white_bg[...,3], fixed_imgs[...,3])
        loss = mask_loss + 0.1 * mse_loss
        
        # laplas smooth
        
        lap_smooth_Loss = laplace_regularizer_const((vertices+ offsets),faces).mean()
        loss = loss + lap_weight*lap_smooth_Loss
        
        # # small offset loss
        # small_offset_loss = (vertices-ori_vertices).abs().mean()
        # loss = loss + 1e-2 * small_offset_loss
        
        
        
        if loss < best_loss:
            best_loss = loss.detach().clone()
            best_vertices = (vertices+ offsets).detach().clone()
            best_rendered_img = rendered_imgs_white_bg
        ## visualize
        if save_path is not None:
            if epoch == 0:
                initial_rendered_imgs_white_bg = rendered_imgs_white_bg.clone()
            if epoch == epochs-1:
                import kiui
                cat = torch.cat([fixed_imgs[0],best_rendered_img[0],
                                (fixed_imgs[0]+best_rendered_img[0])/2,
                                (fixed_imgs[0]-best_rendered_img[0]).abs(),
                                ],dim=1)[...,:3]
                cat2 = torch.cat([fixed_imgs[0],initial_rendered_imgs_white_bg[0],
                                (fixed_imgs[0]+initial_rendered_imgs_white_bg[0])/2,
                                (fixed_imgs[0]-initial_rendered_imgs_white_bg[0]).abs(),
                                ],dim=1)[...,:3]
                cat = torch.cat([cat2,cat],dim=0)
                # kiui.vis.plot_image(cat)
                Image.fromarray((cat*255.0).detach().cpu().numpy().astype(np.uint8)).save(save_path)

        ## print
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, mse_loss: {mse_loss.item()}")
            
        optim.zero_grad()
        loss.backward()
        optim.step()
    return best_vertices,faces,vertex_colors    
        