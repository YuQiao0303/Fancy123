import datetime
from fancy123.optimize.mesh_deformation import optimize_mesh_deformation
from fancy123.utils.logger_util import get_logger
import os
import traceback
from einops import rearrange
from lightning_fabric import seed_everything
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
# from idea1_repaint_refine import coarse_stage_opt, fine_stage_opt

from src.run.mesh_gen import erode_alpha_torch_NHW4
from src.utils.camera_util import elevation_azimuth_radius_to_xyz
from src.utils.infer_util import remove_background
from src.utils.mesh_util import loadobj, loadobj_color, save_obj
import torch.nn.functional as F
from rembg import new_session, remove
from unique3d.app.utils import make_image_grid, rgba_to_rgb, split_image
from unique3d.mesh_reconstruction.opt import MeshOptimizer
# from unique3d.scripts.project_mesh import get_cameras_list_azim_ele, multiview_color_projection
from unique3d.scripts.refine_lr_to_sr import refine_lr_with_sd, run_sr_fast
from kiui.lpips import LPIPS

from fancy123.mesh_deform.jacobian_deform import cleanup_mesh, jacob_mesh_deformation,no_jacob_mesh_deformation

from fancy123.optimize.deform_2d import opt_mv_images_all
from fancy123.optimize.img_registration import apply_deformation, optimize_deformation, visualize_results
from fancy123.optimize.mesh_optimize import calculate_first_derivative, run_mesh_refine, taubin_smooth
from fancy123.render.general_renderer import GeneralRenderer
from fancy123.steps import deform_2d, fit_front_view_lgm, opt_geo, fit_input_view_instantmesh, opt_mv_cameras, project_front_view_colors, sr_mv_imgs,deform_input_img
from fancy123.unproject.NBF import NBF_prepare_per_kernel_per_view_mv_img_validation, NBF_prepare_per_kernel_per_view_shrinked_per_vert_visibility
from fancy123.utils.mesh_utils import simplify_mesh
from fancy123.utils.normal_utils import divide_mv_img_zero123pp, gen_mv_normal_zero123pp
from fancy123.utils.temp_utils import apply_view_color2mesh, complete_vert_colors_by_neighbors, get_fixed_area, load_test_data_instantmesh
from fancy123.utils.utils2d import scharr_edge_RGB_torch, sobel_edge_torch

    



def refine_one_sample(name,
                        vertices,faces,vertex_colors,
                        input_image_HW4, mv_imgs_BHW4, mv_normals_BHW4,
                        camera_positions_B3,renderer,
                        logger,cfg,
                        geo_refine=True,appearance_refine = True, fidelity_refine = True,
                        lgm = False):
    output_root = cfg.output_root
    # output_root = 'outputs_hhh'
    use_alpha = cfg.use_alpha # True
    refine_sr_by_sd = cfg.refine_sr_by_sd # False
    fast_sr = cfg.fast_sr # True
    crop_input_view = cfg.crop_input_view # True
    deform_3D_method = cfg.deform_3D_method # Jacobian
    lap_weight = cfg.lap_weight
    input_all_0_elevation = cfg.input_all_0_elevation
    skip_3D_deform_but_still_unproject = cfg.skip_3D_deform_but_still_unproject
    opt_geo_coarse2fine = cfg.opt_geo_coarse2fine
    use_vertex_wise_NBF = cfg.use_vertex_wise_NBF
    
    
    dist = camera_positions_B3[0].norm().detach().item()
    
    only_refine_fidelity = fidelity_refine and (not appearance_refine) and (not geo_refine)
    
    # background_color =  [247.0/255.0,247.0/255.0,247.0/255.0,0] # default back ground color of instantmesh generated multiview images
    background_color =  [1.0,1.0,1.0,0] # default back ground color of instantmesh generated multiview images
    
    os.makedirs(os.path.join(output_root,name),exist_ok=True)
    if refine_sr_by_sd:
        from unique3d.app.all_models import model_zoo
        model_zoo.init_models()
    else:
        model_zoo = None
    rembg_session = None # here we don't use it
    
    device = vertices.device

    
    camera_positions_np = camera_positions_B3.detach().cpu().numpy()
    
    ori_input_image_pil = Image.fromarray((input_image_HW4.detach().cpu().numpy()*255).astype(np.uint8))
    
    save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(), vertex_colors.detach().cpu().numpy(),
                 os.path.join(output_root,name,'instantmesh.obj'),flip_normal=False)

    input_image_HW4[input_image_HW4[...,3]==0] = torch.tensor([1.0,1.0,1.0,0],device=device) # background to white
    
    if not only_refine_fidelity:
        if mv_imgs_BHW4.shape[3] == 3: # error occurs when generating mv normals and remove background
            mv_imgs_BHW4 = torch.cat([mv_imgs_BHW4,torch.ones_like(mv_imgs_BHW4[:,:,:,0:1])],axis=3)
            # input_image_pil = remove_background(
            #     [Image.fromarray(mv_imgs_BHW4[i,...,:3].detach().cpu().numpy().astype(np.uint8)) for i in range(mv_imgs_BHW4.shape[0])], 
            #     rembg_session)
            # use_alpha = False
            temp_use_alpha = False
            geo_refine = False 
        else:
            temp_use_alpha = use_alpha
            
        mv_imgs_BHW4[mv_imgs_BHW4[...,3]==0] = torch.tensor([1.0,1.0,1.0,0],device=device) # background to white
        
        
        # save initial mesh: no opt geo, no deformation, but directly unproject mv images
        B,H,W,_ = mv_imgs_BHW4.shape
        # if H==1024:
        #     erode_kernel = 11
        # elif H == 320 or H==2048:
        #     erode_kernel = 3
        erode_kernel = 3
        view_weights_B=torch.ones(B).float().to(device)
        # view_weights_B[0] *= 10
        mv_imgs_BHW4_erode_alpha = erode_alpha_torch_NHW4(mv_imgs_BHW4,kernel_size=erode_kernel)
        initial_vertex_colors,final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=vertex_colors,
                            camera_positions=camera_positions_B3,
                            imgs_BHW4 = mv_imgs_BHW4_erode_alpha,
                            view_weights_B=view_weights_B,
                            eps = 0.05,res=2048, renderer=renderer,mode='fuse')
        
        initial_vertex_colors = complete_vert_colors_by_neighbors(vertices.detach(), faces.detach(),
                                initial_vertex_colors.detach(),  final_visible_vertex_mask_V.detach())
        save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),initial_vertex_colors.detach().cpu().numpy(),
            os.path.join(output_root,name,f'unproject_wo_2D_deform.obj'),flip_normal=False)
        
        # Not used Step: opt mv cameras
        opt_all_cameras = False
        if opt_all_cameras:
            camera_positions_np,camera_positions_B3 = opt_mv_cameras(vertices,faces,vertex_colors,mv_imgs_BHW4,camera_positions_np,renderer,
                            debug=False,output_root = output_root,name=name)

            
        # Step: opt geometry

        B = mv_imgs_BHW4.shape[0]
        
        if geo_refine:
            logger.info('Opt geometry')
            vertices,faces = opt_geo(vertices,faces,vertex_colors,mv_normals_BHW4,camera_positions_np,
                debug=False,output_root=output_root,name=name,renderer=renderer,coarse2fine=opt_geo_coarse2fine)
        


        # Step:  2D deformation (deform 2D mv imgs to increase multiview consistency)
        logger.info('2D deformation')
        # mv_images_pil.save(os.path.join(output_root,name,f'2D_deform_undeformed_imgs.png')) # TODO

        if appearance_refine:
            # deformed_imgs_pil = deform_2d(vertices,faces,vertex_colors,mv_imgs_BHW4,camera_positions_B3,renderer,background_color,
            deformed_imgs_pil = deform_2d(vertices,faces,initial_vertex_colors,mv_imgs_BHW4,camera_positions_B3,renderer,background_color,
                    output_root=output_root,name=name,use_alpha=temp_use_alpha)
        else:
            deformed_imgs_whole = rearrange(mv_imgs_BHW4.permute(0,3,1,2), '(n m) c h w -> c (n h) (m w)', n=B//2, m=2)       # (C,960, 640)
            deformed_imgs_pil = Image.fromarray((deformed_imgs_whole.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)).convert('RGBA')
        
        
        if refine_sr_by_sd or fast_sr:
            deformed_imgs_pil,ori_input_image_pil = sr_mv_imgs(ori_input_image_pil,deformed_imgs_pil, 
                refine_sr_by_sd = refine_sr_by_sd,fast_sr = fast_sr,
                output_root=output_root,name=name,model_zoo = model_zoo, rembg_session=rembg_session)
        
        deformed_imgs_whole = np.asarray(deformed_imgs_pil, dtype=np.float32) / 255.0

        deformed_imgs_whole = torch.from_numpy(deformed_imgs_whole).permute(2, 0, 1).contiguous().float().to(device)     # (C, 960, 640)
        B = mv_imgs_BHW4.shape[0]
        deformed_imgs_B4HW = rearrange(deformed_imgs_whole, 'c (n h) (m w) -> (n m) c h w', n=B//2, m=2)        # (6, C, 320, 320)
        deformed_imgs_BHW4 = deformed_imgs_B4HW.permute(0,2,3,1)
        
        # save mesh
        deformed_mesh_2d_path = os.path.join(output_root,name,f'2D_deform_deformed.obj')
        if not os.path.exists(deformed_mesh_2d_path):
        # if True:
            
            per_view_shrinked_per_vert_visibility = None
            if use_vertex_wise_NBF:
                
                B,H,W,_ = deformed_imgs_BHW4.shape
                vertex_colors,final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
                        camera_positions=camera_positions_B3,
                        imgs_BHW4 = deformed_imgs_BHW4,
                        eps = 0.05,res=2048, renderer=renderer,mode='fuse',
                        per_view_shrinked_per_vert_visibility=None) # if want to visualize NBF, apply color first
                border_area_masks_NHW,border_edge_masks_NHW = NBF_prepare_per_kernel_per_view_mv_img_validation(vertices,faces,
                                    camera_positions_B3,H,
                                    debug=False,mv_imgs_NHWC=deformed_imgs_BHW4,
                                    vertex_colors = vertex_colors,
                                    renderer = renderer,
                                    save_img_path=os.path.join(output_root,name,'NBF_mv_imgs.png'))
            
                new_alpha = deformed_imgs_BHW4[:, :, :, 3] * (1-border_area_masks_NHW.float())
                imgs_to_project_BHW4 = torch.cat([deformed_imgs_BHW4[...,:3],new_alpha.unsqueeze(-1)],dim=-1)
            
            else:
                imgs_to_project_BHW4 = deformed_imgs_BHW4
                
            vertex_colors,final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
                        camera_positions=camera_positions_B3,
                        imgs_BHW4 = imgs_to_project_BHW4,
                        eps = 0.05,res=2048, renderer=renderer,mode='fuse',
                        per_view_shrinked_per_vert_visibility=per_view_shrinked_per_vert_visibility)
                        # per_view_shrinked_per_vert_visibility=None)
            
            vertex_colors = complete_vert_colors_by_neighbors(vertices, faces,vertex_colors, final_visible_vertex_mask_V)
            save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(), 
                    deformed_mesh_2d_path,flip_normal=False)
        else:
            vertices,faces,vertex_colors = loadobj_color(deformed_mesh_2d_path,device=device)
    

    # visualize_results(mv_imgs_BHW4.permute(0,3,1,2), mv_imgs_BHW4.permute(0,3,1,2), 
    #                   deformed_imgs_BHW4.permute(0,3,1,2), deformation_field_B2HW, output_path='output')
    
    # # render BxB img
    # rendered_rgbas_BBHW4 = render_visible_area(vertices,faces,deformed_imgs_BHW4,camera_positions_B3,
    #                     res=320,fov_in_degrees=30,renderer=renderer)
    # save_BxB_rgba_img(rendered_rgbas_BBHW4,save_path=os.path.join(output_root,name,'rendered_rgbas_BBHW4.png'))
    
    

    # # Step: opt input view camera

    if fidelity_refine:
        for input_cam_setting_i in range(2):
            if not lgm:
                if input_cam_setting_i == 0: # only deal with 0 elevation
                    if not input_all_0_elevation:
                        continue
                    input_cameras_position_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=dist)
                    input_cameras_position_13 = torch.tensor(input_cameras_position_np).float().to(device)
                    end_str = '_0'
                    
                else: # deal with non-zero elevation
                    if input_all_0_elevation: 
                        continue
                    input_cameras_position_13,input_cameras_position_np,vertices = fit_input_view_instantmesh(vertices,faces,vertex_colors,
                                            input_image_HW4,renderer,output_root=output_root,name=name)
 
                        
                    
                    end_str = ''
                if geo_refine and (not appearance_refine) and fidelity_refine:
                    end_str += '_no_2D_deform'
            else:
                
                if input_cam_setting_i == 0: 
                    continue
                print('LGM here!!!')
                import kiui
                kiui.lo(vertices)
                input_cameras_position_13,input_cameras_position_np,vertices = fit_front_view_lgm(vertices,faces,vertex_colors,
                                                input_image_HW4,renderer,output_root=output_root,name=name)
                end_str = ''
                
                kiui.lo(input_cameras_position_np)
                # print('lgm camera extimation',input_cameras_position_np)
                # print('lgm camera input_cameras_position_13',input_cameras_position_13)
            if not skip_3D_deform_but_still_unproject:
                # Step: 3D deformation to fit front view 
                if deform_3D_method == 'Jacobian' :
                    if lap_weight == 1e5:
                        deformed_mesh_3d_path = os.path.join(output_root,name,f'3D_deformed_mesh{end_str}.obj')
                    else:
                        deformed_mesh_3d_path = os.path.join(output_root,name,f'3D_deformed_mesh{end_str}_{deform_3D_method}_{lap_weight}.obj')
                else:
                    deformed_mesh_3d_path = os.path.join(output_root,name,f'3D_deformed_mesh{end_str}_{deform_3D_method}_{lap_weight}.obj')
                if not os.path.exists(deformed_mesh_3d_path):
                # if True:
                    try:
                        logger.info('3D deformation')
                        default_camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=dist) 
                        if deform_3D_method == 'Jacobian': # Jacobian, vertex_replacement, 3D_deformation_field
                            new_vertices,new_faces,new_vertex_colors  = \
                                                                    jacob_mesh_deformation(vertices,faces,vertex_colors,
                                                                    input_image_HW4.unsqueeze(0),input_cameras_position_np,
                                                                    crop=crop_input_view, # crop_input_view
                                                                    save_path=os.path.join(output_root,name,f'3D_mesh_deform{end_str}.png'),
                                                                    lap_weight = lap_weight, #
                                                                    renderer=renderer 
                                                                    )
                        elif deform_3D_method == 'vertex_replacement':
                            new_vertices,new_faces,new_vertex_colors  = \
                                                    no_jacob_mesh_deformation(vertices,faces,vertex_colors,
                                                    input_image_HW4.unsqueeze(0),input_cameras_position_np,
                                                    crop=crop_input_view, # crop_input_view
                                                    save_path=os.path.join(output_root,name,f'3D_mesh_deform{end_str}_{deform_3D_method}_{lap_weight}.png'),
                                                    lap_weight = lap_weight, #
                                                    renderer=renderer  
                                                    )
                        elif deform_3D_method == '3D_deformation_field':
                            new_vertices = optimize_mesh_deformation(vertices,faces,vertex_colors,
                                                                    input_image_HW4.unsqueeze(0),input_cameras_position_np,
                                                                    crop=crop_input_view, # crop_input_view
                                                                    save_path=os.path.join(output_root,name,f'3D_mesh_deform{end_str}_{deform_3D_method}_{lap_weight}.png'),
                                                                    lap_weight = lap_weight,
                                                                    renderer=renderer  
                                                                    )
                            new_faces = faces
                            new_vertex_colors = vertex_colors
                        if new_vertices is not None:
                            vertices = new_vertices
                            faces = new_faces
                            vertex_colors= new_vertex_colors
                    except:
                        deformed_vertices = vertices
                        traceback.print_exc()
                        print('failed to conduct 3D mesh deformation to fit the input view' )
                    
                    save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(),
                        deformed_mesh_3d_path,flip_normal=False)
                    
                else:
                    vertices,faces,vertex_colors = loadobj_color(deformed_mesh_3d_path,device=device)

                
                
                
            # Step: deform input view (No need to do so since we already have 3D deformation to increase consistency between mesh and input img)
            deform_input_view = False
            if deform_input_view:
                deformed_input_img_pil,deformed_input_img_HWC = deform_input_img(vertices,faces,vertex_colors,
                                input_image_HW4,input_cameras_position_13,renderer,crop_input_view,output_root,name)
            
            else:
                # deformed_input_img_HWC = input_image_HW4 # TODO
                ori_input_img_HW4 = np.asarray(ori_input_image_pil).astype(np.float32) / 255.0
                ori_input_img_HW4 = torch.tensor(ori_input_img_HW4).to(device)
                deformed_input_img_HWC = ori_input_img_HW4
                
            # apply input view color
            with torch.no_grad():
                # first erode alpha for all views to avoid artifact 
                # deformed_imgs_BHW4 = erode_alpha_torch_NHW4(deformed_imgs_BHW4)
                deformed_input_img_HWC = erode_alpha_torch_NHW4(deformed_input_img_HWC.unsqueeze(0))[0]
                if skip_3D_deform_but_still_unproject:
                    final_mesh_path = os.path.join(output_root,name,f'final_mesh{end_str}_skip_3D_deform_but_still_unproject.obj')
                else:
                    if deform_3D_method == 'Jacobian':
                        if lap_weight == 0:
                            final_mesh_path = os.path.join(output_root,name,f'final_mesh{end_str}_{deform_3D_method}_{lap_weight}.obj')
                        else:
                            final_mesh_path = os.path.join(output_root,name,f'final_mesh{end_str}.obj')
                    else:
                        final_mesh_path = os.path.join(output_root,name,f'final_mesh{end_str}_{deform_3D_method}_{lap_weight}.obj')
                old_vertex_colors = vertex_colors.clone()
                vertex_colors = project_front_view_colors(vertices,faces,vertex_colors,
                                                        deformed_input_img_HWC,input_cameras_position_13,renderer,
                                                        crop_input_view,output_root,name,end_str= end_str,use_vertex_wise_NBF=use_vertex_wise_NBF)
                save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(),
                final_mesh_path,flip_normal=False)
                print('final mesh saved:',final_mesh_path)
                

    