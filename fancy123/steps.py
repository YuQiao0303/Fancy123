from einops import rearrange
import torch
import numpy as np
import os
from PIL import Image
from kiui.lpips import LPIPS
from fancy123.optimize.camera_optimize import optimize_my_camera, select_best_camera_position, select_best_camera_position_textured_mesh, select_best_elevation, select_best_mesh_scale
from fancy123.optimize.deform_2d import opt_mv_images_all
from fancy123.optimize.img_registration import apply_deformation, optimize_deformation
from fancy123.optimize.mesh_optimize import run_mesh_refine
from fancy123.unproject.NBF import NBF_prepare_per_kernel_per_view_mv_img_validation, NBF_prepare_per_kernel_per_view_shrinked_per_vert_visibility
from fancy123.utils.mesh_utils import subdivide_mesh
from fancy123.utils.normal_utils import gen_mv_normal_zero123pp
from fancy123.utils.temp_utils import apply_view_color2mesh, complete_vert_colors_by_neighbors, sr_6_view_imgs_by_SD
from src.utils.camera_util import elevation_azimuth_radius_to_xyz
from src.utils.mesh_util import loadobj_color, save_obj
from rembg import new_session, remove

from unique3d.app.utils import make_image_grid, split_image
from unique3d.scripts.refine_lr_to_sr import run_sr_fast
from torchvision.transforms import transforms
# not used
def opt_mv_cameras(vertices,faces,vertex_colors,mv_imgs_BHW4,camera_positions_np,renderer,debug=False,output_root=None,name=None):
    debug_save_path = os.path.join(output_root,name,debug_save_path)
    original_rendered_rgba_NHW4 = renderer.render(vertices,vertex_colors*2-1,
                                                  faces,camera_positions_np=camera_positions_np,rotate_normal=False)
    camera_positions_B3 = optimize_my_camera(mv_imgs_BHW4,vertices,faces,vertex_colors,
                                                camera_positions_np=camera_positions_np,
                                                debug=debug,debug_save_path=debug_save_path)
    camera_positions_np = camera_positions_B3.detach().cpu().numpy()
    save_opt_cameras = True
    if save_opt_cameras:
        rendered_rgba_NHW4 = renderer.render(vertices,vertex_colors*2-1,faces,camera_positions_np=camera_positions_np,rotate_normal=False)
        N,H,W,_ = rendered_rgba_NHW4.shape
        rendered_rgba_row = rendered_rgba_NHW4.reshape(-1,W,4)
        input_row = mv_imgs_BHW4.reshape(-1,W,4)

        original_rendered_rgba_row = original_rendered_rgba_NHW4.reshape(-1,W,4)
                    
        merge1 = (input_row+original_rendered_rgba_row)/2
        merge2 = (input_row+rendered_rgba_row)/2

        cat = torch.cat([merge1,original_rendered_rgba_row,input_row,rendered_rgba_row,merge2],axis=1)
        Image.fromarray((cat.detach().cpu().numpy()*255).astype(np.uint8)).save(os.path.join(output_root,name,f'opt_mv_camera.png'))
    return camera_positions_np,camera_positions_B3
        
        
def opt_geo(vertices,faces,vertex_colors,mv_normals_BHW4,camera_positions_np,
            debug=False,output_root=None,name=None,renderer=None,coarse2fine=False):   
    device = vertices.device
    B,H,W,_ = mv_normals_BHW4.shape
    refined_geo_path = os.path.join(output_root,name,f'refined_geo.obj')

    mv_normal_pil_list = [Image.fromarray((mv_normals_BHW4[i].detach().cpu().numpy()*255).astype(np.uint8)) for i in range(B)]
    if not os.path.exists(refined_geo_path):
    # if True:
        if coarse2fine:
            # save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(), 
            #     os.path.join(output_root,name,f'refined_geo_before_refine.obj'),flip_normal=False)
            # vertices,faces = run_mesh_refine(vertices, faces,mv_normal_pil_list, 
            #             camera_positions_np = camera_positions_np,
            #             fixed_vertex_ids=None,
            #             res = 320,
            #             steps=100, start_edge_len=0.02, end_edge_len=0.01, decay=0.99, update_normal_interval=20, 
            #         update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False,
            #         remesh=False,
            #         expand=True,debug=False,view_weights=None,
            #         background_color =None,use_debug_img=False,renderer=renderer)
            # save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(), 
            #     os.path.join(output_root,name,f'refined_geo_coarse.obj'),flip_normal=False)
            # vertices,faces = subdivide_mesh(vertices,faces,iterations=4)
            # vertices,faces = run_mesh_refine(vertices, faces,mv_normal_pil_list, 
            #             camera_positions_np = camera_positions_np,
            #             fixed_vertex_ids=None,
            #             res = 2048,
            #             steps=100, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=20, 
            #         update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False,
            #         remesh=True,
            #         expand=True,debug=False,view_weights=None,
            #         background_color =None,use_debug_img=False,renderer=renderer)
            # vertices,faces = subdivide_mesh(vertices,faces,iterations=2)
            vertices,faces = run_mesh_refine(vertices, faces,mv_normal_pil_list, 
                        camera_positions_np = camera_positions_np,
                        fixed_vertex_ids=None,
                        res = 2048,
                        steps=300, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=20, 
                    update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False,
                    remesh=True,
                    expand=True,debug=False,view_weights=None,
                    background_color =None,use_debug_img=False,renderer=renderer)
            vertices,faces = subdivide_mesh(vertices,faces,iterations=2)
        else:
            # 
            vertices,faces = run_mesh_refine(vertices, faces,mv_normal_pil_list, 
                        camera_positions_np = camera_positions_np,
                        fixed_vertex_ids=None,
                        res = 2048,
                        steps=300, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=20, 
                    update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False,
                    remesh=True,
                    expand=True,debug=False,view_weights=None,
                    background_color =None,use_debug_img=False,renderer=renderer)
        
        # vertices = taubin_smooth(vertices,faces)
        save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(), 
                refined_geo_path,flip_normal=False)
    else:
        vertices,faces,vertex_colors = loadobj_color(refined_geo_path,device=device)
    return vertices,faces


def deform_2d(vertices,faces,vertex_colors,mv_imgs_BHW4,camera_positions_B3,renderer,background_color,
              output_root=None,name=None,use_alpha=False,fix_first_view = False):
    def save_cat_img_for_check(vertices,faces,vertex_colors,mv_imgs_BHW4,camera_positions_B3,renderer,background_color,
              output_root=None,name=None,use_alpha=False,end_str = ''):
        rgba_BHW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                camera_positions_np=camera_positions_B3.detach().cpu().numpy(),
                                rotate_normal=False,background_color=background_color)
        B,H,W,_ = rgba_BHW4.shape
        deformed_imgs_BHW4_resize = transforms.Resize((H,W))(mv_imgs_BHW4.permute(0,3,1,2)).permute(0,2,3,1)
        deformed_imgs_BH_W4 = deformed_imgs_BHW4_resize.reshape(-1,W,4)
        rgba_BH_W4 = rgba_BHW4.reshape(-1,W,4)
        merge = (deformed_imgs_BH_W4 + rgba_BH_W4) /2
        diff = (rgba_BH_W4 - deformed_imgs_BH_W4).abs()
        cat = torch.cat([deformed_imgs_BH_W4,rgba_BH_W4,merge,diff],dim=1)[...,:3]
        
        
        if not use_alpha:
            rendered_foreground_img_BHW3 = rendered_foreground_mask.unsqueeze(-1).repeat(1,1,1,3).float()
            rendered_foreground_img_BH_W3 = rendered_foreground_img_BHW3.reshape(-1,W,3).float()
            img_foreground_img_BHW3 = img_foreground_mask.unsqueeze(-1).repeat(1,1,1,3).float()
            img_foreground_img_BH_W3 = img_foreground_img_BHW3.reshape(-1,W,3).float()
            merge2 = (img_foreground_img_BH_W3 + rendered_foreground_img_BH_W3) /2
            diff2 = (rendered_foreground_img_BH_W3 - img_foreground_img_BH_W3).abs()
            cat2 = torch.cat([img_foreground_img_BH_W3,rendered_foreground_img_BH_W3,merge2,diff2],dim=1)
            cat = torch.cat([cat,cat2],dim=0)
            
        
        Image.fromarray((cat[...,:3].detach().cpu().numpy()*255).astype(np.uint8)).save(
            os.path.join(output_root,name,f'2d_deform_check_{end_str}.png'))
        
    ori_mv_imgs_BHW4 = mv_imgs_BHW4.clone().to(vertices.device)
    mv_imgs_BHW4 = transforms.Resize((renderer.default_res,renderer.default_res))(mv_imgs_BHW4.permute(0,3,1,2)).permute(0,2,3,1)
    deformed_mv_img_path = os.path.join(output_root,name,f'2D_deform_deformed_imgs.png')
    
    # save before deformation images # if opt geo, don't do this since the vertices has changed
    # save_cat_img_for_check(vertices,faces,vertex_colors,mv_imgs_BHW4,camera_positions_B3,renderer,background_color,
    #     output_root=output_root,name=name,use_alpha=use_alpha,end_str = '_before')
    if not os.path.exists(deformed_mv_img_path): # skip exist
    # if True:

     
        deformed_imgs_BHW4,deformation_field_B2HW,vertex_colors,final_visible_vertex_mask_V, \
            rendered_foreground_mask,img_foreground_mask = \
            opt_mv_images_all(vertices,faces, # TODO, add detach here, only opt imgs
                            mv_imgs_BHW4,camera_positions_B3,
                            input_image_HW4=None,input_cameras_position_13=None,
                        res=renderer.default_res,fov_in_degrees=renderer.fov_in_degrees,renderer=renderer,
                        lr = 1e-1, epochs=100,scale_factor=16,debug=False,use_alpha=use_alpha,
                        background_color=background_color,fix_first_view=fix_first_view)
        # save deformed_imgs
        B,H,W,_ = ori_mv_imgs_BHW4.shape
        deformed_imgs_B4HW = apply_deformation(ori_mv_imgs_BHW4.permute(0,3,1,2), deformation_field_B2HW*H/renderer.default_res)
        deformed_imgs_BHW4 = deformed_imgs_B4HW.permute(0,2,3,1).contiguous()
        
        # save_obj(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(), 
        #             os.path.join(output_root,name,f'2D_deformed_obj_using_color_after_opt.obj'),flip_normal=False)
        
        if fix_first_view:
            deformed_imgs_BHW4[0] = ori_mv_imgs_BHW4[0]
        B,H,W,_ = deformed_imgs_BHW4.shape

        deformed_imgs_whole = rearrange(deformed_imgs_BHW4.permute(0,3,1,2), '(n m) c h w -> c (n h) (m w)', n=B//2, m=2)       # (C,960, 640)
 
        deformed_imgs_pil = Image.fromarray((deformed_imgs_whole.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)).convert('RGBA')
        deformed_imgs_pil.save(deformed_mv_img_path) 
        
        
        # # also save rendered imgs
        # save_cat_img_for_check(vertices,faces,vertex_colors,deformed_imgs_BHW4,camera_positions_B3,renderer,background_color,
        #     output_root=output_root,name=name,use_alpha=use_alpha,end_str = '_after')
      
        # # save compare img
        # deformed_resized_imgs_BHW4 = transforms.Resize((renderer.default_res,renderer.default_res))(deformed_imgs_BHW4.permute(0,3,1,2)).permute(0,2,3,1)
        # mv_resized_imgs_BHW4 = transforms.Resize((renderer.default_res,renderer.default_res))(mv_imgs_BHW4.permute(0,3,1,2)).permute(0,2,3,1)
        # deformed_resized_imgs_BH_W4 = deformed_resized_imgs_BHW4.reshape(-1,renderer.default_res,4)
        # mv_resized_imgs_BH_W4 = mv_resized_imgs_BHW4.reshape(-1,renderer.default_res,4)
        # merge = (deformed_resized_imgs_BH_W4 + mv_resized_imgs_BH_W4) /2
        # diff = (deformed_resized_imgs_BH_W4 - mv_resized_imgs_BH_W4).pow(2)
        
        # compare_img = torch.cat([deformed_resized_imgs_BH_W4,mv_resized_imgs_BH_W4,merge,diff],dim=1)[...,:3]
        # Image.fromarray((compare_img.detach().cpu().numpy()*255).astype(np.uint8)).save(os.path.join(output_root,name,'2D_deform_compare.png'))

    else:
        B,H,W,_ = mv_imgs_BHW4.shape
        deformed_imgs_pil = Image.open(deformed_mv_img_path)
    return deformed_imgs_pil







def sr_mv_imgs(input_image_pil,deformed_imgs_pil, 
               refine_sr_by_sd = False,fast_sr = False,output_root=None,name=None,model_zoo = None, rembg_session=None):
    if refine_sr_by_sd:
        if not os.path.exists(os.path.join(output_root,name,'deformed_imgs_sr.png')):
            deformed_imgs_pil,deformed_imgs_pil_list = sr_6_view_imgs_by_SD(input_image_pil,deformed_imgs_pil,model_zoo)
            deformed_imgs_pil_list_rmbg = []
            for i,deformed_img_pil in enumerate(deformed_imgs_pil_list):
                img = remove(deformed_img_pil, session=rembg_session)
                deformed_imgs_pil_list_rmbg.append(img)
            deformed_imgs_pil_list = deformed_imgs_pil_list_rmbg
            deformed_imgs_pil = make_image_grid(deformed_imgs_pil_list,rows=3,cols=2)
            deformed_imgs_pil.save(os.path.join(output_root,name,'deformed_imgs_sr.png'))
            
        else:
            deformed_imgs_pil = Image.open(os.path.join(output_root,name,'deformed_imgs_sr.png')).convert('RGBA')
    if fast_sr:
        if not os.path.exists(os.path.join(output_root,name,'deformed_imgs_sr.png')):
            deformed_imgs_pil_list = split_image(deformed_imgs_pil, rows=3,cols=2)
            deformed_imgs_pil_list = run_sr_fast(deformed_imgs_pil_list, scale=4)
            # deformed_imgs_pil = make_image_grid(deformed_imgs_pil_list,rows=3,cols=2)
            # deformed_imgs_pil_list_rmbg = []
            # for i,deformed_img_pil in enumerate(deformed_imgs_pil_list):
            #     img = remove(deformed_img_pil, session=rembg_session)
            #     deformed_imgs_pil_list_rmbg.append(img)
            # deformed_imgs_pil_list = deformed_imgs_pil_list_rmbg
            deformed_imgs_pil = make_image_grid(deformed_imgs_pil_list,rows=3,cols=2)
            deformed_imgs_pil.save(os.path.join(output_root,name,'deformed_imgs_sr.png'))
            
            # input_image_pil = run_sr_fast([input_image_pil], scale=4)[0] # also sr input img
        else:
            deformed_imgs_pil = Image.open(os.path.join(output_root,name,'deformed_imgs_sr.png')).convert('RGBA')
    return deformed_imgs_pil,input_image_pil



def fit_input_view_instantmesh(vertices,faces,vertex_colors,input_image_HW4,renderer,output_root=None,name=None,
                          further_optimize = True,crop=True,save_opt_cameras = False):
    device = vertices.device
    input_view_camera_position_path = os.path.join(output_root,name,f'input_cameras_position_13.pt')
    input_image_HW4 =  transforms.Resize((320,320))(input_image_HW4.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)
    if crop:
        crop_views = [0]
    else:
        crop_views = None
    if not os.path.exists(input_view_camera_position_path):
    # if True:
        
        default_camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=4) 
        default_camera_positions = torch.tensor(default_camera_positions_np).float().to(device)
        with torch.no_grad():
            best_lpips,select_input_cameras_position_np = select_best_elevation(input_image_HW4, vertices,faces,vertex_colors, 
                                                                            fov_in_degrees = 30,res=320,device = device,
                                                                            fine_level = not further_optimize,)
        input_cameras_position_np = select_input_cameras_position_np.copy()
        if save_opt_cameras:
            default_view_rendered_rgba_NHW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                                    camera_positions_np=default_camera_positions_np, 
                                                    rotate_normal=False,background_color=[1,1,1,0],crop_views=crop_views)
            selected_view_rendered_rgba_NHW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                                    camera_positions_np=select_input_cameras_position_np, #select_input_cameras_position_np,
                                                    rotate_normal=False,background_color=[1,1,1,0],crop_views=crop_views)
        if further_optimize:
            # optimize camera position
            input_cameras_position_13,best_rendered_img = optimize_my_camera(input_image_HW4.unsqueeze(0),vertices,faces,vertex_colors,
                                                        camera_positions_np=input_cameras_position_np,
                                                        debug=True,debug_save_path=f'{name}_cam_opt_debug.gif',
                                                        crop_views=crop_views)
            
            input_cameras_position_np = input_cameras_position_13.detach().cpu().numpy()
            
            # optimize vertices scale
            with torch.no_grad():
                best_scale_factor,scaled_vertices,best_id,best_lpips = select_best_mesh_scale(vertices,faces,vertex_colors,
                                                                                              input_cameras_position_np,
                                                                                              input_image_HW4=input_image_HW4,
                                renderer=renderer, res=renderer.default_res,save_for_debug = False,save_path = None)
                vertices = scaled_vertices
        else:
            input_cameras_position_13 = torch.tensor(input_cameras_position_np).float().to(device)
            

        
        torch.save(input_cameras_position_13,input_view_camera_position_path)
        
        if save_opt_cameras:
            
            
            scaled_rendered_rgba_NHW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                                    camera_positions_np=input_cameras_position_np,rotate_normal=False,
                                                    crop_views=crop_views,background_color=[1,1,1,0])

            
            rendered_rgba_NHW4 = best_rendered_img
            N,H,W,_ = rendered_rgba_NHW4.shape
      
            opt_rgba_row = rendered_rgba_NHW4.reshape(-1,W,4)
            input_row = input_image_HW4.unsqueeze(0).reshape(-1,W,4)
            scaled_rendered_rgba_row = scaled_rendered_rgba_NHW4.unsqueeze(0).reshape(-1,W,4)
            selected_view_rendered_rgba_row = selected_view_rendered_rgba_NHW4.reshape(-1,W,4)
            default_view_rendered_rgba_row = default_view_rendered_rgba_NHW4.reshape(-1,W,4)
                        
            merge0 = (input_row+default_view_rendered_rgba_row)/2
            merge1 = (input_row+selected_view_rendered_rgba_row)/2
            merge2 = (input_row+opt_rgba_row)/2
            merge3 = (input_row+scaled_rendered_rgba_row)/2
            
            diff0 = (input_row-default_view_rendered_rgba_row).abs()
            diff1 = (input_row-selected_view_rendered_rgba_row).abs()
            diff2 = (input_row-opt_rgba_row).abs()
            diff3 = (input_row - scaled_rendered_rgba_row).abs()

            cat0 = torch.cat([input_row, default_view_rendered_rgba_row,merge0,diff0],axis=1)
            cat = torch.cat([input_row, selected_view_rendered_rgba_row,merge1,diff1],axis=1)
            cat2 = torch.cat([input_row, opt_rgba_row,merge2,diff2],axis=1)
            cat3 = torch.cat([input_row, scaled_rendered_rgba_row,merge3,diff3],axis=1)
            cat = torch.cat([cat0,cat,cat2,cat3],axis=0)[...,:3]
         
            Image.fromarray((cat.detach().cpu().numpy()*255).astype(np.uint8)).save(os.path.join(output_root,name,f'opt_input_camera.png'))

        
        
    else:
        input_cameras_position_13 = torch.load(input_view_camera_position_path).to(device)
        input_cameras_position_np = input_cameras_position_13.detach().cpu().numpy()
    
    # print('input_cameras_position_13 after loading',input_cameras_position_13)
    # rgba = renderer.render(vertices,vertex_colors*2-1,faces,camera_positions_np=input_cameras_position_np,rotate_normal=False)
    # import kiui
    # kiui.vis.plot_image(rgba[0])
    return input_cameras_position_13,input_cameras_position_np,vertices


def fit_front_view_lgm(vertices, faces, vertex_colors, input_img_HW4,renderer,dist=1.5,output_root=None,name=None,
                          further_optimize = True,crop=True,save_opt_cameras = True,res=320,
                       ):
    device = vertices.device
   
    lpips = LPIPS(net='vgg').to(device)
    
    
    input_img_HW4 = transforms.Resize((res,res))(input_img_HW4.permute(2,0,1).unsqueeze(0))[0].permute(1,2,0).to(device)
    # 1. coarse
    coarse_elevation_step = 10
    elevation_start = -90
    elevation_end = 90

    coarse_azimuth_step = 10
    azimuth_start = -90
    azimuth_end = 90
    num = (elevation_end - elevation_start) // coarse_elevation_step +1 

    elevation_in_degrees = np.linspace(elevation_start,elevation_end,num=(elevation_end - elevation_start) // coarse_elevation_step +1 )
    azimuth_in_degrees = np.linspace(azimuth_start,azimuth_end,num=(azimuth_end - azimuth_start) // coarse_azimuth_step +1 )

    elevation_in_degrees = np.repeat(elevation_in_degrees, len(azimuth_in_degrees))
    azimuth_in_degrees = np.tile(azimuth_in_degrees, num)
    
    
    camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees,azimuth_in_degrees,radius=dist) 
    best_campos_id,best_lpips = \
        select_best_camera_position(vertices, faces, vertex_colors,
                                              camera_positions_np,input_rgbs_1HW4=input_img_HW4.unsqueeze(0),
                                              
                                renderer=renderer,lpips = lpips,
                                fov_in_degrees = None,
                                res=res,batch_size=10,save_for_debug = False,save_path = 'coarse')
    # print('coarse, best id',best_campos_id,best_lpips)
    best_elevation = elevation_in_degrees[best_campos_id]
    best_azimuth = azimuth_in_degrees[best_campos_id]
    
    # 2. fine
    fine_elevation_step = 2
    fine_elevation_start = best_elevation-coarse_elevation_step+1
    fine_elevation_end = best_elevation+coarse_elevation_step-1
    
    fine_azimuth_step = 2
    fine_azimuth_start = best_azimuth-coarse_azimuth_step+1
    fine_azimuth_end = best_azimuth+coarse_azimuth_step-1
    num = int((fine_elevation_end - fine_elevation_start) / fine_elevation_step +1)
    
    elevation_in_degrees = np.linspace(fine_elevation_start ,fine_elevation_end,
                                           num=int((fine_elevation_end - fine_elevation_start) / fine_elevation_step +1) )
    azimuth_in_degrees = np.linspace(fine_azimuth_start,fine_azimuth_end,
                                     num=int((fine_azimuth_end - fine_azimuth_start) / fine_azimuth_step +1) )
    
    elevation_in_degrees = np.repeat(elevation_in_degrees, len(azimuth_in_degrees))
    azimuth_in_degrees = np.tile(azimuth_in_degrees, num)
    
    
    camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees,azimuth_in_degrees,radius=dist) 
    best_campos_id,best_lpips = \
        select_best_camera_position(vertices, faces, vertex_colors,
                                              camera_positions_np,input_rgbs_1HW4=input_img_HW4.unsqueeze(0),
                                renderer=renderer,lpips = lpips,
                                fov_in_degrees = None,
                                res=320,batch_size=10,save_for_debug = False,save_path = 'fine')
    # print('fine, best id',best_campos_id,best_lpips)
    best_elevation = elevation_in_degrees[best_campos_id]
    best_azimuth = azimuth_in_degrees[best_campos_id]
    best_cam_position_np = camera_positions_np[best_campos_id:best_campos_id+1].astype(np.float32)
    
    best_cam_position = torch.from_numpy(best_cam_position_np).to(device)
    return best_cam_position,best_cam_position_np,vertices
    

# not used
def pred_normal(deformed_imgs_BHW4,input_image_pil,output_root=None,name=None):
    '''
    TODO: code long time ago, not used now, need rewriting if want to use
    '''
    # pred normal
    B,H,W,_ = deformed_imgs_BHW4.shape
    device = deformed_imgs_BHW4.device
    camera_positions_0_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=4) 

    deformed_mv_normal_path = os.path.join(output_root,name,f'deformed_mv_normal.png')
    if not os.path.exists(deformed_mv_normal_path):
        deformed_imgs_whole = rearrange(deformed_imgs_BHW4.permute(0,3,1,2), '(n m) c h w -> c (n h) (m w)', n=3, m=2)       # (C,H,W)
        deformed_imgs_pil = Image.fromarray((deformed_imgs_whole.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))

        # deformed_imgs_pil_transparent_bg, mv_normal_pil,normal_pipeline = gen_mv_normal_zero123pp(input_image_pil,deformed_imgs_pil,
        #                                   normal_pipeline=normal_pipeline,input_247=False,device='cuda') # use original input img
        # deformed_imgs_pil_transparent_bg.save(deformed_mv_img_path.replace('.png','_transparent_bg.png'))
        _, mv_normal_pil,normal_pipeline = gen_mv_normal_zero123pp(input_image_pil,deformed_imgs_pil,
                                normal_pipeline=normal_pipeline,input_247=True,device='cuda') # use original input img
        
        mv_normal_pil.save(deformed_mv_normal_path)
    else:
        mv_normal_pil = Image.open(deformed_mv_normal_path)
    mv_normals_whole = np.asarray(mv_normal_pil, dtype=np.float32) / 255.0
    mv_normals_whole = torch.from_numpy(mv_normals_whole).permute(2, 0, 1).contiguous().float().to(device)      # (C, 960, 640)
    mv_normals = rearrange(mv_normals_whole, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)       # (6, C, 320, 320)
    mv_normals_BHW4 = mv_normals.permute(0,2,3,1)# 6,H,W,4
    
    
    # # opt geometry
    # refined_geo_path = os.path.join(output_root,name,f'refined_geo.obj')
    # mv_normal_pil_list = [Image.fromarray((mv_normals_BHW4[i].detach().cpu().numpy()*255).astype(np.uint8)) for i in range(B)]
    
    # if not os.path.exists(refined_geo_path):
    #     vertices,faces = opt_geo_by_normal_coarse2fine(vertices,faces,camera_positions_B3,mv_normal_pil_list)
    # else:
    #     vertices,faces = loadobj(refined_geo_path)
    #     vertices = torch.tensor(vertices).to(device).float()
    #     faces = torch.tensor(faces).to(device).long()
    return mv_normal_pil,mv_normals_BHW4


# not used
def deform_input_img(vertices,faces,vertex_colors,input_image_HW4,input_cameras_position_13,renderer,crop_input_view,output_root,name):

    device = vertices.device
    H,W,_ = input_image_HW4.shape
    input_view_camera_position = input_cameras_position_13
    if crop_input_view:
        crop_views = [0]
    else:
        crop_views = None
    # deform input view
    rgba_1HW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                            camera_positions_np=input_view_camera_position.detach().cpu().numpy(),
                            rotate_normal=False,crop_views=crop_views) # background default black

    fixed_imgs = rgba_1HW4.permute(0,3,1,2).detach()
    input_image_HW4[input_image_HW4[...,3]==0] = torch.tensor([0.0,0.0,0.0,0.0]).to(device) # background set to black
    moving_imgs = input_image_HW4.unsqueeze(0).permute(0,3,1,2).detach()



    deformed_input_view_path = os.path.join(output_root,name,f'deformed_input_view.png')
    if not os.path.exists(deformed_input_view_path):
        deformed_input_img_1CHW = optimize_deformation(fixed_imgs = fixed_imgs, moving_imgs=moving_imgs, num_epochs=300, 
                        learning_rate=0.3, scale_factor=16,output_path=None)
        deformed_input_img_HWC = deformed_input_img_1CHW[0].permute(1,2,0)
    
        cat = torch.cat([deformed_input_img_HWC,(rgba_1HW4[0] + deformed_input_img_HWC)/2,
                        input_image_HW4,(rgba_1HW4[0] + input_image_HW4)/2,rgba_1HW4[0] ],dim=1)
        deformed_input_img_pil = Image.fromarray((cat.detach().cpu().numpy()*255).astype(np.uint8))
        deformed_input_img_pil.save(deformed_input_view_path)
        
        # import kiui
        # cat = torch.cat([rgba_1HW4[0],input_image_HW4],dim=1)
        # kiui.vis.plot_image(cat)
    else:
        deformed_input_img_pil = Image.open(deformed_input_view_path)
        deformed_input_img_HWC = np.asarray(deformed_input_img_pil, dtype=np.float32) / 255.0
        
        deformed_input_img_HWC = deformed_input_img_HWC[:,:W,:]
        deformed_input_img_HWC = torch.from_numpy(deformed_input_img_HWC).contiguous().float().to(device)
    return deformed_input_img_pil,deformed_input_img_HWC


def project_front_view_colors(vertices,faces,vertex_colors,deformed_input_img_HWC,
                              input_cameras_position_13,renderer,crop_input_view,output_root,name,end_str = '',use_vertex_wise_NBF=True):
    input_view_camera_position = input_cameras_position_13
    if crop_input_view:
        crop_views=[0]
    else:
        crop_views=None
    only_update_input_view = True
    if only_update_input_view:

        per_view_shrinked_per_vert_visibility = None
        if use_vertex_wise_NBF:
            
            H,W,_ = deformed_input_img_HWC.shape
            # print('input_view_camera_position before NBF',input_view_camera_position)
            border_area_masks_NHW,border_edge_masks_NHW = NBF_prepare_per_kernel_per_view_mv_img_validation(vertices,faces,
                    input_view_camera_position,mv_img_res=H,crop_views=crop_views,
                    debug=False,mv_imgs_NHWC=deformed_input_img_HWC.unsqueeze(0),
                    vertex_colors = vertex_colors,
                    renderer=renderer,
                    save_img_path = None
                    # save_img_path=os.path.join(output_root,name,f'NBF_input_view{end_str}.png')
                    )
            new_alpha = deformed_input_img_HWC.unsqueeze(0)[:, :, :, 3] * (1-border_area_masks_NHW.float())
            imgs_to_project_BHW4 = torch.cat([deformed_input_img_HWC.unsqueeze(0)[...,:3],new_alpha.unsqueeze(-1)],dim=-1)
        else:
            imgs_to_project_BHW4 = deformed_input_img_HWC.unsqueeze(0)

        input_view_vertex_colors,input_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=vertex_colors,
                            camera_positions=input_view_camera_position,
                            imgs_BHW4 =imgs_to_project_BHW4,
                            eps = 0 ,#0.1,
                            res=2048, renderer=renderer,mode='fuse',crop_views=crop_views,
                            per_view_shrinked_per_vert_visibility=per_view_shrinked_per_vert_visibility)
        vertex_colors = input_view_vertex_colors
        # vertex_colors[input_visible_vertex_mask_V] = input_view_vertex_colors[input_visible_vertex_mask_V]
        # vertex_colors = complete_vert_colors_by_neighbors(vertices, faces,vertex_colors, final_visible_vertex_mask_V|input_visible_vertex_mask_V)
    # else:
        
    #     # vertex_colors, final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
    #     #                                     camera_positions=torch.cat([camera_positions_B3[1:-1],input_view_camera_position],dim=0),
    #     #                                     imgs_BHW4 = torch.cat([deformed_imgs_BHW4[1:-1],deformed_input_img_HWC.unsqueeze(0)],dim=0),
    #     #                                     eps=0.05,res=2048, renderer=renderer,mode='fuse',view_weights_B=[1,1,1,1,100])
    #     if crop_input_view:
    #         crop_views=[6]
    #     else:
    #         crop_views=None
    #     vertex_colors, final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
    #                                         camera_positions=torch.cat([camera_positions_B3,input_view_camera_position],dim=0),
    #                                         imgs_BHW4 = torch.cat([deformed_imgs_BHW4,deformed_input_img_HWC.unsqueeze(0)],dim=0),
    #                                         eps=0.05,res=2048, renderer=renderer,mode='fuse',view_weights_B=[0.001,1,1,1,1,0.001,100],
    #                                         crop_views=crop_views)
    #     vertex_colors = complete_vert_colors_by_neighbors(vertices, faces,vertex_colors, final_visible_vertex_mask_V)
    return vertex_colors