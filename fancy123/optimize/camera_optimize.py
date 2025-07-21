# ====================================================================================================================
# Given a 3D mesh of an object and an image of the object from unknown camera parameters, the code tries to get the camera parameters
# ====================================================================================================================


import sys
import logging
from tqdm import tqdm

from src.utils.camera_util import elevation_azimuth_radius_to_xyz
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
# from .my_renderer import NeuralRender, GeneralRenderer
from ..render.general_renderer import  GeneralRenderer





def optimize_my_camera(input_images, vertices,faces,vertex_colors,
                    fov_in_degrees = 30, camera_positions_np = None,res=320,device = 'cuda',
                    lr=0.05, max_iter=100,debug=False,debug_save_path = None,crop_views=[0]):
    '''
    input_images: tensor of shape [B, H, W,4]
    camera_positions_np: np array of shape [B,3]
    '''
    logger = logging.getLogger('logger')
    logger.info('Optimizing camera...')
    from src.utils.camera_util import elevation_azimuth_radius_to_xyz
    

    # input_image = resize_foreground_torch(input_image, 1.0)
    input_images = transforms.Resize((res,res))(input_images.permute(0,3,1,2)).permute(0,2,3,1)
   
    input_rgbs = input_images[..., :3] * input_images[..., 3:4] + (1 - input_images[..., 3:4]) # B,H,W,3 # rgba to rgb white bg
    input_alphas = input_images[...,3:4]
    # res = input_images.shape[1]


    ## Load renderer
    
    renderer = GeneralRenderer(device=device)

    if debug:
        ## Prepare for saving gif
        filename_output = debug_save_path
        # writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    
    # # Init camera
    camera_positions = camera_positions_np
    if camera_positions is None:
        camera_positions = elevation_azimuth_radius_to_xyz(elevation_in_degrees=[0],azimuth_in_degrees=[0],radius=4) 
    camera_positions = torch.tensor(camera_positions).float().to(device).requires_grad_()
    up_world = torch.tensor([0,1,0]).float().to(device)



    # # Save for later so we have some comparison of what changed
    best_camera_positions = camera_positions.detach().clone()
    best_loss = 100.0
    best_rendered_img = None

    # Train     
    optimizer = optim.Adam([camera_positions], lr=lr)
    lpips = LPIPS(net='vgg').to(device)

    for idx in tqdm(range(max_iter),desc='Optimizing camera'):
    
        optimizer.zero_grad()

        # render
        rgba_imgs = renderer.render(vertices,vertex_colors.detach()*2-1,faces,
                                    camera_positions=camera_positions,
                                    fov_in_degrees=fov_in_degrees,
                                    res=res,
                                    rotate_normal=False,crop_views=crop_views,background_color=[1,1,1,0])
        # rgba_img = rgba_imgs[0] # H,W,4

        # rgba_img = resize_foreground_torch(rgba_img, 1.0)
        # rgba_img = transforms.Resize((res,res))(rgba_img.permute(2,0,1)).permute(1,2,0)

        
        rendered_rgbs = rgba_imgs[...,:3] * rgba_imgs[...,3:4] + (1 - rgba_imgs[...,3:4]) # H,W,3 # rgba to rgb white bg
        rendered_alphas = rgba_imgs[...,3:4]

    
        loss = 0
        mask_mse_loss = torch.nn.functional.mse_loss(rendered_alphas, input_alphas).mean()
        rgb_mse_loss = torch.nn.functional.mse_loss(rendered_rgbs, input_rgbs).mean()
        rgb_lpips_loss = lpips(rendered_rgbs.permute(0,3,1,2)*2-1, input_rgbs.permute(0,3,1,2)*2-1).mean() # input [B, 3, H, W] image in [-1, 1]
        # loss += mask_mse_loss
        # loss += 0.001* rgb_mse_loss
        # # loss +=  0.001 *rgb_lpips_loss
        
        loss += 0.01 * mask_mse_loss
        loss += 0.01* rgb_mse_loss
        loss +=  1.0 *rgb_lpips_loss

        if loss<best_loss:
                best_loss = loss.detach().clone()
                best_camera_positions = camera_positions.detach().clone()
                best_rendered_img = rgba_imgs # H,W,4
        loss.backward()
        optimizer.step()
        
        if idx%10 == 0:
            logger.info(f'{idx}: Loss: {loss.item()}, mask: {mask_mse_loss.item()}, rgb: {rgb_mse_loss.item()}, lpips: {rgb_lpips_loss.item()}')
        # if debug and idx%10 == 0:
        #     # print(f'Iteration {idx}:')
        #     # print(f'Loss: {loss.item()}')
        #     print(f'{idx}: Loss: {loss.item()}, mask: {mask_mse_loss.item()}, rgb: {rgb_mse_loss.item()}, lpips: {rgb_lpips_loss.item()}')

        #     merge_img =(input_rgbs + rendered_rgbs)/2
        #     cat_img = torch.cat([rendered_rgbs,input_rgbs],dim=1)
        #     cat_img = torch.cat([cat_img,merge_img],dim=1)
        #     # now cat images of different cameras
        #     cat_img = torch.cat(torch.unbind(cat_img, dim=0), dim=1)
        #     cat_img = cat_img.clip(0,1).detach().squeeze().cpu().numpy()
        
        #     cat_img = img_as_ubyte(cat_img)
     
        #     writer.append_data(cat_img)
    

    # save result
    # if debug:
    #     merge_img =(input_rgbs+best_rendered_img)/2
    #     cat_img = torch.cat([best_rendered_img,input_rgbs],dim=1)
    #     cat_img = torch.cat([cat_img,merge_img],dim=1)
    #     cat_img = torch.cat(torch.unbind(cat_img, dim=0), dim=1)
    #     cat_img = cat_img.clip(0,1).detach().squeeze().cpu().numpy()
    #     cat_img = img_as_ubyte(cat_img)
    #     writer.append_data(cat_img)
    #     writer.close()

    
    # print("best loss:",best_loss)
    # print("best_camera_positions:",best_camera_positions)
   

    return best_camera_positions,best_rendered_img
    




def select_best_camera_position(vertices,faces,vertex_colors,camera_positions_np,input_rgbs_1HW4,
                                renderer,lpips,fov_in_degrees = 30,res=320,batch_size=10,save_for_debug = False,save_path = None):
    device = vertices.device
    i=0
    rgb_lpips_all = torch.ones(len(camera_positions_np)).to(device) * -100
    with torch.no_grad():
        while i<len(camera_positions_np):
            
            if i+batch_size>camera_positions_np.shape[0]:
                batch_size = camera_positions_np.shape[0]-i
            crop_views = np.arange(batch_size)
            rgba_imgs = renderer.render(vertices,vertex_colors*2-1,faces,fov_in_degrees = fov_in_degrees,
                                camera_positions_np=camera_positions_np[i:i+batch_size],res=res,rotate_normal=False,
                                background_color=[1.0,1.0,1.0,0.0],crop_views=crop_views)
            # rgba_imgs = [resize_foreground_torch(rgba_img,1.0) for rgba_img in rgba_imgs] 
            # rgba_imgs = [transforms.Resize((res,res))(rgba_img.unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1).squeeze(0) for rgba_img in rgba_imgs]
            # rgba_imgs = torch.stack(rgba_imgs)
            rendered_rgbs = rgba_imgs[...,:3] * rgba_imgs[...,3:4] + (1 - rgba_imgs[...,3:4]) # B,H,W,3 # rgba to rgb white bg
            rendered_alphas = rgba_imgs[...,3:4]
            

            # calculate lpips
            input_rgbs = input_rgbs_1HW4.repeat(len(rendered_rgbs),1,1,1)
            if input_rgbs.shape[3] == 4: # added 2025.01.24
                input_rgbs = input_rgbs[...,:3] * input_rgbs[...,3:4] + (1 - input_rgbs[...,3:4])
            # input_alphas = input_alphas.repeat(len(elevation_in_degrees),1,1,1)

            rgb_lpips_loss = lpips(rendered_rgbs.permute(0,3,1,2)*2-1, input_rgbs.permute(0,3,1,2)*2-1)#rgb_lpips_loss: [B,1,1,1]  ; input [B, 3, H, W] image in [-1, 1]
            rgb_lpips_loss = rgb_lpips_loss.squeeze() # [B] or sometimes a single value
            rgb_lpips_all[i:i+batch_size] = rgb_lpips_loss
            
            if save_for_debug:
                for j in range(batch_size):
                    img_HW4 = rendered_rgbs[j] 
                    cat = torch.cat([img_HW4,input_rgbs_1HW4[0],(input_rgbs_1HW4[0]+img_HW4)/2],dim=1)
                    img_pil = Image.fromarray((cat*255).detach().cpu().numpy().astype(np.uint8))
                    img_pil.save(f'{save_path}_{i+j}_{rgb_lpips_all[i+j].item()}.png')
            i+=batch_size
    
    best_campos_id = torch.argmin(rgb_lpips_all)
    best_lpips = rgb_lpips_all[best_campos_id]
    # best_elevation = elevation_in_degrees[best_campos_id:best_campos_id+1]
    # best_camera_positions_np = camera_positions_np[best_campos_id:best_campos_id+1]
    if save_for_debug:
        print('best_campos_id',best_campos_id)
    return best_campos_id,best_lpips

def select_best_camera_position_textured_mesh(vertices, faces, face_uvs, triangle_material_ids, texture_list_13HW,
                                              camera_positions_np,input_rgbs_1HW4,
                                renderer,lpips,fov_in_degrees = 30,res=320,batch_size=10,save_for_debug = False,save_path = None):
    device = vertices.device
    i=0
    cameras_positions = torch.tensor(camera_positions_np).to(device).float()
    rgb_lpips_all = torch.ones(len(camera_positions_np)).to(device) * -100
    
    with torch.no_grad():
        while i<len(camera_positions_np):
            
            if i+batch_size>camera_positions_np.shape[0]:
                batch_size = camera_positions_np.shape[0]-i
            crop_views = np.arange(batch_size)
            # rgba_imgs = renderer.render(vertices,vertex_colors*2-1,faces,fov_in_degrees = fov_in_degrees,
            #                     camera_positions_np=camera_positions_np[i:i+batch_size],res=res,rotate_normal=False,
            #                     background_color=[1.0,1.0,1.0,0.0],crop_views=crop_views)
            rgba_imgs = renderer.render_with_texture(
                vertices,faces,
                uvs=None,face_uvs_idx=None,  
                material_list_of_13HW = texture_list_13HW, 
                face_material_idx=triangle_material_ids,face_uvs_F32 = face_uvs, 
                fov_in_degrees = fov_in_degrees,res=res,
                camera_positions=cameras_positions[i:i+batch_size],
                background_color = [1.0,1.0,1.0,0.0],
                crop_views=crop_views)
            
            rendered_rgbs = rgba_imgs[...,:3] * rgba_imgs[...,3:4] + (1 - rgba_imgs[...,3:4]) # B,H,W,3 # rgba to rgb white bg
            rendered_alphas = rgba_imgs[...,3:4]
            

            # calculate lpips
            input_rgbs = input_rgbs_1HW4[...,:3].repeat(len(rendered_rgbs),1,1,1)
            # input_alphas = input_alphas.repeat(len(elevation_in_degrees),1,1,1)

            rgb_lpips_loss = lpips(rendered_rgbs.permute(0,3,1,2)*2-1, input_rgbs.permute(0,3,1,2)*2-1)#rgb_lpips_loss: [B,1,1,1]  ; input [B, 3, H, W] image in [-1, 1]
            rgb_lpips_loss = rgb_lpips_loss.squeeze() # [B] or sometimes a single value
            rgb_lpips_all[i:i+batch_size] = rgb_lpips_loss
            
            if save_for_debug:
                for j in range(batch_size):
                    img_HW4 = rgba_imgs[j] 
  
                    cat = torch.cat([img_HW4,input_rgbs_1HW4[0],(input_rgbs_1HW4[0]+img_HW4)/2],dim=1)
                    img_pil = Image.fromarray((cat*255).detach().cpu().numpy().astype(np.uint8))
                    img_pil.save(f'{save_path}_{i+j}_{rgb_lpips_all[i+j].item()}.png')
            i+=batch_size
    
    best_campos_id = torch.argmin(rgb_lpips_all)
    best_lpips = rgb_lpips_all[best_campos_id]
    # best_elevation = elevation_in_degrees[best_campos_id:best_campos_id+1]
    # best_camera_positions_np = camera_positions_np[best_campos_id:best_campos_id+1]
    if save_for_debug:
        print('best_campos_id',best_campos_id)
    return best_campos_id,best_lpips



  
def select_best_elevation(input_image_HW4, vertices,faces,vertex_colors,
                    fov_in_degrees = 30,res=320,batch_size = 10,device = 'cuda',save_for_debug=False,fine_level=False):
    renderer = GeneralRenderer(device=device)
    lpips = LPIPS(net='vgg').to(device)
    # process input images
    input_image_1HW4 = input_image_HW4.unsqueeze(0)
    input_image_1HW4 = transforms.Resize((res,res))(input_image_1HW4.permute(0,3,1,2)).permute(0,2,3,1)
    input_rgbs_1HW4 = input_image_1HW4[..., :3] * input_image_1HW4[..., 3:4] + (1 - input_image_1HW4[..., 3:4]) # B,H,W,3 # rgba to rgb white bg
    # input_alphas = input_image_1HW4[..., 3:4]
    
    coarse_elevation_step = 3
    elevation_start = -90
    elevation_end = 90
    
    # 1. render all coarse elevation angles
    elevation_in_degrees = np.linspace(-90,90,num=(elevation_end - elevation_start) // coarse_elevation_step +1 )
    camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=elevation_in_degrees,azimuth_in_degrees=[0],radius=4) 

    
    best_elevation_id,best_lpips = select_best_camera_position(vertices,faces,vertex_colors,camera_positions_np,input_rgbs_1HW4,
                                renderer,lpips,fov_in_degrees = fov_in_degrees,res=res,batch_size=batch_size,save_for_debug = save_for_debug,
                                save_path='1coarse_elevation')

    best_elevation = elevation_in_degrees[best_elevation_id:best_elevation_id+1]
    best_camera_positions_np = camera_positions_np[best_elevation_id:best_elevation_id+1]
    
    if fine_level:
        # 2. render detailed elevation angles
        elevation_in_degrees = np.linspace(best_elevation[0]-coarse_elevation_step+1,
                                           best_elevation[0]+coarse_elevation_step-1,
                                           num=coarse_elevation_step*2-1)
        camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=elevation_in_degrees,azimuth_in_degrees=[0],radius=4) 

        
        best_elevation_id,best_lpips = select_best_camera_position(vertices,faces,vertex_colors,camera_positions_np,input_rgbs_1HW4,
                                    renderer,lpips,fov_in_degrees = fov_in_degrees,res=res,batch_size=batch_size,
                                    save_for_debug = save_for_debug,save_path='2fine_elevation')
        best_elevation = elevation_in_degrees[best_elevation_id:best_elevation_id+1]
        best_camera_positions_np = camera_positions_np[best_elevation_id:best_elevation_id+1]
        
        # print('best_elevation',best_elevation)
        # print('best_camera_positions_np',best_camera_positions_np)
        # 3. render coarse azimuth angles
        coarse_azimuth_step = 6
        azimuth_start = -30
        azimuth_end = 30
        azimuth_in_degrees = np.linspace(azimuth_start,azimuth_end,(azimuth_end-azimuth_start)//coarse_azimuth_step+1)

        camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=best_elevation.repeat(len(azimuth_in_degrees)),
                                                              azimuth_in_degrees=azimuth_in_degrees,radius=4) 

        
        best_azimuth_id,best_lpips = select_best_camera_position(vertices,faces,vertex_colors,camera_positions_np,input_rgbs_1HW4,
                                    renderer,lpips,fov_in_degrees = fov_in_degrees,res=res,batch_size=batch_size,
                                    save_for_debug = save_for_debug,save_path='3coarse_azimuth')
        best_azimuth = azimuth_in_degrees[best_azimuth_id:best_azimuth_id+1]
        best_camera_positions_np = camera_positions_np[best_azimuth_id:best_azimuth_id+1]
        
        
        # 4. render find azimuth angles
        azimuth_in_degrees = np.linspace(best_azimuth[0]-coarse_azimuth_step+1,
                                        best_azimuth[0]+coarse_azimuth_step-1,
                                        num=coarse_azimuth_step*2-1)
        camera_positions_np = elevation_azimuth_radius_to_xyz(elevation_in_degrees=best_elevation.repeat(len(azimuth_in_degrees)),
                                                              azimuth_in_degrees=azimuth_in_degrees,radius=4) 

        best_azimuth_id,best_lpips = select_best_camera_position(vertices,faces,vertex_colors,camera_positions_np,input_rgbs_1HW4,
                                    renderer,lpips,fov_in_degrees = fov_in_degrees,res=res,batch_size=batch_size,
                                    save_for_debug = save_for_debug,save_path='4fine_azimuth')
        best_azimuth = azimuth_in_degrees[best_azimuth_id:best_azimuth_id+1]
        best_camera_positions_np = camera_positions_np[best_azimuth_id:best_azimuth_id+1]
    
        # print('best_azimuth',best_azimuth)
        # print('best_camera_positions_np',best_camera_positions_np)
    return best_lpips,best_camera_positions_np
    

def scale_mesh(vertices,scale_factor):
    vertex_max = vertices.max(0)[0]
    vertex_min = vertices.min(0)[0]
    vertices = (vertices - (vertex_max+vertex_min)/2.0) * scale_factor + (vertex_max+vertex_min)/2.0
    return vertices

def select_best_mesh_scale(vertices,faces,vertex_colors,camera_positions_np,input_image_HW4,
                                renderer, scale_factors = [0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6],fov_in_degrees = 30,res=320,save_for_debug = False,save_path = None):
    
    device = vertices.device
    original_vertices = vertices.clone()
    lpips = LPIPS(net='vgg').to(device)

    rgb_lpips_all = torch.ones(len(scale_factors)).to(device) * -100
    
    # process input images
    input_image_1HW4 = input_image_HW4.unsqueeze(0)
    input_image_1HW4 = transforms.Resize((res,res))(input_image_1HW4.permute(0,3,1,2)).permute(0,2,3,1)
    input_rgbs_1HW4 = input_image_1HW4[..., :3] * input_image_1HW4[..., 3:4] + (1 - input_image_1HW4[..., 3:4]) # B,H,W,3 # rgba to rgb white bg
    
    for i in range(len(scale_factors)):
        vertices = scale_mesh(original_vertices,scale_factors[i])
        rgba_imgs = renderer.render(vertices,vertex_colors*2-1,faces,fov_in_degrees = fov_in_degrees,
                            camera_positions_np=camera_positions_np,res=res,rotate_normal=False,
                            background_color=[1.0,1.0,1.0,0.0],crop_views=[0])
        # rgba_imgs = [resize_foreground_torch(rgba_img,1.0) for rgba_img in rgba_imgs] 
        # rgba_imgs = [transforms.Resize((res,res))(rgba_img.unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1).squeeze(0) for rgba_img in rgba_imgs]
        # rgba_imgs = torch.stack(rgba_imgs)
        rendered_rgbs = rgba_imgs[...,:3] * rgba_imgs[...,3:4] + (1 - rgba_imgs[...,3:4]) # B,H,W,3 # rgba to rgb white bg
        rendered_alphas = rgba_imgs[...,3:4]
        

        # calculate lpips
        input_rgbs = input_rgbs_1HW4.repeat(len(rendered_rgbs),1,1,1)
        # input_alphas = input_alphas.repeat(len(elevation_in_degrees),1,1,1)

        rgb_lpips_loss = lpips(rendered_rgbs.permute(0,3,1,2)*2-1, input_rgbs.permute(0,3,1,2)*2-1)#rgb_lpips_loss: [B,1,1,1]  ; input [B, 3, H, W] image in [-1, 1]
        rgb_lpips_loss = rgb_lpips_loss.squeeze() # [B] or sometimes a single value
        rgb_lpips_all[i] = rgb_lpips_loss
        
        if save_for_debug:
            img_HW4 = rendered_rgbs[0] 
            cat = torch.cat([img_HW4,input_rgbs_1HW4[0],(input_rgbs_1HW4[0]+img_HW4)/2],dim=1)
            img_pil = Image.fromarray((cat*255).detach().cpu().numpy().astype(np.uint8))
            img_pil.save(f'{save_path}_{i}_{rgb_lpips_all[i].item()}.png')

    
    best_id = torch.argmin(rgb_lpips_all)
    best_lpips = rgb_lpips_all[best_id]
    # best_elevation = elevation_in_degrees[best_campos_id:best_campos_id+1]
    # best_camera_positions_np = camera_positions_np[best_campos_id:best_campos_id+1]
    if save_for_debug:
        print('best_campos_id',best_id)
    best_scale_factor = scale_factors[best_id]
    scaled_vertices = scale_mesh(original_vertices,best_scale_factor)
    return best_scale_factor,scaled_vertices,best_id,best_lpips