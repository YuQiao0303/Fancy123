import datetime
import logging
import imageio
import pytz
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from fancy123.optimize.img_registration import apply_deformation, optimize_deformation, visualize_results


from fancy123.render.general_renderer import GeneralRenderer
from fancy123.unproject.NBF import NBF_prepare_per_kernel_per_view_mv_img_validation
from fancy123.utils.temp_utils import apply_view_color2mesh, complete_vert_colors_by_neighbors, get_fixed_area
from fancy123.utils.utils2d import  sobel_edge_torch

from kiui.lpips import LPIPS
from tqdm import tqdm

def opt_mv_images_all(vertices,faces,mv_imgs_BHW4,camera_positions_B3,
                      input_image_HW4=None,input_cameras_position_13=None,
                  res=320,fov_in_degrees=30,renderer=None,
                  lr = 0.3, epochs=120,scale_factor=8,debug=False,use_alpha=False,
                  background_color=[247.0/255.0,247.0/255.0,247.0/255.0,0], fix_first_view=False,
                  save_animation=False):
    logger = logging.getLogger('logger')
    if save_animation:
        timestr = datetime.datetime.now(pytz.timezone('Etc/GMT-8')).strftime('%Y.%m.%d.%H.%M')

        filename_output = f"./{timestr}_2D_deform.mp4"
        writer = imageio.get_writer(filename_output, mode='I', fps=15) #duration=0.15 fps=18
        
    B,H,W,_ = mv_imgs_BHW4.shape
    device = mv_imgs_BHW4.device
    deformation_field_B2HW = torch.zeros((B, 2, H//scale_factor, W//scale_factor), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deformation_field_B2HW], lr=lr)
    view_weights_B_ones = torch.ones(B).float().to(device)
    
    if input_image_HW4 is not None and input_cameras_position_13 is not None:
        lpips = LPIPS(net='vgg').to(device) # if we also want to optimize the input view to look like the inpug image
    if use_alpha:
        mv_imgs_BHW4[...,:3] = mv_imgs_BHW4[..., :3] * mv_imgs_BHW4[..., 3:] + (1 - mv_imgs_BHW4[..., 3:]) # background 2 white
        background_color =[1.0, 1.0, 1.0, 0] # background 2 white

    
    best_loss = 100.0
    best_deformed_imgs_BHW4 = None
    best_deformation_field_B2HW = None
    
    for epoch in tqdm(range(epochs),desc='deform_2d'):
        optimizer.zero_grad()
        deformed_imgs_B4HW = apply_deformation(mv_imgs_BHW4.permute(0,3,1,2), deformation_field_B2HW,scale_factor=scale_factor)
        deformed_imgs_BHW4 = deformed_imgs_B4HW.permute(0,2,3,1)
        
        
        if fix_first_view:
            deformed_imgs_BHW4[0] = mv_imgs_BHW4[0]
            view_weights_B = view_weights_B_ones.clone() 
            view_weights_B[0] *= 10
            if epoch >80:
                view_weights_B = view_weights_B_ones#.clone() 
        else:
            view_weights_B = view_weights_B_ones.clone() * 1000
            # view_weights_B[epochs%B] = 1 #0.0001  # weird bug: we mean epoch%B here, but somehow that will make the deformation field not update at all

        # # view_weights_B = view_weights_B.detach()
        # # print('view_weights_B',view_weights_B)
        # # import kiui
        # # kiui.lo(deformation_field_B2HW)
        # print('epoch',epoch,epoch%B, type(epoch))
        # print('epochs',epochs,epochs%B,type(epochs))
            
        vertex_colors,final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
                            camera_positions=camera_positions_B3,
                            imgs_BHW4 = deformed_imgs_BHW4,
                            eps = 0.05,res=2048, renderer=renderer,mode='fuse',view_weights_B=view_weights_B)

        rgba_BHW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                    camera_positions_np=camera_positions_B3.detach().cpu().numpy(),
                                    fov_in_degrees=fov_in_degrees,
                                    rotate_normal=False,res=320,background_color=background_color)
        # import kiui
        # kiui.vis.plot_image(rgba_BHW4[0,...,:3])
        if use_alpha:
            mse_loss = (rgba_BHW4 - deformed_imgs_BHW4).pow(2).mean()
            img_foreground_mask = (deformed_imgs_BHW4[...,3]>0.5)
            rendered_foreground_mask = (rgba_BHW4[...,3]>0.5)
            
        else:
            mse_loss = (rgba_BHW4[...,:3] - deformed_imgs_BHW4[...,:3]).pow(2).mean()
            
            thresh = 5.0/255.0
            img_pseudo_background_mask = (deformed_imgs_BHW4[...,:3] - torch.tensor(background_color[:3]).to(device)).abs().max(-1)[0] < thresh
            img_pseudo_foreground_mask = ~img_pseudo_background_mask
            img_foreground_mask = img_pseudo_foreground_mask
            
            rendered_pseudo_background_mask = (rgba_BHW4[...,:3] - torch.tensor(background_color[:3]).to(device)).abs().max(-1)[0] < thresh
            rendered_pseudo_foreground_mask = ~rendered_pseudo_background_mask
            rendered_foreground_mask = rendered_pseudo_foreground_mask
        
        # get mask loss and foreground_rgb_loss    
        foreground_mask =  rendered_foreground_mask | img_foreground_mask 
        foreground_rgb_loss = F.mse_loss(rgba_BHW4[...,:3][foreground_mask], deformed_imgs_BHW4[...,:3][foreground_mask])
        mask_loss = F.mse_loss(rendered_foreground_mask.float(), img_foreground_mask.float())
        # mse_loss = mask_loss + 1.0 * foreground_rgb_loss
        
        loss = mse_loss 
        
        if input_image_HW4 is not None and input_cameras_position_13 is not None:
            input_rgba_1HW4 =  renderer.render(vertices,vertex_colors*2-1,faces,
                                    camera_positions_np=input_cameras_position_13.detach().cpu().numpy(),
                                    fov_in_degrees=fov_in_degrees,
                                    rotate_normal=False,res=320,background_color=background_color,crop_views=[0])  
            # if epoch > epochs - B:
            #     import kiui
            #     kiui.vis.plot_image(input_rgba_1HW4[0,:,:,:3]) 
            if use_alpha:
                input_image_13HW = input_image_HW4[...,:3].unsqueeze(0).permute(0,3,1,2)
                input_rgba_13HW = input_rgba_1HW4[...,:3].permute(0,3,1,2)
                input_view_loss = lpips(input_rgba_13HW*2-1, 
                                        input_image_13HW*2-1).mean()#rgb_lpips_loss: [B,1,1,1]  ; input [B, 3, H, W] image in [-1, 1]
           
            else:
                input_view_loss = lpips(input_rgba_1HW4[...,:3].permute(0,3,1,2)*2-1, input_image_HW4[...,:3].unsqueeze(0).permute(0,3,1,2)*2-1).mean()#rgb_lpips_loss: [B,1,1,1]  ; input [B, 3, H, W] image in [-1, 1]
            loss = loss + 1e5 * input_view_loss

        # Smoothness loss for deformation field
        smoothness_loss = torch.mean(torch.abs(deformation_field_B2HW[:, :, :-1, :] - deformation_field_B2HW[:, :, 1:, :])) + \
                            torch.mean(torch.abs(deformation_field_B2HW[:, :, :, :-1] - deformation_field_B2HW[:, :, :, 1:]))

        # small deformation loss
        smalll_deform_loss = deformation_field_B2HW.pow(2).mean()
        loss = loss + 0.001 * smoothness_loss #+ 0.001 * smalll_deform_loss

        if loss<best_loss:
            best_loss = loss.detach().clone()
            best_deformed_imgs_BHW4 = deformed_imgs_BHW4.detach().clone()
            best_deformation_field_B2HW = deformation_field_B2HW.detach().clone()
            best_rendered_img = rgba_BHW4.detach().clone()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, \
                  mse_loss: {mse_loss.item()}, \
                  mask_loss: {mask_loss.item()}, \
                  foreground_rgb_loss: {foreground_rgb_loss.item()}, \
                  smoothness_loss: {smoothness_loss.item()}, \
                  small_deform_loss: {smalll_deform_loss.item()}\
                  ")
        getting_better = loss < best_loss
        if best_loss < loss:
            deformed_imgs_BHW4 = best_deformed_imgs_BHW4
            deformation_field_B2HW = best_deformation_field_B2HW
            view_weights_B = view_weights_B_ones
            vertex_colors,final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
                                camera_positions=camera_positions_B3,
                                imgs_BHW4 = deformed_imgs_BHW4,
                                eps = 0.05,res=2048, renderer=renderer,mode='fuse',view_weights_B=view_weights_B)

            rgba_BHW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                        camera_positions_np=camera_positions_B3.detach().cpu().numpy(),
                                        fov_in_degrees=fov_in_degrees,
                                        rotate_normal=False,res=320,background_color=background_color)
            
            if use_alpha:
                img_foreground_mask = (deformed_imgs_BHW4[...,3]>0.5)
                rendered_foreground_mask = (rgba_BHW4[...,3]>0.5)
                
            else:
                # mse_loss = (rgba_BHW4[...,:3] - deformed_imgs_BHW4[...,:3]).pow(2).mean()
                
                thresh = 5.0/255.0
                img_pseudo_background_mask = (deformed_imgs_BHW4[...,:3] - torch.tensor(background_color[:3]).to(device)).abs().max(-1)[0] < thresh
                img_pseudo_foreground_mask = ~img_pseudo_background_mask
                img_foreground_mask = img_pseudo_foreground_mask
                
                rendered_pseudo_background_mask = (rgba_BHW4[...,:3] - torch.tensor(background_color[:3]).to(device)).abs().max(-1)[0] < thresh
                rendered_pseudo_foreground_mask = ~rendered_pseudo_background_mask
                rendered_foreground_mask = rendered_pseudo_foreground_mask
        # if save_gif and (epoch<30 or (epoch<60 and epoch%5==0) or epoch%10==0):
        if save_animation and ((epoch%2==0 and epoch<50) or (epoch%5==0 and epoch>=50)): # cup
        # if save_gif and (epoch<20) : # gameon lunchbag
            from skimage import img_as_ubyte
            front_view_camera_positions_np = np.array([[0,0,4]])
            if True: #epoch==0 or getting_better:
                
                # vertex_colors has already been applied with best iteration's view colors, now we need to do NBF and complete
                NBF = False
                if NBF:
                    border_area_masks_NHW,border_edge_masks_NHW = NBF_prepare_per_kernel_per_view_mv_img_validation(vertices,faces,
                                            camera_positions_B3,H,
                                            debug=False,mv_imgs_NHWC=deformed_imgs_BHW4,
                                            vertex_colors = vertex_colors,
                                            save_img_path=None)
                    new_alpha = deformed_imgs_BHW4[:, :, :, 3] * (1-border_area_masks_NHW.float())
                    imgs_to_project_BHW4 = torch.cat([deformed_imgs_BHW4[...,:3],new_alpha.unsqueeze(-1)],dim=-1)
                    
                    vertex_colors,final_visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors=None,
                            camera_positions=camera_positions_B3,
                            imgs_BHW4 = imgs_to_project_BHW4,
                            eps = 0.05,res=2048, renderer=renderer,mode='fuse',
                            per_view_shrinked_per_vert_visibility=None)
                            # per_view_shrinked_per_vert_visibility=None)
                        
                
                complete_vertex_colors = complete_vert_colors_by_neighbors(vertices.detach(), faces.detach(),
                                    vertex_colors.detach(),  final_visible_vertex_mask_V.detach())
            
            temp_img =  renderer.render(vertices,complete_vertex_colors*2-1,faces,
                                            camera_positions_np=front_view_camera_positions_np,
                                            fov_in_degrees=fov_in_degrees,
                                            rotate_normal=False,res=320,background_color=background_color,
                                            crop_views=np.arange(len(front_view_camera_positions_np))
                                            )
            # merge_img =(deformed_imgs_BHW4 + temp_img)/2
            # diff_img = (deformed_imgs_BHW4-temp_img).abs()
            # margin = torch.full((H, 10, 4), 1.0).float().to(device)
           
            # cat_img = torch.cat([temp_img[0],margin,merge_img[0],margin,diff_img[0]],dim=1)
          
            cat_img = temp_img[0]
            cat_img = cat_img.clip(0,1).detach().squeeze().cpu().numpy()
            
            
            cat_img = img_as_ubyte(cat_img[...,:3])
            writer.append_data(cat_img)
    if save_animation:
        # merge_img =(deformed_imgs_BHW4+best_rendered_img)/2
        # diff_img = (deformed_imgs_BHW4-best_rendered_img).abs()
        # margin = torch.full((H, 10, 4), 1.0).float().to(device)
        # cat_img = torch.cat([best_rendered_img[0],margin,merge_img[0],margin,diff_img[0]],dim=1)
        
        cat_img = temp_img[0]
        cat_img = cat_img.clip(0,1).detach().squeeze().cpu().numpy()
        cat_img = img_as_ubyte(cat_img[...,:3])
        writer.append_data(cat_img)
        writer.close()

            
    return deformed_imgs_BHW4,deformation_field_B2HW,vertex_colors,final_visible_vertex_mask_V,rendered_foreground_mask,img_foreground_mask







def render_visible_area(vertices,faces,mv_imgs_BHW4,camera_positions_B3,
                        res=320,fov_in_degrees=30,renderer=None,eps=0):
    device = vertices.device
    if renderer is None:
        renderer = GeneralRenderer(device=vertices.device)
    B,H,W,_ = mv_imgs_BHW4.shape # B: number of cameras, batch size
    vertex_colors = torch.zeros_like(vertices).to(device)
    
    rendered_rgbas_BBHW4 = torch.zeros((B,B,H,W,4)).to(device)
    for i in range(B):
        vertex_colors = torch.zeros_like(vertices).to(device)
        vertex_colors,visible_vertex_mask_V = apply_view_color2mesh(vertices,faces,vertex_colors,
                                        camera_positions_B3[i].unsqueeze(0),
                                        mv_imgs_BHW4[i].unsqueeze(0),
                                        eps=eps)
 

        face_fixed_mask = visible_vertex_mask_V[faces].min(-1)[0].bool()
        vis=False
        if vis:
            from fancy123.utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
            
            vis_actors_vtk([
                get_colorful_pc_actor_vtk(vertices.detach().cpu().numpy(),vertex_colors.detach().cpu().numpy(),opacity=1),
            ])
            
        

        for j in range(B):
            if i==j:
            # if False:
                rendered_rgbas_BBHW4[i,j] = mv_imgs_BHW4[i]
            else:
                rgba_ij_1HW4 = renderer.render(vertices,vertex_colors*2-1,faces,
                                camera_positions_np=camera_positions_B3[j].unsqueeze(0).detach().cpu().numpy(),
                                fov_in_degrees=fov_in_degrees,
                                rotate_normal=False)
                # rendered_rgbas_BBHW4[i,j] = rgba_ij_1HW4[0]
                
                pixel_fixed_mask_HW, pixel_fixed_dilated_mask_HW,per_pixel_face_ids_HW = get_fixed_area(vertices,faces, 
                    face_fixed_mask,camera_positions_B3[j].unsqueeze(0).detach().cpu().numpy(),renderer,res)
                temp1 = rgba_ij_1HW4[0,:,:,:3]
                temp1 = temp1.clip(0,1)
                temp2 = pixel_fixed_mask_HW.float().unsqueeze(-1)
                rendered_rgbas_BBHW4[i,j] = torch.cat([temp1,temp2],dim=2)
            
            
    return rendered_rgbas_BBHW4

def save_BxB_rgba_img(rendered_rgbas_BBHW4,save_path):
    B,_,H,W,_ = rendered_rgbas_BBHW4.shape
    rendered_rgbas_np = rendered_rgbas_BBHW4.detach().cpu().numpy()
    big_image = rendered_rgbas_np.reshape(B, B, H, W, 4)
    big_image = big_image.transpose(0, 2, 1, 3, 4)

    big_image = big_image.reshape(B * H, B * W, 4)
    big_image[big_image[..., 3] == 0] = np.array([1,1,1,0])
    big_image_pil = Image.fromarray((big_image*255).astype(np.uint8)).convert('RGB')
    big_image_pil.save(save_path)



def opt_mv_images_pairwise(vertices,faces,mv_imgs_BHW4,camera_positions_B3,
                  res=320,fov_in_degrees=30,renderer=None,
                  lr = 0.3, epochs=100,scale_factor=8,debug=False):
    device = vertices.device
    B, H,W,_ = mv_imgs_BHW4.size()
    rendered_rgbas_BBHW4 = render_visible_area(vertices,faces,mv_imgs_BHW4,camera_positions_B3,
                        res=res,fov_in_degrees=fov_in_degrees,renderer=renderer)
    N = B * (B-1)
    n = 0
    fixed_imgs_NHW4 = torch.zeros((N, H, W,4), device=device)
    moving_imgs_NHW4 = torch.zeros((N, H, W,4), device=device)
    for i in range(B):
        for j in range(B):
            if i==j:
                continue
            moving_imgs_NHW4[n] = mv_imgs_BHW4[i]
            fixed_imgs_NHW4[n] = rendered_rgbas_BBHW4[j,i]
            save = True
            if save:
                moving_img = (moving_imgs_NHW4[n]*255.0).detach().cpu().numpy().astype(np.uint8)
                moving_img_pil = Image.fromarray(moving_img)
                moving_img_pil.save(f'temp/ikun_register/moving_img{n}.png')
                
                fixed_img = (fixed_imgs_NHW4[n]*255.0).detach().cpu().numpy().astype(np.uint8)
                fixed_img_pil = Image.fromarray(fixed_img)
                fixed_img_pil.save(f'temp/ikun_register/fixed_img{n}.png')
            n = n+1
            
        
    fixed_imgs_NHW4 = fixed_imgs_NHW4.detach().requires_grad_(False)
    moving_imgs_NHW4 = moving_imgs_NHW4.detach().requires_grad_(False)
    # optimize_deformation(fixed_imgs_NHW4.permute(0,3,1,2), moving_imgs_NHW4.permute(0,3,1,2), num_epochs=300, 

    edge_fixed_imgs_N3HW = sobel_edge_torch(fixed_imgs_NHW4[..., :3].permute(0,3,1,2)).detach().requires_grad_(False)# N,H,W,3
    edge_moving_imgs_N3HW = sobel_edge_torch(moving_imgs_NHW4[..., :3].permute(0,3,1,2)).detach().requires_grad_(False) # N,H,W,3

    deformation_field_N2HW = torch.zeros((B*(B-1), 2, H//scale_factor, W//scale_factor), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deformation_field_N2HW], lr=lr)

    # visualize_results(fixed_imgs_NHW4.permute(0,3,1,2), moving_imgs_NHW4.permute(0,3,1,2), 
    #                 moving_imgs_NHW4.permute(0,3,1,2), deformation_field_N2HW, output_path='.')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        deformed_imgs_N4HW = apply_deformation(moving_imgs_NHW4.permute(0,3,1,2), deformation_field_N2HW,scale_factor=scale_factor)
        deformed_imgs_NHW4 = deformed_imgs_N4HW.permute(0,2,3,1)
        deformed_imgs_NHW4[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([0.0, 0.0, 0.0, 0]).float().to(device) # set background to black
        deformed_imgs_N4HW = deformed_imgs_NHW4.permute(0,3,1,2)
        
       
        fixed_imgs_NHW4[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([0.0, 0.0, 0.0, 0]).float().to(device) # set background to black
        
        # mse loss
        mask = fixed_imgs_NHW4[...,3] > 0
        # mse_loss = F.mse_loss(deformed_imgs_NHW4[...,:3][mask], fixed_imgs_NHW4[...,:3][mask])
        mse_loss = F.mse_loss(deformed_imgs_NHW4, fixed_imgs_NHW4)
        
        # edge loss
        
        # deforemd_edge_N3HW = apply_deformation(edge_moving_imgs_N3HW, deformation_field_N2HW,scale_factor=scale_factor)
        # deforemd_edge_NHW3 = deforemd_edge_N3HW.permute(0,2,3,1)
        # deforemd_edge_NHW3[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([0.0, 0.0, 0.0]).float().to(device) # set background to black
        # deforemd_edge_N3HW = deforemd_edge_NHW3.permute(0,3,1,2)
        
        # edge_loss = (deforemd_edge_N3HW - edge_fixed_imgs_N3HW).pow(2).mean()
       

        # Smoothness loss for deformation field
        smoothness_loss = torch.mean(torch.abs(deformation_field_N2HW[:, :, :-1, :] - deformation_field_N2HW[:, :, 1:, :])) + \
                            torch.mean(torch.abs(deformation_field_N2HW[:, :, :, :-1] - deformation_field_N2HW[:, :, :, 1:]))

        loss = mse_loss + 0.001 * smoothness_loss #+ 0.1 * edge_loss

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # print(f"Epoch {epoch}, mse_loss: {mse_loss.item()}, smoothness_loss: {smoothness_loss.item()},edge_loss: {edge_loss.item()}")
            print(f"Epoch {epoch}, mse_loss: {mse_loss.item()}, smoothness_loss: {smoothness_loss.item()}")
    
    # save visualization results

    moving_imgs_NHW4_clone = moving_imgs_NHW4.clone()
    moving_imgs_NHW4_clone[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([1.0, 1.0, 1.0, 1]).float().to(device) # set background to white
    
    deformed_imgs_NHW4 = deformed_imgs_N4HW.permute(0,2,3,1)
    deformed_imgs_NHW4[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([1.0, 1.0, 1.0, 1]).float().to(device) # set background to white
    deformed_imgs_N4HW = deformed_imgs_NHW4.permute(0,3,1,2)
    
    fixed_imgs_NHW4[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([1.0, 1.0, 1.0, 1]).float().to(device) # set background to white
    visualize_results(fixed_imgs_NHW4.permute(0,3,1,2), moving_imgs_NHW4_clone.permute(0,3,1,2), 
                      deformed_imgs_N4HW, deformation_field_N2HW, output_path='.')
    
    deformed_imgs_N4HW = apply_deformation(moving_imgs_NHW4.permute(0,3,1,2), deformation_field_N2HW,scale_factor=scale_factor)
    img = deformed_imgs_N4HW.permute(0,2,3,1) # NHW4
    img = img.reshape(-1,W,4)
    img = img.detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save('result2.png')
