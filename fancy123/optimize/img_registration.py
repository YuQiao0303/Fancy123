import sys
sys.path.append('..')
sys.path.append('.')
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import torchvision.utils as vutils

from fancy123.render.general_renderer import GeneralRenderer
from fancy123.utils.temp_utils import complete_vert_colors_by_neighbors,apply_view_color2mesh
from src.utils.camera_util import elevation_azimuth_radius_to_xyz
from src.utils.mesh_util import loadobj_color, save_obj
# from unique3d.scripts.project_mesh import get_cameras_list_azim_ele, multiview_color_projection

def load_image(image_path):
    """Load an image and convert it to a tensor."""
    image = Image.open(image_path).convert('RGBA')
    image = np.array(image, dtype=np.float32) 
    image[image[..., 3] == 0] = np.array([255, 255, 255, 0]) # set background to white
    image = image / 255.0
    image = torch.tensor(image).permute(2, 0, 1)  # [C, H, W]
    return image


def apply_deformation(moving_img, deformation_field, scale_factor=8):
    B, C, H, W = moving_img.size()
    # Dimension of the low-resolution deformation field
    # low_H, low_W = H // scale_factor, W // scale_factor

    # Create the original high-resolution grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=moving_img.device), 
                                    torch.arange(W, device=moving_img.device), indexing='ij')
    high_res_grid = torch.stack((grid_x, grid_y), dim=-1).float()  # Combined into [H, W, 2]
    high_res_grid = high_res_grid.unsqueeze(0).expand(B, -1, -1, -1)  # Expand batch size

    # Interpolate the low-resolution deformation field to high-resolution
    deformation_field = F.interpolate(deformation_field, size=(H, W), mode='bilinear', align_corners=True)
    deformation_field = deformation_field.permute(0, 2, 3, 1)  # Rearrange to [B, H, W, 2]

    # Apply the deformation field to the high-resolution grid
    deformed_grid = high_res_grid + deformation_field

    # Normalize the grid to [-1, 1]
    deformed_grid[..., 0] = 2.0 * deformed_grid[..., 0] / (W - 1) - 1.0
    deformed_grid[..., 1] = 2.0 * deformed_grid[..., 1] / (H - 1) - 1.0

    # Perform resampling using grid_sample
    deformed_img = F.grid_sample(moving_img, deformed_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return deformed_img




def visualize_results(fixed_img_B4HW, moving_img_B4HW, deformed_img_B4HW, deformation_field_B4HW, output_path):
    B,_,H,W = fixed_img_B4HW.shape
    # 计算 RGB 通道的差值图像（忽略 alpha 通道用于可视化）
    diff_original_B3HW = torch.abs(moving_img_B4HW[:, :3] - fixed_img_B4HW[:, :3])
    diff_deformed_B3HW = torch.abs(deformed_img_B4HW[:, :3] - fixed_img_B4HW[:, :3])

    # 生成变形场示意图
    chessboard_img_1HW = generate_chessboard_img(H, W,margin=10).to(moving_img_B4HW.device)
    deformation_field_img_B4HW = generate_deformation_field_images(deformation_field_B4HW,chessboard_img_1HW)


    # 将批次中的每个类别的图像拼接起来
    # 假设 B, 4, H, W 格式，并仅选择 RGB 通道用于差值图像
    all_images_B3HW = torch.cat([
        moving_img_B4HW[:, :3],  # 原始移动图像（RGB）
        diff_original_B3HW,      # 变形前的差值（RGB）
        deformed_img_B4HW[:, :3], # 变形后的移动图像（RGB）
        diff_deformed_B3HW,      # 变形后的差值（RGB）
        fixed_img_B4HW[:, :3],   # 固定图像（RGB）
        deformation_field_img_B4HW[:, :3]  # 变形场示意图（灰度）
    ], dim=3)  # 沿宽度拼接

    # 使用 torchvision 创建网格并保存图像
    result_grid = vutils.make_grid(all_images_B3HW, nrow=1, padding=2, normalize=True, scale_each=True)

    # 保存网格为图像文件
 
    vutils.save_image(result_grid, os.path.join(output_path, 'result.png'))



def generate_chessboard_img(H, W, margin):
    """
    Generates a black and white chessboard pattern as a PyTorch tensor.

    Parameters:
    - H (int): Height of the generated image.
    - W (int): Width of the generated image.
    - margin (int): The interval between black and white squares.

    Returns:
    - torch.Tensor: PyTorch tensor of shape (1, H, W) representing the chessboard image.
    """

    # Calculate the number of squares that fit into the dimensions
    num_squares_h = H // (2 * margin)
    num_squares_w = W // (2 * margin)

    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # Generate the chessboard pattern using modulo operations
    pattern = ((y // (2 * margin)) % 2 + (x // (2 * margin)) % 2) % 2

    # Convert the pattern to float and normalize it to be between 0 and 1
    pattern = pattern.float() #/ 255.0

    # Reshape the tensor to have the required dimensions
    chessboard_tensor = pattern.unsqueeze(0)

    chessboard_tensor[...,:3] *= 0.8 # grey grid instead of pure white
    return chessboard_tensor

def generate_deformation_field_images(deformation_field_B4HW,chessboard_img_1HW):
    B, _, H, W = deformation_field_B4HW.size()
    chessboard_imgs_B4HW = chessboard_img_1HW.unsqueeze(0).repeat(B, 4, 1, 1)
    deformed_grid_img = apply_deformation(chessboard_imgs_B4HW, deformation_field_B4HW)
    return deformed_grid_img



def optimize_deformation(fixed_imgs, moving_imgs, num_epochs=300, 
                         learning_rate=0.3, scale_factor=16,output_path='output'):
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

  

    B, C, H, W = fixed_imgs.size()
    deformation_field = torch.zeros((B, 2, H//scale_factor, W//scale_factor), requires_grad=True, device=fixed_imgs.device)
    optimizer = torch.optim.Adam([deformation_field], lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        deformed_imgs = apply_deformation(moving_imgs, deformation_field)

        mse_loss = F.mse_loss(deformed_imgs, fixed_imgs)

        # Smoothness loss for deformation field
        smoothness_loss = torch.mean(torch.abs(deformation_field[:, :, :-1, :] - deformation_field[:, :, 1:, :])) + \
                            torch.mean(torch.abs(deformation_field[:, :, :, :-1] - deformation_field[:, :, :, 1:]))

        loss = mse_loss + 0.001 * smoothness_loss

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, mse_loss: {mse_loss.item()}, smoothness_loss: {smoothness_loss.item()}")

    # print(f"Optimization complete for image pair ({fixed_path}, {moving_path}) with final loss: {loss.item()}")

    # Save visualizations 
    if output_path is not None:
        visualize_results(fixed_imgs, moving_imgs, deformed_imgs, deformation_field, output_path)
        # for index in range(len(fixed_imgs)):
        #     import kiui
        #     kiui.lo(fixed_imgs)
        #     visualize_results0(fixed_imgs[index].unsqueeze(0), moving_imgs[index].unsqueeze(0), 
        #                       deformed_img[index].unsqueeze(0), 
        #                       deformation_field[index].unsqueeze(0), output_path, index)
    return deformed_imgs




def opt_mv_images_pairwise( lr = 0.3, epochs=100,scale_factor=8,debug=False):
    device = 'cuda'
    B=6
    H=W=320
    N = B * (B-1)
    n = 0
    fixed_imgs_NHW4 = torch.zeros((N, H, W,4), device=device)
    moving_imgs_NHW4 = torch.zeros((N, H, W,4), device=device)
    for n in range(N):

        moving_img_pil = Image.open(f'temp/ikun_register/moving_img{n}.png')
        moving_imgs_NHW4[n] = torch.tensor(np.array(moving_img_pil), dtype=torch.float32) / 255.0
        fixed_img_pil = Image.open(f'temp/ikun_register/fixed_img{n}.png')
        fixed_imgs_NHW4[n] = torch.tensor(np.array(fixed_img_pil), dtype=torch.float32) / 255.0
   
    
    fixed_imgs_NHW4 = fixed_imgs_NHW4.detach().requires_grad_(False)
    moving_imgs_NHW4 = moving_imgs_NHW4.detach().requires_grad_(False)
    

    
    deformation_field_N2HW = torch.zeros((N, 2, H//scale_factor, W//scale_factor), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deformation_field_N2HW], lr=lr)

    
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

        # Smoothness loss for deformation field
        smoothness_loss = torch.mean(torch.abs(deformation_field_N2HW[:, :, :-1, :] - deformation_field_N2HW[:, :, 1:, :])) + \
                            torch.mean(torch.abs(deformation_field_N2HW[:, :, :, :-1] - deformation_field_N2HW[:, :, :, 1:]))

        loss = mse_loss + 0.001 * smoothness_loss

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, mse_loss: {mse_loss.item()}, smoothness_loss: {smoothness_loss.item()}")
    
    # save visualization results
    moving_imgs_NHW4_masked = moving_imgs_NHW4.clone()
    moving_imgs_NHW4_masked[fixed_imgs_NHW4[...,3] == 0] = torch.tensor([0.0, 0.0, 0.0, 0]).float().to(device) # set background to black
    visualize_results(fixed_imgs_NHW4.permute(0,3,1,2), moving_imgs_NHW4_masked.permute(0,3,1,2), 
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
        
        
