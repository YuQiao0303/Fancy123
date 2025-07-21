import sys

from fancy123.optimize.mesh_optimize import laplace_regularizer_const
sys.path.append('...')
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

from torchvision.transforms import transforms


def deform_mesh(vertices, deformation_field):
    '''
    Differentiable deform a mesh according to the deformation field.
    vertices: [V, 3]. 
    deformation_field: [H, W, D, 3]
    '''
    
    offsets = trilinear_interpolation(vertices,deformation_field)
    deformed_vertices = vertices + offsets
    return deformed_vertices
    

def trilinear_interpolation(input_points, feature_volume,trans_vertices_to01=True):
    # 获取特征体的尺寸
    X, Y, Z, channels = feature_volume.shape
    
    if trans_vertices_to01:
        # Scale vertices to the range [0, 1] for each dimension
        vertices_min = input_points.min(dim=0)[0]
        vertices_max = input_points.max(dim=0)[0]
        vertices_centered = input_points - vertices_min
        sparse_points = vertices_centered / (vertices_max - vertices_min) # from 0 to 1
        # vertices_normalized = vertices_normalized -0.5 # from -0.5 to 0.5
        # vertices_normalized = vertices_normalized*0.8 # from -0.4 to 0.4
        # vertices_normalized = vertices_normalized + 0.5 # from 0.1 to 0.9
    
    # 稀疏点的坐标
    x, y, z = sparse_points[:, 0], sparse_points[:, 1], sparse_points[:, 2]
    x_norm = x * (X-1)
    y_norm = y * (Y-1)
    z_norm = z * (Z-1)
    
    # norm_points = torch.stack([x_norm, y_norm, z_norm], dim=1)
    # print('norm_points',norm_points)


    # 计算插值的权重
    x_low = torch.floor(x_norm).long()
    y_low = torch.floor(y_norm).long()
    z_low = torch.floor(z_norm).long()
    x_high = x_low + 1
    y_high = y_low + 1
    z_high = z_low + 1

    # 处理边界情况
    x_high = torch.clamp(x_high, 0, X - 1)
    y_high = torch.clamp(y_high, 0, Y - 1)
    z_high = torch.clamp(z_high, 0, Z - 1)

    # 计算权重
    wx_high = x_norm - x_low.float() # [N]
    wy_high = y_norm - y_low.float()
    wz_high = z_norm - z_low.float()
    wx_low = 1 - wx_high
    wy_low = 1 - wy_high
    wz_low = 1 - wz_high

    # 进行三线性插值
    value_000 = feature_volume[x_low, y_low, z_low] # [N, value]
    value_001 = feature_volume[x_low, y_low, z_high]
    value_010 = feature_volume[x_low, y_high, z_low]
    value_011 = feature_volume[x_low, y_high, z_high]
    value_100 = feature_volume[x_high, y_low, z_low]
    value_101 = feature_volume[x_high, y_low, z_high]
    value_110 = feature_volume[x_high, y_high, z_low]
    value_111 = feature_volume[x_high, y_high, z_high]


    value_00 = wz_low.view(-1, 1) * value_000 + wz_high.view(-1, 1) * value_001 # # [N, value_]
    value_01 = wz_low.view(-1, 1) * value_010 + wz_high.view(-1, 1) * value_011
    value_10 = wz_low.view(-1, 1) * value_100 + wz_high.view(-1, 1) * value_101
    value_11 = wz_low.view(-1, 1) * value_110 + wz_high.view(-1, 1) * value_111

    value_0 = wy_low.view(-1, 1) * value_00 + wy_high.view(-1, 1) * value_01
    value_1 = wy_low.view(-1, 1) * value_10 + wy_high.view(-1, 1) * value_11

    interpolated_value = wx_low.view(-1, 1) * value_0 + wx_high.view(-1, 1) * value_1

    return interpolated_value



def optimize_mesh_deformation(vertices,faces,vertex_colors,fixed_imgs, camera_position_np, crop=True,
                         lr=1e-4,epochs=800, deformation_field_resolution=20,save_path = None,lap_weight = 0,renderer=None):

    fixed_imgs = transforms.Resize((320,320))(fixed_imgs.permute(0,3,1,2)).permute(0,2,3,1)

    B, C, H, W = fixed_imgs.size()

    

    fixed_imgs[...,:3] = fixed_imgs[..., :3] * fixed_imgs[..., 3:] + (1 - fixed_imgs[..., 3:]) # background to white
    
    X=Y=Z = deformation_field_resolution
    deformation_field = torch.zeros((X,Y,Z,3), requires_grad=True, device=fixed_imgs.device)
    optimizer = torch.optim.Adam([deformation_field], lr=lr)
    
    if crop:
        crop_views = np.arange(len(camera_position_np)) # all views are to be cropped
    else:
        crop_views = None
    
    best_vertices = vertices.detach().clone()
    best_loss = 100.0
    best_rendered_img = None
    for epoch in range(epochs):
        optimizer.zero_grad()

        deformed_vertices = deform_mesh(vertices.detach(), deformation_field)
        rendered_imgs = renderer.render(deformed_vertices,vertex_colors.detach()*2-1,
                                        faces.detach(),camera_positions_np=camera_position_np,rotate_normal=False,crop_views=crop_views)
        rendered_imgs_white_bg = rendered_imgs.clone()
        rendered_imgs_white_bg[...,:3] = rendered_imgs[..., :3] * rendered_imgs[..., 3:] + (1 - rendered_imgs[..., 3:]) # background to white
        
        mse_loss = F.mse_loss(rendered_imgs_white_bg, fixed_imgs)
        
        

        # Smoothness loss for deformation field
        smoothness_loss =   torch.mean(torch.abs(deformation_field[:, :, :-1, :] - deformation_field[:, :, 1:, :])) + \
                            torch.mean(torch.abs(deformation_field[:, :-1, :, :] - deformation_field[:, 1:, :, :])) + \
                            torch.mean(torch.abs(deformation_field[:-1, :, :, :] - deformation_field[1:, :, :, :]))

        loss = mse_loss + 0.001 * smoothness_loss
        
        # laplacian smooth loss
        lap_smooth_Loss = laplace_regularizer_const((deformed_vertices),faces).mean()
        loss = loss + lap_weight*lap_smooth_Loss

        
        if loss < best_loss:
            best_loss = loss.detach().clone()
            best_vertices = (deformed_vertices).detach().clone()
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
                
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, mse_loss: {mse_loss.item()}, smoothness_loss: {smoothness_loss.item()}")

    # print(f"Optimization complete for image pair ({fixed_path}, {moving_path}) with final loss: {loss.item()}")


   
    return best_vertices



def test_deform_mesh():
    name = 'cute_tiger'
    device = 'cuda'
    
    # load mesh
    input_path = 'outputs/_zero123/instant-mesh-large'
    mesh_path_idx = os.path.join(input_path,'fancy123_meshes',name,f'deformed.obj')
    print(os.path.exists(mesh_path_idx),mesh_path_idx)
    vertices, faces, vertex_colors  = loadobj_color(mesh_path_idx,device=device)

    # deform mesh
    deformation_field = torch.ones((20,20,20,3), requires_grad=True, device=device)
    deformation_field = torch.randn((20,20,20,3), requires_grad=True, device=device)*0.02
    vertices = deform_mesh(vertices,deformation_field)
    
    # save
    save_obj(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), vertex_colors.detach().cpu().numpy(), 
             f'{name}_mesh_deformed.obj',flip_normal=False)
    
def test_trilinear_interpolation():
    sparse_points = torch.randn((4, 3))
    sparse_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])*0.5
    feature_volume = torch.randn((3, 3, 3,1))
    
    interpolated_values = trilinear_interpolation(sparse_points, feature_volume)
    print('feature_volume')
    print(feature_volume)
    # print('sparse_points')
    # print(sparse_points)
    print('interpolated_values')
    print(interpolated_values)
    
    print(feature_volume[0,0,0])
    print(feature_volume[1,0,0])
    print(feature_volume[0,1,0])
    print(feature_volume[0,0,1])
    

if __name__ == '__main__':

    test_deform_mesh()
    
    
