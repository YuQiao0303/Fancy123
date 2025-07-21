import os
import argparse
import kiui.vis
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler,ControlNetModel
import copy
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_unique3d_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video
import kiui
import cv2
# for postprocess by normal
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion
from typing import Tuple


def erode_alpha_torch_NHW4(img_NHW4,kernel_size = 3):
    device = img_NHW4.device
    out_img_NHW4 = []
    for idx, img in enumerate(img_NHW4):
        arr = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 
        alpha = cv2.erode(alpha, np.ones((kernel_size, kernel_size), np.uint8), iterations=3)
        alpha = (alpha * 255).astype(np.uint8)
        img = (np.concatenate([arr[:, :, :3], alpha[:, :, None]], axis=-1))
        img = torch.from_numpy(img).to(device).float() / 255.0
        out_img_NHW4.append(img)
    out_img_NHW4 = torch.stack(out_img_NHW4)
    return out_img_NHW4
# Unique3d/scripts/multiview_infernece.py
def erode_alpha(img_list):
    out_img_list = []
    for idx, img in enumerate(img_list):
        arr = np.array(img)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 1px
        
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=3)
        alpha = (alpha * 255).astype(np.uint8)
        img = Image.fromarray(np.concatenate([arr[:, :, :3], alpha[:, :, None]], axis=-1))
        out_img_list.append(img)
    return out_img_list

# modified from erode_alpha()
def erode_alpha_and_dilate_foreground(img_list,kernel_size=3,erode_alpha=False):
    out_img_list = []
    for idx, img in enumerate(img_list):
        arr = np.array(img)
        forground_mask =(arr[:, :, 3]>127).astype(np.uint8)
  
        # kiui.vis.plot_image(np.concatenate([arr[:, :, :3], forground_mask[:, :, None]* 255], axis=-1))
        # kiui.vis.plot_image(arr[:, :, :3])

        if erode_alpha:
            # erode 1px
            new_alpha = cv2.erode(forground_mask, np.ones((3, 3), np.uint8), iterations=1)
            # new_alpha = cv2.dilate(forground_mask, np.ones((3, 3), np.uint8), iterations=13)
            new_alpha = (new_alpha * 255).astype(np.uint8)
            img = Image.fromarray(np.concatenate([arr[:, :, :3], new_alpha[:, :, None]], axis=-1))
        else:

            # dialte foreground
            background_mask = np.logical_not(forground_mask)
            iterations = 2
            kernel = np.ones((kernel_size,kernel_size), 'uint8')
            rgb = arr[:,:,:3]
            rgba=arr
            rgba[:,:,3] = forground_mask*255
            rgb[background_mask] =0
            for j in range(iterations):
            

                dilated_rgba = cv2.dilate(rgba, kernel,iterations=1)  #
                temp = Image.fromarray(dilated_rgba)
                temp2 = Image.fromarray(rgb)
                rgba = rgba * (1 - background_mask[...,np.newaxis]) + dilated_rgba * background_mask[...,np.newaxis]
                rgba = rgba.astype(np.uint8)
                temp3 = Image.fromarray(rgba)

            img = Image.fromarray(rgba)

        
        # kiui.vis.plot_image(rgba)
        
        out_img_list.append(img)
    return out_img_list




def mesh_gen(mv_outputs, model,infer_config,args,mesh_path,device,
             use_input_img_for_recon,load_exist,IS_FLEXICUBES,subdivide=False):

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale,
                                                  use_input_img_for_recon=use_input_img_for_recon).to(device)
    # kiui.lo(input_cameras) # [cam_num,15]
    chunk_size = 20 if IS_FLEXICUBES else 1

    for idx, sample in enumerate(tqdm(mv_outputs)):
        seed_everything(args.seed)
        name = sample['name']
        print(f'[{idx+1}/{len(mv_outputs)}] Creating {name} ...')

        mv_images = sample['mv_images'].unsqueeze(0).to(device) # 1,cam_num,C,H,W
        mv_images = v2.functional.resize(mv_images, 320, interpolation=3, antialias=True).clamp(0, 1)
        input_image = sample['input_image'].to(device)

        indices = None
        # indices = torch.tensor([2,3,4,6]).long().to(device)
        # indices = torch.tensor([5]).long().to(device)
        if args.view == 4:
    
            # indices = torch.tensor([0, 2, 4, 5]).long().to(device)
            indices = torch.tensor([0, 1,3, 4 ]).long().to(device)
            mv_images = mv_images[:, indices]
            input_cameras = input_cameras[:, indices]

    
        with torch.no_grad():
            
            mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

            if load_exist and os.path.exists(mesh_path_idx):
                continue

            # get triplane
            planes = model.forward_planes(mv_images[:,:,:3,...], input_cameras)

            # get mesh
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args.export_texmap,
                subdivide = subdivide,
                **infer_config,
            )
            vertices, faces, vertex_colors = mesh_out
        


                

        with torch.no_grad():
            # # now save the colors
            # new_vertex_colors =  new_meshes.textures.verts_features_packed()
            
            if args.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
            else:
              
                vertices, faces, vertex_colors = mesh_out
            
        
                # Unique3D coordiante system is different from InstantMesh, so we need to transform it:
                vertices_old = vertices.copy()
                vertices[...,0] = vertices_old[...,1]
                vertices[...,1] = vertices_old[...,2]
                vertices[...,2] = vertices_old[...,0]
                # vertices = vertices.ascontiguousarray()
            
                save_obj(vertices, faces, vertex_colors, mesh_path_idx,flip_normal=False)
            print(f"Mesh saved to {mesh_path_idx}")