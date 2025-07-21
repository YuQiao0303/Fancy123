import os
import argparse
import traceback
import kiui.vis
import numpy as np
import torch
import rembg
from rembg import new_session, remove
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
from fancy123.utils.normal_utils import postprocess_normal_zero123pp
# from unique3d.app.utils import make_image_grid, split_image
try:
    from unique3d.scripts.refine_lr_to_sr import run_sr_fast
except:
    print('no unique3d.scripts.refine_lr_to_sr, skip sr')
providers = [
                        ('CUDAExecutionProvider', {
                            'device_id': 0,
                            'arena_extend_strategy': 'kSameAsRequested',
                            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                            'cudnn_conv_algo_search': 'HEURISTIC',
                        })
                        ]

def rescale_zero123plus(single_res, input_image, ratio):
    # Rescale and recenter
    image_arr = np.array(input_image)
    ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio)
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((single_res, single_res), Image.LANCZOS)
    return rgba

def convert_color_instantmesh2zero123plus(image: Image.Image,  from_a = 247.0, to_b = 128.0):
    import numpy as np
    colors_HW3 = torch.from_numpy(np.array(image.convert('RGB'))).float()
    H, W, _ = colors_HW3.size()
    colors_HW3 = colors_HW3.view( H, W, 3)

    a = torch.full(( H * W *3,), from_a)
    x = colors_HW3

    y = (to_b / from_a) * x
    # y[x > from_a] = (1 - to_b) * (x[x > from_a] - from_a) / (1 - from_a) + to_b

    result = y
    result = result.numpy()
    result = Image.fromarray(result.astype('uint8'))

    return result

def mv_gen(input_files,pipeline, normal_pipeline,args,image_path,use_input_img_for_recon,load_exist):
    rembg_session = None 
    if not load_exist:
        print('Preparing background removal session...')
        # rembg_session = None if args.no_rembg else rembg.new_session()
        # rembg_session = rembg.new_session(model_name='isnet-general-use')
    print('loading input image...')
    outputs = []
    for idx, image_file in enumerate(input_files):
        seed_everything(args.seed)
        name = os.path.basename(image_file).split('.')[0]
        print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')
        # if idx+1 < 26:
        #     continue
        
       
        images_path = os.path.join(image_path, f'{name}.png')
        input_image_path = os.path.join(image_path, f'{name}_input.png')
        if load_exist and os.path.exists(input_image_path):
            input_image_pil = Image.open(input_image_path)  # ( H,W,3)
            input_image_pil1 = resize_foreground(input_image_pil, 1.0) 
        else:
            input_image_pil = Image.open(image_file)
            # remove background optionally
            if not args.no_rembg:
                if rembg_session is None:
                    print('loading rembg...',input_image_path, 'not exist')
                    # rembg_session = rembg.new_session(model_name='isnet-general-use')
                    

                    rembg_session = new_session(providers=providers)
                    # rembg_session = rembg.new_session()
                input_image_pil = remove_background(input_image_pil, rembg_session)
            input_image_pil = resize_foreground(input_image_pil, 1.0) 
            input_image_pil1 = resize_foreground(input_image_pil, 1.0) 
            if input_image_pil.size[0] < 256: 
                try:
                    input_image_pil = run_sr_fast([input_image_pil], scale=4)[0]
                except:
                    pass
            input_image_pil.save(os.path.join(image_path, f'{name}_input.png'))
        
            input_image_pil = resize_foreground(input_image_pil, 0.85) # instantmesh defaultly use 0.85
            input_image_pil.save(os.path.join(image_path, f'{name}_pipeline_input.png'))
    
        if load_exist and os.path.exists(images_path):
            mv_image = Image.open(images_path)  # ( 960, 640,3)
        else:
            # sampling
            mv_image = pipeline(
                    input_image_pil, 
                    num_inference_steps=args.diffusion_steps, 
                ).images[0]
            mv_image.save(os.path.join(image_path, f'{name}.png'))
        
            
        # generate normal
        if load_exist and os.path.exists(os.path.join(image_path, f'{name}_normal.png')):
        # if False:
            mv_normal = Image.open(os.path.join(image_path, f'{name}_normal.png'))
        else:
            used_mv_image = mv_image
            
            used_mv_image = convert_color_instantmesh2zero123plus(mv_image) # convert instantmesh's colros  to zero123plus
            used_input_image_pil =input_image_pil # resize_foreground(input_image_pil, 0.85) 

            print('generating normal image...')
            seed_everything(args.seed)
            mv_normal = normal_pipeline(used_input_image_pil, depth_image=used_mv_image,
                prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960).images[0] # default
            
            
            # mv_image, mv_normal = postprocess_normal_zero123pp(mv_image, mv_normal)
            mv_image.save(os.path.join(image_path, f'{name}.png')) # save image default background
            mv_normal.save(os.path.join(image_path, f'{name}_normal.png'))
        # remove background
        try:
            mv_image2, mv_normal = postprocess_normal_zero123pp(mv_image, mv_normal)
            mv_image2.save(os.path.join(image_path, f'{name}_rmbg.png')) # save image with transparent background
            mv_normal.save(os.path.join(image_path, f'{name}_normal_rmbg.png'))
            print(f"Normal Image saved to {os.path.join(image_path, f'{name}_normal.png')}")
        except:
            traceback.print_exc()
        
                
                
            

        mv_images = np.asarray(mv_image, dtype=np.float32) / 255.0
        mv_images = torch.from_numpy(mv_images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
        mv_images = rearrange(mv_images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    
        # input_image = resize_foreground(input_image, 1.0) # [3,H,W]
        input_image = input_image_pil1.resize((320, 320))#.convert('RGB')
        input_image = np.asarray(input_image, dtype=np.float32) / 255.0
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).contiguous().float()    # (C, 320, 320)
        if use_input_img_for_recon:
            mv_images = torch.cat([ mv_images,input_image.unsqueeze(0),], dim=0)
            
        outputs.append({'name': name, 'mv_images': mv_images,'input_image':input_image})


    return outputs