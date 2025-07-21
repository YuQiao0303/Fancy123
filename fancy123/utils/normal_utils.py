import sys
import traceback

from fancy123.utils.utils2d import modify_bg_color_of_rgba_np
# from unique3d.app.utils import remove_color
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
sys.path.append('unique3d')
import os
from PIL import Image
import torch
import numpy as np
from einops import rearrange, repeat
# for postprocess by normal
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion
from typing import Tuple


def gen_normal_unique3d(image_pil,save_path):
    from unique3d.app.custom_models.normal_prediction import predict_normals

    result = predict_normals([image_pil], guidance_scale=2., do_rotate=False, num_inference_steps=30,)
    print(len(result))
    result[0].save(save_path)
    result_img = result[0]
    return result_img

def gen_normals_unique3d(img_pils):
    from unique3d.app.custom_models.normal_prediction import predict_normals
    
    result = predict_normals(img_pils, guidance_scale=2., do_rotate=False, num_inference_steps=30,)

    return result
    

def postprocess_normal_zero123pp(rgb_img: Image.Image, normal_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    # Code modified from zero123
    normal_vecs_pred = np.array(normal_img, dtype=np.float64) / 255.0 * 2 - 1
    alpha_pred = np.linalg.norm(normal_vecs_pred, axis=-1)

    is_foreground = alpha_pred > 0.6
    is_background = alpha_pred < 0.2
    structure = np.ones(
        (4, 4), dtype=np.uint8
    )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(alpha_pred.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = np.array(rgb_img, dtype=np.float64) / 255.0
    trimap_normalized = trimap.astype(np.float64) / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized) #  (960, 640) float64  [0.0, 1.0]
    foreground = estimate_foreground_ml(img_normalized, alpha) # (960, 640, 3) float32
    # foreground[alpha.astype(np.bool_)]=[1.0,1.0,1.0,0]
    cutout = stack_images(foreground, alpha)
    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8) # H,W,4
    
    # # modify background color # cutout[cutout[:, :, 3] < 127] = [247, 247, 247, 0] # for instantmesh, whose background is this color
    # old_bg_color = np.array([128,128,128]) /255.0 # color of zero123plus
    # new_bg_color = np.array([247.0,247,247]) /255.0 # color of instantmesh # if we want to use instantmesh to generate geometry, we should use this color
    # cutout = modify_bg_color_of_rgba_np(cutout.astype(np.float32)/255.0, old_bg_color, new_bg_color)
    # cutout = (cutout*255.0).astype(np.uint8)
    
    cutout = Image.fromarray(cutout)

    normal_vecs_pred = normal_vecs_pred / (np.linalg.norm(normal_vecs_pred, axis=-1, keepdims=True) + 1e-8)
    normal_vecs_pred = normal_vecs_pred * 0.5 + 0.5
    normal_vecs_pred = normal_vecs_pred * alpha[..., None] + 0.5 * (1 - alpha[..., None]) #  (960, 640, 3) float64
    normal_vecs_pred = stack_images(normal_vecs_pred, alpha) # added by Qiao Yu
    normal_image_normalized = np.clip(normal_vecs_pred * 255, 0, 255).astype(np.uint8)
    
    return cutout, Image.fromarray(normal_image_normalized)



def gen_mv_normal_zero123pp(input_image_rgba_pil,mv_imgs_pil,normal_pipeline=None,input_247=False,device='cuda'):
    '''
    
    '''
    if normal_pipeline is None:
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler,ControlNetModel
        import copy
        pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16, local_files_only=False)
        normal_pipeline = copy.copy(pipeline)
        normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16, local_files_only=False), 
                                       conditioning_scale=1.0)
        # pipeline.to(device, torch.float16)
        normal_pipeline.to(device, torch.float16)
    if input_247:
        # not good, don't use
        mv_imgs = np.asarray(mv_imgs_pil, dtype=np.int32) # int32 for the following calculation
        mv_imgs_new = mv_imgs.copy()[:,:,:3].astype(np.uint8)
        r_mask = np.abs(mv_imgs[:,:,0] -247)<4
        g_mask = np.abs(mv_imgs[:,:,1] -247)<4
        b_mask = np.abs(mv_imgs[:,:,2] -247)<4
        rg_mask = np.logical_and(r_mask , g_mask)
        rgb_mask = np.logical_and(rg_mask, b_mask)

        mv_imgs_new[rgb_mask] = np.array([128,128,128],dtype=np.uint8)

        mv_imgs_pil = Image.fromarray(mv_imgs_new)
    else:
        if mv_imgs_pil.mode != 'RGB':
            mv_imgs = np.asarray(mv_imgs_pil, dtype=np.uint8)
            mv_imgs_new = mv_imgs.copy().astype(np.float32) / 255.0
            
            # modify background color # mv_imgs_new[mv_imgs[:,:,-1] == 0] = np.array([128,128,128],dtype=np.uint8)
            old_bg_color = np.array([247.0,247,247]) /255.0 # color of instantmesh
            new_bg_color = np.array([128,128,128]) /255.0 # color of zero123plus
            mv_imgs_new = modify_bg_color_of_rgba_np(mv_imgs_new, old_bg_color, new_bg_color)
            
            mv_imgs_pil = Image.fromarray((mv_imgs_new[...,:3]*255.0).astype(np.uint8))
    mv_normal_pil = normal_pipeline(input_image_rgba_pil, depth_image=mv_imgs_pil,
        prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960).images[0]
    try:
        mv_imgs_pil, mv_normal_pil = postprocess_normal_zero123pp(mv_imgs_pil, mv_normal_pil)
    except:
        traceback.print_exc()
    return  mv_imgs_pil, mv_normal_pil,normal_pipeline

def divide_mv_img_zero123pp(mv_imgs_pil):
    mv_images = np.asarray(mv_imgs_pil, dtype=np.float32) / 255.0
    mv_images = torch.from_numpy(mv_images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    mv_images_6CHW = rearrange(mv_images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    return mv_images_6CHW



def my_remove_color(arr,thresh=60):
    '''
    Code modified from Unique3D remove_color
    '''
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # calc diffs
    base = arr[0, 0]
    diffs = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    alpha = (diffs <= thresh)
    
    arr[alpha] = 255
    alpha = ~alpha
    arr = np.concatenate([arr, alpha[..., None].astype(np.int32) * 255], axis=-1)
    return arr

def my_simple_remove(imgs, thresh=60):
    """Only works for normal
    Code modified from Unique3D simple_remove
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
        single_input = True
    else:
        single_input = False
    rets = []
    for img in imgs:
        arr = np.array(img)
        arr = my_remove_color(arr,thresh=thresh)
        rets.append(Image.fromarray(arr.astype(np.uint8)))
    if single_input:
        return rets[0]
    return rets

if __name__ == '__main__':
    from pytorch_lightning import seed_everything
    seed_everything(42)
    from src.utils.infer_util import remove_background, resize_foreground
    name = 'cute_horse'
    image_path = os.path.join('outputs/instant-mesh-large/images')
    for i in range(6):
        # input_image_path = os.path.join(image_path, f'{name}_input.png')
        input_image_path = os.path.join(image_path, f'{name}_view{i+1}.png')
        
        
        input_image_pil = Image.open(input_image_path)  # ( H,W,3)
        input_image_pil = resize_foreground(input_image_pil, 1.0) 
        
        ### zero123pp
        # images_path = os.path.join(image_path, f'{name}.png')
        # mv_images_pil = Image.open(images_path)  # ( 960, 640,C)
        # import kiui
        # # kiui.vis.plot_image(np.array(mv_images_pil.convert('RGB')))
        # mv_imgs_pil, mv_normal_pil = gen_mv_normal_zero123pp(input_image_pil,mv_images_pil)
        # mv_normal_pil.save(f'temp_{name}_mv_normal.png')
        
        
        # unique3d
        save_path = os.path.join(image_path, f'{name}_u3d_normal_view{i+1}.png')
        gen_normal_unique3d(input_image_pil,save_path)