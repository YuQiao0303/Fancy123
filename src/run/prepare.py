
import os
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

def prepare(args, infer_config,model_config, config_name,device, load_exist,IS_FLEXICUBES):
    pipeline = None
    normal_pipeline = None
    model = None
    # if not load_exist:
    if True:
        
        print('Loading zero123 model ...')
        zero123_plus_pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16, local_files_only=False)
        
            
        normal_pipeline = zero123_plus_pipeline
        normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16, local_files_only=False), conditioning_scale=1.0)
        normal_pipeline.to(device, torch.float16)
        

        # load diffusion model
        print('Loading diffusion model ...')
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        

        # load custom white-background UNet
        print('Loading custom white-background unet ...')
        if os.path.exists(infer_config.unet_path):
            unet_ckpt_path = infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipeline.unet.load_state_dict(state_dict, strict=True)

        pipeline = pipeline.to(device)

        # load reconstruction model
        print('Loading reconstruction model ...')
        model = instantiate_from_config(model_config)
        if os.path.exists(infer_config.model_path):
            model_ckpt_path = infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        model.load_state_dict(state_dict, strict=True)

        model = model.to(device)
        if IS_FLEXICUBES:
            model.init_flexicubes_geometry(device, fovy=30.0)
        model = model.eval()

    # make output directories
    image_path = os.path.join(args.output_path, config_name, 'images')
    mesh_path = os.path.join(args.output_path, config_name, 'meshes')

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)


    # process input files
    if os.path.isdir(args.input_path):
        input_files = [
            os.path.join(args.input_path, file) 
            for file in os.listdir(args.input_path) 
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp') or file.endswith('.jpeg')
        ]
        input_files = sorted(input_files)
    else:
        input_files = [args.input_path]
    print(f'Total number of input images: {len(input_files)}')
    # input_files = [input_files[0],input_files[1],input_files[2],input_files[3],input_files[4]] # debug
    return pipeline, normal_pipeline, model,input_files,image_path,mesh_path