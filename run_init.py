import os
import argparse
import traceback
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


# CUDA_VISIBLE_DEVICES=1 python run.py instantmesh_configs/instant-mesh-large.yaml examples/SCHOOL_BUS.png --output_path outputs_temp/
# CUDA_VISIBLE_DEVICES=1 python run.py instantmesh_configs/instant-mesh-large.yaml examples --output_path outputs/

###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.',default='instantmesh_configs/instant-mesh-large.yaml')
parser.add_argument('--input_path', type=str, default = 'examples',help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.') # 42
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')

args = parser.parse_args()
seed_everything(args.seed)
load_exist = True
use_input_img_for_recon = False # always false


subdivide = True
###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False


device = torch.device('cuda')
from src.run.prepare import prepare
pipeline, normal_pipeline, model,input_files,image_path,mesh_path = prepare(args, infer_config,model_config, config_name,device, load_exist,IS_FLEXICUBES)



###############################################################################
# Stage 1: Multiview generation.
###############################################################################
from src.run.mv_gen import mv_gen
mv_outputs = mv_gen(input_files,pipeline, normal_pipeline,args,image_path,use_input_img_for_recon,load_exist)

# delete pipeline to save memory
try:
    
    del pipeline
    torch.cuda.empty_cache()
    
    
except:
    traceback.print_exc()
try:
    del normal_pipeline
    torch.cuda.empty_cache()
except:
    traceback.print_exc()

###############################################################################
# Stage 2: Reconstruction.
###############################################################################
from src.run.mesh_gen import mesh_gen
mesh_gen(mv_outputs, model,infer_config,args,mesh_path,device,
             use_input_img_for_recon,load_exist,IS_FLEXICUBES,subdivide=subdivide)
