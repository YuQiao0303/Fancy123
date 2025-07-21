import datetime
import sys
import yaml
import os
import argparse
import traceback
from einops import rearrange
from lightning_fabric import seed_everything
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.camera_util import elevation_azimuth_radius_to_xyz
from rembg import new_session, remove
from fancy123.render.general_renderer import GeneralRenderer
from fancy123.utils.temp_utils import apply_view_color2mesh, complete_vert_colors_by_neighbors, get_fixed_area, load_test_data_instantmesh
from fancy123.refine_one_sample import refine_one_sample
from fancy123.utils.logger_util import get_logger

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main_refine_instantmesh(config):
    # Unpack configuration
    output_root = config['output_root']
    input_path = config['input_path']
    fast_sr = config['fast_sr']
    input_all_0_elevation = config['input_all_0_elevation']
    deform_3D_method = config['deform_3D_method']
    lap_weight = config['lap_weight']
    use_vertex_wise_NBF = config['use_vertex_wise_NBF']
    fov = config['fov']
    dist = config['dist']
    azim_list = config['azim_list']
    elevations = config['elevations']
    default_res = config['default_res']
    ortho = config['ortho']
    device = config['device']
    load_normals = config['load_normals']
    use_alpha = config['use_alpha']
    refine_sr_by_sd = config['refine_sr_by_sd']
    crop_input_view = config['crop_input_view']
    opt_geo_coarse2fine = config['opt_geo_coarse2fine']
    skip_3D_deform_but_still_unproject = config['skip_3D_deform_but_still_unproject']

    logger = get_logger(os.path.join(output_root, f'{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}_log.log'))
    
    camera_positions_np = elevation_azimuth_radius_to_xyz(
        elevation_in_degrees=elevations,
        azimuth_in_degrees=azim_list,
        radius=dist
    ) 
    camera_positions_B3 = torch.tensor(camera_positions_np).float().to(device)
    renderer = GeneralRenderer(device=device, fov_in_degrees=fov, default_res=default_res, ortho=ortho)
    
    names = os.listdir('examples')
    names = [name.split('.')[0] for name in names]
    names = sorted(names)

    for name_i in tqdm(range(len(names)), desc='Fancy123'):
        name = names[name_i]
        seed_everything(42)
        logger.info('-' * 100)
        logger.info(f'{name_i+1}/{len(names)}, {name}')
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

        vertices, faces, vertex_colors, mv_imgs_BHW4, mv_images_pil, input_image_HW4, input_image_pil, ori_input_image_pil, mv_normals_BHW4 = \
            load_test_data_instantmesh(
                name=name,
                device=device,
                input_path=input_path,
                load_normals=load_normals
            )
        
        input_image_HW4 = np.array(ori_input_image_pil).astype(np.float32)/255.0
        input_image_HW4 = torch.from_numpy(input_image_HW4).to(device)
            
        cfg = {
            'output_root': output_root,
            'fast_sr': fast_sr,
            'deform_3D_method': deform_3D_method,
            'lap_weight': lap_weight,
            'input_all_0_elevation': input_all_0_elevation,
            'use_vertex_wise_NBF': use_vertex_wise_NBF,
            
            'use_alpha': use_alpha,
            'refine_sr_by_sd': refine_sr_by_sd,
            'crop_input_view': crop_input_view,
            'skip_3D_deform_but_still_unproject': skip_3D_deform_but_still_unproject,
            'opt_geo_coarse2fine': opt_geo_coarse2fine,
            
        }
        
        from collections import namedtuple
        Config = namedtuple('Config', cfg.keys())
        cfg_as_obj = Config(**cfg)
        
        refine_one_sample(
            name, vertices, faces, vertex_colors,
            input_image_HW4, mv_imgs_BHW4, mv_normals_BHW4,
            camera_positions_B3, renderer,
            logger, cfg_as_obj,
            geo_refine=True, appearance_refine=True, fidelity_refine=True
        )

def main(config_path):

    config = load_config(config_path)
    init_method = config['init_method']
    if init_method == 'instantmesh':
        main_refine_instantmesh(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fancy123 Mesh Refinement")
    parser.add_argument('--config', type=str, default='configs/instantmesh.yaml', help="path to the config file")
    args = parser.parse_args()
    main(args.config)