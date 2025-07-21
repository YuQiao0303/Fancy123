import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
import copy
import os

def load_rmbg_session():
    from rembg import new_session, remove
    providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
    ]
    session = new_session(providers=providers)
    return session

def load_depth_pipeline(device='cuda'):
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipeline.add_controlnet(ControlNetModel.from_pretrained(
        "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
    ), conditioning_scale=0.75) # don't use the scheler in zero123plus's example!!!
    pipeline.to(device)
    return pipeline

def load_zero123_plus_pipeline(device='cuda'):
    print('Loading zero123 model ...')
    pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16, local_files_only=False)
    normal_pipeline = copy.copy(pipeline)
    normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
        "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16, local_files_only=False), conditioning_scale=1.0)
    pipeline.to(device, torch.float16)
    normal_pipeline.to(device, torch.float16)
    return pipeline, normal_pipeline

def load_instantmesh_mv_pipeline(device='cuda'):
    from huggingface_hub import hf_hub_download
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
    unet_path =  'ckpts/models--TencentARC--InstantMesh/snapshots/b785b4ecfb6636ef34a08c748f96f6a5686244d0/diffusion_pytorch_model.bin'
    if os.path.exists(unet_path):
        unet_ckpt_path = unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(device)
    return pipeline

def load_sr_stuff():
    from unique3d.app.all_models import model_zoo
    model_zoo.init_models()