# modified from Unique3D/mesh_reconstruction/render.py
# modified from https://github.com/Profactor/continuous-remeshing
import sys
sys.path.append('...')
sys.path.append('..')
sys.path.append('.')
import kiui
import nvdiffrast.torch as dr
import torch
import numpy as np
from typing import Tuple
from pytorch3d.renderer.cameras import look_at_view_transform, OrthographicCameras, CamerasBase
import kaolin as kal


def _warmup(glctx, device=None):
    device = 'cuda' if device is None else device
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device=device, **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

glctx = dr.RasterizeCudaContext( device="cuda")#glctx = dr.RasterizeGLContext(output_db=False, device="cuda")


class Unique3DRenderer:
    _glctx:dr.RasterizeCudaContext = None #_glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            device=None,
            ):

        self._glctx = glctx
        _warmup(self._glctx, device)
    
    def transform_vertices_pt3d(self, vertices, cameras: CamerasBase):
        vertices = cameras.transform_points_ndc(vertices)

        perspective_correct = cameras.is_perspective()
        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if not perspective_correct or znear is None else znear / 2

        # if z_clip:
        #     vertices = vertices[vertices[..., 2] >= cameras.get_znear()][None]    # clip
        vertices = vertices * torch.tensor([-1, -1, 1]).to(vertices)
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1).to(torch.float32)
        return vertices
    
    def render(self,
            vertices: torch.Tensor, #V,3 float
            colors: torch.Tensor, #V,3 float   in [0, 1]
            faces: torch.Tensor, #F,3 long
            cameras:CamerasBase = None, # use 'cameras' or 'proj' and 'mv'
            mv: torch.Tensor = None, #C,4,4
        
            proj: torch.Tensor = None, #C,4,4
            image_size: Tuple[int,int] = (512,512),

            ) ->torch.Tensor: # [C,H,W,4], [C,H,W,1],[C,H,W,1]

        V = vertices.shape[0]
        faces = faces.type(torch.int32)

        if cameras is not None:
            vertices_clip = self.transform_vertices_pt3d(vertices, cameras)
            
         
        else:
            assert proj is not None
            assert mv is not None
            mvp = proj @ mv #C,4,4
            vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
            vertices_clip = vert_hom @ mvp.transpose(-2,-1) #C,V,4, C means camera_num instead of channels
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=image_size, grad_db=False) #C,H,W,4, 4 is (u,v,z/w,triangle_id), triangle id starts from 1
        vert_col = colors #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        face_ids = rast_out[..., -1:].long()-1 #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4

        return col,alpha.bool(),face_ids #[C,H,W,4], [C,H,W,1],[C,H,W,1]

class KaolinRenderer:
    _glctx:dr.RasterizeCudaContext = None #_glctx:dr.RasterizeGLContext = None
    
    def __init__(self, device=None):
        self._glctx = glctx
        _warmup(self._glctx, device)

    
    def render(self,
            vertices: torch.Tensor, #V,3 float
            colors: torch.Tensor, #V,3 float   in [0, 1]
            faces: torch.Tensor, #F,3 long
            cameras = None, 
            image_size: Tuple[int,int] = (512,512),
            ) ->torch.Tensor: # [C,H,W,4], [C,H,W,1],[C,H,W,1]
        device = vertices.device
        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vertices[:,1] *= -1 # kaolin camera system somehow differs from pytorch3d, and the y axis should be flipped
    
        pos = torch.zeros((len(cameras), V, 4), device=device)
        for i, cam in enumerate(cameras):
            transformed_vertices = cam.transform(vertices.unsqueeze(0))
            # Create a fake W (See nvdiffrast documentation)
            pos[i] = torch.nn.functional.pad(
                transformed_vertices, (0, 1), mode='constant', value=1.
            ).contiguous()


        rast_out,_ = dr.rasterize(self._glctx, pos, faces, resolution=image_size, grad_db=False) #C,H,W,4, 4 is (u,v,z/w,triangle_id), triangle id starts from 1

        vert_col = colors #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        face_ids = rast_out[..., -1:].long()-1 #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, pos, faces) #C,H,W,4

        return col,alpha.bool(),face_ids #[C,H,W,4], [C,H,W,1],[C,H,W,1]
        

    
def save_tensor_to_img(tensor, save_dir):
    from PIL import Image
    import numpy as np
    for idx, img in enumerate(tensor):
        img = img[..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(save_dir + f"{idx}.png")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.mesh_util import loadobj_color
    import kiui

    device = 'cuda'
    resolution = 256
    vertices, faces, vertex_colors = loadobj_color('outputs/instant-mesh-large/meshes/clock.obj',device=device)


    kaolin = True
    if kaolin:
        from src.utils.camera_util import get_kaolin_cameras_azim_ele_pers
        
        cameras = get_kaolin_cameras_azim_ele_pers( 
                                            azim=[30.0, 90.0, 150.0, 210.0, 270.0, 330.0], 
                                            elev = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0],
                                            dist=4.0,
                                            device=device,
                                            )

        renderer = KaolinRenderer(device=device)
        colors,alpha,face_ids = renderer.render(vertices, vertex_colors, faces,cameras=cameras, image_size= [resolution,resolution])
        for i in range(len(cameras)):
            kiui.vis.plot_image(colors[i])



    else:

        from unique3d.mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
        from unique3d.mesh_reconstruction.func import make_star_cameras
        cameras = make_star_cameras_orthographic_py3d([0, 270, 180, 90], device="cuda", focal=1., dist=4.0)
        mv,proj = make_star_cameras_orthographic(4, 1)

        
        renderer1 = Unique3DRenderer( device="cuda")
    
        # vertices = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[1,0,0]], device="cuda", dtype=torch.float32)
        # vertex_colors = torch.tensor([[1,1,1],[1,0,0],[0,0,1],[0.5,1,0.5]], device="cuda", dtype=torch.float32)
        # faces = torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], device="cuda", dtype=torch.long)
        
        # import time
        # t0 = time.time()
        # r1 = renderer1.render(vertices, vertex_colors, faces,cameras=cameras, image_size= [resolution,resolution])
        # print("time r1:", time.time() - t0)
        # print(cameras)

        
        
        from unique3d.scripts.project_mesh import get_cameras_list_azim_ele
        
        scale = 1.0
        camera_list = get_cameras_list_azim_ele(
                                            azim_list=[30.0, 90.0, 150.0, 210.0, 270.0, 330.0], 
                                            elevations = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0],
                                            cam_type="fov",
                                            dist=4.0,
                                            device=device,
                                            return_list=False
                                            )
        cameras = camera_list
        print(cameras)
        colors,alpha,face_ids = renderer1.render(vertices, vertex_colors, faces,cameras=cameras, image_size= [resolution,resolution])
        for i in range(len(cameras)):
            kiui.vis.plot_image(colors[i])
