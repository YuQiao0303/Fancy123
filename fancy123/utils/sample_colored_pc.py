# taken from the following link with a few modifications:
# https://github.com/NVIDIAGameWorks/kaolin/blob/6fdb91394f6ef0c991da7c845918fb26832c5991/examples/recipes/preprocess/fast_mesh_sampling.py

# from count import get_shapenet2_mesh_data
import sys
sys.path.append("..")
sys.path.append(".")
import os
import numpy as np
from plyfile import PlyData, PlyElement
# from utils.vtk_basic import vis_actors_vtk, get_colorful_pc_actor_vtk
import traceback
# from utils.logger_util import get_logger
import platform
sys_platform = platform.platform().lower()
import datetime
import pytz # time zone
import torch
import nvdiffrast
import nvdiffrast.torch as dr
device = torch.device('cuda')
try:
    glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device) #
except:
    glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)



import kaolin as kal
import torch




######################################################################
#                           core code
######################################################################

# used for sampling points

def preprocessing_transform(vertices,faces,face_uvs,face_material_idx,materials):

    """This the transform used in shapenet dataset __getitem__.
    Three tasks are done:
    1) Get the areas of each faces, so it can be used to sample points
    2) Get a proper list of RGB diffuse map
    3) Get the material associated to each face

    The inputs should contain:
    input.data is a kaolin mesh
    input.attributes['name'] should be a string
    """
    # mesh = inputs.data
    

    # vertices = mesh.vertices.unsqueeze(0)
    # faces = mesh.faces
    # calculate normal # added by Qiao
    face_vertices=vertices[faces].unsqueeze(0) # 1F33
    face_normals = kal.ops.mesh.face_normals(face_vertices=face_vertices, unit=True) # [1,num_faces,3)
    vertices = vertices.unsqueeze(0)

    # Some materials don't contain an RGB texture map, so we are considering the single value
    # to be a single pixel texture map (1, 3, 1, 1)

    # we apply a modulo 1 on the UVs because ShapeNet follows GL_REPEAT behavior (see: https://open.gl/textures)
    # uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0) % 1, (0, 0, 0, 1)) * 2. - 1.
    # uvs[:, :, 1] = -uvs[:, :, 1]
    # uvs = torch.nn.functional.pad(uvs.unsqueeze(0),
    #                               (0, 0, 0, 1))  # mod 1; *2-1, oposite ::1 should be applied after sampling points

    # face_uvs_idx = mesh.face_uvs_idx
    # materials_order = mesh.materials_order

    # DEBUG = False
    # if DEBUG:
    #     for i,m in enumerate(mesh.materials):
    #         print(m)
    # materials = [
    #     m['map_Kd'].permute(2, 0, 1).unsqueeze(0).float() / 255. if 'map_Kd' in m else
    #     m['Kd'].reshape(1, 3, 1, 1)
    #     for m in mesh.materials
    # ]

    # nb_faces = faces.shape[0]
    # num_consecutive_materials = \
    #     torch.cat([
    #         materials_order[1:, 1],
    #         torch.LongTensor([nb_faces])
    #     ], dim=0) - materials_order[:, 1]

    # face_material_idx = kal.ops.batch.tile_to_packed(
    #     materials_order[:, 0],
    #     num_consecutive_materials
    # ).squeeze(-1)
    # mask = face_uvs_idx == -1
    # face_uvs_idx[mask] = 0
    # face_uvs = kal.ops.mesh.index_vertices_by_faces(
    #     uvs, face_uvs_idx
    # )
    # face_uvs[:, mask] = 0.


   

    # print('face_normals',face_normals,face_normals.shape) # [1,num_faces,3)
    face_areas = kal.ops.mesh.face_areas(vertices, faces)
    face_uvs = face_uvs.unsqueeze(0)
    outputs = {
        'vertices': vertices, # [1,vertex_num,3]
        'faces': faces, # [face_num,3]
        'face_areas': face_areas, # 1,face_num
        'face_uvs': face_uvs, # [1,face_num,3,2]
        'materials': materials,
       'face_material_idx': face_material_idx,
        
        'face_normals': face_normals # [1,num_faces,3)
    }


    return outputs


# used for sampling points
class SamplePointsTransform(object):
    """

    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, inputs):
        # print('uv max',inputs['face_uvs'].max().item())
        coords, face_idx, feature_uvs = kal.ops.mesh.sample_points(
            inputs['vertices'],
            inputs['faces'],
            num_samples=self.num_samples,
            areas=inputs['face_areas'],
            face_features=inputs['face_uvs']
        )
        coords = coords.squeeze(0)
        face_idx = face_idx.squeeze(0)


        # we apply a modulo 1 on the UVs because ShapeNet follows GL_REPEAT behavior (see: https://open.gl/textures)
        feature_uvs = feature_uvs.squeeze(0)

        # Interpolate the RGB values from the texture map
        point_materials_idx = inputs['face_material_idx'][face_idx]
        all_point_colors = torch.zeros((self.num_samples, 3)).to(inputs['vertices'].device)

        uvs = feature_uvs
        uvs = (uvs % 1) * 2 - 1
        # uvs[:, 1] = -uvs[:, 1] # only if use kaolin for loading mesh texture
        for i, material in enumerate(inputs['materials']):
            mask = point_materials_idx == i
            point_color = torch.nn.functional.grid_sample(
                material,
                uvs[mask].reshape(1, 1, -1, 2),
                mode='bilinear',
                align_corners=False,
                padding_mode='border')
            all_point_colors[mask] = point_color[0, :, 0, :].permute(1, 0)

        normals = inputs['face_normals'].squeeze(0)[face_idx]

        outputs = {
            'coords': coords,
            'face_idx': face_idx,
            'material_idx': point_materials_idx,
            'uvs': feature_uvs,
            'colors': all_point_colors,
            # 'name': inputs['name'],
            'normals': inputs['face_normals'].squeeze(0)[face_idx]
        }
        return outputs



# sample a given mesh; version before 2023.07.13
def sample_one_mesh(vertices,faces,face_uvs,face_material_ids,materials, point_num=100000, 
                    save_path = None):



    samplePointsTransform = SamplePointsTransform(point_num)
    temp = preprocessing_transform(vertices,faces,face_uvs,face_material_ids,materials)


    outputs = samplePointsTransform(temp)
    # print(outputs['coords'].shape)  # [sampled_V_NUM,  3]
    # print(outputs['face_idx'].shape)  # [sampled_V_NUM]
    # print(outputs['colors'].shape)  # [sampled_V_NUM,  3]
    # print(outputs['material_idx'].shape)  # [sampled_V_NUM]
    # print(outputs['uvs'].shape)  # [sampled_V_NUM,  2]
    # print(outputs['name'])

    # save_one_mesh_npy(outputs, save_root=save_root)

    return outputs







    