import numpy as np
import torch
from trimesh import grouping
from trimesh.geometry import faces_to_edges

import pymeshlab
from pymeshlab import PercentageValue
import open3d as o3d
from src.utils.camera_util import center_looking_at_camera_pose, elevation_azimuth_radius_to_xyz, xyz_to_elevation_azimuth_radius

def poission_recon_mesh(vertices,faces,vertex_colors,poissson_depth=8 ):
    from unique3d.scripts.utils import to_pyml_mesh
    import pymeshlab as ml
    from pymeshlab import PercentageValue
    device = vertices.device
    pyml_mesh = torch2pymeshlab(vertices,faces,vertex_colors)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', threads = 6, depth = poissson_depth, preclean = True)
    
    mesh = ms.current_mesh()
    vertices,faces,vertex_colors = pymeshlab2torch(mesh,device)
    return vertices,faces,vertex_colors
def poission_recon_pc(vertices,vertex_colors,normals,poissson_depth=8 ):
    from unique3d.scripts.utils import to_pyml_mesh
    import pymeshlab as ml
    from pymeshlab import PercentageValue
    device = vertices.device

    # torch to pymeshlab
    vertex_colors4 = torch.cat([vertex_colors,torch.ones(vertices.shape[0],1).to(device)],dim=1) #  4-dim colors as RGBA
    pyml_mesh = pymeshlab.Mesh(
        vertex_matrix=vertices.detach().cpu().float().numpy().astype(np.float64),
        v_color_matrix=vertex_colors4.detach().cpu().float().numpy().astype(np.float64),
        v_normals_matrix = normals.detach().cpu().float().numpy().astype(np.float64)
    )

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")
    # ms.apply_filter('compute_normal_for_point_clouds')
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', threads = 6, depth = poissson_depth, preclean = True)
    
    mesh = ms.current_mesh()

    vertices,faces,vertex_colors = pymeshlab2torch(mesh,device)
    return vertices,faces,vertex_colors
    

def subdivide_mesh(vertices,faces,sub_divide_threshold=0.25,iterations = 2):
    from unique3d.scripts.utils import to_pyml_mesh
    import pymeshlab as ml
    from pymeshlab import PercentageValue
    device = vertices.device
    pyml_mesh = to_pyml_mesh(vertices, faces)
    ms = ml.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")

    # # simplify
    # ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=15000, preservetopology=True)
    # # smooth
    # ms.apply_filter('apply_coord_taubin_smoothing',stepsmoothnum =1) 
    
    # subdevide
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges", method='Remove Faces')
    ms.apply_filter("meshing_surface_subdivision_loop", iterations=iterations, 
                    threshold=PercentageValue(sub_divide_threshold))
    mesh = ms.current_mesh()
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix()) 
    vertices=torch.tensor(vertices).float().to(device).contiguous()
    faces=torch.tensor(faces).to(device).long().contiguous()
    return vertices,faces

def subdivide_mesh_with_color(vertices,faces,vertex_colors,sub_divide_threshold=0.25,iterations = 2):
    from unique3d.scripts.utils import to_pyml_mesh
    import pymeshlab as ml
    from pymeshlab import PercentageValue
    device = vertices.device
    vertex_colors4 = torch.cat([vertex_colors,torch.ones(vertices.shape[0],1).to(device)],dim=1) #  4-dim colors as RGBA
    pyml_mesh = torch2pymeshlab(vertices,faces,vertex_colors)
    ms = ml.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")

    # # simplify
    # ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=15000, preservetopology=True)
    # # smooth
    # ms.apply_filter('apply_coord_taubin_smoothing',stepsmoothnum =1) 
    
    # subdevide
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges", method='Remove Faces')
    ms.apply_filter("meshing_surface_subdivision_loop", iterations=iterations, 
                    threshold=PercentageValue(sub_divide_threshold))
    mesh = ms.current_mesh()
    vertices,faces,vertex_colors = pymeshlab2torch(mesh,device)
    return vertices,faces,vertex_colors

def subdivide_with_uv( vertices, faces, face_uv_idx,uvs, face_index=None):
    """
    Modified from:
    https://github.com/mikedh/trimesh/blob/85b4bd1f410d8d8361009c6f27266719a3d2b97d/trimesh/remesh.py#L15

    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_uv_idx : (F, 3) int
      Indexes of uvs for each vertex of each face
    uvs : (uv_num, 2) float
      UV coordinates
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
   


    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    new_uvs : (uv_num, 2) float
      Remeshed uvs
    new_face_uv_idx : (F, 3) int
      Indexes of uvs for each vertex of each face
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]
    face_uv_subset = face_uv_idx[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # get new uv
    edges_uv = np.sort(faces_to_edges(face_uv_subset), axis=1)
    unique_uv, inverse_uv = grouping.unique_rows(edges_uv)
    mid_uv = uvs[edges_uv[unique_uv]].mean(axis=1)
    mid_idx_uv = inverse_uv.reshape((-1, 3)) + len(uvs)

    # the new faces_subset with correct winding
    f = np.column_stack(
        [
            faces_subset[:, 0],
            mid_idx[:, 0],
            mid_idx[:, 2],
            mid_idx[:, 0],
            faces_subset[:, 1],
            mid_idx[:, 1],
            mid_idx[:, 2],
            mid_idx[:, 1],
            faces_subset[:, 2],
            mid_idx[:, 0],
            mid_idx[:, 1],
            mid_idx[:, 2],
        ]
    ).reshape((-1, 3))

    f_uv = np.column_stack(
    [
        face_uv_subset[:, 0],
        mid_idx_uv[:, 0],
        mid_idx_uv[:, 2],
        mid_idx_uv[:, 0],
        face_uv_subset[:, 1],
        mid_idx_uv[:, 1],
        mid_idx_uv[:, 2],
        mid_idx_uv[:, 1],
        face_uv_subset[:, 2],
        mid_idx_uv[:, 0],
        mid_idx_uv[:, 1],
        mid_idx_uv[:, 2],
    ]
    ).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))

    new_face_uv_idx = np.vstack((face_uv_idx[~face_mask], f_uv))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    new_uvs = np.vstack((uvs, mid_uv))

    return new_vertices,new_faces,new_uvs,new_face_uv_idx

def test_subdivide_with_uv(input_file):
    def savemeshtes2(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, fname):
      import os
      fol, na = os.path.split(fname)
      na, _ = os.path.splitext(na)

      matname = os.path.join(fol, f'model_normalized.mtl')  #matname = '%s/%s.mtl' % (fol, na)
      fid = open(matname, 'w')

      fid.write('newmtl material_0\n')
      fid.write('Kd 1 1 1\n')
      fid.write('Ka 0 0 0\n')
      fid.write('Ks 0.4 0.4 0.4\n')
      fid.write('Ns 10\n')
      fid.write('illum 2\n')
      fid.write('map_Kd %s.png\n' % na)
      fid.close()
      print('save',matname)
      ####

      fid = open(fname, 'w')
      fid.write('mtllib %s.mtl\n' % na)

      for pidx, p in enumerate(pointnp_px3):
          pp = p
          fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

      for pidx, p in enumerate(tcoords_px2):
          pp = p
          fid.write('vt %f %f\n' % (pp[0], pp[1]))

      fid.write('usemtl material_0\n')
      for i, f in enumerate(facenp_fx3):
          f1 = f + 1
          f2 = facetex_fx3[i] + 1
          fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
      fid.close()

    import kaolin as kal
    import torch
    import kiui
    device = 'cpu'
    mesh = kal.io.obj.import_mesh(input_file, with_materials=True)


    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    uvs = mesh.uvs.to(device)
    face_uvs_idx = mesh.face_uvs_idx.to(device)
    materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).to(device).float() / 255. if 'map_Kd' in m else
                      m['Kd'].reshape(1, 3, 1, 1).to(device)
                      for m in mesh.materials]
    atlas_img = materials[0]
    kiui.lo(atlas_img)
    kiui.lo(vertices)
    kiui.lo(faces)
    kiui.lo(uvs)
    kiui.lo(face_uvs_idx)
    face_vert_uvs = uvs[face_uvs_idx]
    kiui.lo(face_vert_uvs)
    new_vertices,new_faces,new_uvs,new_face_uv_idx = subdivide_with_uv(vertices, faces, face_uvs_idx,uvs)

    kiui.lo(new_vertices)
    kiui.lo(new_faces)
    kiui.lo(new_uvs)
    kiui.lo(new_face_uv_idx)

    savemeshtes2(
        new_vertices, # pointnp_px3
        new_uvs, # tcoords_px2
        new_faces, # facenp_fx3
        new_face_uv_idx, # facetex_fx3
        'temp.obj') # fname




def simplify_mesh_with_colors(vertices,faces,vertex_colors,edge_len_percentage=0.25):
    device = vertices.device
    # torch to pymeshlab
    vertex_colors4 = torch.cat([vertex_colors,torch.ones(vertices.shape[0],1).to(device)],dim=1) #  4-dim colors as RGBA
    pyml_mesh = pymeshlab.Mesh(
        vertex_matrix=vertices.detach().cpu().float().numpy().astype(np.float64),
        face_matrix=faces.detach().cpu().long().numpy().astype(np.int32),
        v_color_matrix=vertex_colors4.detach().cpu().float().numpy().astype(np.float64),
    )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")

    # simplify
    ms.apply_filter('meshing_decimation_clustering', threshold =PercentageValue(edge_len_percentage))

    
    # pymeshlab to torch
    mesh = ms.current_mesh()
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix()) 
    vertex_colors = np.array(mesh.vertex_color_matrix())
    vertices=torch.tensor(vertices).float().to(device).contiguous()
    faces=torch.tensor(faces).to(device).long().contiguous()
    vertex_colors=torch.tensor(vertex_colors)[...,:3].float().to(device).contiguous()
    return vertices,faces,vertex_colors

def torch2pymeshlab(vertices,faces,vertex_colors):
    device = vertices.device
    # torch to pymeshlab
    vertex_colors4 = torch.cat([vertex_colors,torch.ones(vertices.shape[0],1).to(device)],dim=1) #  4-dim colors as RGBA
    pyml_mesh = pymeshlab.Mesh(
        vertex_matrix=vertices.detach().cpu().float().numpy().astype(np.float64),
        face_matrix=faces.detach().cpu().long().numpy().astype(np.int32),
        v_color_matrix=vertex_colors4.detach().cpu().float().numpy().astype(np.float64),
    )
    return pyml_mesh
def pymeshlab2torch(pyml_mesh,device):
        
    mesh = pyml_mesh
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix()) 
    vertex_colors = np.array(mesh.vertex_color_matrix())
    vertices=torch.tensor(vertices).float().to(device).contiguous()
    faces=torch.tensor(faces).to(device).long().contiguous()
    vertex_colors=torch.tensor(vertex_colors)[...,:3].float().to(device).contiguous()
    return vertices,faces,vertex_colors

def simplify_mesh(vertices,faces,targetfacenum=10000):
    device = vertices.device
 
    pyml_mesh = pymeshlab.Mesh(
        vertex_matrix=vertices.detach().cpu().float().numpy().astype(np.float64),
        face_matrix=faces.detach().cpu().long().numpy().astype(np.int32),
    )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pyml_mesh, "cube_mesh")

    # simplify
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=targetfacenum, preservetopology=True)

    

    mesh = ms.current_mesh()
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix()) 
    vertices=torch.tensor(vertices).float().to(device).contiguous()
    faces=torch.tensor(faces).to(device).long().contiguous()
    return vertices,faces



import torch
import xatlas
import numpy as np
import nvdiffrast.torch as dr


# ==============================================================================================


def xatlas_uvmap_w_face_id(ctx, mesh_v, mesh_pos_idx, resolution):
    def interpolate(attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')
    vmapping, indices, uvs = xatlas.parametrize(mesh_v.detach().cpu().numpy(), mesh_pos_idx.detach().cpu().numpy())

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
    mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
    # mesh_v_tex. ture
    uv_clip = uvs[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), (resolution, resolution))

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
    mask = rast[..., 3:4] > 0

    per_pixel_face_idx = (rast[..., -1].long() - 1).contiguous()  # 1,res,res
    return uvs, mesh_tex_idx, gb_pos, mask,per_pixel_face_idx
  
  
def load_textured_mesh_by_kaolin(mesh_file,device='cuda'):
    import kaolin as kal
    mesh = kal.io.obj.import_mesh(mesh_file, with_materials=True)

    vertices = mesh.vertices.to(device)

    faces = mesh.faces.to(device)
    # face_normals = kal.ops.mesh.face_normals(face_vertices=mesh.vertices[mesh.faces].unsqueeze(0),
    #                                          unit=True)[0].to(device)  # [F,3]

    # Here we are preprocessing the materials, assigning faces to materials and
    # using single diffuse color as backup when map doesn't exist (and face_uvs_idx == -1)
    uvs = mesh.uvs.to(device)
    uvs[...,1] =  - uvs[...,1] # somehow need this
    # uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0).to(device), (0, 0, 0, 1)) #% 1. # don't %1 here. %1 after interpolate
    face_uvs_idx = mesh.face_uvs_idx.to(device)
    face_uvs_idx[face_uvs_idx == -1] = 0
    
    material_list_of_13HW = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).to(device).float() / 255. if 'map_Kd' in m else
                m['Kd'].reshape(1, 3, 1, 1).to(device)
                for m in mesh.materials]
    if len(material_list_of_13HW) > 1:
        # face_material_idx = mesh.material_assignments.to(device) # my kaolin version is too low, use the following instead
        materials_order = mesh.materials_order
        nb_faces = faces.shape[0]

        num_consecutive_materials = \
            torch.cat([
                materials_order[1:, 1],
                torch.LongTensor([nb_faces])
            ], dim=0) - materials_order[:, 1]


        face_material_idx = kal.ops.batch.tile_to_packed(
            materials_order[:, 0],
            num_consecutive_materials
        ).squeeze(-1).to(device)

    else:
        face_material_idx = torch.zeros(faces.shape[0]).to(device)
    return vertices,faces,uvs,face_uvs_idx,material_list_of_13HW,face_material_idx


def load_glb_vertex_color(mesh_file,device=None):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)

    return vertices, faces, vertex_colors


def load_mesh_with_uv(mesh_file,device=None):
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    F = len(faces)
    if mesh.has_triangle_uvs():
        face_uvs = np.asarray(mesh.triangle_uvs)

        # face_uvs = face_uvs.reshape(3,F,2)
        # face_uvs = face_uvs.transpose(1,0,2) # F,3,2
        
        face_uvs = face_uvs.reshape(F,3,2)
        triangle_material_ids = np.asarray(mesh.triangle_material_ids)
        unique_material_ids = np.unique(triangle_material_ids)

        
        textures = []
        for i,texture in enumerate(mesh.textures):
            if i in unique_material_ids:
                try:
                    texture = np.asarray(texture)
                    texture = (texture.astype(np.float32))/255.0
                    textures.append(texture)
                except:
                    pass
            else:
                textures.append(np.zeros((1,1,3)))
        if device is not None:
            vertices=torch.tensor(vertices, dtype=torch.float32, device=device)
            faces = torch.tensor(faces, dtype=torch.long, device=device)
            face_uvs = torch.tensor(face_uvs, dtype=torch.float32, device=device)
            triangle_material_ids = torch.tensor(triangle_material_ids, dtype=torch.long, device=device)
            textures = [torch.tensor(texture, dtype=torch.float32, device=device) for texture in textures]
            
    else:
        face_uvs = None
        triangle_material_ids = None
        textures = None


    return vertices, faces, face_uvs, triangle_material_ids, textures

def load_mesh_and_sample_pc_o3d(mesh_file,device=None):
    """
    Loads a mesh from an OBJ file and samples 10,000 points from it using Open3D.

    Args:
        mesh_file (str): The path to the OBJ file containing the mesh.
        device (torch.device, optional): The device to use for the point cloud. Defaults to None.

    Returns:
        None
    """
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    F = len(faces)
    # if mesh.has_triangle_uvs():
    #     face_uvs = np.asarray(mesh.triangle_uvs)
    #     # face_uvs = face_uvs.reshape(3,F,2)
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    o3d.visualization.draw_geometries([pcd])

def trans_mesh_vertices_to_fit_view(vertices, camera_position_np_13, elevation=None, azimuth=None, distance=None):
    """
    Transforms the mesh vertices so that the mesh appears the same when viewed from a new camera position.
    
    Args:
        vertices (Tensor): A tensor of shape [V, 3] representing the vertices of the mesh.
        elevation (float): The elevation angle of the original camera in degrees.
        azimuth (float): The azimuth angle of the original camera in degrees.
        distance (float): The distance of the original camera from the origin.
        
    Returns:
        Tensor: A tensor of shape [V, 3] representing the transformed vertices.
    """
    from math import sin, cos, radians
    device = vertices.device

    if camera_position_np_13 is not None:
        elevation, azimuth, distance = xyz_to_elevation_azimuth_radius(camera_position_np_13)
    else:
        assert elevation is not None
        assert azimuth is not None
        assert distance is not None
    # Convert angles from degrees to radians
    elevation_rad = radians(elevation)
    azimuth_rad = radians(azimuth)

    # # Calculate the original camera position
    # cam_x = distance * sin(elevation_rad) * cos(azimuth_rad)
    # cam_y = distance * sin(elevation_rad) * sin(azimuth_rad)
    # cam_z = distance * cos(elevation_rad)

    # Calculate rotation matrices for elevation and azimuth
    Rx = torch.tensor([ 
        [1, 0, 0],
        [0, cos(elevation_rad), -sin(elevation_rad)],
        [0, sin(elevation_rad), cos(elevation_rad)]
    ]).to(device)
    
    Ry = torch.tensor([
        [cos(azimuth_rad), 0, sin(-azimuth_rad)],
        [0, 1, 0],
        [-sin(-azimuth_rad), 0, cos(azimuth_rad)]
    ]).to(device)

    # Combine rotations
    R = torch.matmul(Ry, Rx)

    # Apply inverse rotation to vertices
    # Note: PyTorch uses column-major order for matrix multiplication, so we transpose R
    rotated_vertices = torch.mm(vertices, R.t())

    return rotated_vertices

if __name__ == '__main__':
  test_subdivide_with_uv('temp/meshes/model.obj')