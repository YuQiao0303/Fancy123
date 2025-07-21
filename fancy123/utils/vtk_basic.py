import numpy as np
# import trimesh
# from plyfile import PlyData, PlyElement
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import vtk
import seaborn as sns
import random

from vtkmodules.vtkCommonDataModel import vtkDataObject

palette = np.array(sns.color_palette("hls", 8))
red = [1,0.67,0.6]
red = np.array([1,0.67,0.6])
blue = [0.6,0.67,1]
green = [0.6,1,0.67]
orange = [0.92,0.78,0.52]

darkred = [0.5,0.1,0.1]
brightred = [0.7,0.1,0.1]



color1 = palette[1]
color2 = palette[5]
color3 = palette[7]
color_wrong = palette[0]

palette_red = palette[0]
palette_yellow = palette[1]


shapenet_cat_id_str2int = {None:-100, '04379243':0, '03001627':1, '02871439':2, '04256520':3, '02747177':4,
                           '02933112':5, '03211117':6, '02808440':7}
transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

import platform
sys_platform = platform.platform().lower()
if "windows" in sys_platform:
    LINUX = False
    meshlab_path = "D:\software\VCG\MeshLab\meshlab.exe "
elif "linux" in sys_platform:
    LINUX = True
    meshlab_path = "/home/me/download/Meshlab2022.02/AppRun "

def visFilesByMeshlab(files):
    '''
    Windows only
    :param files:
    :return:
    '''
    command = meshlab_path
    for file in files:
        command += file
        command += " "
    # print(command)
    os.system(command)

# save pc xyz
def export_pc_xyz(pc,path):
    vertices = np.empty(pc.shape[0],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'] = pc[:, 0].astype('f4')
    vertices['y'] = pc[:, 1].astype('f4')
    vertices['z'] = pc[:, 2].astype('f4')
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(path)

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# -------------transformation
def rescale_cano_pc(cano_pc,box):
    pc = cano_pc
    xmin = pc[:, 0].min()
    xmax = pc[:, 0].max()
    ymin = pc[:, 1].min()
    ymax = pc[:, 1].max()
    zmin = pc[:, 2].min()
    zmax = pc[:, 2].max()

    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    pc -= center
    scales = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    scale = scales.max()
    # print(xmin,xmax,ymin,ymax,zmin,zmax)
    # print(scale)
    pc = pc / scales
    pc[:,0] = pc[:,0] * box[4]
    pc[:,1] = pc[:,1] * box[5]
    pc[:,2] = pc[:,2] * box[3]
    pc = pc/np.max(box[3:6])


    return pc
# ----------------   vtk_utils   ----------------
def mkVtkIdList(it):
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil
def read_trimesh_vtk(mesh):
    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)
    nodes = v
    elements = f

    # Make the building blocks of polyData attributes
    Mesh = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()

    # Load the point and cell's attributes
    for i in range(len(nodes)):
        Points.InsertPoint(i, nodes[i])

    for i in range(len(elements)):
        Cells.InsertNextCell(mkVtkIdList(elements[i]))

    # Assign pieces to vtkPolyData
    Mesh.SetPoints(Points)
    Mesh.SetPolys(Cells)
    return Mesh
def read_off_vtk(meshfile):
    mesh = trimesh.load(meshfile)
    return read_trimesh_vtk(mesh)
def set_actor( mapper):
    '''
    vtk general actor
    :param mapper: vtk shape mapper
    :return: vtk actor
    '''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


# ------------------------ vis actors or renderers
def vtk_window_to_numpy(window):
    '''
    :param window:
    :return: arr of shape [C,H,W], channesl are RGB. ranging between 0 and 1
    '''


    vtk_win_im = vtk.vtkWindowToImageFilter()
    vtk_win_im.SetInput(window)
    vtk_win_im.Update()

    vtk_image = vtk_win_im.GetOutput()

    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()

    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
    arr  = arr.astype(float) / 255. # [H,W,3]
    arr = arr[::-1,:,:] # flip vertically
    # print('arr',arr,arr.shape)
    arr = arr.transpose((2,0,1)) # [3,H,W]
    arr = np.ascontiguousarray(arr) # so that when it's converted to tensor, there will be no error

    return arr



def vis_actors_vtk(actors,info=None,parallel=False,arrows = True,backgroundGradient = True,cameraParam = None,
                   save_path = None,return_np = False, vis_now=True):
    '''set renderer'''
    renderer = vtk.vtkRenderer()
    # renderer.SetViewport(0, 0, 0.25, 0.5)  # comment this line if necessary
    if backgroundGradient:
        renderer.SetBackground(1, 1, 1)  
        renderer.SetBackground2(0.529, 0.8078, 0.92157)  
        renderer.SetGradientBackground(1)
    else:
        renderer.SetBackground(1,1,1) 

    # Renderer Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    if vis_now:
        window.FullScreenOn()  # same as SetFullScreen(True)
    window.BordersOn()  # same as SetBorders(True)

    # System Event
    win_render = vtk.vtkRenderWindowInteractor()
    win_render.SetRenderWindow(window)

    # Style


    # interactor_camera.AutoAdjustCameraClippingRangeOff()
    # interactor_camera.SetClippingRange(0, 1000000)  # default is (0.1,1000) # not working
    win_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera()) # what if we comment it here
    # win_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())



    if cameraParam is None:

        cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        camera = vtk.vtkCamera()
        camera.SetViewAngle((2 * np.arctan(cam_K[1][2] / cam_K[0][0])) / np.pi * 180) #77.31
        # camera.SetViewAngle(77.31) #77.31
        # global default
        # camera.SetPosition(np.array([0.6241536419507486, 0.681298692302482, 1.9342278328738134]))
        # camera.SetFocalPoint(np.array([0.0,0.0,0.0]))
        # camera.SetViewUp(np.array([-0.08724089052693657, 0.9480912654069744, -0.3057972849439141]))
        # local default
        # camera.SetPosition(np.array([0.40166213226865183, 0.3741911026625985, -0.6191230736363628]))
        # camera.SetFocalPoint(np.array([-0.015854752134160377, 0.03631870210278588, 0.008997414441827861]))
        # camera.SetViewUp(np.array([-0.2138756546473049, 0.9125111473575531, 0.3486841124820417]))
        camera.SetPosition(np.array([0.594316678162462, 0.39478630943700577, -0.9846444525311533]))
        camera.SetFocalPoint(np.array([-0.01696979229169535, -0.09989267222261602, -0.06501324593587415]))
        camera.SetViewUp(np.array([-0.21387565464730487, 0.912511147357553, 0.3486841124820417]))

        renderer.SetActiveCamera(camera)
    else:
        cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        camera = vtk.vtkCamera()
        camera.SetPosition(cameraParam[0])
        camera.SetFocalPoint(cameraParam[1])
        camera.SetViewUp(cameraParam[2])
        # camera.SetViewUp(np.array([0, 1, 0]))
        camera.SetViewAngle((2 * np.arctan(cam_K[1][2] / cam_K[0][0])) / np.pi * 180) #77.31
        # camera.SetViewAngle(77.31)  # 77.31

        renderer.SetActiveCamera(camera)


    if parallel:
        '''set camera''' # this part is new added
        camera = renderer.GetActiveCamera()
        renderer.ResetCamera()
        # camera = vtk.vtkCamera()
        # cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        # camera_center = np.array([0, -3, 3])
        # centroid = camera_center
        # camera = set_camera(centroid, [[0., 0., 0.], [-centroid[0], -centroid[1],
        #                                                    centroid[0] ** 2 / centroid[2] + centroid[1] ** 2 / centroid[
        #                                                        2]]], cam_K)
        camera.ParallelProjectionOn()
        # camera.SetClippingRange(0,1000000) # default is (0.1,1000) # not working
        # camera.SetObliqueAngles(0,100) # (45,90) by default. # not working too
        # camera.SetViewAngle(90) # 30 by default
        renderer.SetActiveCamera(camera)
    # Insert Actor
    for actor in actors:
        renderer.AddActor(actor)
    # add axis actor
    if arrows:
        axes = vtk.vtkAxesActor()
        renderer.AddActor(axes)


    # set point picker
    if info is not None:
        # Create a point picker
        picker = vtk.vtkPointPicker()

        # Define a callback function for picking
        def pick_point(obj, event):
            # Get the display position of the mouse
            x, y = obj.GetEventPosition()

            # Pick a point from the point cloud
            picker.Pick(x, y, 0, renderer)

            # Get the id and position of the picked point
            point_id = picker.GetPointId()
            point_pos = picker.GetPickPosition()

            # Get the color of the picked point
            # point_color = colors.GetTuple(point_id)


            # Print the information of the picked point
            print(f"Point id: {point_id}")
            print(f"Point position: {point_pos}")
            print(f"Point info: {info[point_id]}")
            print('-'*50)
            # print(f"Point color: {point_color}")

        # Add the callback function to the interactor
        win_render.AddObserver("RightButtonPressEvent", pick_point) # LeftButtonPressEvent

    # visaulize

    win_render.Initialize()
    if save_path is not None:
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(window)
        windowToImageFilter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(save_path)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()
    if vis_now:
        window.OffScreenRenderingOff()
        win_render.Start()
        # print camera parameters
        camera = window.GetRenderers().GetFirstRenderer().GetActiveCamera()
        print(camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
    else:
        window.OffScreenRenderingOn()  # don't show window, directly save png

    if return_np:
        return vtk_window_to_numpy(window)

def vis_renderers(renderers,save_path = None,share_camera = True):
    # Renderer Window
    window = vtk.vtkRenderWindow()
    camera = renderers[0].GetActiveCamera()
    for renderer in renderers:
        if share_camera:
            renderer.SetActiveCamera(camera)
        window.AddRenderer(renderer)
    window.FullScreenOn()  # SetFullScreen(True)
    window.BordersOn()  # SetBorders(True)

    # System Event
    win_render = vtk.vtkRenderWindowInteractor()
    win_render.SetRenderWindow(window)

    # Style
    win_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())  # what if we comment it here

    # visaulize
    win_render.Initialize()
    if save_path is not None:
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(window)
        windowToImageFilter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(save_path)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    win_render.Start()

def get_renderer(actors,cameraParam = None,parallel=False,arrows = False):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  

    if cameraParam is None:
        # cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        camera = vtk.vtkCamera()
        # camera.SetPosition(np.array([0.6241536419507486, 0.681298692302482, 1.9342278328738134]))
        # camera.SetFocalPoint(np.array([0.0, 0.0, 0.0]))
        # camera.SetViewUp(np.array([-0.08724089052693657, 0.9480912654069744, -0.3057972849439141]))
        # # camera.SetViewUp(np.array([0, 1, 0]))
        # # camera.SetViewAngle((2 * np.arctan(cam_K[1][2] / cam_K[0][0])) / np.pi * 180)  # 77.31
        # # camera.SetViewAngle(77.31) #77.31
        #
        renderer.SetActiveCamera(camera)
    else:
        cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        camera = vtk.vtkCamera()
        # print(cameraParam)
        camera.SetPosition(cameraParam[0])
        camera.SetFocalPoint(cameraParam[1])
        camera.SetViewUp(cameraParam[2])
        # camera.SetViewUp(np.array([0, 1, 0]))
        camera.SetViewAngle((2 * np.arctan(cam_K[1][2] / cam_K[0][0])) / np.pi * 180)  # 77.31
        # camera.SetViewAngle(77.31)  # 77.31

        renderer.SetActiveCamera(camera)

    if parallel:
        '''set camera'''  # this part is new added
        camera = renderer.GetActiveCamera()
        renderer.ResetCamera()
        # camera = vtk.vtkCamera()
        # cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        # camera_center = np.array([0, -3, 3])
        # centroid = camera_center
        # camera = set_camera(centroid, [[0., 0., 0.], [-centroid[0], -centroid[1],
        #                                                    centroid[0] ** 2 / centroid[2] + centroid[1] ** 2 / centroid[
        #                                                        2]]], cam_K)
        camera.ParallelProjectionOn()
        # camera.SetClippingRange(0,1000000) # default is (0.1,1000) # not working
        # camera.SetObliqueAngles(0,100) # (45,90) by default. # not working too
        # camera.SetViewAngle(90) # 30 by default
        renderer.SetActiveCamera(camera)
        # Insert Actor
    for actor in actors:
        renderer.AddActor(actor)
        # add axis actor
    if arrows:
        axes = vtk.vtkAxesActor()
        renderer.AddActor(axes)
    return renderer

def set_renderer_pos(renderer, row,col,row_num, col_num, x_length=None,y_length=None, x_margin=0,y_margin=0):
    if x_length == None:
        x_length = 1.0/col_num
    if y_length == None:
        y_length = 1.0/row_num
    xmin = col * (x_length+x_margin)
    xmax = xmin + x_length
    ymin = row * (y_length + y_margin)
    ymax = ymin + y_length

    xmin += 0.5- (x_length*col_num)/2
    xmax += 0.5- (x_length*col_num)/2

    ymin += 0.5- (y_length*row_num)/2
    ymax += 0.5- (y_length*row_num)/2

    renderer.SetViewport(xmin, ymin, xmax, ymax)
    return renderer

def get_seaborn_main_color(palette_name,i=50):
    color = np.array(sns.color_palette(palette_name, n_colors=100))[i]
    return color

def convert_torch2np(param):
    if not isinstance(param, np.ndarray):
        param = param.detach().cpu().numpy()
    return param

# ------------------------ actors
def get_colorful_pc_actor_vtk(pc_np,point_colors = None,point_size = 3, opacity = 0.3,palette_name = 'crest_r',light = 1,cut=True):
    pc_np = convert_torch2np(pc_np)
    if point_colors is not None:
        point_colors = convert_torch2np(point_colors)
    if cut:
        depth_palette = np.array(sns.color_palette(palette_name, n_colors=200))
        depth_palette = depth_palette[100:]
    else:
        depth_palette = np.array(sns.color_palette(palette_name, n_colors=100))
    centroid = np.array([3, 0, 3])
    def set_points_property( point_clouds, point_colors):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        x3 = point_clouds[:, 0]
        y3 = point_clouds[:, 1]
        z3 = point_clouds[:, 2]

        for x, y, z, c in zip(x3, y3, z3, point_colors):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point
    colors = np.linalg.norm(pc_np - centroid, axis=1)
    colors = depth_palette[np.int16((colors - colors.min()) / (colors.max() - colors.min()) * 99)]

    mapper = vtk.vtkPolyDataMapper()
    if point_colors is None:
        point_colors =255 * colors* light
    if point_colors.max() <=1.0:
        point_colors *=255
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(set_points_property(pc_np, point_colors))
    else:
        mapper.SetInputData(set_points_property(pc_np, point_colors))

    point_actor = vtk.vtkActor()
    point_actor.SetMapper(mapper)

    point_actor.GetProperty().SetPointSize(point_size)
    point_actor.GetProperty().SetOpacity(opacity)
    point_actor.GetProperty().SetInterpolationToPBR()
    return point_actor

def get_pc_actor_vtk(pc_np,color = (0,0,1),opacity = 1,point_size = 7 ):
    if isinstance(pc_np,str):
        if os.path.exists(pc_np):
            pc_np = read_ply(pc_np)
        else:
            # print('wrong pc input:',pc_np)
            pc_np = np.zeros((1,3))
            opacity = 0

    pc_np = convert_torch2np(pc_np)
    obj_points = pc_np


    
    points = vtk.vtkPoints()

    points.SetData(numpy_to_vtk(obj_points))
   
    polydata = vtk.vtkPolyData()
  
    polydata.SetPoints(points)

 
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)

 
    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputConnection(vertex.GetOutputPort())


    actor = vtk.vtkActor()

    actor.SetMapper(mapper)

    # actor.GetProperty().SetColor(1, 0, 0)  
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetPointSize(point_size)
    actor.GetProperty().SetOpacity(opacity)

    return actor

def get_mesh_actor_vtk(meshfile,GT_box_z_up = None,proposal_box_z_up = None,color = [1.3,1.3,1.3],opacity=1,PBR=True):
    '''get data'''

    if isinstance(meshfile, str):
        if ".ply" in meshfile:  # proposal mesh
            vtk_object = vtk.vtkPLYReader()
            vtk_object.SetFileName(meshfile)
            vtk_object.Update()
            polydata = vtk_object.GetOutput()

        elif ".obj" in meshfile:  # GT mesh
            vtk_object = vtk.vtkOBJReader()
            vtk_object.SetFileName(meshfile)
            vtk_object.Update()
            polydata = vtk_object.GetOutput()

        elif ".off" in meshfile:
            polydata = read_off_vtk(meshfile)
            # print("off", type(vtk_object))
        else:
            print("wrong", meshfile)
    else:
        polydata = read_trimesh_vtk(meshfile)

    '''replace aligned points'''
    # polydata = plydata.GetOutput()
    # points_array = numpy_to_vtk(self.vertices[..., :3], deep=True)
    # # Update the point information of vtk
    # polydata.GetPoints().SetData(points_array)
    # # update changes
    # plydata.Update()
    '''set mapper'''
    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputConnection(vtk_object.GetOutputPort()) # vtk_object is vtkmodules.vtkCommonDataModel.vtkPolyData

    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)
    '''set actor'''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if color is not None:
        actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    if PBR:
        actor.GetProperty().SetInterpolationToPBR()
    return actor
# def get_cube_actor_vtk(color =  np.array(sns.color_palette("crest_r", n_colors=100))[25],opacity = 0.05, xyzwhl = [0,0,0,1,1,1],rotation=[0,0,0]):
def get_cube_actor_vtk(color = (0.2,0,0),opacity = 0.05, xyzwhl = [0,0,0,1,1,1],rotation=[0,0,0]):
    from vtkmodules.vtkCommonColor import (
        vtkColorSeries,
        vtkNamedColors
    )

    # Create a cube
    from vtkmodules.vtkFiltersSources import (
        vtkConeSource,
        vtkCubeSource,
        vtkCylinderSource,
        vtkSphereSource
    )


    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderWindow,
        vtkRenderWindowInteractor,
        vtkRenderer
    )

    cubeSource = vtkCubeSource()
    # cubeSource.SetCenter(0.0, 0.0, 0.0)
    cubeSource.SetCenter(xyzwhl[0], xyzwhl[1], xyzwhl[2])
    cubeSource.SetXLength(xyzwhl[3])
    cubeSource.SetYLength(xyzwhl[4])
    cubeSource.SetZLength(xyzwhl[5])

    cubeSource.Update()
    # colors = vtkNamedColors()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cubeSource.GetOutputPort())

    actor = vtkActor()
    # my_trans = vtkTransform()
    # actor.SetUserTransform()
    actor.RotateY(rotation[1])



    actor.SetMapper(mapper)

    # actor.GetProperty().SetColor(colors.GetColor3d('Tomato'))

    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    # actor.GetProperty().SetEdgeVisibility(True) # show edges
    # actor.GetProperty().SetEdgeColor(color) # show edges
    # actor.GetProperty().SetEdgeWidth(3) # now such input_pc_generate_method
    return actor

def get_voxel_actor_mc_vtk(voxel):
    # Create a vtkImageData object with the same dimensions as the voxel grid
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(*voxel.shape)

    # Set the voxel spacing to 1.0 in all dimensions
    image_data.SetSpacing(1.0, 1.0, 1.0)

    # Allocate memory for the voxel data and copy the voxel grid into it
    voxel_data = vtk.vtkUnsignedCharArray()
    voxel_data.SetNumberOfComponents(1)
    voxel_data.SetArray(voxel.ravel(), voxel.size, 1)
    image_data.GetPointData().SetScalars(voxel_data)

    # Create a MarchingCubes filter and set the input data
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(image_data)

    # Set the iso value to 0.5 (i.e., the threshold for voxelization)
    marching_cubes.SetValue(0, 0.5)

    # Create a mapper and actor to display the voxel grid
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(marching_cubes.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.SetScale(0.5 / max(voxel.shape))

    # actor.GetProperty().SetColor(color)
    # actor.GetProperty().SetOpacity(opacity)
    return actor



def get_voxel_actor_vtk(voxel,color = (0,0,1.0),opacity = 1):
    # Get indices where voxel grid is equal to 1
    indices = np.where(voxel == 1)
    # Convert indices to coordinates
    coordinates = np.stack(indices, axis=1) - (voxel.shape[0] - 1) / 2.0

    # Create vtkPolyData object from coordinates
    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    # points.SetData(vtk.vtkFloatArray.FromArray(coordinates.astype(np.float32), 3))
    points.SetData(numpy_to_vtk(coordinates.astype(np.float32)))
    poly_data.SetPoints(points)


    # Create a vtkCubeSource to generate cubes for each voxel
    cube = vtk.vtkCubeSource()
    cube.SetXLength(1.0)
    cube.SetYLength(1.0)
    cube.SetZLength(1.0)

    # Create a vtkGlyph3D filter to place cubes at the center of each voxel
    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(poly_data)
    glyph.SetSourceConnection(cube.GetOutputPort())
    glyph.SetScaleModeToDataScalingOff()

    # Set scalar values for each cube to 1
    scalar_values = vtk.vtkIntArray()
    scalar_values.SetNumberOfTuples(coordinates.shape[0])
    scalar_values.FillComponent(0, 1)
    poly_data.GetPointData().SetScalars(scalar_values)

    # Create a mapper and actor to display the voxel grid
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    actor.SetScale(1 / max(voxel.shape))
    return actor


# ------------------- bboxes and bbox actors----------------
def get_bbox_center_vectors(box):
    center = box[:3]
    orientation = box[6]
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    vectors = np.diag(box[3:6] / 2.).dot(axis_rectified)
    return center,vectors

def world_box2cano_bbox(box):
    scale = np.max(box[3:6])
    cano_box = np.array([0, 0, 0, box[4] / scale, box[5] / scale, box[3] / scale, 0])
    return cano_box

def get_box_corners( center, vectors):
    '''
    Convert box center and vectors to the corner-form
    :param center:
    :param vectors:
    :return: corner points and faces related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

    corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
    corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)]

    return corner_pnts, faces

def set_bbox_line_actor(corners, faces, color):
    edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
    edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
    edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
    edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
    edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    pts = vtk.vtkPoints()
    for corner in corners:
        pts.InsertNextPoint(corner)

    lines = vtk.vtkCellArray()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    for edge in edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines.InsertNextCell(line)
        colors.InsertNextTuple3(*color)

    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(pts)
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(colors)

    return linesPolyData

# def get_bbox_line_actor( center, vectors, color, opacity, width=7):
#     if color.max()<1:
#         color *=255
#     corners, faces = get_box_corners(center, vectors)
#
#     mapper = vtk.vtkPolyDataMapper()
#     if vtk.VTK_MAJOR_VERSION <= 5:
#         mapper.SetInput(set_bbox_line_actor(corners, faces, color))
#     else:
#         mapper.SetInputData(set_bbox_line_actor(corners, faces, color))
#
#     bbox_actor = set_actor(mapper)
#     bbox_actor.GetProperty().SetOpacity(opacity)
#     bbox_actor.GetProperty().SetLineWidth(width)
#     # bbox_actor.GetProperty().SetColor(color)
#     return bbox_actor

def get_bbox_line_actor(box= np.array([0,0,0,1,1,1,0]),color=red,opacity=1,width =7):
    center ,vectors = get_bbox_center_vectors(box)
    if color.max() < 1:
        color *= 255
    corners, faces = get_box_corners(center, vectors)

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(set_bbox_line_actor(corners, faces, color))
    else:
        mapper.SetInputData(set_bbox_line_actor(corners, faces, color))

    bbox_actor = set_actor(mapper)
    bbox_actor.GetProperty().SetOpacity(opacity)
    bbox_actor.GetProperty().SetLineWidth(width)
    # bbox_actor.GetProperty().SetColor(color)
    return bbox_actor
# ------------------ axis and axis actors --------------
def set_arrow_actor( startpoint, vector):
    '''
    Design an actor to draw an arrow from startpoint to startpoint + vector.
    :param startpoint: 3D point
    :param vector: 3D vector
    :return: an vtk arrow actor
    '''
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipLength(0.2)
    arrow_source.SetTipRadius(0.08)
    arrow_source.SetShaftRadius(0.02)

    vector = vector / np.linalg.norm(vector) * 0.5

    endpoint = startpoint + vector

    # compute a basis
    normalisedX = [0 for i in range(3)]
    normalisedY = [0 for i in range(3)]
    normalisedZ = [0 for i in range(3)]

    # the X axis is a vector from start to end
    math = vtk.vtkMath()
    math.Subtract(endpoint, startpoint, normalisedX)
    length = math.Norm(normalisedX)
    math.Normalize(normalisedX)

    # the Z axis is an arbitrary vector cross X
    arbitrary = [0 for i in range(3)]
    arbitrary[0] = random.uniform(-10, 10)
    arbitrary[1] = random.uniform(-10, 10)
    arbitrary[2] = random.uniform(-10, 10)
    math.Cross(normalisedX, arbitrary, normalisedZ)
    math.Normalize(normalisedZ)

    # the Y axis is Z cross X
    math.Cross(normalisedZ, normalisedX, normalisedY)

    # create the direction cosine matrix
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i, 0, normalisedX[i])
        matrix.SetElement(i, 1, normalisedY[i])
        matrix.SetElement(i, 2, normalisedZ[i])

    # apply the transform
    transform = vtk.vtkTransform()
    transform.Translate(startpoint)
    transform.Concatenate(matrix)
    transform.Scale(length, length, length)

    # create a mapper and an actor for the arrow
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()

    mapper.SetInputConnection(arrow_source.GetOutputPort())
    actor.SetUserMatrix(transform.GetMatrix())
    actor.SetMapper(mapper)

    return actor
def get_arrow_actors(center,vectors,colors =None):
    '''

    :param center: [3]
    :param vectors: [N,3]
    :param colors:
    :return:
    '''

    actors = []
    for index in range(vectors.shape[0]):
        arrow_actor = set_arrow_actor(center, vectors[index])
        if colors is not None:
            arrow_actor.GetProperty().SetColor(colors[index])

        actors.append(arrow_actor)
    return actors

def get_one_arrow_actor(center,vector,color =(1,0,0)):
    '''

    :param center: [3]
    :param vector: [3]
    :param color:
    :return:
    '''
    arrow_actor = set_arrow_actor(center, vector)
    arrow_actor.GetProperty().SetColor(color)
    return arrow_actor

# -------------------- transformation -------------------
def get_rotation_matrix (theta, axis):
    if axis == 'x':
        rotation_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rotation_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        rotation_matrix = np.array(
            [[np.cos(theta), np.sin(-theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]])
    else:
        print('wrong axis:',axis)
        rotation_matrix = 'error'
    return rotation_matrix

def rotate(points,theta,axis):
    matrix = get_rotation_matrix(theta,axis)
    points = np.dot(points ,  matrix )
    return points

if __name__ == '__main__':
    pc_path = 'datasets/ShapeNetCore.v2.pc/03001627/f46ccdbf92b738e64b3c42e318f3affc'
    coords = np.load(os.path.join(pc_path,'coords.npy'))
    colors = np.load(os.path.join(pc_path,'colors.npy'))
    # vis_actors_vtk([
    #     get_pc_actor_vtk(coords)
    # ])
    vis_actors_vtk([
        get_colorful_pc_actor_vtk(pc_np = coords,point_colors=colors,opacity=1.0)
    ])


