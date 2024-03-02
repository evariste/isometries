
import os, numpy as np, xmltodict

from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray, vtkFloatArray, reference as vtkReference, vtkPoints
from vtkmodules.vtkFiltersCore import (vtkAppendPolyData, vtkTriangleFilter, vtkPolyDataNormals, vtkConnectivityFilter,
                                       vtkCleanPolyData, vtkStripper)
from vtkmodules.vtkCommonDataModel import vtkPointLocator, vtkCellLocator, vtkPolyData

from vtkmodules.vtkCommonCore import (VTK_CHAR, VTK_UNSIGNED_CHAR, VTK_SIGNED_CHAR,
                                      VTK_SHORT, VTK_UNSIGNED_SHORT,
                                      VTK_INT, VTK_UNSIGNED_INT,
                                      VTK_LONG, VTK_UNSIGNED_LONG, VTK_LONG_LONG,
                                      VTK_ID_TYPE,
                                      VTK_TYPE_INT64, VTK_TYPE_UINT64)

from vtkmodules.vtkCommonCore import VTK_FLOAT, VTK_DOUBLE
from vtkmodules.numpy_interface import dataset_adapter

from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)

from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

from vtkmodules.numpy_interface.dataset_adapter import numpy_support


VTK_INTEGER_TYPES = [VTK_CHAR, VTK_UNSIGNED_CHAR, VTK_SIGNED_CHAR,
                     VTK_SHORT, VTK_UNSIGNED_SHORT,
                     VTK_INT, VTK_UNSIGNED_INT,
                     VTK_LONG, VTK_UNSIGNED_LONG, VTK_LONG_LONG,
                     VTK_ID_TYPE,
                     VTK_TYPE_INT64, VTK_TYPE_UINT64]

VTK_FLOAT_TYPES = [VTK_FLOAT, VTK_DOUBLE]

colours_basic = {
    'black': [0, 0, 0],
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'magenta': [255, 0, 255],
    'cyan': [0, 255, 255],
    'white': [255, 255, 255],
}



def vtk_make_sphere(x, y, z, radius):
    sph_src = vtkSphereSource()
    sph_src.SetCenter(x, y, z)
    sph_src.SetRadius(radius)
    # noinspection PyArgumentList
    sph_src.Update()
    return sph_src.GetOutput()

def vtk_get_colour_array(polydata, colour):
    col_array = vtkUnsignedCharArray()
    col_array.SetNumberOfComponents(3)
    col_array.SetName("Colors")

    for k in range(polydata.GetNumberOfPoints()):
        col_array.InsertNextTuple3(*colours_basic[colour])

    return col_array

def vtk_get_coloured_sphere(x, y, z, radius, colour):
    sph = vtk_make_sphere(x, y, z, radius)
    col_array = vtk_get_colour_array(sph, colour)
    sph.GetPointData().SetScalars(col_array)
    return sph

def vtk_append_polydata(pds):
    assert len(pds) > 0
    append = vtkAppendPolyData()
    for pd in pds:
        append.AddInputData(pd)
    # noinspection PyArgumentList
    append.Update()
    return append


def run_triangle_filter(pd):

    tri_filter = vtkTriangleFilter()
    tri_filter.SetInputData(pd)
    # noinspection PyArgumentList
    tri_filter.Update()

    return tri_filter.GetOutput()


def run_normals_filter(pd):

    normals_filter = vtkPolyDataNormals()
    normals_filter.SetInputData(pd)
    normals_filter.SetSplitting(True)
    normals_filter.SetConsistency(True)
    normals_filter.SetFlipNormals(False)
    normals_filter.SetNonManifoldTraversal(True)
    normals_filter.SetComputePointNormals(True)
    normals_filter.SetComputeCellNormals(True)
    normals_filter.SetFeatureAngle(30)
    # noinspection PyArgumentList
    normals_filter.Update()

    return normals_filter.GetOutput()


def calc_polydata_cell_dists(polys_A, polys_B, scalars_name='cell_distance'):
    """
    For each point in polys_A, calculate distance to nearest cell in
    polys_B. Assign to scalar array in the polys_A.
    """

    n_pts_A = polys_A.GetNumberOfPoints()

    # Locator for source surface.
    cell_locator = vtkCellLocator()
    cell_locator.SetDataSet(polys_B)
    cell_locator.BuildLocator()

    # Where to store distances for each target point.
    dists_arr = vtkFloatArray()
    dists_arr.SetNumberOfComponents(1)
    dists_arr.SetNumberOfTuples(n_pts_A)
    dists_arr.SetName(scalars_name)

    # Variables for call to find closest points in source.
    c = [0.0, 0.0, 0.0]
    subId = vtkReference(0)
    dist2 = vtkReference(0.0)
    cellId = vtkReference(0)

    for k in range(n_pts_A):
        p = polys_A.GetPoint(k)
        cell_locator.FindClosestPoint(p, c, cellId, subId, dist2)
        dists_arr.SetTuple1(k, np.sqrt(dist2.get()))

    polys_A.GetPointData().AddArray(dists_arr)

    return

def polydata_point_dists_numpy(polys_A, polys_B):
    """
    For each point in polys_A, calculate the distance to the nearest point
    in polys_B, return as a numpy array.
    :param polys_A:
    :param polys_B:
    :return: distances, (N,) vector where N is the number of points in polys_A
    """

    array_name = 'point_distance'
    calc_polydata_point_dists(polys_A, polys_B, scalars_name=array_name)

    dists = polydata_get_pt_array_by_name(polys_A, array_name)

    polydata_remove_pt_array_by_name(polys_A, array_name)

    return dists

def calc_polydata_point_dists(polys_A, polys_B, scalars_name='point_distance'):
    """
    For each point in polys_A, calculate distance to nearest point in
    polys_B. Assign to scalar array in the target.
   """

    n_pts_A = polys_A.GetNumberOfPoints()

    # Locator for source surface.
    pt_locator = vtkPointLocator()
    pt_locator.SetDataSet(polys_B)
    pt_locator.SetNumberOfPointsPerBucket(1)
    pt_locator.BuildLocator()

    # Where to store distances for each target point.
    dists_arr = vtkFloatArray()
    dists_arr.SetNumberOfComponents(1)
    dists_arr.SetNumberOfTuples(n_pts_A)
    dists_arr.SetName(scalars_name)

    for k in range(n_pts_A):
        p = polys_A.GetPoint(k)
        ix_B = pt_locator.FindClosestPoint(p)
        q = polys_B.GetPoint(ix_B)
        disp = np.asarray(p) - np.asarray(q)
        d = np.sqrt(np.sum(disp * disp))
        dists_arr.SetTuple1(k, d)

    polys_A.GetPointData().AddArray(dists_arr)

    return

def polydata_nearest_point_IDs(points, pd):
    """
    :param points: Nx3 array of point coordinates
    :param pd: PolyData
    :return: List of N IDs of nearest point in pd to each of the points.
    """
    pts = points.copy()
    n_pts, dim = pts.shape
    if not (dim == 3):
        assert n_pts == 3, 'Points have invalid shape. Must be Nx3'
        pts = pts.T
        n_pts, dim = dim, n_pts

    pt_locator = vtkPointLocator()
    pt_locator.SetDataSet(pd)
    pt_locator.SetNumberOfPointsPerBucket(1)
    pt_locator.BuildLocator()

    pt_ids = []
    for k in range(n_pts):
        p = pts[k]
        idx = pt_locator.FindClosestPoint(p)
        pt_ids.append(idx)

    return pt_ids



def append_polydata(pd_list):
    n_polys = len(pd_list)
    appender = vtkAppendPolyData()
    appender.SetUserManagedInputs(True)
    appender.SetNumberOfInputs(n_polys)
    for k in range(n_polys):
        appender.SetInputDataByNumber(k, pd_list[k])
    # noinspection PyArgumentList
    appender.Update()
    return appender.GetOutput()

def vtkMatrix_to_numpy(vtk_mat):
    class_name = vtk_mat.GetClassName()
    if class_name == 'vtkMatrix4x4':
        sz = (4,4)
    elif class_name == 'vtkMatrix3x3':
        sz = (3,3)
    else:
        raise Exception('Type not supported.')

    ij = [(i, j) for i in range(sz[0]) for j in range(sz[1])]

    dat = np.zeros(sz)

    for i, j in ij:
        val = vtk_mat.GetElement(i, j)
        dat[i, j] = val

    return dat

def polydata_centroid(polydata):

    pts_npy = np.asarray(dataset_adapter.WrapDataObject(polydata).Points)
    centroid = np.mean(pts_npy, axis=0)

    return centroid

def polydata_bounds(pd):

    pts_npy = np.asarray(dataset_adapter.WrapDataObject(pd).Points)
    xlo, ylo, zlo = np.min(pts_npy, axis=0)
    xhi, yhi, zhi = np.max(pts_npy, axis=0)

    return xlo, xhi, ylo, yhi, zlo, zhi

def polydata_apply_affine(polydata_in, affine):
    """
    Apply an affine matrix transformation to all the points in a PolyData object.

    :param polydata_in: PolyData object.
    :param affine: 4x4 Homogeneous affine matrix.

    :return: New PolyData with transformed points.
    """

    n_tgt = polydata_in.GetNumberOfPoints()

    new_points = vtkPoints()
    new_points.SetNumberOfPoints(n_tgt)

    for k in range(n_tgt):
        p = polydata_in.GetPoint(k)
        x = np.asarray(p + (1,))
        y = affine.dot(x)
        new_points.SetPoint(k, y[:3])

    polydata_out = vtkPolyData()
    polydata_out.DeepCopy(polydata_in)
    polydata_out.SetPoints(new_points)

    return polydata_out

def get_polydata_components(pd, strip=False):
    conn = vtkConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToAllRegions()
    # noinspection PyArgumentList
    conn.Update()
    n_regions = conn.GetNumberOfExtractedRegions()

    conn = vtkConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToSpecifiedRegions()
    conn.InitializeSpecifiedRegionList()

    pd_components = []

    for region_id in range(n_regions):
        conn.AddSpecifiedRegion(region_id)
        # noinspection PyArgumentList
        conn.Update()
        comp = conn.GetOutput()

        clean = vtkCleanPolyData()
        clean.SetInputData(comp)
        # noinspection PyArgumentList
        clean.Update()
        comp_clean = clean.GetOutput()

        if strip:
            stripper = vtkStripper()
            stripper.SetInputData(comp_clean)
            # noinspection PyArgumentList
            stripper.Update()
            comp_clean_strip = stripper.GetOutput()
            pd_components.append(comp_clean_strip)
        else:
            pd_components.append(comp_clean)

        conn.DeleteSpecifiedRegion(region_id)

    return pd_components

def polydata_remove_pt_array_by_name(pd, array_name):
    """
    Remove a point data array with the given name.
    :param pd:
    :param array_name:
    """

    n_arrays = pd.GetPointData().GetNumberOfArrays()
    array_names = [pd.GetPointData().GetArrayName(k) for k in range(n_arrays)]

    if not array_name in array_names:
        print(f'Cannot remove array with name {array_name}. No such array.')
        return
    k = array_names.index(array_name)

    pd.GetPointData().RemoveArray(k)

    return

def polydata_get_pt_array_names(pd):
    """
    Get a list of names of all point data arrays (in order).
    :param pd:
    :return: List of names.
    """

    n_arrays = pd.GetPointData().GetNumberOfArrays()
    array_names = [pd.GetPointData().GetArrayName(k) for k in range(n_arrays)]

    return array_names

def polydata_get_pt_array_by_name(pd, array_name):
    """
    Retrieve a point data array with a given name.
    :param pd:
    :param array_name:
    :return: Numpy array containing the desired array, size (N,D) where N is
    the number of points in the PolyData and D is the dimension of the scalar
    array (number of components).
    """

    array_names = polydata_get_pt_array_names(pd)
    assert array_name in array_names, f'Cannot find array with name: {array_name}'

    k = array_names.index(array_name)
    return np.asarray(pd.GetPointData().GetArray(k))


def polydata_pts_to_numpy(pd):
    """
    Get a numpy array with the coordinates of the points in a
    PolyData object.

    :param pd: PolyData with N points.
    :return: Coordinates array with size (N,3).
    """

    xyz = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    return xyz

def polyline_pts_to_numpy(pl, return_ids=False):
    pt_ids = []
    pts = []
    n_pts = pl.GetNumberOfPoints()
    id_list = pl.GetPointIds()
    for k in range(n_pts):
        pt_id = id_list.GetId(k)
        pt_ids.append(pt_id)
        pts.append(pl.GetPoints().GetPoint(pt_id))

    if return_ids:
        return np.asarray(pts), pt_ids
    else:
        return np.asarray(pts)


def parse_paraview_camera_config(camera_config_file):

    assert os.path.isfile(camera_config_file), f'Cannot find camera config file: {camera_config_file}'

    with open(camera_config_file, 'r') as f:
        camera_config_str = f.read()

    camera_config = xmltodict.parse(camera_config_str)
    try:
        props = camera_config['CustomViewpointsConfiguration']['CustomViewpointButton0']['Configuration']['PVCameraConfiguration']['Proxy']['Property']
    except:
        raise Exception(f'Cannot parse camera config file {camera_config_file}')

    config = {}
    for prop in props:
        name = prop['@name']
        n_elements = int(prop['@number_of_elements'])
        assert n_elements > 0, 'Expect at least one element'
        elts = prop['Element']

        if n_elements == 1:
            val = float(elts['@value'])
        else:
            val = [float(e['@value']) for e in elts]

        config[name] = val

    if 'CameraParallelProjection' in config:
        config['CameraParallelProjection'] = bool(config['CameraParallelProjection'])


    return config


def set_camera_config(camera, camera_config):
    """
    :param camera: vtkCamera
    :param camera_config: Configuration dictionary (see parse_paraview_camera_config)
    :return:
    """
    pos = camera_config['CameraPosition']
    camera.SetPosition(*pos)

    focus = camera_config['CameraFocalPoint']
    camera.SetFocalPoint(*focus)

    view_up = camera_config['CameraViewUp']
    camera.SetViewUp(*view_up)

    view_angle = camera_config['CameraViewAngle']
    camera.SetViewAngle(view_angle)

    parallel_projection = camera_config['CameraParallelProjection']
    camera.SetParallelProjection(parallel_projection)

    parallel_scale = camera_config['CameraParallelScale']
    camera.SetParallelScale(parallel_scale)

    return

def render_window_to_image_file(file_name, render_window, rgba=True):
    """
    From https://kitware.github.io/vtk-examples/site/Python/IO/ImageWriter/

    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param file_name: The file name, if no extension then PNG is assumed.
    :param render_window: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    """

    img_writers = {
        '.bmp': vtkBMPWriter(),
        '.jpg': vtkJPEGWriter(),
        '.jpeg': vtkJPEGWriter(),
        '.pnm': vtkPNMWriter(),
        '.ps': vtkPostScriptWriter(),
        '.tiff': vtkTIFFWriter(),
        '.png': vtkPNGWriter()
    }


    # Select the writer to use.
    path, ext = os.path.splitext(file_name)
    ext = ext.lower()
    if not ext:
        ext = '.png'
        file_name = file_name + ext

    assert ext in img_writers, f'Unsupported image type: {ext}'
    writer = img_writers[ext]

    if ext == '.ps':
        rgba = False

    win_to_image = vtkWindowToImageFilter()
    win_to_image.SetInput(render_window)
    win_to_image.SetScale(1)  # image quality

    if rgba:
        win_to_image.SetInputBufferTypeToRGBA()
    else:
        win_to_image.SetInputBufferTypeToRGB()
        # Read from the front buffer.
        win_to_image.ReadFrontBufferOff()
        win_to_image.Update()

    writer.SetFileName(file_name)
    writer.SetInputConnection(win_to_image.GetOutputPort())
    writer.Write()
    return

