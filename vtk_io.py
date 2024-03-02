"""
Created by: Paul Aljabar
Date: 21/10/2020
"""

import os
import numpy as np
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter, vtkPolyDataReader
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints, vtkIntArray, vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPolygon
from vtkmodules.vtkFiltersCore import vtkTriangleFilter, vtkPolyDataNormals
from vtkmodules.vtkIOGeometry import vtkBYUReader, vtkOBJReader, vtkSTLReader, vtkBYUWriter, vtkOBJWriter, vtkSTLWriter
from vtkmodules.vtkIOPLY import vtkPLYReader, vtkPLYWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter


########################################################

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

def polydata_save(poly_data, file_name, clobber=True):

    """
    Use extension of file name to choose a suitable writer for saving
    a PolyData object.
    :param poly_data: PolyData object.
    :param file_name: File to save to.
    :param clobber: Overwrite output if it already exists.
    """

    if (not clobber) and os.path.isfile(file_name):
        print('polydata_save: File exists')
        return

    path, ext = os.path.splitext(file_name)
    ext = ext.lower()

    if ext == '.ply':
        writer = vtkPLYWriter()
        writer.SetFileName(file_name)
    elif ext == '.vtp':
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(file_name)
    elif ext == '.obj':
        writer = vtkOBJWriter()
        writer.SetFileName(file_name)
    elif ext == '.stl':
        writer = vtkSTLWriter()
        writer.SetFileName(file_name)
    elif ext == '.vtk':
        writer = vtkPolyDataWriter()
        writer.SetFileName(file_name)
    elif ext == '.g':
        writer = vtkBYUWriter()
        writer.SetGeometryFileName(file_name)
    else:
        raise Exception(f'Unsupported file format: {ext}. Exiting.')

    try:
        writer.SetInputData(poly_data)
        writer.Update()
    except:
        raise Exception(f'Cannot write {file_name}. Exiting.')

    print(f'Saved PolyData file: {file_name}')

    del writer

    return

########################################################

def polydata_read(in_file):
    """
    Given an input file, use suffix to decide which VTK reader to use.
    Apply the reader to load the data, returning a PolyData object.
    :param in_file:  Input file.
    :return: PolyData.
    """

    assert os.path.isfile(in_file), f'Cannot find input file {in_file}'

    path, ext = os.path.splitext(in_file)
    ext = ext.lower()

    if ext == '.ply':
        reader = vtkPLYReader()
        reader.SetFileName(in_file)
    elif ext == '.vtp':
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(in_file)
    elif ext == '.obj':
        reader = vtkOBJReader()
        reader.SetFileName(in_file)
    elif ext == '.stl':
        reader = vtkSTLReader()
        reader.SetFileName(in_file)
    elif ext == '.vtk':
        reader = vtkPolyDataReader()
        reader.SetFileName(in_file)
    elif ext == '.g':
        reader = vtkBYUReader()
        reader.SetGeometryFileName(in_file)
    else:
        raise Exception(f'Unsupported file format: {ext}. Exiting.')


    try:
        reader.Update()
    except:
        raise Exception('Cannot read input file. Exiting.')

    return reader.GetOutput()

########################################################

def make_vtk_ID_list(it):
    # Iterable to a vtk ID list.

    vtk_id_list = vtkIdList()
    for i in it:
        vtk_id_list.InsertNextId(int(i))

    return vtk_id_list

########################################################

def save_legacy_vtk_polydata(pts_list, faces, out_file):
    """
    Save in a (very) old VTK format.

    NB. Assumes constant number of points per face.

    E.g.

    # vtk DataFile Version 2.0
    Cube example
    ASCII
    DATASET POLYDATA
    POINTS 8 float
    0.0 0.0 0.0
    1.0 0.0 0.0
    1.0 1.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    1.0 0.0 1.0
    1.0 1.0 1.0
    0.0 1.0 1.0
    POLYGONS 6 30
    4 0 1 2 3
    4 4 5 6 7
    4 0 1 5 4
    4 2 3 7 6
    4 0 4 7 3
    4 1 2 6 5

    :param out_file: What to save.
    :param pts_list: List of iterables (each of length three).
    :param faces: Numpy array. Shape is n_faces x pts per face.
    :return:
    """

    no_of_pts = len(pts_list)

    n_faces = len(faces)
    n_face_entries = 0
    pts_per_face = 0

    if n_faces > 0:
        n_faces, pts_per_face = faces.shape
        n_face_entries = (1 + pts_per_face) * n_faces

    header = """# vtk DataFile Version 2.0
Polydata
ASCII
DATASET POLYDATA
"""
    with open(out_file, 'w') as f_handle:

        f_handle.write(header)
        f_handle.write(f'POINTS {no_of_pts} FLOAT\n')
        for p in pts_list:
            s = '{:f} {:f} {:f}\n'.format(p[0], p[1], p[2])
            f_handle.write(s)

        if n_faces < 1:
            return

        f_handle.write(f'POLYGONS {n_faces} {n_face_entries}\n')
        for face in faces:
            s = ['{:d}'.format(pts_per_face)]
            s += [' {:d}'.format(k) for k in face]
            s = ''.join(s)
            s += '\n'
            f_handle.write(s)


    return

########################################################

def make_vtk_polydata(pts_list, cell_info, lines=False):
    """

    :param pts_list: List of points, or Numpy array of shape Nx3
    :param cell_info: Iterable whose elements are lists (or iterables)
                     to define the point indices for each cell. E.g.,
                     list of lists or 2-D Numpy array (if every cell
                     has the same number of points).
    :param lines: Set to true if the cells only contain lines (no faces).
    :return: Polydata object.
    """
    #
    # No scalars specified, save ID as per-point data.

    pd = vtkPolyData()

    pts_vtk = vtkPoints()
    # noinspection PyArgumentList
    cells = vtkCellArray()

    no_of_pts = len(pts_list)

    for n in range(no_of_pts):
        pts_vtk.InsertPoint(n, pts_list[n])


    pt_id_scalar_array = make_vtk_scalar_array(np.arange(no_of_pts),
                                               'Point ID')

    for pt_ids in cell_info:
        cells.InsertNextCell(make_vtk_ID_list(pt_ids))

    pd.SetPoints(pts_vtk)

    if lines:
        pd.SetLines(cells)
    else:
        pd.SetPolys(cells)

    pd.GetPointData().SetScalars(pt_id_scalar_array)

    return pd

########################################################

def make_vtk_scalar_array(np_array: np.ndarray, name: str):
    dt_kind = np_array.dtype.kind
    if dt_kind in 'iu':
        vtk_arr = vtkIntArray()
    elif dt_kind == 'f':
        vtk_arr = vtkFloatArray()
    else:
        raise Exception('Unsupported kind of data type.')

    n_pts = np_array.size
    dat = np_array.flatten()

    for k in range(n_pts):
        vtk_arr.InsertTuple1(k, dat[k])

    vtk_arr.SetName(name)

    return vtk_arr



########################################################

def vtk_polydata_from_contour_components(components):
    """
    Take a set of components representing contour polygons and create a vtkPolyData
    object to contain them.
    """

    component_sizes = [c.shape[0] for c in components]
    n_pts_total = sum(component_sizes)

    n_components = len(components)

    # noinspection PyArgumentList
    cellArr = vtkCellArray()
    cellArr.InitTraversal()

    pts = vtkPoints()

    pts.SetNumberOfPoints(n_pts_total)

    ix_pt = 0

    for k in range(n_components):
        pol = vtkPolygon()
        c = components[k]
        n_pts = c.shape[0]
        pol.GetPointIds().SetNumberOfIds(n_pts)

        for n in range(n_pts):
            pol.GetPointIds().SetId(n, ix_pt)
            pts.InsertPoint(ix_pt, (c[n][0], c[n][1], c[n][2]))
            ix_pt += 1

        cellArr.InsertNextCell(pol)

    # Polydata containing polygons
    pd = vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cellArr)

    return pd


def triangulate_polydata(pd):
    tri_filt = vtkTriangleFilter()
    tri_filt.SetInputData(pd)
    # noinspection PyArgumentList
    tri_filt.Update()
    pd_tri = tri_filt.GetOutput()
    return pd_tri


def vtk_get_cells_as_list(pd):

    cells_list = []
    n_cells = pd.GetNumberOfCells()

    for n in range(n_cells):
        cell = pd.GetCell(n)
        n_pts = cell.GetNumberOfPoints()
        pt_ids = cell.GetPointIds()

        cell_info = [n_pts]
        for k in range(n_pts):
            cell_info.append(pt_ids.GetId(k))

        cells_list.append(cell_info)

    return cells_list

