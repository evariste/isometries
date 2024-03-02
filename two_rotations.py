"""
Created by: Paul Aljabar
Date: 08/04/2023
"""
import os
import sys, numpy as np
from matplotlib import pyplot as plt
from colour import get_Nth_colour, get_color_list
from vtk_io import make_vtk_polydata, polydata_save
from utilities import ensure_vec_3d, nearest_point_on_line
from objects import Glyph2D, Glyph3D, Line
from transform import Rotation, compose_rotatations


def main():

    # example_2d()
    example_3d()

    return 0

def example_3d():
    
    working_dir = 'output'
    os.makedirs(working_dir, exist_ok=True)

    glyph0 = Glyph3D()


    cent_A = [0, -10, 0]
    cent_B = [5, 0, 5]

    angle_A = np.pi / 4.0
    angle_B = np.pi / 3

    axis_A = [1, -1, 0]
    axis_B = [1, 1, 2]

    rot_1 = Rotation(cent_A, axis_A, angle_A)
    rot_2 = Rotation(cent_B, axis_B, angle_B)

    glyph1 = glyph0.apply_transformation(rot_1)
    glyph2 = glyph1.apply_transformation(rot_2)

    tr_comp = compose_rotatations(rot_1, rot_2)
    glyph3 = glyph0.apply_transformation(tr_comp)
    glyph4 = glyph0.apply_transformation(tr_comp, t=0.5)


    u = ensure_vec_3d(glyph0.points[:,0])
    L = Line(rot_1.centre, rot_1.axis)
    v = nearest_point_on_line(L, u)
    w = ensure_vec_3d(glyph1.points[:,0])
    y = v - 10 * L.direction
    z = v + 10 * L.direction
    uvw = np.hstack([u, v, w, y, z])
    cell_info = [[0,1], [1,2], [3,4]]
    pd = make_vtk_polydata(uvw.T, cell_info, lines=True)
    polydata_save(pd, f'{working_dir}/lines1.vtk')



    u = ensure_vec_3d(glyph1.points[:,0])
    L = Line(rot_2.centre, rot_2.axis)
    v = nearest_point_on_line(L, u)
    w = ensure_vec_3d(glyph2.points[:,0])
    y = v - 10 * L.direction
    z = v + 10 * L.direction
    uvw = np.hstack([u, v, w, y, z])
    cell_info = [[0,1], [1,2], [3,4]]
    pd = make_vtk_polydata(uvw.T, cell_info, lines=True)
    polydata_save(pd, f'{working_dir}/lines2.vtk')

    u = ensure_vec_3d(glyph0.points[:, 0])
    L = Line(tr_comp.rot.centre, tr_comp.rot.axis)
    uu = nearest_point_on_line(L, u)

    w = ensure_vec_3d(glyph4.points[:, 0])
    ww = nearest_point_on_line(L, w)

    y = ww - 10 * L.direction
    z = ww + 10 * L.direction

    p = ensure_vec_3d(glyph3.points[:,0])
    pp = nearest_point_on_line(L, p)

    dat = np.hstack([u, uu, w, ww, y, z, p, pp])
    cell_info = [[0,1], [2,3], [4,5], [6,7]]
    pd = make_vtk_polydata(dat.T, cell_info, lines=True)
    polydata_save(pd, f'{working_dir}/lines3.vtk')






    glyph0.save(f'{working_dir}/glyph0.vtk')
    glyph1.save(f'{working_dir}/glyph1.vtk')
    glyph2.save(f'{working_dir}/glyph2.vtk')

    glyph3.save(f'{working_dir}/glyph3.vtk')

    glyph4.save(f'{working_dir}/glyph4.vtk')

    return 0



def example_2d():
    glyph0 = Glyph2D()

    fig, ax = plt.subplots()

    z_axis = [0, 0, 1]
    cent_A = np.random.rand(2) * 10
    cent_B = np.random.rand(2) * 10

    angle_A = np.random.rand() * np.pi * 2.0 - np.pi
    angle_B = np.random.rand() * np.pi * 2.0 - np.pi
    # rotA = Rotation([4, 0], [0, 0, 1], np.pi / 3.0)
    # rotB = Rotation([6,3],[0,0,1],np.pi/2.0)

    rotA = Rotation(cent_A, z_axis, angle_A)
    rotB = Rotation(cent_B, z_axis, angle_B)

    glyph1 = glyph0.apply_transformation(rotA)

    glyph2 = glyph1.apply_transformation(rotB)

    # translate = Translation([4,5])
    # glyph4 = glyph0.apply_transformation(translate)
    # glyphs = [glyph0, glyph1, glyph2, glyph4]


    glyphs = [glyph0, glyph1, glyph2]

    colors = get_color_list(len(glyphs))
    labels = ['0', '1', '2']
    patches = [g.get_patch(facecolor=c, label=l) for g,c,l in zip(glyphs, colors, labels)]

    for p in patches:
        ax.add_patch(p)

    n_glyphs = len(glyphs)


    rotC = compose_rotatations(rotA, rotB)
    glyph5 = glyph0.apply_transformation(rotC)
    n_glyphs += 1
    c = get_Nth_colour(n_glyphs)
    p = glyph5.get_patch(c)

    p.set_alpha(0.2)
    p.set_linestyle(':')
    p.set_linewidth(2)
    p.set_edgecolor('k')
    p.set_label('direct')
    ax.add_patch(p)

    ax.axis('equal')

    cent_C = rotC.centre.flatten()[:2]
    angle_C = rotC.angle
    axis_C = rotC.axis
    print(f'Angle C: {angle_C:0.2f}')
    print(f'Axis C: {axis_C}')


    ax.plot(*cent_A, 'x', label='cent_A')
    ax.plot(*cent_B, 'x', label='cent_B')
    ax.plot(*cent_C, 'x', label='cent_C')

    ax.legend()
    msg = f'angles: A: {angle_A:0.2f}  B: {angle_B:0.2f}'
    ax.set_title(msg)

    plt.show()
    return 0




if __name__ == '__main__':
    sys.exit(main())
