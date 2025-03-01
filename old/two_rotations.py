"""
Created by: Paul Aljabar
Date: 08/04/2023
"""
import os
import sys, numpy as np
from matplotlib import pyplot as plt
from isom.utils.colour import get_Nth_colour, get_color_list
from isom.io.vtk import make_vtk_polydata, polydata_save
from isom.utils.general import (
    ensure_vec_3d, nearest_point_on_line, validate_pts, 
    rotation_matrix_from_axis_and_angle, 
    axis_from_rotation_matrix, angle_from_rotation_matrix
)
from isom.geom.objects import Glyph2D, Glyph3D, Line3D
from isom.geom.transform import Transform



class Translation(Transform):
    def __init__(self, t):
        super().__init__()
        self.vec = ensure_vec_3d(t)

    def apply(self, points):
        pts_out = validate_pts(points)
        pts_out = pts_out + self.vec

        dim, _ = points.shape
        if dim == 2:
            return pts_out[:2]

        return pts_out

    def get_matrix(self):
        M = np.eye(4)
        # [I v]
        # [0 1]
        M[:3, 3:] = self.vec
        return M


class Rotation(Transform):
    def __init__(self, centre, axis, angle):
        super().__init__()
        self.centre = ensure_vec_3d(centre)
        self.axis = ensure_vec_3d(axis)
        self.angle = angle

    def __repr__(self):
        c = np.round(self.centre.flatten(), 2)
        ax = np.round(self.axis.flatten(), 2)
        ang = np.round(self.angle, 2)
        return f'Rotation(\n{c},\n {ax},\n {ang}\n)'

    def apply(self, points):

        pts_out = validate_pts(points)

        R = rotation_matrix_from_axis_and_angle(self.axis, self.angle)

        pts_out = pts_out - self.centre
        pts_out = R @ pts_out
        pts_out = pts_out + self.centre
        # pts_out = self.R @ pts_out + (np.eye(3) - self.R) @ self.centre

        dim, _ = points.shape
        if dim == 2:
            return pts_out[:2]

        return pts_out

    def get_matrix(self):
        M = np.eye(4)

        # [I t] [R 0] [I -t]
        # [0 1] [0 1] [0  1]
        #
        # [R t] [I -t]
        # [0 1] [0  1]
        #
        # [R  -Rt + t ]
        # [0      1   ]
        R = rotation_matrix_from_axis_and_angle(self.axis, self.angle)
        M[:3, :3] = R
        M[:3, 3:] = -1.0 * R @ self.centre + self.centre

        return M


class Screw(Transform):
    def __init__(self, centre, axis, angle, translate_dist):
        """
        Screw transformation: Combination of rotation and translation along
        a single axis.

        @param centre: Point on rotation axis
        @param axis: of rotation
        @param angle: of rotation
        @param translate_dist: (signed) distance along axis to perform translation.
        """
        super().__init__()

        self.rot = Rotation(centre, axis, angle)

        self.tra = Translation(self.rot.axis * translate_dist)
        return

    def get_matrix(self):
        M1 = self.rot.get_matrix()
        M2 =self.tra.get_matrix()
        return M2 @ M1




    def apply(self, points):
        pts_out = validate_pts(points)
        pts_out = self.rot.apply(pts_out)
        pts_out = self.tra.apply(pts_out)
        return pts_out



def compose_rotatations(rot_A, rot_B):
    """
    Generate a rotation rot_C such that rot_C (x) = rot_B ( rot_A (x) )

    """

    M_A = rot_A.get_matrix()
    M_B = rot_B.get_matrix()
    M = M_B @ M_A

    axis = axis_from_rotation_matrix(M[:3, :3])

    v = M[:3, 3:]

    v_along = (axis @ v) * axis.T
    v_perp = v - v_along

    M_rot = M.copy()
    M_rot[:3, 3:] = v_perp

    Z = M_rot - np.eye(4)
    Z[3, :3] = axis

    #
    centre, _, _, _ = np.linalg.lstsq(Z[:, :3], -Z[:, 3], rcond=None)

    cent_hom = np.hstack((centre, 1)).T
    assert np.allclose(M_rot @ cent_hom, cent_hom), 'Error in finding centre.'

    angle = angle_from_rotation_matrix(M[:3, :3])

    rot = Rotation(centre, axis, angle)


    if not np.allclose(rot.get_matrix(), M_rot):
        # Try inverting the axis
        axis *= -1.0
        rot = Rotation(centre, axis, angle)
        assert np.allclose(rot.get_matrix(), M_rot), 'Error finding rotation.'

    translate_dist = np.sqrt(np.sum(v_along * v_along))

    if np.isclose(translate_dist, 0):
        # Composition is a rotation.
        return rot

    # Composition is a screw transformation.
    screw = Screw(centre, axis, angle, translate_dist)

    if not np.allclose(screw.get_matrix(), M):
        # Try inverting the translation
        translate_dist *= -1.0
        screw = Screw(centre, axis, angle, translate_dist)
        assert np.allclose(screw.get_matrix(), M), 'Error finding screw transform.'

    return screw

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
    glyph4 = glyph0.apply_transformation(tr_comp)


    u = ensure_vec_3d(glyph0.points[:,0])
    L = Line3D(rot_1.centre, rot_1.axis)
    v = nearest_point_on_line(L, u)
    w = ensure_vec_3d(glyph1.points[:,0])
    y = v - 10 * L.direction
    z = v + 10 * L.direction
    uvw = np.hstack([u, v, w, y, z])
    cell_info = [[0,1], [1,2], [3,4]]
    pd = make_vtk_polydata(uvw.T, cell_info, lines=True)
    polydata_save(pd, f'{working_dir}/lines1.vtk')



    u = ensure_vec_3d(glyph1.points[:,0])
    L = Line3D(rot_2.centre, rot_2.axis)
    v = nearest_point_on_line(L, u)
    w = ensure_vec_3d(glyph2.points[:,0])
    y = v - 10 * L.direction
    z = v + 10 * L.direction
    uvw = np.hstack([u, v, w, y, z])
    cell_info = [[0,1], [1,2], [3,4]]
    pd = make_vtk_polydata(uvw.T, cell_info, lines=True)
    polydata_save(pd, f'{working_dir}/lines2.vtk')

    u = ensure_vec_3d(glyph0.points[:, 0])
    L = Line3D(tr_comp.rot.centre, tr_comp.rot.axis)
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
