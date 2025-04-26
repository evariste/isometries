"""
Decompose a specific Twist into a pair of rotations in 3D.
"""
import os
import sys
import numpy as np
from isom.geom.transform_3d import Rotation3D, Twist3D, OrthoRotation3D, compose_ortho_3d, Translation3D, compose_3d

from isom.geom.objects import Plane3D, Glyph3D, Line3D, get_glyph_bounds, LineSegment
from isom.utils.general import ensure_vec, vecs_perpendicular


def main():


    rot_1, rot_2, twist = get_transforms()

    create_glyphs(rot_1, rot_2, twist)

    return 0


def get_transforms():

    # Some orthogonal rotations, these will later be used to generate general rotations.

    # Rotation 1
    axis_1 = [1, 1, 1]
    theta_1 = np.pi / 3.0
    orot_1 = OrthoRotation3D(axis_1, theta_1)

    # Rotation part of final composition.
    z_axis = [0, 0, 1]
    theta_3 = np.pi / 3.0
    orot_3 = OrthoRotation3D(z_axis, theta_3)

    # We want to find orot_2 such that orot_3 = orot_2 o orot_1

    orot_2 = compose_ortho_3d(
        orot_1.inverse(),
        orot_3
    )

    # Set rot_1 = t_1 orot_1 and rot_2 = t_2 orot_2


    # Want the composition rot_2 rot_1 to be a twist along the z-axis


    axis_1 = orot_1.axis
    axis_2 = orot_2.axis

    # t_1 must be perpendicular to axis_1, same for t_2, axis_2.

    origin = [0, 0, 0]
    # Plane perp. to axis_1.
    plane_1 = Plane3D(axis_1, origin)

    # Plane perp. to axis_2.
    plane_2 = Plane3D(axis_2, origin)

    u, v = plane_1.get_basis()

    n_2 = plane_2.normal

    # a u + b v gives a point on plane 1. This can represent translation t_1.

    # The composition is
    #  t_2 orot_2  t_1 orot_1

    # This can be re-written
    # t_2 t_3  orot_2 orot_1
    # where t_3 is the rotation of t_1 under orot_2

    # We want to have t_2 t_3 result in a displacement to a point (0, 0, k) on the z-axis.

    # The plane containing t_1 can be rotated to give the plane containing t_3.

    # a ur + b vr gives a point for t_3

    ur = orot_2.apply(u)
    vr = orot_2.apply(v)


    # Working through gives a ur . n_2 + b vr . n_2 = k n_2_z

    n_2_z = n_2[-1]

    ur_dot_n = np.sum(ur * n_2)
    vr_dot_n = np.sum(vr * n_2)

    twist_displacement = 8

    assert np.abs(ur_dot_n) > 0

    b = 2

    a = (twist_displacement * n_2_z - b * vr_dot_n) / ur_dot_n

    pt_1_r = a * ur + b * vr

    Z = ensure_vec([0, 0, twist_displacement])
    pt_2 = Z - pt_1_r


    # Now write pt_1 in terms of u and v
    pt_1 = a * u + b * v


    t_1 = Translation3D(pt_1)
    t_2 = Translation3D(pt_2)

    rot_1 = Rotation3D.from_two_step_form(orot_1, t_1)
    rot_2 = Rotation3D.from_two_step_form(orot_2, t_2)

    tr_composed = compose_3d(rot_1, rot_2)

    assert isinstance(rot_1, Rotation3D)
    assert isinstance(rot_2, Rotation3D)
    assert isinstance(tr_composed, Twist3D)



    M_1 = rot_1.get_matrix()
    M_2 = rot_2.get_matrix()
    M_composed = tr_composed.get_matrix()

    assert np.allclose(M_composed, M_2 @ M_1)

    assert np.isclose(twist_displacement, tr_composed.displacement)


    return rot_1, rot_2, tr_composed

def create_glyphs(rot_1: Rotation3D, rot_2: Rotation3D, twist: Twist3D):
    """
    The twist is rotation rot_1 followed by rot_2.
    """

    K = twist.displacement

    g_0 = Glyph3D()

    g_1 = g_0.apply_transformation(rot_1)

    g_2 = g_1.apply_transformation(rot_2)

    g_composed = g_0.apply_transformation(twist)


    assert np.allclose(g_composed.points, g_2.points)

    bounds = get_glyph_bounds([g_0, g_1, g_2])

    x0, y0, z0, x1, y1, z1 = bounds

    xc = (x0 + x1) / 2.0
    yc = (y0 + y1) / 2.0
    zc = (z0 + z1) / 2.0

    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    rz = (z1 - z0) / 2.0

    pad = 6.5

    x0 = xc - pad * rx
    x1 = xc + pad * rx
    y0 = yc - pad * ry
    y1 = yc + pad * ry
    z0 = zc - pad * rz
    z1 = zc + pad * rz


    plane_x0 = Plane3D([1, 0, 0], [x0, 0, 0])
    plane_x1 = Plane3D([1, 0, 0], [x1, 0, 0])
    plane_y0 = Plane3D([0, 1, 0], [0, y0, 0])
    plane_y1 = Plane3D([0, 1, 0], [0, y1, 0])
    plane_z0 = Plane3D([0, 0, 1], [0, 0, z0])
    plane_z1 = Plane3D([0, 0, 1], [0, 0, z1])

    planes_box = [plane_x0, plane_y0, plane_z0, plane_x1, plane_y1, plane_z1]
    line_1 = Line3D(rot_1.point, rot_1.ortho_rot.axis)
    line_2 = Line3D(rot_2.point, rot_2.ortho_rot.axis)

    bounds_lo = ensure_vec([x0, y0, z0])
    bounds_hi = ensure_vec([x1, y1, z1])

    seg_1_pts = []
    for plane in planes_box:
        try:
            p = plane.intersection(line_1)
        except:
            continue
        if np.all(p >= bounds_lo) and np.all(p <= bounds_hi):
            seg_1_pts.append(p)

    assert len(seg_1_pts) == 2



    seg_2_pts = []
    for plane in planes_box:
        try:
            p = plane.intersection(line_2)
        except:
            continue
        if np.all(p >= bounds_lo) and np.all(p <= bounds_hi):
            seg_2_pts.append(p)

    assert len(seg_2_pts) == 2

    seg_1 = LineSegment(seg_1_pts[0], seg_1_pts[1])
    seg_2 = LineSegment(seg_2_pts[0], seg_2_pts[1])
    seg_0 = LineSegment([0, 0, -3], [0, 0, K+6])

    out_dir = 'output/two_rotations_3d'
    os.makedirs(out_dir, exist_ok=True)

    g_0.save(f'{out_dir}/g_0.vtk')
    g_1.save(f'{out_dir}/g_1.vtk')
    g_2.save(f'{out_dir}/g_2.vtk')

    seg_0.save(f'{out_dir}/axis_0.vtk', width=0.1)
    seg_1.save(f'{out_dir}/axis_1.vtk', width=0.1)
    seg_2.save(f'{out_dir}/axis_2.vtk', width=0.1)

    return


if __name__ == '__main__':
    sys.exit(main())
