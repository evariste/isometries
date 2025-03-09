import os
import sys

import numpy as np

from isom.utils.general import apply_hom_matrix_to_points, cross_product, vecs_parallel
from isom.geom.objects import Glyph3D
from isom.geom.transform_3d import (
    random_ortho_rotation_3d, random_rotation_3d, OrthoRotation3D,
    compose_ortho_3d,
)

# TODO: Two step forms and composition.

def main():
    print('*' * 80)
    print(f'Running test {os.path.basename(__file__)}')
    test_matrices()

    test_inversion()

    test_null_rotation()

    test_quaternions()

    test_axis_calculation()

    return 0


def test_quaternions():
    print('    - Test composing via quaternions.')

    rot_A = random_ortho_rotation_3d()
    rot_B = random_ortho_rotation_3d()
    rot_C = compose_ortho_3d(rot_A, rot_B)

    q_A = rot_A.to_quaternion()
    q_B = rot_B.to_quaternion()
    q_C = rot_C.to_quaternion()

    q_B_q_A = q_B * q_A

    if q_B_q_A.real < 0:
        print('Flip quaternion so that real part is positive (-pi <= theta <= pi).')
        q_B_q_A *= -1.0

    assert np.isclose(q_B_q_A, q_C), 'Quaternions do not match.'


def test_axis_calculation():
    print('    - Test calculating the axis of a composition manually.')

    rot_A = random_ortho_rotation_3d()
    rot_B = random_ortho_rotation_3d()

    rot_C = compose_ortho_3d(rot_A, rot_B)

    axis_A = rot_A.axis
    theta_A = rot_A.angle

    axis_B = rot_B.axis
    theta_B = rot_B.angle

    # Axis calculated from scratch.
    c_theta_half = float(np.cos(theta_A / 2.0))
    c_phi_half = float(np.cos(theta_B / 2.0))

    s_theta_half = float(np.sin(theta_A / 2.0))
    s_phi_half = float(np.sin(theta_B / 2.0))

    axis_calc = (
            c_theta_half * s_phi_half * axis_B +
            c_phi_half * s_theta_half * axis_A +
            s_theta_half * s_phi_half * cross_product(axis_B, axis_A)
    )

    # C = B A
    axis_C = rot_C.axis

    success = vecs_parallel(axis_C, axis_calc)
    assert success, 'Expect axes to be aligned.'

    return



def test_null_rotation():
    print('    - Test null rotation.')
    axis = [1, 0, 0]
    angle = 0
    rot = OrthoRotation3D(axis, angle)

    glyph = Glyph3D()
    pts = glyph.points

    pts2 = rot.apply(pts)

    assert np.allclose(pts, pts2), 'Do not expect change in points.'

    return

def test_inversion():
    print('    - Test inversion.')
    glyph = Glyph3D()
    pts = glyph.points

    ortho_rot = random_ortho_rotation_3d()
    axis = ortho_rot.axis
    theta = ortho_rot.angle

    ortho_rot_B = OrthoRotation3D(axis, -1.0 * theta)

    pts_B = ortho_rot.apply(pts)
    pts_B = ortho_rot_B.apply(pts_B)
    assert np.allclose(pts, pts_B), 'Expected inverse failed.'

    ortho_rot_C = OrthoRotation3D(-1.0 * axis, theta)

    pts_C = ortho_rot.apply(pts)
    pts_C = ortho_rot_C.apply(pts_C)
    assert np.allclose(pts, pts_C), 'Expected inverse failed.'


    return


def test_matrices():
    print('    - Test matrices equivalent.')
    glyph = Glyph3D()
    pts = glyph.points

    ortho_rot = random_ortho_rotation_3d()
    M_ortho_rot = ortho_rot.get_matrix()

    pts_B = glyph.apply_transformation(ortho_rot).points
    pts_C = apply_hom_matrix_to_points(M_ortho_rot, pts)
    assert np.allclose(pts_B, pts_C), 'Result from matrix multiplication differ'

    rot = random_rotation_3d()
    M_rot = rot.get_matrix()

    pts_D = glyph.apply_transformation(rot).points
    pts_E = apply_hom_matrix_to_points(M_rot, pts)
    assert np.allclose(pts_D, pts_E),  'Result from matrix multiplication differ'

    return


if __name__ == '__main__':
    sys.exit(main())