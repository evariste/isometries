import sys

import numpy as np

from isom.utils.general import apply_hom_matrix_to_points
from isom.geom.objects import Glyph3D
from isom.geom.transform_3d import random_ortho_rotation_3d, random_rotation_3d, OrthoRotation3D

# TODO: Two step forms and composition.

def main():

    test_matrices()

    test_inversion()

    test_null_rotation()

    return 0

def test_null_rotation():
    axis = [1, 0, 0]
    angle = 0
    rot = OrthoRotation3D(axis, angle)

    glyph = Glyph3D()
    pts = glyph.points

    pts2 = rot.apply(pts)

    assert np.allclose(pts, pts2), 'Do not expect change in points.'

    return

def test_inversion():

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