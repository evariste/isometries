
import sys
import numpy as np
from twor.geom.transform_2d import (
    Rotation2D, OrthoRotation2D, OrthoReflection2D, compose_ortho_2d, OrthoTransform2D, ortho2D_to_reflections
)
from twor.geom.objects import Glyph2D
from twor.utils.general import apply_hom_matrix_to_points, apply_transform_sequence_to_glyph

def main():

    test_rotation()

    test_reflection_decomp()

    run_composition_tests()

    test_inverses()

    print('Done.')

    return 0

def test_rotation():
    # Random general rotation.
    alpha = np.random.rand() * 2.0 * np.pi
    C = np.random.rand(2) * 10
    rot = Rotation2D(C, alpha)

    #############################################################################

    # Apply to glyph
    glyph = Glyph2D()
    glyph_rot = glyph.apply_transformation(rot)

    M = rot.get_matrix()
    pts_v2 = apply_hom_matrix_to_points(M, glyph.points)

    assert np.allclose(pts_v2, glyph_rot.points), 'Result from homogeneous matrix and direct transformation do not match.'

    return

def test_reflection_decomp():

    alpha = np.random.rand() * 2.0 * np.pi
    # Random orthogonal rotation
    ortho_rot = OrthoRotation2D(alpha)

    glyph = Glyph2D()

    # Decompose to reflections.
    reflections = ortho_rot.get_reflections()

    glyph_rotate = glyph.apply_transformation(ortho_rot)

    glyph_2refls = apply_transform_sequence_to_glyph(reflections, glyph)

    assert np.allclose(glyph_rotate.points, glyph_2refls.points), 'Results from reflections and direct orthogonal transformation do not match.'

    # Decompose using general function
    reflections_B = ortho2D_to_reflections(ortho_rot)

    glyph_B = apply_transform_sequence_to_glyph(reflections_B, glyph)

    assert np.allclose(glyph_rotate.points, glyph_B.points), 'Results from reflections and direct orthogonal transformation do not match.'


    # Random reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl = OrthoReflection2D(v)

    reflections_C = ortho2D_to_reflections(ortho_refl)

    assert len(reflections_C) == 1, 'Unexpected number of reflections.'
    refl_C = reflections_C[0]

    glyph_C = glyph.apply_transformation(refl_C)
    glyph_D = glyph.apply_transformation(ortho_refl)
    assert np.allclose(glyph_C.points, glyph_D.points), 'Results from reflections and direct orthogonal transformation do not match.'


    return


def run_composition_tests():

    alpha = np.random.rand() * 2.0 * np.pi
    # Random orthogonal rotation
    ortho_rot_1 = OrthoRotation2D(alpha)

    glyph = Glyph2D()


    # Compose rotation with itself.
    test_composition(ortho_rot_1, ortho_rot_1, glyph)

    #############################################################################
    # Random reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl_1 = OrthoReflection2D(v)

    #############################################################################
    # Rotation then random orthogonal reflection
    test_composition(ortho_rot_1, ortho_refl_1, glyph)

    # Orthogonal reflection then rotation
    test_composition(ortho_refl_1, ortho_rot_1, glyph)

    # Reflection with itself
    test_composition(ortho_refl_1, ortho_refl_1, glyph)

    #############################################################################
    # Second random reflection.

    w = np.random.rand(2) - [0.5, 0.5]
    ortho_refl_2 = OrthoReflection2D(w)

    # Compose two reflections
    test_composition(ortho_refl_1, ortho_refl_2, glyph)

    return

def test_composition(
        ortho_T1: OrthoTransform2D,
        ortho_T2: OrthoTransform2D,
        glyph: Glyph2D
):

    ortho_composition = compose_ortho_2d(ortho_T1, ortho_T2)

    glyph_A = glyph.apply_transformation(ortho_T1)
    glyph_A = glyph_A.apply_transformation(ortho_T2)

    glyph_B = glyph.apply_transformation(ortho_composition)

    assert np.allclose(glyph_A.points, glyph_B.points), 'Composition result incorrect.'

    return

def test_inverses():

    alpha = np.random.rand() * 2.0 * np.pi

    # Random orthogonal rotation
    ortho_rot = OrthoRotation2D(alpha)

    # Random reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl = OrthoReflection2D(v)

    inv_ortho_rot_1 = ortho_rot.inverse()

    mat_prod = ortho_rot.get_matrix() @ inv_ortho_rot_1.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'

    inv_ortho_refl_1 = ortho_refl.inverse()

    mat_prod = ortho_refl.get_matrix() @ inv_ortho_refl_1.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'

    return


if __name__ == '__main__':
    sys.exit(main())