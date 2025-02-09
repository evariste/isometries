
import sys
import numpy as np
from twor.geom.transform_2d import (
    OrthoReflection2D, compose_ortho_2d, OrthoTransform2D, ortho2D_to_reflections,
    random_rotation_2d, random_ortho_rotation_2d, random_ortho_reflection_2d
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

    rot = random_rotation_2d()

    #############################################################################

    # Apply to glyph
    glyph = Glyph2D()
    glyph_rot = glyph.apply_transformation(rot)

    M = rot.get_matrix()
    pts_v2 = apply_hom_matrix_to_points(M, glyph.points)

    assert np.allclose(pts_v2, glyph_rot.points), 'Result from homogeneous matrix and direct transformation do not match.'

    return

def test_reflection_decomp():

    ortho_rot = random_ortho_rotation_2d()

    glyph = Glyph2D()

    # Decompose to reflections.
    reflections = ortho_rot.get_reflections()

    glyph_rotate = glyph.apply_transformation(ortho_rot)

    glyph_2refls = apply_transform_sequence_to_glyph(reflections, glyph)

    assert glyph_rotate.is_close_to(glyph_2refls), 'Results from reflections and direct orthogonal transformation do not match.'

    # Decompose using general function
    reflections_B = ortho2D_to_reflections(ortho_rot)

    glyph_B = apply_transform_sequence_to_glyph(reflections_B, glyph)

    assert glyph_rotate.is_close_to(glyph_B), 'Results from reflections and direct orthogonal transformation do not match.'


    # Random reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl = OrthoReflection2D(v)

    reflections_C = ortho2D_to_reflections(ortho_refl)

    assert len(reflections_C) == 1, 'Unexpected number of reflections.'
    refl_C = reflections_C[0]

    glyph_C = glyph.apply_transformation(refl_C)
    glyph_D = glyph.apply_transformation(ortho_refl)
    assert glyph_C.is_close_to(glyph_D), 'Results from reflections and direct orthogonal transformation do not match.'


    return


def run_composition_tests():

    ortho_rot_1 = random_ortho_rotation_2d()

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

    assert glyph_A.is_close_to(glyph_B), 'Composition result incorrect.'

    return

def test_inverses():

    # Random orthogonal rotation.
    ortho_rot = random_ortho_rotation_2d()

    # Random orthogonal reflection.
    ortho_refl = random_ortho_reflection_2d()

    inv_ortho_rot = ortho_rot.inverse()

    mat_prod = ortho_rot.get_matrix() @ inv_ortho_rot.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'

    inv_ortho_refl = ortho_refl.inverse()

    mat_prod = ortho_refl.get_matrix() @ inv_ortho_refl.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'

    return


if __name__ == '__main__':
    sys.exit(main())