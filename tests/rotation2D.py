
import sys
import numpy as np
from twor.geom.transform_2d import Rotation2D, OrthoRotation2D, OrthoReflection2D, compose_ortho_2d
from twor.geom.objects import Glyph2D, Line2D
from twor.utils.general import apply_hom_matrix_to_points, apply_transform_sequence_to_glyph

def main():
    #############################################################################

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

    #############################################################################

    # Random orthogonal rotation
    ortho_rot_1 = OrthoRotation2D(alpha)

    #############################################################################

    # Decompose to reflections.
    reflections = ortho_rot_1.get_reflections()

    glyph_rot_N = glyph.apply_transformation(ortho_rot_1)

    glyph_2 = apply_transform_sequence_to_glyph(reflections, glyph)

    assert np.allclose(glyph_rot_N.points, glyph_2.points), 'Results from reflections and direct orthogonal transformation do not match.'

    #############################################################################

    # Compose rotation with itself.

    ortho_comp_1 = compose_ortho_2d(ortho_rot_1, ortho_rot_1)

    glyph_3 = glyph.apply_transformation(ortho_rot_1)
    glyph_3 = glyph_3.apply_transformation(ortho_rot_1)

    glyph_4 = glyph.apply_transformation(ortho_comp_1)

    assert np.allclose(glyph_3.points, glyph_4.points), 'Composition result incorrect.'

    #############################################################################
    # Random reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl_1 = OrthoReflection2D(v)

    #############################################################################
    # Rotation then random orthogonal reflection

    ortho_comp_2 = compose_ortho_2d(ortho_rot_1, ortho_refl_1)

    glyph_5 = glyph.apply_transformation(ortho_rot_1)
    glyph_5 = glyph_5.apply_transformation(ortho_refl_1)

    glyph_6 = glyph.apply_transformation(ortho_comp_2)

    assert np.allclose(glyph_5.points, glyph_6.points), 'Composition result incorrect.'

    #############################################################################
    # Orthogonal reflection then rotation

    ortho_comp_3 = compose_ortho_2d(ortho_refl_1, ortho_rot_1)

    glyph_7 = glyph.apply_transformation(ortho_refl_1)
    glyph_7 = glyph_7.apply_transformation(ortho_rot_1)

    glyph_8 = glyph.apply_transformation(ortho_comp_3)

    assert np.allclose(glyph_7.points, glyph_8.points), 'Composition result incorrect.'

    #############################################################################
    # Second random reflection.

    w = np.random.rand(2) - [0.5, 0.5]
    ortho_refl_2 = OrthoReflection2D(w)

    #############################################################################

    # Compose two reflections
    ortho_comp_4 = compose_ortho_2d(ortho_refl_1, ortho_refl_2)

    glyph_9 = glyph.apply_transformation(ortho_refl_1)
    glyph_9 = glyph_9.apply_transformation(ortho_refl_2)

    glyph_10 = glyph.apply_transformation(ortho_comp_4)

    assert np.allclose(glyph_9.points, glyph_10.points), 'Composition result incorrect.'


    #############################################################################

    # Reflection with itself
    ortho_comp_5 = compose_ortho_2d(ortho_refl_1, ortho_refl_1)

    glyph_11 = glyph.apply_transformation(ortho_refl_1)
    glyph_11 = glyph_11.apply_transformation(ortho_refl_1)

    glyph_12 = glyph.apply_transformation(ortho_comp_5)

    assert np.allclose(glyph_11.points, glyph_12.points), 'Composition result incorrect.'



    #############################################################################

    # Inverses

    inv_ortho_rot_1 = ortho_rot_1.inverse()

    mat_prod = ortho_rot_1.get_matrix() @ inv_ortho_rot_1.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'

    inv_ortho_refl_1 = ortho_refl_1.inverse()

    mat_prod = ortho_refl_1.get_matrix() @ inv_ortho_refl_1.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'


    #############################################################################



    print('Done.')

    return 0


if __name__ == '__main__':
    sys.exit(main())