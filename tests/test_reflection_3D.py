import sys

import numpy as np

from isom.geom.transform import is_identity
from isom.utils.general import apply_hom_matrix_to_points, rotate_vector_3d
from isom.geom.objects import Glyph3D
from isom.geom.transform_3d import (
    random_reflection_3d, random_ortho_reflection_3d,
    compose_3d,
    OrthoReflection3D,
    OrthoRotation3D, Rotation3D,
    OrthoImproperRotation3D, ImproperRotation3D,
)


def main():

    test_matrices()

    test_self_inversion()

    test_composition(ortho=True)

    test_composition()

    test_composition_3()

    print('Done.')
    
    return 0

def test_composition(ortho=False):
    print('Testing composition.')

    rand_func = random_reflection_3d
    target_class = Rotation3D
    if ortho:
        print('  -- Orthogonal transformations.')
        rand_func = random_ortho_reflection_3d
        target_class = OrthoRotation3D

    refl_0 = rand_func()
    refl_1 = rand_func()

    transf = compose_3d(refl_0, refl_1)
    assert isinstance(transf, target_class) or is_identity(transf), 'Unexpected type of result.'

    R0 = refl_0.get_matrix()
    R1 = refl_1.get_matrix()
    T = transf.get_matrix()
    assert np.allclose(R1 @ R0, T), 'Composition mismatch.'

    return


def test_composition_3():
    print('Testing composition of three reflections.')

    # Three reflections in planes through the origin.

    refl_0 = random_ortho_reflection_3d()
    refl_1 = random_ortho_reflection_3d()
    refl_2 = random_ortho_reflection_3d()

    transf = OrthoImproperRotation3D.from_reflections(refl_0, refl_1, refl_2)

    assert isinstance(transf, OrthoImproperRotation3D), 'Unexpected type of transformation.'

    R0 = refl_0.get_matrix()
    R1 = refl_1.get_matrix()
    R2 = refl_2.get_matrix()

    T = transf.get_matrix()

    assert np.allclose(R2 @ R1 @ R0, T), 'Composition mismatch.'



    # Three reflections with planes that intersect in a single line.

    refl_0 = random_ortho_reflection_3d()
    refl_1 = random_ortho_reflection_3d()

    axis = refl_0.plane.intersection(refl_1.plane)

    alpha = (np.random.rand() * 2.0 - 1) * np.pi

    normal = rotate_vector_3d(refl_1.normal, axis.direction, alpha)

    refl_2 = OrthoReflection3D(normal)

    transf = OrthoImproperRotation3D.from_reflections(refl_0, refl_1, refl_2)

    assert isinstance(transf, OrthoReflection3D), 'Unexpected type of transformation.'

    R0 = refl_0.get_matrix()
    R1 = refl_1.get_matrix()
    R2 = refl_2.get_matrix()

    T = transf.get_matrix()

    assert np.allclose(R2 @ R1 @ R0, T), 'Composition mismatch.'


    return



def test_self_inversion():

    print('Test self-inversion')

    ortho_refl = random_ortho_reflection_3d()

    glyph = Glyph3D()

    glyph_B = glyph.apply_transformation(ortho_refl).apply_transformation(ortho_refl)
    assert glyph.is_close_to(glyph_B), 'Failed self inversion check'

    ortho_refl_inv = ortho_refl.inverse()
    glyph_C = glyph.apply_transformation(ortho_refl).apply_transformation(ortho_refl_inv)
    assert glyph.is_close_to(glyph_C), 'Failed self inversion check'

    refl = random_reflection_3d()

    glyph_D = glyph.apply_transformation(refl).apply_transformation(refl)
    assert glyph.is_close_to(glyph_D), 'Failed self inversion check'

    return

def test_matrices():
    print('Test matrices.')

    ortho_refl = random_ortho_reflection_3d()

    glyph = Glyph3D()

    pts = glyph.points

    M_ortho_refl = ortho_refl.get_matrix()

    pts_B = glyph.apply_transformation(ortho_refl).points

    pts_C = apply_hom_matrix_to_points(M_ortho_refl, pts)

    assert np.allclose(pts_B, pts_C), 'Result from matrix multiplication differ'

    refl = random_reflection_3d()

    M_refl = refl.get_matrix()

    pts_D = glyph.apply_transformation(refl).points

    pts_E = apply_hom_matrix_to_points(M_refl, pts)

    assert np.allclose(pts_D, pts_E),  'Result from matrix multiplication differ'

    return


if __name__ == '__main__':
    sys.exit(main())