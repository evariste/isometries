import sys
import numpy as np
import os
from isom.geom.objects import Glyph2D
from isom.geom.transform_2d import random_reflection_2d, random_ortho_reflection_2d, compose_2d
from isom.utils.general import apply_hom_matrix_to_points

def main():
    print('*' * 80)
    print(f'Running test {os.path.basename(__file__)}')

    test_apply_reflection()

    test_two_step_form()

    test_composition()

    test_inverses()

    return 0



def test_apply_reflection():
    refl = random_reflection_2d()

    # Apply to a glyph.
    glyph = Glyph2D()
    glyph_refl =  glyph.apply_transformation(refl)

    # Apply via the matrix.
    M = refl.get_matrix()
    pts_v2 = apply_hom_matrix_to_points(M,  glyph.points)

    assert np.allclose(pts_v2, glyph_refl.points), 'Homogeneous matrix gives different answer.'

    return



def test_two_step_form():
    refl = random_reflection_2d()
    glyph = Glyph2D()

    # Get the two step form
    [N, t] = refl.two_step_form()

    glyph_refl = glyph.apply_transformation(refl)

    glyph_B = glyph.apply_transformation(N).apply_transformation(t)

    assert glyph_B.is_close_to(glyph_refl), 'Two-step form gives different answer.'

    return

def test_composition():
    refl_1 = random_reflection_2d()
    refl_2 = random_reflection_2d()
    glyph = Glyph2D()

    # Composition
    transf_comp = compose_2d(refl_1, refl_2)
    glyph_comp = glyph.apply_transformation(transf_comp)

    M = refl_1.get_matrix()
    N = refl_2.get_matrix()

    NM = N @ M

    pts_via_matrices = apply_hom_matrix_to_points(NM, glyph.points)

    assert np.allclose(pts_via_matrices, glyph_comp.points), 'Composition result different from one obtained with hom. matrices.'

    return

def test_inverses():

    # Random orthogonal reflection.
    ortho_refl = random_ortho_reflection_2d()

    inv_ortho_refl = ortho_refl.inverse()

    mat_prod = ortho_refl.get_matrix() @ inv_ortho_refl.get_matrix()

    assert np.allclose(mat_prod, np.eye(3)), 'Inverse incorrect.'

    return


if __name__ == '__main__':
    sys.exit(main())
