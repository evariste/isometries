import sys

import numpy as np

from tests.rot_to_reflections import ref_L
from twor.utils.general import apply_hom_matrix_to_points
from twor.geom.objects import Glyph3D
from twor.geom.transform_3d import random_reflection_3d, random_ortho_reflection_3d


def main():

    test_matrices()

    test_self_inversion()

    return 0


def test_self_inversion():

    ortho_refl = random_ortho_reflection_3d()

    glyph = Glyph3D()

    glyph_B = glyph.apply_transformation(ortho_refl).apply_transformation(ortho_refl)
    assert glyph.is_close_to(glyph_B), 'Failed self inversion check'

    refl = random_reflection_3d()

    glyph_C = glyph.apply_transformation(refl).apply_transformation(refl)
    assert glyph.is_close_to(glyph_C), 'Failed self inversion check'

    return

def test_matrices():

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