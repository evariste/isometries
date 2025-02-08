import sys

import numpy as np

from twor.geom.transform_2d import (
    random_reflection2d, random_rotation2d, random_ortho_reflection2d, random_ortho_rotation2d, Transform2D,
)
from twor.geom.objects import Glyph2D

def main():

    run_tests_two_step_form()

    return 0


def run_tests_two_step_form():

    o_refl = random_ortho_reflection2d()
    test_two_step_form(o_refl)

    o_rot = random_ortho_rotation2d()
    test_two_step_form(o_rot)

    refl = random_reflection2d()
    test_two_step_form(refl)

    rot = random_rotation2d()
    test_two_step_form(rot)

    return

def test_two_step_form(transf: Transform2D):

    M, t = transf.two_step_form()

    glyph = Glyph2D()

    # Direct transformation.
    glyph_A = glyph.apply_transformation(transf)

    # Transform in two steps.
    glyph_B = glyph.apply_transformation(M).apply_transformation(t)

    pts_A = glyph_A.points
    pts_B = glyph_B.points

    assert np.allclose(pts_A, pts_B), 'Two step form does not match original transform.'

    return


if __name__ == '__main__':
    sys.exit(main())