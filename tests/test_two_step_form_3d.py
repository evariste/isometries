from __future__ import annotations

import sys

from isom.geom.objects import Glyph3D
from isom.geom.transform_3d import (
random_ortho_reflection_3d, random_ortho_rotation_3d, random_ortho_improper_rotation_3d,
random_reflection_3d, random_rotation_3d, random_improper_rotation_3d,
Transform3D,
transf_3d_from_two_step,
flip_two_step_form_3D,
compose_3d,
)

def main():

    run_tests_two_step_form_equivalence()

    run_tests_back_conversion()

    run_tests_flip_two_step()

    return 0


def test_back_conversion(transf: Transform3D):
    # M is orthogonal, t is a translation.
    M, t = transf.two_step_form()

    transf_B = transf_3d_from_two_step(M, t)

    assert transf.matrix_equals(transf_B), 'Recovered transform is not the same.'

    return


def test_two_step_form_equivalence(transf: Transform3D):
    # M is orthogonal, t is a translation.
    M, t = transf.two_step_form()

    glyph = Glyph3D()

    # Direct transformation.
    glyph_A = glyph.apply_transformation(transf)

    # Transform in two steps.
    glyph_B = glyph.apply_transformation(M).apply_transformation(t)

    assert glyph_A.is_close_to(glyph_B), 'Two step form does not match original transform.'

    return

def test_flip_two_step(transf: Transform3D):

    # M is orthogonal, t is a translation.
    M, t = transf.two_step_form()

    s, N = flip_two_step_form_3D([M, t])

    transf_B = compose_3d(s, N)

    assert transf.matrix_equals(transf_B), 'Flipped two-step form does not match original.'

    return


def run_tests_two_step_form_equivalence():

    print('Running tests for two-step equivalence.')

    o_refl = random_ortho_reflection_3d()
    test_two_step_form_equivalence(o_refl)

    o_rot = random_ortho_rotation_3d()
    test_two_step_form_equivalence(o_rot)

    o_imp_rot = random_ortho_improper_rotation_3d()
    test_two_step_form_equivalence(o_imp_rot)

    refl = random_reflection_3d()
    test_two_step_form_equivalence(refl)

    rot = random_rotation_3d()
    test_two_step_form_equivalence(rot)

    imp_rot = random_improper_rotation_3d()
    test_two_step_form_equivalence(imp_rot)

    return


def run_tests_flip_two_step():

    print('Running tests for flipping two-step form.')

    o_refl = random_ortho_reflection_3d()
    test_flip_two_step(o_refl)

    o_rot = random_ortho_rotation_3d()
    test_flip_two_step(o_rot)

    refl = random_reflection_3d()
    test_flip_two_step(refl)

    rot = random_rotation_3d()
    test_flip_two_step(rot)

    return


def run_tests_back_conversion():

    print('Running tests for back conversion.')

    o_refl = random_ortho_reflection_3d()
    test_back_conversion(o_refl)

    o_rot = random_ortho_rotation_3d()
    test_back_conversion(o_rot)

    refl = random_reflection_3d()
    test_back_conversion(refl)

    rot = random_rotation_3d()
    test_back_conversion(rot)

    return


if __name__ == '__main__':
    sys.exit(main())