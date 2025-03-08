import sys

from isom.geom.transform_2d import (
    random_reflection_2d, random_rotation_2d, random_ortho_reflection_2d, random_ortho_rotation_2d, Transform2D,
    flip_two_step_form_2D, compose_2d, transf_2d_from_two_step, random_glide_reflection_2d,
    random_transformation_2d, random_ortho_transformation_2d
)
from isom.geom.objects import Glyph2D

def main():

    run_tests_two_step_form_equivalence()

    run_tests_back_conversion()

    run_tests_flip_two_step()

    test_random_transfs(ortho=True)

    test_random_transfs()

    return 0


def test_back_conversion(transf: Transform2D):
    # M is orthogonal, t is a translation.
    M, t = transf.two_step_form()

    transf_B = transf_2d_from_two_step(M, t)

    assert transf.matrix_equals(transf_B), 'Recovered transform is not the same.'

    return


def test_two_step_form_equivalence(transf: Transform2D):
    # M is orthogonal, t is a translation.
    M, t = transf.two_step_form()

    glyph = Glyph2D()

    # Direct transformation.
    glyph_A = glyph.apply_transformation(transf)

    # Transform in two steps.
    glyph_B = glyph.apply_transformation(M).apply_transformation(t)

    assert glyph_A.is_close_to(glyph_B), 'Two step form does not match original transform.'

    return

def test_flip_two_step(transf: Transform2D):

    # M is orthogonal, t is a translation.
    M, t = transf.two_step_form()

    s, N = flip_two_step_form_2D([M, t])

    transf_B = compose_2d(s, N)

    assert transf.matrix_equals(transf_B), 'Flipped two-step form does not match original.'

    return


def run_tests_two_step_form_equivalence():

    print('Running tests for two-step equivalence.')

    transfs = generate_random_transforms()

    for transf in transfs:
        print(transf.__class__.__name__)
        test_two_step_form_equivalence(transf)
    print('-' * 80)
    return


def run_tests_flip_two_step():

    print('Running tests for flipping two-step form.')

    transfs = generate_random_transforms()

    for transf in transfs:
        print(transf.__class__.__name__)
        test_flip_two_step(transf)
    print('-' * 80)
    return


def run_tests_back_conversion():
    print('Running tests for back conversion.')

    transfs = generate_random_transforms()

    for transf in transfs:
        print(transf.__class__.__name__)
        test_back_conversion(transf)

    print('-' * 80)

    return

def generate_random_transforms():

    o_refl = random_ortho_reflection_2d()

    o_rot = random_ortho_rotation_2d()

    refl = random_reflection_2d()

    rot = random_rotation_2d()

    grefl = random_glide_reflection_2d()

    return [o_refl, o_rot, refl, rot, grefl]


def test_random_transfs(reps=10, ortho=False):
    print(f'Running tests on {reps} random transformations.')

    rand_func = random_transformation_2d
    if ortho:
        print('Restrict to orthogonal transformations.')
        rand_func = random_ortho_transformation_2d

    for _ in range(reps):
        transf = rand_func()

        print(transf.__class__.__name__)

        test_back_conversion(transf)

        test_flip_two_step(transf)

        test_two_step_form_equivalence(transf)

    print('-' * 80)
    return



if __name__ == '__main__':
    sys.exit(main())