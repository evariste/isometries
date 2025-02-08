import sys
import numpy as np
from matplotlib import pyplot as plt


from twor.geom.objects import Line2D, Glyph2D
from twor.geom.transform_2d import Reflection2D
from twor.utils.general import apply_hom_matrix_to_points
from twor.utils.plotting import set_axis_glyph_bounds

def main():

    test_apply_reflection()

    test_two_step_form()

    test_composition()

    visualise_reflection()

    return 0



def test_apply_reflection():
    refl = make_random_reflection()

    # Apply to a glyph.
    glyph = Glyph2D()
    glyph_refl =  glyph.apply_transformation(refl)

    # Apply via the matrix.
    M = refl.get_matrix()
    pts_v2 = apply_hom_matrix_to_points(M,  glyph.points)

    assert np.allclose(pts_v2, glyph_refl.points), 'Homogeneous matrix gives different answer.'

    return

def make_random_reflection():
    # Random reflection.
    P = np.random.rand(2) * 10
    v = np.random.rand(2)
    line_1 = Line2D(P, v)
    refl = Reflection2D(line_1)

    return refl


def test_two_step_form():
    refl = make_random_reflection()
    glyph = Glyph2D()

    # Get the two step form
    [N, t] = refl.two_step_form()

    glyph_refl = glyph.apply_transformation(refl)

    glyph_B = glyph.apply_transformation(N).apply_transformation(t)

    assert np.allclose(glyph_B.points, glyph_refl.points), 'Two-step form gives different answer.'

    return

def test_composition():
    refl_1 = make_random_reflection()
    refl_2 = make_random_reflection()
    glyph = Glyph2D()

    # Composition
    transf_comp = refl_1.followed_by(refl_2)
    glyph_comp = glyph.apply_transformation(transf_comp)

    M = refl_1.get_matrix()
    N = refl_2.get_matrix()

    NM = N @ M

    pts_via_matrices = apply_hom_matrix_to_points(NM, glyph.points)

    assert np.allclose(pts_via_matrices, glyph_comp.points), 'Composition result different from one obtained with hom. matrices.'

    return

def visualise_reflection():
    refl = make_random_reflection()
    line = refl.line
    glyph = Glyph2D()


    glyph_refl = glyph.apply_transformation(refl)

    fig, ax = plt.subplots()

    patch1 = glyph.get_patch(facecolor='g')
    patch2 = glyph_refl.get_patch(facecolor='r', alpha=0.5)

    for p in [patch1, patch2]:
        ax.add_patch(p)

    xy0, xy1 = set_axis_glyph_bounds(ax, [glyph, glyph_refl])

    f_x0 = line.f_x(xy0)
    f_x1 = line.f_x(xy1)

    ax.plot([xy0, xy1], [f_x0, f_x1], 'k:', linewidth=1)

    ax.set_aspect('equal', 'box')

    plt.show()

    return


if __name__ == '__main__':
    sys.exit(main())
