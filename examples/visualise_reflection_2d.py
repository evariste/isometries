import sys
from matplotlib import pyplot as plt
from isom.geom.objects import Glyph2D
from isom.geom.transform_2d import random_reflection_2d
from isom.utils.plotting import set_axis_glyph_bounds

def main():

    visualise_reflection()

    return 0


def visualise_reflection():
    refl = random_reflection_2d()
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
