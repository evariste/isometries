"""
Example of how two 2D rotations can compose to make a translation.
"""
import sys
import numpy as np
from matplotlib import pyplot as plt
from isom.geom.objects import Glyph2D, get_glyph_bounds
from isom.geom.transform_2d import Rotation2D, compose_2d


def main():

    A = [5, 0]
    B = [5, 10]

    theta = -1.0 * np.pi / 3.0
    phi = -1.0 * theta

    rot_A = Rotation2D(A, theta)
    rot_B = Rotation2D(B, phi)

    rot_C = compose_2d(rot_A, rot_B)

    glyph_start = Glyph2D()

    glyph_A = glyph_start.apply_transformation(rot_A)
    glyph_BA = glyph_A.apply_transformation(rot_B)

    glyph_C = glyph_start.apply_transformation(rot_C)

    fig3, ax = plt.subplots()

    patch_start = glyph_start.get_patch(facecolor='g')

    patch_A = glyph_A.get_patch(facecolor='gray')
    patch_BA = glyph_BA.get_patch(facecolor='r')

    patch_C = glyph_C.get_patch(facecolor='orange')

    glyphs = [glyph_start, glyph_A, glyph_BA, glyph_C]

    centres = [rot_A.centre, rot_B.centre]

    patches = [patch_start, patch_A, patch_BA, patch_C]

    x0, y0, x1, y1 = get_glyph_bounds(glyphs)
    pt_data = np.asarray([[x0, x1], [y0, y1]])
    pt_data = np.hstack(centres + [pt_data])
    min_vals = np.squeeze(np.min(pt_data, axis=1))
    max_vals = np.squeeze(np.max(pt_data, axis=1))

    x0 = min_vals[0]
    y0 = min_vals[1]
    x1 = max_vals[0]
    y1 = max_vals[1]

    for p in patches:
        ax.add_patch(p)

    patch_C.set_edgecolor('k')
    patch_C.set_alpha(0.3)
    patch_C.set_linestyle('dashed')
    patch_C.set_linewidth(2)

    for c, col in zip(centres, ['gray', 'r', 'orange']):
        plt.plot(c[0], c[1], marker='o', color='k')
        plt.plot(c[0], c[1], marker='x', color=col)

    xy0 = min(x0, y0)
    xy1 = max(x1, y1)
    xyrange = xy1 - xy0
    xy0 -= 0.1 * xyrange
    xy1 += 0.1 * xyrange
    ax.set_xlim([xy0, xy1])
    ax.set_ylim([xy0, xy1])

    ax.set_aspect('equal', 'box')

    def lines_to_centre(k, glyph_0, rotation, glyph_1, colour):
        p_0 = glyph_0.points[:, k]
        p_cent = np.squeeze(rotation.centre)
        p_1 = glyph_1.points[:, k]
        xy = np.vstack([p_0, p_cent, p_1])
        plt.plot(xy[:,0], xy[:,1], ':', color=colour)

    lines_to_centre(1, glyph_start, rot_A, glyph_A, 'k')
    lines_to_centre(5, glyph_A, rot_B, glyph_BA, 'g')

    # lines_to_centre(4, glyph_start, rot_C, glyph_C, 'orange')
    def line_glyph_to_glyph(k, glyph_0, glyph_1, colour):
        p_0 = glyph_0.points[:, k]
        p_1 = glyph_1.points[:, k]
        xy = np.vstack([p_0, p_1])
        plt.plot(xy[:, 0], xy[:, 1], ':', color=colour)

    for j in [1, 5]:
        line_glyph_to_glyph(j, glyph_start, glyph_C, 'gray')

    plt.savefig('pics/two_rotations_2D_trans.png')
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())