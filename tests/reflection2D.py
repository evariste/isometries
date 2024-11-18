import numpy as np
from matplotlib import pyplot as plt
from twor.geom.objects import Line2D, Glyph2D
from twor.geom.transform import Reflection2D
from twor.utils.general import apply_hom_matrix_to_points
from twor.utils.plotting import set_axis_glyph_bounds

P = np.random.rand(2) * 10
v = np.random.rand(2)

l = Line2D(P, v)

refl = Reflection2D(l)

glyph = Glyph2D()
glyph_refl =  glyph.apply_transformation(refl)

M = refl.get_matrix()

pts_v2 = apply_hom_matrix_to_points(M,  glyph.points)

assert np.allclose(pts_v2, glyph_refl.points), 'Homogeneous matrix gives different answer.'


fig, ax = plt.subplots()

patch1 = glyph.get_patch(facecolor='g')
patch2 = glyph_refl.get_patch(facecolor='r', alpha=0.5)

for p in [patch1, patch2]:
    ax.add_patch(p)

xy0, xy1 = set_axis_glyph_bounds(ax, [glyph, glyph_refl])

f_x0 = l.f_x(xy0)
f_x1 = l.f_x(xy1)

ax.plot([xy0, xy1], [f_x0, f_x1], 'k:', linewidth=1)

ax.set_aspect('equal', 'box')

plt.show()