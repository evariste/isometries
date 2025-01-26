import numpy as np
from matplotlib import pyplot as plt
from twor.geom.objects import Line2D, Glyph2D
from twor.geom.transform import Reflection2D
from twor.utils.general import apply_hom_matrix_to_points
from twor.utils.plotting import set_axis_glyph_bounds

P = np.random.rand(2) * 10
v = np.random.rand(2)

Q = np.random.rand(2) * 10
u = np.random.rand(2)

line_1 = Line2D(P, v)
line_2 = Line2D(Q, u)

refl_1 = Reflection2D(line_1)
refl_2 = Reflection2D(line_2)

glyph = Glyph2D()
glyph_refl =  glyph.apply_transformation(refl_1)

M = refl_1.get_matrix()

pts_v2 = apply_hom_matrix_to_points(M,  glyph.points)

assert np.allclose(pts_v2, glyph_refl.points), 'Homogeneous matrix gives different answer.'


fig, ax = plt.subplots()

patch1 = glyph.get_patch(facecolor='g')
patch2 = glyph_refl.get_patch(facecolor='r', alpha=0.5)

for p in [patch1, patch2]:
    ax.add_patch(p)

xy0, xy1 = set_axis_glyph_bounds(ax, [glyph, glyph_refl])

f_x0 = line_1.f_x(xy0)
f_x1 = line_1.f_x(xy1)

ax.plot([xy0, xy1], [f_x0, f_x1], 'k:', linewidth=1)

ax.set_aspect('equal', 'box')

plt.show()