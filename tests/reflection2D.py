import numpy as np
from matplotlib import pyplot as plt
from twor.geom.objects import Line2D, Glyph2D
from twor.geom.transform import Reflection2D
from twor.utils.general import apply_hom_matrix_to_points
from twor.utils.plotting import set_axis_glyph_bounds

# Random reflection.
P = np.random.rand(2) * 10
v = np.random.rand(2)
line_1 = Line2D(P, v)
refl_1 = Reflection2D(line_1)

# Apply to a glyph.
glyph = Glyph2D()
glyph_refl =  glyph.apply_transformation(refl_1)

# Apply via the matrix.
M = refl_1.get_matrix()
pts_v2 = apply_hom_matrix_to_points(M,  glyph.points)

assert np.allclose(pts_v2, glyph_refl.points), 'Homogeneous matrix gives different answer.'


# Get the two step form
[N, t] = refl_1.two_step_form()

pts_v3 = t.apply(N.apply(glyph.points))
assert np.allclose(pts_v2, glyph_refl.points), 'Two-step form gives different answer.'

# Another reflection
Q = np.random.rand(2) * 10
u = np.random.rand(2)
line_2 = Line2D(Q, u)
refl_2 = Reflection2D(line_2)

# Composition
refl_21 = refl_1.followed_by(refl_2)
glyph_comp = glyph.apply_transformation(refl_21)

S = refl_2.get_matrix()

pts_v4 = apply_hom_matrix_to_points(S @ M, glyph.points)
assert np.allclose(pts_v4, glyph_comp.points), 'Composition result different from one obtained with hom. matrices.'


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