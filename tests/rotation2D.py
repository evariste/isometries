
import numpy as np
from twor.geom.transform import Rotation2D
from twor.geom.objects import Glyph2D
from twor.utils.general import apply_hom_matrix_to_points

alpha = np.random.rand() * 2.0 * np.pi

C = np.random.rand(2) * 10

rot = Rotation2D(C, alpha)


glyph = Glyph2D()

glyph_rot = glyph.apply_transformation(rot)

M = rot.get_matrix()
pts_v2 = apply_hom_matrix_to_points(M, glyph.points)

assert np.allclose(pts_v2, glyph_rot.points)


print('Done.')