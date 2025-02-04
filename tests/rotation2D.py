
import numpy as np
from twor.geom.transform import Rotation2D, OrthoRotation2D
from twor.geom.objects import Glyph2D
from twor.utils.general import apply_hom_matrix_to_points, apply_transform_sequence_to_glyph

alpha = np.random.rand() * 2.0 * np.pi

C = np.random.rand(2) * 10

rot = Rotation2D(C, alpha)


glyph = Glyph2D()

glyph_rot = glyph.apply_transformation(rot)

M = rot.get_matrix()
pts_v2 = apply_hom_matrix_to_points(M, glyph.points)

assert np.allclose(pts_v2, glyph_rot.points), 'Result from homogeneous matrix and direct transformation do not match.'


N = OrthoRotation2D(alpha)

reflections = N.get_reflections()

glyph_rot_N = glyph.apply_transformation(N)

glyph_2 = apply_transform_sequence_to_glyph(reflections, glyph)


assert np.allclose(glyph_rot_N.points, glyph_2.points), 'Results from reflections and direct orthogonal transformation do not match.'


print('Done.')