import numpy as np
from twor.utils.general import random_rotation_3D
from twor.geom.transform import OriginRotation3D


ax, theta = random_rotation_3D()
rot_A = OriginRotation3D(ax, theta)

ax, theta = random_rotation_3D()
rot_B = OriginRotation3D(ax, theta)

rot_C = rot_A.followed_by(rot_B)

M_A = rot_A.homogeneous_matrix()
M_B = rot_B.homogeneous_matrix()

M_C = rot_C.homogeneous_matrix()

assert np.allclose(M_C, M_B @ M_A)


M_A2 = rot_A.homogeneous_matrix_B()
M_B2 = rot_B.homogeneous_matrix_B()

assert np.allclose(M_A, M_A2)
assert np.allclose(M_B, M_B2)