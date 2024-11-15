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



q_A = rot_A.to_quaternion()
q_B = rot_B.to_quaternion()

q_C = rot_C.to_quaternion()

v = q_C.imag
s = np.sqrt(np.sum(v * v))

c = q_C.real

q_theta = np.arctan2(s, c)

q_B_q_A = q_B * q_A

success = np.isclose(q_B_q_A, q_C) or np.isclose(-1.0 * q_B_q_A, q_C)

assert success

print('Done.')