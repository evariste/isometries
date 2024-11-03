import numpy as np
from twor.utils.general import random_rotation_3D, ensure_vec
from twor.geom.transform import Rotation3D, OriginRotation3D, Translation3D

ax, theta = random_rotation_3D()

P = np.random.rand(3) * 10
P = ensure_vec(P)

rot_A = Rotation3D(P, ax, theta)

M_A = rot_A.homogeneous_matrix()

print(rot_A)

rot_B = OriginRotation3D(ax, theta)

Q = rot_B.apply(P)

trans = Translation3D(P - Q)

M_trans = trans.homogeneous_matrix()
M_B = rot_B.homogeneous_matrix()

M_C = M_trans @ M_B

assert np.allclose(M_A, M_C)