import numpy as np
from twor.utils.general import random_rotation_3D, ensure_vec
from twor.geom.transform import Rotation3D, TransRotation3D

ax, theta = random_rotation_3D()

P = np.random.rand(3) * 10
P = ensure_vec(P)

rot_A = Rotation3D(P, ax, theta)

M_A = rot_A.homogeneous_matrix()


transf_B = rot_A.to_transrot()

assert isinstance(transf_B, TransRotation3D)

print(rot_A)


print(transf_B)