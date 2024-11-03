import numpy as np
from twor.utils.general import random_rotation_3D, ensure_vec
from twor.geom.transform import Rotation3D, TransOriginRotation3D

def gen_random_rot3D():

    ax, theta = random_rotation_3D()

    P = np.random.rand(3) * 10
    P = ensure_vec(P)

    return Rotation3D(P, ax, theta)

rot_A = gen_random_rot3D()

rot_B = gen_random_rot3D()

A_then_B = rot_A.followed_by(rot_B)



M_A = rot_A.homogeneous_matrix()
M_B = rot_B.homogeneous_matrix()

M_C = A_then_B.homogeneous_matrix()

assert np.allclose(M_B @ M_A, M_C)

print('Rotation A')
print(rot_A)

print('Rotation B')
print(rot_B)

print('Transformation from B o A')
print(A_then_B)


