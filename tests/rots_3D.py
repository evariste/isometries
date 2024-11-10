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

# Two rotations about different points with parallel axes.

ax, theta = random_rotation_3D()
P = np.random.rand(3) * 10
P = ensure_vec(P)

rot_C = Rotation3D(P, ax, theta)

alpha = np.random.rand() * 2.0 * np.pi
Q = np.random.rand(3) * 10
Q = ensure_vec(Q)


rot_D = Rotation3D(Q, ax, alpha)

transf_E = rot_C.followed_by(rot_D)

M_C = rot_C.homogeneous_matrix()
M_D = rot_D.homogeneous_matrix()

M_E = transf_E.homogeneous_matrix()

success = np.allclose(M_D @ M_C, M_E)
assert success


# Two rotations about different points with arbitrary axes.

ax, theta = random_rotation_3D()
P = np.random.rand(3) * 10
P = ensure_vec(P)

rot_C = Rotation3D(P, ax, theta)

ax2, alpha = random_rotation_3D()
Q = np.random.rand(3) * 10
Q = ensure_vec(Q)


rot_D = Rotation3D(Q, ax2, alpha)

transf_E = rot_C.followed_by(rot_D)

M_C = rot_C.homogeneous_matrix()
M_D = rot_D.homogeneous_matrix()

M_E = transf_E.homogeneous_matrix()

success = np.allclose(M_D @ M_C, M_E)
assert success





print('x')
