import numpy as np
from isom.utils.general import random_rotation_3D, ensure_vec
from isom.geom.transform_3d import Rotation3D, OrthoRotation3D, Translation3D

ax, theta = random_rotation_3D()
P = np.random.rand(3) * 10
P = ensure_vec(P)

rot_A = Rotation3D(P, ax, theta)

M_A = rot_A.get_matrix()

print(rot_A)

rot_B = OrthoRotation3D(ax, theta)

Q = rot_B.apply(P)

trans = Translation3D(P - Q)

M_trans = trans.get_matrix()
M_B = rot_B.get_matrix()

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

M_C = rot_C.get_matrix()
M_D = rot_D.get_matrix()

M_E = transf_E.get_matrix()

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

M_C = rot_C.get_matrix()
M_D = rot_D.get_matrix()

M_E = transf_E.get_matrix()

success = np.allclose(M_D @ M_C, M_E)
assert success





print('x')
