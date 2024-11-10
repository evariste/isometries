import numpy as np
from twor.utils.general import random_rotation_3D, ensure_vec
from twor.geom.transform import Rotation3D, TransOriginRotation3D, TransRotation3D

ax, theta = random_rotation_3D()

P = np.random.rand(3) * 10
P = ensure_vec(P)

rot_A = Rotation3D(P, ax, theta)

M_A = rot_A.homogeneous_matrix()

transf_B = rot_A.to_trans_origin_rot()

assert isinstance(transf_B, TransOriginRotation3D)

M_B = transf_B.homogeneous_matrix()

assert np.allclose(M_A, M_B)

print(rot_A)

print(transf_B)


transf_C = transf_B.to_trans_rot()


assert transf_C.is_close(transf_C)


v, alpha = random_rotation_3D()
Q = np.random.rand(3) * 10
Q = ensure_vec(Q)

rot_D = Rotation3D(Q, v, alpha)



transf_E = rot_A.followed_by(rot_D)

assert isinstance(transf_E, TransRotation3D)

ax = transf_E.gen_rot.orig_rot.axis

print(transf_E)



