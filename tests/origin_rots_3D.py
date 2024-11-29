import numpy as np

from twor.utils.general import random_rotation_3D, cross_product, vecs_parallel
from twor.geom.transform import OriginRotation3D

# Compose two pure rotations:

ax, theta = random_rotation_3D()
rot_A = OriginRotation3D(ax, theta)

ax, theta = random_rotation_3D()
rot_B = OriginRotation3D(ax, theta)

rot_C = rot_A.followed_by(rot_B)

# Check matrices match.

M_A = rot_A.get_matrix()
M_B = rot_B.get_matrix()

M_C = rot_C.get_matrix()

assert np.allclose(M_C, M_B @ M_A)

# Check alternative matrices match.

M_A2 = rot_A.get_matrix_B()
M_B2 = rot_B.get_matrix_B()

assert np.allclose(M_A, M_A2)
assert np.allclose(M_B, M_B2)

# Compose via quaternions.

q_A = rot_A.to_quaternion()
q_B = rot_B.to_quaternion()

q_C = rot_C.to_quaternion()

q_B_q_A = q_B * q_A

if q_B_q_A.real < 0:
    print('Flip quaternion so that real part is positive (-pi <= theta <= pi).')
    q_B_q_A *= -1.0
assert np.isclose(q_B_q_A, q_C)

# Calculating the axis of a composition manually.

u = rot_A.axis
theta = rot_A.angle

v = rot_B.axis
phi = rot_B.angle

# C = B A
w = rot_C.axis


c_theta_half = float(np.cos(theta / 2.0))
c_phi_half = float(np.cos(phi / 2.0))

s_theta_half = float(np.sin(theta / 2.0))
s_phi_half = float(np.sin(phi / 2.0))

new_axis = (
        c_theta_half * s_phi_half * v +
        c_phi_half * s_theta_half * u +
        s_theta_half * s_phi_half * cross_product(v, u)
)

success = vecs_parallel(w, new_axis)
assert success


print('Done.')