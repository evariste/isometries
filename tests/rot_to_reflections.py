"""

Given a random rotation matrix, we can define two reflections that
will generate it.

"""
import numpy as np
from twor.utils.general import random_rotation_matrix_3D, ensure_vec
from twor.geom.objects import Plane3D
from twor.geom.transform_3d import Reflection3D

R = random_rotation_matrix_3D()


origin = ensure_vec([0, 0, 0])

# The starting coordinate frame.
i = ensure_vec([1, 0, 0])
j = ensure_vec([0, 1, 0])
k = ensure_vec([0, 0, 1])

# The rotated frame is represented by the columns of the rotation matrix.
u = ensure_vec(R[:, 0])
v = ensure_vec(R[:, 1])
w = ensure_vec(R[:, 2])

# The reflection i -> u = R(i)
K = Plane3D(u - i, origin)
ref_K = Reflection3D(K)

assert np.allclose(ref_K.apply(i), u)

# Where does the above reflection take j?
Kj = ref_K.apply(j)

# Find a reflection that:
# - takes ref_K(j) to v = R(j)
# - fixes u = R(i)
L = Plane3D.from_points(origin, u, v + Kj)
ref_L = Reflection3D(L)

assert np.allclose(ref_L.apply(u), u)

assert np.allclose(ref_L.apply(Kj), v)



MK = ref_K.get_matrix()
ML = ref_L.get_matrix()

MK = MK[:3, :3]
ML = ML[:3, :3]

ML_MK = ML @ MK
assert np.allclose(R, ML_MK)

# Ensure the resulting matrix takes k -> w = R(w)
assert np.allclose(ML_MK @ k, w)