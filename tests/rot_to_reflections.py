import numpy as np
from twor.utils.general import random_rotation_matrix_3D, ensure_vec
from twor.geom.objects import Plane3D
from twor.geom.transform import Reflection3D



R = random_rotation_matrix_3D()

origin = ensure_vec([0, 0, 0])

i = ensure_vec([1, 0, 0])
j = ensure_vec([0, 1, 0])
k = ensure_vec([0, 0, 1])

u = ensure_vec(R[:, 0])
v = ensure_vec(R[:, 1])
w = ensure_vec(R[:, 2])

K = Plane3D(u - i, origin)
ref_K = Reflection3D(K)

assert np.allclose(ref_K.apply(i), u)

Kj = ref_K.apply(j)

L = Plane3D.from_points(origin, u, v + Kj)
ref_L = Reflection3D(L)

assert np.allclose(ref_L.apply(u), u)

assert np.allclose(ref_L.apply(Kj), v)

MK = ref_K.get_matrix()
ML = ref_L.get_matrix()

MK = MK[:3, :3]
ML = ML[:3, :3]

assert np.allclose(R, ML @ MK)