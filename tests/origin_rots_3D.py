import sys
import numpy as np

from isom.utils.general import random_rotation_3D, cross_product, vecs_parallel
from isom.geom.transform_3d import OrthoRotation3D, random_ortho_rotation_3d, compose_ortho_3d

def main():
    # Compose two orthogonal rotations:
    rot_A = random_ortho_rotation_3d()

    rot_B = random_ortho_rotation_3d()

    rot_C = compose_ortho_3d(rot_A, rot_B)

    # Check matrices match.
    M_A = rot_A.get_matrix()
    M_B = rot_B.get_matrix()

    M_C = rot_C.get_matrix()

    assert np.allclose(M_C, M_B @ M_A), 'Matrices do not match.'

    # Check matching with alternative method of getting matrices.
    M_A_new = rot_A.get_matrix_B()
    M_B_new = rot_B.get_matrix_B()

    assert np.allclose(M_A, M_A_new), 'Matrices do not match.'
    assert np.allclose(M_B, M_B_new), 'Matrices do not match.'

if __name__ == '__main__':
    sys.exit(main())