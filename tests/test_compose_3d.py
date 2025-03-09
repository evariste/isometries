from __future__ import annotations
import os
import numpy as np

"""
Pairs of transforms.
Ordered composition.
Also with their inverses.
"""

import sys
import itertools
from isom.geom.transform_3d import random_transformation_3d, compose_3d, TRANSF_TYPES_3D, Transform3D
from isom.geom.transform import is_identity

def main():
    print('*' * 80)
    print(f'Running test {os.path.basename(__file__)}')

    test_inverses()

    test_compositions()

    return 0

def test_compositions():
    rand_transfs = get_random_transfs()

    for t1, t2 in itertools.product(TRANSF_TYPES_3D, TRANSF_TYPES_3D):
        f1 = rand_transfs[t1]
        f2 = rand_transfs[t2]

        test_composition(f1, f2)

        test_composition(f2, f1)

    return

def test_composition(f1: Transform3D, f2: Transform3D):

    g = compose_3d(f1, f2)

    M1 = f1.get_matrix()
    M2 = f2.get_matrix()

    Mg = g.get_matrix()

    assert np.allclose(M2 @ M1, Mg), 'Composition matrix does not match matrix product.'

    return



def test_inverses():

    rand_transfs = get_random_transfs()

    for t_type in TRANSF_TYPES_3D:
        f = rand_transfs[t_type]

        f_inv = f.inverse()
        g = compose_3d(f, f_inv)
        assert is_identity(g), 'Expect identity transform.'

        g = compose_3d(f_inv, f)
        assert is_identity(g), 'Expect identity transform.'




    return

def get_random_transfs():

    ret = {}
    for t_type in TRANSF_TYPES_3D:
        ret[t_type] = random_transformation_3d(t_type=t_type)

    return ret



if __name__ == '__main__':
    sys.exit(main())