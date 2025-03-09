import sys
import os
import numpy as np
from isom.geom.transform import Identity, transforms_equal
from isom.geom.transform_2d import OrthoReflection2D, OrthoRotation2D, Reflection2D, Rotation2D, Translation2D
from isom.geom.objects import Line2D

# TODO: 3D transforms.

def main():
    print('*' * 80)
    print(f'Running test {os.path.basename(__file__)}')

    test_copy_2D()

    return 0

def test_copy_2D():

    I = Identity(2)
    I_B = I.copy()

    assert transforms_equal(I, I_B), 'Copy fail: identity'

    v_dir = np.random.rand(2) - [0.5, 0.5]
    alpha = np.random.rand() * 2.0 * np.pi

    pt = np.random.rand(2) * 10
    line = Line2D(pt, v_dir)


    T = OrthoReflection2D(v_dir)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B), 'Copy fail: OrthoReflection2D'

    T = OrthoRotation2D(alpha)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B), 'Copy fail: OrthoRotation2D'

    T = Reflection2D(line)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B), 'Copy fail: Reflection2D'

    T = Rotation2D(pt, alpha)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B), 'Copy fail: Rotation2D'

    T = Translation2D(pt)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B), 'Copy fail: Translation2D'

    return


if __name__ == '__main__':
    sys.exit(main())