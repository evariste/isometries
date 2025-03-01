import sys
import numpy as np
from twor.geom.transform import Identity, transforms_equal
from twor.geom.transform_2d import OrthoReflection2D, OrthoRotation2D, Reflection2D, Rotation2D, Translation2D
from twor.geom.objects import Line2D

# TODO: 3D transforms.

def main():

    test_copy_2D()

    return 0

def test_copy_2D():

    I = Identity(2)
    I_B = I.copy()

    assert transforms_equal(I, I_B)

    v_dir = np.random.rand(2) - [0.5, 0.5]
    alpha = np.random.rand() * 2.0 * np.pi

    pt = np.random.rand(2) * 10
    line = Line2D(pt, v_dir)


    T = OrthoReflection2D(v_dir)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B)

    T = OrthoRotation2D(alpha)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B)

    T = Reflection2D(line)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B)

    T = Rotation2D(pt, alpha)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B)

    T = Translation2D(pt)
    T_B = T.copy()
    print('-' * 40)
    print(T)
    assert transforms_equal(T, T_B)

    return


if __name__ == '__main__':
    sys.exit(main())