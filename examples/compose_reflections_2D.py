"""

Compose three reflections in non-concurrent lines to get a glide reflection.

"""
import sys
import numpy as np
from isom.geom.objects import Line2D
from isom.geom.transform_2d import Reflection2D, compose_2d, GlideReflection2D


def main():

    print('Reflecting in lines:')
    # y = 2 - x
    line_1 = Line2D((1, 1), (1, -1))
    print(f'Line 1: {line_1}')
    refl_1 = Reflection2D(line_1)

    # y = 0
    line_2 = Line2D((0, 0), (1, 0))
    print(f'Line 2: {line_2}')
    refl_2 = Reflection2D(line_2)

    # x = 1
    line_3 = Line2D((1, 0), (0, 1))
    print(f'Line 3: {line_3}')
    refl_3 = Reflection2D(line_3)

    comp_12 = compose_2d(refl_1, refl_2)

    comp_123 = compose_2d(comp_12, refl_3)

    M_1 = refl_1.get_matrix()
    M_2 = refl_2.get_matrix()
    M_3 = refl_3.get_matrix()

    M_prod = M_3 @ M_2 @ M_1

    M_123 = comp_123.get_matrix()


    assert np.allclose(M_123, M_prod), 'Expect matrices to be close.'

    print('Composition of reflections:')
    print(comp_123)

    assert isinstance(comp_123, GlideReflection2D), 'Expect a glide reflection.'


    return 0



if __name__ == '__main__':
    sys.exit(main())