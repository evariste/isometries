"""
Example of two orthogonal rotations applied to a glyph in 3D.

Results are saved in VTK files for visualisation in a tool like Paraview.
"""

import sys
import os
import numpy as np
from isom.geom.objects import Glyph3D
from isom.geom.transform_3d import OrthoRotation3D, Translation3D

def main():
    os.makedirs('output/two_ortho_rots_3D', exist_ok=True)

    g = Glyph3D()

    t = Translation3D([6, 0, 0])

    g = g.apply_transformation(t)

    R_A = OrthoRotation3D([0, 1, 0], -1.0 * np.pi / 4.0)
    R_B = OrthoRotation3D([0, 0, 1], np.pi / 3.0)

    g_A = g.apply_transformation(R_A)
    g_B = g_A.apply_transformation(R_B)

    g.save(f'output/ex3/glyph0.vtk')
    g_A.save(f'output/ex3/glyph1.vtk')
    g_B.save(f'output/ex3/glyph2.vtk')

    return 0

if __name__ == '__main__':
    sys.exit(main())