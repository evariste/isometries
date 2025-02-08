import os
import numpy as np
from twor.geom.objects import Glyph3D
from twor.geom.transform_3d import OriginRotation3D, Translation3D

os.makedirs('output/ex3', exist_ok=True)

g = Glyph3D()

t = Translation3D([6, 0, 0])

g = g.apply_transformation(t)

R_A = OriginRotation3D([0, 1, 0], -1.0 * np.pi / 4.0)
R_B = OriginRotation3D([0, 0, 1], np.pi / 3.0)

g_A = g.apply_transformation(R_A)
g_B = g_A.apply_transformation(R_B)

g.save(f'output/ex3/glyph0.vtk')
g_A.save(f'output/ex3/glyph1.vtk')
g_B.save(f'output/ex3/glyph2.vtk')
