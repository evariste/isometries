from __future__ import annotations

import sys
import numpy as np
from isom.utils.general import random_rotation_3D, ensure_vec, cross_product, ensure_unit_vec
from isom.geom.transform_3d import Rotation3D, Transform3D, OrthoRotation3D, Translation3D

def main():
    ax, theta = random_rotation_3D()

    P = np.random.rand(3) * 10
    P = ensure_vec(P)

    rot_A = Rotation3D(P, ax, theta)

    M_A = rot_A.get_matrix()

    transf_B = rot_A.to_trans_origin_rot()

    assert isinstance(transf_B, TransOriginRotation3D)

    M_B = transf_B.get_matrix()

    assert np.allclose(M_A, M_B)

    print(rot_A)


    print(transf_B)

    return 0




class TransOriginRotation3D(Transform3D):
    """
    A two-step transformation of the form
    T M : x -> T ( M (x) )
    where
     - M is an origin rotation
     - T is a translation.
    """
    def __init__(self, transvector, axis, theta):
        super().__init__()
        self.origin_rot = OrthoRotation3D(axis, theta)
        self.tra = Translation3D(transvector)

        return

    @classmethod
    def from_transforms(cls, originRotation: OrthoRotation3D, trans: Translation3D):
        v = trans.vec
        ax = originRotation.axis
        ang = originRotation.angle
        return cls(v, ax, ang)

    def to_trans_rot(self):

        v = self.tra.vec

        c = self.origin_rot.axis
        theta = self.origin_rot.angle

        v_para = c * (c.T @ v)
        u = v - v_para

        # TODO: what if u is zero vec?

        c_cross_u = cross_product(c, u)

        # TODO what if c x u is zero.

        w = ensure_unit_vec(c_cross_u)

        len_u = np.sqrt(np.sum(u * u))

        OA = len_u / 2.0 / np.tan(theta / 2.0)

        OA_vec = OA * w

        p = OA_vec + u / 2.0


        r3d = Rotation3D(p, c, theta)

        tra_new = Translation3D(v_para)

        trans_rot = TransRotation3D.from_transforms(r3d, tra_new)


        return trans_rot

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_matrix(self):
        M = self.origin_rot.get_matrix()
        T = self.tra.get_matrix()
        return T @ M

    def apply(self, points):
        pts = self.origin_rot.apply(points)
        pts = self.tra.apply(pts)
        return pts


    def __repr__(self):
        strs = ['TransOriginRotation3D', repr(self.origin_rot), repr(self.tra), ')']
        return '\n'.join(strs)

class TransRotation3D(Transform3D):
    """
    A two-step transformation of the form
    T M : x -> T ( M (x) )

    where:
     - M is an general rotation
     - T is a translation, either zero or parallel to the axis of M.
    """


    def __init__(self, point, axis, theta, transvector):
        super().__init__()

        self.gen_rot = Rotation3D(point, axis, theta)
        self.tra = Translation3D(transvector)

        return

    @classmethod
    def from_transforms(cls, rotation: Rotation3D, trans: Translation3D):

        pt = rotation.point
        ax = rotation.ortho_rot.axis
        ang = rotation.ortho_rot.angle
        v = trans.vec

        return cls(pt, ax, ang, v)

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def apply(self, points):
        pts = self.gen_rot.apply(points)
        pts = self.tra.apply(pts)
        return pts

    def get_matrix(self):
        M = self.gen_rot.get_matrix()
        T = self.tra.get_matrix()
        return T @ M


    def is_close(self, other: TransRotation3D):

        r3d = self.gen_rot
        r3d_other = other.gen_rot

        v = self.tra.vec
        v_other = other.tra.vec


        return r3d.is_close(r3d_other) and np.allclose(v, v_other)



    def __repr__(self):
        strs = ['TransRotation3D', repr(self.gen_rot), repr(self.tra)]
        return '\n'.join(strs)



if __name__ == '__main__':
    sys.exit(main())