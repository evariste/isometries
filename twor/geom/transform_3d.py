from __future__ import annotations

import numpy as np
from quaternion import quaternion
from abc import ABC, abstractmethod
from typing import List
from twor.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi, rotate_vector, cross_product,
    angle_between_vectors, vecs_parallel, rotation_matrix_from_axis_and_angle
)
from twor.geom.transform import Transform, Identity
from twor.geom.objects import Plane3D

class Transform3D(Transform, ABC):
    pass

class OrthoTransform3D(Transform3D, ABC):

    @abstractmethod
    def get_reflections(self) -> List[OrthoReflection3D]:
        """
        Return one or two reflections for the orthogonal transformation.
        """

class OrthoReflection3D(OrthoTransform3D):
    """
    Reflection in a plane through the origin.
    """

    def __init__(self, normal):
        """
        normal is the plane normal.
        """

        super(OrthoReflection3D, self).__init__()

        self.normal = ensure_unit_vec(normal)
        self.plane = Plane3D(self.normal, [0, 0, 0])


        return

    def get_reflections(self):
        M = OrthoReflection3D(self.normal)
        return [M]

    def two_step_form(self):
        M = OrthoReflection3D(self.normal)
        I = Identity(3)
        return [M, I]

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def apply(self, points):

        pts = validate_pts(points)
        n = self.normal

        # Component of points in normal direction.
        comp_norm = n @ (n.T @ pts)
        ret = pts - 2 * comp_norm
        return ret


    def get_matrix(self):
        # TODO:
        pass

class OrthoRotation3D(OrthoTransform3D):
    """
    Rotation about an axis going through (0, 0, 0).
    """
    def __init__(self, axis, angle):

        super().__init__()

        self.axis = ensure_unit_vec(axis)
        self.angle = wrap_angle_minus_pi_to_pi(angle)

        origin = [0, 0, 0]
        z_vec = [0, 0, 1]

        axis_plane = Plane3D(self.axis, origin)
        xy_plane = Plane3D(z_vec,  origin)

        O = ensure_vec([0, 0, 0])

        if axis_plane.parallel_to(xy_plane):
            u = [1, 0, 0]
            v = rotate_vector(u, self.axis, self.angle / 2.0)
            plane_0 = Plane3D(u, O)
            plane_1 = Plane3D(v, O)
        else:
            l = axis_plane.intersection(xy_plane)

            assert l.contains_point(O), 'Unexpected line of intersection.'
            l.set_start_point(O)

            P = l(10)
            Q = O + 10 * self.axis
            R = rotate_vector(P, self.axis, self.angle / 2.0)

            plane_0 = Plane3D.from_points(O, P, Q)
            plane_1 = Plane3D.from_points(O, R, Q)

        self.refl_0 = Reflection3D(plane_0)
        self.refl_1 = Reflection3D(plane_1)

        return

    def two_step_form(self):
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_reflections(self):
        # TODO:
        pass


    def to_quaternion(self):
        v = self.axis
        theta = self.angle

        c = float(np.cos(theta / 2.0))
        s = float(np.sin(theta / 2.0))

        # sin(t / 2) v
        sv = s * v.flatten()

        q = quaternion(c, *sv)

        return q

    @classmethod
    def from_planes(cls, plane_0: Plane3D, plane_1: Plane3D):
        if plane_0.parallel_to(plane_1):
            return cls([1, 0, 0], 0)

        n_0 = plane_0.normal
        n_1 = plane_1.normal

        axis = cross_product(n_0, n_1)

        theta = angle_between_vectors(n_0, n_1)

        return cls(axis, 2.0 * theta)

    def followed_by(self, other: OrthoRotation3D):

        if vecs_parallel(self.axis, other.axis):
            if np.allclose(self.axis, other.axis):
                return OrthoRotation3D(self.axis, self.angle + other.angle)
            else:
                # Axes are opposing each other.
                return OrthoRotation3D(self.axis, self.angle - other.angle)

        O = ensure_vec([0, 0, 0])
        P = 10 * self.axis
        Q = 10 * other.axis
        plane_shared = Plane3D.from_points(O, P, Q)
        n_shared = plane_shared.normal

        n_0 = rotate_vector(n_shared, self.axis, -0.5 * self.angle)

        n_1 = rotate_vector(n_shared, other.axis, 0.5 * other.angle)

        plane_0 = Plane3D(n_0, O)
        plane_1 = Plane3D(n_1, O)

        return OrthoRotation3D.from_planes(plane_0, plane_1)

    def get_matrix(self):
        """
        Get a homogeneous matrix from the composed reflections.
        """
        M0 = self.refl_0.get_matrix()
        M1 = self.refl_1.get_matrix()
        return M1 @ M0

    def get_matrix_B(self):
        """
        Get a homogeneous matrix via an external function.
        """
        R = rotation_matrix_from_axis_and_angle(self.axis, self.angle)
        M = np.eye(4)
        M[:3, :3] = R
        return M

    def apply(self, points):
        pts = validate_pts(points)
        pts = self.refl_0.apply(pts)
        pts = self.refl_1.apply(pts)
        return pts

    def __repr__(self):
        ax = np.round(self.axis.flatten(), 2)
        ang = np.round(self.angle, 2)
        return f'OriginRotation3D(\n {ax},\n {ang}\n)'


class OrthoImproperRotation(OrthoTransform3D):

    def __init__(self):

        super(OrthoImproperRotation, self).__init__()

        return

    def two_step_form(self):
        # TODO
        pass

    def apply(self, points):
        # TODO
        pass

    def get_matrix(self):
        # TODO
        pass

    def get_reflections(self):
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass





class Reflection3D(Transform3D):

    def __init__(self, plane: Plane3D):
        super().__init__()

        self.plane = plane
        # TODO: Implement with an OrthoReflection3D

        return


    def apply(self, points):

        pts = validate_pts(points)

        X = self.plane.pt
        n = self.plane.normal

        disps = pts - X

        coeff_norm = n.T @ disps

        comp_norm = n @ coeff_norm

        ret = disps - 2 * comp_norm + X

        return ret

    def two_step_form(self):
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_matrix(self):

        pt = self.plane.pt
        n = self.plane.normal

        T_inv = Translation3D(-1.0 * pt).get_matrix()
        T = Translation3D(pt).get_matrix()

        M = np.eye(4)

        I3 = np.eye(3)
        coeff = n.T @ I3
        comp_norm = n @ coeff
        M[:3, :3] = I3 - 2 * comp_norm

        H = T @ M @ T_inv
        return H

    def __repr__(self):
        normal = np.round(self.plane.normal.flatten(), 2)
        pt = np.round(self.plane.pt.flatten(), 2)
        return f'Reflection3D(\n {normal},\n {pt}\n)'


class Rotation3D(Transform3D):

    def __init__(self, point, axis_dir, angle):
        """
        A rotation through 'angle' about an axis that
        goes through 'point' with the direction 'axis_dir'.
        """

        super().__init__()
        self.orig_rot = OrthoRotation3D(axis_dir, angle)
        self.point = ensure_vec(point)
        self.T_inv = Translation3D(-1.0 * self.point)
        self.T = Translation3D(self.point)

        return

    def followed_by(self, other: Rotation3D):
        L = self.orig_rot
        K = other.orig_rot
        M = L.followed_by(K)

        p = self.point
        r = other.point
        u = r - K.apply(r - p) - M.apply(p)
        trans = Translation3D(u)

        trans_orig_rot = TransOriginRotation3D.from_transforms(M, trans)

        return trans_orig_rot.to_trans_rot()

    def two_step_form(self):
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def to_trans_origin_rot(self):
        vec = self.point - self.orig_rot.apply(self.point)
        return TransOriginRotation3D(vec, self.orig_rot.axis, self.orig_rot.angle)

    def apply(self, points):
        pts = validate_pts(points)
        pts = self.T_inv.apply(pts)
        pts = self.orig_rot.apply(pts)
        pts = self.T.apply(pts)
        return pts


    def get_matrix(self):
        M_T_inv = self.T_inv.get_matrix()
        M_rot = self.orig_rot.get_matrix()
        M_T = self.T.get_matrix()

        return M_T @ M_rot @ M_T_inv

    def __repr__(self):
        c = np.round(self.point.flatten(), 2)
        ax = np.round(self.orig_rot.axis.flatten(), 2)
        ang = np.round(self.orig_rot.angle, 2)
        return f'Rotation3D(\n {c},\n {ax},\n {ang}\n)'

    def is_close(self, other: Rotation3D):

        R = self.orig_rot.get_matrix()
        R_other = other.orig_rot.get_matrix()

        p = self.point
        p_other = other.point

        return np.allclose(R, R_other) and np.allclose(p, p_other)

class ImproperRotation3D(Transform3D):

    def __init__(self):
        super(ImproperRotation3D, self).__init__()

        return

    def two_step_form(self):
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_matrix(self):
        # TODO
        pass

    def apply(self, points):
        # TODO
        pass

    def matrix_equals(self, other: Transform):
        # TODO
        pass



class TransOriginRotation3D(Transform3D):
    """
    A two-step transformation of the form
    T M : x -> T ( M (x) )
    where
     - M is an origin rotation
     - T is a translation.
    """
    def __init__(self, transvector, axis, angle):
        super().__init__()
        self.origin_rot = OrthoRotation3D(axis, angle)
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
        strs = ['TransOriginRotation3D', repr(self.origin_rot), repr(self.tra)]
        return '\n'.join(strs)

class TransRotation3D(Transform3D):
    """
    A two-step transformation of the form
    T M : x -> T ( M (x) )

    where:
     - M is an general rotation
     - T is a translation, either zero or parallel to the axis of M.
    """


    def __init__(self, point, axis, angle, transvector):
        super().__init__()

        self.gen_rot = Rotation3D(point, axis, angle)
        self.tra = Translation3D(transvector)

        return

    @classmethod
    def from_transforms(cls, rotation: Rotation3D, trans: Translation3D):

        pt = rotation.point
        ax = rotation.orig_rot.axis
        ang = rotation.orig_rot.angle
        v = trans.vec

        return cls(pt, ax, ang, v)

    def two_step_form(self):
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





class Translation3D(Transform3D):

    def __init__(self, v):

        super().__init__()
        self.vec = ensure_vec(v)

    def apply(self, points):
        pts = validate_pts(points)
        return pts + self.vec

    def get_matrix(self):
        T = np.eye(4)
        T[:3, -1] = np.squeeze(self.vec)
        return T

    def two_step_form(self):
        I = Identity(3)
        t = Translation3D(self.vec)
        return [I, t]

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def __repr__(self):
        v = np.round(self.vec.flatten(), 2)
        return f'Translation3D(\n {v}\n)'

