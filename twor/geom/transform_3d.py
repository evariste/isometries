from __future__ import annotations

import numpy as np
from quaternion import quaternion
from abc import ABC, abstractmethod
from typing import List

from twor.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi, rotate_vector_3d, cross_product,
    angle_between_vectors, vecs_parallel, rotation_matrix_from_axis_and_angle, rotate_vectors_3d
)
from twor.geom.transform import Transform, Identity, is_identity
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

        M = np.eye(4)

        I = np.eye(3)
        I_transf = self.apply(I)
        M[:3, :3] = I_transf

        return M

    def __repr__(self):
        n = np.round(self.normal.flatten(), 2).tolist()
        return f'OrthoReflection3D(\n {n}\n)'


class OrthoRotation3D(OrthoTransform3D):
    """
    Rotation about an axis going through (0, 0, 0).
    """
    def __init__(self, axis, theta):

        super().__init__()

        self.axis = ensure_unit_vec(axis)
        self.angle = wrap_angle_minus_pi_to_pi(theta)

        ijk = np.eye(3)

        uvw = rotate_vectors_3d(ijk, self.axis, self.angle)

        reflections = reflections_for_frame(uvw)

        self.refl_0 = reflections[0]
        self.refl_1 = reflections[1]

        return

    def two_step_form(self):
        # TODO
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_reflections(self):
        R0 = OrthoReflection3D(self.refl_0.normal)
        R1 = OrthoReflection3D(self.refl_1.normal)
        return [R0, R1]


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
            return Identity

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

        n_0 = rotate_vector_3d(n_shared, self.axis, -0.5 * self.angle)

        n_1 = rotate_vector_3d(n_shared, other.axis, 0.5 * other.angle)

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
        ax = np.round(self.axis.flatten(), 2).tolist()
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

        self.ortho_reflection = OrthoReflection3D(self.plane.normal)

        self.point = self.plane.point

        return


    def apply(self, points):

        p = self.point
        T = Translation3D(p)
        T_inv = Translation3D(-1.0 * p)

        pts = validate_pts(points)

        pts = T_inv.apply(pts)
        pts = self.ortho_reflection.apply(pts)
        pts = T.apply(pts)

        return pts


    def two_step_form(self):
        # TODO
        # q = self.ortho apply p
        # v = p - q
        # return self.ortho, t_v
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_matrix(self):

        pt = self.plane.point

        T_inv = Translation3D(-1.0 * pt).get_matrix()
        T = Translation3D(pt).get_matrix()

        M = self.ortho_reflection.get_matrix()

        return T @ M @ T_inv

    def __repr__(self):
        normal = np.round(self.plane.normal.flatten(), 2).tolist()
        pt = np.round(self.plane.point.flatten(), 2).tolist()
        return f'Reflection3D(\n Plane3D(\n {normal},\n {pt}\n))'


class Rotation3D(Transform3D):

    def __init__(self, point, axis_dir, theta):
        """
        A rotation through 'theta' about an axis that
        goes through 'point' with the direction 'axis_dir'.
        """

        super().__init__()
        self.orig_rot = OrthoRotation3D(axis_dir, theta)
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
        c = np.round(self.point.flatten(), 2).tolist()
        ax = np.round(self.orig_rot.axis.flatten(), 2).tolist()
        ang = np.round(self.orig_rot.angle, 2).tolist()
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

    def __repr__(self):
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


    def __init__(self, point, axis, theta, transvector):
        super().__init__()

        self.gen_rot = Rotation3D(point, axis, theta)
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


def frame_orthonormal(uvw):
    """
    Given three vectors [u, v, w] in a 3x3 array
    check if they are orthonormal.
    """
    assert uvw.shape == (3, 3), 'Expect 3x3 array.'
    return np.allclose(
        uvw.T @ uvw,
        np.eye(3)
    )


def reflections_for_frame(uvw):
    """
    The usual orthonormal basis for 3D is
    i=(1,0,0)^T, j=(0,1,0)^T, k=(0,0,1)^T

    If i, j, k are mapped to u, v, w under some orthogonal 3D
    transformation, find the sequence of reflections that can
    represent the tranformation. There will be no more than
    three.
    """
    assert frame_orthonormal(uvw), 'Expect orthonormal frame.'

    ijk = np.eye(3)

    if np.allclose(uvw, ijk):
        return Identity(3)

    i = ijk[:, [0]]
    j = ijk[:, [1]]
    k = ijk[:, [2]]

    u = uvw[:, [0]]
    v = uvw[:, [1]]
    w = uvw[:, [2]]


    # First reflection takes i to u.
    if np.allclose(i, u):
        R = Identity(3)
    else:
        normal = u - i
        R = OrthoReflection3D(normal)

    assert np.allclose(u, R.apply(i))

    R_j = R.apply(j)

    if np.allclose(R_j, v):
        S = Identity(3)
    else:
        normal = cross_product(u, v + R_j)
        S = OrthoReflection3D(normal)

    SR_i = S.apply(R.apply(i))
    SR_j = S.apply(R.apply(j))
    SR_k = S.apply(R.apply(k))

    assert np.allclose(SR_i, u)
    assert np.allclose(SR_j, v)

    if np.allclose(SR_k, w):
        T = Identity(3)
    else:
        T = OrthoReflection3D(w)

    reflections = [R, S, T]
    reflections = [r for r in reflections if not is_identity(r)]

    return reflections

def random_ortho_reflection_3d():
    # Random orthogonal 3D reflection.
    # Hacky method.
    v = np.random.rand(3) - [0.5, 0.5, 0.5]
    ortho_refl = OrthoReflection3D(v)
    return ortho_refl


def random_reflection_3d():
    # Random general 3D reflection.
    # Hacky method.
    n = np.random.rand(3) - [0.5, 0.5, 0.5]
    P = np.random.rand(3) * 10
    plane = Plane3D(n, P)
    return Reflection3D(plane)


def random_ortho_rotation_3d():
    # Random orthogonal 3D rotation.
    axis = np.random.rand(3) - [0.5, 0.5, 0.5]
    alpha = np.random.rand() * 2.0 * np.pi
    ortho_rot = OrthoRotation3D(axis, alpha)
    return ortho_rot

def random_rotation_3d():
    pass