from __future__ import annotations

import numpy as np
from quaternion import quaternion
from abc import ABC, abstractmethod
from typing import List

from isom.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi, rotate_vector_3d, cross_product,
    angle_between_vectors, vecs_parallel, rotation_matrix_from_axis_and_angle, rotate_vectors_3d
)
from isom.geom.transform import Transform, Identity, is_identity
from isom.geom.objects import Plane3D, Line3D

class Transform3D(Transform, ABC):
    """
    Abstract base class for 3D isometries.
    """
    def copy(self):
        return eval(self.__repr__())


class OrthoTransform3D(Transform3D, ABC):
    """
    Abstract base class for 3D orthogonal transformations.
    """
    @abstractmethod
    def get_reflections(self) -> List[OrthoReflection3D]:
        """
        Return between one and three reflections for the orthogonal transformation.
        """

    @abstractmethod
    def inverse(self):
        """
        Return the inverse.
        """


class OrthoReflection3D(OrthoTransform3D):
    """
    Reflection in a plane through the origin.
    """

    def __init__(self, normal):
        """
        Reflection in a plane with the given normal.
        """
        super(OrthoReflection3D, self).__init__()
        self.normal = ensure_unit_vec(normal)
        self.plane = Plane3D(self.normal, [0, 0, 0])
        return

    def get_reflections(self):
        M = OrthoReflection3D(self.normal)
        return [M]

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """
        return [self.copy(), Identity(3)]

    @classmethod
    def from_two_step_form(cls, M, t):
        assert is_identity(t), 'Expect no translation.'

        if is_identity(M):
            return Identity(3)

        assert isinstance(M, OrthoReflection3D), 'Expect first transform to be orthogonal reflection.'
        return M.copy()


    def inverse(self):
        return self.copy()

    def apply(self, points):
        pts = validate_pts(points)
        n = self.normal

        # Components of points in direction of normal.
        comp_norm = n @ (n.T @ pts)
        ret = pts - 2.0 * comp_norm
        return ret

    def get_matrix(self):
        M = np.eye(4)
        I = np.eye(3)
        I_transf = self.apply(I)
        M[:3, :3] = I_transf
        return M

    def __repr__(self):
        n = self.normal.tolist()
        return f'OrthoReflection3D(\n    {n}\n)'

    def __str__(self):
        n = np.round(self.normal.flatten(), 2).tolist()
        return f'OrthoReflection3D(\n    {n}\n)'


class OrthoRotation3D(OrthoTransform3D):
    """
    Rotation about an axis going through (0, 0, 0)^T.
    """
    def __init__(self, axis, theta):
        super().__init__()

        self.axis = ensure_unit_vec(axis)
        self.angle = wrap_angle_minus_pi_to_pi(theta)

        ijk = np.eye(3)

        uvw = rotate_vectors_3d(ijk, self.axis, self.angle)

        reflections = reflections_for_frame(uvw)

        if len(reflections) == 1:
            assert is_identity(reflections[0]), 'Unexpected non-trivial reflection.'
            # Fake an identity transformation with two identical reflections.
            reflections = [OrthoReflection3D([1, 0, 0]), OrthoReflection3D([1, 0, 0])]

        assert len(reflections) == 2, 'Expect two reflections'

        self.refl_0 = reflections[0]
        self.refl_1 = reflections[1]

        return

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """
        return [self.copy(), Identity(3)]

    @classmethod
    def from_two_step_form(cls, M, t):
        assert is_identity(t), 'Expect no translation.'

        if is_identity(M):
            return Identity(3)

        assert isinstance(M, OrthoRotation3D), 'Expect first transform to be orthogonal rotation.'
        return M.copy()


    def inverse(self):
        M = OrthoRotation3D(self.axis, -1.0 * self.angle)
        return M

    def get_reflections(self):
        return [self.refl_0.copy(), self.refl_1.copy()]


    def to_quaternion(self):
        """
        Generate a quaternion to represent the orthogonal rotation.
        """
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

    @classmethod
    def from_reflections(cls, refl_0: OrthoReflection3D, refl_1: OrthoReflection3D):
        plane_0 = refl_0.plane
        plane_1 = refl_1.plane
        return OrthoRotation3D.from_planes(plane_0, plane_1)


    def followed_by(self, other: OrthoRotation3D):
        """
        Compose with another orthogonal rotation using geometric objects.
        """

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
        ax = self.axis.tolist()
        ang = self.angle
        return f'OrthoRotation3D(\n {ax},\n {ang}\n)'

    def __str__(self):
        ax = np.round(self.axis.flatten(), 2).tolist()
        ang = np.round(self.angle, 2)
        return f'OrthoRotation3D(\n {ax},\n {ang}\n)'


class OrthoImproperRotation3D(OrthoTransform3D):

    def __init__(self, axis, theta):

        super(OrthoImproperRotation3D, self).__init__()
        self.axis = ensure_unit_vec(axis)
        self.angle = wrap_angle_minus_pi_to_pi(theta)

        # A rotation and a reflection.
        self.rot = OrthoRotation3D(self.axis, self.angle)
        self.refl = OrthoReflection3D(self.axis)

        return

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """
        return [self.copy(), Identity(3)]

    @classmethod
    def from_two_step_form(cls, M, t):
        assert is_identity(t), 'Expect no translation.'

        if is_identity(M):
            return Identity(3)

        assert isinstance(M, OrthoImproperRotation3D), 'Expect first transform to be orthogonal improper rotation.'
        return M.copy()

    def apply(self, points):
        pts = validate_pts(points)
        pts = self.rot.apply(pts)
        pts = self.refl.apply(pts)
        return pts

    def get_matrix(self):
        M = self.rot.get_matrix()
        N = self.refl.get_matrix()
        NM = M @ N

        return NM

    def get_reflections(self):
        R0, R1 = self.rot.get_reflections()
        R2 = self.refl
        return [R0, R1, R2]

    def inverse(self):
        # TODO
        pass

    def __repr__(self):
        ax = self.axis.tolist()
        theta = self.angle
        return f"""OrthoImproperRotation(
{ax},
{theta},
)"""

    def __str__(self):
        ax = np.round(self.axis, 2).tolist()
        theta = np.round(self.angle, 2)
        return f"""OrthoImproperRotation(
{ax},
{theta},
)"""





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
        """
        [OrthoTransform3D, Translation3D]
        """
        return [Identity(3), self.copy()]

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def __repr__(self):
        v = self.vec.tolist()
        return f'Translation3D(\n {v}\n)'

    def __str__(self):
        v = np.round(self.vec.flatten(), 2)
        return f'Translation3D(\n {v}\n)'


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
        """
        [OrthoTransform3D, Translation3D]
        """
        p = self.point
        q = self.ortho_reflection.apply(p)
        v = p - q
        t_v = Translation3D(v)
        M = self.ortho_reflection.copy()
        return [M, t_v]

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
        plane_repr = repr(self.plane)
        return f'Reflection3D(\n{plane_repr}\n))'

    def __str__(self):
        plane_str = str(self.plane)
        return f'Reflection3D(\n{plane_str}\n))'


class Rotation3D(Transform3D):

    def __init__(
            self,
            point,
            axis_dir,
            theta
    ):
        """
        A rotation through 'theta' about an axis that
        goes through 'point' with the direction 'axis_dir'.
        """

        super().__init__()
        self.ortho_rot = OrthoRotation3D(axis_dir, theta)

        line = Line3D(point, axis_dir)
        self.point = line.nearest([0, 0, 0])

        self.T_inv = Translation3D(-1.0 * self.point)
        self.T = Translation3D(self.point)


        return

    def followed_by(self, other: Rotation3D):
        # TODO:
        pass
        # L = self.ortho_rot
        # K = other.ortho_rot
        # M = L.followed_by(K)
        #
        # p = self.point
        # r = other.point
        # u = r - K.apply(r - p) - M.apply(p)
        # trans = Translation3D(u)
        #
        # trans_orig_rot = TransOriginRotation3D.from_transforms(M, trans)
        #
        # return trans_orig_rot.to_trans_rot()

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """

        p = self.point
        q = self.ortho_rot.apply(p)
        v = p - q
        t_v = Translation3D(v)
        M = self.ortho_rot.copy()
        return [M, t_v]


    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def to_trans_origin_rot(self):
        # TODO: ?? If at all.
        pass
        # vec = self.point - self.ortho_rot.apply(self.point)
        # return TransOriginRotation3D(vec, self.ortho_rot.axis, self.ortho_rot.angle)

    def apply(self, points):
        pts = validate_pts(points)
        pts = self.T_inv.apply(pts)
        pts = self.ortho_rot.apply(pts)
        pts = self.T.apply(pts)
        return pts


    def get_matrix(self):
        M_T_inv = self.T_inv.get_matrix()
        M_rot = self.ortho_rot.get_matrix()
        M_T = self.T.get_matrix()

        return M_T @ M_rot @ M_T_inv

    def is_close(self, other: Rotation3D):

        R = self.ortho_rot.get_matrix()
        R_other = other.ortho_rot.get_matrix()

        p = self.point
        p_other = other.point

        return np.allclose(R, R_other) and np.allclose(p, p_other)

    def __repr__(self):
        c = self.point.tolist()
        ax = self.ortho_rot.axis.tolist()
        ang = self.ortho_rot.angle.tolist()
        return f'Rotation3D(\n {c},\n {ax},\n {ang}\n)'


    def __str__(self):
        c = np.round(self.point.flatten(), 2).tolist()
        ax = np.round(self.ortho_rot.axis.flatten(), 2).tolist()
        ang = np.round(self.ortho_rot.angle, 2).tolist()
        return f'Rotation3D(\n {c},\n {ax},\n {ang}\n)'


class ImproperRotation3D(Transform3D):

    def __init__(
            self,
            point,
            axis_dir,
            theta
    ):
        super(ImproperRotation3D, self).__init__()

        self.axis = ensure_unit_vec(axis_dir)
        self.angle = wrap_angle_minus_pi_to_pi(theta)

        self.point = ensure_vec(point)
        self.ortho_imp_rot = OrthoImproperRotation3D(self.axis, self.angle)

        return

    def two_step_form(self):
        """
        [OrthoTransform3D, Translation3D]
        """
        p = self.point
        q = self.ortho_imp_rot.apply(p)
        v = p - q
        t_v = Translation3D(v)
        M = self.ortho_imp_rot.copy()
        return [M, t_v]

    @classmethod
    def from_two_step_form(cls, M, t):
        # TODO
        pass

    def get_matrix(self):
        t_p = Translation3D(self.point)
        t_p_inv = Translation3D(-1.0 * self.point)

        T_p = t_p.get_matrix()
        T_p_inv = t_p_inv.get_matrix()
        M = self.ortho_imp_rot.get_matrix()

        return T_p @ M @ T_p_inv

    def apply(self, points):
        pts = validate_pts(points)

        t_p = Translation3D(self.point)
        t_p_inv = Translation3D(-1.0 * self.point)

        pts = t_p_inv.apply(pts)
        pts = self.ortho_imp_rot.apply(pts)
        pts = t_p.apply(pts)

        return pts

    def __repr__(self):
        p = self.point.tolist()
        ax = self.axis.tolist()
        ang = self.angle
        return f"""ImproperRotation3D(
{p},
{ax},
{ang}
)"""

    def __str__(self):
        p = np.round(self.point, 2).tolist()
        ax = np.round(self.axis, 2).tolist()
        ang = np.round(self.angle, 2)
        return f"""ImproperRotation3D(
{p},
{ax},
{ang}
)"""


class GlideReflection3D(Transform3D):
    """
    A combination of a reflection and a translation
    parallel to the plane of reflection.
    """

    def __init__(self):
        super().__init__()

        return

    def apply(self, points):
        pass

    def get_matrix(self):
        pass

    def two_step_form(self):
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        pass


    def __repr__(self):
        pass

    def __str__(self):
        pass


class Twist3D(Transform3D):
    """
    A combination of a rotation and a translation
    parallel to the axis of rotation.
    """

    def __init__(self):
        super().__init__()

        return

    def apply(self, points):
        pass

    def get_matrix(self):
        pass

    def two_step_form(self):
        pass

    @classmethod
    def from_two_step_form(cls, M, t):
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass


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

def transf_3d_from_two_step(M, t):
    # TODO:
    pass

def flip_two_step_form_3D(Mt):
    # TODO:
    pass

def compose_3d(A, B):
    # TODO:
    pass

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
        I = Identity(3)
        return [I]

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
    # Random 3D rotation.
    P = np.random.rand(3) * 10
    axis = np.random.rand(3) - [0.5, 0.5, 0.5]
    alpha = np.random.rand() * 2.0 * np.pi

    rot = Rotation3D(P, axis, alpha)
    return rot

def random_ortho_improper_rotation_3d():
    # Random orthogonal 3D improper rotation.
    axis = np.random.rand(3) - [0.5, 0.5, 0.5]
    alpha = np.random.rand() * 2.0 * np.pi

    return OrthoImproperRotation3D(axis, alpha)

def random_improper_rotation_3d():
    # Random 3D improper rotation.
    P = np.random.rand(3) * 10
    axis = np.random.rand(3) - [0.5, 0.5, 0.5]
    alpha = np.random.rand() * 2.0 * np.pi
    return ImproperRotation3D(P, axis, alpha)

