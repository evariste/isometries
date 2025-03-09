from __future__ import annotations

import numpy as np
from quaternion import quaternion
from abc import ABC, abstractmethod
from typing import List

from isom.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi, rotate_vector_3d, cross_product,
    angle_between_vectors, vecs_parallel, rotation_matrix_from_axis_and_angle, rotate_vectors_3d, vecs_perpendicular
)
from isom.geom.transform import Transform, Identity, is_identity
from isom.geom.objects import Plane3D, Line3D
from isom.utils.general import vector_pair_to_rotation_axis_angle


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
    def from_reflections(cls, refl):
        if is_identity(refl):
            return Identity(3)
        assert isinstance(refl, OrthoReflection3D), 'Expect a reflection.'
        return refl.copy()

    @classmethod
    def from_planes(cls, plane):
        assert isinstance(plane, Plane3D), 'Expect a plane.'
        return cls(plane.normal)

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

        axis = ensure_unit_vec(cross_product(n_0, n_1))

        theta = 2.0 * angle_between_vectors(n_0, n_1)

        return cls(axis, theta)

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

    @classmethod
    def from_reflections(
            cls,
            r_0: OrthoReflection3D,
            r_1: OrthoReflection3D,
            r_2: OrthoReflection3D,
    ):
        plane_0 = r_0.plane
        plane_1 = r_1.plane
        plane_2 = r_2.plane

        # Any pair of successive identical reflections will cancel each other out.
        if plane_0.parallel_to(plane_1):
            return OrthoReflection3D(plane_2.normal)

        if plane_1.parallel_to(plane_2):
            return OrthoReflection3D(plane_0.normal)

        axis_01 = plane_0.intersection(plane_1)
        axis_12 = plane_1.intersection(plane_2)

        if vecs_parallel(axis_01.direction, axis_12.direction):
            # All three planes intersect in a single line.

            n_0 = plane_0.normal
            n_1 = plane_1.normal
            n_2 = plane_2.normal

            # The rotation that takes n_1 to n_2.
            axis, theta = vector_pair_to_rotation_axis_angle(n_1, n_2)

            # Apply this to n_0
            n_0_R = rotate_vector_3d(n_0, axis, theta)
            # The rotated version of plane 1 will coincide with plane 2.
            # So r_1 and r_2 will cancel.
            return OrthoReflection3D(n_0_R)



        # Convert the three reflections into canonical form
        # where the plane for the last is perpendicular to
        # the planes for the first two.

        # Step 1. Rotate (s_0, s_1) until s_1 perp s_2
        # Step 2. Rotate (s_1, s_2) until s_2 perp s_0

        # Step 1
        n_0 = r_0.plane.normal
        n_1 = r_1.plane.normal
        n_2 = r_2.plane.normal

        # Step 1.
        n_01 = ensure_unit_vec(cross_product(n_0, n_1))

        if vecs_parallel(n_01, n_2):
            # s_1 plane is already perpendicular to s_2 plane.
            pass
        else:
            # Rotate (s_0, s_1)
            # A vector perpendicular to n01 and s2 plane.
            w = ensure_unit_vec(cross_product(n_2, n_01))

            # Rotate (s_0, s_1) about n_01 until s_1 aligns with w (i.e., perpendicular to s_2).
            axis, theta = vector_pair_to_rotation_axis_angle(n_1, w)

            n_0_R = rotate_vector_3d(n_0, axis, theta)
            n_1_R = rotate_vector_3d(n_1, axis, theta)

            # Replace s_0, s_1
            r_0 = OrthoReflection3D(n_0_R)
            r_1 = OrthoReflection3D(n_1_R)

            assert np.isclose(n_1_R.T @ n_2, 0.0), 'Expect perpendicular vectors.'

            n_0 = n_0_R
            n_1 = n_1_R


        # Step 2
        n_12 = ensure_unit_vec(cross_product(n_1, n_2))

        if vecs_parallel(n_12, n_0):
            # s_2 plane is already perpendicular to s_0 plane.
            pass
        else:
            # Rotate (s_1, s_2)
            # A vector perpendicular to n_12 and s0 plane.
            w = ensure_unit_vec(cross_product(n_0, n_12))

            # Rotate (s_1, s_2) about n_12 until s_2 aligns with w (so it is perpendicular to s_0).
            axis, theta = vector_pair_to_rotation_axis_angle(n_2, w)

            n_1_R = rotate_vector_3d(n_1, axis, theta)
            n_2_R = rotate_vector_3d(n_2, axis, theta)

            assert np.isclose(n_2_R.T @ n_0, 0.0), 'Expect perpendicular vectors.'

            r_1 = OrthoReflection3D(n_1_R)
            # r_2 = OrthoReflection3D(n_2_R)

            n_1 = n_1_R
            # n_2 = n_2_R


        theta = 2.0 * angle_between_vectors(n_0, n_1)
        line = r_0.plane.intersection(r_1.plane)
        return OrthoImproperRotation3D(line.direction, theta)

    @classmethod
    def from_planes(
            cls,
            plane_0: Plane3D,
            plane_1: Plane3D,
            plane_2: Plane3D,
    ):

        r_0 = OrthoReflection3D(plane_0)
        r_1 = OrthoReflection3D(plane_1)
        r_2 = OrthoReflection3D(plane_2)
        return OrthoImproperRotation3D.from_reflections(r_0, r_1, r_2)



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
        R0, R1, R2 = self.get_reflections()
        rev_refls = [R2, R1, R0]
        return OrthoImproperRotation3D.from_reflections(*rev_refls)


    def __repr__(self):
        ax = self.axis.tolist()
        theta = self.angle
        return f"""OrthoImproperRotation3D(
{ax},
{theta},
)"""

    def __str__(self):
        ax = np.round(self.axis, 2).tolist()
        theta = np.round(self.angle, 2)
        return f"""OrthoImproperRotation3D(
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
        assert is_identity(M), 'Expect identity for orthogonal transformation.'

        if is_identity(t):
            return Identity(3)

        assert isinstance(t, Translation3D), 'Expect second transformation to be a translation.'
        return t.copy()

    def __repr__(self):
        v = self.vec.tolist()
        return f'Translation3D(\n {v}\n)'

    def __str__(self):
        v = np.round(self.vec, 2).tolist()
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

        if is_identity(M):
            if is_identity(t):
                return Identity(3)
            else:
                assert isinstance(t, Translation3D), 'Expect second transformation to be a Translation.'
                return t.copy()

        assert isinstance(M, OrthoReflection3D), 'Expect orthogonal transformation to be an orthogonal reflection.'

        if is_identity(t):
            return M.copy()

        plane_normal = M.plane.normal
        t_vec = t.vec
        if not vecs_parallel(plane_normal, t_vec):
            # TODO:
            return GlideReflection3D()

        # Both M and t are non-trivial and translation vector
        # is perpendicular to the plane of reflection of M.
        O = ensure_vec([0, 0, 0])
        P = t.apply(M.apply(O))
        # Point on the reflection plane.
        Q = (O + P) / 2.0

        plane = Plane3D(M.plane.normal, Q)
        return Reflection3D(plane)

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

        if is_identity(M):
            if is_identity(t):
                return Identity(3)
            else:
                assert isinstance(t, Translation3D), 'Expect second transformation to be a Translation.'
                return t.copy()

        assert isinstance(M, OrthoRotation3D), 'Expect orthogonal transformation to be an orthogonal rotation.'

        if is_identity(t):
            return M.copy()

        # Both M and t are non-trivial.

        if not vecs_perpendicular(M.axis, t.vec):
            return Twist3D.from_two_step_form(M, t)

        # Translation is perpendicular to the rotation axis.
        t_dir = ensure_unit_vec(t.vec)

        # Let u be the translation vector for t.
        # Let theta be the angle for rotation M.

        # We want to find p such that p - p' = u
        # where p' is the image of p under M and t, i.e., p' = t(M(p)).

        # Let A be the midpoint of p and p'.
        # O, p, p' make an isosceles triangle.
        # O A p make a right-angled triangle where the angle at O is theta / 2.

        u = t.vec

        len_u = np.sqrt(np.sum(u * u))

        theta = M.angle
        tan_theta_half = np.tan(theta / 2.0)
        assert np.abs(tan_theta_half) > 0, 'Should not have zero angle.'

        # A is
        len_OA = len_u / 2.0 / tan_theta_half

        dir_OA = ensure_unit_vec(cross_product(M.axis, t_dir))

        OA = len_OA * dir_OA

        p = OA + u / 2.0

        return Rotation3D(p, M.axis, M.angle)


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

        if is_identity(M):
            if is_identity(t):
                return Identity(3)
            else:
                assert isinstance(t, Translation3D), 'Expect second transformation to be a Translation.'
                return t.copy()

        msg = 'Expect orthogonal transformation to be an orthogonal improper rotation.'
        assert isinstance(M, OrthoImproperRotation3D), msg

        if is_identity(t):
            return M.copy()

        # Both M and t are non-trivial.

        # What is the image of some point under the transformation?
        # Pick the origin as an example.
        O = ensure_vec([0, 0, 0])
        P = t.apply(M.apply(O))

        # The displacement from O to P will have a component parallel
        # to the axis of rotation and a component perpendicular.
        OP = P - O
        OP_para = M.axis.T @ OP * M.axis
        OP_perp = OP - OP_para

        # An orthogonal rotation about an axis parallel to the one for M.
        N = OrthoRotation3D(M.axis, M.angle)

        # The rotation part of our improper rotation.
        R = Rotation3D.from_two_step_form(N, Translation3D(OP_perp))

        # The axis line for the rotation part.
        R_line = Line3D(R.point, R.ortho_rot.axis)

        # The projection of the origin onto the axis.
        Q = R_line.nearest(O)

        # The fixed point of the improper rotation is along the axis
        # from Q, half way along the parallel component of the displacement OP
        X = Q + 0.5 * OP_para

        return ImproperRotation3D(X, M.axis, M.angle)

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


def transf_3d_from_two_step(M: OrthoTransform3D, t):
    """
    Return the single transformation that is equivalent to
    applying the orthogonal transformation M followed by the
    translation t.
    """

    assert isinstance(t, Translation3D) or is_identity(
        t), 'Expect second transformation to be a translation or identity.'

    if isinstance(M, OrthoReflection3D):
        return Reflection3D.from_two_step_form(M, t)
    elif isinstance(M, OrthoRotation3D):
        return Rotation3D.from_two_step_form(M, t)
    elif isinstance(M, OrthoImproperRotation3D):
        return ImproperRotation3D.from_two_step_form(M, t)
    elif is_identity(M):
        return Translation3D.from_two_step_form(M, t)
    else:
        raise Exception('Unexpected type for orthogonal transformation.')


def flip_two_step_form_3D(two_step_transf):
    """
    two_step_transf is a list with a pair of transformations.
    For an orthogonal transformation M and a translation t,
    the form is either [M, t] (orthogonal first) or [t, M]
    (translation first).

    Return an two step form with the order reversed that is
    equivalent.
    """

    assert len(two_step_transf) == 2, 'Expect length of two-step form to be 2'

    t0 = two_step_transf[0]
    t1 = two_step_transf[1]
    I = Identity(3)

    if is_identity(t0):
        # Put t1 at index 0
        return [t1, I]

    if is_identity(t1):
        # Put t0 at index 1
        return [I, t0]

    # Neither t0, nor t1, are the identity.

    if is_ortho_3d(t0):
        # Form [M, t]
        assert is_translation_3d(t1), 'Expect second transform to be a translation.'
        # Translation vector
        p = t1.vec

        # Alias for orthogonal part:
        M = t0
        M_inv = M.inverse()

        q = M_inv.apply(p)

        t_q = Translation3D(q)

        return [t_q, M]

    # First transformation is not orthogonal.
    assert is_translation_3d(t0), 'Expect first tranform to be a translation'
    assert is_ortho_3d(t1), 'Expect second transform to be orthogonal.'

    # Form is [t, M]

    # Translation vector
    p = t0.vec

    # Alias for orthogonal part
    M = t1

    q = M.apply(p)

    t_q = Translation3D(q)

    return [M, t_q]



def compose_3d(transf_A: Transform3D, transf_B: Transform3D):
    """
    Compose two 3D transformations.
    Return the 3D transformation that results from applying
    transf_A followed by transf_B.
    """

    M_a, t_a = transf_A.two_step_form()
    M_b, t_b = transf_B.two_step_form()

    if is_identity(transf_A):
        return transf_3d_from_two_step(M_b, t_b)

    if is_identity(transf_B):
        return transf_3d_from_two_step(M_a, t_a)


    # Application sequence (starting from the left:
    # M_a t_a M_b t_b

    # Flip the middle pair.
    M_c, t_c = flip_two_step_form_3D([t_a, M_b])

    # Sequence can now be:
    # M_a M_c t_c t_b

    # We should have M_c == M_b
    assert M_b.matrix_equals(M_c), 'Unexpected change in orthogonal part after flip.'

    # Form is
    # M_a M_b  t_c t_b

    # Orthogonal part of result.
    M_out = compose_ortho_3d(M_a, M_b)

    v = ensure_vec([0, 0, 0])
    # Accumulate translation vectors (if they are not identity transforms).
    if is_translation_3d(t_b):
        v += t_b.vec
    if is_translation_3d(t_c):
        v += t_c.vec

    if np.allclose(v, [0, 0, 0]):
        # No translation happened, we can just return the orthogonal part.
        return M_out

    t_out = Translation3D(v)

    result = transf_3d_from_two_step(M_out, t_out)

    return result



def compose_ortho_3d(t_a: OrthoTransform3D, t_b: OrthoTransform3D):
    """
    Compose a pair of orthogonal 3D transformations.
    """

    valid_a = is_ortho_3d(t_a) or is_identity(t_a)
    valid_b = is_ortho_3d(t_b) or is_identity(t_b)
    assert valid_a or valid_b, 'Invalid transformation type(s).'

    if is_identity(t_a):
        return t_b

    if is_identity(t_b):
        return t_a


    ijk = np.eye(3)
    uvw = t_b.apply(t_a.apply(ijk))
    refls = reflections_for_frame(uvw)

    if len(refls) == 0:
        return Identity(3)

    if len(refls) == 1:
        return refls[0]

    if len(refls) == 2:
        return OrthoRotation3D.from_reflections(*refls)

    if len(refls) == 3:
        return OrthoImproperRotation3D.from_reflections(*refls)

    raise Exception('Unexpected number of reflections.')



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

    assert np.allclose(u, R.apply(i)), 'Expect R(i) = u'

    R_j = R.apply(j)

    if np.allclose(R_j, v):
        S = Identity(3)
    else:
        normal = cross_product(u, v + R_j)
        S = OrthoReflection3D(normal)

    SR_i = S.apply(R.apply(i))
    SR_j = S.apply(R.apply(j))
    SR_k = S.apply(R.apply(k))

    assert np.allclose(SR_i, u), 'Expect S(R(i)) = u'
    assert np.allclose(SR_j, v), 'Expect S(R(j)) = v'

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


def random_glide_reflection_3d():
    # TODO:
    pass

def random_twist_3d():
    # TODO:
    pass

def is_ortho_3d(transf: Transform3D):
    return isinstance(transf, OrthoTransform3D)

def is_translation_3d(transf: Transform3D):
    return isinstance(transf, Translation3D)
