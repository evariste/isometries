from __future__ import annotations


"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

from abc import ABC, abstractmethod
import numpy as np
from quaternion import quaternion
from twor.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi, rotate_vector, cross_product,
    angle_between_vectors, vecs_parallel, rotation_matrix_from_axis_and_angle
)
from twor.geom.objects import Line2D, Plane3D



class Transform(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def apply(self, points):
        """
        Apply to some points and return result.
        """
    @abstractmethod
    def get_matrix(self):
        """
        Return a homogeneous matrix for a transform.
        """


class Identity(Transform):
    def __init__(self, dimension):
        super().__init__()
        self.dim = dimension
        self.matrix = np.eye(1 + self.dim)

    def apply(self, points):
        return validate_pts(points)

    def get_matrix(self):
        return self.matrix

class OrthoTransform2D(Transform, ABC):
    pass

class Transform2D(Transform, ABC):
    pass

class OrthoReflection2D(OrthoTransform2D):
    """
    Orthogonal (linear) reflection in 2-D.
    """
    def __init__(self, direction):

        super().__init__()
        self.direction = ensure_unit_vec(direction)
        normal = [-1.0 * self.direction[1], self.direction[0]]
        self.normal = ensure_unit_vec(normal)

        return

    def apply(self, points):

        points_out = validate_pts(points)

        comp_along = self.direction @ (self.direction.T @ points_out)
        comp_perp = points_out - comp_along

        points_out = comp_along - comp_perp

        return points_out

    def get_matrix(self, ):

        M = np.eye(3)
        M[:2, :2] = self.apply(np.eye(2))

        return M

    def followed_by(self, other: OrthoReflection2D):
        line_1 = Line2D((0, 0), self.direction)
        line_2 = Line2D((0, 0), other.direction)
        return OrthoRotation2D.from_lines(line_1, line_2)

class Reflection2D(Transform2D):
    """
    Reflection in 2-D.
    """

    def __init__(self, line: Line2D):

        super().__init__()

        self.line = line
        self.direction = line.direction
        self.ortho_reflection = OrthoReflection2D(self.direction)
        self.pt = line.nearest_point_on_line_to([0, 0])

        return

    def apply(self, points):

        points_out = validate_pts(points)

        points_out = points_out - self.pt

        points_out = self.ortho_reflection.apply(points_out)

        points_out = points_out + self.pt

        return points_out

    def followed_by(self, other: Reflection2D):
        line_1 = self.line
        line_2 = other.line
        return Rotation2D.from_lines(line_1, line_2)


    def get_matrix(self, ):

        P = self.pt

        T = Translation2D(P).get_matrix()
        M = self.ortho_reflection.get_matrix()
        T_inv = Translation2D(-1.0 * P).get_matrix()

        return T @ M @ T_inv

    def __repr__(self):
        pt = np.round(self.pt.flatten(), 2)
        direction = np.round(self.line.direction.flatten(), 2)
        return f'Reflection2D(\n {pt},\n {direction}\n)'

class OrthoRotation2D(OrthoTransform2D):

    def __init__(self, angle):

        super().__init__()

        self.angle = wrap_angle_minus_pi_to_pi(angle)

        # Set up a pair of reflections that can be used
        # to execute this rotation.
        half_angle = self.angle / 2.0
        O = [0, 0]

        line_1 = Line2D(O, [1, 0])
        line_2 = Line2D(O, [np.cos(half_angle), np.sin(half_angle)])

        self.ref_1 = Reflection2D(line_1)
        self.ref_2 = Reflection2D(line_2)

        return

    @classmethod
    def from_lines(cls, line_1: Line2D, line_2: Line2D):

        assert np.isclose(line_1.f_x(0), 0.0), 'Line must intersect origin.'
        assert np.isclose(line_2.f_x(0), 0.0), 'Line must intersect origin.'

        dir_1 = line_1.direction
        dir_2 = line_2.direction

        theta_1 = np.arctan2(dir_1[1], dir_1[0])
        theta_2 = np.arctan2(dir_2[1], dir_2[0])

        alpha = 2.0 * (theta_2 - theta_1)
        return OrthoRotation2D(alpha)


    def apply(self, points):
        pts = validate_pts(points)

        # Apply the pair of reflections
        pts = self.ref_1.apply(pts)
        pts = self.ref_2.apply(pts)

        return pts

    def get_matrix(self):
        M = np.eye(3)
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        R = np.asarray([
            [c, -1.0 * s],
            [s, c]
        ])

        M[:2, :2] = R

        return M


class Rotation2D(Transform2D):


    def __init__(self, centre, angle):

        super().__init__()
        self.centre = ensure_vec(centre)
        self.angle = wrap_angle_minus_pi_to_pi(angle)

        self.ortho_rotation = OrthoRotation2D(self.angle)

        return

    @classmethod
    def from_reflections(cls, refl_1: Reflection2D, refl_2: Reflection2D):
        return cls.from_lines(refl_1.line, refl_2.line)


    @classmethod
    def from_lines(cls, line_1: Line2D, line_2: Line2D):
        """
        Create the transformation that results from
        reflecting in line_1 then in line_2.
        """

        if line_1.parallel_to(line_2):
            # Translation.

            perp_dir = line_1.perp
            A = line_1.get_point_on_line()
            perp_line = Line2D(A, perp_dir)

            X = perp_line.intersection(line_2)

            disp_l_to_n = X - A
            dist_l_to_n = np.sqrt(disp_l_to_n.T @ disp_l_to_n)
            u_vec_l_to_n = disp_l_to_n / dist_l_to_n
            displacement = 2.0 * dist_l_to_n * u_vec_l_to_n

            return Translation2D(displacement)

        # Lines are not parallel.
        P = line_1.intersection(line_2)
        alpha = line_1.angle_to(line_2)
        return Rotation2D(P, 2.0 * alpha)


    def apply(self, points):
        pts = validate_pts(points)

        pts = pts - self.centre
        pts = self.ortho_rotation.apply(pts)
        pts = pts + self.centre

        return pts

    def followed_by(self, other):
        A = self.centre
        B = other.centre
        theta = self.angle
        phi = other.angle
        if np.allclose(A, B):
            result = Rotation2D(A, theta + phi)
            return result

        # A and B are distinct.
        m = Line2D(A, B - A)
        m_dir = np.squeeze(m.direction)
        m_ang = np.arctan2(m_dir[1], m_dir[0])

        l_ang = m_ang - theta / 2.0
        l_dir = [np.cos(l_ang), np.sin(l_ang)]
        l = Line2D(A, l_dir)

        n_ang = m_ang + phi / 2.0
        n_dir = [np.cos(n_ang), np.sin(n_ang)]
        n = Line2D(B, n_dir)

        return Rotation2D.from_lines(l, n)

    def __matmul__(self, other):
        """
        Overload @ operator.
        """
        return self.followed_by(other)


    def get_matrix(self):

        T = Translation2D(self.centre).get_matrix()
        R = self.ortho_rotation.get_matrix()
        T_inv = Translation2D(-1.0 * self.centre).get_matrix()

        return T @ R @ T_inv

    def __repr__(self):
        centre = np.round(self.centre.flatten(), 2)
        angle = np.round(self.angle, 2)
        return f'Rotation2D(\n {centre},\n {angle}\n)'


class Translation2D(Transform):

    def __init__(self, v):

        super().__init__()
        self.vec = ensure_vec(v)

    def apply(self, points):
        pts = validate_pts(points)
        return pts + self.vec

    def get_matrix(self):
        T = np.eye(3)
        T[:2, -1] = np.squeeze(self.vec)
        return T

    def __repr__(self):
        v = np.round(self.vec.flatten(), 2)
        return f'Translation2D(\n {v}\n)'


class Translation3D(Transform):

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

    def __repr__(self):
        v = np.round(self.vec.flatten(), 2)
        return f'Translation3D(\n {v}\n)'


class Reflection3D(Transform):

    def __init__(self, plane: Plane3D):
        super().__init__()

        self.plane = plane

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



class OriginRotation3D(Transform):
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

    def followed_by(self, other: OriginRotation3D):

        if vecs_parallel(self.axis, other.axis):
            if np.allclose(self.axis, other.axis):
                return OriginRotation3D(self.axis, self.angle + other.angle)
            else:
                # Axes are opposing each other.
                return OriginRotation3D(self.axis, self.angle - other.angle)

        O = ensure_vec([0, 0, 0])
        P = 10 * self.axis
        Q = 10 * other.axis
        plane_shared = Plane3D.from_points(O, P, Q)
        n_shared = plane_shared.normal

        n_0 = rotate_vector(n_shared, self.axis, -0.5 * self.angle)

        n_1 = rotate_vector(n_shared, other.axis, 0.5 * other.angle)

        plane_0 = Plane3D(n_0, O)
        plane_1 = Plane3D(n_1, O)

        return OriginRotation3D.from_planes(plane_0, plane_1)

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

class Rotation3D(Transform):

    def __init__(self, point, axis_dir, angle):
        """
        A rotation through 'angle' about an axis that
        goes through 'point' with the direction 'axis_dir'.
        """

        super().__init__()
        self.orig_rot = OriginRotation3D(axis_dir, angle)
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

class TransOriginRotation3D(Transform):
    """
    A two-step transformation of the form
    T M : x -> T ( M (x) )
    where
     - M is an origin rotation
     - T is a translation.
    """
    def __init__(self, transvector, axis, angle):
        super().__init__()
        self.origin_rot = OriginRotation3D(axis, angle)
        self.tra = Translation3D(transvector)

        return

    @classmethod
    def from_transforms(cls, originRotation: OriginRotation3D, trans: Translation3D):
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

class TransRotation3D(Transform):
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

