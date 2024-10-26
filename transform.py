"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

from abc import ABC, abstractmethod
import numpy as np
from utilities import *

from objects import *

class Transform(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def apply(self, points):
        """
        Apply to some points and return result.
        """
    @abstractmethod
    def homogeneous_matrix(self):
        """
        Return a homogeneous 4x4 matrix for a transform.
        """


class Translation(Transform):
    def __init__(self, t):
        super().__init__()
        self.vec = ensure_vec_3d(t)

    def apply(self, points):
        pts_out = validate_pts(points)
        pts_out = pts_out + self.vec

        dim, _ = points.shape
        if dim == 2:
            return pts_out[:2]

        return pts_out

    def homogeneous_matrix(self):
        M = np.eye(4)
        # [I v]
        # [0 1]
        M[:3, 3:] = self.vec
        return M


class Rotation(Transform):
    def __init__(self, centre, axis, angle):
        super().__init__()
        self.centre = ensure_vec_3d(centre)
        self.axis = ensure_vec_3d(axis)
        self.angle = angle

    def __repr__(self):
        return f'Rotation({self.centre.flatten()},\n {self.axis.flatten()},\n {self.angle})'

    def apply(self, points):

        pts_out = validate_pts(points)

        R = rotation_matrix_from_axis_and_angle(self.axis, self.angle)

        pts_out = pts_out - self.centre
        pts_out = R @ pts_out
        pts_out = pts_out + self.centre
        # pts_out = self.R @ pts_out + (np.eye(3) - self.R) @ self.centre

        dim, _ = points.shape
        if dim == 2:
            return pts_out[:2]

        return pts_out

    def homogeneous_matrix(self):
        M = np.eye(4)

        # [I t] [R 0] [I -t]
        # [0 1] [0 1] [0  1]
        #
        # [R t] [I -t]
        # [0 1] [0  1]
        #
        # [R  -Rt + t ]
        # [0      1   ]
        R = rotation_matrix_from_axis_and_angle(self.axis, self.angle)
        M[:3, :3] = R
        M[:3, 3:] = -1.0 * R @ self.centre + self.centre

        return M


class Screw(Transform):
    def __init__(self, centre, axis, angle, translate_dist):
        """
        Screw transformation: Combination of rotation and translation along
        a single axis.

        @param centre: Point on rotation axis
        @param axis: of rotation
        @param angle: of rotation
        @param translate_dist: (signed) distance along axis to perform translation.
        """
        super().__init__()

        self.rot = Rotation(centre, axis, angle)

        self.tra = Translation(self.rot.axis * translate_dist)
        return

    def homogeneous_matrix(self):
        M1 = self.rot.homogeneous_matrix()
        M2 =self.tra.homogeneous_matrix()
        return M2 @ M1




    def apply(self, points):
        pts_out = validate_pts(points)
        pts_out = self.rot.apply(pts_out)
        pts_out = self.tra.apply(pts_out)
        return pts_out



def compose_rotatations(rot_A, rot_B):
    """
    Generate a rotation rot_C such that rot_C (x) = rot_B ( rot_A (x) )

    """

    M_A = rot_A.homogeneous_matrix()
    M_B = rot_B.homogeneous_matrix()
    M = M_B @ M_A

    axis = axis_from_rotation_matrix(M[:3, :3])

    v = M[:3, 3:]

    v_along = (axis @ v) * axis.T
    v_perp = v - v_along

    M_rot = M.copy()
    M_rot[:3, 3:] = v_perp

    Z = M_rot - np.eye(4)
    Z[3, :3] = axis

    #
    centre, _, _, _ = np.linalg.lstsq(Z[:, :3], -Z[:, 3], rcond=None)

    cent_hom = np.hstack((centre, 1)).T
    assert np.allclose(M_rot @ cent_hom, cent_hom), 'Error in finding centre.'

    angle = angle_from_rotation_matrix(M[:3, :3])

    rot = Rotation(centre, axis, angle)


    if not np.allclose(rot.homogeneous_matrix(), M_rot):
        # Try inverting the axis
        axis *= -1.0
        rot = Rotation(centre, axis, angle)
        assert np.allclose(rot.homogeneous_matrix(), M_rot), 'Error finding rotation.'

    translate_dist = np.sqrt(np.sum(v_along * v_along))

    if np.isclose(translate_dist, 0):
        # Composition is a rotation.
        return rot

    # Composition is a screw transformation.
    screw = Screw(centre, axis, angle, translate_dist)

    if not np.allclose(screw.homogeneous_matrix(), M):
        # Try inverting the translation
        translate_dist *= -1.0
        screw = Screw(centre, axis, angle, translate_dist)
        assert np.allclose(screw.homogeneous_matrix(), M), 'Error finding screw transform.'

    return screw


class Reflection2D(Transform):

    def __init__(self, line: Line2D):

        super().__init__()

        self.line = line
        self.direction = line.direction
        normal = [-1.0 * line.direction[1], line.direction[0]]
        self.normal = ensure_unit_vec_2d(normal)


    def apply(self, points):

        points_out = validate_pts(points)

        points_out = points_out - self.line.point

        comp1 = self.direction.T @ points_out
        comp1 = self.direction @ comp1
        comp2 = points_out - comp1

        points_out = comp1 - comp2

        points_out = points_out + self.line.point

        return points_out

    def homogeneous_matrix(self,):
        P = self.line.get_point_on_line()
        O_line = Line2D([0, 0], self.direction)
        O_refl = Reflection2D(O_line)

        T_inv = Translation2D(-1.0 * P).homogeneous_matrix()
        T = Translation2D(P).homogeneous_matrix()

        M = np.eye(3)


        M[:2, :2] = O_refl.apply(np.eye(2))

        return T @ M @ T_inv

class Rotation2D(Transform):


    def __init__(self, centre, angle):

        super().__init__()
        self.centre = ensure_vec_2d(centre)
        self.angle = wrap_angle_minus_pi_to_pi(angle)

        # Set up a pair of reflections that can be used
        # to execute this rotation.
        half_angle = self.angle / 2.0

        line_1 = Line2D(self.centre, [1, 0])
        line_2 = Line2D(self.centre, [np.cos(half_angle), np.sin(half_angle)])

        self.ref_1 = Reflection2D(line_1)
        self.ref_2 = Reflection2D(line_2)

        return

    @classmethod
    def from_reflections(cls, refl_1: Reflection2D, refl_2: Reflection2D):
        return cls.from_lines(refl_1.line, refl_2.line)


    @classmethod
    def from_lines(cls, line_1: Line2D, line_2: Line2D):

        if line_1.parallel_to(line_2):

            perp_dir = line_1.perp
            A = line_1.get_point_on_line()
            perp_line = Line2D(A, perp_dir)

            X = perp_line.intersection(line_2)

            disp_l_to_n = X - A
            dist_l_to_n = np.sqrt(disp_l_to_n.T @ disp_l_to_n)
            u_vec_l_to_n = disp_l_to_n / dist_l_to_n
            displacement = 2.0 * dist_l_to_n * u_vec_l_to_n

            return Translation2D(displacement)


        P = line_1.intersection(line_2)
        alpha = line_1.angle_to(line_2)
        return Rotation2D(P, 2.0 * alpha)


    def apply(self, points):
        # Apply the pair of reflections
        pts = self.ref_1.apply(points)
        pts = self.ref_2.apply(pts)

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


    def homogeneous_matrix(self):
        M = np.eye(3)
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        R = np.asarray([
            [c, -1.0 * s],
            [s, c]
        ])
        T_inv = Translation2D(-1.0 * self.centre).homogeneous_matrix()
        T = Translation2D(self.centre).homogeneous_matrix()

        M[:2, :2] = R

        return T @ M @ T_inv


class Translation2D(Transform):

    def __init__(self, v):

        super().__init__()
        self.v = ensure_vec_2d(v)

    def apply(self, points):
        pts = validate_pts(points)
        return pts + self.v

    def homogeneous_matrix(self):
        T = np.eye(3)
        T[:2, -1] = np.squeeze(self.v)
        return T


class Translation3D(Transform):

    def __init__(self, v):

        super().__init__()
        self.v = ensure_vec_3d(v)

    def apply(self, points):
        pts = validate_pts(points)
        return pts + self.v

    def homogeneous_matrix(self):
        T = np.eye(4)
        T[:3, -1] = np.squeeze(self.v)
        return T



class Reflection3D(Transform):

    def __init__(self, plane: Plane3D):
        super().__init__()

        self.plane = plane

        return


    def apply(self, points):
        X = self.plane.pt
        n = self.plane.normal

        disps = points - X

        coeff_norm = n.T @ disps

        comp_norm = n @ coeff_norm

        ret = disps - 2 * comp_norm + X

        return ret



    def homogeneous_matrix(self):

        pt = self.plane.pt
        n = self.plane.normal

        T_inv = Translation3D(-1.0 * pt).homogeneous_matrix()
        T = Translation3D(pt).homogeneous_matrix()

        M = np.eye(4)

        I3 = np.eye(3)
        coeff = n.T @ I3
        comp_norm = n @ coeff
        M[:3, :3] = I3 - 2 * comp_norm

        H = T @ M @ T_inv
        return H


