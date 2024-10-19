"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

import numpy as np
from abc import ABC, abstractmethod
from utilities import (
    ensure_vec_3d, ensure_pts_3d,
    ensure_unit_vec_2d, ensure_pts_2d,
    rotation_matrix_from_axis_and_angle,
    angle_from_rotation_matrix,
    axis_from_rotation_matrix
)

from objects import Line2D

class Transform(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def apply(self, points, t=1.0):
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

    def apply(self, points, t=1.0):
        pts_out = ensure_pts_3d(points)
        pts_out = pts_out + t * self.vec

        dim, _ = points.shape
        if dim == 2:
            return pts_out[:2]

        return pts_out

    def homogeneous_matrix(self, t=1.0):
        M = np.eye(4)
        # [I v]
        # [0 1]
        M[:3, 3:] = t * self.vec
        return M


class Rotation(Transform):
    def __init__(self, centre, axis, angle):
        super().__init__()
        self.centre = ensure_vec_3d(centre)
        self.axis = ensure_vec_3d(axis)
        self.angle = angle

    def __repr__(self):
        return f'Rotation({self.centre.flatten()},\n {self.axis.flatten()},\n {self.angle})'

    def apply(self, points, t=1.0):

        pts_out = ensure_pts_3d(points)

        R = rotation_matrix_from_axis_and_angle(self.axis, t * self.angle)

        pts_out = pts_out - self.centre
        pts_out = R @ pts_out
        pts_out = pts_out + self.centre
        # pts_out = self.R @ pts_out + (np.eye(3) - self.R) @ self.centre

        dim, _ = points.shape
        if dim == 2:
            return pts_out[:2]

        return pts_out

    def homogeneous_matrix(self, t=1.0):
        M = np.eye(4)

        # [I t] [R 0] [I -t]
        # [0 1] [0 1] [0  1]
        #
        # [R t] [I -t]
        # [0 1] [0  1]
        #
        # [R  -Rt + t ]
        # [0      1   ]
        R = rotation_matrix_from_axis_and_angle(self.axis, t * self.angle)
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

    def homogeneous_matrix(self, t=1.0):
        M1 = self.rot.homogeneous_matrix(t=t)
        M2 =self.tra.homogeneous_matrix(t=t)
        return M2 @ M1




    def apply(self, points, t=1.0):
        pts_out = ensure_pts_3d(points)
        pts_out = self.rot.apply(pts_out, t=t)
        pts_out = self.tra.apply(pts_out, t=t)
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


class Reflection2D:

    def __init__(self, line: Line2D):

        self.line = line
        self.direction = line.direction
        normal = [-1.0 * line.direction[1], line.direction[0]]
        self.normal = ensure_unit_vec_2d(normal)


    def apply(self, points):

        points_out = ensure_pts_2d(points)

        points_out = points_out - self.line.point

        comp1 = self.direction.T @ points_out
        comp1 = self.direction @ comp1
        comp2 = points_out - comp1

        points_out = comp1 - comp2

        points_out = points_out + self.line.point

        return points_out



