from __future__ import annotations

from twor.geom.transform import Transform, Identity
from abc import ABC, abstractmethod
import numpy as np

from twor.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi,
)

from twor.geom.objects import Line2D
from typing import List


class Transform2D(Transform, ABC):
    pass

class OrthoTransform2D(Transform2D, ABC):

    @abstractmethod
    def get_reflections(self) -> List[OrthoReflection2D]:
        """
        Return one or two reflections for the orthogonal transformation.
        """

def compose_ortho_2d(t_a: OrthoTransform2D, t_b: OrthoTransform2D):
    """
    Compose a pair of orthogonal 2D transformations.
    """

    refls = t_a.get_reflections() + t_b.get_reflections()

    assert 1 < len(refls) < 5, 'Unexpected number of reflections'

    lines = [r.line for r in refls]
    l0 = lines[0]
    l1 = lines[1]

    if len(refls) == 2:
        if l0.parallel_to(l1):
            return Identity(2)

        return OrthoRotation2D.from_lines(l0, l1)

    # The sequence has three or four reflections.
    l2 = lines[2]

    angle_12 = l1.angle_to(l2)
    rot = OrthoRotation2D(angle_12)

    # Rotate lines l0 and l1 so that l1 coincides with l2.
    # Then we can drop l1 and l2

    l0_rot = l0.apply_transformation(rot)

    if len(refls) == 3:
        return OrthoReflection2D(l0_rot.direction)

    # Four reflections in original list.
    # Second and third have been cancelled.
    l3 = lines[3]
    return OrthoRotation2D.from_lines(l0_rot, l3)


def ortho2D_to_reflections(ortho2d_transf: OrthoTransform2D):
    """
    Break up a generic orthogonal 2D transform into a sequence of
    reflections.
    """

    # x = (1, 0)   y = (0, 1)
    xy = validate_pts([[1, 0], [0, 1]])
    x = xy[:, [0]]
    y = xy[:, [1]]

    uv = ortho2d_transf.apply(xy)
    u = uv[:, [0]]
    v = uv[:, [1]]

    # x -> u and y -> v

    if np.allclose(u, x):
        R1 = Identity(2)
    else:
        # Reflect in a line that goes half way between x and u
        l = Line2D([0, 0], x + u)
        R1 = Reflection2D(l)

    if np.allclose(R1.apply(y), v):
        return [R1]

    l = Line2D([0, 0], u)
    R2 = Reflection2D(l)

    assert np.allclose(R2.apply(R1.apply(x)), u), 'Error in reflections'
    assert np.allclose(R2.apply(R1.apply(y)), v), 'Error in reflections'

    return [R1, R2]

def flip_two_step_form_2D(two_step_transf):
    """
    Given a two step form for a 2D transformation consisting of
    a translation and orthogonal 2D transform, in some order,
    return the equivalent two step form with order reversed.

    Let M be an orthogonal transformation and t be a translation

    The input two-step form either applies M first or t first.

    Return the two-step form that reverses the order of application.

    We represent a two step form by a length 2 list where the index of
    the elements determines the order of application. E.g., [M, t]
    means the orthogonal transformation is applied first.
    """

    assert len(two_step_transf) == 2, 'Unexpected length of two-step form.'

    t0 = two_step_transf[0]
    t1 = two_step_transf[1]
    I = Identity(2)

    if is_identity(t0):
        return [t1, I]

    if is_identity(t1):
        return [t1, I]

    # Neither t0, nor t1, are the identity.

    if is_ortho2d(t0):
        # Form should be [M, t]
        assert is_translation2d(t1), 'Expect second tranform to be a translation.'
        # Translation vector
        p = t1.vec

        # Alias for orthogonal part:
        M = t0
        M_inv = M.inverse()

        q = M_inv.apply(p)

        t_q = Translation2D(q)

        return [t_q, M]

    # First transformation is not orthogonal.
    assert is_translation2d(t0), 'Expect first tranform to be a translation'
    assert is_ortho2d(t1), 'Expect second transform to be orthogonal.'

    # Form is [t, M]

    # Translation vector
    p = t0.vec

    # Alias for orthogonal part
    M = t1

    q = M.apply(p)

    t_q = Translation2D(q)

    return [M, t_q]


def is_identity(transf: Transform2D):
    return isinstance(transf, Identity)

def is_ortho2d(transf: Transform2D):
    return isinstance(transf, OrthoTransform2D)

def is_translation2d(transf: Transform2D):
    return isinstance(transf, Translation2D)

def random_reflection2d():
    # Random general 2D reflection.
    P = np.random.rand(2) * 10
    v = np.random.rand(2)
    line_1 = Line2D(P, v)
    refl = Reflection2D(line_1)
    return refl

def random_rotation2d():
    # Random general 2D rotation.
    alpha = np.random.rand() * 2.0 * np.pi
    C = np.random.rand(2) * 10
    rot = Rotation2D(C, alpha)
    return rot

def random_ortho_rotation2d():
    # Random orthogonal 2D rotation.
    alpha = np.random.rand() * 2.0 * np.pi
    # Random orthogonal rotation
    ortho_rot = OrthoRotation2D(alpha)
    return ortho_rot

def random_ortho_reflection2d():
    # Random orthogonal 2D reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl = OrthoReflection2D(v)
    return ortho_refl


class OrthoReflection2D(OrthoTransform2D):
    """
    Orthogonal (linear) reflection in 2-D.
    """
    def __init__(self, direction):

        super().__init__()
        self.direction = ensure_unit_vec(direction)
        normal = [-1.0 * self.direction[1], self.direction[0]]
        self.normal = ensure_unit_vec(normal)

        self.line = Line2D([0, 0], self.direction)

        return

    @classmethod
    def from_angle(cls, angle):
        direction = [np.cos(angle), np.sin(angle)]
        return OrthoReflection2D(direction)

    def inverse(self):
        # Same as self.
        return OrthoReflection2D(self.direction)

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

    def two_step_form(self):
        M = OrthoReflection2D(self.direction)
        I = Identity(2)
        return [M, I]

    def get_reflections(self):
        M = OrthoReflection2D(self.direction)
        return [M]

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

    def two_step_form(self):
        pt2 = self.ortho_reflection.apply(self.pt)
        u = self.pt - pt2

        t = Translation2D(u)
        M = OrthoReflection2D(self.direction)
        return [M, t]

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

        # Set up an initial pair of reflections that can be used
        # to execute this rotation.
        half_angle = self.angle / 2.0

        self.refl_1 = OrthoReflection2D([1, 0])
        self.refl_2 = OrthoReflection2D.from_angle(half_angle)

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

    @classmethod
    def from_reflections(cls, refl_1: Reflection2D, refl_2: Reflection2D):
        return OrthoRotation2D.from_lines(refl_1.line, refl_2.line)

    def inverse(self):
        refls = self.get_reflections()
        lines = [r.line for r in refls]
        l0, l1 = lines
        return self.from_lines(l1, l0)


    def apply(self, points):
        pts = validate_pts(points)

        # Apply the pair of reflections
        pts = self.refl_1.apply(pts)
        pts = self.refl_2.apply(pts)

        return pts

    def get_matrix(self):
        M = np.eye(3)
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        R = np.asarray([
            [c, -1.0 * s],
            [s, c]
        ])

        M[:2, :2] = R.squeeze()

        return M

    def two_step_form(self):
        M = OrthoRotation2D(self.angle)
        I = Identity(2)
        return [M, I]

    def get_reflections(self):
        return [self.refl_1, self.refl_2]


class Rotation2D(Transform2D):


    def __init__(self, centre, angle):

        super().__init__()
        self.centre = ensure_vec(centre)
        self.angle = wrap_angle_minus_pi_to_pi(angle)

        self.ortho_rotation = OrthoRotation2D(self.angle)

        return

    def two_step_form(self):
        centre_rot = self.ortho_rotation.apply(self.centre)
        u = self.centre - centre_rot

        M = OrthoRotation2D(self.angle)

        t = Translation2D(u)

        return [M, t]


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

    def two_step_form(self):
        I = Identity(2)
        t = Translation2D(self.vec)
        return [I, t]

