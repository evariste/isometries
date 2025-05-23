from __future__ import annotations

from isom.geom.transform import Transform, Identity, is_identity
from abc import ABC, abstractmethod
import numpy as np
from random import shuffle

from isom.utils.general import (
    ensure_unit_vec, ensure_vec, validate_pts, wrap_angle_minus_pi_to_pi, vecs_parallel, vecs_perpendicular,
    ensure_scalar,
)

from isom.geom.objects import Line2D
from typing import List



class Transform2D(Transform, ABC):

    @abstractmethod
    def inverse(self):
        """
        Get the inverse.
        """


class OrthoTransform2D(Transform2D, ABC):

    @abstractmethod
    def get_reflections(self) -> List[OrthoReflection2D]:
        """
        Return one or two reflections for the orthogonal transformation.
        """


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
    def from_angle(cls, theta):
        direction = [np.cos(theta), np.sin(theta)]
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

    def two_step_form(self):
        M = OrthoReflection2D(self.direction)
        I = Identity(2)
        return [M, I]

    @classmethod
    def from_two_step_form(cls, M, t):
        assert is_identity(t), 'Expect no translation.'

        if is_identity(M):
            return Identity(2)

        assert isinstance(M, OrthoReflection2D), 'Expect first transform to be orthogonal reflection.'
        return M.copy()


    def get_reflections(self):
        M = OrthoReflection2D(self.direction)
        return [M]

    def copy(self):
        return OrthoReflection2D(self.direction)

    def __repr__(self):
        return f'OrthoReflection2D({self.direction.tolist()})'

    def __str__(self):
        d = np.round(self.direction, 2).tolist()
        return f'OrthoReflection2D({d})'


class OrthoRotation2D(OrthoTransform2D):

    def __init__(self, theta):

        super().__init__()

        self.angle = wrap_angle_minus_pi_to_pi(theta)

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
        alpha = wrap_angle_minus_pi_to_pi(alpha)

        if np.isclose(np.abs(alpha), 0) or np.isclose(np.abs(alpha), np.pi):
            return Identity(2)

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

    @classmethod
    def from_two_step_form(cls, M, t):
        assert is_identity(t), 'Expect no translation.'

        if is_identity(M):
            return Identity(2)

        assert isinstance(M, OrthoRotation2D), 'Expect first transform to be orthogonal rotation.'
        return M.copy()

    def get_reflections(self):
        return [self.refl_1, self.refl_2]

    def copy(self):
        return OrthoRotation2D(self.angle)

    def __repr__(self):
        return f'OrthoRotation2D({self.angle})'

    def __str__(self):
        a = np.round(self.angle, 2)
        return f'OrthoRotation2D({a})'


class Translation2D(Transform2D):

    def __init__(self, v):

        super().__init__()
        self.vec = ensure_vec(v)

    def apply(self, points):
        pts = validate_pts(points)
        return pts + self.vec

    def inverse(self):
        return Translation2D(-1.0 * self.vec)

    def get_matrix(self):
        T = np.eye(3)
        T[:2, -1] = np.squeeze(self.vec)
        return T

    def two_step_form(self):
        I = Identity(2)
        t = Translation2D(self.vec)
        return [I, t]

    @classmethod
    def from_two_step_form(cls, M, t):
        assert is_identity(M), 'Expect first transform to be identity.'

        if is_identity(t):
            return Identity(2)

        assert is_translation_2d(t), 'Expect second transform to be a translation.'
        return t.copy()

    def copy(self):
        return Translation2D(self.vec)

    def __repr__(self):
        v = self.vec.flatten().tolist()
        return f'Translation2D({v})'

    def __str__(self):
        v = np.round(self.vec, 2).tolist()
        return f'Translation2D({v})'



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

    def inverse(self):
        return self.copy()

    def two_step_form(self):
        pt2 = self.ortho_reflection.apply(self.pt)
        u = self.pt - pt2

        t = Translation2D(u)
        M = OrthoReflection2D(self.direction)
        return [M, t]


    @classmethod
    def from_two_step_form(cls, M, t):
        """
        M: First transform of two-steps. Orthogonal reflection or identity.
        t: Second step. Translation or identity.

        Make a general reflection object from the above.
        """

        if is_identity(M):
            return t

        assert isinstance(M, OrthoReflection2D), 'Expect first transform to be an orthogonal reflection.'

        if is_identity(t):
            return M

        assert is_translation_2d(t), 'Expect second transform to be a translation.'

        if not vecs_perpendicular(t.vec, M.direction):
            # Some component of translation is parallel to the line
            # of reflection. Return a glide reflection.
            return GlideReflection2D.from_two_step_form(M, t)

        # t is perpendicular to the reflection line.

        # Origin.
        O = ensure_vec([0, 0])

        # Image of O under the transformation.
        P = t.apply(M.apply(O))

        if np.allclose(O, P):
            # Origin is on the plane of reflection.
            # Transform is orthogonal. Should not happen given checking t
            # for identity above. But anyway ...
            return M

        assert vecs_parallel(M.normal, P - O), 'Expect pure reflection.'

        # O and P are distinct and the line OP is perpendicular to the line of reflection.

        # Point on reflection line.
        Q = 0.5 * (O + P)

        # Line of reflection.
        line = Line2D(Q, M.direction)

        return Reflection2D(line)


    def apply(self, points):

        points_out = validate_pts(points)

        points_out = points_out - self.pt

        points_out = self.ortho_reflection.apply(points_out)

        points_out = points_out + self.pt

        return points_out


    def get_matrix(self, ):

        P = self.pt

        T = Translation2D(P).get_matrix()
        M = self.ortho_reflection.get_matrix()
        T_inv = Translation2D(-1.0 * P).get_matrix()

        return T @ M @ T_inv

    def copy(self):
        line = Line2D(self.pt, self.direction)
        return Reflection2D(line)

    def __repr__(self):
        l = self.line.__repr__()
        return f"""Reflection2D(
{l}
)"""

    def __str__(self):
        return f"""Reflection2D(
{self.line}
)"""


class Rotation2D(Transform2D):


    def __init__(self, centre, theta):

        super().__init__()
        self.centre = ensure_vec(centre)
        self.angle = wrap_angle_minus_pi_to_pi(theta)

        self.ortho_rotation = OrthoRotation2D(self.angle)

        return

    def inverse(self):
        return Rotation2D(self.centre, -1.0 * self.angle)

    def two_step_form(self):
        centre_rot = self.ortho_rotation.apply(self.centre)
        u = self.centre - centre_rot

        M = OrthoRotation2D(self.angle)

        t = Translation2D(u)

        return [M, t]

    @classmethod
    def from_two_step_form(cls, M, t):
        """
        M: First transform of two-steps. Orthogonal rotation or identity.
        t: Second step. Translation or identity.

        Make a general rotation object from the above.
        """

        if is_identity(M):
            return t

        assert isinstance(M, OrthoRotation2D), 'Expect first transform to be a rotation.'

        if is_identity(t):
            return M

        assert isinstance(t, Translation2D), 'Expect second transform to be a translation.'

        # Origin.
        O = ensure_vec([0, 0])

        # Image of O under the transformation.
        P = t.apply(M.apply(O))

        if np.allclose(O, P):
            # Origin is fixed the composition of M and t.
            # Axis passes through O, result is orthogonal.
            return M.copy()

        # O and P are distinct.

        # Image of P under the transformation.
        Q = t.apply(M.apply(P))

        if np.allclose(O, Q):
            # Transformation is a half-turn.
            # Centre is half way between O and P
            C = 0.5 * (O + P)
            theta = np.pi
            return Rotation2D(C, theta)

        assert not np.allclose(P, Q), 'This should not happen!'

        A = 0.5 * (O + P)
        x, y = P
        line_dir_A = ensure_vec([-1.0 * y, x])
        line_A = Line2D(A, line_dir_A)

        B = 0.5 * (P + Q)
        x, y = Q - P
        line_dir_B = ensure_vec([-1.0 * y, x])
        line_B = Line2D(B, line_dir_B)

        C = line_A.intersection(line_B)

        theta = line_A.angle_to(line_B)

        return Rotation2D(C, theta)



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

            distance = np.sqrt(np.sum(displacement * displacement))
            if np.isclose(distance, 0):
                return Identity(2)

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


    def get_matrix(self):

        T = Translation2D(self.centre).get_matrix()
        R = self.ortho_rotation.get_matrix()
        T_inv = Translation2D(-1.0 * self.centre).get_matrix()

        return T @ R @ T_inv

    def copy(self):
        return Rotation2D(self.centre, self.angle)

    def __repr__(self):
        return f"""Rotation2D(
{self.centre.tolist()},
{self.angle},
)"""

    def __str__(self):
        c = np.round(self.centre, 2).tolist()
        a = np.round(self.angle, 2)
        return f"""Rotation2D(
{c},
{a},
)"""

class GlideReflection2D(Transform2D):

    def __init__(self, line: Line2D, displacement: float):
        """
        Displacement is in direction of 'direction'.
        Can be positive or negative.
        """
        super().__init__()

        self.displacement = ensure_scalar(displacement)
        self.line = line

        self.reflection = Reflection2D(self.line)
        self.translation = Translation2D(self.displacement * self.reflection.direction)

        return

    def inverse(self):
        return GlideReflection2D(self.line, -1.0 * self.displacement)

    def apply(self, points):
        pts = validate_pts(points)
        pts = self.reflection.apply(pts)
        pts = self.translation.apply(pts)
        return pts

    def get_matrix(self):
        M = self.reflection.get_matrix()
        N = self.translation.get_matrix()
        return N @ M

    def two_step_form(self):
        M, t1 = self.reflection.two_step_form()
        t2 = Translation2D(t1.vec + self.translation.vec)
        return [M, t2]

    @classmethod
    def from_two_step_form(cls, M, t):
        """
        M: First transform of two-steps. Orthogonal reflection or identity.
        t: Second step. Translation or identity.

        Make a glide reflection object from the result of composing
        an orthogonal reflection followed by a translation.
        """
        if is_identity(M):
            return t

        assert isinstance(M, OrthoReflection2D), 'Expect first transform to be an orthogonal reflection.'

        if is_identity(t):
            return M

        assert is_translation_2d(t), 'Expect second transform to be a translation.'

        # Origin.
        O = ensure_vec([0, 0])

        # Image of O under the transformation.
        P = t.apply(M.apply(O))

        if np.allclose(O, P):
            # Origin is on the plane of reflection.
            # Transform is orthogonal. Should not happen given checking t
            # for identity above. But anyway ...
            return M

        if vecs_parallel(M.normal, P - O):
            # The line joining O and P is perpendicular to the line of reflection.
            # We have a 'normal' (non-glide) reflection.
            return Reflection2D.from_two_step_form(M, t)

        disp_vec = ensure_vec(P - O)
        disp_value = M.direction.T @ disp_vec

        # Point on reflection line.
        Q = 0.5 * (O + P)

        # Line of reflection.
        line = Line2D(Q, M.direction)

        return GlideReflection2D(line, disp_value)

    def copy(self):
        return GlideReflection2D(self.line.copy(), self.displacement)

    def __repr__(self):
        line_repr = repr(self.line)
        disp = self.displacement
        return f"""GlideReflection2D(
{line_repr},
{disp},
)
"""

    def __str__(self):
        line_repr = str(self.line)
        disp = np.round(self.displacement, 2)
        return f"""GlideReflection2D(
{line_repr},
{disp},
)
"""


def compose_2d(transf_A: Transform2D, transf_B: Transform2D):
    """
    Compose two 2D transformations.
    Return the 2D transformation that results from applying transf_A
    followed by transf_B.
    """

    M_a, t_a = transf_A.two_step_form()
    M_b, t_b = transf_B.two_step_form()

    # Application sequence (starting from the left:
    # M_a t_a M_b t_b

    # Flip the middle pair.
    M_c, t_c = flip_two_step_form_2D([t_a, M_b])

    # Sequence can now be:
    # M_a M_c t_c t_b

    # We should have M_c == M_b
    assert M_b.matrix_equals(M_c), 'Unexpected change in orthogonal part after flip.'

    # Sequence is:
    # M_a M_b t_c t_b

    # Orthogonal part of result.
    M_out = compose_ortho_2d(M_a, M_b)

    v = ensure_vec([0, 0])
    # Accumulate translation vectors (if they are not identity transforms).
    if is_translation_2d(t_b):
        v += t_b.vec
    if is_translation_2d(t_c):
        v += t_c.vec


    if np.allclose(v, [0, 0]):
        return M_out

    t_out = Translation2D(v)

    return transf_2d_from_two_step(M_out, t_out)

def compose_ortho_2d(t_a: OrthoTransform2D, t_b: OrthoTransform2D):
    """
    Compose a pair of orthogonal 2D transformations.
    """
    refls = []
    if not is_identity(t_a):
        refls += t_a.get_reflections()

    if not is_identity(t_b):
        refls += t_b.get_reflections()

    if len(refls) == 0:
        return Identity(2)

    if len(refls) == 1:
        return refls[0]

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


def transf_2d_from_two_step(M: OrthoTransform2D, t: Translation2D):
    """
    Generate a single tranform object from a two step form.
    """

    if isinstance(M, Identity):
        return t

    if isinstance(t, Identity):
        return M

    if isinstance(M, OrthoReflection2D):
        result = Reflection2D.from_two_step_form(M, t)
    elif isinstance(M, OrthoRotation2D):
        result = Rotation2D.from_two_step_form(M, t)
    else:
        raise Exception('Unexpected type for first transform M')

    return result


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
        # Put t1 at index 0
        return [t1, I]

    if is_identity(t1):
        # Put t0 at index 1
        return [I, t0]

    # Neither t0, nor t1, are the identity.

    if is_ortho_2d(t0):
        # Form should be [M, t]
        assert is_translation_2d(t1), 'Expect second tranform to be a translation.'
        # Translation vector
        p = t1.vec

        # Alias for orthogonal part:
        M = t0
        M_inv = M.inverse()

        q = M_inv.apply(p)

        t_q = Translation2D(q)

        return [t_q, M]

    # First transformation is not orthogonal.
    assert is_translation_2d(t0), 'Expect first tranform to be a translation'
    assert is_ortho_2d(t1), 'Expect second transform to be orthogonal.'

    # Form is [t, M]

    # Translation vector
    p = t0.vec

    # Alias for orthogonal part
    M = t1

    q = M.apply(p)

    t_q = Translation2D(q)

    return [M, t_q]


def is_ortho_2d(transf: Transform2D):
    return isinstance(transf, OrthoTransform2D)

def is_translation_2d(transf: Transform2D):
    return isinstance(transf, Translation2D)

def random_reflection_2d():
    # Random general 2D reflection.
    P = np.random.rand(2) * 10
    v = np.random.rand(2)
    line_1 = Line2D(P, v)
    refl = Reflection2D(line_1)
    return refl

def random_ortho_reflection_2d():
    # Random orthogonal 2D reflection.
    v = np.random.rand(2) - [0.5, 0.5]
    ortho_refl = OrthoReflection2D(v)
    return ortho_refl

def random_rotation_2d():
    # Random general 2D rotation.
    alpha = np.random.rand() * 2.0 * np.pi
    C = np.random.rand(2) * 10
    rot = Rotation2D(C, alpha)
    return rot

def random_ortho_rotation_2d():
    # Random orthogonal 2D rotation.
    alpha = np.random.rand() * 2.0 * np.pi
    ortho_rot = OrthoRotation2D(alpha)
    return ortho_rot


def random_glide_reflection_2d():
    refl = random_reflection_2d()
    line = refl.line
    displacement = np.random.rand() * 20.0 - 10.0
    return GlideReflection2D(line, displacement)

def random_ortho_transformation_2d():
    val = np.random.rand()
    if val < 0.5:
        return random_ortho_reflection_2d()

    return random_ortho_rotation_2d()

def random_translation_2d():
    vec = np.random.rand(2) * 20.0 - 10.0
    t = Translation2D(vec)
    return t

def random_transformation_2d(t_type=None):

    if t_type is None:
        t_types = TRANSF_TYPES_2D.copy()
        # Remove abstract types
        abstract_types = ['Transform2D', 'OrthoTransform2D']
        t_types = list(set(t_types).difference(abstract_types))

        shuffle(t_types)
        t_type = t_types[0]

        return random_transformation_2d(t_type=t_type)

    assert t_type in TRANSF_TYPES_2D, 'Invalid type of transformation specified.'

    k = TRANSF_TYPES_2D.index(t_type)

    rand_func = RAND_TRANS_FUNCS_2D[k]
    return rand_func()


TRANSF_TYPES_2D = [
    'Transform2D'
    'OrthoTransform2D',
    'OrthoReflection2D',
    'OrthoRotation2D',
    'Translation2D',
    'Reflection2D',
    'Rotation2D',
    'GlideReflection2D'
]
RAND_TRANS_FUNCS_2D = [
    random_transformation_2d,
    random_ortho_transformation_2d,
    random_ortho_reflection_2d,
    random_ortho_rotation_2d,
    random_translation_2d,
    random_reflection_2d,
    random_rotation_2d,
    random_glide_reflection_2d
]
