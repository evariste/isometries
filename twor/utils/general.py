"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

import numpy as np


def cross_product(x, y):
    u = ensure_vec_3d(x, transpose=True)
    v = ensure_vec_3d(y, transpose=True)
    return np.transpose(np.cross(u, v))

def vecs_perpendicular(u, v):
    uu = ensure_unit_vec(u)
    vv = ensure_unit_vec(v)
    return np.isclose(np.abs(np.sum(uu * vv)), 0.0)

def vecs_parallel(u, v):
    uu = ensure_unit_vec(u)
    vv = ensure_unit_vec(v)
    return np.isclose(np.abs(np.abs(np.sum(uu * vv))), 1.0)

def make_pts_homogenous(points_in):
    pts = validate_pts(points_in)
    spatial_dim, n_pts = pts.shape
    row_ones = np.ones((1, n_pts))
    return np.vstack([pts, row_ones])

def apply_hom_matrix_to_points(M, pts):
    """
    Apply a homogeneous matrix to a point set.
    pts: 3xN or 2xN
    M: 4x4 or 3x3
    return: transformed points.
    """
    pts_out = M @ make_pts_homogenous(pts)
    return pts_out[:-1]

def rotate_vector(vec, axis, angle):
    """
    Rotate the vector about the given axis by the given angle.
    """
    u = ensure_unit_vec_3d(axis)
    v = ensure_vec_3d(vec)

    if vecs_parallel(u, v):
        return v

    theta = wrap_angle_minus_pi_to_pi(angle)

    comp_u_v = u.T @ v * u

    perp_u_v = v - comp_u_v

    d_perp_u_v = np.sqrt(perp_u_v.T @ perp_u_v)

    w = cross_product(u, v)

    rot_part_1 = np.cos(theta) * d_perp_u_v * perp_u_v

    rot_part_2 = np.sin(theta) * d_perp_u_v * w

    return comp_u_v + rot_part_1 + rot_part_2



def validate_pts(points_in):
    if isinstance(points_in, list):
        points = np.asarray(points_in, dtype=np.float64)
    else:
        assert isinstance(points_in, np.ndarray), 'Invalid type for points.'
        points = points_in.copy().astype(np.float64)

    assert points.ndim == 2, 'Point data array must be 2D'

    r, c = points.shape

    msg = 'Must have at least one point.'
    # Expect rows to correspond to spatial dimension in first instance
    if r == 2:
        assert c > 0, msg
        return points

    if r == 3:
        assert c > 0, msg
        return points

    # If we are here, then the columns might be the spatial dimension.
    if c == 2:
        assert r > 0, msg
        return np.transpose(points)

    if c == 3:
        assert r > 0, msg
        return np.transpose(points)

    raise Exception('Invalid dimensions for points. Cannot validate.')



def ensure_vec(vec):
    sz = np.size(vec)
    if sz == 2:
        return ensure_vec_2d(vec)
    elif sz == 3:
        return ensure_vec_3d(vec)
    else:
        raise Exception('Invalid dimension for vector')

def ensure_unit_vec(vec):
    sz = np.size(vec)
    if sz == 2:
        return ensure_unit_vec_2d(vec)
    elif sz == 3:
        return ensure_unit_vec_3d(vec)
    else:
        raise Exception('Invalid dimension for vector')


def ensure_vec_3d(vec: list, transpose=False):
    assert np.size(vec) == 3, 'Invalid dimension for 3D vector.'

    v = vec.copy()

    if isinstance(v, list):
        v = np.asarray(v)
    v = np.reshape(v, (3, 1)).astype(np.float64)
    if transpose:
        v = np.transpose(v)
    return v


def ensure_unit_vec_3d(vec):
    v = ensure_vec_3d(vec)
    d = np.sqrt(np.sum(v * v))
    if np.abs(d) > 0:
        v = v / d
    else:
        raise Exception('Cannot return unit from zero vector.')
    return v


def ensure_vec_2d(vec: list, transpose=False):
    assert np.size(vec) == 2, 'Invalid dimension for 2D vector.'

    v = vec.copy()

    if isinstance(v, list):
        v = np.asarray(v)

    v = np.reshape(v, (2, 1)).astype(np.float64)

    if transpose:
        v = np.transpose(v)

    return v


def ensure_unit_vec_2d(vec):
    v = ensure_vec_2d(vec)

    d = np.sqrt(np.sum(v * v))
    if np.abs(d) > 0:
        v = v / d
    else:
        raise Exception('Cannot return unit from zero vector.')
    return v


def wrap_angle_minus_pi_to_pi(alpha):
    return np.arctan2(np.sin(alpha), np.cos(alpha))


def angle_to_2d_line_direction(alpha):
    """
    Return an angle in the range -pi/2 to pi/2 to indicate a line direction in 2D.

    We identify angles in quadrants to the left of the y axis with their offset
    by half a turn to get an angle in the quadrants on the right of the y axis.
    """

    direction = wrap_angle_minus_pi_to_pi(alpha)
    if direction > np.pi / 2.0:
        direction -= np.pi
    if direction < -1.0 * np.pi / 2.0:
        direction += np.pi

    return direction


def nearest_point_on_line(line, point):
    pt = ensure_vec_3d(point)
    disp = pt - line.pt

    comp_par = (disp.T @ line.direction) * line.direction
    comp_perp = disp - comp_par
    nearest_pt = pt - comp_perp

    return nearest_pt


def dist_between_pts(pt0, pt1):
    v0 = ensure_vec_3d(pt0)
    v1 = ensure_vec_3d(pt1)

    disp = v0 - v1
    d = np.sqrt(np.sum(disp * disp))
    return d


def angle_from_three_points(vertex, first, second):
    v0 = ensure_vec_3d(vertex)
    v1 = ensure_vec_3d(first)
    v2 = ensure_vec_3d(second)

    vec1 = v1 - v0
    vec2 = v2 - v0

    return angle_between_vectors(vec1, vec2)


def angle_between_vectors(v_first, v_second):
    v0 = ensure_vec_3d(v_first)
    v1 = ensure_vec_3d(v_second)

    norm0 = np.sqrt(np.sum(v0 * v0))
    norm1 = np.sqrt(np.sum(v1 * v1))

    assert norm0 > 0, 'Zero vector passed to angle function.'
    assert norm1 > 0, 'Zero vector passed to angle function.'

    cosine = np.sum(v0 * v1) / norm0 / norm1

    angle = np.arccos(cosine)

    return angle

    # RA, tA = rot_A.R, rot_A.centre
    # RB, tB = rot_B.R, rot_B.centre
    #
    # # Rotation matrix.
    # RC = RB @ RA
    #
    # angleC = angle_from_rotation_matrix(RC)
    # axisC = axis_from_rotation_matrix(RC)
    #
    # # Check that angle and axis orientation are consistent with rotation matrix.
    # test_RC = rotation_matrix_from_axis_and_angle(axisC, angleC)
    # if not np.allclose(RC, test_RC):
    #     axisC = -1.0 * axisC
    #     test_RC = rotation_matrix_from_axis_and_angle(axisC, angleC)
    #     assert np.allclose(RC, test_RC)
    #
    # I = np.eye(3)
    #
    # # If fA(x) = RA ( x - tA ) + tA and similar for fB(x)
    # # Let fC(x) = fB ( fA (x))
    # # We want to find tC where fC(x) = RC ( x - tC ) + tC
    # # RC already found earlier.
    # # Expanding for fB ( fA (x)) and solving for tC leads to the below stuff.
    # M = I - RC
    # b = RB @ (I - RA) @ tA + (I - RB) @ tB
    # # Solve Mx=b to get tC, M is a rank 2 matrix. It represents a transformation
    # # that takes each point to the displacement vector to its rotated version.
    # # All points along any line parallel to the rotation axis map to the same
    # # point, so it is a projection. Therefore we use least squares to solve the
    # # under-determined system.
    # tC, residuals, rank, sing_vals = np.linalg.lstsq(M, b, rcond=None)
    #
    # return Rotation(tC.flatten(), axisC, angleC)


def get_normal_vector(v):
    """
    Return an arbitrary normal vector to v
    """
    assert v.size == 3, 'Expect 3D vector.'

    x, y, z = v.flatten().tolist()

    if np.abs(z) > 0:
        a, b = 1, 0
        c = -1.0 * x / z
    else:
        # z == 0
        if np.abs(y) > 0:
            a, c = 1, 0
            b = -1.0 * x / y
        else:
            # y == z == 0
            assert np.abs(x) > 0
            a, b, c = 0, 1, 0

    mag = np.sqrt(a * a + b * b + c * c)
    w = np.asarray([a, b, c]) / mag
    w = np.reshape(w, (3, 1))
    return w


def axis_from_rotation_matrix(R: np.ndarray):
    evals, evecs = np.linalg.eig(R)
    idxs = np.isclose(evals, 1.0)
    assert np.any(idxs), 'Expect at least one eigenvalue to be equal to 1'
    if np.allclose(evals, 1.0):
        # Identity matrix.
        return np.asarray([0.0, 0.0, 1.0])

    k = np.argwhere(idxs)[0]
    axis = evecs[:, k]
    assert np.allclose(np.imag(axis), 0.0), 'Expect real axis.'

    axis = np.real(axis)
    axis = axis.T
    return axis


def angle_from_rotation_matrix(R):
    return np.arccos(0.5 * (np.trace(R) - 1.0))


def rotation_matrix_from_axis_and_angle(u, theta):
    """
    Return a 3x3 rotation matrix for axis u and angle theta.
    Uses Rodrigues' formula.

    Rotation is in anti-clockwise sense looking backwards along the axis of
    rotation in a right-handed frame.

    Rotation is in anti-clockwise sense looking forwards along the axis of
    rotation in a left-handed frame.

    :param u: axis vector
    :param theta: angle
    :return: rotation matrix.
    """

    if isinstance(u, list):
        u = np.asarray(u)

    if not np.size(u) == 3:
        raise Exception('Vector must be 3D')

    # Ensure u is a column vector.
    u = u.reshape((3, 1)).astype(float)

    mag = np.sqrt(np.sum(u * u))
    assert mag > 0, 'Vector must have non-zero magnitude.'

    # Normalize.
    u = u / mag

    # Outer product
    uut = u.dot(u.T)

    I = np.eye(3)

    c, s = np.cos(theta), np.sin(theta)

    skew_sym = skew_symmetric_matrix(u)

    R = uut + c * (I - uut) + s * skew_sym

    return R


############################################################################

def skew_symmetric_matrix(v):
    """
    Return the skew symmetric matrix form of the vector v.

        v = (v1 v2 v3)^T

            [  0  -v2  v1 ]
       s =  [  v2  0  -v0 ]
            [ -v1  v0  0  ]

    :param v:
    :return:
    """

    if not v.flatten().shape == (3,):
        raise Exception('Vector must be 3D')

    s = np.zeros((3, 3))

    s[0, 2] = v[1]
    s[1, 0] = v[2]
    s[2, 1] = v[0]

    return s + -1 * s.T


############################################################################


def matrix2params_affine_3D(matrix):
    """
    Obtain the parameters of translation, rotation, shearing and scaling that correspond to
    a decomposition of the given matrix as described in  params2matrix_affine_IRTK_3D above.
    Assumes matrix contains no perspective transformation component.

    See:
    Spencer W. Thomas, 1991. Decomposing a matrix into simple transformations.
    In Graphics Gems II (pp. 320-323). Morgan Kaufmann.

    :param matrix:
    :return: parameters in order  tx ty tz rx ry rz sx sy sz sxy syz sxz
    """

    m = np.copy(matrix)

    assert m.shape == (4, 4)

    assert np.abs(1.0 - m[-1, -1]) < 0.000001

    det = np.linalg.det(m)

    assert det > 0.000001

    if det < 0.001:
        print('Determinant suspiciously small.')

    if det > 1000:
        print('Determinant suspiciously large.')

    tx, ty, tz = m[:3, -1]

    c0 = m[:3, 0]
    c1 = m[:3, 1]
    c2 = m[:3, 2]

    sx = np.linalg.norm(c0)
    c0 /= sx

    tansxy = c0.dot(c1)
    c1 -= tansxy * c0

    sy = np.linalg.norm(c1)
    c1 /= sy

    tansxy /= sy

    tansxz = c0.dot(c2)
    c2 -= tansxz * c0

    tansyz = c1.dot(c2)
    c2 -= tansyz * c1

    sz = np.linalg.norm(c2)
    c2 /= sz

    tansxz /= sz
    tansyz /= sz

    cross_c1_c2 = cross_product(c1, c2)

    if c0.dot(cross_c1_c2) < 0:
        sx *= -1
        sy *= -1
        sz *= -1
        c0 *= -1
        c1 *= -1
        c2 *= -1

    ry = np.arcsin(-1 * c2[0])

    if np.abs(np.cos(ry)) > 0:
        rx = np.arctan2(c2[1], c2[2])
        rz = np.arctan2(c1[0], c0[0])
    else:
        rx = np.arctan2(-1 * c2[0] * c0[2], -1 * c2[0] * c0[2])
        rz = 0

    sxy = np.arctan(tansxy)
    syz = np.arctan(tansyz)
    sxz = np.arctan(tansxz)

    return np.asarray([tx, ty, tz,
                       rx, ry, rz,
                       sx, sy, sz,
                       sxy, syz, sxz])



def random_rotation_3D():
    R = random_rotation_matrix_3D()
    axis = axis_from_rotation_matrix(R)
    angle = angle_from_rotation_matrix(R)
    return ensure_vec_3d(axis), angle

def random_rotation_matrix_3D():
    """
    Julie C Mitchell, Sampling Rotation Groups by Successive Orthogonal
    Images, SIAM J Sci comput. 30(1), 2008, pp 525-547

    Generate R = [u0 u1 u2], where the columns are the images of the unit axis
    vectors under the rotation.

    :return:
    """

    # u2 uniformly sampled from a sphere (see
    # http://mathworld.wolfram.com/SpherePointPicking.html

    phi = 2 * np.pi * np.random.rand()
    theta   = np.arccos(2 * np.random.rand() - 1)

    u2 = np.asarray([np.cos(phi) * np.sin(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(theta)])

    u2 = np.atleast_2d(u2)
    if u2.shape[0] == 1:
        u2 = u2.T

    # Sample u1 uniformly from the circle that is the intersection of the unit
    # sphere with the plane through O and orthogonal to u2

    eps = 0.000001

    # Find a point w in the xy plane that is also in the plane orthogonal to u2
    # and is one unit from the origin.

    u2_0, u2_1 = u2.flat[:2]

    if np.abs(u2_0) < eps:
        w = np.asarray([0, 1, 0])
    elif np.abs(u2_1) < eps:
        w = np.asarray([1, 0, 0])
    else:
        w = np.asarray([u2_1, -u2_0, 0])
        w = w / np.sqrt(np.sum(w * w))

    w = np.atleast_2d(w)
    if w.shape[0] == 1:
        w = w.T

    # Rotate w by a random angle around the axis u2
    theta_w = 2 * np.pi * np.random.rand()
    Rw = rotation_matrix_from_axis_and_angle(u2, theta_w)

    u1 = Rw.dot(w)

    # Disable warnings.
    cross1 = lambda x, y: np.cross(x, y)
    u0 = cross1(u1.T, u2.T).T

    R = np.hstack([u0, u1, u2])

    return R

