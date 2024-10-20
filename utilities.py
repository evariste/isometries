"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

import numpy as np

def ensure_pts_3d(points):
    assert points.ndim == 2, 'Points array must be 2D'

    spatial_dim, n_pts = points.shape

    assert n_pts > 0, 'Must have at least one point.'

    if spatial_dim == 2:
        pts_out = np.vstack([points, np.zeros((1, n_pts))])
    elif spatial_dim == 3:
        pts_out = points.copy()
    else:
        raise Exception('Invalid point dimension.')

    return pts_out.astype(np.float64)

def ensure_pts_2d(points):
    assert points.ndim == 2, 'Points array must be 2D'

    spatial_dim, n_pts = points.shape

    assert spatial_dim == 2, 'Must be 2D points.'

    assert n_pts > 0, 'Must have at least one point.'

    return points.copy().astype(np.float64)




def ensure_vec_3d(vec: list):
    if np.size(vec) == 2:
        tt = list(vec) + [0]
    else:
        tt = list(vec)

    assert np.size(tt) == 3
    if isinstance(tt, list):
        tt = np.asarray(tt)
    return np.reshape(tt, (3,1)).astype(np.float64)

def ensure_unit_vec_3d(vec):

    v = ensure_vec_3d(vec)
    d = np.sqrt(np.sum(v * v))
    if np.abs(d) > 0:
        v = v / d
    else:
        raise Exception('Cannot return unit from zero vector.')
    return v

def ensure_vec_2d(vec: list):
    assert np.size(vec) == 2, 'Invalid size for 2D vector.'

    v = vec.copy()

    if isinstance(v, list):
        v = np.asarray(v)

    return np.reshape(v, (2,1)).astype(np.float64)

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
    w = np.reshape(w, (3,1))
    return w




def axis_from_rotation_matrix(R: np.ndarray):
    evals, evecs = np.linalg.eig(R)
    idxs = np.isclose(evals, 1.0)
    assert np.any(idxs), 'Expect at least one eigenvalue to be equal to 1'
    if np.allclose(evals, 1.0):
        # Identity matrix.
        return np.asarray([0.0,0.0,1.0])

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
    u = u.reshape((3,1)).astype(float)

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

    s = np.zeros((3,3))

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

    assert m.shape == (4,4)

    assert np.abs(1.0 - m[-1,-1]) < 0.000001

    det = np.linalg.det(m)

    assert det > 0.000001

    if det < 0.001:
        print('Determinant suspiciously small.')

    if det > 1000:
        print('Determinant suspiciously large.')

    tx, ty, tz = m[:3, -1]

    c0 = m[:3,0]
    c1 = m[:3,1]
    c2 = m[:3,2]

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
        rx = np.arctan2(-1*c2[0]*c0[2], -1*c2[0]*c0[2])
        rz = 0

    sxy = np.arctan(tansxy)
    syz = np.arctan(tansyz)
    sxz = np.arctan(tansxz)

    return np.asarray([tx, ty, tz,
                       rx, ry, rz,
                       sx, sy, sz,
                       sxy, syz, sxz])


def cross_product(v1, v2):
    # Wrapper prevents lint warnings if np.cross is placed directly in blocks of code.
    return np.cross(v1, v2)