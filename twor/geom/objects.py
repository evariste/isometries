from __future__ import annotations


"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

import numpy as np
from twor.utils.general import (
    ensure_vec, ensure_unit_vec, vecs_parallel, vecs_perpendicular, cross_product, wrap_angle_minus_pi_to_pi,
    rotation_matrix_from_axis_and_angle
)
from twor.utils.vtk import run_triangle_filter
from twor.io.vtk import make_vtk_polydata, polydata_save
from matplotlib.patches import Polygon



class Line3D(object):

    def __init__(self, pt, direction):

        self.pt = ensure_vec(pt)
        self.direction = ensure_unit_vec(direction)

        return

    def __call__(self, mu):
        return self.pt + mu * self.direction

    def nearest(self, point):

        v = ensure_vec(point)

        pt_to_v = v - self.pt

        coeff_along = np.sum(pt_to_v * self.direction)

        nearest_pt = self.point_from_parameter(coeff_along)

        return nearest_pt

    def point_from_parameter(self, mu):
        return self.pt + mu * self.direction

    def contains_point(self, X):
        V = ensure_vec(X)

        if np.allclose(V, self.pt):
            return True

        u = V - self.pt
        return vecs_parallel(u, self.direction)

    def set_start_point(self, P):
        self.pt = ensure_vec(P)

    def set_direction(self, v):
        self.direction = ensure_unit_vec(v)


class Plane3D:
    def __init__(self, normal, pt):

        self.normal = ensure_unit_vec(normal)

        p = ensure_vec(pt)

        self.const = np.sum(self.normal * p)

        self.pt = self.const * self.normal

        return

    def parallel_to(self, other: Plane3D):
        return vecs_parallel(self.normal, other.normal)

    @classmethod
    def from_points(cls, O, P, Q):
        o = ensure_vec(O)
        p = ensure_vec(P)
        q = ensure_vec(Q)

        OP = p - o
        OQ = q - o

        if vecs_parallel(OP, OQ):
            raise Exception('Collinear points.')

        n = ensure_unit_vec(cross_product(OP, OQ))

        return cls(n, o)



    def intersection(self, other):

        if isinstance(other, Plane3D):
            return self.intersection_with_plane(other)
        elif isinstance(other, Line3D):
            return self.intersection_with_line(other)
        else:
            raise Exception('Invalid type for intersecting object.')


    def intersection_with_line(self, l: Line3D):
        if vecs_perpendicular(self.normal, l.direction):
            msg = 'Line either lies in plane or is parallel and separate.'
            raise Exception(msg)

        c = self.const
        n = self.normal
        p = l.pt
        u = l.direction

        u_dot_n = np.sum(u * n)
        n_dot_p = np.sum(n * p)
        mu = (c - n_dot_p) / u_dot_n

        pt_intersection = l.point_from_parameter(mu)

        raise pt_intersection

    def intersection_with_plane(self, other: Plane3D):

        if self.parallel_to(other):
            raise Exception('Planes are parallel.')

        direction = cross_product(self.normal, other.normal)

        M = np.zeros((2,3))
        M[0, :] = self.normal.flatten()
        M[1, :] = other.normal.flatten()

        b = np.zeros((2, 1))
        b[0] = self.const
        b[1] = other.const

        pt_inter, residuals, rank, sing_vals = np.linalg.lstsq(M, b, rcond=None)

        intersection = Line3D(pt_inter, direction)

        return intersection




class Glyph(object):
    def __init__(self):
        pass

class Glyph3D(object):
    labelled_points = {
        'A': [0,0,0],
        'B': [2,0,0],
        'C': [1.5,1,0],
        'D': [0,1,0],
        'E': [2,0,1],
        'F': [1.5,1,1],
        'G': [1,1,1],
        'H': [1,1,2],
        'I': [0,1,2],
        'J': [1,0,1],
        'K': [0,0,3],
        'L': [1,0,3],
    }
    labelled_polys = [
        'BEFC',
        'JGFE',
        'CFGHID',
        'KLJEBA',
        'JLHG',
        'DIKA',
        'HLKI',
        'ABCD'
    ]
    def __init__(self, points=None, cells=None):
        if points is None:
            self.points = self.get_default_points()
        else:
            self.points = points

        if cells is None:
            self.cells = self.get_default_cells()
        else:
            self.cells = cells

    def get_default_points(self):
        ordered_point_labels = sorted(list(self.labelled_points.keys()))
        pts = []
        for l in ordered_point_labels:
            pts.append(self.labelled_points[l])
        return np.transpose(np.asarray(pts)).astype(np.float64)

    def get_default_cells(self):
        if self.points is None:
            self.points = self.get_default_points()

        cells = []
        ordered_point_labels = sorted(list(self.labelled_points.keys()))
        for poly_vertex_labels in self.labelled_polys:
            vertex_inds = [ordered_point_labels.index(l) for l in poly_vertex_labels]
            cells.append(vertex_inds)

        return cells

    def save(self, file_name):
        pd = make_vtk_polydata(self.points.T, self.cells)
        pd = run_triangle_filter(pd)
        polydata_save(pd, file_name)

    def apply_transformation(self, transform, in_place=False):
        new_points = transform.apply(self.points)
        if in_place:
            self.points = new_points
            return self
        else:
            return Glyph3D(new_points)


class Glyph2D(object):
    default_points = np.asarray(
        [
            [0, 0],
            [2, 0],
            [2, 2],
            [1, 1],
            [1, 4],
            [0, 4]
        ]
    ).T

    def __init__(self, points=None, facecolor='blue'):
        if points is None:
            self.points = self.default_points
        else:
            self.points = points

        self.facecolor = facecolor


    def apply_transformation(self, transform, in_place=False):
        new_points = transform.apply(self.points)
        if in_place:
            self.points = new_points
            return self
        else:
            return Glyph2D(new_points)

    def get_patch(self, facecolor=None, label=None, **kwargs):
        if facecolor is None:
            facecolor = self.facecolor

        return Polygon(self.points.T, closed=True, facecolor=facecolor, label=label, **kwargs)

    def bounds(self):
        min_vals = np.min(self.points, axis=1)
        max_vals = np.max(self.points, axis=1)
        x0 = min_vals[0]
        y0 = min_vals[1]
        x1 = max_vals[0]
        y1 = max_vals[1]

        return x0, y0, x1, y1


def get_glyph_bounds(glyphs):
    all_bounds = [list(g.bounds()) for g in glyphs]
    all_bounds = np.asarray(all_bounds)
    min_vals = np.min(all_bounds, axis=0)
    max_vals = np.max(all_bounds, axis=0)
    x0 = min_vals[0]
    y0 = min_vals[1]
    x1 = max_vals[2]
    y1 = max_vals[3]
    return x0, y0, x1, y1

class Line2D:
    parallel_tol_angle = 0.001

    def __init__(self, pt, direction):
        self.point = ensure_vec(pt)
        self.direction = ensure_unit_vec(direction)
        self.perp = [-1.0 * self.direction[1], self.direction[0]]

        u, v = self.direction

        p0, p1 = self.point

        # ax + by = c
        self.a = float(v)
        self.b = -1.0 * float(u)
        self.c = float(v * p0 - u * p1)

        self.theta = np.arctan2(v, u)
        return

    def apply_transformation(self, transf):
        p = self.point
        q = p + 10.0 * self.direction

        p2 = transf.apply(p)
        q2 = transf.apply(q)

        d2 = q2 - p2

        return Line2D(p2, d2)


    def f_x(self, x):
        if np.abs(self.b) > 0:
            y = (self.c - self.a * x) / self.b
        else:
            raise Exception('Line is of form x = constant')
        return y

    def get_point_on_line(self):
        x = 0
        try:
            y = self.f_x(x)
        except:
            x = self.c / self.a
            y = 0
        return ensure_vec([x, y])

    def angle_to(self, other: Line2D):
        return wrap_angle_minus_pi_to_pi(other.theta - self.theta)

    def parallel_to(self, other: Line2D):
        angle_diff = np.abs(self.angle_to(other))
        return  (angle_diff < self.parallel_tol_angle) or (np.abs(angle_diff - np.pi) < self.parallel_tol_angle)

    def intersection(self, other: Line2D):

        M = np.asarray([
            [self.a, self.b],
            [other.a, other.b]
        ])

        consts = ensure_vec([self.c, other.c])

        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

        if np.abs(det) < 0.001:
            raise Exception('Lines are parallel or near to it.')


        Minv = np.array([
            [other.b, -1 * self.b],
            [-1 * other.a, self.a]
        ]) / det

        xy = Minv @ consts

        return xy


    def nearest_point_on_line_to(self, other):
        """
        Get a point on this line that is closest to the give one.
        """
        p_other = ensure_vec(other)

        line_other = Line2D(p_other, self.perp)

        pt = self.intersection(line_other)

        return pt




class Icosahedron(object):

    def __init__(self, scale=1.0):

        self.n_vertices = 12
        self.n_edges = 30
        self.n_faces = 20

        self.vertex_label = 'ABCDEFGHIJKL'

        # For default scale, 1.0, the edge length of the icosahedron is 2.
        self.scale = scale

        phi = (1 + np.sqrt(5)) / 2
        mphi = -1.0 * phi

        vertex = np.asarray(
            [
                [1, phi, 0],  # 0 A
                [1, mphi, 0],  # 1 B
                [-1, phi, 0],  # 2 C
                [-1, mphi, 0],  # 3 D
                [phi, 0, 1],  # 4 E
                [mphi, 0, 1],  # 5 F
                [phi, 0, -1],  # 6 G
                [mphi, 0, -1],  # 7 H
                [0, 1, phi],  # 8 I
                [0, 1, mphi],  # 9 J
                [0, -1, phi],  # 10 K
                [0, -1, mphi]  # 11 L
            ]
        )

        self.vertex_coord = vertex * scale

        # Index of vertex opposite
        #                   0  1  2  3  4  5  6  7   8   9 10 11
        self.v_index_opp = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]

        # Anti-clockwise viewed from outside.
        face_label = ['ACI', 'CFI', 'FKI', 'FDK', 'KDB', 'BEK', 'EIK', 'EAI',  # Above xy plane
                      'AEG', 'BGE', 'CHF', 'HDF',  # Cross xy plane
                      'AJC', 'CJH', 'HJL', 'DHL', 'DLB', 'BLG', 'GLJ', 'AGJ']  # Below xy plane

        self.face_label = face_label

        self.edge_label = self.init_edge_label()

        # Each face is [v0, v1, v2], indices of vertices - order as above.
        self.face = self.init_face_index()

        # Each edge is [v0, v1], indices of vertices with v0 < v1
        self.edge = self.init_edge_index()

        # List of six pairs of opposite vertices (by their index in the vertex arrays).
        self.opp_vertices = self.init_opposite_vertices()

        # List of fifteen pairs of opposite edges (by their index into the edge array).
        self.opp_edges = self.init_opposite_edges()

        # List of ten pairs of opposite faces (by their index into the face array).
        self.opp_faces = self.init_opposite_faces()

        self.face_centroid = self.init_face_centroids()

        self.edge_midpoint = self.init_edge_midpoints()

        return

    def init_edge_midpoints(self):
        edge_midpoint = []
        for e in self.edge:
            idx_v0, idx_v1 = e
            v0 = self.vertex_coord[idx_v0]
            v1 = self.vertex_coord[idx_v1]
            midpt = (v0 + v1) / 2.0
            edge_midpoint.append(midpt)

        return np.asarray(edge_midpoint)

    def init_face_centroids(self):
        face_centroid = []
        for f in self.face:
            idx_v0, idx_v1, idx_v2 = f
            v0 = self.vertex_coord[idx_v0]
            v1 = self.vertex_coord[idx_v1]
            v2 = self.vertex_coord[idx_v2]
            centroid = (v0 + v1 + v2) / 3.0
            face_centroid.append(centroid)

        return np.asarray(face_centroid)

    def generate_rotations(self):
        yield np.eye(3)
        yield from self.gen_face_rotations()
        yield from self.gen_edge_rotations()
        yield from self.gen_vertex_rotations()


    def gen_vertex_rotations(self):
        theta_1 = 2.0 * np.pi / 5.0
        angles = [n * theta_1 for n in [1.0, 2.0, 3.0, 4.0]]

        for k, k_opp in self.opp_vertices:
            v = self.vertex_coord[k]
            v_opp = self.vertex_coord[k_opp]
            axis = v_opp - v

            for theta in angles:
                M = rotation_matrix_from_axis_and_angle(axis, theta)
                yield M

    def gen_edge_rotations(self):

        for e0, _ in self.opp_edges:
            idx_v0, idx_w0 = self.edge[e0]
            v0 = self.vertex_coord[idx_v0]
            w0 = self.vertex_coord[idx_w0]
            axis = w0 - v0

            M = rotation_matrix_from_axis_and_angle(axis, np.pi)
            yield M

    def gen_face_rotations(self):
        theta1 = 2.0 * np.pi / 3.0
        angles = [theta1, 2.0 * theta1]
        for f0, f1 in self.opp_faces:

            centroid0 = self.face_centroid[f0]
            centroid1 = self.face_centroid[f1]
            axis = centroid1 - centroid0

            for theta in angles:
                M = rotation_matrix_from_axis_and_angle(axis, theta)
                yield M

    def init_opposite_vertices(self):
        opp_vertices = []
        for k in range(self.n_vertices):

            k_opp = self.opp_vertex_index(k)

            opp_pair = [k, k_opp]
            if k > k_opp:
                opp_pair = [k_opp, k]

            if opp_pair in opp_vertices:
                continue

            opp_vertices.append(opp_pair)

        return opp_vertices

    def init_opposite_edges(self):
        opp_edges = []
        for k in range(self.n_edges):
            k_opp = self.opp_edge_index(k)

            opp_pair = [k, k_opp]
            if k > k_opp:
                opp_pair = [k_opp, k]

            if opp_pair in opp_edges:
                continue

            opp_edges.append(opp_pair)

        return opp_edges

    def init_opposite_faces(self):

        opp_faces = []

        for k in range(self.n_faces):
            k_opp = self.opp_face_index(k)

            opp_pair = [k, k_opp]
            if k > k_opp:
                opp_pair = [k_opp, k]

            if opp_pair in opp_faces:
                continue

            opp_faces.append(opp_pair)

        return opp_faces


    def init_edge_label(self):
        edge_label = []
        for face_l in self.face_label:
            # This will double count the edges:
            v0, v1, v2 = face_l
            e1 = sorted([v0, v1])
            e2 = sorted([v1, v2])
            e3 = sorted([v2, v0])
            edge_label += [''.join(e) for e in [e1, e2, e3]]

        # Fix double counting.
        return list(set(edge_label))

    def init_face_index(self):
        face_index = []

        for face_l in self.face_label:
            # Store each face's vertex indices.
            v0_l, v1_l, v2_l = face_l
            v0_idx = self.vertex_label.index(v0_l)
            v1_idx = self.vertex_label.index(v1_l)
            v2_idx = self.vertex_label.index(v2_l)
            face_index.append([v0_idx, v1_idx, v2_idx])

        return np.asarray(face_index, dtype='int')

    def init_edge_index(self):
        edge_index = []

        for v1_label, v2_label in self.edge_label:
            v1_idx = self.vertex_label.index(v1_label)
            v2_idx = self.vertex_label.index(v2_label)
            edge_index.append([v1_idx, v2_idx])

        return edge_index

    def to_vtk_polydata(self):

        pd = make_vtk_polydata(self.vertex_coord, self.face)
        return pd

    def save_as_vtk_polydata(self, file_name, clobber=True):

        pd = self.to_vtk_polydata()
        polydata_save(pd, file_name, clobber=clobber)
        return

    def opp_vertex_index(self, v_index):
        assert -1 < v_index < 12, 'Index out of range.'
        return self.v_index_opp[v_index]

    def opp_vertex_label(self, v_label):
        assert v_label in self.vertex_label, 'Invalid vertex label.'
        v_idx = self.vertex_label.index(v_label)
        v_idx_opp = self.opp_vertex_index(v_idx)
        v_label_opp = self.vertex_label[v_idx_opp]
        return v_label_opp

    def opp_edge_index(self, e_index):
        assert -1 < e_index < self.n_edges, 'Invalid edge index'

        v0, v1 = self.edge[e_index]
        v0_opp = self.opp_vertex_index(v0)
        v1_opp = self.opp_vertex_index(v1)
        if v0_opp > v1_opp:
            v0_opp, v1_opp = v1_opp, v0_opp
        e_opp = [v0_opp, v1_opp]
        return self.edge.index(e_opp)

    def opp_edge_label(self, e_label):
        assert len(e_label) == 2, 'Invalid edge label.'

        v0_label, v1_label = e_label

        assert v0_label in self.vertex_label, 'Invalid vertex label.'
        assert v1_label in self.vertex_label, 'Invalid vertex label.'

        # Edges are are written alphabetically.
        if v0_label > v1_label:
            e_label = f'{v1_label}{v0_label}'

        assert e_label in self.edge_label, 'Edge does not exist.'

        e_index = self.edge_label.index(e_label)

        e_idx_opp = self.opp_edge_index(e_index)

        return self.edge_label[e_idx_opp]

    def opp_face_index(self, f_index):

        assert -1 < f_index < self.n_faces, 'Invalid face index'

        v0, v1, v2 = self.face[f_index]

        v0_opp = self.opp_vertex_index(v0)
        v1_opp = self.opp_vertex_index(v1)
        v2_opp = self.opp_vertex_index(v2)

        f_opp_vs_sort = sorted([v0_opp, v1_opp, v2_opp])

        all_f_vs_sort = [sorted(vs) for vs in self.face]

        assert f_opp_vs_sort in all_f_vs_sort, 'Cannot find opposing face index.'

        return all_f_vs_sort.index(f_opp_vs_sort)

    def opp_face_label(self, f_label):
        assert self.face_with_vertex_labels_exists(f_label), 'No face with given vertex labels.'
        f_label = self.standardise_face_label(f_label)

        f_idx = self.face_label.index(f_label)
        f_idx_opp = self.opp_face_index(f_idx)
        return self.face_label[f_idx_opp]

    def face_with_vertex_labels_exists(self, face_label):
        """
        Check a face exists with the three vertices given in the face label.
        They might not be specified in the right order.
        """
        vs_sort = sorted(face_label)
        all_vs_sort = [sorted(vs) for vs in self.face_label]
        return vs_sort in all_vs_sort

    def standardise_face_label(self, face_label):
        """
        Given a triplet of vertices for a face, find the representation
        listed in standard form.
        """
        assert self.face_with_vertex_labels_exists(face_label)
        vs_sort = sorted(face_label)
        all_vs_sort = [sorted(vs) for vs in self.face_label]
        f_idx = all_vs_sort.index(vs_sort)
        return self.face_label[f_idx]


class PointList(object):

    def __init__(self, points=None):

        if points is None:
            self.points = []
            return

        for p in points:
            self.append_point(p)

        return

    def append_point(self, q):
        q = ensure_vec(q)
        ix = np.argwhere([np.allclose(p, q) for p in self.points]).flatten()
        if len(ix) < 1:
            ix = len(self.points)
            self.points.append(q)
            return ix

        assert len(ix) == 1
        return int(ix[0])





