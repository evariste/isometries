from __future__ import annotations


"""
Created by: Paul Aljabar
Date: 08/06/2023
"""

from abc import ABC, abstractmethod
import numpy as np
from twor.utils.general import validate_pts


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

    @abstractmethod
    def two_step_form(self):
        """
        The two step form represents the transformation as an orthogonal part
        followed by a translation, i.e.,

        t M in mathematical notation (M applied first).

        This function returns a list [M t] i.e., the index of the transformation
        in the list represents the order of application.

        We could have chosen the form to be translation first, but will make the
        convention that the 'standard' two-step form applies the orthogonal
        transformaation first then the translation second.
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

    def two_step_form(self):
        return [Identity(2), Identity(2)]
