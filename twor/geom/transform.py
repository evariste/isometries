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


class Identity(Transform):
    def __init__(self, dimension):
        super().__init__()
        self.dim = dimension
        self.matrix = np.eye(1 + self.dim)

    def apply(self, points):
        return validate_pts(points)

    def get_matrix(self):
        return self.matrix
