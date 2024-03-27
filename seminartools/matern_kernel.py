import pymc as pm
from .utils import geo_distance
from pymc.gp.cov import Matern32
import numpy as np


class MaternGeospatial(pm.gp.cov.Stationary):
    """
    Custom Matern kernel wrapper that is used to convert geographical distances between countries
    into a prior covariance matrix for a Gaussian process.
    """

    def __init__(
        self,
        input_dims,
        ls,
        active_dims=None,
        distance_function: callable = geo_distance,
        distance_scaling: float = 1000,
        matern: pm.gp.cov.BaseCovariance = Matern32,
    ):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.distance_function = distance_function
        self.matern = matern(input_dims, ls, active_dims)
        self.distance_scaling = distance_scaling

    def _geographical_distance_matrix(self, countries):
        """
        Create a matrix of geographical distances between countries based on their coordinates.
        """
        distance_matrix = np.zeros((len(countries), len(countries)))

        for i, country_i in enumerate(countries):
            for j, country_j in enumerate(countries):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    distance_matrix[i, j] = self.distance_function(country_i, country_j)
        return distance_matrix

    def full(self, X, Xs=None):
        """
        Returns the full covariance matrix
        """
        distance_matrix = self._geographical_distance_matrix(X)
        return self.matern.full_from_distance(distance_matrix / self.distance_scaling)
