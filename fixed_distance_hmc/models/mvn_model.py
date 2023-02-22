from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from fixed_distance_hmc.models.abstract_model import Model
from scipy.stats import wishart
from fixed_distance_hmc.abstract_sampler import Sampler


class MvnOracleSampler(Sampler):
    def __init__(self, model):
        super().__init__(model=model, name='MVN', alias='MVN', color='k')
        assert type(model) == ExperimentalMultiVarNormalModel
        self.cov = model.A
        assert self.cov.shape[0] == self.cov.shape[1]
        self.mean = np.zeros(self.cov.shape[0])

    def next_sample(self, current_sample):
        return np.random.multivariate_normal(self.mean, self.cov)


class ExperimentalMultiVarNormalModel(Model):
    def __init__(self, dim=250, A=None, seed=10000):
        self.dim = dim
        if A is None:
            self.A = wishart.rvs(df=dim, scale=np.eye(dim), random_state=seed)
        else:
            assert A.shape == (dim, dim)
            self.A = A

    def pr(self, q):
        return np.exp(-0.5 * q @ self.A @ q)

    def grad_u(self, q):
        return self.A @ q

    @staticmethod
    def name():
        return 'mvn'



