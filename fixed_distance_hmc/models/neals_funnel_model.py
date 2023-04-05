#!/usr/env/python
# Copyright 2022 The Centre for Translational Data Science (CTDS)
# at the University of Sydney. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Created by Hadi Afshar.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fixed_distance_hmc.models.abstract_model import Model
import numpy as np


class NealsFunnel(Model):
    def __init__(self, dim: int, sigma2=1.0, k=2.0):
        """
        :param dim: n + 1
        :param sigma2:
        :param k:
        returns Neals_Funnel(x, y_1, ..., y_n)) = N(x; 0, sigma2) Prod_{i=1}^n N(y_i; 0, e^{kx})
        """
        assert dim > 1
        self.n = dim - 1
        self.k = k
        self.sig2 = sigma2
        self.scale = sigma2 ** 0.5

    def pr(self, w):
        return np.exp(-self.u(w))

    def u(self, q):
        x = q[0]
        y = q[1:]
        assert y.shape[0] == self.n
        return 0.5 * x * x / self.sig2 + 0.5 * self.n * self.k * x + 0.5 * y.dot(y) * np.exp(-self.k * x)

    def grad_u(self, q):
        x = q[0]
        y = q[1:]
        assert y.shape[0] == self.n
        ekx = np.exp(-self.k * x)
        assert np.isfinite(ekx)

        dudx = x / self.sig2 + 0.5 * self.n * self.k - 0.5 * self.k * y.dot(y) * ekx
        dudy = y * ekx
        return np.concatenate([[dudx], dudy])

    def oracle_draw_sample(self):
        # Drawing from N(x; 0, sigma2)
        x = np.random.normal(loc=0., scale=self.scale)

        # Drawing from Prod_{i=1}^n N(y_i; 0, e^{kx})
        y = np.random.normal(loc=0., scale=np.exp(self.k * x) ** 0.5, size=self.n)
        return np.concatenate([[x], y])

    @staticmethod
    def name():
        return 'Funnel'
