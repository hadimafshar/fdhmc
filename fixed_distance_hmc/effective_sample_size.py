#!/usr/env/python
# Copyright 2021 The Centre for Translational Data Science (CTDS)
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

import numpy as np


def __rho_s_hat_f(mu_hat_f, sig2_hat_f, M, s, f_arr):
    result = 0.0
    for m in np.arange(s + 1, M + 1):
        result += (f_arr[m - 1] - mu_hat_f) * (f_arr[m - s - 1] - mu_hat_f)  # since indexes starts from 0 rather than 1
    return result / (sig2_hat_f * (M - s))


def ess_per_dim(theta_array, ref_samples, cutoff=0.05):
    """
    :param theta_array:  MxDim array where M is the sample size and Dim is the dimensionality of the distribution
    :param ref_samples: reference NxDim array (where N >> M)
    :param cutoff: parameter, see "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
Matthew" appendix
    :return: Effective sample size per dim
    """
    dim = theta_array.shape[1]
    assert dim == ref_samples.shape[1]

    true_mean = ref_samples.mean(axis=0)
    true_sigma2 = ref_samples.var(axis=0)
    assert true_mean.shape[0] == dim
    return np.array([effective_sample_size1d(f_arr=theta_array[:, d], mu_hat_f=true_mean[d], sig2_hat_f=true_sigma2[d],
                                    cutoff=cutoff) for d in range(dim)])


def effective_sample_size1d(f_arr, mu_hat_f, sig2_hat_f, cutoff=0.05):
    assert f_arr.ndim == 1
    M = len(f_arr)

    a = 0.0
    for s in np.arange(1, M):  # so its is up to M - 1
        rho_s_hat_f = __rho_s_hat_f(mu_hat_f, sig2_hat_f, M, s, f_arr)

        if rho_s_hat_f < 0:
            break

        a += (1 - s / M) * rho_s_hat_f
        if rho_s_hat_f < cutoff:
            break

    return M / (1 + 2 * a)


