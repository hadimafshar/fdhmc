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

from fixed_distance_hmc.abstract_sampler import *
from fixed_distance_hmc.hmc_utils import *
import numpy as np


class AdaptiveHmc(Sampler):
    def __init__(self, model,
                 num_samples_for_tuning, init_sample, lambda_simulation_length, delta_dual_av_param=0.65,
                 # init_eps=1.,
                 name='Adapt-HMC', alias='Adapt-HMC', color='g'):
        """
        :param model: model
        :param num_samples_for_tuning: number of draws for tuning
        :param init_sample: initial state starting from which tuning is done
        :param lambda_simulation_length: simulation length = epsilon*L
        :param delta_dual_av_param: dual averaging (ideal acceptance rate) parameter in (0, 1)
        :param name: samplers name
        """
        super().__init__(model=model, name=name, alias=alias, color=color)
        self.u = self.model.u
        self.grad_u = self.model.grad_u

        num_trials = 0
        while True:
            num_trials += 1
            try:
                self.L, self.epsilon = self.adapt_choose_L_epsilon(
                    num_samples_for_tuning=num_samples_for_tuning,
                    init_sample=init_sample,  # init_eps=init_eps,
                    lambda_simulation_length=lambda_simulation_length,
                    delta_dual_av_param=delta_dual_av_param)
                if self.L < 3 or self.L > 200:
                    print('UNSTBLE DUAL TUNING, L=', self.L)
                    raise Exception()
                break
            except Exception as e:
                print("Exception caught in Adapt-HMC initialization: ", e)
                lambda_simulation_length *= 0.5
                print('Halving simulation length to {a}'.format(a=lambda_simulation_length))
                assert lambda_simulation_length > 0
                if num_trials > 4:
                    self.L = 10
                    self.epsilon = lambda_simulation_length / self.L
                    break

        self._name = name
        print('{s}: \t Best L: {l}, eps: {e}'.format(s=name, l=self.L, e=self.epsilon))

    def hamiltonian(self, p, q):
        return 0.5 * p.dot(p) + self.u(q)

    def adapt_choose_L_epsilon(self, num_samples_for_tuning, init_sample, lambda_simulation_length,
                               delta_dual_av_param, init_eps=None, eps0bar=1.0, H_bar=0.0, gamma=0.05, t0=10.0,
                               kappa=0.75):

        eps0 = find_reasonable_epsilon(model=self.model, q=init_sample, p_with_expected_magnitude=np.random.normal(
            size=init_sample.shape[0])) if init_eps is None else init_eps

        mu = np.log(10 * eps0)

        dim = init_sample.shape[0]
        current_sample = init_sample.copy()
        ln_eps = np.log(eps0)  # log eps_{m-1}
        ln_eps_bar = np.log(eps0bar)  # log eps_bar_{m-1}

        for m in range(1, num_samples_for_tuning):
            p = np.random.normal(size=dim)  # r~
            q = current_sample.copy()  # theta~
            h0 = self.hamiltonian(p=p, q=q)
            assert np.isfinite(h0)
            _eps = np.exp(ln_eps)
            Lm = max(1, round(lambda_simulation_length / _eps))
            for l in range(Lm):
                p = p - 0.5 * _eps * self.grad_u(q)
                assert np.isfinite(p).all()
                q = q + _eps * p
                assert np.isfinite(q).all()
                p = p - 0.5 * _eps * self.grad_u(q)
            h = self.hamiltonian(p=p, q=q)
            assert np.isfinite(h)
            delta_h = h - h0
            accept = min(1., np.exp(-delta_h))
            if np.random.uniform() < accept:
                current_sample = q

            H_bar = (1. - 1. / (m + t0)) * H_bar + (1. / (m + t0)) * (delta_dual_av_param - accept)
            ln_eps = mu - ((m ** 0.5) / gamma) * H_bar
            ln_eps_bar = (m ** (-kappa)) * ln_eps + (1 - m ** (-kappa)) * ln_eps_bar  # see eq (6)

        L = max(1, round(lambda_simulation_length / np.exp(ln_eps_bar)))
        return L, np.exp(ln_eps_bar)

    def next_sample(self, current_q):
        sample, info = self.next_sample_info(current_q)
        return sample

    def next_sample_info(self, current_q, current_p=None, verbose_info=False):
        info = {INFO_NUM_GRADS: 2 + self.L}  # for the first and last half-step

        p = np.random.normal(size=current_q.shape[0])
        q = current_q
        h0 = self.hamiltonian(p=p, q=q)

        for l in range(self.L):
            p = p - 0.5 * self.epsilon * self.grad_u(q)
            q = q + self.epsilon * p
            p = p - 0.5 * self.epsilon * self.grad_u(q)

        h = self.hamiltonian(p=p, q=q)

        delta_h = h - h0
        if np.random.uniform() < np.exp(-delta_h):
            return q, info
        else:
            return current_q, info
