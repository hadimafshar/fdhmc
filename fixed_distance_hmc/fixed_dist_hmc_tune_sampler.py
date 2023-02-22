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
from fixed_distance_hmc.abstract_sampler import *
from scipy.stats import chi
from fixed_distance_hmc.hmc_utils import find_reasonable_epsilon
from tqdm import tqdm

PROPOSAL_DISTANCE_FORM_CURRENT_Q = 'info.proposal.distance.form.current.q'

TOTAL_TRAVERSED_DISTANCE = 'info.total.traversed.distance'


class AdaptiveEpsChiFixedDistSampler(Sampler):
    def __init__(self, model, fixed_distance, init_epsilon,
                 init_q,  # starting point for adaptively selecting epsilon
                 num_adapt_iterations,  # M_adapt
                 target_mean_acceptance_prob_delta=0.65,  # delta
                 name='AdaptEpsFD_HMC', alias='AdaptEpsFD_HMC', color='k'):
        super().__init__(model=model, name=name, alias=alias, color=color)
        self.model = model
        self.fixed_distance_mean = fixed_distance  # D

        assert 0 < target_mean_acceptance_prob_delta < 1
        self.target_mean_accept_prob_delta = target_mean_acceptance_prob_delta

        self.epsilon = init_epsilon
        self.mu = np.log(1 * self.epsilon)
        self.epsilon_bar = 1.0
        self.H_bar = 0.0
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75

        current_q = init_q.copy()
        for m in tqdm(range(1, num_adapt_iterations + 1)):
            current_q, _ = self.next_sample_info(current_q,
                                                 fixed_distance=self.fixed_distance_mean,
                                                 adapt_phase_m=m)
        self.epsilon = self.epsilon_bar

    def next_sample(self, current_q, adapt_phase_m=None):
        sample, info = self.next_sample_info(current_q=current_q, adapt_phase_m=adapt_phase_m)
        return sample

    def next_sample_info(self, current_q, current_p=None, fixed_distance=None, verbose_info=False, adapt_phase_m=None):
        info = {INFO_NUM_GRADS: 0}

        if fixed_distance is None:
            fixed_distance = np.random.uniform(low=self.fixed_distance_mean * 0.7, high=self.fixed_distance_mean * 1.2)

        q = current_q.copy()
        if current_p is None:
            # p = np.random.normal(size=q.shape[0])
            p = self.generate_momentum(dim=q.shape[0])
        else:
            assert current_p.shape == current_q.shape
            p = current_p

        norm_p1 = np.linalg.norm(p)
        h1 = 0.5 * norm_p1 * norm_p1 + self.model.u(q)
        assert np.isfinite(h1)

        tau1 = np.random.uniform(0, self.epsilon)  # initial position evolution

        # initial evolution of q for time tau1:
        q += tau1 * p

        remaining_dist = fixed_distance - tau1 * norm_p1  # the remained distance

        if remaining_dist <= 0.0:  # checking this condition does not affect the results but only saves one gradient
            tau = fixed_distance / norm_p1
            q = current_q + tau * p
            norm_p2 = norm_p1
        else:
            info[INFO_NUM_GRADS] += 1
            p -= self.epsilon * self.model.grad_u(q)  # full-step momentum evolution

            if verbose_info:
                info[INFO_INIT_EVOLVE_Q] = np.array([current_q.copy(), q.copy()])
                info[INFO_LEAPFROG_QS] = q.copy()[np.newaxis, :]
                info[INFO_LEAPFROG_PS] = p.copy()[np.newaxis, :]

            while np.linalg.norm(p) * self.epsilon < remaining_dist:
                # full-step q:
                q += self.epsilon * p

                remaining_dist -= np.linalg.norm(p) * self.epsilon  # update remaining distance

                p -= self.epsilon * self.model.grad_u(q)  # full-step p

                info[INFO_NUM_GRADS] += 1
                if verbose_info:
                    info[INFO_LEAPFROG_QS] = np.append(info[INFO_LEAPFROG_QS], q.copy()[np.newaxis, :], axis=0)
                    info[INFO_LEAPFROG_PS] = np.append(info[INFO_LEAPFROG_PS], p.copy()[np.newaxis, :], axis=0)

            # final evolution of position for time rho/||p||:
            norm_p2 = np.linalg.norm(p)

            tau2 = remaining_dist / norm_p2

            q += p * tau2

            if verbose_info:
                info[INFO_FINAL_EVOLVE_Q] = np.array([info[INFO_LEAPFROG_QS][-1, :], q.copy()])

        # Hamiltonian of the evolved state (x', v'):
        h2 = 0.5 * norm_p2 * norm_p2 + self.model.u(q)
        assert np.isfinite(h2)

        accept_prob = np.min([1, np.exp(-h2 + h1)])
        info[INFO_ACCEPT_PROB] = accept_prob
        info[PROPOSAL_DISTANCE_FORM_CURRENT_Q] = np.linalg.norm(q - current_q)

        if adapt_phase_m is not None:
            # then we are in the adapt mode:
            self.H_bar = (1 - (1 / (adapt_phase_m + self.t0))) * self.H_bar + \
                         (1 / (adapt_phase_m + self.t0)) * (self.target_mean_accept_prob_delta - accept_prob)
            self.epsilon = np.exp(self.mu - (np.sqrt(adapt_phase_m) / self.gamma) * self.H_bar)
            self.epsilon_bar = np.exp((adapt_phase_m ** (-self.kappa)) * np.log(self.epsilon) +
                                      (1 - adapt_phase_m ** (-self.kappa)) * np.log(self.epsilon_bar))

        if np.random.uniform(low=0, high=1) < accept_prob:  # (np.exp(-h2) * norm_p1) / (np.exp(-h1) * norm_p2):
            return q, info  # the evolved state
        else:
            return current_q, info

    @staticmethod
    def generate_momentum(dim):
        s = np.random.normal(size=dim)
        direction = s / np.linalg.norm(s)  # a vector drawn from dim-sphere of radius 1

        magnitude = chi.rvs(df=dim + 1)
        return magnitude * direction

    @staticmethod
    def generate_expected_magnitude_momentum(dim):
        s = np.random.normal(size=dim)
        direction = s / np.linalg.norm(s)  # a vector drawn from dim-sphere of radius 1

        mean_magnitude = chi.stats(df=dim + 1, moments='m')
        return mean_magnitude * direction


class AutoTuneChiFixedDistSampler(Sampler):
    def __init__(self, model,  # fixed_distance, init_epsilon,
                 init_q,  # starting point for adaptively selecting epsilon
                 num_iters1_eps_adapt=100,  # M_adapt (of the auxiliary sampler to tune its epsilon)
                 num_iters2_aux_sampler_tune_dist=500,  # no. draws from the auxiliary sampler to estimate Distance
                 target_mean_acceptance_prob_delta=0.65,  # delta
                 sufficiently_large_D_param=10,
                 name='AutoTuned_FDHMC', alias='FDHMC', color='k'):
        super().__init__(model=model, name=name, alias=alias, color=color)

        # these parameters are used for tuning when activated
        self.__init_q = init_q
        self.__num_iters1_eps_adapt = num_iters1_eps_adapt
        self.__num_iters2_aux_sampler_tune_dist = num_iters2_aux_sampler_tune_dist
        self.__target_mean_acceptance_prob_delta = target_mean_acceptance_prob_delta
        self.__sufficiently_large_D_param = sufficiently_large_D_param

    def activate(self):
        p_with_expected_magnitude = AdaptiveEpsChiFixedDistSampler.generate_expected_magnitude_momentum(
            dim=self.__init_q.shape[0])
        reasonable_eps = find_reasonable_epsilon(
            model=self.model, q=self.__init_q, p_with_expected_magnitude=p_with_expected_magnitude)

        reasonable_p_magnitude = np.linalg.norm(p_with_expected_magnitude)
        sufficiently_large_distance = reasonable_eps * reasonable_p_magnitude * self.__sufficiently_large_D_param

        auxiliary_sampler = AdaptiveEpsChiFixedDistSampler(model=self.model, fixed_distance=sufficiently_large_distance,
                                                           init_q=self.__init_q,
                                                           init_epsilon=reasonable_eps,
                                                           num_adapt_iterations=self.__num_iters1_eps_adapt,
                                                           target_mean_acceptance_prob_delta=self.__target_mean_acceptance_prob_delta,
                                                           name='AUXILARY SAMPLER')
        sum_proposal_dist_from_current_q = 0.
        num_aux_samples_to_tune_distance = self.__num_iters2_aux_sampler_tune_dist
        sample = self.__init_q.copy()
        for i in tqdm(range(num_aux_samples_to_tune_distance)):
            sample, info = auxiliary_sampler.next_sample_info(current_q=sample)
            sum_proposal_dist_from_current_q += info[PROPOSAL_DISTANCE_FORM_CURRENT_Q]

        average_proposal_distance = sum_proposal_dist_from_current_q / num_aux_samples_to_tune_distance

        self.inner_sampler = AdaptiveEpsChiFixedDistSampler(model=self.model, fixed_distance=average_proposal_distance,
                                                            init_q=self.__init_q,
                                                            init_epsilon=auxiliary_sampler.epsilon,
                                                            num_adapt_iterations=self.__num_iters1_eps_adapt)

    def next_sample(self, current_q, adapt_phase_m=None):
        return self.inner_sampler.next_sample(current_q=current_q, adapt_phase_m=adapt_phase_m)

    def next_sample_info(self, current_q, current_p=None,
                         fixed_distance=None, verbose_info=False,
                         adapt_phase_m=None):
        return self.inner_sampler.next_sample_info(current_q=current_q, current_p=current_p,
                                                   fixed_distance=fixed_distance, adapt_phase_m=adapt_phase_m)


