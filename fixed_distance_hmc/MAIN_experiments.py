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
import math

import pickle
import time
from tqdm import tqdm
from fixed_distance_hmc.abstract_sampler import *
from fixed_distance_hmc.adaptive_hmc_sampler import AdaptiveHmc
from fixed_distance_hmc.models.mvn_model import MvnOracleSampler, ExperimentalMultiVarNormalModel
from fixed_distance_hmc.models.bayesian_lr_models import GermanCreditBlrModel, SpectBlrModel, AustralianCreditBlrModel
from fixed_distance_hmc.models.neals_funnel_model import NealsFunnel
from fixed_distance_hmc.nuts_efficient_sampler_dual_averaging import EffectiveNUTSSamplerWithDualAveraging
from fixed_distance_hmc.fixed_dist_hmc_tune_sampler import AutoTuneChiFixedDistSampler
from fixed_distance_hmc.effective_sample_size import *
from fixed_distance_hmc.utils import *


class Experiment:
    def __init__(self, experiment_type, model, init_sample, num_samples,
                 mcmc_chains, samplers, path, reference_sampler=None, num_reference_samples=0,
                 num_burn_in_samples=0, max_sampling_seconds=None):
        self.experiment_type = experiment_type  # 'trace_itr_based'  # 'itr_based' # 'time_based'
        self.model = model
        self.__init_sample = init_sample

        self.dim = self.__init_sample().shape[0] if callable(self.__init_sample) else self.__init_sample.shape[0]

        self.num_samples = num_samples
        if experiment_type == 'itr_based':
            prefix = 'Itr{i}_'.format(i=num_samples)
        elif experiment_type == 'time_based':
            prefix = 'Time{i}_'.format(i=max_sampling_seconds)
        else:
            raise

        self.mcmc_chains = [prefix + chain for chain in mcmc_chains]
        self.num_burn_in_samples = num_burn_in_samples
        self.max_sampling_seconds = max_sampling_seconds

        self.samplers = samplers
        self.path = path
        self.reference_sampler = reference_sampler
        self.num_reference_samples = num_reference_samples

    def init_sample(self):
        if callable(self.__init_sample):
            return self.__init_sample()
        else:
            return self.__init_sample.copy()


def generate_samples_times_total_grads(sampler, init_sample, number_of_samples, max_time_sec):
    current_sample = init_sample
    dim = current_sample.shape[0]

    samples = np.zeros(shape=(number_of_samples, dim), dtype=float)
    times = np.zeros(shape=number_of_samples, dtype=float)

    start_time = time.time()

    total_grads = 0.
    for sample_id in tqdm(range(0, number_of_samples)):
        current_sample, info = sampler.next_sample_info(current_sample.copy())
        assert not np.isnan(current_sample).any()
        if INFO_NUM_GRADS in info:
            total_grads += info[INFO_NUM_GRADS]
        else:
            raise

        samples[sample_id] = current_sample
        sampling_time = time.time() - start_time
        times[sample_id] = sampling_time
        if max_time_sec is not None and sampling_time > max_time_sec:
            print('sampling terminated after {n} samples due to time out'.format(n=sample_id + 1))
            return samples[:sample_id + 1], times[:sample_id + 1]

    # print('::::total_grads>', total_grads)
    return samples, times, total_grads


def save_samples_times_errors_grads(path, samples, times, errors, total_grads,
                                    model_name, sampler_name, dimension, mcmc_chain, num_samples=None):
    if num_samples is None:
        num_samples = len(times)  # for itr experiments it is None, for time experiments it is manually set

    obj = {'samples': samples, 'times': times, 'errors': errors, INFO_NUM_GRADS: total_grads}

    # save error measure:
    with open(path + '/{m}_{a}_dim_{d}_chain_{c}.pickle'.format(m=model_name,
                                                                a=sampler_name,
                                                                d=dimension,
                                                                c=mcmc_chain), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_samples_times_errors(path, model_name, sampler_name, dimension, mcmc_chain, num_samples):
    with open(path + '/{m}_{a}_dim_{d}_chain_{c}_.pickle'.format(m=model_name,
                                                                 a=sampler_name,
                                                                 d=dimension,
                                                                 c=mcmc_chain),
              'rb') as f:
        dct = pickle.load(f)
        return dct['samples'], dct['times'], dct['errors']


def fetchsamples(sampler, sample, num_samples):
    dim = sample.shape[0]
    samples = np.zeros([num_samples, dim])

    for sample_id in tqdm(range(num_samples)):
        sample = sampler.next_sample(sample.copy())
        samples[sample_id, :] = sample

    return samples


def run_save_samplers(e):
    if e.reference_sampler is not None and e.num_reference_samples > 0:
        e.reference_sampler.activate()
        print('Taking {s} Reference samples...'.format(s=e.num_reference_samples))
        ref_samples = fetchsamples(e.reference_sampler, e.init_sample(), num_samples=e.num_reference_samples)
        with open(e.path + '/{m}_dim_{d}_Ref_samples.pickle'.format(m=e.model.name(), d=e.dim), 'wb') as f:
            pickle.dump(ref_samples, f, pickle.HIGHEST_PROTOCOL)

    for mcmc_chain in e.mcmc_chains:
        init_sample_for_all_samplers = e.init_sample()
        assert e.model.pr(init_sample_for_all_samplers) > 0

        for sampler in e.samplers:
            print("-----\n")

            print("sampling method: {s} \t model: {m} \t dim: {d} \t chain: {c}\n".format(s=sampler.name(),
                                                                                          m=e.model.name(),
                                                                                          d=e.dim, c=mcmc_chain))

            sampler.activate()

            current_sample = init_sample_for_all_samplers.copy()
            if e.num_burn_in_samples > 0:
                print("burn in...")
                for i in range(e.num_burn_in_samples):
                    current_sample, _ = sampler.next_sample_info(current_sample.copy())

            samples, times, total_grads = generate_samples_times_total_grads(sampler, current_sample, e.num_samples,
                                                                             e.max_sampling_seconds)
            errors = None  # error_measure(samples=samples, weights=weights, times=times)

            save_samples_times_errors_grads(path=e.path, samples=samples, times=times,
                                            errors=errors,
                                            total_grads=total_grads,
                                            model_name=e.model.name(), sampler_name=sampler.name(), dimension=e.dim,
                                            mcmc_chain=mcmc_chain)


def calc_ess(e):
    print('calc_ess...')
    # func = lambda x: x[0]
    with open(e.path + '/{m}_dim_{d}_Ref_samples.pickle'.format(m=e.model.name(), d=e.dim), 'rb') as f:
        ref_samples = pickle.load(f)
    true_mean = np.mean(ref_samples, axis=0)
    true_sigma2 = np.var(ref_samples, axis=0)
    assert true_mean.shape[0] == e.dim
    num_chains = len(e.mcmc_chains)

    for sampler in e.samplers:
        min_sampler_esses = np.zeros(num_chains)
        # mean_sampler_esses = np.zeros(num_chains)
        min_sampler_ess_per_secs = np.zeros(num_chains)
        min_sampler_ess_per_grads = np.zeros(num_chains)
        # mean_sampler_ess_per_secs = np.zeros(num_chains)
        for i, chain in enumerate(tqdm(e.mcmc_chains)):
            # print('chain:', chain)
            ess_per_dim, ess_sec_per_dim, ess_grads_per_dim = calc_chain_ess_and_ess_sec_and_ess_grads__per_dim(
                chain=chain, sampler=sampler, e=e,
                true_mean=true_mean,
                true_sigma2=true_sigma2)
            min_sampler_esses[i] = ess_per_dim.min()
            # mean_sampler_esses[i] = ess_per_dim.mean()
            min_sampler_ess_per_secs[i] = ess_sec_per_dim.min()
            min_sampler_ess_per_grads[i] = ess_grads_per_dim.min()
            # mean_sampler_ess_per_secs[i] = ess_sec_per_dim.mean()

        print('----')
        print(sampler.name(), '\t minESS:', str_mean_confidence_interval(min_sampler_esses))
        # print(sampler.name(), '\t meanESS:', str_mean_confidence_interval(mean_sampler_esses))
        print(sampler.name(), '\t minESS/sec:', str_mean_confidence_interval(min_sampler_ess_per_secs))
        # print(sampler.name(), '\t meanESS/sec', str_mean_confidence_interval(mean_sampler_ess_per_secs))
        print(sampler.name(), '\t minESS/grads:', str_mean_confidence_interval(min_sampler_ess_per_grads))
        print('----')


def calc_chain_ess_and_ess_sec_and_ess_grads__per_dim(chain, sampler, e, true_mean, true_sigma2):
    with open(e.path + '/{m}_{a}_dim_{d}_chain_{c}.pickle'.format(m=e.model.name(),
                                                                  a=sampler.name(),
                                                                  d=e.dim,
                                                                  c=chain),
              'rb') as f:
        dct = pickle.load(f)  # dct['samples'], dct['weights'], dct['times'], dct['errors']
        samples = dct['samples']
        last_time = dct['times'][-1]
        seconds = last_time / len(dct['times'])
        total_grads = dct[INFO_NUM_GRADS] if INFO_NUM_GRADS in dct else np.nan
        # print('seconds: ', seconds)
        ess_per_dim = np.array(
            [effective_sample_size1d(samples[:, d], mu_hat_f=true_mean[d], sig2_hat_f=true_sigma2[d]) for d in
             range(e.dim)])

        ess_per_dim_per_grad = ess_per_dim / total_grads if total_grads > 0 else np.nan
        return ess_per_dim, ess_per_dim / seconds, ess_per_dim_per_grad


##############################################################################
def main_MVN(path, dim):
    print('main_MVN . . . (dim={d})'.format(d=dim))

    model = ExperimentalMultiVarNormalModel(dim=dim)
    init_sample = np.ones(dim) * 0.01
    samplers = [
        AdaptiveHmc(model=model, num_samples_for_tuning=900,
                    init_sample=init_sample.copy(),
                    lambda_simulation_length=2, color='c'),
        EffectiveNUTSSamplerWithDualAveraging(model=model, init_q=init_sample.copy(), num_adapt_iterations=900,
                                              max_possible_depth=10, target_mean_acceptance_prob_delta=0.6, color='r'),
        AutoTuneChiFixedDistSampler(model=model, init_q=init_sample.copy(),
                                    num_iters1_eps_adapt=200,
                                    num_iters2_aux_sampler_tune_dist=500,
                                    sufficiently_large_D_param=10, color='k'),
    ]
    mvn_experiment = Experiment(experiment_type="itr_based",
                                model=model,
                                init_sample=init_sample, num_samples=1000, num_burn_in_samples=200,
                                mcmc_chains=['C{c}'.format(c=i) for i in range(0, 50)],
                                samplers=samplers,
                                path=path,
                                reference_sampler=MvnOracleSampler(model=model),
                                num_reference_samples=500000
                                )
    run_save_samplers(mvn_experiment)
    calc_ess(mvn_experiment)


def main_GrCr(path):
    print('main_GrCr . . .')
    model = GermanCreditBlrModel(sigma2=1.0)
    dim = model.dim()  # 25
    init_sample = np.ones(dim) * 0.01
    samplers = [
        AdaptiveHmc(model=model, num_samples_for_tuning=900,
                    init_sample=init_sample.copy(),
                    lambda_simulation_length=2., color='c', name='Adapt-HMC'),
        EffectiveNUTSSamplerWithDualAveraging(model=model, init_q=init_sample.copy(), num_adapt_iterations=900,
                                              max_possible_depth=10),
        AutoTuneChiFixedDistSampler(model=model, init_q=init_sample.copy(),
                                    num_iters1_eps_adapt=200,
                                    num_iters2_aux_sampler_tune_dist=500,
                                    sufficiently_large_D_param=10)
    ]
    german_experiment = Experiment(experiment_type="itr_based",
                                   model=model,
                                   init_sample=init_sample, num_samples=1000, num_burn_in_samples=200,
                                   mcmc_chains=['C{c}'.format(c=i) for i in range(0, 50)],
                                   samplers=samplers,
                                   path=path,
                                   reference_sampler=AutoTuneChiFixedDistSampler(
                                       model=model, init_q=-init_sample.copy(),
                                       num_iters1_eps_adapt=200,
                                       num_iters2_aux_sampler_tune_dist=1000,
                                       sufficiently_large_D_param=10,
                                       name='REFERENCE SAMPLER'),
                                   num_reference_samples=500000
                                   )
    run_save_samplers(german_experiment)
    calc_ess(german_experiment)


def main_SPECT(path):
    print('main_SPECT . . .')
    model = SpectBlrModel(sigma2=1)
    dim = model.dim()  # 25
    init_sample = np.ones(dim) * 0.01
    samplers = [
        AdaptiveHmc(model=model, num_samples_for_tuning=900,
                    init_sample=init_sample.copy(),
                    lambda_simulation_length=2., color='c', name='Adapt-HMC'),
        EffectiveNUTSSamplerWithDualAveraging(model=model, init_q=init_sample.copy(), num_adapt_iterations=900,
                                              max_possible_depth=10),
        AutoTuneChiFixedDistSampler(model=model, init_q=init_sample.copy(),
                                    num_iters1_eps_adapt=200,
                                    num_iters2_aux_sampler_tune_dist=500,
                                    sufficiently_large_D_param=10)
    ]
    spect_experiment = Experiment(experiment_type="itr_based",
                                  model=model,
                                  init_sample=init_sample, num_samples=1000, num_burn_in_samples=200,
                                  mcmc_chains=['C{c}'.format(c=i) for i in range(0, 50)],
                                  samplers=samplers,
                                  path=path,
                                  reference_sampler=AutoTuneChiFixedDistSampler(
                                      model=model, init_q=-init_sample.copy(),
                                      num_iters1_eps_adapt=200,  # 100,
                                      num_iters2_aux_sampler_tune_dist=1000,  # 200,
                                      sufficiently_large_D_param=10,
                                      name='REFERENCE SAMPLER'),
                                  num_reference_samples=500000
                                  )
    run_save_samplers(spect_experiment)
    calc_ess(spect_experiment)


class ReasonableInitalSampleDrawer:
    def __init__(self, model, num_trials):
        self.model = model
        self.num_trials = num_trials

    def find_a_reasonable_init_sample(self):
        least_found_u = np.inf
        chosen_q = np.full(shape=(self.model.dim()), fill_value=np.nan)
        for _ in tqdm(range(self.num_trials)):
            q = np.random.uniform(low=-.05, high=.05, size=self.model.dim())
            u = self.model.u(q)
            if u < least_found_u:
                least_found_u = u
                chosen_q = q
        assert np.isfinite(least_found_u)
        assert self.model.pr(q) > 0
        return chosen_q


def main_AusCr(path):
    print('main_AusCr . . .')
    model = AustralianCreditBlrModel(
        data_path='./data/australian.dat', max_data_points=100, sigma2=1.0)

    initial_sample_drawer_class = ReasonableInitalSampleDrawer(model=model, num_trials=50)
    init_sample_for_tuning = initial_sample_drawer_class.find_a_reasonable_init_sample()
    init_sample_callable = initial_sample_drawer_class.find_a_reasonable_init_sample

    samplers = [
        AdaptiveHmc(model=model, num_samples_for_tuning=900,
                    init_sample=init_sample_for_tuning.copy(),
                    lambda_simulation_length=2., color='c', name='Adapt-HMC'),
        EffectiveNUTSSamplerWithDualAveraging(model=model, init_q=init_sample_for_tuning.copy(),
                                              num_adapt_iterations=900,
                                              max_possible_depth=10),
        AutoTuneChiFixedDistSampler(model=model, init_q=init_sample_for_tuning.copy(),
                                    num_iters1_eps_adapt=200,
                                    num_iters2_aux_sampler_tune_dist=500,
                                    sufficiently_large_D_param=10)
    ]
    spect_experiment = Experiment(experiment_type="itr_based",
                                  model=model,
                                  init_sample=init_sample_callable, num_samples=1000, num_burn_in_samples=200,
                                  mcmc_chains=['C{c}'.format(c=i) for i in range(0, 50)],
                                  samplers=samplers,
                                  path=path,
                                  reference_sampler=AutoTuneChiFixedDistSampler(
                                      model=model, init_q=-init_sample_for_tuning.copy(),
                                      num_iters1_eps_adapt=200,
                                      num_iters2_aux_sampler_tune_dist=1000,
                                      sufficiently_large_D_param=10,
                                      name='REFERENCE SAMPLER'),
                                  num_reference_samples=500000
                                  )
    run_save_samplers(spect_experiment)
    calc_ess(spect_experiment)


def main_FNNL(path, dim):
    print('main_FNNL. . .(dim={d})'.format(d=dim))
    model = NealsFunnel(dim=dim, k=3)  # k=5.)

    init_sample_for_tuning = model.oracle_draw_sample()
    init_sample_callable = model.oracle_draw_sample

    class NealsFunnelOracleSampler(Sampler):
        def __init__(self, model):
            super().__init__(model=model, name='Oracle_NF', alias='ONF', color='k')
            assert type(model) == NealsFunnel

        def next_sample(self, current_sample):
            return model.oracle_draw_sample()

    samplers = [
        AdaptiveHmc(model=model, num_samples_for_tuning=900,
                    init_sample=init_sample_for_tuning.copy(),
                    lambda_simulation_length=1., color='c', name='Adapt-HMC'),
        EffectiveNUTSSamplerWithDualAveraging(model=model, init_q=init_sample_for_tuning.copy(),
                                              num_adapt_iterations=900,
                                              target_mean_acceptance_prob_delta=0.6, max_possible_depth=10),
        AutoTuneChiFixedDistSampler(model=model, init_q=init_sample_for_tuning.copy(),
                                    num_iters1_eps_adapt=200,
                                    num_iters2_aux_sampler_tune_dist=500,
                                    sufficiently_large_D_param=10)
    ]
    nf_experiment = Experiment(experiment_type="itr_based",
                               model=model,
                               init_sample=init_sample_callable, num_samples=1000, num_burn_in_samples=200,
                               mcmc_chains=['C{c}'.format(c=i) for i in range(0, 50)],
                               samplers=samplers,
                               path=path,
                               reference_sampler=NealsFunnelOracleSampler(
                                   model=model),
                               num_reference_samples=500000
                               )
    run_save_samplers(nf_experiment)
    calc_ess(nf_experiment)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MVN', 'FNNL', 'SPECT', 'GrCr', 'AusCr'])
    parser.add_argument('--dim', type=int, default=10)

    args = parser.parse_args()

    if args.model == 'MVN':
        main_MVN(path='./results/MVN', dim=args.dim)
    elif args.model == 'FNNL':
        main_FNNL(path='./results/FNNL', dim=args.dim)
    elif args.model == 'SPECT':
        main_SPECT(path='./results/SPECT')
    elif args.model == 'GrCr':
        main_GrCr(path='./results/GrCr')
    elif args.model == 'AusCr':
        main_AusCr(path='./results/AusCr')
    else:
        print('Unknown model: ', args.model)
