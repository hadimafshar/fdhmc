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

import numpy as np
from fixed_distance_hmc.models.abstract_model import Model


class DataSetParser:
    def fetch_Xy(self):
        raise


class BayesianLogisticRegressionModel(Model):
    def dim(self):
        # alpha + [beta_1, ... beta_n] where n is the number of features
        return self.X.shape[1] + 1

    def __init__(self,
                 dataset_parser: DataSetParser,
                 sigma2=100.):
        self.sigma2 = sigma2
        self.X, self.y = dataset_parser.fetch_Xy()

        # normalize:
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

        # y should only contain -1, 1:
        assert min(self.y) == -1
        assert max(self.y) == 1

        self.NUM_DATA = self.X.shape[0]
        assert self.y.shape == (self.NUM_DATA,)

    def pr(self, q):
        assert q.shape[0] == self.dim()
        return np.exp(-self.u(q))

    def u(self, q):
        assert np.isfinite(q).all()
        alpha = q[0]
        beta = q[1:]
        u = 0.5 * alpha * alpha / self.sigma2 + 0.5 * beta.dot(beta) / self.sigma2
        u += (np.log(
            1 + np.exp(
                -self.y * (alpha + self.X.dot(beta))
            )
        )).sum()
        assert np.isfinite(u)
        return u

    def grad_u(self, q):
        alpha = q[0]
        beta = q[1:]
        partial_u_alpha = alpha / self.sigma2
        partial_u_beta = beta / self.sigma2
        log_inv_times_exp = 1 / (
                1 + np.exp(-self.y * (alpha + self.X.dot(beta)))
        ) * np.exp(
            -self.y * (alpha + self.X.dot(beta))
        )
        assert log_inv_times_exp.shape == (self.NUM_DATA,)
        if np.isnan(log_inv_times_exp).any():
            np.nan_to_num(log_inv_times_exp, copy=False, nan=1.0)

        assert np.isfinite(log_inv_times_exp).all()

        partial_u_alpha += log_inv_times_exp.dot(-self.y)
        partial_u_beta += (log_inv_times_exp * (-self.y)).dot(self.X)

        return np.concatenate((np.array([partial_u_alpha]), partial_u_beta))

    @staticmethod
    def name():
        return 'BLR'


class SPECTdatasetParser(DataSetParser):
    def __init__(self,
                 data_path='./data/SPECT.train'):
        """
            Data comes from  https://archive.ics.uci.edu/ml/machine-learning-databases/spect/
        """

        with open(data_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 80
            num_columns = len(lines[0].split(','))
            data = np.full(shape=(len(lines), num_columns), fill_value=np.nan, dtype=np.float64)
            for i, l in enumerate(lines):
                data[i] = np.array([x for x in l.split(',')])

        self.X = data[:, 1:]

        # the first column is y:
        self.y = data[:, 0]
        # Y should contain 1 and -1:
        self.y = np.array([1. if e == 1. else -1. for e in self.y])

    def fetch_Xy(self):
        return self.X, self.y


class AustralianCreditParser(DataSetParser):
    def __init__(self,
                 data_path='./data/australian.dat',
                 max_data_points=None):
        """
            Data comes from  https://archive.ics.uci.edu/ml/machine-learning-databases/spect/
        """

        with open(data_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 690
            num_columns = len(lines[0].split())
            data = np.full(shape=(len(lines), num_columns), fill_value=np.nan, dtype=np.float64)
            for i, l in enumerate(lines):
                data[i] = np.array([x for x in l.split()])

        if max_data_points is not None:
            data = data[:max_data_points, :]

        self.X = data[:, :-1]

        # the last column is y:
        self.y = data[:, -1]
        # Y should contain 1 and -1:
        self.y = np.array([1. if e == 1. else -1. for e in self.y])

    def fetch_Xy(self):
        return self.X, self.y


class GermanCreditParser(DataSetParser):
    def __init__(self,
                 data_path='./data/german.data-numeric'):
        self.NUM_DATA = 1000
        """
            Data comes from  https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
        """

        data = np.full(shape=(self.NUM_DATA, 25), fill_value=np.nan, dtype=np.float64)
        with open(data_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == self.NUM_DATA
            for i, l in enumerate(lines):
                data[i] = np.array([x for x in l.split()])

        self.X = data[:, 0:-1]

        self.y = data[:, -1]
        # (1 = Good --> 1, 2 = Bad/should be denied --> -1)
        self.y = np.array([1. if e == 1. else -1. for e in self.y])

    def fetch_Xy(self):
        return self.X, self.y


class SpectBlrModel(BayesianLogisticRegressionModel):
    def __init__(self,
                 data_path='./data/SPECT.train',
                 sigma2=100):
        super().__init__(dataset_parser=SPECTdatasetParser(data_path=data_path), sigma2=sigma2)

    @staticmethod
    def name():
        return 'SpectCredit'


class GermanCreditBlrModel(BayesianLogisticRegressionModel):
    def __init__(self,
                 data_path='./data/german.data-numeric',
                 sigma2=100):
        super().__init__(dataset_parser=GermanCreditParser(data_path=data_path), sigma2=sigma2)

    @staticmethod
    def name():
        return 'GermanCredit'


class AustralianCreditBlrModel(BayesianLogisticRegressionModel):
    def __init__(self,
                 data_path='./data/australian.dat',
                 max_data_points=None,
                 sigma2=100):
        super().__init__(dataset_parser=AustralianCreditParser(data_path=data_path, max_data_points=max_data_points, ),
                         sigma2=sigma2)

    @staticmethod
    def name():
        return 'AustralianCredit'
