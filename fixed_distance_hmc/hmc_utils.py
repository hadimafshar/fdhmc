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


def hamiltonian(model, q, p):
    return 0.5 * p.dot(p) + model.u(q)


def jump_prob(model, q_prim, p_prim, q_curr, p_curr):
    h_prim = hamiltonian(model=model, q=q_prim, p=p_prim)
    h_curr = hamiltonian(model=model, q=q_curr, p=p_curr)
    a = np.exp(-h_prim + h_curr)
    assert np.isfinite(a)
    return np.min([1., a])


def find_reasonable_epsilon(model, q, p_with_expected_magnitude):
    eps = 1.0
    p = p_with_expected_magnitude

    (q_prim, p_prim) = leapfrog(model=model, q=q, p=p, time=eps)

    # if prob(model, q_prim, p_prim) / prob_qp > 0.5:
    if jump_prob(model=model, q_prim=q_prim, p_prim=p_prim, q_curr=q, p_curr=p) > 0.5:
        while True:
            eps *= 2.
            (q_prim, p_prim) = leapfrog(model=model, q=q, p=p, time=eps)
            # if prob(model, q_prim, p_prim) / prob_qp <= 0.5:
            if jump_prob(model=model, q_prim=q_prim, p_prim=p_prim, q_curr=q, p_curr=p) <= 0.5:
                break
    else:  # prob(model, q_prim, p_prim) / prob_qp < 0.5:
        while True:
            eps /= 2.
            print('eps:', eps)
            (q_prim, p_prim) = leapfrog(model=model, q=q, p=p, time=eps)
            # if prob(model, q_prim, p_prim) / prob_qp >= 0.5:
            if jump_prob(model=model, q_prim=q_prim, p_prim=p_prim, q_curr=q, p_curr=p) >= 0.5:
                break

    assert eps != 1.0  # if a nan happens this may be the case
    return eps


def leapfrog(model, q, p, time):
    q = q.copy()
    p = p.copy()
    # half-step:
    p -= 0.5 * time * model.grad_u(q=q)
    assert not np.isnan(p).any()

    # full-step:
    q += time * p
    assert not np.isnan(q).any()

    # half-step:
    p -= 0.5 * time * model.grad_u(q=q)
    assert not np.isnan(p).any()
    return q, p


def main():
    pass


if __name__ == "__main__":
    main()
