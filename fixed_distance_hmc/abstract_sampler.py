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

INFO_LEAPFROG_PS = 'info.leapfrog.ps'

INFO_LEAPFROG_QS = 'info.leapfrog.qs'

INFO_NUM_GRADS = 'info.num.grads'

INFO_ACCEPT_PROB = 'info.accept.prob'

INFO_INIT_EVOLVE_Q = 'info.fd.init.evolve.q'

INFO_FINAL_EVOLVE_Q = 'info.fd.final.evolve.q'


class Sampler:
    def __init__(self, model, name, alias, color='k'):
        self.model = model
        self._name = name
        self.alias = alias
        self.color = color
        self._activate = False

    def activate(self):
        self._activate = True

    def next_sample(self, current_sample):
        if not self._activate:
            raise Exception('the sampler should be activated before the first sample is taken')

        sample, info = self.next_sample_info(current_sample=current_sample)
        return sample

    def next_sample_info(self, current_sample):
        if not self._activate:
            raise Exception('the sampler should be activated before the first sample is taken')

        raise Exception('not implemented')

    def name(self):
        return self._name
