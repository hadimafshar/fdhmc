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
import scipy.stats as stat


def mean_lower_upper_confidence(data, confidece_percent=0.95):
    mean = np.mean(data)
    if len(data) <= 30:
        lower, upper = stat.t.interval(alpha=confidece_percent, df=len(data) - 1, loc=mean, scale=stat.sem(data))
    else:
        lower, upper = stat.norm.interval(alpha=confidece_percent, loc=mean, scale=stat.sem(data))
    return mean, lower, upper


def str_mean_confidence_interval(data, confidece_percent=0.95):
    m, l, u = mean_lower_upper_confidence(data, confidece_percent)
    return "{m}\t  $\pm$  {i}".format(m=m, i=u - m)


