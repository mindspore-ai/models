# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
python initializer.py
"""
from mindspore.common.initializer import One, Zero, Normal, Uniform

def initializer(cfg):
    """
    return a initializer

    Args:
        cfg(dict): configuration

    Returns:
        initializer
    """
    init = None
    if cfg['name'] == 'one':
        init = One()
    elif cfg['name'] == 'zero':
        init = Zero()
    elif cfg['name'] == 'normal':
        sigma = cfg['sigma'] if 'sigma' in cfg else 0.01
        mean = cfg['mean'] if 'mean' in cfg else 0
        init = Normal(sigma, mean)
    elif cfg['name'] == 'uniform':
        scale = cfg['scale'] if 'scale' in cfg else 0.07
        init = Uniform(scale)
    else:
        print('unsupported initializer.')

    return init
