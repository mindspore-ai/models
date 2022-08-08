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
"""hub config."""
from src.model_utils.config import config as cfg
from src.fibinet import PredictWithSigmoid, FiBiNetModel


def get_fibinet_net(config):
    """
    Get network of fibinet model.
    """
    fibinet_net = FiBiNetModel(config)
    eval_net = PredictWithSigmoid(fibinet_net)
    return eval_net

def create_network(name, *args, **kwargs):
    """create_network about fibinet"""
    if name == 'fibinet':
        eval_net = get_fibinet_net(cfg)
        return eval_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
