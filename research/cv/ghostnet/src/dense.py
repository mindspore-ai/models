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
"""Adjust the fully connected layer and load the pretrained network"""

import mindspore as ms

from src.config import config


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""

    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def init_weight(net, param_dict):
    """init_weight"""

    # if config.pre_trained:
    if param_dict:
        if config.filter_weight:
            filter_list = [x.name for x in net.classifier.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        ms.load_param_into_net(net, param_dict)
