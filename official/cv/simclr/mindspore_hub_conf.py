# Copyright 2021 Huawei Technologies Co., Ltd
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
"""hub config"""
from src.simclr_model import SimCLR
from src.resnet import resnet50 as resnet

def simclr_net(base_net, *args, **kwargs):
    projection_dimension = kwargs.get("projection_dimension", 128)
    return SimCLR(base_net, projection_dimension, base_net.end_point.in_channels)

def create_network(name, *args, **kwargs):
    if name == "simclr":
        width_multiplier = kwargs.get("width_multiplier", 1)
        dataset = kwargs.get("dataset", "cifar10") == "cifar10"
        base_net = resnet(1, width_multiplier, cifar_stem=dataset)
        return simclr_net(base_net, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
