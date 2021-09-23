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
from src.resnetv2 import PreActResNet50 as resnetv2_50
from src.resnetv2 import PreActResNet101 as resnetv2_101
from src.resnetv2 import PreActResNet152 as resnetv2_152
from src.config import config1 as config_cifar10
from src.config import config2 as config_cifar100
from src.config import config3 as config_imagenet2012

def create_network(name, *args, **kwargs):
    """create_network about resnetv2_net"""
    dataset = kwargs.get("dataset", "cifar10")
    if name == "resnetv2_101":
        if dataset == "imagent2012":
            net = resnetv2_101(config_imagenet2012.class_num, config_imagenet2012.low_memory)
        elif dataset == "cifar10":
            net = resnetv2_101(config_cifar10.class_num, config_cifar10.low_memory)
        elif dataset == "cifar100":
            net = resnetv2_101(config_cifar100.class_num, config_cifar100.low_memory)
        else:
            raise NotImplementedError(f"dataset is not implemented in the repo")
    elif name == "resnetv2_152":
        if dataset == "imagent2012":
            net = resnetv2_152(config_imagenet2012.class_num, config_imagenet2012.low_memory)
        elif dataset == "cifar10":
            net = resnetv2_152(config_cifar10.class_num, config_cifar10.low_memory)
        elif dataset == "cifar100":
            net = resnetv2_152(config_cifar100.class_num, config_cifar100.low_memory)
        else:
            raise NotImplementedError(f"dataset is not implemented in the repo")
    elif name == "resnetv2_50":
        if dataset == "imagent2012":
            net = resnetv2_50(config_imagenet2012.class_num, config_imagenet2012.low_memory)
        elif dataset == "cifar10":
            net = resnetv2_50(config_cifar10.class_num, config_cifar10.low_memory)
        elif dataset == "cifar100":
            net = resnetv2_50(config_cifar100.class_num, config_cifar100.low_memory)
        else:
            raise NotImplementedError(f"dataset is not implemented in the repo")
    else:
        raise NotImplementedError(f"{name} is not implemented in the repo")
    return net
