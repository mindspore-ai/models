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
"""hub config."""
from src.vgg import vgg19 as VGG19
from src.config import cifar_cfg, imagenet_cfg



def vgg19(*args, **kwargs):
    return VGG19(*args, **kwargs)


def create_network(name, *args, **kwargs):
    """create_network about vgg19_net"""
    if name == "vgg19":
        dataset = kwargs.get("dataset", "cifar10")
        if "dataset" in kwargs:
            del kwargs["dataset"]
        if dataset == "cifar10":
            net = vgg19(args=cifar_cfg)
        elif dataset == "imagenet2012":
            net = vgg19(args=imagenet_cfg)
        else:
            raise NotImplementedError(f"dataset {dataset} is not implemented in the repo")
        return net
    raise NotImplementedError(f"{name} is not implemented in the repo")
