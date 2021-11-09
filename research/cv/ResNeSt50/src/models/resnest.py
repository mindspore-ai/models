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
"""ResNeSt models"""
from mindspore.train.serialization import load_param_into_net

from src.models.resnet import ResNet, Bottleneck
from src.models.utils import Resume

__all__ = ['resnest50', 'resnest101', 'resnest200']

def resnest50(pretrained=False, root='~/output/ckpt_0/resnest50-270_2502.ckpt', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        param = Resume(model, root)
        load_param_into_net(model, param)
    return model

def resnest101(pretrained=False, root='~/output/ckpt_0/resnest101-270_2502.ckpt', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        param = Resume(model, root)
        load_param_into_net(model, param)
    return model

def resnest200(pretrained=False, root='~/output/ckpt_0/resnest200-270_2502.ckpt', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        param = Resume(model, root)
        load_param_into_net(model, param)
    return model

def get_network(net_name, resume=False, resume_path='./output/ckpt_0/resnest50-270_2502.ckpt', **kwargs):
    """Get the network from ['resnest50', 'resnest101', 'resnest200']"""
    if net_name not in ['resnest50', 'resnest101', 'resnest200']:
        raise NotImplementedError(f"The network {net_name} not in [resnest50, resnest101, resnest200].")

    net = resnest50()

    if resume:
        if net_name == 'resnest50':
            net = resnest50(pretrained=True, root=resume_path, **kwargs)
        elif net_name == 'resnest101':
            net = resnest101(pretrained=True, root=resume_path, **kwargs)
        else:
            net = resnest200(pretrained=True, root=resume_path, **kwargs)
    else:
        if net_name == 'resnest50':
            net = resnest50(pretrained=False, **kwargs)
        elif net_name == 'resnest101':
            net = resnest101(pretrained=False, **kwargs)
        else:
            net = resnest200(pretrained=False, **kwargs)
    return net
