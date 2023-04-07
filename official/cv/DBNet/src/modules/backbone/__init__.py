# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
from .resnet import ResNet, Bottleneck, BasicBlock
from .mobilenetv3 import MobileNetV3


def mobilenetv3(pretrained=True, backbone_ckpt=None, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        ms.load_checkpoint(backbone_ckpt, model)
    return model


def resnet18(pretrained=True, backbone_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ms.load_checkpoint(backbone_ckpt, model)
    return model


def deformable_resnet18(pretrained=True, backbone_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=True, **kwargs)
    if pretrained:
        ms.load_checkpoint(backbone_ckpt, model)
    return model


def resnet50(pretrained=True, backbone_ckpt=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ms.load_checkpoint(backbone_ckpt, model)
    return model


def deformable_resnet50(pretrained=True, backbone_ckpt=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=True, **kwargs)
    if pretrained:
        ms.load_checkpoint(backbone_ckpt, model)
    return model


def get_backbone(initializer):
    backbone_dict = {
        "mobilenetv3": mobilenetv3,
        "resnet18": resnet18,
        "deformable_resnet18": deformable_resnet18,
        "resnet50": resnet50,
        "deformable_resnet50": deformable_resnet50,
    }
    if initializer not in backbone_dict:
        raise ValueError(f"get_backbone initializer {initializer} is not supported.")
    return backbone_dict[initializer]
