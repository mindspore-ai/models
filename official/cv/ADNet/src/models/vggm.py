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
from __future__ import print_function, division, absolute_import

from mindspore import nn, ops
from mindspore import load_checkpoint, load_param_into_net
# source: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/vggm.py

__all__ = ['vggm']

pretrained_settings = {
    'vggm': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth',
            'input_space': 'BGR',
            'input_size': [3, 221, 221],
            'input_range': [0, 255],
            'mean': [123.68, 116.779, 103.939],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}

class VGGM(nn.Cell):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.ops_LRN = wrapper_LRN()
        self.features = nn.SequentialCell([
            nn.Conv2d(3, 96, (7, 7), (2, 2)),  # conv1
            nn.ReLU(),
            self.ops_LRN,
            nn.MaxPool2d((3, 3), (2, 2), 'valid'),
            nn.Conv2d(96, 256, (5, 5), (2, 2), 'pad', 1),  # conv2
            nn.ReLU(),
            self.ops_LRN,
            nn.MaxPool2d((3, 3), (2, 2), 'valid'),
            nn.Conv2d(256, 512, (3, 3), (1, 1), 'pad', 1),  # conv3
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 'pad', 1),  # conv4
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 'pad', 1),  # conv5
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), 'valid')
        ])
        self.classifier = nn.SequentialCell([
            nn.Dense(18432, 4096),  # 18432 = 4 * 3 * 3 * 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(4096, num_classes)
        ])

    def construct(self, x):
        x = self.features(x)
        # x = x.(x.size(0), -1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def vggm(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['vggm'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        import os
        model = VGGM(num_classes=num_classes)
        load_param_into_net(model, (load_checkpoint(os.path.dirname(__file__)+'/vggm.ckpt')))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = VGGM(num_classes=num_classes)
    return model


class wrapper_LRN(nn.Cell):
    def __init__(self, depth_=5, bias=2., alpha=5e-4, beta=0.75):
        super(wrapper_LRN, self).__init__()
        self.lrn = ops.LRN(depth_radius=depth_, bias=bias, alpha=alpha, beta=beta)

    def construct(self, x):
        return self.lrn(x)
