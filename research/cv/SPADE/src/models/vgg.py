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
""" Vgg19 network """

from mindspore import nn, load_checkpoint, load_param_into_net

class Vgg19(nn.Cell):
    def __init__(self, opt, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = self.getVggFeatures()
        save_path = opt.vgg_ckpt_path
        param_dict = load_checkpoint(save_path)
        load_param_into_net(vgg_pretrained_features, param_dict)
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        list5 = []
        for x in range(2):
            list1.append(vgg_pretrained_features[x])
        for x in range(2, 7):
            list2.append(vgg_pretrained_features[x])
        for x in range(7, 12):
            list3.append(vgg_pretrained_features[x])
        for x in range(12, 21):
            list4.append(vgg_pretrained_features[x])
        for x in range(21, 30):
            list5.append(vgg_pretrained_features[x])

        self.slice1 = nn.SequentialCell(list1)
        self.slice2 = nn.SequentialCell(list2)
        self.slice3 = nn.SequentialCell(list3)
        self.slice4 = nn.SequentialCell(list4)
        self.slice5 = nn.SequentialCell(list5)

    def construct(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def getVggFeatures(self):
        return nn.SequentialCell([
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
