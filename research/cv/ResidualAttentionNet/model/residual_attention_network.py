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
import mindspore.nn as nn
import mindspore.ops.operations as P
from src.conv2d_ops import _conv3x3_pad, _conv7x7_pad, _fc
from model.basic_layers import ResidualBlock
from model.attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from model.attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar

class ResidualAttentionModel_448input(nn.Cell):
    # for input size 448
    def __init__(self):
        super(ResidualAttentionModel_448input, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                      pad_mode='pad', padding=3, has_bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU()
        ])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128)
        self.residual_block1 = ResidualBlock(128, 256, 2)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.SequentialCell([
            nn.BatchNorm2d(num_features=2048, momentum=0.9),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')
        ])
        self.fc = nn.Dense(in_channels=2048, out_channels=10)

    def construct(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.fc(out)
        return out

class ResidualAttentionModel_92(nn.Cell):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_92, self).__init__()
        self.conv1 = nn.SequentialCell([
            _conv7x7_pad(in_channels=3, out_channels=64, stride=2),
            nn.BatchNorm2d(num_features=64, eps=1.0e-05, momentum=0.1),
            nn.ReLU()
        ])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.SequentialCell([
            nn.BatchNorm2d(num_features=2048, eps=1.0e-05, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')
        ])
        self.fc = _fc(2048, 1000)

    def construct(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.fc(out)
        return out

class ResidualAttentionModel_56(nn.Cell):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_56, self).__init__()
        self.conv1 = nn.SequentialCell([
            _conv7x7_pad(in_channels=3, out_channels=64, stride=2),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU()
        ])
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.SequentialCell([
            nn.BatchNorm2d(num_features=2048, momentum=0.9),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')
        ])
        self.fc = _fc(2048, 1000)

    def construct(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.fc(out)
        return out

class ResidualAttentionModel_92_32input(nn.Cell):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionModel_92_32input, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                      stride=1, pad_mode='pad', padding=2, has_bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU()
        ])
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 16*16
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128)  # 16*16
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 8*8
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256)  # 8*8
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256)  # 8*8
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.residual_block4 = ResidualBlock(512, 1024)  # 4*4
        self.residual_block5 = ResidualBlock(1024, 1024)  # 4*4
        self.residual_block6 = ResidualBlock(1024, 1024)  # 4*4
        self.mpool2 = nn.SequentialCell([
            nn.BatchNorm2d(num_features=1024, momentum=0.9),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=1, pad_mode='valid')
        ])
        self.fc = nn.Dense(in_channels=1024, out_channels=10)

    def construct(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.fc(out)
        return out

class ResidualAttentionModel_92_32input_cifar100_update(nn.Cell):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionModel_92_32input_cifar100_update, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      stride=1, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU()
        ])  # 32*32
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.SequentialCell([
            nn.BatchNorm2d(num_features=1024, momentum=0.9),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=8, pad_mode='valid')
        ])
        self.fc = nn.Dense(in_channels=1024, out_channels=100)

    def construct(self, x):
        out = self.conv1(x)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.fc(out)
        return out

class ResidualAttentionModel_92_32input_update(nn.Cell):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionModel_92_32input_update, self).__init__()
        self.conv1 = nn.SequentialCell([
            _conv3x3_pad(in_channels=3, out_channels=32, stride=1),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU()
        ])
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.SequentialCell([
            nn.BatchNorm2d(num_features=1024, momentum=0.9),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=8, pad_mode='valid')
        ])
        self.fc = _fc(1024, 10)

    def construct(self, x):
        out = self.conv1(x)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.fc(out)
        return out
