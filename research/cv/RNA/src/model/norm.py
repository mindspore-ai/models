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
"""norm for models"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class GroupBatchNorm2d(nn.Cell):
    def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(GroupBatchNorm2d, self).__init__()
        if affine:
            self.weight = ms.Parameter(ops.Ones(1, num_features, 1, 1))
            self.bias = ms.Parameter(ops.Zeros(1, num_features, 1, 1))
        else:
            self.weight, self.bias = None, None
        self.running_mean = ops.Zeros(num_groups)
        self.running_var = ops.Ones(num_groups)

        self.reset_parameters()

        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = momentum

    def reset_parameters(self):
        self.running_mean = ops.ZerosLike(self.running_mean)
        self.running_var = ops.Fill(ms.float32, self.running_var.shape, 1)

    def extra_repr(self):
        s = ('{num_groups}, {num_features}, eps={eps}'
             ', affine={affine}')
        return s.format(**self.__dict__)

    def construct(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        assert C % G == 0

        x_reshape = x.view(N, G, int(C/G), H, W)
        mean = x_reshape.mean([0, 2, 3, 4])
        var = x_reshape.var([0, 2, 3, 4])

        if self.training:
            tmp_mean = (self.momentum * self.running_mean) + (1.0 - self.momentum) * mean
            self.running_mean = tmp_mean.copy()
            tmp_var = (self.momentum * self.running_var) + (1.0 - self.momentum) * (N / (N - 1) * var)
            self.running_var = tmp_var.copy()
        else:
            mean = self.running_mean.copy()
            var = self.running_var.copy()

        # change shape
        current_mean = mean.view([1, G, 1, 1, 1]).expand_as(x_reshape)
        current_var = var.view([1, G, 1, 1, 1]).expand_as(x_reshape)
        x_reshape = (x_reshape-current_mean) / (current_var+self.eps).Sqrt()
        x = x_reshape.view(N, C, H, W)

        if self.affine:
            results = x * self.weight + self.bias
        else:
            results = x

        return results

class USNorm(nn.Cell):
    def __init__(self, num_features, norm_list):
        super(USNorm, self).__init__()
        self.num_features = num_features
        self.norm_list = norm_list

        # define list of normalizations
        normalization = []
        for item in self.norm_list:
            if item == 'bn':
                normalization.append(nn.BatchNorm2d(num_features))
            elif item == 'in':
                normalization.append(nn.InstanceNorm2d(num_features, affine=False))
            else:
                norm_type = item[:item.index('_')]
                num_group = int(item[item.index('_') + 1:])
                if 'gn' in norm_type:
                    if 'r' in norm_type:
                        if int(num_features/num_group) > num_features:
                            normalization.append(nn.GroupNorm(num_features, num_features))
                        else:
                            normalization.append(nn.GroupNorm(int(num_features/num_group), num_features))
                    else:
                        if int(num_features / num_group) > num_features:
                            normalization.append(nn.GroupNorm(num_features, num_features))
                        else:
                            normalization.append(nn.GroupNorm(num_group, num_features))
                elif 'gbn' in norm_type:
                    if 'r' in norm_type:
                        normalization.append(GroupBatchNorm2d(int(num_features/num_group), num_features))
                    else:
                        normalization.append(GroupBatchNorm2d(num_group, num_features))

        self.norm = nn.CellList(normalization)

        self.norm_type = None
        self.print = ops.Print()

    def set_norms(self, norm_type=None):
        self.norm_type = norm_type

    def set_norms_mixed(self):
        self.norm_type = np.random.choice(self.norm_list)

    def construct(self, x):
        if self.norm is None:
            assert self.norm is not None
        else:
            assert self.norm_type in self.norm_list
            idx = self.norm_list.index(self.norm_type)
        y = self.norm[idx](x)

        return y
