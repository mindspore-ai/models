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
'''loss'''
import mindspore.nn as nn
import mindspore.ops as ops


class MyLoss(nn.Cell):
    ''''Myloss'''
    def __init__(self):
        super(MyLoss, self).__init__()
        self.Norm = ops.L2Normalize(axis=1)
        self.criterion = nn.MSELoss(reduction='sum')

    def construct(self, fs_list, ft_list):
        '''construct'''
        tot_loss = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            _, _, h, w = fs.shape
            fs_norm = self.Norm(fs)
            ft_norm = self.Norm(ft)
            f_loss = (0.5 / (w * h)) * self.criterion(fs_norm, ft_norm)
            tot_loss += f_loss

        return tot_loss
