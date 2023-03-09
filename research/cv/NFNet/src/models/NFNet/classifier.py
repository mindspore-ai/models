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
"""
Classifier head and layer factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from mindspore import nn
from mindspore import ops


def create_pool(pool_type='avg'):
    assert pool_type in ["avg", "max"]
    if pool_type == 'avg':
        global_pool = ops.ReduceMean(keep_dims=False)
    elif pool_type == 'max':
        global_pool = ops.ReduceMax(keep_dims=False)
    return global_pool


class ClassifierHead(nn.Cell):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = create_pool(pool_type)
        self.drop_out = nn.Dropout(p=self.drop_rate) if self.drop_rate > 0. else ops.Identity()
        self.fc = nn.Dense(in_chs, num_classes)

    def construct(self, x):
        x = self.global_pool(x, (2, 3))
        x = self.drop_out(x)
        x = self.fc(x)
        return x
