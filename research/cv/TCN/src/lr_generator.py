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
######################## dynamic learning rate ########################
"""


def get_lr(config, base_step):
    """dynamic learning rate generator"""
    base_lr = config.lr
    lr = []
    now_lr = base_lr
    for i in range(config.epoch_size):
        if i % config.epoch_change == 0 and i != 0:
            now_lr = now_lr / 10
        for _ in range(base_step):
            lr.append(now_lr)
    return lr
