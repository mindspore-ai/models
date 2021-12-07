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
"""Util class or function."""
import numpy as np

def process_trainable_params(trainable_params):
    """process trainable params"""
    group1 = []
    group2 = []
    group3 = []
    group4 = []
    group5 = []
    for x in trainable_params:
        if x.name == "base_model.conv1_7x7_s2.conv1_7x7_s2.weight":
            group1.append(x)
        elif x.name == "base_model.conv1_7x7_s2.conv1_7x7_s2.bias":
            group2.append(x)
        elif "bn" in x.name:
            group5.append(x)
        else:
            if "weight" in x.name:
                group3.append(x)
            else:
                group4.append(x)
    return group1, group2, group3, group4, group5

def get_lr(learning_rate, gamma, epochs, steps_per_epoch, lr_steps):
    """generate lr"""
    lr_each_step = []
    base_lr = learning_rate
    for epoch in range(1, epochs+1):
        #decay = gamma ** (sum(epoch >= np.array(lr_steps)))
        #base_lr = base_lr * decay
        if epoch%lr_steps == 0:
            base_lr = base_lr*gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(base_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step
