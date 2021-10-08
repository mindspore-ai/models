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
# returns action history as one-hot form
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_action_history_onehot.m
import mindspore.numpy as nps

def get_action_history_onehot(action_history, opts):
    onehot = nps.zeros((opts['num_actions'] * len(action_history),))
    for i in range(len(action_history)):
        start_idx = i * opts['num_actions']
        if action_history[i] >= 0 and action_history[i] < opts['num_actions']:
            onehot[start_idx + action_history[i]] = 1.
    return onehot
