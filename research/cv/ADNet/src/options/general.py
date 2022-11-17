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
# source: https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/init_params.m

#parameter settings
show_visualization = 0
record_video = 0
GT_anno_interval = 1


# ============================
# NETWORK PARAMETERS
# ============================
opts = {
    'imgSize': [112, 112, 3],
    'train_dbs': ['vot15', 'vot14', 'vot13'],
    'test_db': 'otb',
    'train': {
        'weightDecay': 0.0005,
        'momentum': 0.9,
        'learningRate': 10e-5,
        'conserveMemory': True,
        'gt_skip': 1,
        'rl_num_batches': 5,
        'RL_steps': 10
    },
    'minibatch_size': 32,
    'numEpoch': 30,
    'numInnerEpoch': 3,
    'continueTrain': False,
    'samplePerFrame_large': 40,
    'samplePerFrame_small': 10,
    'inputSize': [112, 112, 3],
    'stopIou': 0.93,
    'meta': {
        'inputSize': [112, 112, 3]
    },
    'use_finetune': True,
    'scale_factor': 1.05,

    # test
    'finetune_iters': 20,
    'finetune_iters_online': 10,
    'finetune_interval': 30,
    'posThre_init': 0.7,
    'negThre_init': 0.3,
    'posThre_online': 0.7,
    'negThre_online': 0.5,
    'nPos_init': 200,
    'nNeg_init': 150,
    'nPos_online': 30,
    'nNeg_online': 15,
    'finetune_scale_factor': 3.0,
    'redet_scale_factor': 3.0,
    'finetune_trans': 0.10,
    'redet_samples': 256,

    'successThre': 0.5,
    'failedThre': 0.5,

    'nFrames_long': 100, # long-term period (in matlab code, for positive samples... while current implementation just with history for now...)
    'nFrames_short': 20, # short-term period (for negative samples)

    'nPos_train': 150,
    'nNeg_train': 50,
    'posThre_train': 0.5,
    'negThre_train': 0.3,

    'random_perturb': {
        'x': 0.15,
        'y': 0.15,
        'w': 0.03,
        'h': 0.03
    },

    'action_move': {
        'x': 0.03,
        'y': 0.03,
        'w': 0.03,
        'h': 0.03,
        'deltas': [
            [-1, 0, 0, 0],  # left
            [-2, 0, 0, 0],  # left x2
            [+1, 0, 0, 0],  # right
            [+2, 0, 0, 0],  # right x2
            [0, -1, 0, 0],  # up
            [0, -2, 0, 0],  # up x2
            [0, +1, 0, 0],  # down
            [0, +2, 0, 0],  # down x2
            [0, 0, 0, 0],  # stop
            [0, 0, -1, -1],  # smaller
            [0, 0, +1, +1]   # bigger
        ]
    },

    'num_actions': 11,
    'stop_action': 8,
    'num_show_actions': 20,
    'num_action_step_max': 20,
    'num_action_history': 10,

    'visualize': True,
    'printscreen': True,

    'means': [104, 117, 123]  # https://github.com/amdegroot/ssd.pytorch/blob/8dd38657a3b1df98df26cf18be9671647905c2a0/data/config.py

}
