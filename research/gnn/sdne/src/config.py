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
"""
python config.py
"""
from easydict import EasyDict

cfg = EasyDict({
    'is_modelarts': False,
    'WIKI': {
        'batch': 3000,
        'delimiter': ' ',
        'hidden_size': [256, 128],
        'node_size': 2405,
        'weight_init': {
            'name': 'uniform',
        },
        'optim': {
            'name': 'Adam',
            'learning_rate': 0.002,
            'weight_decay': 0.0001,
        },
        'act': 'relu',
        'alpha': 1,
        'beta': 5,
        'gamma': 1.0e-6,
        'ckpt_step': 32,
        'ckpt_max': 10,
        'generate_emb': False,
        'data_path': 'Wiki_edgelist.txt',
        'label_path': 'wiki_labels.txt',
        'reconstruction': {
            'check': True,
            'k_query': [1, 10, 20, 100, 200, 1000, 2000, 6000, 8000, 10000],
        },
        'classify': {
            'check': False,
            'tr_frac': 0.8,
        },
    },
    'GRQC': {
        'batch': 32,
        'delimiter': ' ',
        'hidden_size': [100],
        'node_size': 5242,
        'weight_init': {
            'name': 'normal',
            'sigma': 1,
        },
        'optim': {
            'name': 'RMSProp',
            'learning_rate': 0.01,
            'weight_decay': 1,
        },
        'act': 'sigmoid',
        'alpha': 100,
        'beta': 10,
        'gamma': 1,
        'ckpt_step': 32,
        'ckpt_max': 1,
        'generate_emb': False,
        'data_path': 'ca-Grqc.txt',
        'label_path': '',
        'reconstruction': {
            'check': True,
            'k_query': [10, 100],
        },
        'classify': {
            'check': False,
        },
    },
})
