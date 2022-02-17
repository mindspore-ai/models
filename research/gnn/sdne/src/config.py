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
python config.py
"""
from easydict import EasyDict

cfg = EasyDict({
    'is_modelarts': False,
    'WIKI': {
        'batch': 3000,
        'delimiter': ' ',
        'node_size': 2405,
        'hidden_size': [256, 128],
        'weight_init': 'uniform',
        'learning_rate': 0.002,
        'weight_decay': 1.0e-4,
        'alpha': 1.0e-6,
        'beta': 5,
        'dataset_path': './Wiki_edgelist.txt',
        'ckpt_step': 32,
        'ckpt_max': 10,
        'generate_emb': True,
        'generate_rec': False,
        'eval': {
            'frac': 1,
            'use_rand': False,
            'k_query': [1, 10, 20, 100, 200, 1000, 2000, 6000, 8000, 10000],
        },
        'classify': {
            'label': './wiki_labels.txt',
            'skip_head': False,
            'has_index': True,
            'tr_frac': 0.8,
        },
        'linkpred': None,
    },
    'BLOGCATALOG': {
        'batch': 2048,
        'delimiter': ',',
        'hidden_size': [10300, 1000, 100],
        'weight_init': 'uniform',
        'learning_rate': 0.01,
        'weight_decay': 1.0e-4,
        'alpha': 2,
        'beta': 2,
        'dataset_path': './edges.csv',
        'ckpt_step': 32,
        'ckpt_max': 10,
        'generate_emb': True,
        'generate_rec': False,
        'eval': {
            'frac': 0.3,
            'use_rand': True,
            'k_query': [1, 10, 20, 100, 200, 1000, 2000, 6000, 8000, 10000],
        },
        'classify': {
            'label': './group-edges.csv',
            'skip_head': False,
            'has_index': True,
            'tr_frac': 0.8,
        },
        'linkpred': None,
    },
})
