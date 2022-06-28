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
"""hyper-parameters."""

from easydict import EasyDict as edict
import mindspore as ms

SEQ_LEN = 512

soft_masked_bert_cfg = edict({
    'model': edict({
        'bert_ckpt': 'bert-base-chinese',
        'device': 'Ascend',
        'name': 'SoftMaskedBertModel',
        'gpu_ids': [0],
        'hyper_params': [0.8]
    }),
    'dataset': edict({
        'train': 'datasets/csc/train.json',
        'valid': 'datasets/csc/dev.json',
        'test': 'datasets/csc/test.json'
    }),
    'solver': edict({
        'base_lr': 0.0001,
        'weight_decay': 5e-8,
        'batch_size': 4,
        'max_epoch': 10,
        'accumulate_grad_batches': 4
    }),
    'test': edict({
        'batch_size': 16
    }),
    'task': edict({
        'name': 'csc'
    }),
    'output_dir': 'checkpoints/SoftMaskedBert'
})

optimizer_cfg = edict({
    'batch_size': 2,
    'optimizer': 'AdamWeightDecay',
    'AdamWeightDecay': edict({
        'learning_rate': 2e-5,
        'end_learning_rate': 1e-7,
        'power': 1.0,
        'weight_decay': 1e-5,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
        'eps': 1e-6,
    }),
    'Lamb': edict({
        'learning_rate': 2e-5,
        'end_learning_rate': 1e-7,
        'power': 1.0,
        'weight_decay': 0.01,
        'decay_filter': lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
    }),
    'Momentum': edict({
        'learning_rate': 2e-5,
        'momentum': 0.9,
    }),
})

bert_cfg = edict({
    'seq_length': SEQ_LEN, #128
    'vocab_size': 21128,
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'hidden_act': "gelu",
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': SEQ_LEN,
    'type_vocab_size': 2,
    'initializer_range': 0.02,
    'use_relative_positions': False,
    'dtype': ms.float32,
    'compute_type': ms.float32,
    'pad_token_id': 0,
    'layer_norm_eps': 1e-12
    })

gru_cfg = edict({
    'encoder_embedding_size': 768,
    'hidden_size': 384,
    'max_length': SEQ_LEN,
    'is_training': True
})
