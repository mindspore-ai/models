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
Process the .npz checkpoint to .ckpt format.
For ViT-base-16 only.
"""

import mindspore as ms
import numpy as np


def extract_encoder_weights(ref_ws, index):
    """extract weights from encoder layers and transform shape"""
    src_prefix = f'Transformer/encoderblock_{index}/'
    tgt_prefix = f'transformer.encoder.layer.{index}.'

    tgt_ws = {}

    # Attention
    src_att_name = src_prefix + 'MultiHeadDotProductAttention_1/'
    tgt_att_name = tgt_prefix + 'attn.'

    tgt_ws[tgt_att_name + 'query.weight'] = ref_ws[src_att_name + 'query/kernel'].reshape(768, 768).T
    tgt_ws[tgt_att_name + 'key.weight'] = ref_ws[src_att_name + 'key/kernel'].reshape(768, 768).T
    tgt_ws[tgt_att_name + 'value.weight'] = ref_ws[src_att_name + 'value/kernel'].reshape(768, 768).T
    tgt_ws[tgt_att_name + 'out.weight'] = ref_ws[src_att_name + 'out/kernel'].reshape(768, 768).T

    tgt_ws[tgt_att_name + 'query.bias'] = ref_ws[src_att_name + 'query/bias'].reshape(768)
    tgt_ws[tgt_att_name + 'key.bias'] = ref_ws[src_att_name + 'key/bias'].reshape(768)
    tgt_ws[tgt_att_name + 'value.bias'] = ref_ws[src_att_name + 'value/bias'].reshape(768)
    tgt_ws[tgt_att_name + 'out.bias'] = ref_ws[src_att_name + 'out/bias']

    tgt_ws[tgt_prefix + 'attention_norm.gamma'] = ref_ws[src_prefix + 'LayerNorm_0/scale']
    tgt_ws[tgt_prefix + 'attention_norm.beta'] = ref_ws[src_prefix + 'LayerNorm_0/bias']

    # Feed forward
    tgt_ws[tgt_prefix + 'ffn_norm.gamma'] = ref_ws[src_prefix + 'LayerNorm_2/scale']
    tgt_ws[tgt_prefix + 'ffn_norm.beta'] = ref_ws[src_prefix + 'LayerNorm_2/bias']

    tgt_ws[tgt_prefix + 'ffn.fc1.weight'] = ref_ws[src_prefix + 'MlpBlock_3/Dense_0/kernel'].T
    tgt_ws[tgt_prefix + 'ffn.fc1.bias'] = ref_ws[src_prefix + 'MlpBlock_3/Dense_0/bias']
    tgt_ws[tgt_prefix + 'ffn.fc2.weight'] = ref_ws[src_prefix + 'MlpBlock_3/Dense_1/kernel'].T
    tgt_ws[tgt_prefix + 'ffn.fc2.bias'] = ref_ws[src_prefix + 'MlpBlock_3/Dense_1/bias']

    return tgt_ws


def extract_embeddings(ref_ws):
    """extract weights from embeddings and transform shape"""
    tgt_ws = dict()

    tgt_ws['transformer.embeddings.position_embeddings'] = ref_ws['Transformer/posembed_input/pos_embedding']
    tgt_ws['transformer.embeddings.cls_token'] = ref_ws['cls']
    tgt_ws['transformer.embeddings.patch_embeddings.weight'] = np.transpose(ref_ws['embedding/kernel'], (3, 2, 0, 1))
    tgt_ws['transformer.embeddings.patch_embeddings.bias'] = ref_ws['embedding/bias']

    return tgt_ws


def prepare_weights(weights_data):
    """prepare weights from every encoder layer"""
    new_weights = {}

    # Extract encoder data
    for i in range(12):
        new_weights.update(extract_encoder_weights(weights_data, index=i))

    # Extract something
    new_weights['transformer.encoder.encoder_norm.gamma'] = weights_data['Transformer/encoder_norm/scale']
    new_weights['transformer.encoder.encoder_norm.beta'] = weights_data['Transformer/encoder_norm/bias']

    # Extract embeddings
    new_weights.update(extract_embeddings(weights_data))

    # Extract head
    new_weights['head.weight'] = weights_data['head/kernel'].T
    new_weights['head.bias'] = weights_data['head/bias']

    # Take first ten head weights
    head_indexes = np.arange(0, 10, 1, dtype=int)
    new_weights.update(
        {
            'head.weight': new_weights['head.weight'][head_indexes],
            'head.bias': new_weights['head.bias'][head_indexes]
        }
    )

    # Turn numpy data into parameters
    new_weights = {
        k: ms.Parameter(v.astype(np.float32))
        for k, v in new_weights.items()
    }

    return new_weights


def npz2ckpt(npz_path):
    """
    Takes weights from .npz format.
    If necessary prepare it's shape to mindspore format
    and create dictionary ready to load into mindspore net

    Note:
        Supports ViT-base-16 only.

    Returns:
        weight dict of mindspore format
    """

    ref_weights = np.load(npz_path)
    prepared_weights = prepare_weights(ref_weights)

    return prepared_weights
