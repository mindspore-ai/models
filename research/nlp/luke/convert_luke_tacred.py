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
"""convert luke pretrain model"""
import collections

import torch
from mindspore import Tensor
from mindspore import save_checkpoint

from src.model_utils.config_args import args_config as args


def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_maps = collections.OrderedDict({
        'embeddings.word_embeddings.weight': "embeddings.word_embeddings.embedding_table",
        'embeddings.position_embeddings.weight': "embeddings.position_embeddings.embedding_table",
        'embeddings.token_type_embeddings.weight': "embeddings.token_type_embeddings.embedding_table",
        'embeddings.LayerNorm.weight': 'embeddings.LayerNorm.gamma',
        'embeddings.LayerNorm.bias': 'embeddings.LayerNorm.beta',
        'entity_embeddings.entity_embeddings.weight': 'entity_embeddings.entity_embeddings.embedding_table',
        'entity_embeddings.entity_embedding_dense.weight': 'entity_embeddings.entity_embedding_dense.weight',
        'entity_embeddings.position_embeddings.weight': 'entity_embeddings.position_embeddings.embedding_table',
        'entity_embeddings.token_type_embeddings.weight':
            'entity_embeddings.token_type_embeddings.embedding_table',
        'entity_embeddings.LayerNorm.weight': 'entity_embeddings.LayerNorm.gamma',
        'entity_embeddings.LayerNorm.bias': 'entity_embeddings.LayerNorm.beta'

    })
    # add attention layers
    for i in range(attention_num):
        weight_maps[f'encoder.layer.{i}.attention.self.query.weight'] = \
            f'encoder.layer.{i}.attention.self1.query.weight'
        weight_maps[f'encoder.layer.{i}.attention.self.query.bias'] = \
            f'encoder.layer.{i}.attention.self1.query.bias'
        weight_maps[f'encoder.layer.{i}.attention.self.key.weight'] = \
            f'encoder.layer.{i}.attention.self1.key.weight'
        weight_maps[f'encoder.layer.{i}.attention.self.key.bias'] = \
            f'encoder.layer.{i}.attention.self1.key.bias'
        weight_maps[f'encoder.layer.{i}.attention.self.value.weight'] = \
            f'encoder.layer.{i}.attention.self1.value.weight'
        weight_maps[f'encoder.layer.{i}.attention.self.value.bias'] = \
            f'encoder.layer.{i}.attention.self1.value.bias'
        weight_maps[f'encoder.layer.{i}.attention.output.dense.weight'] = \
            f'encoder.layer.{i}.attention.output.dense.weight'
        weight_maps[f'encoder.layer.{i}.attention.output.dense.bias'] = \
            f'encoder.layer.{i}.attention.output.dense.bias'
        weight_maps[f'encoder.layer.{i}.attention.output.LayerNorm.weight'] = \
            f'encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_maps[f'encoder.layer.{i}.attention.output.LayerNorm.bias'] = \
            f'encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_maps[f'encoder.layer.{i}.output.LayerNorm.bias'] = \
            f'encoder.layer.{i}.output.LayerNorm.beta'

        weight_maps[
            f'encoder.layer.{i}.intermediate.dense.weight'] = f'encoder.layer.{i}.intermediate.dense.weight'
        weight_maps[f'encoder.layer.{i}.intermediate.dense.bias'] = f'encoder.layer.{i}.intermediate.dense.bias'
        weight_maps[f'encoder.layer.{i}.output.dense.weight'] = f'encoder.layer.{i}.output.dense.weight'
        weight_maps[f'encoder.layer.{i}.output.dense.bias'] = f'encoder.layer.{i}.output.dense.bias'
        weight_maps[f'encoder.layer.{i}.output.LayerNorm.weight'] = \
            f'encoder.layer.{i}.output.LayerNorm.gamma'
        weight_maps[f'encoder.layer.{i}.output.LayerNorm.bias'] = \
            f'encoder.layer.{i}.output.LayerNorm.beta'

    return weight_maps


def convert_ckpt(data_dir):
    if args.model_type == "luke-large":
        count = 24
    else:
        count = 12

    state_dict = []

    t_params = torch.load(data_dir, map_location=torch.device('cpu'))

    weight_map = build_params_map(count)
    for weight_name, weight_value in t_params.items():
        parameter = weight_value
        if weight_name not in weight_map.keys():
            print(weight_name, "not in")
            state_dict.append({'name': weight_name, 'data': Tensor(parameter.numpy())})
            continue
        state_dict.append({'name': weight_map[weight_name], 'data': Tensor(parameter.numpy())})
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    print("over")
    return state_dict


if __name__ == "__main__":
    input_dir = f'{args.model_file}/pytorch_model.bin'
    output_dir = f"{args.model_file}/luke.ckpt"

    ms_state_dict = convert_ckpt(input_dir)
    save_checkpoint(ms_state_dict, output_dir)
