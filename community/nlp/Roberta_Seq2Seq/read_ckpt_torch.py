# Copyright 2020 Huawei Technologies Co., Ltd
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
""" transform the torch bin """
import torch
import mindspore.context as context
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

context.set_context(device_target="Ascend", device_id=3)


trans_dict = {
    'roberta.embeddings.word_embeddings.weight': 'roberta.embeddings.word_embeddings.embedding_table',
    'roberta.embeddings.position_embeddings.weight': 'roberta.embeddings.position_embeddings.embedding_table',
}
# decoder
'''
'.LayerNorm.weight':'.LayerNorm.gamma'
'.LayerNorm.bias':'.LayerNorm.beta'
'''


def cover_encoder(path, save_path):
    """
    cover the encoder
    """
    torch_dict = torch.load(path)

    ms_encoder_params_list = []

    for name in torch_dict:
        param_dict = {}
        parameter = torch_dict[name]
        print(name)
        if name in trans_dict:
            name = trans_dict[name]
        elif name.endswith('.LayerNorm.weight'):
            name = name.replace('.LayerNorm.weight', '.LayerNorm.gamma')
        elif name.endswith('.LayerNorm.bias'):
            name = name.replace('.LayerNorm.bias', '.LayerNorm.beta')
        if 'roberta.' in name:
            name = name.replace('roberta.', '')
        if 'attention.self' in name:
            name = name.replace('attention.self', 'attention.self_attention')

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        ms_encoder_params_list.append(param_dict)
    save_checkpoint(ms_encoder_params_list, save_path)


def cover_decoder(path, save_path):
    """cover decoder"""
    torch_dict = torch.load(path)
    ms_decoder_params_list = []
    for name in torch_dict:
        param_dict = {}
        parameter = torch_dict[name]
        print(name)
        if name in trans_dict:
            name = trans_dict[name]
        elif name.endswith('.LayerNorm.weight'):
            name = name.replace('.LayerNorm.weight', '.LayerNorm.gamma')
        elif name.endswith('.LayerNorm.bias'):
            name = name.replace('.LayerNorm.bias', '.LayerNorm.beta')
        elif name.endswith('layer_norm.weight'):
            name = name.replace('.layer_norm.weight', '.layer_norm.gamma')
        elif name.endswith('layer_norm.bias'):
            name = name.replace('.layer_norm.bias', '.layer_norm.beta')

        if 'attention.self' in name:
            name = name.replace('attention.self', 'attention.self_attention')
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        ms_decoder_params_list.append(param_dict)

    save_checkpoint(ms_decoder_params_list, save_path)


def cover_encoder_decoder_model(path, save_path):
    """ cover encoder_decoder"""
    torch_dict = torch.load(path, map_location=torch.device('cpu'))
    ms_decoder_params_list = []
    for name in torch_dict:
        param_dict = {}
        parameter = torch_dict[name]
        print(name)
        if name.endswith('embeddings.weight'):
            name = name.replace('.embeddings.weight', 'embeddings.embedding_table')
        elif name.endswith('.LayerNorm.weight'):
            name = name.replace('.LayerNorm.weight', '.LayerNorm.gamma')
        elif name.endswith('.LayerNorm.bias'):
            name = name.replace('.LayerNorm.bias', '.LayerNorm.beta')
        elif name.endswith('layer_norm.weight'):
            name = name.replace('.layer_norm.weight', '.layer_norm.gamma')
        elif name.endswith('layer_norm.bias'):
            name = name.replace('.layer_norm.bias', '.layer_norm.beta')

        if 'attention.self' in name:
            name = name.replace('attention.self', 'attention.self_attention')
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        ms_decoder_params_list.append(param_dict)

    save_checkpoint(ms_decoder_params_list, save_path)


if __name__ == '__main__':
    cover_encoder('../roberta_base/pytorch_model.bin', '../roberta_base_ms/robarta_base_encoder_ms.ckpt')
    cover_decoder('../roberta_base/pytorch_model.bin', '../roberta_base_ms/robarta_base_decoder_ms.ckpt')
