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
import pickle
import logging
from mindspore import Tensor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model utils')


def update_param(net_param_dict, params, ms_full_name, torch_full_name):
    old_param = net_param_dict[ms_full_name]
    new_param = Tensor(params[torch_full_name], old_param.data.dtype)
    old_param.set_data(new_param)


def load_visual_encoder(net, param_dict):
    for mindspore_full_name, torch_full_name in [
            ('class_embedding', 'visual.class_embedding'),
            ('positional_embedding', 'visual.positional_embedding'),
            ('proj', 'visual.proj'),
            ('conv1.weight', 'visual.conv1.weight'),
            ('ln_pre.gamma', 'visual.ln_pre.weight'),
            ('ln_pre.beta', 'visual.ln_pre.bias'),
            ('ln_post.gamma', 'visual.ln_post.weight'),
            ('ln_post.beta', 'visual.ln_post.bias')
    ]:
        update_param(net, param_dict, mindspore_full_name, torch_full_name)
    for i in range(24):
        mindspore_prefix = 'transformer.resblocks.'
        torch_prefix = 'visual.transformer.resblocks.'
        for mindspore_name, torch_name in [
                ('attn.attn.in_proj.weight', 'attn.in_proj_weight'),
                ('attn.attn.in_proj.bias', 'attn.in_proj_bias'),
                ('attn.attn.out_proj.weight', 'attn.out_proj.weight'),
                ('attn.attn.out_proj.bias', 'attn.out_proj.bias'),
                ('ln_1.gamma', 'ln_1.weight'),
                ('ln_1.beta', 'ln_1.bias'),
                ('ln_2.gamma', 'ln_2.weight'),
                ('ln_2.beta', 'ln_2.bias'),
                ('c_fc.weight', 'mlp.c_fc.weight'),
                ('c_fc.bias', 'mlp.c_fc.bias'),
                ('c_proj.weight', 'mlp.c_proj.weight'),
                ('c_proj.bias', 'mlp.c_proj.bias')
        ]:
            mindspore_full_name = '{}{}.{}'.format(mindspore_prefix, i, mindspore_name)
            torch_full_name = '{}{}.{}'.format(torch_prefix, i, torch_name)
            update_param(net, param_dict, mindspore_full_name, torch_full_name)


def load_text_encoder(net, param_dict):
    for mindspore_full_name, torch_full_name in [
            ('embedding_table', 'transformer.token_embedding.weight'),
            ('positional_embedding', 'transformer.positional_embedding'),
            ('text_projection', 'transformer.text_projection'),
            ('ln_final.gamma', 'transformer.ln_final.weight'),
            ('ln_final.beta', 'transformer.ln_final.bias')
    ]:
        update_param(net, param_dict, mindspore_full_name, torch_full_name)
    mindspore_prefix = 'transformer_layer.resblocks.'
    torch_prefix = 'transformer.resblocks.'
    for i in range(12):
        for mindspore_name, torch_name in [
                ('attn.attn.in_proj.weight', 'attn.in_proj_weight'),
                ('attn.attn.in_proj.bias', 'attn.in_proj_bias'),
                ('attn.attn.out_proj.weight', 'attn.out_proj.weight'),
                ('attn.attn.out_proj.bias', 'attn.out_proj.bias'),
                ('ln_1.gamma', 'ln_1.weight'),
                ('ln_1.beta', 'ln_1.bias'),
                ('c_fc.weight', 'mlp.c_fc.weight'),
                ('c_fc.bias', 'mlp.c_fc.bias'),
                ('c_proj.weight', 'mlp.c_proj.weight'),
                ('c_proj.bias', 'mlp.c_proj.bias'),
                ('ln_2.gamma', 'ln_2.weight'),
                ('ln_2.beta', 'ln_2.bias')
        ]:
            mindspore_full_name = '{}{}.{}'.format(mindspore_prefix, i, mindspore_name)
            torch_full_name = '{}{}.{}'.format(torch_prefix, i, torch_name)
            update_param(net, param_dict, mindspore_full_name, torch_full_name)


def load_model(ckpt_path, visual_encoder, text_encoder):
    with open(ckpt_path, 'rb') as ckpt_fp:
        param_dict = pickle.load(ckpt_fp)
    visual_encoder_param = visual_encoder.parameters_dict()
    text_encoder_param = text_encoder.parameters_dict()
    load_visual_encoder(visual_encoder_param, param_dict)
    load_text_encoder(text_encoder_param, param_dict)
    logger.info("model loaded")
