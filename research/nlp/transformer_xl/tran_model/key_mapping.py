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
Build mapping file for model parameter transformation
"""


# mindspore -> pytroch
def msp2torch():
    param_dict = {}
    param_dict["r_w_bias"] = 'r_w_bias'
    param_dict["r_r_bias"] = 'r_r_bias'
    param_dict['word_emb.emb_layers.0.embedding_table'] = 'word_emb.emb_layers.0.weight'
    for i in range(0, 12):
        param_dict[str(i) + '.attn.qkv_net.weight'] = 'layers.' + str(i) + '.dec_attn.qkv_net.weight'
        param_dict[str(i) + '.attn.o_net.weight'] = 'layers.' + str(i) + '.dec_attn.o_net.weight'
        param_dict[str(i) + '.attn.layer_norm.gamma'] = 'layers.' + str(i) + '.dec_attn.layer_norm.weight'
        param_dict[str(i) + '.attn.layer_norm.beta'] = 'layers.' + str(i) + '.dec_attn.layer_norm.bias'
        param_dict[str(i) + '.attn.r_net.weight'] = 'layers.' + str(i) + '.dec_attn.r_net.weight'
        param_dict[str(i) + '.pos_ff.CoreNet.0.weight'] = 'layers.' + str(i) + '.pos_ff.CoreNet.0.weight'
        param_dict[str(i) + '.pos_ff.CoreNet.0.bias'] = 'layers.' + str(i) + '.pos_ff.CoreNet.0.bias'
        param_dict[str(i) + '.pos_ff.CoreNet.3.weight'] = 'layers.' + str(i) + '.pos_ff.CoreNet.3.weight'
        param_dict[str(i) + '.pos_ff.CoreNet.3.bias'] = 'layers.' + str(i) + '.pos_ff.CoreNet.3.bias'
        param_dict[str(i) + '.pos_ff.layer_norm.gamma'] = 'layers.' + str(i) + '.pos_ff.layer_norm.weight'
        param_dict[str(i) + '.pos_ff.layer_norm.beta'] = 'layers.' + str(i) + '.pos_ff.layer_norm.bias'
    param_dict['crit.out_layers.0.weight'] = 'crit.out_layers.0.weight'
    param_dict['crit.out_layers.0.bias'] = 'crit.out_layers.0.bias'
    param_dict['pos_emb.inv_freq'] = 'pos_emb.inv_freq'
    with open('msp2torch_base.txt', 'w') as f:
        for key, value in param_dict.items():
            line = '%s:%s\n' % (key, value)
            f.write(line)
    return param_dict


# tf -> msp
def tf2msp():
    param_dict = {}
    param_dict["transformer/r_w_bias"] = 'r_w_bias'
    param_dict["transformer/r_r_bias"] = 'r_r_bias'
    param_dict['transformer/adaptive_embed/lookup_table'] = 'word_emb.emb_layers.0.embedding_table'
    for i in range(0, 24):
        param_dict['transformer/layer_' + str(i) + '/rel_attn/qkv/kernel'] = str(i) + '.attn.qkv_net.weight'
        param_dict['transformer/layer_' + str(i) + '/rel_attn/o/kernel'] = str(i) + '.attn.o_net.weight'
        param_dict['transformer/layer_' + str(i) + '/rel_attn/r/kernel'] = str(i) + '.attn.r_net.weight'
        param_dict['transformer/layer_' + str(i) + '/rel_attn/LayerNorm/gamma'] = str(i) + '.attn.layer_norm.gamma'
        param_dict['transformer/layer_' + str(i) + '/rel_attn/LayerNorm/beta'] = str(i) + '.attn.layer_norm.beta'
        param_dict['transformer/layer_' + str(i) + '/ff/layer_1/kernel'] = str(i) + '.pos_ff.CoreNet.0.weight'
        param_dict['transformer/layer_' + str(i) + '/ff/layer_1/bias'] = str(i) + '.pos_ff.CoreNet.0.bias'
        param_dict['transformer/layer_' + str(i) + '/ff/layer_2/kernel'] = str(i) + '.pos_ff.CoreNet.3.weight'
        param_dict['transformer/layer_' + str(i) + '/ff/layer_2/bias'] = str(i) + '.pos_ff.CoreNet.3.bias'
        param_dict['transformer/layer_' + str(i) + '/ff/LayerNorm/gamma'] = str(i) + '.pos_ff.layer_norm.gamma'
        param_dict['transformer/layer_' + str(i) + '/ff/LayerNorm/beta'] = str(i) + '.pos_ff.layer_norm.beta'
    with open('tf2msp_large.txt', 'w') as f:
        for key, value in param_dict.items():
            line = '%s:%s\n' % (key, value)
            f.write(line)
    return param_dict


if __name__ == '__main__':
    msp2torch()
    tf2msp()
