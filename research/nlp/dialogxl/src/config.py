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

import argparse
import mindspore as ms
from mindspore import save_checkpoint

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_dir', type=str, default='pretrained_model/dialogXL.ckpt')

    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--num_heads', nargs='+', type=int, default=[5, 5, 1, 1],
                        help='number of heads: [n_local, n_global, n_speaker, n_listener]')

    parser.add_argument('--max_sent_len', type=int, default=300, help='max content length for each text')
    parser.add_argument('--mem_len', type=int, default=400, help='max memory length')
    parser.add_argument('--windowp', type=int, default=5, help='local attention window size')

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--output_dropout', type=float, default=.0)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--device', type=str, default='GPU')

    args = parser.parse_args()
    return args

class DialogXLConfig:
    def __init__(self):
        self.d_head = 64
        self.d_inner = 3072
        self.d_model = 768
        self.dropout = 0.1
        self.ff_activation = 'gelu'
        self.layer_norm_eps = 1e-12
        self.n_head = 12
        self.n_layer = 12
        self.vocab_size = 32000
        self.initializer_range = 0.02

    def set_args(self, args):
        for params in vars(args):
            setattr(self, params, getattr(args, params))

def convert_pt_to_ms(state_dict, ckpt_file_name='dialogXL'):
    ms_ckpt = []
    for k, v in state_dict.items():
        if k == 'mask_emb' or 'seg_embed' in k or 'r_s_bias' in k:
            continue
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
        if 'layer_norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'word_embedding' in k:
            k = k.replace('weight', 'embedding_table')

        ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
    save_checkpoint(ms_ckpt, ckpt_file_name)
