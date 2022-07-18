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

import sys
import argparse
import pickle
import mindspore
import mindspore.ops as ops
from mindspore import context
from mindspore.dataset import GeneratorDataset
from mindspore import save_checkpoint
from src.metric.calc import bpc
from src.model.mem_transformer import MemTransformerLM
from src.model_utils.config import config
from src.utils.dataset_util import get_dataset
from src.callback.eval import doEval

sys.path.insert(0, '../')

parser = argparse.ArgumentParser(description='PyTorch Model Trans MindSpore Model.')
parser.add_argument('--datadir', default='./data/enwik8',
                    help='Directory contains enwik8 dataset.')
parser.add_argument('--dataset', default='enwik8',
                    help='Dataset Name.', choices=["enwik8", "text8"])
parser.add_argument('--pt_path', default="./model.pt", help='Directory of model param.')
parser.add_argument("--device", type=str, default="GPU", help="Device Target, default GPU",
                    choices=["Ascend", "GPU"])
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
args = parser.parse_args()
datadir = args.datadir
dataset = args.dataset
pt_path = args.pt_path
device_id = args.device_id
print(datadir)
print(dataset)
print(pt_path)

numpy_param_path = pt_path
with open(numpy_param_path, 'rb') as f:
    tf_dict = pickle.load(f, encoding='bytes')

dataset = get_dataset(datadir, dataset)
ntokens = len(dataset.vocab)

context.set_context(device_id=device_id)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", max_device_memory="39.0GB",
                    enable_graph_kernel=True)

test_dataset = GeneratorDataset(source=dataset.get_test_generator(), column_names=['data', 'target'],
                                shuffle=False)

cutoffs = []
net = MemTransformerLM(ntokens, config.n_layer, config.n_head, config.d_model,
                       config.d_head, config.d_inner, config.dropout, config.dropatt, batch_size=config.batch_size,
                       d_embed=config.d_embed, div_val=config.div_val,
                       pre_lnorm=config.pre_lnorm, tgt_len=config.tgt_len,
                       ext_len=config.ext_len, mem_len=config.mem_len, eval_tgt_len=config.eval_tgt_len,
                       cutoffs=cutoffs, same_length=config.same_length, clamp_len=config.clamp_len)

net_dict = {}
with open('./tf2msp_large.txt', 'r') as f:
    for line in f.readlines():
        tf_name, msp_name = line.strip().split(":")
        net_dict[msp_name] = tf_dict[tf_name]

transpose = ops.Transpose()

for k in net.parameters_dict():
    if k in ('mems', 'valid_mems', 'empty_valid_mems'):
        continue
    if k in ('pos_emb.inv_freq', 'crit.out_layers.0.weight', 'crit.out_layers.0.bias'):
        continue
    if 'attn.qkv_net.weight' in k or 'attn.r_net.weight' in k or \
            'attn.o_net.weight' in k or 'pos_ff.CoreNet.0.weight' in k or \
            'pos_ff.CoreNet.3.weight' in k:
        a = mindspore.Tensor(net_dict[k].transpose((1, 0)))
        net.parameters_dict()[k].set_data(a)
    else:
        net.parameters_dict()[k].set_data(mindspore.Tensor(net_dict[k]))

print('load net param')

save_path = './tf2msp_model/' + str(args.dataset) + '_model.ckpt'
save_checkpoint(net, save_path)

test_loss = doEval(net, test_dataset, config.tgt_len, config.ext_len, config.mem_len, config.eval_tgt_len)

print('=' * 100)
if config.dataset in ['enwik8', 'text8']:
    print('| End of test | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, bpc(test_loss)))

print('=' * 100)
