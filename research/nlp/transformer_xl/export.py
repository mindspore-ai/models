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
"""export checkpoint file into air, onnx, mindir models"""

import argparse

import numpy as np
from mindspore import Tensor, load_checkpoint, export, context

from src.model.mem_transformer_for_ascend import MemTransformerLM as MemTransformerLMAscend

from src.model_utils.config import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer-XL export')
    parser.add_argument('--ckpt_path', default="./model0.ckpt", help='Directory of model.')
    parser.add_argument("--file_name", type=str, default="model_output", help="output file name.")
    parser.add_argument('--file_format', type=str, default='MINDIR', help='file format')
    parser.add_argument("--device_target", type=str, default="Ascend", help="Device Target, default GPU",
                        choices=["Ascend", "GPU"])
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")

    args = parser.parse_args()

    ascend = 'ascend'
    gpu = 'gpu'
    ntokens = 204

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    cutoffs = []
    net = MemTransformerLMAscend(ntokens, config.n_layer, config.n_head, config.d_model, config.d_head, config.d_inner,
                                 config.dropout, config.dropatt, batch_size=config.batch_size, d_embed=config.d_embed,
                                 div_val=config.div_val, pre_lnorm=config.pre_lnorm, tgt_len=config.tgt_len,
                                 ext_len=config.ext_len, mem_len=config.mem_len, eval_tgt_len=config.eval_tgt_len,
                                 cutoffs=cutoffs, same_length=config.same_length, clamp_len=config.clamp_len)

    model_filename = args.ckpt_path
    print(model_filename)
    if gpu in config.device_target:
        args.file_name = args.file_name + '_' + gpu
    else:
        args.file_name = args.file_name + '_' + ascend
    load_checkpoint(net=net, ckpt_file_name=model_filename, filter_prefix=['mems', 'valid_mems', 'empty_valid_mems'])
    data = Tensor(np.ones((80, config.batch_size), np.int32))
    target = Tensor(np.ones((80, config.batch_size), np.int32))
    export(net, data, target, file_name=args.file_name, file_format=args.file_format)
    print('export success:)')
