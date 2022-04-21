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
"""Training entry file"""

import argparse
import ast

import numpy as np
from absl import logging
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, export, Tensor
from mindspore.common import set_seed
from ncf import NCFModel, PredictWithSigmoid
import src.constants as rconst
from src.config import cfg


set_seed(1)

logging.set_verbosity(logging.INFO)


parser = argparse.ArgumentParser(description='ncf export')

parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "ml-20m"], help="Dataset.")
parser.add_argument("--file_name", type=str, default="ncf", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
parser.add_argument("--is_row_vector_input", type=ast.literal_eval, default=True,
                    help="Change model input into row vector for MindX SDK inference")
args = parser.parse_args()

def export_for_infer():
    """export method"""
    if args.dataset == "ml-1m":
        num_eval_users = 6040
        num_eval_items = 3706
    elif args.dataset == "ml-20m":
        num_eval_users = 138493
        num_eval_items = 26744
    else:
        raise ValueError("not supported dataset")

    layers = cfg.layers
    num_factors = cfg.num_factors
    ncf_net = NCFModel(num_users=num_eval_users,
                       num_items=num_eval_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16,
                       is_row_vector_input=args.is_row_vector_input)

    frozen_to_air_args = {'ckpt_file': args.ckpt_file,
                          'topk': rconst.TOP_K,
                          'num_eval_neg': rconst.NUM_EVAL_NEGATIVES,
                          'file_name': args.file_name,
                          'file_format': args.file_format}
    frozen_to_air(ncf_net, frozen_to_air_args)

def frozen_to_air(net, args_net):
    """frozen net parameters in the format of air"""
    param_dict = load_checkpoint(args_net.get("ckpt_file"))
    load_param_into_net(net, param_dict)
    network = PredictWithSigmoid(net, args_net.get("topk"), args_net.get("num_eval_neg"))
    users = Tensor(np.zeros([1, cfg.eval_batch_size]).astype(np.int32))
    items = Tensor(np.zeros([1, cfg.eval_batch_size]).astype(np.int32))
    masks = Tensor(np.zeros([1, cfg.eval_batch_size]).astype(np.float32))
    input_data = [users, items, masks]
    export(network, *input_data, file_name=args_net.get("file_name"), file_format=args_net.get("file_format"))

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    export_for_infer()
