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

import pickle
from argparse import ArgumentParser
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from models.nri import NRIModel

def init_args():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='GPU', help='Type of device: GPU or Ascend.')
    parser.add_argument('--dataset', type=str, default='spring', help='Type of dynamics: spring, charged or kuramoto.')
    parser.add_argument('--ckpt_file', type=str, default='./checkpoints/spring.ckpt',
                        help='Where to load the best model.')
    parser.add_argument('--file_name', type=str, default='nri_mpm', help='output file name.')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'MINDIR'], default='MINDIR', help='file format')
    return parser.parse_args()

def main():
    args = init_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    with open("data/{}.pkl".format(args.dataset), "rb") as f:
        _, es = pickle.load(f)

    dim = 3 if args.dataset == "kuramoto" else 4
    skip = args.dataset == "kuramoto"
    nri_mpm = NRIModel(dim, 256, 2, 0.0, skip, 5, es)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(nri_mpm, param_dict)

    enc_states = Tensor(np.ones((128, 49, 5, dim)), ms.float32)
    dec_states = Tensor(np.ones((128, 49, 5, dim)), ms.float32)
    export(nri_mpm, enc_states, dec_states, file_name=args.file_name, file_format=args.file_format)
    print('=========================================')
    print('{}.mindir exported successfully!'.format(args.file_name))
    print('=========================================')

if __name__ == "__main__":
    main()
