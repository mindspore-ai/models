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
from mindspore import context
from mindspore.communication import init
from models.train_and_eval import TrainWrapper

def init_args():
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='GPU', help='Type of device: GPU or Ascend.')
    parser.add_argument('--dataset', type=str, default='spring', help='Type of dynamics: spring, charged or kuramoto.')
    parser.add_argument('--size', type=int, default=5, help='Number of particles.')
    parser.add_argument('--dim', type=int, default=4, help='Dimension of the input states.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. 0 for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--reg', type=float, default=100, help='Penalty factor for the symmetric prior.')
    parser.add_argument('--drop_out', type=float, default=0.0, help='rate of dropout.')
    parser.add_argument('--batch_size', type=int, default=2 ** 7, help='Batch size.')
    parser.add_argument('--skip', action='store_true', default=False, help='Skip the last type of edge.')
    parser.add_argument('--hidden', type=int, default=2 ** 8, help='Dimension of the hidden layers.')
    parser.add_argument('--edge_type', type=int, default=2, help='edge type.')
    parser.add_argument('--train_step', type=int, default=49, help='train step.')
    parser.add_argument('--test_step', type=int, default=20, help='test step.')
    parser.add_argument('--parallel', action='store_true', default=False, help='parallel training')
    return parser.parse_args()

def main():
    args = init_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    if args.parallel and args.device == 'GPU':
        init("nccl")
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True)
    with open("data/{}.pkl".format(args.dataset), "rb") as f:
        dataset, es = pickle.load(f)
    train_net = TrainWrapper(args, dataset, es)
    train_net.train()

if __name__ == "__main__":
    main()
