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
"""SGCN runner."""
import os
from mindspore import Tensor
from mindspore import export
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore.common import set_seed
from mindspore.communication import init
from mindspore.communication.management import get_rank
from mindspore.context import ParallelMode
import moxing as mox

from src.ms_utils import read_graph
from src.ms_utils import score_printer
from src.ms_utils import tab_printer
from src.param_parser import parameter_parser
from src.sgcn import SignedGCNTrainer
from src.sgcn import SignedGraphConvolutionalNetwork

def remove_self_loops(edge_index):
    """
    remove self loops
    Args:
        edge_index (LongTensor): The edge indices.

    Returns:
        Tensor(edge_index): removed self loops
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return Tensor(edge_index)

def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    set_seed(args.seed)
    device_id = int(os.getenv('DEVICE_ID', args.device_id))

    CKPT_OUTPUT_PATH = "../"
    mox.file.copy_parallel(args.data_path, '/cache')

    args.edge_path = args.data_path + "/bitcoin_" + args.data_type + ".csv"
    args.features_path = args.data_path + "/bitcoin_" + args.data_type + ".csv"

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)
    args.rank_log_save_ckpt_flag = 1
    if args.distributed:
        if args.device_target == 'Ascend':
            init()
        else:
            init('nccl')
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        args.rank = get_rank()
        if args.rank != 0:
            args.rank_log_save_ckpt_flag = 0
    edges = read_graph(args)
    if args.rank_log_save_ckpt_flag:
        tab_printer(args)
        trainer = SignedGCNTrainer(args, edges)
        print('******************** set up dataset... ********************')
        dataset = trainer.setup_dataset()
        print('******************** set up dataset! ********************')
        print("\nTraining started.\n")
        trainer.create_and_train_model()
        print('******************** finish training! ********************')
        if args.test_size > 0:
            score_printer(trainer.logs)
    else:
        trainer = SignedGCNTrainer(args, edges)
        dataset = trainer.setup_dataset()
        trainer.create_and_train_model()

    print('******************** export! ********************')
    input_x, pos_edg, neg_edg = dataset[0], dataset[1], dataset[2]
    repos, reneg = remove_self_loops(pos_edg), remove_self_loops(neg_edg)
    net = SignedGraphConvolutionalNetwork(input_x, args.norm, args.norm_embed, args.bias)
    # Load parameters from checkpoint into network
    param_dict = load_checkpoint(args.checkpoint_file + '_auc.ckpt')
    load_param_into_net(net, param_dict)
    # export
    export(net, repos, reneg,
           file_name="sgcn_auc", file_format="AIR")

    param_dict = load_checkpoint(args.checkpoint_file + '_f1.ckpt')
    load_param_into_net(net, param_dict)
    # export
    export(net, repos, reneg,
           file_name="sgcn_f1", file_format="AIR")

    print("==========================================")
    print("sgcn_auc.air and sgcn_f1.air exported successfully!")
    print("==========================================")
    mox.file.copy_parallel(CKPT_OUTPUT_PATH, args.save_ckpt)

if __name__ == "__main__":
    main()
