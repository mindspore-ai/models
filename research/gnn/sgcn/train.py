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

from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init
from mindspore.communication.management import get_rank
from mindspore.context import ParallelMode

from src.ms_utils import read_graph
from src.ms_utils import score_printer
from src.ms_utils import tab_printer
from src.param_parser import parameter_parser
from src.sgcn import SignedGCNTrainer


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
        trainer.setup_dataset()
        print('******************** set up dataset! ********************')
        print("\nTraining started.\n")
        trainer.create_and_train_model()
        print('******************** finish training! ********************')
        if args.test_size > 0:
            score_printer(trainer.logs)
    else:
        trainer = SignedGCNTrainer(args, edges)
        trainer.setup_dataset()
        trainer.create_and_train_model()


if __name__ == "__main__":
    main()
