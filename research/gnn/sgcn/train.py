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
from mindspore.context import ParallelMode
from src.sgcn import SignedGCNTrainer
from src.param_parser import parameter_parser
from src.ms_utils import tab_printer, read_graph, score_printer


def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)
    if args.distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    tab_printer(args)
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    print('******************** set up dataset... ********************')
    trainer.setup_dataset()
    print('******************** set up dataset! ********************')
    trainer.create_and_train_model()
    print('******************** finish training! ********************')
    if args.test_size > 0:
        score_printer(trainer.logs)


if __name__ == "__main__":
    main()
