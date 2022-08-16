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
"""eval script"""

import os
import time

from mindspore import context, set_seed
from mindspore.train.model import Model, ParallelMode
from mindspore.communication.management import init
from mindspore.profiler.profiling import Profiler
from mindspore.train.serialization import load_checkpoint

from src.hlg import get_network
from src.dataset import get_dataset
from src.optimizer import get_optimizer
from src.eval_engine import get_eval_engine
from src.logging import get_logger
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.metric import ClassifyCorrectEval

try:
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')

    device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
    local_rank = int(os.getenv('RANK_ID'))    # local_rank
    device_num = int(os.getenv('RANK_SIZE'))  # world_size
    print("distribute")
except TypeError:
    device_id = 0   # 0 ~ 7
    local_rank = 0    # local_rank
    device_num = 1  # world_size
    print("standalone")

def add_static_args(args):
    """add_static_args"""
    args.train_image_size = args.eval_image_size
    args.weight_decay = 0.05
    args.no_weight_decay_filter = ""
    args.gc_flag = 0
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.loss_scale = 1024

    args.dataset_name = 'imagenet'
    args.save_checkpoint_path = './outputs'
    args.eval_engine = 'imagenet'
    args.auto_tune = 0
    args.seed = 1

    args.device_id = device_id
    args.local_rank = local_rank
    args.device_num = device_num

    return args

def modelarts_pre_process():
    '''modelarts pre process function.'''

    val_file = os.path.join(config.data_path, 'val/imagenet_val.tar')
    # train_file = os.path.join(config.data_path, 'train/imagenet_train.tar')
    tar_file = val_file

    print('tar_files:{}'.format(tar_file))
    if os.path.exists(tar_file):
        tar_dir = os.path.dirname(tar_file)
        print('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
        os.system('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
        os.system('cd {}; rm -rf {}'.format(tar_dir, tar_file))
    else:
        print('file no exists:', tar_file)

@moxing_wrapper(pre_process=modelarts_pre_process)
def eval_net():
    """eval_net"""
    args = add_static_args(config)
    set_seed(args.seed)
    args.logger = get_logger(args.save_checkpoint_path, rank=local_rank)

    context.set_context(device_id=device_id,
                        mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        save_graphs=False)

    if args.auto_tune:
        context.set_context(auto_tune_mode='GA')
    elif args.device_num == 1:
        pass
    else:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if args.open_profiler:
        profiler = Profiler(output_path="data_{}".format(local_rank))

    # init the distribute env
    if not args.auto_tune and args.device_num > 1:
        init()

    # network
    net = get_network(backbone_name=args.backbone, args=args)

    if os.path.isfile(args.pretrained):
        load_checkpoint(args.pretrained, net, strict_load=False)

    # evaluation dataset
    eval_dataset = get_dataset(dataset_name=args.dataset_name,
                               do_train=False,
                               dataset_path=args.eval_path,
                               args=args)
    step_size = eval_dataset.get_dataset_size()

    opt, _ = get_optimizer(optimizer_name='adamw',
                           network=net,
                           lrs=1.0,
                           args=args)

    # evaluation engine
    if args.auto_tune or args.open_profiler or eval_dataset is None:
        args.eval_engine = ''
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # model
    eval_network = ClassifyCorrectEval(net)
    model = Model(net, loss_fn=None, optimizer=opt,
                  metrics=eval_engine.metric, eval_network=eval_network,
                  loss_scale_manager=None, amp_level=args.amp_level)
    args.logger.save_args(args)

    t0 = time.time()
    output = model.eval(eval_dataset)
    t1 = time.time()

    print_str = 'accuracy={:.6f}'.format(float(output['acc']))
    print_str += ', per step time: {:.4f}s'.format((t1 - t0) / step_size)
    print_str += ', total cost time: {:.4f}s'.format(t1-t0)
    print(print_str)

    if args.open_profiler:
        profiler.analyse()

if __name__ == '__main__':
    eval_net()
