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
""" Train Knowledge Graph Embedding Model """
import os
import time
import argparse
import moxing
import numpy as np
from mindspore import context, set_seed
from mindspore.train.serialization import save_checkpoint
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from eval import KGEModel
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num
from src.dataset import create_dataset
from src.dataset import get_entity_and_relation
from src.rotate import ModelBuilder

set_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser(description='rotate ')
parser.add_argument('--data_url', type=str, default=None, help='Location of Data')
parser.add_argument('--train_url', type=str, default='', help='Location of training outputs')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device where the code will be implemented')
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument("--ckpt_file", type=str, default='/cache/data/rotate.ckpt', help="Checkpoint file path.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")

args_opt, unparsed = parser.parse_known_args()


def modelarts_pre_process():
    pass

def export_rotate():
    """ export_rotate """
    entity2id, relation2id = get_entity_and_relation('/cache/data/wn18rr')
    config.num_entity, config.num_relation = len(entity2id), len(relation2id)

    model_builder = ModelBuilder(config)
    network = model_builder.get_eval_net()
    model_params = load_checkpoint(ckpt_file_name=config.eval_checkpoint)
    load_param_into_net(net=network, parameter_dict=model_params)
    infer_net_head = KGEModel(network=network, mode='head-mode')
    infer_net_tail = KGEModel(network=network, mode='tail-mode')
    infer_net_head.set_train(False)
    infer_net_tail.set_train(False)

    positive_sample = Tensor(np.ones([config.batch_size, 3]).astype(np.int32))
    negative_sample = Tensor(np.ones([config.batch_size, config.negative_sample_size]).astype(np.int32))
    filter_bias = Tensor(np.ones([config.batch_size, config.negative_sample_size]).astype(np.float32))

    input_data = [positive_sample, negative_sample, filter_bias]
    export(
        infer_net_head,
        *input_data,
        file_name='/cache/data/rotate'+'-head',
        file_format=args_opt.file_format

    )
    export(
        infer_net_tail,
        *input_data,
        file_name='/cache/data/rotate'+'-tail',
        file_format=args_opt.file_format
    )


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_kge():

    """ Train RotatE Model """

    config.rank_size = get_device_num()
    moxing.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/data')
    if config.rank_size > 1:
        config.is_distribute = True
        config.max_steps = int(config.max_steps / config.rank_size)
        if config.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=config.rank_size,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank_id = int(os.environ.get('RANK_ID'))
        elif config.device_target == "GPU":
            init()
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=config.rank_size,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank_id = get_rank()
        else:
            print("Unsupported device_target ", config.device_target)
            exit()
    else:
        config.is_distribute = False
        if config.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        elif config.device_target == "GPU":
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        config.rank_size = None
        rank_id = None

    if not os.path.exists(config.train_url):
        try:
            os.mkdir(config.train_url)
        except OSError:
            pass

    num_entity, num_relation, train_dataloader = create_dataset(
        data_path='/cache/data/wn18rr',
        config=config,
        is_train=True,
        device_num=config.rank_size,
        rank_id=rank_id
    )
    config.num_entity, config.num_relation = num_entity, num_relation

    model_builder = ModelBuilder(config)
    train_net_head, train_net_tail = model_builder.get_train_net()

    start_time = time.time()
    train_net_head.set_train()
    train_net_tail.set_train()
    for _step in range(config.max_steps):
        positive_sample, negative_sample, subsampling_weight, mode = next(train_dataloader)
        start = time.time()
        if mode == 'head-mode':
            loss, cond, _ = train_net_head(
                positive_sample,
                negative_sample,
                subsampling_weight
            )
        elif mode == 'tail-mode':
            loss, cond, _ = train_net_tail(
                positive_sample,
                negative_sample,
                subsampling_weight
            )
        else:
            raise ValueError("Wrong sample mode! ")
        if cond:
            print("======== Over Flow ========")
        print('step{} cost time: {:.2f}ms loss={:.6f} '.format(_step, (time.time()-start)*1000, loss.asnumpy().item()))
    if (config.is_distribute and rank_id == 0) or not config.is_distribute:
        save_checkpoint(
            train_net_tail.network.network,
            os.path.join(config.train_url, '{}.ckpt'.format(config.experiment_name))
        )
    print("Training is done ... ")
    print("Training process costs: {} ".format(time.time() - start_time))

    print('start export')
    config.data_path = config.data_url
    config.batch_size = 1
    config.negative_sample_size = 40943
    config.eval_checkpoint = os.path.join(config.train_url, '{}.ckpt'.format(config.experiment_name))
    config.file_format = 'AIR'
    export_rotate()
    print("export finished")
    moxing.file.copy_parallel(src_url='/cache/data/', dst_url=args_opt.train_url)

if __name__ == '__main__':
    train_kge()
