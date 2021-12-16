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
""" Train Knowledge Graph Embedding Model """
import os
import time
import numpy as np
from mindspore import context, set_seed
from mindspore.train.serialization import save_checkpoint
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num
from src.dataset import create_dataset
from src.rotate import ModelBuilder

set_seed(1)
np.random.seed(1)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_kge():
    """ Train RotatE Model """
    config.rank_size = get_device_num()
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

    if not os.path.exists(config.output_path):
        try:
            os.mkdir(config.output_path)
        except OSError:
            pass

    num_entity, num_relation, train_dataloader = create_dataset(
        data_path=config.data_path,
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
            os.path.join(config.output_path, '{}.ckpt'.format(config.experiment_name))
        )
    print("Training is done ... ")
    print("Training process costs: {} ".format(time.time() - start_time))


if __name__ == '__main__':
    train_kge()
