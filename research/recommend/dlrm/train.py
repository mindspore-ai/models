# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""
train criteo
"""
import os

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed

from src.dlrm import ModelBuilder, AccMetric
from src.dataset import create_dataset, DataType
from src.callback import EvalCallBack, LossCallBack
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num

config.do_eval = config.do_eval == 'True'
config.rank_size = get_device_num()

set_seed(config.random_seed)

def modelarts_pre_process():
    pass

@moxing_wrapper()
def train_dlrm():
    """Train DLRM
    """
    if config.rank_size > 1:
        if config.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            rank_id = int(os.environ.get('RANK_ID'))
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
        elif config.device_target == "GPU":
            device_num = int(os.getenv("RANK_SIZE"))
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
            init("nccl")
            context.reset_auto_parallel_context()
            rank_id = get_rank()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            print("Unsupported device_target", config.device_targe)

    else:
        if config.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        config.rank_size = None
        rank_id = None

    ds_train = create_dataset(
        config.dataset_path,
        train_mode=True,
        epochs=1,
        batch_size=config.batch_size,
        data_type=DataType(config.data_format),
        line_per_sample=1,
        rank_size=config.rank_size,
        rank_id=rank_id
    )


    steps_size = ds_train.get_dataset_size()
    print(steps_size)

    if config.convert_dtype:
        config.convert_dtype = config.device_target != 'CPU'

    model_builder = ModelBuilder(config, config)
    train_net, eval_net = model_builder.get_train_eval_net()
    acc_metric = AccMetric()
    model = Model(train_net, eval_network=eval_net, metrics={'acc': acc_metric})

    loss_callback = LossCallBack(loss_file_path=config.loss_file_name,
                                 data_size=ds_train.get_dataset_size())
    callback_list = [loss_callback]

    if config.save_checkpoint:
        if config.rank_size:
            config.ckpt_file_name_prefix = config.ckpt_file_name_prefix + str(get_rank())
            config.ckpt_path = os.path.join(config.ckpt_path, 'ckpt_' + str(get_rank()) + '/')

        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=config.ckpt_file_name_prefix,
                                  directory=config.ckpt_path,
                                  config=config_ck)
        callback_list.append(ckpt_cb)

    if config.do_eval:
        ds_eval = create_dataset(config.dataset_path, train_mode=False, epochs=1,
                                 batch_size=config.test_batch_size,
                                 data_type=DataType(config.data_format),
                                 line_per_sample=1)
        eval_callback = EvalCallBack(model, ds_eval, acc_metric, eval_file_path=config.eval_file_name)
        callback_list.append(eval_callback)
    if config.device_target == 'GPU':
        model.train(config.train_epochs,
                    ds_train,
                    callbacks=callback_list,
                    sink_size=steps_size,
                    dataset_sink_mode=False)
    else:
        model.train(config.train_epochs, ds_train, callbacks=callback_list)
if __name__ == '__main__':
    train_dlrm()
