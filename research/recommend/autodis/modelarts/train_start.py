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
"""train_criteo."""
import os
import sys
import datetime
import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.common import set_seed
from mindspore.train.serialization import export, load_checkpoint

from src.autodis import ModelBuilder, AUCMetric
from src.dataset_modelarts import create_dataset, DataType
from src.callback import EvalCallBack, LossCallBack
from src.model_utils.moxing_adapter_modelarts import moxing_wrapper
from src.model_utils.config_modelarts import config, train_config, data_config, model_config
from src.model_utils.device_adapter_modelarts import get_device_id, get_device_num, get_rank_id

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

set_seed(1)

def get_latest_ckpt():
    '''get latest ckpt'''
    ckpt_path = config.ckpt_path
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_path) if ckpt_file.endswith(".ckpt")]
    if not ckpt_files:
        return None
    latest_ckpt_file = sorted(ckpt_files)[-1]
    return os.path.join(ckpt_path, latest_ckpt_file)

def export_air_onxx():
    '''export air or onxx'''
    ckpt_file = get_latest_ckpt()
    if not ckpt_file:
        print("Not found ckpt file")
        return
    config.ckpt_file = ckpt_file
    config.file_name = os.path.join(config.ckpt_path, config.file_name)

    print("starting export AIR and ONNX")

    model_config.batch_size = 1
    model_builder = ModelBuilder(model_config, train_config)
    _, network = model_builder.get_train_eval_net()
    network.set_train(False)

    load_checkpoint(config.ckpt_file, net=network)

    data_config.batch_size = 1
    batch_ids = Tensor(np.zeros([data_config.batch_size, data_config.data_field_size]).astype(np.int32))
    batch_wts = Tensor(np.zeros([data_config.batch_size, data_config.data_field_size]).astype(np.float32))
    labels = Tensor(np.zeros([data_config.batch_size, 1]).astype(np.float32))
    input_data = [batch_ids, batch_wts, labels]

    config.file_format = "AIR"
    export(network, *input_data, file_name=config.file_name, file_format=config.file_format)
    config.file_format = "MINDIR"
    export(network, *input_data, file_name=config.file_name, file_format=config.file_format)
    # mox.file.copy(config.file_name+".air", config.ckpt_path+"/../autodis.air")


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.train_data_dir = config.data_path
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''convert data to mindrecord'''
    parser = argparse.ArgumentParser(description='Autodis')
    parser.add_argument('--data_url', type=str, default='')
    parser.add_argument('--train_url', type=str, default='')
    parser.add_argument('--train_epochs', type=int, default=15)
    args_opt, _ = parser.parse_known_args()
    train_config.train_epochs = args_opt.train_epochs
    # get data_url and train_url
    config.train_data_dir = args_opt.data_url
    config.ckpt_path = config.train_url

    # train function
    config.do_eval = config.do_eval == 'True'
    rank_size = get_device_num()
    if rank_size > 1:
        if config.device_target == "Ascend":
            device_id = get_device_id()
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
            rank_id = get_rank_id()
        else:
            print("Unsupported device_target ", config.device_target)
            exit()
    else:
        if config.device_target == "Ascend":
            device_id = get_device_id()
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
        else:
            print("Unsupported device_target ", config.device_target)
            exit()
        rank_size = None
        rank_id = None

    # Init Profiler
    data_config.data_format = 1
    ds_train = create_dataset(config.train_data_dir,
                              train_mode=True,
                              epochs=1,
                              batch_size=train_config.batch_size,
                              data_type=DataType(data_config.data_format),
                              rank_size=rank_size,
                              rank_id=rank_id)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    # steps_size = ds_train.get_dataset_size()

    model_builder = ModelBuilder(model_config, train_config)
    train_net, eval_net = model_builder.get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallBack(loss_file_path=config.loss_file_name)
    callback_list = [time_callback, loss_callback]

    if train_config.save_checkpoint:
        config.ckpt_path = os.path.join(config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        if rank_size:
            train_config.ckpt_file_name_prefix = train_config.ckpt_file_name_prefix + str(get_rank())
            config.ckpt_path = os.path.join(config.ckpt_path, 'ckpt_' + str(get_rank()) + '/')
        config_ck = CheckpointConfig(save_checkpoint_steps=train_config.save_checkpoint_steps,
                                     keep_checkpoint_max=train_config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=train_config.ckpt_file_name_prefix,
                                  directory=config.ckpt_path,
                                  config=config_ck)
        callback_list.append(ckpt_cb)

    if config.do_eval:
        ds_eval = create_dataset(config.train_data_dir, train_mode=False,
                                 epochs=1,
                                 batch_size=train_config.batch_size,
                                 data_type=DataType(data_config.data_format))
        eval_callback = EvalCallBack(model, ds_eval, auc_metric,
                                     eval_file_path=config.eval_file_name)
        callback_list.append(eval_callback)
    model.train(train_config.train_epochs, ds_train, callbacks=callback_list)
    export_air_onxx()

if __name__ == '__main__':
    run_train()
