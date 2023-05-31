# Copyright 2020 Huawei Technologies Co., Ltd
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
"""train distribute on parameter server."""


import os
import sys
import mindspore.dataset as ds
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.common import set_seed

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.model_utils.config import config as cfg
from src.model_utils.moxing_adapter import moxing_wrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_wide_deep_net(config):
    """
    Get network of wide&deep model.
    """
    wide_deep_net = WideDeepModel(config)
    loss_net = NetWithLossClass(wide_deep_net, config)
    train_net = TrainStepWrap(loss_net, parameter_server=bool(config.parameter_server),
                              sparse=config.sparse, cache_enable=(config.vocab_cache_size > 0))
    eval_net = PredictWithSigmoid(wide_deep_net)
    return train_net, eval_net


class ModelBuilder():
    """
    ModelBuilder
    """

    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_train_hook(self):
        hooks = []
        callback = LossCallBack()
        hooks.append(callback)
        if int(os.getenv('DEVICE_ID')) == 0:
            pass
        return hooks

    def get_net(self, config):
        return get_wide_deep_net(config)


def train_and_eval(config):
    """
    test_train_eval
    """
    set_seed(1000)
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    parameter_server = bool(config.parameter_server)
    if cache_enable:
        config.full_batch = True
    print("epochs is {}".format(epochs))
    if config.full_batch and os.getenv("MS_ROLE") == "MS_WORKER":
        context.set_auto_parallel_context(full_batch=True)
        ds.config.set_seed(1)
        ds_train = create_dataset(data_path, train_mode=True,
                                  batch_size=batch_size*get_group_size(), data_type=dataset_type)
        ds_eval = create_dataset(data_path, train_mode=False,
                                 batch_size=batch_size*get_group_size(), data_type=dataset_type)
    else:
        ds_train = create_dataset(data_path, train_mode=True,
                                  batch_size=batch_size, rank_id=get_rank(),
                                  rank_size=get_group_size(), data_type=dataset_type)
        ds_eval = create_dataset(data_path, train_mode=False,
                                 batch_size=batch_size, rank_id=get_rank(),
                                 rank_size=get_group_size(), data_type=dataset_type)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()

    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()
    auc_metric = AUCMetric()

    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    if cache_enable:
        config.stra_ckpt = os.path.join(config.stra_ckpt + "-{}".format(get_rank()), "strategy.ckpt")
        context.set_auto_parallel_context(strategy_ckpt_save_file=config.stra_ckpt)

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    callback = LossCallBack(config=config)
    ms_role = os.getenv("MS_ROLE")
    if ms_role == "MS_WORKER":
        if cache_enable:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size()*epochs,
                                          keep_checkpoint_max=1, integrated_save=False)
        else:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    else:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train',
                                 directory=config.ckpt_path + '/ckpt_' + str(get_rank()) + '/',
                                 config=ckptconfig)
    callback_list = [TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback]
    if get_rank() == 0:
        callback_list.append(ckpoint_cb)
    model.train(epochs, ds_train,
                callbacks=callback_list,
                dataset_sink_mode=(parameter_server and cache_enable))


def modelarts_pre_process():
    cfg.ckpt_path = cfg.output_path

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
cache_enable = cfg.vocab_cache_size > 0


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_wide_and_deep():
    """ train_wide_and_deep """
    if cache_enable and cfg.device_target != "GPU":
        context.set_context(max_device_memory="24GB")
    context.set_ps_context(enable_ps=True)
    init()
    context.set_context(save_graphs_path='./graphs_of_device_id_'+str(get_rank()))

    if cache_enable:
        if os.getenv("MS_ROLE") == "MS_WORKER":
            context.set_auto_parallel_context(
                parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    else:
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=get_group_size())
        cfg.sparse = True

    if cfg.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    train_and_eval(cfg)

if __name__ == "__main__":
    train_wide_and_deep()
