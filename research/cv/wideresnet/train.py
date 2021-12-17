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
"""
#################train WideResNet example on cifar10########################
python train.py
"""
import os
import numpy as np

from mindspore.train.summary.summary_record import SummaryRecord
from mindspore.common import set_seed
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.callbacks import CustomLossMonitor, TimeMonitor, EvalCallback
from src.wide_resnet import wideresnet
from src.dataset import create_dataset
from src.model_utils.config import config as cfg
from src.generator_lr import get_lr
from src.cross_entropy_smooth import CrossEntropySmooth

set_seed(1)

def get_device_num():
    _device_num = os.getenv('RANK_SIZE', '1')
    return int(_device_num)

def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

if __name__ == '__main__':
    target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if target == "Ascend":
        device_num = int(os.getenv('RANK_SIZE'))
    else: #GPU
        device_num = get_device_num()

    if cfg.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    dataset_sink_mode = True

    if cfg.use_summary_collector:
        dataset_sink_mode = False

    if cfg.modelart:
        import moxing as mox
        data_path = '/cache/data_path'
        mox.file.copy_parallel(src_url=cfg.data_url, dst_url=data_path)
    else:
        data_path = cfg.data_path

    ds_train = create_dataset(dataset_path=data_path,
                              do_train=True,
                              batch_size=cfg.batch_size,
                              target=target,
                              infer_910=cfg.infer_910,
                              distribute=cfg.run_distribute)
    if cfg.run_eval:
        ds_eval = create_dataset(dataset_path=cfg.eval_data_path,
                                 do_train=False,
                                 batch_size=cfg.batch_size,
                                 target=target,
                                 infer_910=cfg.infer_910)
    step_size = ds_train.get_dataset_size()

    net = wideresnet()
    #Init weights
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.pre_trained)
        if cfg.filter_weight:
            filter_list = [x.name for x in net.linear.get_parameters()] # clean the classifier head
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(
                    weight_init.XavierUniform(gain=np.sqrt(2)),
                    cell.weight.shape,
                    cell.weight.dtype))

    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=cfg.label_smooth_factor,
                              num_classes=cfg.num_classes,
                              )
    loss_scale = FixedLossScaleManager(loss_scale=cfg.loss_scale, drop_overflow_update=False)

    lr = get_lr(total_epochs=cfg.epoch_size, steps_per_epoch=step_size, lr_init=cfg.lr_init)
    lr = Tensor(lr)

    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    metrics = {"top_1_accuracy"}

    group_params = [{'params': decayed_params, 'weight_decay': cfg.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params,
                   learning_rate=lr,
                   momentum=cfg.momentum,
                   loss_scale=cfg.loss_scale,
                   use_nesterov=True,
                   weight_decay=cfg.weight_decay)

    model = Model(net,
                  amp_level="O2",
                  loss_fn=loss,
                  optimizer=opt,
                  loss_scale_manager=loss_scale,
                  metrics=metrics,
                  keep_batchnorm_fp32=False,
                  )
    if cfg.run_eval:
        new_net = wideresnet()
        eval_model = Model(new_net,
                           loss_fn=loss,
                           metrics=metrics,
                           )

    output_path = os.path.join(cfg.output_path, "exp_" + cfg.experiment_label)
    ckpt_save_dir = output_path + cfg.save_checkpoint_path
    summary_save_dir = output_path + cfg.summary_dir
    if cfg.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
        summary_save_dir = summary_save_dir + "device_" + str(get_rank()) + "/"

    with SummaryRecord(summary_save_dir) as summary_record:

        loss_cb = CustomLossMonitor(summary_record=summary_record,
                                    mode="train",
                                    frequency=cfg.collection_freq)
        time_cb = TimeMonitor()
        cb = [loss_cb, time_cb]
        if cfg.save_checkpoint:
            config_ck = \
                CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_size,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="wideresnet",
                                      directory=ckpt_save_dir,
                                      config=config_ck)
            cb += [ckpt_cb]

        if cfg.run_eval:
            eval_save_cb = EvalCallback(eval_model, model, ds_eval, ckpt_save_dir, cfg.modelart, summary_record, 10, 5)
            cb += [eval_save_cb]

        model.train(epoch=cfg.epoch_size, train_dataset=ds_train, callbacks=cb,
                    dataset_sink_mode=dataset_sink_mode)
