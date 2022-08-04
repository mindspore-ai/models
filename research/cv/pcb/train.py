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
"""train.py"""

import sys
import os
import mindspore.nn as nn
from mindspore import context, set_seed
from mindspore.communication.management import init, get_rank
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode

from src.model_utils.config import config
from src.dataset import create_dataset
from src.pcb import PCB, NetWithLossCell
from src.rpp import RPP
from src.eval_utils import apply_eval
from src.eval_callback import EvalCallBack
from src.logging import Logger
from src.lr_generator import get_lr
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_rank_id, get_device_id

set_seed(config.seed)

def build_model(num_classes):
    """Build model"""
    model = None
    if config.model_name == "PCB":
        model = PCB(num_classes)
    elif config.model_name == "RPP":
        model = RPP(num_classes)
    return model

def init_group_params(model, train_batch_num):
    """Init group params"""
    group_params = None
    backbone_params = model.base.trainable_params()
    other_params = list(filter(lambda x: "base" not in x.name, model.trainable_params()))
    lr_backbone = get_lr(lr_init=0, lr_max=config.learning_rate * config.lr_mult, \
total_steps=train_batch_num * config.epoch_size, warmup_steps=train_batch_num * config.warmup_epoch_size, \
decay_steps=train_batch_num * config.decay_epoch_size)
    lr_other = get_lr(lr_init=0, lr_max=config.learning_rate, \
total_steps=train_batch_num * config.epoch_size, warmup_steps=train_batch_num * config.warmup_epoch_size, \
decay_steps=train_batch_num * config.decay_epoch_size)
    group_params = [{"params": backbone_params, "lr": lr_backbone}, \
{"params": other_params, "lr": lr_other}, \
{'order_params': model.trainable_params()}]
    return group_params

def set_log_save_dir():
    """Set log save dir"""
    log_save_dir = os.path.join(config.output_path, config.log_save_path)
    if config.enable_modelarts and config.run_distribute:
        log_save_dir = os.path.join(log_save_dir, 'train_log{}/'.format(get_rank_id()))
    else:
        if config.run_distribute:
            log_save_dir = os.path.join(log_save_dir, 'train_log{}/'.format(get_rank()))
    return log_save_dir

def set_ckpt_save_dir():
    """Set checkpoint save dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_save_path)
    if config.enable_modelarts and config.run_distribute:
        ckpt_save_dir = os.path.join(ckpt_save_dir, 'ckpt_{}/'.format(get_rank_id()))
    else:
        if config.run_distribute:
            ckpt_save_dir = os.path.join(ckpt_save_dir, 'ckpt_{}/'.format(get_rank()))
    return ckpt_save_dir

@moxing_wrapper()
def train_net():
    """train_net"""
    target = config.device_target
    # init context
    if config.mode_name == 'GRAPH':
        context.set_context(mode=context.GRAPH_MODE, device_target=target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = get_device_id()
        device_num = config.device_num
        context.set_context(device_id=device_id)
        if config.run_distribute:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, \
parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
    log_save_dir = set_log_save_dir()
    if not os.path.isdir(log_save_dir):
        os.makedirs(log_save_dir)
    sys.stdout = Logger(os.path.join(log_save_dir, 'log.txt'))
    #create dataset
    train_dataset, train_set = create_dataset(dataset_name=config.dataset_name, \
dataset_path=config.dataset_path, subset_name="train", batch_size=config.batch_size, \
num_parallel_workers=config.num_parallel_workers, distribute=config.run_distribute)
    train_batch_num = train_dataset.get_dataset_size()
    num_classes = train_set.num_ids
    #net
    network = build_model(num_classes)
    if config.checkpoint_file_path != "":
        print("loading checkpoint from " + config.checkpoint_file_path)
        param_dict = load_checkpoint(config.checkpoint_file_path)
        load_param_into_net(network, param_dict)
    #optimizer
    loss_scale = float(config.loss_scale)
    group_params = init_group_params(network, train_batch_num)
    optimizer = nn.SGD(group_params, momentum=float(config.momentum), \
weight_decay=float(config.weight_decay), nesterov=config.nesterov, loss_scale=loss_scale)
    net = NetWithLossCell(network)
    net = nn.TrainOneStepCell(net, optimizer, sens=loss_scale)
    model = Model(net)
    # define callbacks
    time_cb = TimeMonitor(data_size=train_batch_num)
    loss_cb = LossMonitor()
    callbacks = [time_cb, loss_cb]
    # checkpoint
    if config.save_checkpoint:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * train_batch_num, \
keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_save_dir = set_ckpt_save_dir()
        ckpoint_cb = ModelCheckpoint(prefix=config.model_name, directory=ckpt_save_dir, config=ckpt_config)
        callbacks += [ckpoint_cb]
    if config.save_checkpoint and config.run_eval:
        eval_network = network
        query_dataset, query_set = create_dataset(dataset_name=config.dataset_name, \
dataset_path=config.dataset_path, subset_name="query", batch_size=config.batch_size, \
num_parallel_workers=config.num_parallel_workers)
        gallery_dataset, gallery_set = create_dataset(dataset_name=config.dataset_name, \
dataset_path=config.dataset_path, subset_name="gallery", batch_size=config.batch_size, \
num_parallel_workers=config.num_parallel_workers)
        eval_param_dict = {"net": eval_network, "query_dataset": query_dataset, \
"gallery_dataset": gallery_dataset, "query_set": query_set.data, "gallery_set": gallery_set.data}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=ckpt_save_dir, best_ckpt_name="best.ckpt",
                               metrics_name=("mAP", "CMC"), cmc_topk=(1, 5, 10))
        callbacks += [eval_cb]
    dataset_sink_mode = False
    if config.sink_mode and config.device_target != "CPU":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start training {}, the first epoch will be slower because of the graph compilation.".format(\
config.model_name))
    model.train(config.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)

if __name__ == "__main__":
    train_net()
