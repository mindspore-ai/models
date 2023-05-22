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
import os
import numpy as np
import mindspore as ms
from mindspore import Model, context, Tensor
import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from model.residual_attention_network import ResidualAttentionModel_92_32input_update
from model.residual_attention_network import ResidualAttentionModel_56
from src.eval_callback import EvalCallBack
from src.lr_generator import get_lr
from src.model_utils.config import config
from src.dataset import create_dataset1, create_dataset2
if config.enable_modelarts:
    from src.moxing_adapter import get_device_id, get_device_num, get_rank_id
else:
    from src.local_adapter import get_device_id, get_device_num, get_rank_id

if config.dataset == "cifar10":
    ResidualAttentionModel = ResidualAttentionModel_92_32input_update
else:
    ResidualAttentionModel = ResidualAttentionModel_56

def lr_steps_cifar10(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.9 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr_each_step.append(lr_max)
        elif i < decay_epoch_index[1]:
            lr_each_step.append(lr_max * 0.1)
        elif i < decay_epoch_index[2]:
            lr_each_step.append(lr_max * 0.01)
        else:
            lr_each_step.append(lr_max * 0.001)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate

def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    eval_net = eval_param["net"]
    eval_net.set_train(False)
    res = eval_model.eval(eval_ds)
    eval_net.set_train(True)
    return res[metrics_name]

def run_eval(test_dataset, model, net, ckpt_save_dir, cb, rank):
    """run_eval"""
    eval_param_dict = {"model": model, "net": net, "dataset": test_dataset, "metrics_name": "acc"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, rank, interval=1,
                           eval_start_epoch=1, save_best_ckpt=True,
                           ckpt_directory=ckpt_save_dir, best_ckpt_name="best_acc3.ckpt",
                           metrics_name="acc")
    cb += [eval_cb]

def train():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    device_num = get_device_num()
    if config.enable_modelarts:
        import moxing as mox
        obs_data_url = config.data_url
        config.data_url = '/home/work/user-job-dir/data/'
        DATA_DIR = config.data_url
        if not os.path.exists(config.data_url):
            os.mkdir(config.data_url)
        mox.file.copy_parallel(obs_data_url, config.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, config.data_url))
    else:
        DATA_DIR = config.data_path
    if config.device_target == 'Ascend':
        device_id = get_device_id()
        if device_num == 1:
            context.set_context(device_id=config.device_id)
        else:
            context.set_context(device_id=device_id)
        if device_num > 1:
            init()
            device_num = get_device_num()
            rank = get_rank_id()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            rank = 0
    else:
        rank = get_rank_id()
    if config.dataset == 'cifar10':
        train_dataset = create_dataset1(dataset_path=DATA_DIR, do_train=True,
                                        batch_size=config.batch_size,
                                        target=config.device_target)
        test_dataset = create_dataset1(dataset_path=DATA_DIR, do_train=False,
                                       batch_size=config.batch_size,
                                       target=config.device_target)
    elif config.dataset == "imagenet":
        if config.enable_modelarts:
            DATA_DIR = os.path.join(DATA_DIR, 'imagenet')
        train_pth = os.path.join(DATA_DIR, 'train')
        test_pth = os.path.join(DATA_DIR, 'val')
        train_dataset = create_dataset2(dataset_path=train_pth, do_train=True,
                                        batch_size=config.batch_size,
                                        target=config.device_target)
        test_dataset = create_dataset2(dataset_path=test_pth, do_train=False,
                                       batch_size=config.batch_size,
                                       target=config.device_target)
    else:
        raise ValueError("Unsupported dataset.")
    batch_num = train_dataset.get_dataset_size()
    net = ResidualAttentionModel()
    if config.dataset == 'cifar10':
        lr = lr_steps_cifar10(0, lr_max=config.lr, total_epochs=config.epoch_size, steps_per_epoch=batch_num)
    else:
        lr = ms.Tensor(get_lr(lr_init=config.lr_init, lr_end=config.lr_end,
                              lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                              total_epochs=config.epoch_size, steps_per_epoch=batch_num,
                              lr_decay_mode=config.lr_decay_mode))
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(net.get_parameters(), learning_rate=Tensor(lr),
                            momentum=config.momentum, use_nesterov=True,
                            weight_decay=config.weight_decay)
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={'acc'})
    workroot = '/home/work/user-job-dir'
    train_dir = workroot + '/model/'
    if config.enable_modelarts:
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        ckpt_save_dir = os.path.join(train_dir, 'ckpt_' + str(rank) + '/')
    else:
        ckpt_save_dir = "./"
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor()
    ckpt_name = "mindspore-rank"+str(rank)
    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num*config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=ckpt_name, directory=ckpt_save_dir, config=config_ck)
    cb = [time_cb, loss_cb, ckpoint_cb]
    if config.save_best_ckpt:
        run_eval(test_dataset, model, net, ckpt_save_dir, cb, rank)
    model.train(config.epoch_size, train_dataset, dataset_sink_mode=config.dataset_sink_mode, callbacks=cb)
    print("train_finnish")
    if config.enable_modelarts:
        import moxing as mox
        obs_train_url = config.train_url
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))

if __name__ == '__main__':
    set_seed(1)
    train()
