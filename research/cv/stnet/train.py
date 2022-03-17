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
"""train"""
import os
from pprint import pprint
from time import time

import numpy as np

import mindspore
from mindspore import Model, context, Parameter
from mindspore.nn import Accuracy
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, \
                                     LossMonitor, TimeMonitor, LearningRateScheduler
from mindspore.common import set_seed
from mindspore.dataset import config
from mindspore.train.callback import Callback
from mindspore.train.callback import SummaryCollector
from mindspore.profiler import Profiler

from src.config import config as cfg
from src.dataset import create_dataset
from src import Stnet_Res_model
from src.eval_callback import EvalCallBack
from src.model_utils.moxing_adapter import moxing_wrapper
from src.CrossEntropySmooth import CrossEntropySmooth


class StopAtStep(Callback):
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(output_path=cfg.summary_dir, start_profile=False)
    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
    def end(self, run_context):
        self.profiler.analyse()


def learning_rate_function(lr, cur_step_num):
    steps_size = cfg.steps_size
    if (cur_step_num // steps_size) % cfg.lr_decay_rate == 0:
        lr = lr * 0.1
    return lr


def change_weights(model, state):
    """
        The pretrained params are ResNet50 pretrained on ImageNet.
        However, conv1_weights' shape of StNet is not the same as that in ResNet50 because the input are super-image
        concatanated by a series of images. So it is recommendated to treat conv1_weights specifically.
        The process is as following:
          1, load params from pretrain
          2, get the value of conv1_weights in the state_dict and transform it
          3, set the transformed value to conv1_weights in prog
    """
    pretrained_dict = {}
    for name, _ in state.items():
        if "xception" in name:
            continue
        if "temp1" in name or "temp2" in name:
            continue
        if name.startswith("conv1"):
            state[name] = state[name].mean(axis=1, keep_dims=True) / model.N
            pretrained_dict[name] = np.repeat(state[name], model.N * 3, axis=1)
            pretrained_dict[name] = Parameter(pretrained_dict[name], requires_grad=True)
        else:
            pretrained_dict[name] = state[name]
            pretrained_dict[name] = Parameter(pretrained_dict[name], requires_grad=True)
    return pretrained_dict


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def run_eval(model, ckpt_save_dir, cb):
    """run_eval"""
    if cfg.run_eval:
        eval_dataset = create_dataset(data_dir=cfg.dataset_path, config=cfg, shuffle=False, do_trains='val',
                                      num_worker=cfg.workers, list_path=cfg.local_val_list)
        eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=cfg.eval_interval,
                               eval_start_epoch=cfg.eval_start_epoch, save_best_ckpt=cfg.save_best_ckpt,
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="acc")
        cb += [eval_cb]


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = cfg.checkpoint_path

    if cfg.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "_" + str(get_rank()) + "/"

    return ckpt_save_dir


def set_summary_dir():
    """set summary dir"""
    summary_dir = cfg.summary_dir

    if cfg.run_distribute:
        summary_dir = summary_dir + "/rank_" + str(get_rank()) + "/"

    return summary_dir


def set_parameter():
    """set_parameter"""
    # init context
    if cfg.run_distribute:
        if cfg.mode == 'GRAPH':
            context.set_context(mode=context.GRAPH_MODE, device_target=cfg.target, save_graphs=False)
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.target, save_graphs=False)

        if cfg.target == "Ascend":
            init()
            cfg.device_num = get_group_size()
            cfg.rank_id = get_rank()
            context.set_auto_parallel_context(device_num=cfg.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              parameter_broadcast=True)
        elif cfg.target == "GPU":
            init('nccl')
            cfg.device_num = get_group_size()
            cfg.rank_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=cfg.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        else:
            raise NotImplementedError("Only GPU and Ascend training supported")

    else:
        if cfg.mode == 'GRAPH':
            context.set_context(mode=context.GRAPH_MODE, device_target=cfg.target, save_graphs=False,
                                device_id=cfg.device_id)
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.target, save_graphs=False,
                                device_id=cfg.device_id)

    config.set_enable_shared_mem(False) # we may get OOM when it set to 'True'


@moxing_wrapper()
def run_train():
    """train function"""

    if cfg.profile:
        print("INFO: performance profiling mode enabled")
        print("Profiling mode is performed only for 1 epoch with dataset_sink_mode=False")
        cfg.dataset_sink_mode = False
        cfg.num_epochs = 1

    set_parameter()

    dataset_path = cfg.dataset_path    # label path

    # Load the data
    print(f'Loading the data... {dataset_path}')

    video_datasets_train = create_dataset(data_dir=dataset_path, config=cfg, do_trains='train',
                                          num_worker=cfg.workers,
                                          list_path=cfg.local_train_list)

    print('Starting to training...')
    step_size_train = video_datasets_train.get_dataset_size()
    print('The size of training set is {}'.format(step_size_train))
    cfg.steps_size = step_size_train

    pprint(cfg)

    # define net
    net = Stnet_Res_model.stnet50(input_channels=3, num_classes=cfg.class_num, T=cfg.T, N=cfg.N)

    # load pretrain_resnet50
    if cfg.pre_res50 or cfg.pre_res50_art_load_path:
        path = cfg.pre_res50
        if cfg.run_online:
            path = cfg.pre_res50_art_load_path
        if os.path.isfile(path):
            net_parmerters = load_checkpoint(path)
            net_parmerters = change_weights(net, net_parmerters)
            load_param_into_net(net, net_parmerters, strict_load=True)
        else:
            raise RuntimeError('no such file{}'.format(path))

    # load pretrain model
    if cfg.resume or cfg.best_acc_art_load_path:
        resume = os.path.join(cfg.resume)
        if cfg.run_online:
            resume = cfg.best_acc_art_load_path
        if os.path.isfile(resume):
            net_parmerters = load_checkpoint(resume)
            load_param_into_net(net, net_parmerters, strict_load=True)
        else:
            raise RuntimeError('no such file{}'.format(resume))

    # define loss function
    loss = CrossEntropySmooth(sparse=True, reduction='mean', num_classes=cfg.class_num)

    # lr
    lr = cfg.lr
    optimizer_ft = mindspore.nn.Momentum(params=net.trainable_params(), learning_rate=lr,
                                         momentum=cfg.momentum,
                                         weight_decay=cfg.weight_decay)

    # define model
    model = Model(net, loss_fn=loss, optimizer=optimizer_ft, metrics={'acc': Accuracy()})

    # define callback
    callback = []
    time_cb = TimeMonitor(data_size=1)
    loss_cb = LossMonitor(1)
    lr_scheduler = LearningRateScheduler(learning_rate_function)
    callback.append(time_cb)
    callback.append(loss_cb)
    callback.append(lr_scheduler)

    ckpt_save_dir = set_save_ckpt_dir()
    if cfg.save_checkpoint:
        cfg_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_size_train,
                                  keep_checkpoint_max=cfg.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="stnet", directory=ckpt_save_dir, config=cfg_ck)
        callback.append(ckpt_cb)

    # Init a SummaryCollector callback instance, and use it in model.train or model.eval
    summary_dir = set_summary_dir()
    summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1)
    callback.append(summary_collector)

    if cfg.profile:
        if cfg.dataset_sink_mode:
            print("Dataset sink mode is not recommended while profiling, switching it to False")
            cfg.dataset_sink_mode = False
        profiler_cb = StopAtStep(start_step=10, stop_step=20)
        callback.append(profiler_cb)

    run_eval(model, ckpt_save_dir, callback)  # setting 'run_eval' config to True slow down train

    # train
    print("\nStarted training")

    start_time = time()
    model.train(cfg.num_epochs, video_datasets_train,
                callbacks=callback, dataset_sink_mode=cfg.dataset_sink_mode)
    total_time = time() - start_time
    print(f"\nFinished training. Time taken: {int(total_time) // 60} min {int(total_time) % 60} sec")


if __name__ == '__main__':
    set_seed(1)
    run_train()
