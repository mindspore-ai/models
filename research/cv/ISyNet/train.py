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
"""train IsyNet."""
import os
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.eval_callback import EvalCallBack
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num
from src.KLLossAscend import KLwithCELoss
from src.dataset import create_dataset2 as create_dataset
from src.adamw import AdamW
from ISyNet.model import ISyNet

if config.enable_modelarts:
    import moxinf as mox
set_seed(1)


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

def set_parameter():
    """set_parameter"""
    target = config.device_target
    if target == "CPU":
        config.run_distribute = False

    # init context
    if config.mode_name == 'GRAPH':
        context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    if config.parameter_server:
        context.set_ps_context(enable_ps=True)
    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            init()
        # GPU target
        else:
            init()
            context.set_auto_parallel_context(device_num=get_device_num(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    """conv init"""
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def init_weight(net):
    """init_weight"""
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        if config.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if config.conv_init == "XavierUniform":
                    cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                                 cell.weight.shape,
                                                                 cell.weight.dtype))
                elif config.conv_init == "HeNormal":
                    cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                                 cell.weight.shape,
                                                                 cell.weight.dtype))
                elif config.conv_init == "TruncatedNormal":
                    weight = conv_variance_scaling_initializer(cell.in_channels,
                                                               cell.out_channels,
                                                               cell.kernel_size[0])
                    cell.weight.set_data(weight)
            if isinstance(cell, nn.Dense):
                if config.dense_init == "TruncatedNormal":
                    cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                                 cell.weight.shape,
                                                                 cell.weight.dtype))
                elif config.dense_init == "RandomNormal":
                    in_channel = cell.in_channels
                    out_channel = cell.out_channels
                    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                    weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                    cell.weight.set_data(weight)

def init_lr(step_size):
    """init lr"""
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    return lr

def init_loss_scale():
    """init loss scale"""
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        if config.dml > 1:
            loss = KLwithCELoss(sparse=True, reduction="mean",
                                smooth_factor=config.label_smooth_factor, num_classes=config.class_num,
                                dml=config.dml)
        else:
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


def init_group_params(net):
    """init group params"""
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        #print(param)
        if 'beta' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params

def run_eval(target, model, ckpt_save_dir, cb):
    """run_eval"""
    if config.run_eval:
        if config.eval_dataset_path is None or (not os.path.isdir(config.eval_dataset_path)):
            raise ValueError(f'{config.eval_dataset_path} is not a existing path.')
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False,
                                      batch_size=config.batch_size, target=target, enable_cache=config.enable_cache,
                                      cache_session_id=config.cache_session_id)
        eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=config.save_best_ckpt,
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="acc")
        cb += [eval_cb]


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    if config.enable_modelarts and config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank_id()) + "/"
    else:
        if config.run_distribute:
            ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


def train_net():
    """train net"""
    target = config.device_target
    set_parameter()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target,
                             distribute=config.run_distribute,
                             autoaugment=config.autoaugment,
                             randaugment=config.randaugment)
    step_size = dataset.get_dataset_size()
    net = ISyNet(num_classes=config.class_num,
                 json_arch_file_backbone=config.jsonFile if config.dml < 2 \
                 else [config.jsonFile, config.jsonFileTeacher],
                 dropout=config.dropout,
                 weight_standardization=config.weight_standardization,
                 last_bn=config.lastbn,
                 dml=config.dml)
    if config.parameter_server:
        net.set_param_ps()
    init_weight(net=net)
    lr = Tensor(init_lr(step_size=step_size))
    # define opt
    group_params = init_group_params(net)
    if config.optimizer == 'adamw':
        opt = AdamW(group_params, lr, config.beta1, config.beta2, loss_scale=config.loss_scale)
    else:
        opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    loss = init_loss_scale()
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    dist_eval_network = ClassifyCorrectCell(net) if config.run_distribute else None
    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                  amp_level=config.acc_mode, eval_network=dist_eval_network)


    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    ckpt_save_dir = set_save_ckpt_dir()
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="isynet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    run_eval(target, model, ckpt_save_dir, cb)
    # train model
    dataset_sink_mode = (not config.parameter_server) and target != "CPU"
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
    if config.enable_modelarts:
        if config.save_checkpoint:
            #import moxing as mox
            ckpts = os.listdir(ckpt_save_dir)
            for ckpt in ckpts:
                abs_ckpt = os.path.join(ckpt_save_dir, ckpt)
                roma_weights_fp = os.path.join(config.train_url, config.checkpoint_path, ckpt)
                roma_weights_dirname = os.path.dirname(roma_weights_fp)
                if not mox.file.exists(roma_weights_dirname):
                    mox.file.make_dirs(roma_weights_dirname)
                mox.file.copy(abs_ckpt, roma_weights_fp)


if __name__ == '__main__':
    train_net()
