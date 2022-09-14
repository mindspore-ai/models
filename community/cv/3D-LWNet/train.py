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
"""
######################## train net ########################
"""
import os
import mindspore
from mindspore import context, set_seed, Model, load_checkpoint, load_param_into_net
from mindspore.common import initializer
from mindspore.nn import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
import mindspore.nn as nn
from src.models_lw_3D import dict_lwnet
from src.data_preprocess import create_dataset
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.loss import NllLoss


set_seed(1)


class EvalCallback(Callback):
    """
    Evaluation per epoch, and save the best accuracy checkpoint.
    """
    def __init__(self, model, eval_ds, begin_eval_epoch=1, save_path="./"):
        self.model = model
        self.eval_ds = eval_ds
        self.begin_eval_epoch = begin_eval_epoch
        self.best_acc = 0
        self.save_path = save_path

    def epoch_end(self, run_context):
        """
        evaluate at epoch end.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.begin_eval_epoch:
            res = self.model.eval(self.eval_ds, dataset_sink_mode=False)
            acc = res["accuracy"]
            if acc > self.best_acc:
                self.best_acc = acc
                mindspore.save_checkpoint(cb_params.train_network, os.path.join(self.save_path, "best_acc.ckpt"))
                print("the best epoch is", cur_epoch, "best acc is", self.best_acc)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def set_parameter():
    """set_parameter"""
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)


def init_weight(net):
    """init_weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv3d):
            cell.weight.set_data(initializer.initializer(initializer.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
    return net


@moxing_wrapper()
def train_net():
    """train net"""
    set_parameter()
    trained_model_dir = os.path.join(config.checkpoint_path, 'train_{}/'.format(config.dataset_name))

    data_train_dir = './data_list/{}_train.txt'.format(config.dataset_name)
    data_test_dir = './data_list/{}_val.txt'.format(config.dataset_name)
    train_dataset = create_dataset(config, data_train_dir)
    test_dataset = create_dataset(config, data_test_dir)
    step_size = train_dataset.get_dataset_size()

    model = dict_lwnet()[config.model_name](num_classes=config.class_num, dropout_keep_prob=0)

    if config.pre_trained:
        param_dict = load_checkpoint(config.load_path)
        filter_list = [x.name for x in model.fc.get_parameters()]
        filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(model, param_dict)
    else:
        model = init_weight(model)

    lr = nn.dynamic_lr.exponential_decay_lr(
        config.lr, config.warm, config.epoch_size * step_size, step_size, config.low_begin_epoch, is_stair=True)

    decayed_params = []
    no_decayed_params = []
    for param in model.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': model.trainable_params()}]
    opt = Momentum(group_params, lr, config.momentum)

    loss = NllLoss(reduction="mean", num_classes=config.class_num)

    net = Model(model, loss_fn=loss, optimizer=opt, metrics={'accuracy'})

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor(per_print_times=step_size)

    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="3D-LwNet", directory=trained_model_dir, config=config_ck)
        cb += [ckpt_cb]

    eval_callback = EvalCallback(net, test_dataset, save_path=trained_model_dir)
    cb += [eval_callback]

    net.train(config.epoch_size, train_dataset, callbacks=cb, sink_size=step_size, dataset_sink_mode=False)


if __name__ == '__main__':
    train_net()
