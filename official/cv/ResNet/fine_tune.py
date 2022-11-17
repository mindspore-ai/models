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
"""train resnet34."""

import os
import mindspore as ms
import mindspore.nn as nn

from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from train import LossCallBack
from src.resnet import resnet34
from src.dataset import create_dataset2
from src.model_utils.config import config
from src.eval_callback import EvalCallBack


ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, save_graphs=False)


def import_data():
    dataset_train = create_dataset2(dataset_path=os.path.join(config.data_path, "train"), do_train=True,
                                    batch_size=config.batch_size, train_image_size=config.train_image_size,
                                    eval_image_size=config.eval_image_size, target=config.device_target,
                                    distribute=False, enable_cache=False, cache_session_id=None)
    dataset_val = create_dataset2(dataset_path=os.path.join(config.data_path, "test"), do_train=True,
                                  batch_size=config.batch_size, train_image_size=config.train_image_size,
                                  eval_image_size=config.eval_image_size, target=config.device_target,
                                  distribute=False, enable_cache=False, cache_session_id=None)
    #
    data = next(dataset_train.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    print("Tensor of image", images.shape)
    print("Labels:", labels)

    return dataset_train, dataset_val


# define the head layer
class DenseHead(nn.Cell):
    def __init__(self, input_channel, num_classes):
        super(DenseHead, self).__init__()
        self.dense = nn.Dense(input_channel, num_classes)

    def construct(self, x):
        return self.dense(x)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def init_weight(net, param_dict):
    """init_weight"""
    if param_dict:
        if config.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        ms.load_param_into_net(net, param_dict)


def eval_net(net, dataset):
    """eval net"""
    net.set_train(False)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res)


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    print("res:", res)
    return res[metrics_name]


def run_eval(model, ckpt_save_dir, eval_dataset, cb):
    """run_eval"""
    eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "top_1_accuracy"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=1,
                           eval_start_epoch=0, save_best_ckpt=True,
                           ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                           metrics_name="acc")
    cb += [eval_cb]


def finetune_train():
    dataset_train, data_val = import_data()

    ckpt_param_dict = ms.load_checkpoint(config.checkpoint_path)
    net = resnet34(class_num=1001)
    init_weight(net=net, param_dict=ckpt_param_dict)
    print("net parameter:")
    for param in net.get_parameters():
        print("param:", param)

    # fully Connected layer the size of the input layer
    src_head = net.end_point
    in_channels = src_head.in_channels
    # the number of output channels is 5
    head = DenseHead(in_channels, config.class_num)
    # reset the fully connected layer
    net.end_point = head

    print("net.get_parameters():", net.get_parameters())
    print("net.trainable_params():", net.trainable_params())
    # freeze all parameters except the last layer
    for param in net.get_parameters():
        if param.name not in ["end_point.dense.weight", "end_point.dense.bias"]:
            param.requires_grad = False
        if param.name == "end_point.dense.weight":
            param.name = "end_point.weight"
        if param.name == "end_point.dense.bias":
            param.name = "end_point.bias"

    # define optimizer and loss function
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=config.learning_rate, momentum=config.momentum)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # instantiating the model
    model = Model(net, loss, opt, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # define callbacks
    step_size = dataset_train.get_dataset_size()
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(config.has_trained_epoch)
    cb = [time_cb, loss_cb]

    run_eval(model, "./", data_val, cb)

    num_epochs = config.epoch_size
    model.train(num_epochs, dataset_train, callbacks=cb)

    eval_net(net, data_val)


if __name__ == '__main__':
    finetune_train()
