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
python start.py
"""
import argparse
import os
import glob
import numpy as np

from src.loss import SoftmaxCrossEntropyExpand
from src.resnet_ibn import resnet50_ibn_a
from src.dataset import create_dataset_ImageNet as create_dataset, create_evalset
from src.lr_generator import lr_generator
from src.config import cfg

import mindspore.nn as nn
from mindspore import context
from mindspore import export
from mindspore import Tensor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init, get_rank
import moxing as mox

DATA_PATH = "/cache/data/"
EVAL_PATH = "/cache/eval/"
CKPT_PATH = "/cache/ckpt/"

parser = argparse.ArgumentParser(description='Mindspore ImageNet Training')

# Datasets
parser.add_argument('--train_url', default='', type=str, help='train path')
parser.add_argument('--data_url', default='', type=str, help='data path')
parser.add_argument('--train_path', default='/train/', type=str, help='train subpath')
parser.add_argument('--eval_path', default='/eval/', type=str, help='eval subpath')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--class_num', default=1000, type=int, metavar='N',
                    help='number of class')
parser.add_argument('--train_batch', default=64, type=int, metavar='N',
                    help='number of train batch')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='number of test batch')

# Device options
parser.add_argument('--device_target', type=str,
                    default='Ascend', choices=['GPU', 'Ascend'])
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()


class EvalCallBack(Callback):
    """
    Precision verification using callback function.
    """
    # define the operator required
    def __init__(self, models, eval_ds, epochs_per_eval, file_name):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_ds
        self.epochs_per_eval = epochs_per_eval
        self.file_name = file_name

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch > 90:
            acc = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["acc"].append(acc["Accuracy"])
            print(acc)

def frozen_to_air(input_net, input_args):
    """
    Frozen ckpt file to air file.
    """
    ckpt_param_dict = load_checkpoint(input_args.get("ckpt_file"))
    load_param_into_net(input_net, ckpt_param_dict)

    input_arr = Tensor(np.zeros([input_args.get("batch_size"), 3, input_args.get("height"),
                                 input_args.get("width")], np.float32))
    export(input_net, input_arr, file_name=input_args.get("file_name"), file_format=input_args.get("file_format"))


if __name__ == "__main__":
    ckpt_save_path = CKPT_PATH
    train_epoch = args.epochs
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target, save_graphs=False)
    device_id = args.device_id
    if args.device_num > 1:
        if target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
        elif target == 'GPU':
            init()
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
        ckpt_save_path = CKPT_PATH + "ckpt_" + str(get_rank()) + "/"
    else:
        context.set_context(device_id=device_id)

    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path, 0o755)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, 0o755)
    mox.file.copy_parallel(args.data_url + args.train_path, DATA_PATH)
    print("training data finish copy to %s." % DATA_PATH)
    train_dataset = create_dataset(dataset_path=DATA_PATH, do_train=True, repeat_num=1,
                                   batch_size=args.train_batch, target=target)

    if not os.path.exists(EVAL_PATH):
        os.makedirs(EVAL_PATH, 0o755)
    mox.file.copy_parallel(args.data_url + args.eval_path, EVAL_PATH)
    print("evaluating data finish copy to %s." % EVAL_PATH)
    eval_dataset = create_evalset(dataset_path=EVAL_PATH, do_train=False, repeat_num=1,
                                  batch_size=args.test_batch, target=target)

    net = resnet50_ibn_a(num_classes=args.class_num)
    criterion = SoftmaxCrossEntropyExpand(sparse=True)
    step = train_dataset.get_dataset_size()
    lr = lr_generator(cfg.lr, train_epoch, steps_per_epoch=step)
    optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr,
                       momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    model = Model(net, loss_fn=criterion, optimizer=optimizer,
                  metrics={"Accuracy": Accuracy()})

    config_ck = CheckpointConfig(
        save_checkpoint_steps=step, keep_checkpoint_max=cfg.keep_checkpoint_max)

    prefix = "IBNNet_" + str(device_id)
    ckpoint_cb = ModelCheckpoint(prefix=prefix, config=config_ck,
                                 directory=ckpt_save_path)
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, eval_dataset, epoch_per_eval, "ibn")
    cb = [ckpoint_cb, time_cb, loss_cb, eval_cb]
    if args.device_num == 1:
        model.train(train_epoch, train_dataset, callbacks=cb)
    elif args.device_num > 1 and get_rank() % 8 == 0:
        model.train(train_epoch, train_dataset, callbacks=cb)
    else:
        model.train(train_epoch, train_dataset)

    # find the newest ckpt file
    ckpt_list = glob.glob(CKPT_PATH + prefix + "*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    # frozen to ait file
    net = resnet50_ibn_a(num_classes=args.class_num)
    frozen_to_air_args = {'ckpt_file': ckpt_model,
                          'batch_size': 1,
                          'height': 224,
                          'width': 224,
                          'file_name': (CKPT_PATH + prefix),
                          'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)
    mox.file.copy_parallel(CKPT_PATH, args.train_url)
