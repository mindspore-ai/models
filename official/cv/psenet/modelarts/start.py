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
import os
import operator
import argparse
import subprocess
import ast
from src.dataset import train_dataset_creator
from src.PSENET.psenet import PSENet
from src.PSENET.dice_loss import DiceLoss
from src.network_define import WithLossCell, TrainOneStepCell, LossCallBack
from src.lr_schedule import dynamic_lr
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id
from src.model_utils.config import config
import moxing as mox
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed


parser = argparse.ArgumentParser('mindspore psenet training')
# output url
parser.add_argument('--train_url', type=str, default='',
                    help='where training log and ckpts saved')
# dataset url
parser.add_argument('--data_url', type=str, default='',
                    help='path of dataset')

parser.add_argument('--enable_modelarts', type=str, default=True,
                    help='enable_modelarts')
parser.add_argument('--run_distribute', type=str, default=True,
                    help='run_distribute')
parser.add_argument('--TRAIN_MODEL_SAVE_PATH', type=str, default='/cache/train/outputs/',
                    help='TRAIN_MODEL_SAVE_PATH')
parser.add_argument('--TRAIN_ROOT_DIR', type=str, default='/cache/data/ic15/',
                    help='TRAIN_ROOT_DIR')
parser.add_argument('--pre_trained', type=str, default='',
                    help='pre_trained')

parser.add_argument('--TRAIN_REPEAT_NUM', type=int, default=1,
                    help='TRAIN_REPEAT_NUM')


args = parser.parse_args()
mox.file.copy_parallel(args.data_url, '/cache')
config.enable_modelarts = args.enable_modelarts
config.TRAIN_MODEL_SAVE_PATH = args.TRAIN_MODEL_SAVE_PATH
config.TRAIN_ROOT_DIR = args.TRAIN_ROOT_DIR
config.pre_trained = args.pre_trained
config.TRAIN_REPEAT_NUM = args.TRAIN_REPEAT_NUM

set_seed(1)

binOps = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod
}


def arithmeticeval(s):
    node = ast.parse(s, mode='eval')

    def _eval(node):
        if isinstance(node, ast.BinOp):
            return binOps[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        raise Exception('unsupported type{}'.format(node))

    return _eval(node.body)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=device_target,
                        device_id=get_device_id())

    rank_id = 0
    config.BASE_LR = arithmeticeval(config.BASE_LR)
    config.WARMUP_RATIO = arithmeticeval(config.WARMUP_RATIO)

    device_num = get_device_num()
    if config.run_distribute:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
        if device_target == 'Ascend':
            rank_id = get_rank_id()
        else:
            rank_id = get_rank()

    # dataset/network/criterion/optim
    ds = train_dataset_creator(rank_id, device_num)
    step_size = ds.get_dataset_size()
    print('Create dataset done!')

    config.INFERENCE = False
    net = PSENet(config)
    net = net.set_train()

    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict, strict_load=True)
        print('Load Pretrained parameters done!')

    criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)

    lrs = dynamic_lr(config.BASE_LR, config.TRAIN_TOTAL_ITER,
                     config.WARMUP_STEP, config.WARMUP_RATIO)
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs,
                 momentum=0.99, weight_decay=5e-4)

    # warp model
    net = WithLossCell(net, criterion)
    if config.run_distribute:
        net = TrainOneStepCell(net, opt, reduce_flag=True, mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net, opt)

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(per_print_times=10)
    # set and apply parameters of check point config.TRAIN_MODEL_SAVE_PATH
    ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=3)
    ckpoint_cb = ModelCheckpoint(prefix="PSENet",
                                 config=ckpoint_cf,
                                 directory="{}/ckpt_{}".format(config.TRAIN_MODEL_SAVE_PATH,
                                                               rank_id))

    model = Model(net)
    model.train(config.TRAIN_REPEAT_NUM,
                ds,
                dataset_sink_mode=True,
                callbacks=[time_cb, loss_cb, ckpoint_cb])


if __name__ == '__main__':
    train()

    print('training is over')

    export_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "export.py")

    cmd = ['python', export_file,
           f'--ckpt=/cache/train/outputs/ckpt_0/PSENet-1_250.ckpt',
           f'--file_name=/cache/train/output/psenet',
           f'--file_format=AIR']
    process = subprocess.Popen(cmd, shell=False)
    process.wait()
    mox.file.copy_parallel('/cache/train', args.train_url)
