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
train model
"""
import argparse
import os
import time
import ast
from datetime import datetime
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.nn import FixedLossScaleUpdateCell
from mindspore import context, load_checkpoint, load_param_into_net, export
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import RunContext, _InternalCallbackParam
from mindspore.context  import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from src.ecapa_tdnn import ECAPA_TDNN, Classifier
from src.reader import DatasetGeneratorBatch as DatasetGenerator
from src.util import AdditiveAngularMargin
from src.loss_scale import TrainOneStepWithLossScaleCellv2 as TrainOneStepWithLossScaleCell
from src.model_utils.config import config as hparams
from src.sampler import DistributedSampler

parser = argparse.ArgumentParser(description='ecapatdnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_url', type=str, default=None, help='Location of Data')
parser.add_argument('--train_url', type=str, default='', help='Location of training outputs')
parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=True, help='choose modelarts')
args, unknown = parser.parse_known_args()

def save_ckpt_to_air(save_ckpt_path, path):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    in_channels = 80
    channels = 1024
    emb_size = 192
    net = ECAPA_TDNN(in_channels, channels=[channels, channels, channels, channels, channels * 3],
                     lin_neurons=emb_size, global_context=False)

    # assert config.ckpt_file is not None, "config.ckpt_file is None."
    param_dict = load_checkpoint(path)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.ones([1, 301, 80]), ms.float32)
    export(net, input_arr, file_name=save_ckpt_path+'ecapatdnn', file_format="AIR")

def create_dataset(cfg, data_home, shuffle=False):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        data_home(string): the path of dataset.
        batch_size(int): the batch size of dataset.
        repeat_num(int): the repeat times of dataset. Default: 1
    Returns:
        dataset
    """

    dataset_generator = DatasetGenerator(data_home)
    distributed_sampler = None
    if cfg.run_distribute:
        distributed_sampler = DistributedSampler(len(dataset_generator), cfg.group_size, cfg.rank, shuffle=True)
    vox2_ds = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=shuffle, sampler=distributed_sampler)
    cnt = int(len(dataset_generator) / cfg.group_size)
    return vox2_ds, cnt

class CorrectLabelNum(nn.Cell):
    def __init__(self):
        super(CorrectLabelNum, self).__init__()
        self.argmax = ms.ops.Argmax(axis=1)
        self.sum = ms.ops.ReduceSum()

    def construct(self, output, target):
        output = self.argmax(output)
        correct = self.sum((output == target).astype(ms.dtype.float32))
        return correct

class BuildTrainNetwork(nn.Cell):
    '''Build train network.'''
    def __init__(self, my_network, classifier, lossfunc, my_criterion, train_batch_size, class_num_):
        super(BuildTrainNetwork, self).__init__()
        self.network = my_network
        self.classifier = classifier
        self.criterion = my_criterion
        self.lossfunc = lossfunc
        # Initialize self.output
        self.output = ms.Parameter(Tensor(np.ones((train_batch_size, class_num_)), ms.float32), requires_grad=False)
        self.onehot = ms.nn.OneHot(depth=class_num_, axis=-1, dtype=ms.float32)

    def construct(self, input_data, label):
        output = self.network(input_data)
        label_onehot = self.onehot(label)
        # Get the network output and assign it to self.output
        logits = self.classifier(output)
        output = self.lossfunc(logits, label_onehot)
        self.output = output
        loss0 = self.criterion(output, label_onehot)
        return loss0

def update_average(loss_, avg_loss, step):
    avg_loss -= avg_loss / step
    avg_loss += loss_ / step
    return avg_loss

def train_net(rank, model, epoch_max, data_train, ckpt_cb, steps_per_epoch,
              train_batch_size):
    """define the training method"""
    # Create dict to save internal callback object's parameters
    cb_params = _InternalCallbackParam()
    cb_params.train_network = model
    cb_params.epoch_num = epoch_max
    cb_params.batch_num = steps_per_epoch
    cb_params.cur_epoch_num = 0
    cb_params.cur_step_num = 0
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)
    if rank == 0:
        print("============== Starting Training ==============")
    correct_num = CorrectLabelNum()
    correct_num.set_train(False)

    for epoch in range(epoch_max):
        t_start = time.time()
        train_loss = 0
        avg_loss = 0
        train_loss_cur = 0
        train_correct_cur = 0
        train_correct = 0
        print_dur = 3000
        i = 0
        for idx, (data, gt_classes) in enumerate(data_train):
            i = i + 1
            if i == 1000:
                break
            model.set_train()
            batch_loss, _, _, output = model(data, gt_classes)
            correct = correct_num(output, gt_classes)
            train_loss += batch_loss
            train_correct += correct.sum()
            train_loss_cur += batch_loss
            avg_loss = update_average(batch_loss, avg_loss, idx+1)
            train_correct_cur += correct.sum()
            if rank == 0 and idx % print_dur == 0:
                cur_loss = train_loss_cur.asnumpy()
                acc = correct.sum().asnumpy() / float(train_batch_size)
                total_avg = train_loss.asnumpy() / float(idx+1)
                if idx > 0:
                    cur_loss = train_loss_cur.asnumpy()/float(print_dur)
                    acc = train_correct_cur.asnumpy() / float(train_batch_size *print_dur)
                print(f"{datetime.now()}, epoch:{epoch + 1}/{epoch_max}, iter-{idx}/{steps_per_epoch},"
                      f'cur loss:{cur_loss:.4f}, aver loss:{avg_loss.asnumpy():.4f},'
                      f'total_avg loss:{total_avg:.4f}, acc_aver:{acc:.4f}')
                train_loss_cur = 0
                train_correct_cur = 0
            # Update current step number
            cb_params.cur_step_num += 1
            # Check whether save checkpoint or not
            if rank == 0:
                ckpt_cb.step_end(run_context)

        cb_params.cur_epoch_num += 1
        my_train_loss = train_loss/steps_per_epoch
        my_train_accuracy = 100 * train_correct / (train_batch_size * steps_per_epoch)
        time_used = time.time() - t_start
        fps = train_batch_size*steps_per_epoch / time_used
        if rank == 0:
            print('epoch[{}], {:.2f} imgs/sec'.format(epoch, fps))
            print('Train Loss:', my_train_loss)
            print('Train Accuracy:', my_train_accuracy, '%')

def triangular():
    """
    triangular for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    return 1.0

def triangular2(cycle):
    """
    triangular2 for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    return 1.0 / (2.**(cycle - 1))

def learning_rate_clr_triangle_function(step_size, max_lr, base_lr, clr_iterations):
    """
    get learning rate for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    cycle = math.floor(1 + clr_iterations / (2 * step_size))
    x = abs(clr_iterations / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x)) * triangular()

def train():
    # init distributed
    if hparams.run_distribute:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        init()
        hparams.rank = get_rank()
        hparams.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=8,
                                          parameter_broadcast=True)
    else:
        hparams.rank = 0
        hparams.group_size = 1
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=hparams.device_id)
    data_dir = args.data_url
    in_channels = hparams.in_channels
    channels = hparams.channels
    base_lrate = hparams.base_lrate
    max_lrate = hparams.max_lrate
    weight_decay = hparams.weight_decay
    num_epochs = 1
    minibatch_size = hparams.minibatch_size
    emb_size = hparams.emb_size
    clc_step_size = hparams.step_size
    class_num = 7205
    ckpt_save_dir = args.train_url
    # Configure operation information

    mymodel = ECAPA_TDNN(in_channels, channels=(channels, channels, channels, channels, channels * 3),
                         lin_neurons=emb_size)
    # Construct model
    ds_train, steps_per_epoch_train = create_dataset(hparams, data_dir)
    print(f'group_size:{hparams.group_size}, data total len:{steps_per_epoch_train}')
    # Define the optimizer and model
    my_classifier = Classifier(1, 0, emb_size, class_num)
    aam = AdditiveAngularMargin(0.2, 30)
    lr_list = []
    lr_list_total = steps_per_epoch_train * num_epochs
    for i in range(lr_list_total):
        lr_list.append(learning_rate_clr_triangle_function(clc_step_size, max_lrate, base_lrate, i))

    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')

    loss_scale_manager = FixedLossScaleUpdateCell(loss_scale_value=2**14)
    model_constructed = BuildTrainNetwork(mymodel, my_classifier, aam, loss, minibatch_size, class_num)
    opt = nn.Adam(model_constructed.trainable_params(), learning_rate=lr_list, weight_decay=weight_decay)
    model_constructed = TrainOneStepWithLossScaleCell(model_constructed, opt,
                                                      scale_sense=loss_scale_manager)

    if hparams.pre_trained:
        pre_trained_model = os.path.join(ckpt_save_dir, hparams.checkpoint_path)
        param_dict = load_checkpoint(pre_trained_model)
        # load parameter to the network
        load_param_into_net(model_constructed, param_dict)
    # CheckPoint CallBack definition
    save_steps = int(steps_per_epoch_train/10)
    config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                 keep_checkpoint_max=hparams.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="train_ecapa_vox12",
                                 directory=ckpt_save_dir, config=config_ck)

    train_net(hparams.rank, model_constructed, num_epochs, ds_train, ckpoint_cb, steps_per_epoch_train, minibatch_size)
    print("============== End Training ==============")
    path = os.path.join(ckpt_save_dir, 'train_ecapa_vox12-0_936.ckpt')
    print("ckpt_save_dir  ", ckpt_save_dir, "path  ", path)
    save_ckpt_to_air(ckpt_save_dir, path)

if __name__ == "__main__":
    train()
