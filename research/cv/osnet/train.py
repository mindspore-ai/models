# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""train OSNet and get checkpoint files."""

import warnings
import os
import os.path as osp
import time
import numpy as np


import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore import Tensor, Model, context
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor,\
                                      TimeMonitor, SummaryCollector
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank


from src.osnet import create_osnet
from src.dataset import dataset_creator
from src.lr_generator import step_lr, multi_step_lr
from src.cross_entropy_loss import CrossEntropyLoss
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


set_seed(1)
class LossCallBack(LossMonitor):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()

    def step_end(self, run_context):
        '''check loss at the end of each step.'''
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss), flush=True)

    def end(self, run_context):
        cb_params = run_context.original_args()
        end_time = time.time()
        print("total_time:", (end_time-cb_params.init_time)*1000, "ms")


def init_lr(num_batches):
    '''initialize learning rate.'''
    if config.lr_scheduler == 'single_step':
        lr = step_lr(config.lr, config.step_size, num_batches, config.max_epoch, config.gamma)
    elif config.lr_scheduler == 'multi_step':
        lr = multi_step_lr(config.lr, config.step_size, num_batches, config.max_epoch, config.gamma)
    elif config.lr_scheduler == 'cosine':
        lr = np.array(nn.cosine_decay_lr(0., config.lr, num_batches * config.max_epoch, num_batches,
                                         config.max_epoch)).astype(np.float32)
    return lr


def check_isfile(fpath):
    '''check whether the path is a file.'''
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def load_from_checkpoint(net):
    '''load parameters when resuming from a checkpoints for training.'''
    param_dict = load_checkpoint(config.checkpoint_file_path)
    if param_dict:
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            config.start_epoch = int(param_dict["epoch_num"].data.asnumpy())
            config.start_step = int(param_dict["step_num"].data.asnumpy())
        else:
            config.start_epoch = 0
            config.start_step = 0
        load_param_into_net(net, param_dict)
    else:
        raise ValueError("Checkpoint file:{} is none.".format(config.checkpoint_file_path))


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path, config.source)
    if config.enable_modelarts and config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    else:
        if config.run_distribute:
            ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.source)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.source + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_rank() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_rank(), zip_file_1, save_dir_1))


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_net():
    """train net"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    device_num = get_device_num()
    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel',
                                              gradients_mean=True)
            init()
            rank_id = get_rank()
        else:
            rank_id = 0
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel',
                                              gradients_mean=True)
            init()
            rank_id = get_rank()
        else:
            rank_id = 0


    num_classes, dataset1 = dataset_creator(root=config.data_path, height=config.height, width=config.width,
                                            transforms=config.transforms, dataset=config.source,
                                            norm_mean=config.norm_mean, norm_std=config.norm_std,
                                            batch_size_train=config.batch_size_train, workers=config.workers,
                                            cuhk03_labeled=config.cuhk03_labeled,
                                            cuhk03_classic_split=config.cuhk03_classic_split, mode='train')
    num_classes, dataset2 = dataset_creator(root=config.data_path, height=config.height, width=config.width,
                                            transforms=config.transforms, dataset=config.source,
                                            norm_mean=config.norm_mean, norm_std=config.norm_std,
                                            batch_size_train=config.batch_size_train, workers=config.workers,
                                            cuhk03_labeled=config.cuhk03_labeled,
                                            cuhk03_classic_split=config.cuhk03_classic_split, mode='train')
    num_batches = dataset1.get_dataset_size()


    if config.checkpoint_file_path and check_isfile(config.checkpoint_file_path):
        fpath = osp.abspath(osp.expanduser(config.checkpoint_file_path))
        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        net = create_osnet(num_classes=num_classes)
        load_from_checkpoint(net)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pretrained_dir = os.path.join(current_dir, config.pretrained_dir)
        net = create_osnet(num_classes=num_classes, pretrained=True, pretrained_dir=pretrained_dir)

    crit = CrossEntropyLoss(num_classes=num_classes, label_smooth=config.label_smooth)
    lr = init_lr(num_batches=num_batches)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if config.LogSummary:
        summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=1)
    time_cb = TimeMonitor(data_size=num_batches)


    if config.start_epoch == config.fixbase_epoch:
        net.stop_layer = ops.stop_gradient
        lr1 = Tensor(lr[:config.start_epoch * num_batches])
        opt1 = nn.Adam(net.classifier.trainable_params(), learning_rate=lr1, beta1=config.adam_beta1,
                       beta2=config.adam_beta2, weight_decay=config.weight_decay, loss_scale=config.loss_scale)
        model1 = Model(network=net, optimizer=opt1, loss_fn=crit, amp_level="O3",
                       loss_scale_manager=loss_scale)
        loss_cb1 = LossCallBack(has_trained_epoch=0)
        if config.LogSummary:
            cb1 = [time_cb, loss_cb1, summary_collector]
        else:
            cb1 = [time_cb, loss_cb1]
        model1.train(config.fixbase_epoch, dataset1, cb1, dataset_sink_mode=True)

    loss_cb2 = LossCallBack(config.start_epoch)
    net.stop_layer = ops.Identity()
    lr2 = Tensor(lr[config.start_epoch * num_batches:])
    opt2 = nn.Adam(net.trainable_params(), learning_rate=lr2, beta1=config.adam_beta1, beta2=config.adam_beta2,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    model2 = Model(network=net, optimizer=opt2, loss_fn=crit, amp_level="O3",
                   loss_scale_manager=loss_scale)
    if config.LogSummary:
        cb2 = [time_cb, loss_cb2, summary_collector]
    else:
        cb2 = [time_cb, loss_cb2]

    if config.save_checkpoint:
        if not config.run_distribute or (config.run_distribute and rank_id % 8 == 0):
            ckpt_append_info = [{"epoch_num": config.start_epoch, "step_num": config.start_epoch}]
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * num_batches,
                                         keep_checkpoint_max=10, append_info=ckpt_append_info)
            ckpt_cb = ModelCheckpoint(prefix="osnet", directory=set_save_ckpt_dir(), config=config_ck)
            cb2 += [ckpt_cb]
    model2.train(config.max_epoch-config.start_epoch, dataset2, cb2, dataset_sink_mode=True)
    print("train success")


if __name__ == '__main__':
    train_net()
