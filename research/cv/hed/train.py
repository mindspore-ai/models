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
'''TRAIN'''
import os
import datetime
import time
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from src.loss import BinaryCrossEntropyLoss
from src.lr_schedule import get_lr
from src.dataset import create_dataset
from src.model import HED
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

set_seed(666)

class Print_info(Callback):
    def __init__(self, lr_init=None):
        super(Print_info, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}".format(epoch_mseconds,
                                                                   per_step_mseconds))
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)

        print(datetime.datetime.now(), "end epoch", cb_params.cur_epoch_num,
              self.lr_init[cb_params.cur_step_num - 1], np.mean(self.losses))

def learning_rate_function(lr, epoch_num):
    if epoch_num % 250 == 0:
        lr = lr*0.1
        print("current lr: ", str(lr))
    return lr

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
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
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

def get_files(folder, name_filter=None, extension_filter=None):
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    if name_filter is None:
        def name_cond(filename):
            return True
    else:
        def name_cond(filename):
            return name_filter in filename

    if extension_filter is None:
        def ext_cond(filename):
            return True
    else:
        def ext_cond(filename):
            return filename.endswith(extension_filter)

    filtered_files = []
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)
    return filtered_files

def train_path_set():
    ''''new'''
    train_path = os.path.join(config.data_path, 'output/train.lst')
    train_img = get_files(os.path.join(config.data_path, 'BSDS500/data/images/train'),
                          extension_filter='.jpg')
    train_label = get_files(os.path.join(config.data_path, 'BSDS500/data/labels/train'),
                            extension_filter='.jpg')
    f = open(train_path, "w")
    for img, label in zip(train_img, train_label):
        f.write(str(img) + " " + str(label))
        f.write('\n')
    f.close()

def param_set():
    param_dict = load_checkpoint(config.vgg_ckpt_path)
    filter_list = ["classifier.0.bias", "classifier.3.bias", "classifier.6.bias",
                   "classifier.0.weight", "classifier.3.weight", "classifier.6.weight",
                   "global_step", "momentum", "moments.layers.0.weight", "moments.layers.2.weight",
                   "moments.layers.5.weight", "moments.layers.7.weight", "moments.layers.10.weight",
                   "moments.layers.12.weight", "moments.layers.14.weight", "moments.layers.17.weight",
                   "moments.layers.19.weight", "moments.layers.21.weight", "moments.layers.24.weight",
                   "moments.layers.26.weight", "moments.layers.28.weight", "learning_rate"]
    for key in list(param_dict.keys()):
        for filter_key in filter_list:
            if filter_key not in key:
                continue
            print('filter {}'.format(key))
            del param_dict[key]
    key_mapping = {'conv1_1.weight': 'layers.0.weight',
                   'conv1_2.weight': 'layers.2.weight',
                   'conv2_1.weight': 'layers.5.weight',
                   'conv2_2.weight': 'layers.7.weight',
                   'conv3_1.weight': 'layers.10.weight',
                   'conv3_2.weight': 'layers.12.weight',
                   'conv3_3.weight': 'layers.14.weight',
                   'conv4_1.weight': 'layers.17.weight',
                   'conv4_2.weight': 'layers.19.weight',
                   'conv4_3.weight': 'layers.21.weight',
                   'conv5_1.weight': 'layers.24.weight',
                   'conv5_2.weight': 'layers.26.weight',
                   'conv5_3.weight': 'layers.28.weight',
                  }
    for k, v in key_mapping.items():
        param_dict[k] = param_dict.pop(v)
    return param_dict
@moxing_wrapper(pre_process=modelarts_pre_process)
def train_hed_dis():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    if os.getenv("DEVICE_ID", "not_set").isdigit():
        context.set_context(device_id=get_device_id())
    init()
    device_num = get_device_num()
    rank = get_rank_id()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=device_num)
    net = HED()
    train_path = os.path.join(config.data_path, 'output/train.lst')
    train_loader = create_dataset(train_path, True, True, batch_size=config.batch_size,
                                  num_parallel_workers=config.para_workers, device_num=device_num, rank=rank)
    dataset_size = train_loader.get_dataset_size()
    #para_dict_new
    param_dict = param_set()
    load_param_into_net(net, param_dict)
    print('load_model {} success'.format(config.vgg_ckpt_path))
    #tune lr
    net_parameters_id = {}
    for pname, p in net.parameters_and_names():
        if pname in ['conv1_1.weight', 'conv1_2.weight', 'conv2_1.weight', 'conv2_2.weight', 'conv3_1.weight',
                     'conv3_2.weight', 'conv3_3.weight', 'conv4_1.weight', 'conv4_2.weight', 'conv4_3.weight']:
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias', 'conv1_2.bias', 'conv2_1.bias', 'conv2_2.bias', 'conv3_1.bias',
                       'conv3_2.bias', 'conv3_3.bias', 'conv4_1.bias', 'conv4_2.bias', 'conv4_3.bias']:
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight', 'conv5_2.weight', 'conv5_3.weight']:
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias', 'conv5_2.bias', 'conv5_3.bias']:
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['score_dsn1.weight', 'score_dsn2.weight', 'score_dsn3.weight',
                       'score_dsn4.weight', 'score_dsn5.weight']:
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias', 'score_dsn2.bias', 'score_dsn3.bias', 'score_dsn4.bias', 'score_dsn5.bias']:
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)
    lr = Tensor(get_lr(global_step=0, lr_init=config.lr, total_epochs=config.epoch_size, steps_per_epoch=dataset_size))
    group_params = [
        {'params': net_parameters_id['conv1-4.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['conv1-4.bias'], 'lr': lr, 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['conv5.bias'], 'lr': lr, 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': lr, 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['score_final.bias'], 'lr': lr, 'weight_decay': 0.},
        ]
    optimizer = nn.SGD(group_params, learning_rate=lr, momentum=config.momentum, weight_decay=config.weight_decay)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2**24, scale_factor=2, scale_window=2000)
    b_loss = BinaryCrossEntropyLoss(net)
    model = Model(b_loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager)

    #training callbacks
    checkpoint_config = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
    print_cb = Print_info(lr_init=lr.asnumpy())
    loss_monitor_cb = LossMonitor(per_print_times=dataset_size)
    ckpt_save_dir = config.output_path +'/ckpt_{}/'.format(rank)
    ckpoint_cb = ModelCheckpoint('hed_mindspore', directory=ckpt_save_dir, config=checkpoint_config)
    cb = [print_cb, loss_monitor_cb]

    if rank == 0:
        cb += [ckpoint_cb]
    model.train(config.epoch_size, train_loader, callbacks=cb, dataset_sink_mode=False)

@moxing_wrapper(pre_process=modelarts_pre_process)
def train_hed_sin():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    rank = 0
    device_num = 1
    context.set_context(device_id=get_device_id())
    net = HED()
    train_path = os.path.join(config.data_path, 'output/train.lst')
    train_loader = create_dataset(train_path, True, True, batch_size=config.batch_size,
                                  num_parallel_workers=config.para_workers, device_num=device_num, rank=rank)
    dataset_size = train_loader.get_dataset_size()
    #para_dict_new
    param_dict = param_set()
    load_param_into_net(net, param_dict)
    print('load_model {} success'.format(config.vgg_ckpt_path))
    #tune lr
    net_parameters_id = {}
    for pname, p in net.parameters_and_names():
        if pname in ['conv1_1.weight', 'conv1_2.weight', 'conv2_1.weight', 'conv2_2.weight', 'conv3_1.weight',
                     'conv3_2.weight', 'conv3_3.weight', 'conv4_1.weight', 'conv4_2.weight', 'conv4_3.weight']:
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias', 'conv1_2.bias', 'conv2_1.bias', 'conv2_2.bias', 'conv3_1.bias',
                       'conv3_2.bias', 'conv3_3.bias', 'conv4_1.bias', 'conv4_2.bias', 'conv4_3.bias']:
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight', 'conv5_2.weight', 'conv5_3.weight']:
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias', 'conv5_2.bias', 'conv5_3.bias']:
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['score_dsn1.weight', 'score_dsn2.weight', 'score_dsn3.weight',
                       'score_dsn4.weight', 'score_dsn5.weight']:
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias', 'score_dsn2.bias', 'score_dsn3.bias', 'score_dsn4.bias', 'score_dsn5.bias']:
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)
    lr = Tensor(get_lr(global_step=0, lr_init=config.lr, total_epochs=config.epoch_size, steps_per_epoch=dataset_size))
    group_params = [
        {'params': net_parameters_id['conv1-4.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['conv1-4.bias'], 'lr': lr, 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['conv5.bias'], 'lr': lr, 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': lr, 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight'], 'lr': lr, 'weight_decay': 2e-4},
        {'params': net_parameters_id['score_final.bias'], 'lr': lr, 'weight_decay': 0.},
        ]
    optimizer = nn.SGD(group_params, learning_rate=lr, momentum=config.momentum, weight_decay=config.weight_decay)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2**24, scale_factor=2, scale_window=2000)
    b_loss = BinaryCrossEntropyLoss(net)
    model = Model(b_loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager)

    #training callbacks
    checkpoint_config = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
    print_cb = Print_info(lr_init=lr.asnumpy())
    loss_monitor_cb = LossMonitor(per_print_times=dataset_size)
    ckpt_save_dir = config.output_path +'/ckpt_{}/'.format(rank)
    ckpoint_cb = ModelCheckpoint('hed_mindspore', directory=ckpt_save_dir, config=checkpoint_config)
    cb = [print_cb, loss_monitor_cb]

    cb += [ckpoint_cb]
    model.train(config.epoch_size, train_loader, callbacks=cb, dataset_sink_mode=False)
if __name__ == "__main__":
    train_path_set()
    if config.distribute:
        train_hed_dis()
    else:
        train_hed_sin()
