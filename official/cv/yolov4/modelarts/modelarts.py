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
"""YoloV4 train."""
import argparse
import os
import time
import datetime
import numpy as np

import mindspore
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import CheckpointConfig
from mindspore.common import set_seed
from mindspore.profiler.profiling import Profiler

from src.yolo import YOLOV4CspDarkNet53, YoloWithLossCell, TrainingWrapper
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov4_params
from src.eval_utils import apply_eval, EvalCallBack

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

set_seed(1)
parser = argparse.ArgumentParser(description='YOLOV4')
parser.add_argument('--enable_modelarts', type=bool, default='True', help='use modelarts')
parser.add_argument('--data_url', type=str, default='', help='Dataset directory')
parser.add_argument('--train_url', type=str, default='', help='The path model saved')
parser.add_argument('--checkpoint_url', type=str, default='', help='The path pre-model saved')
parser.add_argument('--is_distributed', type=int, default=0, help='do not distributed')
parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epoch')
parser.add_argument('--epoch', type=int, default=1, help='train epoch')
parser.add_argument('--training_shape', type=int, default=416, help='training shape')
args_opt, _ = parser.parse_known_args()

def set_default():
    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(config.data_path, exist_ok=True)

    config.run_eval = True
    config.eval_start_epoch = 0
    config.max_epoch = args_opt.epoch
    config.warmup_epochs = args_opt.warmup_epochs
    config.is_distributed = args_opt.is_distributed
    config.enable_modelarts = args_opt.enable_modelarts
    config.checkpoint_url = args_opt.checkpoint_url
    config.pretrained_backbone = args_opt.checkpoint_url
    config.training_shape = args_opt.training_shape
    config.per_batch_size = 1
    config.file_name = os.path.join(args_opt.train_url, "yolov4")
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.t_max:
        config.t_max = config.max_epoch

    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.data_root = os.path.join(args_opt.data_url, 'train2017')
    config.annFile = os.path.join(args_opt.data_url, 'annotations/instances_train2017.json')

    config.data_val_root = os.path.join(args_opt.data_url, 'val2017')
    config.ann_val_file = os.path.join(args_opt.data_url, 'annotations/instances_val2017.json')

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False, device_id=device_id)

    if config.need_profiler:
        profiler = Profiler(output_path=config.checkpoint_url, is_detail=True, is_show_op_path=True)
    else:
        profiler = None

    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(args_opt.train_url,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    return profiler


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network_, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network_
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss_ = self.criterion(output, label)
        return loss_


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

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


def get_network(net, cfg, learning_rate):
    opt = Momentum(params=get_param_groups(net),
                   learning_rate=Tensor(learning_rate),
                   momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay,
                   loss_scale=cfg.loss_scale)
    net = TrainingWrapper(net, opt)
    net.set_train()
    return net


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():

    profiler = set_default()
    loss_meter = AverageMeter('loss')
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)

    network = YOLOV4CspDarkNet53()
    if config.run_eval:
        network_eval = network
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov4_params(config, network)

    network = YoloWithLossCell(network)
    config.logger.info('finish get network')

    ds, data_size = create_yolo_dataset(image_dir=config.data_root, anno_path=config.annFile, is_training=True,
                                        batch_size=config.per_batch_size, max_epoch=config.max_epoch,
                                        device_num=config.group_size, rank=config.rank, default_config=config)
    config.logger.info('Finish loading dataset')

    config.steps_per_epoch = int(data_size / config.per_batch_size / config.group_size)

    if config.ckpt_interval <= 0: config.ckpt_interval = config.steps_per_epoch

    lr = get_lr(config)
    network = get_network(network, config, lr)
    network.set_train(True)

    if config.rank_save_ckpt_flag or config.run_eval:
        cb_params = InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)

    if config.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = 10
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=save_ckpt_path, prefix='{}'.format(config.rank))
        ckpt_cb.begin(run_context)

    if config.run_eval:
        data_val_root = config.data_val_root
        ann_val_file = config.ann_val_file
        save_ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank) + '/')
        input_val_shape = Tensor(tuple(config.test_img_shape), mindspore.float32)
        # init detection engine
        eval_dataset, eval_data_size = create_yolo_dataset(data_val_root, ann_val_file, is_training=False,
                                                           batch_size=1, max_epoch=1, device_num=1,
                                                           rank=0, shuffle=False, default_config=config)
        eval_param_dict = {"net": network_eval, "dataset": eval_dataset, "data_size": eval_data_size,
                           "anno_json": ann_val_file, "input_shape": input_val_shape, "args": config}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=save_ckpt_path, besk_ckpt_name="best_map.ckpt", metrics_name="mAP")

    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)

    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        config.logger.info('iter[%d], shape%d', i + 1, input_shape[0])

        images = Tensor.from_numpy(images)
        batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
        batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
        batch_y_true_2 = Tensor.from_numpy(data['bbox3'])
        batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
        batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])
        batch_gt_box2 = Tensor.from_numpy(data['gt_box3'])

        input_shape = Tensor(tuple(input_shape[::-1]), mindspore.float32)
        loss = network(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                       batch_gt_box2, input_shape)
        loss_meter.update(loss.asnumpy())

        # ckpt progress
        if config.rank_save_ckpt_flag:
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if (i + 1) % config.log_interval == 0:
            time_used = time.time() - t_end
            epoch = int((i + 1) / config.steps_per_epoch)
            fps = config.per_batch_size * (i - old_progress) * config.group_size / time_used
            if config.rank == 0:
                config.logger.info('epoch[{}], iter[{}], {}, per step time: {:.2f} ms, fps: {:.2f}, lr:{}'.format(
                    epoch, i, loss_meter, 1000 * time_used / (i - old_progress), fps, lr[i]))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if (i + 1) % config.steps_per_epoch == 0 and (config.run_eval or config.rank_save_ckpt_flag):
            if config.run_eval:
                eval_cb.epoch_end(run_context)
                network.set_train()
            cb_params.cur_epoch_num += 1

        if config.need_profiler and profiler is not None:
            if i == 10:
                profiler.analyse()
                break

    ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank), '0-1_117266.ckpt')

    network_export = YOLOV4CspDarkNet53()
    network_export.set_train(False)

    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network_export, param_dict)
    input_data = Tensor(np.zeros([config.batch_size, 3, config.testing_shape, config.testing_shape]), mindspore.float32)

    export(network_export, input_data, file_name=config.file_name, file_format="AIR")

if __name__ == "__main__":
    run_train()
