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
"""YoloV5 train."""
import os
import time
from collections import deque
import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm
from mindspore import load_checkpoint, Parameter, save_checkpoint

from src.yolo import YOLOV5, YoloWithLossCell
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups, cpu_affinity, EvalWrapper, DetectionEngine
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov5_params

from model_utils.config import config
from model_utils.device_adapter import get_device_id

# only useful for huawei cloud modelarts.
from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process, modelarts_post_process


ms.set_seed(1)


def init_distribute():
    comm.init()
    config.rank = comm.get_rank()
    config.group_size = comm.get_group_size()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                 device_num=config.group_size)


def train_preprocess():
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.T_max:
        config.T_max = config.max_epoch

    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.train_img_dir = os.path.join(config.data_dir, config.train_img_dir)
    config.train_ann_file = os.path.join(config.data_dir, config.train_ann_file)
    device_id = get_device_id()
    if config.device_target == "Ascend":
        device_id = get_device_id()
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)

    if config.is_distributed:
        # init distributed
        init_distribute()

    # for promoting performance in GPU device
    if config.device_target == "GPU" and config.bind_cpu:
        cpu_affinity(config.rank, min(config.group_size, config.device_num))

    # logger module is managed by config, it is used in other function. e.x. config.logger.info("xxx")
    config.logger = get_logger(config.output_dir, config.rank)
    config.logger.save_args(config)


def get_val_dataset():
    config.val_img_dir = os.path.join(config.data_dir, config.val_img_dir)
    config.val_ann_file = os.path.join(config.data_dir, config.val_ann_file)
    ds_val = create_yolo_dataset(config.val_img_dir, config.val_ann_file,
                                 is_training=False,
                                 batch_size=config.per_batch_size,
                                 device_num=config.group_size,
                                 rank=config.rank, config=config)
    config.logger.info("Finish loading val dataset!")
    return ds_val


def load_parameters(val_network, train_network):
    config.logger.info("Load parameters of train network")
    param_dict_new = {}
    for key, values in train_network.parameters_and_names():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(val_network, param_dict_new)
    config.logger.info('Load train network success')


def load_best_results():
    best_ckpt_path = os.path.join(config.output_dir, 'best.ckpt')
    if os.path.exists(best_ckpt_path):
        param_dict = load_checkpoint(best_ckpt_path)
        best_result = param_dict['best_result'].asnumpy().item()
        best_epoch = param_dict['best_epoch'].asnumpy().item()
        config.logger.info('cur best result %s at epoch %s', best_result, best_epoch)
        return best_result, best_epoch
    return 0.0, 0


def save_best_checkpoint(network, best_result, best_epoch):
    param_list = [{'name': 'best_result', 'data': Parameter(best_result)},
                  {'name': 'best_epoch', 'data': Parameter(best_epoch)}]
    for name, param in network.parameters_and_names():
        param_list.append({'name': name, 'data': param})
    save_checkpoint(param_list, os.path.join(config.output_dir, 'best.ckpt'))


def is_val_epoch(epoch_idx: int):
    epoch = epoch_idx + 1
    return (
        (epoch >= config.eval_start_epoch) and
        ((epoch_idx + 1) % config.eval_epoch_interval == 0 or (epoch_idx + 1) == config.max_epoch)
    )

@moxing_wrapper(pre_process=modelarts_pre_process, post_process=modelarts_post_process, pre_args=[config])
def run_train():
    train_preprocess()
    config.eval_parallel = config.run_eval and config.is_distributed and config.eval_parallel
    loss_meter = AverageMeter('loss')
    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    network = YOLOV5(is_training=True, version=dict_version[config.yolov5_version])
    val_network = YOLOV5(is_training=False, version=dict_version[config.yolov5_version])
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov5_params(config, network)
    network = YoloWithLossCell(network)

    ds = create_yolo_dataset(image_dir=config.train_img_dir, anno_path=config.train_ann_file, is_training=True,
                             batch_size=config.per_batch_size, device_num=config.group_size,
                             rank=config.rank, config=config)
    config.logger.info('Finish loading train dataset')
    ds_val = get_val_dataset()

    steps_per_epoch = ds.get_dataset_size()
    lr = get_lr(config, steps_per_epoch)
    opt = nn.Momentum(params=get_param_groups(network), momentum=config.momentum, learning_rate=ms.Tensor(lr),
                      weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    network = nn.TrainOneStepCell(network, opt, config.loss_scale // 2)
    network.set_train()

    data_loader = ds.create_tuple_iterator(do_copy=False)
    first_step = True
    t_end = time.time()
    # Load previous best results if exists
    best_result, best_epoch = load_best_results()
    engine = DetectionEngine(config, config.test_ignore_threshold)
    eval_wrapper = EvalWrapper(config, val_network, ds_val, engine)
    ckpt_queue = deque()
    for epoch_idx in range(config.max_epoch):
        for step_idx, data in enumerate(data_loader):
            images = data[0]
            input_shape = images.shape[1:3]
            input_shape = ms.Tensor(input_shape, ms.float32)
            loss = network(images, data[2], data[3], data[4], data[5], data[6],
                           data[7], input_shape)
            loss_meter.update(loss.asnumpy())

            # it is used for loss, performance output per config.log_interval steps.
            if (epoch_idx * steps_per_epoch + step_idx) % config.log_interval == 0:
                time_used = time.time() - t_end
                if first_step:
                    fps = config.per_batch_size * config.group_size / time_used
                    per_step_time = time_used * 1000
                    first_step = False
                else:
                    fps = config.per_batch_size * config.log_interval * config.group_size / time_used
                    per_step_time = time_used / config.log_interval * 1000
                config.logger.info('epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, '
                                   'lr:{}, per step time: {}ms'.format(epoch_idx + 1, step_idx + 1,
                                                                       loss_meter, fps, lr[step_idx], per_step_time))
                t_end = time.time()
                loss_meter.reset()
        if config.rank == 0 and (epoch_idx % config.save_ckpt_interval == 0):
            ckpt_name = os.path.join(config.output_dir, "yolov5_{}_{}.ckpt".format(epoch_idx + 1, steps_per_epoch))
            ms.save_checkpoint(network, ckpt_name)
            if len(ckpt_queue) == config.save_ckpt_max_num:
                ckpt_to_remove = ckpt_queue.popleft()
                os.remove(ckpt_to_remove)
            ckpt_queue.append(ckpt_name)

        if is_val_epoch(epoch_idx):
            # Load weights of train network to val network
            load_parameters(val_network, train_network=network)
            eval_wrapper.inference()
            eval_result, mAP = eval_wrapper.get_results(cur_epoch=epoch_idx + 1, cur_step=steps_per_epoch)
            if mAP >= best_result:
                best_result = mAP
                best_epoch = epoch_idx + 1
                if config.rank == 0:
                    save_best_checkpoint(network, best_result, best_epoch)
                config.logger.info("Best result %s at %s epoch", best_result, best_epoch)
            config.logger.info(eval_result)
            config.logger.info('Ending inference...')


    config.logger.info('==========end training===============')


if __name__ == "__main__":
    run_train()
