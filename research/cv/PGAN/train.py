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
#################train pgan########################
"""
import datetime
import os
import pathlib

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as ds
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.context import ParallelMode

from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.dataset import ImageDataset
from src.image_transform import Normalize, NumpyResize, TransporeAndDiv, Crop
from src.network_D import DNet4_4_Train, DNetNext_Train, DNet4_4_Last, DNetNext_Last
from src.network_G import GNet4_4_Train, GNet4_4_last, GNetNext_Train, GNetNext_Last
from src.optimizer import AllLossD, AllLossG
from src.time_monitor import TimeMonitor


def set_every(num):
    """set random seed"""
    set_seed(num)
    np.random.seed(num)


set_every(1)
ds.config.set_prefetch_size(16)


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


def cell_deepcopy(gnet, avg_gnet):
    """cell_deepcopy"""
    for param, avg_param in zip(gnet.trainable_params(),
                                avg_gnet.trainable_params()):
        avg_param.set_data(param.clone())


def cell_deepcopy_update(gnet, avg_gnet):
    """cell_deepcopy_update"""
    for param, avg_param in zip(gnet.trainable_params(),
                                avg_gnet.trainable_params()):
        new_p = avg_param * 0.999 + param * 0.001
        avg_param.set_data(new_p)


def save_checkpoints(avg, gnet, dnet, ckpt_dir, i_batch):
    """save_checkpoint"""
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_checkpoint(gnet, os.path.join(ckpt_dir, f"G_{i_batch}.ckpt"))
    save_checkpoint(avg, os.path.join(ckpt_dir, f"AvG_{i_batch}.ckpt"))
    save_checkpoint(dnet, os.path.join(ckpt_dir, f"D_{i_batch}.ckpt"))


def load_checkpoints(gnet, dnet, cfg):
    """load_checkpoints"""
    param_dict_g = load_checkpoint(cfg.resume_check_g)
    param_dict_d = load_checkpoint(cfg.resume_check_d)
    load_param_into_net(gnet, param_dict_g)
    load_param_into_net(dnet, param_dict_d)
    return gnet, dnet


def modelarts_pre_process():
    """modelarts pre process function."""
    config.ckpt_save_dir = os.path.join(config.output_path, config.ckpt_save_dir)


def get_dataset(args, size=None):
    """getDataset

    Returns:
        output.
    """
    transform_list = [Crop(), NumpyResize(size), TransporeAndDiv(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return ImageDataset(args.train_data_path, transform=transform_list)


def cal_each_batch_alpha():
    """buildNoiseData"""
    each_batch_alpha = []
    for index in range(len(config.scales)):
        this_batch = config.num_batch[index]
        new_batch_alpha = []
        alphas = -1
        new_jumps = config.alpha_jumps[index] / config.device_num
        for i in range(this_batch):
            if i % config.alpha_size_jumps[index] == 0:
                alphas += 1
            if i < new_jumps * config.alpha_size_jumps[index]:
                new_batch_alpha.append(1 - alphas / new_jumps)
            else:
                new_batch_alpha.append(0.0)
        each_batch_alpha.append(new_batch_alpha)
    return each_batch_alpha


def get_optimize_d(dnet, lr, cfg):
    """getOptimizerD

    Returns:
        output.
    """
    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** cfg.loss_scale_value,
                                            scale_factor=cfg.scale_factor, scale_window=cfg.scale_factor)
    loss_cell = AllLossD(dnet)
    opti = nn.Adam(dnet.trainable_params(), beta1=0.0001, beta2=0.99, learning_rate=lr)
    train_network = nn.TrainOneStepWithLossScaleCell(loss_cell, opti, scale_sense=manager)
    return train_network


def get_optimizer_g(gnet, dnet, args):
    """getOptimizerG

    Returns:
        output.
    """
    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** args.loss_scale_value,
                                            scale_factor=args.scale_factor, scale_window=args.scale_factor)
    loss_cell = AllLossG(gnet, dnet)
    opti = nn.Adam(gnet.trainable_params(),
                   beta1=0.0001, beta2=0.99, learning_rate=args.lr)
    train_network = nn.TrainOneStepWithLossScaleCell(loss_cell, opti, scale_sense=manager)
    return train_network


def build_noise_data(n_samples):
    """buildNoiseData

    Returns:
        output.
    """
    input_latent = np.random.randn(n_samples, 512)
    input_latent = Tensor(input_latent, mstype.float32)
    return input_latent


def prepare_context(cfg):
    """prepare context"""
    if cfg.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if cfg.device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=cfg.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    if cfg.device_target == "GPU":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if cfg.device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=cfg.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)


def construct_network(gnet, dnet, avg_gnet, depth, scale):
    """construct_network"""
    if scale == 4:
        dnet = DNet4_4_Train(depth, leakyReluLeak=0.2, sizeDecisionLayer=1, dimInput=3)
        gnet = GNet4_4_Train(512, depth, leakyReluLeak=0.2, dimOutput=3)
        avg_gnet = GNet4_4_Train(512, depth, leakyReluLeak=0.2, dimOutput=3)
    elif scale == 8:
        last_dnet = DNet4_4_Last(dnet)
        last_gnet = GNet4_4_last(gnet)
        dnet = DNetNext_Train(depth, last_Dnet=last_dnet, leakyReluLeak=0.2, dimInput=3)
        gnet = GNetNext_Train(depth, last_Gnet=last_gnet, leakyReluLeak=0.2, dimOutput=3)
        last_avg_gnet = GNet4_4_last(avg_gnet)
        avg_gnet = GNetNext_Train(depth, last_Gnet=last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
    else:
        last_dnet = DNetNext_Last(dnet)
        last_gnet = GNetNext_Last(gnet)
        dnet = DNetNext_Train(depth, last_Dnet=last_dnet, leakyReluLeak=0.2, dimInput=3)
        gnet = GNetNext_Train(depth, last_gnet, leakyReluLeak=0.2, dimOutput=3)
        last_avg_gnet = GNetNext_Last(avg_gnet)
        avg_gnet = GNetNext_Train(depth, last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
    cell_deepcopy(gnet, avg_gnet)
    return gnet, dnet, avg_gnet


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """run_train"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    config.device_num = get_device_num()
    prepare_context(config)

    if config.lr_list:
        if len(config.lr_list) != len(config.scales):
            raise ValueError(f"len(lr_list) and len(config.scales) must be same")
    else:
        config.lr_list = [config.lr] * len(config.scales)

    if config.batch_size_list:
        if len(config.batch_size_list) != len(config.scales):
            raise ValueError(f"len(lr_list) and len(config.scales) must be same")
    else:
        config.batch_size_list = [config.batch_size] * len(config.scales)

    gnet, dnet, avg_gnet = None, None, None
    each_batch_alpha = cal_each_batch_alpha()
    time_monitor = TimeMonitor(200)
    for scale_index, scale in enumerate(config.scales):
        print('Scale', scale, flush=True)
        this_scale_checkpoint_path = os.path.join(config.ckpt_save_dir, str(scale))

        gnet, dnet, avg_gnet = construct_network(gnet, dnet, avg_gnet, config.depth[scale_index], scale)
        if config.resume_load_scale != -1 and scale <= config.resume_load_scale:
            if scale == config.resume_load_scale:
                gnet, dnet = load_checkpoints(gnet, dnet, config)
            continue

        optimizer_d = get_optimize_d(dnet, config.lr_list[scale_index], config)
        optimizer_g = get_optimizer_g(gnet, dnet, config)
        rank_size, rank_id = _get_rank_info()
        if rank_id:
            this_scale_checkpoint_path = os.path.join(this_scale_checkpoint_path, f"rank_{rank_id}")

        db_loader = get_dataset(config, (scale, scale))
        dataset = ds.GeneratorDataset(db_loader, column_names=["data", "label"], shuffle=True,
                                      num_parallel_workers=4, num_shards=rank_size, shard_id=rank_id)
        dataset = dataset.batch(batch_size=config.batch_size_list[scale_index], drop_remainder=True)
        dataset_iter = dataset.create_tuple_iterator()
        print('Dataset size', dataset.get_dataset_size(), flush=True)
        i_batch = 0
        while i_batch < config.num_batch[scale_index] / config.device_num:
            time_monitor.epoch_begin()
            for data in dataset_iter:
                time_monitor.step_start()
                alpha = each_batch_alpha[scale_index][i_batch]
                alpha = Tensor(alpha, mstype.float32)
                inputs_real = data[0]
                n_samples = inputs_real.shape[0]
                fake_image = gnet(build_noise_data(n_samples), alpha)
                loss_d, overflow, _ = optimizer_d(inputs_real, fake_image.copy(), alpha)
                loss_g, overflow, _ = optimizer_g(build_noise_data(n_samples), alpha)
                cell_deepcopy_update(gnet=gnet, avg_gnet=avg_gnet)
                i_batch += 1
                time_monitor.step_end()
                if i_batch >= config.num_batch[scale_index] / config.device_num:
                    break
                if i_batch % 100 == 0:
                    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print(f'batch_i:{i_batch} alpha:{alpha} loss G:{loss_g} '
                          f'loss D:{loss_d} overflow:{overflow},time:{time_now}')
                if (i_batch + 1) % config.model_save_step == 0:
                    save_checkpoints(avg_gnet, gnet, dnet, this_scale_checkpoint_path, i_batch)
                time_monitor.data_iter_end()
            time_monitor.epoch_end()
        save_checkpoints(avg_gnet, gnet, dnet, this_scale_checkpoint_path, i_batch)


if __name__ == '__main__':
    run_train()
