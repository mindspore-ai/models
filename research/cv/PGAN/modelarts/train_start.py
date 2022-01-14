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
import os
import datetime
import numpy as np
from mindspore import nn
from mindspore.common import set_seed
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank
import mindspore
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.train.serialization import export
from src.image_transform import Normalize, NumpyResize, TransporeAndDiv
from src.dataset import ImageDataset
from src.network_D import DNet4_4_Train, DNetNext_Train, DNet4_4_Last, DNetNext_Last
from src.network_G import GNet4_4_Train, GNet4_4_last, GNetNext_Train, GNetNext_Last
from src.optimizer import AllLossD, AllLossG

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

import moxing as mox

set_seed(1)
np.random.seed(1)

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
    for p, avg_p in zip(gnet.trainable_params(),
                        avg_gnet.trainable_params()):
        avg_p.set_data(p.clone())


def cell_deepcopy_update(gnet, avg_gnet):
    """cell_deepcopy_update"""
    for p, avg_p in zip(gnet.trainable_params(),
                        avg_gnet.trainable_params()):
        new_p = avg_p * 0.999 + p * 0.001
        avg_p.set_data(new_p)

def save_checkpoint_g(avg, gnet, dnet, ckpt_dir, i_batch):
    """save_checkpoint"""
    save_checkpoint(gnet, os.path.join(ckpt_dir, "G_{}.ckpt".format(i_batch)))
    save_checkpoint(avg, os.path.join(ckpt_dir, "AvG_{}.ckpt".format(i_batch)))
    save_checkpoint(dnet, os.path.join(ckpt_dir, "D_{}.ckpt".format(i_batch)))
def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.ckpt_save_dir = os.path.join(config.output_path, config.ckpt_save_dir)
def getDataset(args, size=None):
    """getDataset

    Returns:
        output.
    """
    transformList = [NumpyResize(size), TransporeAndDiv(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return ImageDataset(args.train_data_path, transform=transformList)
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

def getOptimizerD(dnet, args):
    """getOptimizerD

    Returns:
        output.
    """
    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** args.loss_scale_value,
                                            scale_factor=args.scale_factor, scale_window=args.scale_factor)
    lossCell = AllLossD(dnet)
    opti = nn.Adam(dnet.trainable_params(), beta1=0.0001, beta2=0.99, learning_rate=args.lr)
    train_network = nn.TrainOneStepWithLossScaleCell(lossCell, opti, scale_sense=manager)
    return train_network


def getOptimizerG(gnet, dnet, args):
    """getOptimizerG

    Returns:
        output.
    """
    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** args.loss_scale_value,
                                            scale_factor=args.scale_factor, scale_window=args.scale_factor)
    lossCell = AllLossG(gnet, dnet)
    opti = nn.Adam(gnet.trainable_params(),
                   beta1=0.0001, beta2=0.99, learning_rate=args.lr)
    train_network = nn.TrainOneStepWithLossScaleCell(lossCell, opti, scale_sense=manager)
    return train_network


def buildNoiseData(n_samples):
    """buildNoiseData

    Returns:
        output.
    """
    inputLatent = np.random.randn(n_samples, 512)
    inputLatent = mindspore.Tensor(inputLatent, mindspore.float32)
    return inputLatent

def exportOutput(cfg):
    """exportOutput

    Returns:
        output.
    """
    scales = cfg.scales
    depth = cfg.depth
    for scale_index, scale in enumerate(scales):
        if scale == 4:
            avg_gnet = GNet4_4_Train(512, depth[scale_index], leakyReluLeak=0.2, dimOutput=3)
        elif scale == 8:
            last_avg_gnet = GNet4_4_last(avg_gnet)
            avg_gnet = GNetNext_Train(depth[scale_index], last_Gnet=last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
        else:
            last_avg_gnet = GNetNext_Last(avg_gnet)
            avg_gnet = GNetNext_Train(depth[scale_index], last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)

    netG = avg_gnet
    if cfg.ckpt_file == "":
        scale_checkpoint = os.path.join(cfg.ckpt_save_dir, str(cfg.scales[-1]))
        i_batch = int(cfg.num_batch[-1] / cfg.device_num - \
                      cfg.num_batch[-1] / cfg.device_num % cfg.model_save_step - 1)
        param_G = load_checkpoint(os.path.join(scale_checkpoint, "AvG_{}.ckpt".format(i_batch)))
    else:
        mox.file.copy_parallel(cfg.checkpoint_url, cfg.ckpt_save_dir)
        param_G = load_checkpoint(os.path.join(cfg.ckpt_save_dir, cfg.ckpt_file))
    load_param_into_net(netG, param_G)
    netG.set_train(False)
    inputNoise = buildNoiseData(64)
    export(netG, inputNoise, file_name="PGAN_AIR.air", file_format="AIR")
    print("PGAN exported")
    mox.file.copy_parallel("./", cfg.train_url)

def prepareTrain(cfg):
    """prepareTrain

    Returns:
        output.
    """
    if not os.path.exists(cfg.train_data_path):
        os.makedirs(cfg.train_data_path, 0o755)
    if not os.path.exists(cfg.resume_check_d):
        os.makedirs(cfg.resume_check_d, 0o755)
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path, 0o755)
    mox.file.copy_parallel(cfg.data_url, cfg.train_data_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """buildNoiseData"""
    cfg = config
    # copy dataset from obs to container
    prepareTrain(cfg)

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    cfg.device_num = get_device_num()
    if not os.path.exists(cfg.ckpt_save_dir):
        os.mkdir(cfg.ckpt_save_dir)
    if cfg.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if cfg.device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=cfg.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    each_batch_alpha = cal_each_batch_alpha()
    for scale_index, scale in enumerate(cfg.scales):
        this_scale_checkpoint = os.path.join(cfg.ckpt_save_dir, str(scale))
        if not os.path.exists(this_scale_checkpoint):
            os.mkdir(this_scale_checkpoint)
        if scale == 4:
            dnet = DNet4_4_Train(cfg.depth[scale_index], leakyReluLeak=0.2, sizeDecisionLayer=1, dimInput=3)
            gnet = GNet4_4_Train(512, cfg.depth[scale_index], leakyReluLeak=0.2, dimOutput=3)
            avg_gnet = GNet4_4_Train(512, cfg.depth[scale_index], leakyReluLeak=0.2, dimOutput=3)
        elif scale == 8:
            last_dnet = DNet4_4_Last(dnet)
            last_gnet = GNet4_4_last(gnet)
            dnet = DNetNext_Train(cfg.depth[scale_index], last_Dnet=last_dnet, leakyReluLeak=0.2, dimInput=3)
            gnet = GNetNext_Train(cfg.depth[scale_index], last_Gnet=last_gnet, leakyReluLeak=0.2, dimOutput=3)
            last_avg_gnet = GNet4_4_last(avg_gnet)
            avg_gnet = GNetNext_Train(cfg.depth[scale_index], last_Gnet=last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
        else:
            last_dnet = DNetNext_Last(dnet)
            last_gnet = GNetNext_Last(gnet)
            dnet = DNetNext_Train(cfg.depth[scale_index], last_Dnet=last_dnet, leakyReluLeak=0.2, dimInput=3)
            gnet = GNetNext_Train(cfg.depth[scale_index], last_gnet, leakyReluLeak=0.2, dimOutput=3)
            last_avg_gnet = GNetNext_Last(avg_gnet)
            avg_gnet = GNetNext_Train(cfg.depth[scale_index], last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
        cell_deepcopy(gnet, avg_gnet)
        if cfg.resume_load_scale != -1 and scale < cfg.resume_load_scale:
            continue
        elif cfg.resume_load_scale != -1 and scale == cfg.resume_load_scale:
            param_dict_g = load_checkpoint(cfg.resume_check_g)
            param_dict_d = load_checkpoint(cfg.resume_check_d)
            load_param_into_net(gnet, param_dict_g)
            load_param_into_net(dnet, param_dict_d)
            continue
        optimizerD = getOptimizerD(dnet, cfg)
        optimizerG = getOptimizerG(gnet, dnet, cfg)
        dbLoader = getDataset(cfg, (scale, scale))
        rank_size, rank_id = _get_rank_info()
        if rank_id:
            this_scale_checkpoint = os.path.join(this_scale_checkpoint, "rank_{}".format(rank_id))
        if not os.path.exists(this_scale_checkpoint):
            os.mkdir(this_scale_checkpoint)
        dataset = ds.GeneratorDataset(dbLoader, column_names=["data", "label"], shuffle=True,
                                      num_parallel_workers=4, num_shards=rank_size, shard_id=rank_id)
        dataset = dataset.batch(batch_size=cfg.batch_size, drop_remainder=True)
        dataset_iter = dataset.create_tuple_iterator()
        i_batch = 0
        while i_batch < cfg.num_batch[scale_index] / cfg.device_num:
            epoch = 0
            for data in dataset_iter:
                alpha = each_batch_alpha[scale_index][i_batch]
                alpha = mindspore.Tensor(alpha, mindspore.float32)
                inputs_real = data[0]
                n_samples = inputs_real.shape[0]
                inputLatent = buildNoiseData(n_samples)
                fake_image = gnet(inputLatent, alpha)
                lossD, overflow, _ = optimizerD(inputs_real, fake_image, alpha)
                inputNoise = buildNoiseData(n_samples)
                lossG, overflow, _ = optimizerG(inputNoise, alpha)
                cell_deepcopy_update(gnet=gnet, avg_gnet=avg_gnet)
                i_batch += 1
                if i_batch >= cfg.num_batch[scale_index] / cfg.device_num:
                    break
                if i_batch % 100 == 0:
                    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print('batch_i:{} alpha:{} loss G:{} loss D:{} overflow:{},time:{}'.format(i_batch, alpha, lossG,
                                                                                               lossD, overflow,
                                                                                               time_now))
                if (i_batch + 1) % cfg.model_save_step == 0:
                    save_checkpoint_g(avg_gnet, gnet, dnet, this_scale_checkpoint, i_batch)
            epoch += 1
        save_checkpoint_g(avg_gnet, gnet, dnet, this_scale_checkpoint, i_batch)

    # export
    exportOutput(cfg)

if __name__ == '__main__':
    run_train()
