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
'''train'''
import os
import datetime
import time
import glob
import numpy as np
import PIL.Image as Image

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.common.tensor import Tensor
from mindspore import save_checkpoint
from mindspore.dataset import config
from mindspore import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size

from mindspore.ops import operations as P
from mindspore.ops import composite as C

from src.logger import get_logger
from src.dataset import create_Dataset
from src.models import UNet, UNetWithLossCell
from src.util import AverageMeter
from src.dataset import AugmentNoise
from src.config import config as cfg


def get_lr(steps_per_epoch, max_epoch, init_lr, gamma):
    lr_each_step = []
    while max_epoch > 0:
        tem = min(20, max_epoch)
        for _ in range(steps_per_epoch*tem):
            lr_each_step.append(init_lr)
        max_epoch -= tem
        init_lr *= gamma
    return lr_each_step

def space_to_depth(x, block_size):
    '''space_to_depth'''
    n, c, h, w = x.shape #([4, 1, 256, 256])
    unfolded_x = np.zeros((n, block_size*block_size, w*h // (block_size*block_size)))#([4, 4, 16384])
    for j in range(n):
        tx1 = x[j, 0, :, :][::2].reshape(-1, block_size)
        tx2 = x[j, 0, :, :][1::2].reshape(-1, block_size)
        for i in range(block_size):
            unfolded_x[j, i] = tx1[:, i]
        for i in range(block_size):
            unfolded_x[j, i+block_size] = tx2[:, i]

    return unfolded_x.reshape((n, c * block_size**2, h // block_size,
                               w // block_size))

def generate_subimages(img, mask):
    '''generate_subimages'''
    n, c, h, w = img.shape
    subimage = np.zeros((n, c, h // 2, w // 2), dtype=img.dtype)
    # per channel
    for i in range(c): #NCHW
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = np.transpose(img_per_channel, (0, 2, 3, 1)).reshape(-1)
        subimage[:, i:i + 1, :, :] = np.transpose(img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1), (0, 3, 1, 2))
    return subimage


def generate_mask_pair(img):
    '''generate_mask_pair'''
    # prepare masks (N x C x H/2 x W/2)
    n, _, h, w = img.shape
    mask1 = np.zeros(shape=(n * h // 2 * w // 2 * 4,),
                     dtype=bool)
    mask2 = np.zeros(shape=(n * h // 2 * w // 2 * 4,),
                     dtype=bool)
    # prepare random mask pairs
    idx_pair = np.array(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=np.int64)
    rd_idx = np.random.randint(low=0,
                               high=8,
                               size=(n * h // 2 * w // 2,),
                               dtype=np.int64)

    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += np.arange(0,
                             n * h // 2 * w // 2 * 4,
                             4,
                             dtype=np.int64).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def copy_data_from_obs():
    '''copy_data_from_obs'''
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying train data from obs to cache....")
        mox.file.copy_parallel(cfg.train_data, 'cache/dataset')
        cfg.logger.info("copying traindata finished....")
        cfg.train_data = 'cache/dataset/'
        if cfg.eval_while_train:
            cfg.logger.info("copying test data from obs to cache....")
            mox.file.copy_parallel(cfg.test_dir, 'cache/test')
            cfg.logger.info("copying test data finished....")
            cfg.test_dir = 'cache/test/'
        if cfg.resume_path:
            cfg.logger.info("copying resume checkpoint from obs to cache....")
            mox.file.copy_parallel(cfg.resume_path, 'cache/resume_path')
            cfg.logger.info("copying resume checkpoint finished....")
            cfg.resume_path = 'cache/resume_path/'

def copy_data_to_obs():
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(cfg.save_dir, cfg.outer_path)
        cfg.logger.info("copying finished....")

def main():
    '''main'''
    dataset, cfg.steps_per_epoch = create_Dataset(cfg.train_data, cfg.patchsize, \
                        cfg.noisetype, cfg.batch_size, cfg.group_size, cfg.rank, shuffle=True)
    f_model = UNet(in_nc=cfg.n_channel, out_nc=cfg.n_channel, n_feature=cfg.n_feature)
    if cfg.resume_path:
        cfg.resume_path = os.path.join(cfg.resume_path, cfg.resume_name)
        cfg.logger.info('loading resume checkpoint %s into network', str(cfg.resume_path))
        load_param_into_net(f_model, load_checkpoint(cfg.resume_path))
        cfg.logger.info('loaded resume checkpoint %s into network', str(cfg.resume_path))
    model = UNetWithLossCell(f_model)
    model.set_train()
    lr_list = get_lr(cfg.steps_per_epoch, cfg.epoch, float(cfg.lr), cfg.gamma)
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=Tensor(lr_list, mindspore.float32))
    model = nn.TrainOneStepCell(model, optimizer)

    data_loader = dataset.create_dict_iterator()
    loss_meter = AverageMeter('loss')
    for k in range(cfg.epoch):
        model.set_train(True)
        t_end = time.time()
        old_progress = -1
        for i, data in enumerate(data_loader):
            noisy = data["noisy"].asnumpy()
            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)

            noisy_denoised = f_model(data["noisy"]).asnumpy()
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
            Lambda = k / cfg.epoch * cfg.increase_ratio
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            loss = model(Tensor(noisy_sub1, mindspore.float32), \
                         Tensor(noisy_sub2, mindspore.float32), \
                         Tensor(exp_diff, mindspore.float32), \
                         Tensor(Lambda, mindspore.float32))
            loss_meter.update(loss.asnumpy())
            if i % cfg.log_interval == 0:
                if cfg.rank == 0:
                    time_used = time.time()- t_end
                    fps = cfg.batch_size * (i - old_progress) * cfg.group_size / time_used
                    cfg.logger.info(
                        'epoch[{}], iter[{}], {}, {:.2f} imgs/sec, lr:{}'.format(\
                         k, i, loss_meter, fps, lr_list[k*cfg.steps_per_epoch + i]))
                    t_end = time.time()
                    loss_meter.reset()
                    old_progress = i
        if cfg.rank_save_ckpt_flag:
            # checkpoint save
            save_checkpoint(model, os.path.join(cfg.save_dir, str(cfg.rank)+"_last_map.ckpt"))
        if cfg.eval_while_train and (k+1) > cfg.eval_start_epoch and (k+1)%cfg.eval_steps == 0:
            test(f_model)
    cfg.logger.info("training finished....")

def test(model):
    '''test'''
    noise_generator = AugmentNoise(cfg.noisetype)

    model.set_train(False)
    transpose = P.Transpose()
    expand_dims = P.ExpandDims()
    compare_psnr = nn.PSNR()
    compare_ssim = nn.SSIM()
    best_value = 0.
    for filename in os.listdir(cfg.test_dir):
        tem_path = os.path.join(cfg.test_dir, filename)

        psnr = []   #after denoise
        ssim = []   #after denoise
        psnr_b = [] #before denoise
        ssim_b = [] #before denoise
        file_list = glob.glob(os.path.join(tem_path, '*'))

        cfg.logger.info('Start to test on %s', str(tem_path))
        start_time = time.time()
        for file in file_list:
            # read image
            img_clean = np.array(Image.open(file), dtype='float32') / 255.0

            img_test = noise_generator.add_noise(img_clean)
            H = img_test.shape[0]
            W = img_test.shape[1]
            val_size = (max(H, W) + 31) // 32 * 32
            img_test = np.pad(img_test,
                              [[0, val_size - H], [0, val_size - W], [0, 0]],
                              'reflect')
            img_clean = Tensor(img_clean, mindspore.float32) #HWC
            img_test = Tensor(img_test, mindspore.float32) #HWC

            # predict
            img_clean = expand_dims(transpose(img_clean, (2, 0, 1)), 0)#NCHW
            img_test = expand_dims(transpose(img_test, (2, 0, 1)), 0)#NCHW

            prediction = model(img_test)
            y_predict = prediction[:, :, :H, :W]

            # calculate numeric metrics
            img_out = C.clip_by_value(y_predict, 0, 1)

            psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test[:, :, :H, :W]), \
                                        compare_psnr(img_clean, img_out)
            ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test[:, :, :H, :W]), \
                                        compare_ssim(img_clean, img_out)

            psnr.append(psnr_denoised.asnumpy()[0])
            ssim.append(ssim_denoised.asnumpy()[0])
            psnr_b.append(psnr_noise.asnumpy()[0])
            ssim_b.append(ssim_noise.asnumpy()[0])

        psnr_avg = sum(psnr)/len(psnr)
        ssim_avg = sum(ssim)/len(ssim)
        psnr_avg_b = sum(psnr_b)/len(psnr_b)
        ssim_avg_b = sum(ssim_b)/len(ssim_b)
        best_value += (psnr_avg * 0.1 + ssim_avg * 10.)
        cfg.logger.info("Result in:%s", str(tem_path))
        cfg.logger.info('Before denoise: Average PSNR_b = {0:.4f}, SSIM_b = {1:.4f};'\
                        .format(psnr_avg_b, ssim_avg_b))
        cfg.logger.info('After denoise: Average PSNR = {0:.4f}, SSIM = {1:.4f}'\
                        .format(psnr_avg, ssim_avg))
        cfg.logger.info("testing finished....")
        time_used = time.time() - start_time
        cfg.logger.info("time cost:%s seconds!", str(time_used))
    if cfg.best_value < best_value:
        cfg.best_value = best_value
        save_checkpoint(model, os.path.join(cfg.save_dir, str(cfg.rank)+"_best_map.ckpt"))
        cfg.logger.info("Update newly best ckpt! best_value: %s", str(best_value))

if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.join(cfg.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not cfg.use_modelarts and not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, save_graphs=False)
    if cfg.is_distributed:
        if cfg.device_target == "Ascend":
            context.set_context(device_id=device_id)
            init("hccl")
        else:
            assert cfg.device_target == "GPU"
            init("nccl")
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        device_num = cfg.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        if cfg.device_target in ["Ascend", "GPU"]:
            context.set_context(device_id=device_id)
    config.set_enable_shared_mem(False) # we may get OOM when it set to 'True'
    cfg.logger = get_logger(cfg.save_dir, "Neighbor2Neighbor", cfg.rank)
    cfg.logger.save_args(cfg)
    cfg.rank_save_ckpt_flag = not (cfg.is_save_on_master and cfg.rank)
    cfg.best_value = 0.
    copy_data_from_obs()
    main()
    copy_data_to_obs()
    cfg.logger.info('All task finished!')
