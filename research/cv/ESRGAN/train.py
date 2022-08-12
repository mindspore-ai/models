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

"""train scripts"""

import os
from os import path as osp
import argparse
import math
import time
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.nn import piecewise_constant_lr
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore.context import ParallelMode

from src.model.generator import get_generator
from src.model.discriminator import get_discriminator
from src.dataset.traindataset import create_traindataset
from src.dataset.testdataset import create_testdataset
from src.loss.psnr_loss import PSNRLoss
from src.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from src.trainonestep.train_psnr import TrainOnestepPSNR
from src.trainonestep.train_gan import TrainOneStepD
from src.trainonestep.train_gan import TrainOnestepG
from src.util.util import tensor2img, imwrite, calculate_psnr, CosineAnnealingRestartLR

mindspore.set_seed(0)

parser = argparse.ArgumentParser(description="ESRGAN train")
parser.add_argument("--data_url", type=str, default='', help="data url.")
parser.add_argument("--train_url", type=str, default='', help="train url.")

# dataset
parser.add_argument("--train_LR_path", type=str, default='/data/DIV2K/DIV2K_train_LR_bicubic/X4_sub')
parser.add_argument("--train_GT_path", type=str, default='/data/DIV2K/DIV2K_train_HR_sub')
parser.add_argument("--val_PSNR_LR_path", type=str, default='/data/Set5/LRbicx4')
parser.add_argument("--val_PSNR_GT_path", type=str, default='/data/Set5/GTmod12')
parser.add_argument("--val_GAN_LR_path", type=str, default='/data/Set14/LRbicx4')
parser.add_argument("--val_GAN_GT_path", type=str, default='/data/Set14/GTmod12')

parser.add_argument("--vgg_ckpt", type=str, default='/ckpt/vgg19.ckpt')

parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
parser.add_argument("--image_size", type=int, default=128, help="Image size of high resolution image. (default: 128)")
parser.add_argument("--train_batch_size", default=16, type=int, metavar="N", help="batch size for training")
parser.add_argument("--val_batch_size", default=1, type=int, metavar="N", help="batch size for tesing")

parser.add_argument("--start_psnr_epoch", default=0, type=int, metavar='N',
                    help="Manual gan epoch number (useful on restarts). (default: 0)")
parser.add_argument("--psnr_steps", default=1000000, type=int, metavar="N",
                    help="Number of total psnr steps to run. (default: 1000000)")

parser.add_argument("--start_gan_epoch", default=0, type=int, metavar='N',
                    help="Manual gan epoch number (useful on restarts). (default: 0)")
parser.add_argument("--gan_steps", default=400000, type=int, metavar="N",
                    help="Number of total gan epochs to run. (default: 400000)")
parser.add_argument("--sens", default=1024.0, type=float)
parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'))
# distribute
parser.add_argument("--modelArts", type=int, default=0, help="Run cloud, default: false.")
parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: false.")
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
parser.add_argument("--version", type=int, default=0, help="version, default: 0.")
parser.add_argument("--best_ckpt_path", type=str, default="./best_ckpt",
                    help="best ckpt save path, not use in modelarts.")



def evaluate(model, test_data_loader, cur_step, cur_epoch, cur_best_psnr, eval_mode, runs_path):
    """evaluate for every epoch"""
    print("start valing:")
    psnr_list = []
    total = 0
    for test in test_data_loader:
        total += 1
        lq = test['LR']
        gt = test['HR']
        output = model(lq)
        sr_img = tensor2img(output)
        gt_img = tensor2img(gt)
        save_img_path = osp.join(runs_path, f'{cur_epoch}', f'{cur_step}_{eval_mode}_epoch_{total}.png')
        imwrite(sr_img, save_img_path)
        cur_psnr = calculate_psnr(sr_img, gt_img)
        psnr_list.append(cur_psnr)
    psnr_mean = np.mean(psnr_list)
    print("val ending psnr = ", np.mean(psnr_list), cur_best_psnr)
    return psnr_mean


if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    context.set_context(device_target=args.platform)
    # distribute
    if args.modelArts:
        import moxing as mox
        print("modelArts")
        rank_id = int(os.getenv('RANK_ID'))
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))
        device_num = args.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        shard_id = rank_id
        num_shards = device_num
        rank = get_rank()

        device_id = int(os.getenv('DEVICE_ID'))
        data_root = "/cache/ESRGAN"
        local_data_url = os.path.join(data_root, str(args.version))
        device_data_url = os.path.join(local_data_url, "device{0}".format(device_id))

        local_train_runs_path = os.path.join(device_data_url, 'runs')
        local_train_ckpt_path = os.path.join(device_data_url, 'ckpt')
        best_ckpt_path = os.path.join(local_data_url, "best_ckpt")

        mox.file.make_dirs(local_data_url)
        mox.file.make_dirs(device_data_url)
        mox.file.make_dirs(local_train_runs_path)
        mox.file.make_dirs(local_train_ckpt_path)
        if rank == 0:
            mox.file.make_dirs(best_ckpt_path)
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
    else:
        local_train_runs_path = './runs'
        local_train_ckpt_path = './ckpt'
        best_ckpt_path = args.best_ckpt_path
        if args.run_distribute:
            print("distribute")
            if args.platform == 'Ascend':
                rank_id = int(os.getenv('RANK_ID'))
                device_id = int(os.getenv("DEVICE_ID"))
                context.set_context(device_id=device_id)

            device_num = args.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()
            if args.platform == 'GPU':
                rank = get_rank()
                rank_id = rank

            shard_id = rank_id
            num_shards = device_num
            rank = get_rank()

        else:
            context.set_context(device_id=args.device_id)
            device_num = args.device_num
            shard_id = None
            num_shards = None
            if args.platform == 'GPU':
                rank = 0
                shard_id = 0
                num_shards = 1
    # for RRDBNet
    # create dataset
    args.train_batch_size = int(args.train_batch_size // device_num) if args.run_distribute else args.train_batch_size
    train_ds = create_traindataset(args.train_batch_size, args.train_LR_path, args.train_GT_path, args.upscale_factor,
                                   args.image_size, num_shards, shard_id)
    test_psnr_ds = create_testdataset(args.val_batch_size, args.val_PSNR_LR_path, args.val_PSNR_GT_path)
    train_data_loader = train_ds.create_dict_iterator()
    test_psnr_data_loader = test_psnr_ds.create_dict_iterator()
    # definition of network
    generator = get_generator(3, 3)

    # network with loss
    psnr_loss = PSNRLoss(generator)

    loss_scale = args.sens

    lr = CosineAnnealingRestartLR(250000, restart_weights=[1, 1, 1, 1], eta_min=1e-7, total_step=1000000)

    # optimizer
    psnr_optimizer = nn.Adam(generator.trainable_params(), lr, loss_scale=loss_scale)

    # trainonestep
    train_psnr = TrainOnestepPSNR(psnr_loss, psnr_optimizer, sens=loss_scale)
    train_psnr.set_train()

    best_psnr = 0.0

    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt", exist_ok=True)
    if not os.path.exists("./runs"):
        os.makedirs("./runs", exist_ok=True)
    print('start training:')

    print('start training PSNR:')
    # warm up generator
    steps = train_ds.get_dataset_size()
    num_iter_per_epoch = math.ceil(steps)
    total_psnr_iters = int(args.psnr_steps)
    total_psnr_epochs = math.floor(total_psnr_iters / (num_iter_per_epoch))  # 490
    for epoch in range(args.start_psnr_epoch, total_psnr_epochs):
        print("training {:d} epoch:".format(epoch + 1))
        step = 0
        mysince = time.time()
        for data in train_data_loader:
            step = step + 1
            LR = data['LR']
            HR = data['HR']
            PSNR_loss = train_psnr(HR, LR)
        time_elapsed = (time.time() - mysince)
        step_time = time_elapsed / steps
        print('per epoch needs time:{:.0f}ms'.format(time_elapsed * 1000))
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        # val for every epoch
        psnr = evaluate(generator, test_psnr_data_loader, step, epoch, best_psnr, "PSNR", local_train_runs_path)
        # Check whether the evaluation index of the current model is the highest.
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        if is_best and rank == 0:
            print("best_psnr saving ckpt    ", end="")
            print(best_psnr)
            save_checkpoint(generator, os.path.join(best_ckpt_path, 'psnr_best.ckpt'))
            save_ckeckpoint(generator, os.path.join(local_train_ckpt_path, f'{epoch}_psnr_generator.ckpt'))
            print(f"{epoch + 1}/{total_psnr_epochs} epoch finished")
    # for esrgan
    test_gan_ds = create_testdataset(args.val_batch_size, args.val_GAN_LR_path, args.val_GAN_GT_path)
    test_gan_data_loader = test_gan_ds.create_dict_iterator()

    generator = get_generator(3, 3)
    discriminator = get_discriminator(3)
    # load checkpoint
    params = load_checkpoint(os.path.join(best_ckpt_path, 'psnr_best.ckpt'))
    load_param_into_net(generator, params)

    milestone = [50000, 100000, 200000, 300000, 400000]
    learning_rates = [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625]

    lr_gan = piecewise_constant_lr(milestone, learning_rates)

    discriminator_loss = DiscriminatorLoss(discriminator, generator, args)
    generator_loss = GeneratorLoss(discriminator, generator, args.vgg_ckpt, args)
    generator_optimizer = nn.Adam(generator.trainable_params(), lr_gan, loss_scale=loss_scale)
    discriminator_optimizer = nn.Adam(discriminator.trainable_params(), lr_gan, loss_scale=loss_scale)
    train_discriminator = TrainOneStepD(discriminator_loss, discriminator_optimizer, sens=loss_scale)
    train_generator = TrainOnestepG(generator_loss, generator_optimizer, sens=loss_scale)
    print("========================================")
    print('start training GAN :')
    train_discriminator.set_train()
    train_generator.set_train()

    best_psnr = 0.0
    # trainGAN
    total_gan_iters = int(args.gan_steps)
    total_gan_epochs = math.floor(total_gan_iters / (num_iter_per_epoch))
    for epoch in range(args.start_gan_epoch, total_gan_epochs):  # 196
        print('training {:d} epoch'.format(epoch + 1))
        step = 0
        mysince1 = time.time()
        for data in train_data_loader:
            step = step + 1
            LR = data['LR']
            HR = data['HR']
            G_loss = train_generator(HR, LR)
            D_loss = train_discriminator(HR, LR)
        time_elapsed = (time.time() - mysince1)
        step_time = time_elapsed / steps
        print('per epoch needs time:{:.0f}ms'.format(time_elapsed * 1000))
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        # val for every epoch
        psnr = evaluate(generator, test_gan_data_loader, step, epoch, best_psnr, "GAN", local_train_runs_path)
        # Check whether the evaluation index of the current model is the highest.
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        print(best_psnr)
        if is_best and rank == 0:
            print("best_psnr saving ckpt    ", end="")
            print(best_psnr)
            save_checkpoint(generator, os.path.join(best_ckpt_path, 'gan_generator_best.ckpt'))
            save_checkpoint(discriminator, os.path.join(best_ckpt_path, 'gan_discriminator_best.ckpt'))
        # save checkpoint every epoch
            save_checkpoint(generator, os.path.join(local_train_ckpt_path, f'{epoch}_gan_generator.ckpt'))
            save_checkpoint(discriminator, os.path.join(local_train_ckpt_path, f'{epoch}_gan_discriminator.ckpt'))
        if epoch == total_gan_epochs - 1 and rank == 0:
            save_checkpoint(generator, os.path.join(local_train_ckpt_path, 'gan_generator.ckpt'))
        print(f"{epoch + 1}/{total_gan_epochs} epoch finished")
    print("all")

    if args.modelArts:
        mox.file.copy_parallel(src_url=local_train_runs_path, dst_url=args.train_url)
        mox.file.copy_parallel(src_url=local_train_ckpt_path, dst_url=args.train_url)
        if rank == 0:
            mox.file.copy_parallel(src_url=best_ckpt_path, dst_url=args.train_url)
