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

"""
This is the boot file for ModelArts platform.
Firstly, the train datasets are copied from obs to ModelArts.
Then, the string of train shell command is concated and using 'os.system()' to execute
"""

# train scripts

import os
import argparse
import time
import datetime
import numpy as np
import moxing as mox
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio

from mindspore import Tensor
from mindspore.train.serialization import export
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net, nn
from mindspore.context import ParallelMode
import mindspore.ops as ops
from src.model.generator import get_generator
from src.model.discriminator import get_discriminator
from src.dataset.traindataset import create_traindataset
from src.dataset.testdataset import create_testdataset
from src.loss.psnr_loss import PSNRLoss
from src.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from src.trainonestep.train_psnr import TrainOnestepPSNR
from src.trainonestep.train_gan import TrainOneStepD
from src.trainonestep.train_gan import TrainOnestepG

parser = argparse.ArgumentParser(description="SRGAN train")
parser.add_argument("--train_LR_path", type=str, default='/data/DIV2K/LR')
parser.add_argument("--train_GT_path", type=str, default='/data/DIV2K/HR')
parser.add_argument("--val_LR_path", type=str, default='/data/Set5/LR')
parser.add_argument("--val_GT_path", type=str, default='/data/Set5/HR')
parser.add_argument("--vgg_ckpt", type=str,
                    default='/data/pre-models/vgg19/vgg19.ckpt')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument("--image_size", type=int, default=96,
                    help="Image size of high resolution image. (default: 96)")
parser.add_argument("--train_batch_size", default=16, type=int,
                    metavar="N",
                    help="batch size for training")
parser.add_argument("--val_batch_size", default=1, type=int,
                    metavar="N",
                    help="batch size for tesing")
parser.add_argument("--psnr_epochs", default=2000, type=int, metavar="N",
                    help="Number of total psnr epochs to run. (default: 2000)")
parser.add_argument("--start_psnr_epoch", default=0, type=int, metavar='N',
                    help="Manual psnr epoch number (useful on restarts). (default: 0)")
parser.add_argument("--gan_epochs", default=1000, type=int, metavar="N",
                    help="Number of total gan epochs to run. (default: 1000)")
parser.add_argument("--start_gan_epoch", default=0, type=int, metavar='N',
                    help="Manual gan epoch number (useful on restarts). (default: 0)")
parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"),
                    help='network initialization, default is normal.')
parser.add_argument("--scale", type=int, default=4)
# distribute
parser.add_argument("--run_distribute", type=int, default=0,
                    help="Run distribute, default: false.")
parser.add_argument("--device_id", type=int, default=0,
                    help="device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1,
                    help="number of device, default: 0.")
parser.add_argument("--rank_id", type=int, default=0,
                    help="Rank id, default: 0.")
# model art
parser.add_argument("--data_url", type=str, default="./dataset",
                    help='real input file path, include DIV2K(DIV2K_train_HR,DIV2K_train_LR_bicubic_X4)')
parser.add_argument("--modelarts_data_dir", type=str,
                    default="/cache/dataset", help='modelart input path')
parser.add_argument("--modelarts_result_dir", type=str,
                    default="/cache/result", help='modelart output path.')
parser.add_argument("--obs_result_dir", type=str, default="./output",
                    help='real output file path include .ckpt and .air')  # modelarts -> obs
parser.add_argument("--modelarts_attrs", type=str, default="")

print(os.system('env'))


def obs_data2modelarts(argus):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(
        argus.data_url, argus.modelarts_data_dir))
    mox.file.copy_parallel(src_url=argus.data_url,
                           dst_url=argus.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format(
        (end - start).seconds))
    files = os.listdir(argus.modelarts_data_dir)
    print("===>>>Files:", files)
    files = os.listdir(argus.modelarts_data_dir + "/DIV2K/DIV2K_train_HR")
    print("===>>>HR files:", files)
    files = os.listdir(argus.modelarts_data_dir +
                       "/DIV2K/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4")
    print("===>>>LR files:", files)
    files = os.listdir(argus.modelarts_data_dir + "/Set5/HR")
    print("===>>>Test HR files:", files)
    files = os.listdir(argus.modelarts_data_dir + "/Set5/LR")
    print("===>>>Test LR files:", files)
    files = os.listdir(argus.modelarts_data_dir + "/pre-models")
    print("===>>>pretrain model files:", files)

    if not mox.file.exists(argus.obs_result_dir):
        mox.file.make_dirs(argus.obs_result_dir)
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(
        argus.obs_result_dir, argus.modelarts_result_dir))
    mox.file.copy_parallel(src_url=argus.obs_result_dir,
                           dst_url=argus.modelarts_result_dir)
    files = os.listdir(argus.modelarts_result_dir)
    print("===>>>Files:", files)


def modelarts_result2obs(argus):
    """
    Copy result data from modelarts to obs.
    """
    obs_result_dir = argus.obs_result_dir
    if not mox.file.exists(obs_result_dir):
        print(f"obs_result_dir[{obs_result_dir}] not exist!")
        mox.file.make_dirs(obs_result_dir)
    mox.file.copy_parallel(
        src_url='./ckpt', dst_url=os.path.join(obs_result_dir, 'ckpt'))
    print("===>>>Copy Event or Checkpoint from modelarts dir: ./ckpt to obs:{}".format(obs_result_dir))
    mox.file.copy(src_url='SRGAN_G_model_GAN.air',
                  dst_url=os.path.join(obs_result_dir, 'SRGAN_G_model_GAN.air'))
    files = os.listdir(obs_result_dir)
    print("===>>>current Files:", files)


def export_AIR():
    """
    export AIR model from ckpt.
    """
    from src.model.generator import Generator
    # start modelarts export
    net = Generator(4)

    paras = load_checkpoint("ckpt/G_model_GAN_final.ckpt")
    load_param_into_net(generator, paras)

    input_shp = [1, 3, 126, 126]
    input_array = Tensor(
        np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(net, input_array, file_name="SRGAN_G_model_GAN", file_format='AIR')


if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE,
                        device_id=args.device_id, save_graphs=False)
    # distribute
    if args.run_distribute:
        print("distribute")
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))
        device_num = args.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()

        rank = get_rank()

    # copy dataset from obs to modelarts
    obs_data2modelarts(args)

    args.train_LR_path = args.modelarts_data_dir + \
                         "/DIV2K/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4"
    args.train_GT_path = args.modelarts_data_dir + "/DIV2K/DIV2K_train_HR"
    args.val_LR_path = args.modelarts_data_dir + "/Set5/LR"
    args.val_GT_path = args.modelarts_data_dir + "/Set5/HR"
    args.vgg_ckpt = args.modelarts_data_dir + "/pre-models/vgg19.ckpt"

    # for srresnet
    # create dataset
    train_ds = create_traindataset(
        args.train_batch_size, args.train_LR_path, args.train_GT_path)
    test_ds = create_testdataset(
        args.val_batch_size, args.val_LR_path, args.val_GT_path)
    train_data_loader = train_ds.create_dict_iterator()
    test_data_loader = test_ds.create_dict_iterator()
    # definition of network
    generator = get_generator(4, 0.02)

    # network with loss

    psnr_loss = PSNRLoss(generator)

    # optimizer
    psnr_optimizer = nn.Adam(generator.trainable_params(), 1e-4)

    # operation for testing
    op = ops.ReduceSum(keep_dims=False)
    # trainonestep

    train_psnr = TrainOnestepPSNR(psnr_loss, psnr_optimizer)
    train_psnr.set_train()

    bestpsnr = 0
    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt")
    print('start training:')

    print('start training PSNR:')
    # warm up generator
    for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
        print("training {:d} epoch:".format(epoch + 1))
        mysince = time.time()
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            mse_loss = train_psnr(hr, lr)
        steps = train_ds.get_dataset_size()
        time_elapsed = (time.time() - mysince)
        step_time = time_elapsed / steps
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        print("mse_loss:")
        print(mse_loss)
        psnr_list = []
        # val for every epoch
        print("start valing:")
        for test in test_data_loader:
            lr = test['LR']
            gt = test['HR']

            bs, c, h, w = lr.shape[:4]
            gt = gt[:, :, : h * args.scale, : w * args.scale]

            output = generator(lr)
            output = op(output, 0)
            output = output.asnumpy()
            output = np.clip(output, -1.0, 1.0)
            gt = op(gt, 0)

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.asnumpy()
            gt = gt.transpose(1, 2, 0)

            y_output = rgb2ycbcr(
                output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            psnr = peak_signal_noise_ratio(
                y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)

        mean = np.mean(psnr_list)
        print("psnr:")
        print(mean)
        if mean > bestpsnr:
            print("saving ckpt")
            bestpsnr = mean
            if args.run_distribute == 0:
                save_checkpoint(train_psnr, "./ckpt/best.ckpt")
            else:
                if args.device_id == 0:
                    save_checkpoint(train_psnr, "./ckpt/best.ckpt")

        if (epoch + 1) % 200 == 0:
            if args.run_distribute == 0:
                save_checkpoint(
                    train_psnr, './ckpt/pre_trained_model_%03d.ckpt' % (epoch + 1))
            else:
                if args.device_id == 0:
                    save_checkpoint(
                        train_psnr, './ckpt/pre_trained_model_%03d.ckpt' % (epoch + 1))

        print("{:d}/2000 epoch finished".format(epoch + 1))
    # for srgan
    generator = get_generator(4, 0.02)
    discriminator = get_discriminator(96, 0.02)
    if args.run_distribute == 0:
        ckpt = "./ckpt/best.ckpt"
    else:
        ckpt = '../train_parallel0/ckpt/best.ckpt'
    params = load_checkpoint(ckpt)
    load_param_into_net(generator, params)
    discriminator_loss = DiscriminatorLoss(discriminator, generator)
    generator_loss = GeneratorLoss(discriminator, generator, args.vgg_ckpt)
    generator_optimizer = nn.Adam(generator.trainable_params(), 1e-4)
    discriminator_optimizer = nn.Adam(discriminator.trainable_params(), 1e-4)
    train_discriminator = TrainOneStepD(
        discriminator_loss, discriminator_optimizer)
    train_generator = TrainOnestepG(generator_loss, generator_optimizer)
    print("========================================")
    print('start training GAN :')
    # trainGAN
    for epoch in range(args.start_gan_epoch, args.gan_epochs):
        print('training {:d} epoch'.format(epoch + 1))
        mysince1 = time.time()
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            D_loss = train_discriminator(hr, lr)
            G_loss = train_generator(hr, lr)
        time_elapsed1 = (time.time() - mysince1)
        steps = train_ds.get_dataset_size()
        step_time1 = time_elapsed1 / steps
        print('per step needs time:{:.0f}ms'.format(step_time1 * 1000))
        print("D_loss:")
        print(D_loss.mean())
        print("G_loss:")
        print(G_loss.mean())

        if (epoch + 1) % 100 == 0:
            print("saving ckpt")
            if args.run_distribute == 0:
                save_checkpoint(train_generator,
                                './ckpt/G_model_%03d.ckpt' % (epoch + 1))
                save_checkpoint(train_discriminator,
                                './ckpt/D_model_%03d.ckpt' % (epoch + 1))
            else:
                if args.device_id == 0:
                    save_checkpoint(train_generator,
                                    './ckpt/G_model_%03d.ckpt' % (epoch + 1))
                    save_checkpoint(train_discriminator,
                                    './ckpt/D_model_%03d.ckpt' % (epoch + 1))
        print(" {:d}/1000 epoch finished".format(epoch + 1))
    print('save final checkpoint')
    save_checkpoint(train_generator, './ckpt/G_model_GAN_final.ckpt')
    save_checkpoint(train_discriminator, './ckpt/D_model_GAN_final.ckpt')
    # start export air
    export_AIR()
    # copy result from modelarts to obs
    modelarts_result2obs(args)
