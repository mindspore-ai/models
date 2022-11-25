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

"""General-purpose training script for image-to-image translation.
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
Example:
    Train a resnet model:
        python train.py --dataroot ./data/horse2zebra --model ResNet
"""

import os
import argparse
import datetime
import moxing as mox

import mindspore as ms
import mindspore.nn as nn
from src.utils.args import get_args
from src.utils.reporter import Reporter
from src.utils.tools import get_lr, ImagePool, load_ckpt
from src.dataset.cyclegan_dataset import create_dataset
from src.models.losses import DiscriminatorLoss, GeneratorLoss
from src.models.cycle_gan import get_generator, get_discriminator, Generator, TrainOneStepG, TrainOneStepD

ms.set_seed(1)
code_dir = os.path.dirname(__file__)
work_dir = os.getcwd()
print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

parser = argparse.ArgumentParser(description='CycleGAN Training Args')
parser.add_argument("--modelarts_FLAG", type=bool, default=True, help="use modelarts or not")
parser.add_argument('--dataroot', type=str, default='./data/',
                    help='path of images (should have subfolders trainA, trainB, testA, testB, etc).')
parser.add_argument("--outputs_dir", type=str, default="./output/")
parser.add_argument("--training_dataset", type=str, default="/cache/dataset/apple2orange/")
parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset/")
parser.add_argument("--modelarts_result_dir", type=str, default="/cache/train_output/")
parser.add_argument('--max_epoch', type=int, default=2, help='epoch size for training, default is 2.')
parser.add_argument('--n_epochs', type=int, default=1,
                    help='number of epochs with the initial learning rate, default is 1')

args_opt = parser.parse_args()


def obs_data2modelarts(cfg):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(cfg.dataroot, cfg.modelarts_data_dir))
    mox.file.copy_parallel(src_url=cfg.dataroot, dst_url=cfg.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(cfg.modelarts_data_dir)
    print("===>>>before Files:", files)


def modelarts_result2obs(cfg):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=cfg.modelarts_result_dir, dst_url=cfg.outputs_dir)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(cfg.modelarts_result_dir,
                                                                                  cfg.outputs_dir))


def train(cfg):
    """Train function."""
    if cfg.modelarts_FLAG:
        obs_data2modelarts(args_opt)
    args = get_args("train")
    if args.need_profiler:
        from mindspore.profiler.profiling import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)
    if cfg.modelarts_FLAG:
        args.dataroot = cfg.training_dataset
        args.outputs_dir = cfg.modelarts_result_dir
        if not os.path.exists(cfg.modelarts_result_dir):
            os.makedirs(cfg.modelarts_result_dir)
    ds = create_dataset(args)
    G_A = get_generator(args)
    G_B = get_generator(args)
    D_A = get_discriminator(args)
    D_B = get_discriminator(args)
    if args.load_ckpt:
        load_ckpt(args, G_A, G_B, D_A, D_B)
    imgae_pool_A = ImagePool(args.pool_size)
    imgae_pool_B = ImagePool(args.pool_size)
    generator = Generator(G_A, G_B, args.lambda_idt > 0)

    loss_D = DiscriminatorLoss(args, D_A, D_B)
    loss_G = GeneratorLoss(args, generator, D_A, D_B)
    optimizer_G = nn.Adam(generator.trainable_params(), get_lr(args), beta1=args.beta1)
    optimizer_D = nn.Adam(loss_D.trainable_params(), get_lr(args), beta1=args.beta1)

    net_G = TrainOneStepG(loss_G, generator, optimizer_G)
    net_D = TrainOneStepD(loss_D, optimizer_D)

    data_loader = ds.create_dict_iterator()
    if args.rank == 0:
        reporter = Reporter(args)
        reporter.info('==========start training===============')
    for _ in range(args.max_epoch):
        if args.rank == 0:
            reporter.epoch_start()
        for data in data_loader:
            img_A = data["image_A"]
            img_B = data["image_B"]
            res_G = net_G(img_A, img_B)
            fake_A = res_G[0]
            fake_B = res_G[1]
            res_D = net_D(img_A, img_B, imgae_pool_A.query(fake_A), imgae_pool_B.query(fake_B))
            if args.rank == 0:
                reporter.step_end(res_G, res_D)
                reporter.visualizer(img_A, img_B, fake_A, fake_B)
        if args.rank == 0:
            reporter.epoch_end(net_G)
        if args.need_profiler:
            profiler.analyse()
            break
    if args.rank == 0:
        reporter.info('==========end training===============')

    if cfg.modelarts_FLAG:
        modelarts_result2obs(args_opt)


if __name__ == "__main__":
    train(args_opt)
