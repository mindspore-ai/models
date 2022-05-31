# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""train"""
import os
import ast
import argparse
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.common import dtype as mstype
from mindspore.communication.management import init
from src.dataset import create_dataset
from src.model import Generator, Discriminator, init_weights
from src.cell import GenWithLossCell, DisWithLossCell, TrainOneStepG, TrainOneStepD
from src.tools import load_ckpt
from src.reporter import Reporter

def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='device target, Ascend or GPU (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of training (Default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size id of training (Default: 128)')
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='start epochs of training (Default: 0)')
    parser.add_argument('--epoch_size', type=int, default=50,
                        help='epoch size of training (Default: 50)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='.', help='checkpoint dir of CGAN')
    parser.add_argument('--D_ckpt_path', type=str,
                        default=None, help='checkpoint path of discriminator')
    parser.add_argument('--G_ckpt_path', type=str,
                        default=None, help='checkpoint path of generator')
    parser.add_argument('--data_path', type=str, default='data/MNIST_Data/train',
                        help='dataset dir (default data/MNIST_Data/train)')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='output path of training (default ./output)')
    parser.add_argument("--real_valued_mnist", type=bool, default=True,
                        help='If load real valued MNIST dataset(default False)')
    args = parser.parse_args()
    mindspore.common.set_seed(0)
    # if not exists 'ckpt_dir', make it
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target=args.device_target,)
    if args.distribute:
        context.set_auto_parallel_context(gradients_mean=True,
                                          device_num=int(os.getenv('DEVICE_NUM', '1')),
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
        init()
        if args.device_target == 'Ascend':
            context.set_context(device_id=args.device_id)
    else:
        context.set_context(device_id=args.device_id)
    return args


def main():
    # before training, we should set some arguments
    args = preLauch()

    # training argument
    lr = 0.0002
    classes_num = 10
    img_channels = 1
    img_size = 32
    input_dim = 100
    embed_size = 100
    start_epochs = args.start_epochs
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    G_ckpt_path = args.G_ckpt_path
    D_ckpt_path = args.D_ckpt_path
    output_path = args.output_path

    dataset = create_dataset(args.data_path, usage='train', img_size=img_size, batch_size=batch_size,
                             real_valued_mnist=args.real_valued_mnist, num_parallel_workers=1)
    dataset_size = dataset.get_dataset_size()
    dataset = dataset.create_dict_iterator()
    print(dataset_size)
    # create G Cell & D Cell
    netG = Generator(input_dim, img_channels, classes_num, embed_size=embed_size)
    netD = Discriminator(img_channels, classes_num, embed_size=img_size*img_size)
    init_weights(netG)
    init_weights(netD)

    if G_ckpt_path is not None:
        load_ckpt(G_ckpt_path, netG)
    if D_ckpt_path is not None:
        load_ckpt(D_ckpt_path, netD)
    # create WithLossCell
    netG_with_loss = GenWithLossCell(netG, netD)
    netD_with_loss = DisWithLossCell(netG, netD)

    # create optimizer cell
    optimizerG = nn.Adam(netG.trainable_params(), lr, beta1=0.5)
    optimizerD = nn.Adam(netD.trainable_params(), lr, beta1=0.5)

    net_G_train = TrainOneStepG(netG_with_loss, optimizerG)
    net_D_train = TrainOneStepD(netD_with_loss, optimizerD)

    netG.set_train()
    netD.set_train()

    reporter = Reporter(output_path=output_path, stage='train', start_epochs=start_epochs,
                        dataset_size=dataset_size, batch_size=batch_size)
    reporter.info('==========start training===============')
    fixed_noise = Tensor(np.random.normal(size=(batch_size, input_dim)), dtype=mstype.float32)
    fixed_label = Tensor((np.arange(fixed_noise.shape[0]) % 10), dtype=mstype.int32)

    for _ in range(start_epochs, epoch_size):
        reporter.epoch_start()
        for data in dataset:
            real_img = data['image']
            label = data['label']
            noise = Tensor(np.random.normal(size=(batch_size, input_dim)),
                           dtype=mstype.float32)

            res_D = net_D_train(real_img, noise, label)
            fake_img, res_G = net_G_train(noise, label)

            fixed_fake_img = netG(fixed_noise, fixed_label)
            reporter.step_end(res_G, res_D)
            reporter.visualizer(fake_img, fixed_fake_img)
        reporter.epoch_end(net_G_train)
    reporter.end_train()

if __name__ == '__main__':
    main()
