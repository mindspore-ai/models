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
import datetime
import argparse
import numpy as np
import moxing as mox
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_group_size
from mindspore.train.serialization import export
from src.reporter import Reporter
from src.dataset import create_dataset
from src.model import Generator, Discriminator, init_weights
from src.cell import GenWithLossCell, DisWithLossCell, TrainOneStepG, TrainOneStepD


def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size id of training (Default: 128)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='ckpt', help='checkpoint dir of CGAN')
    parser.add_argument('--epoch_size', type=int,
                        default=50, help='epochs of CGAN for training')
    parser.add_argument('--data_path', type=str, default='data/MNIST_Data/train',
                        help='dataset dir (default data/MNISt_Data/train)')
    parser.add_argument("--real_valued_mnist", type=bool, default=True,
                        help='If load real valued MNIST dataset(default False)')
    # model art
    parser.add_argument("--data_url", type=str, default="./dataset", help='real input file path')
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset", help='modelart input path')
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result", help='modelart output path.')
    parser.add_argument("--obs_result_dir", type=str, default="./output", help='real output file path include .ckpt and .air')  # modelarts -> obs
    parser.add_argument("--modelarts_attrs", type=str, default="")
    args = parser.parse_args()

    # if not exists 'imgs4', 'gif' or 'ckpt_dir', make it
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    # deal with the distribute analyze problem
    if args.distribute:
        device_id = args.device_id
        context.set_context(save_graphs=False,
                            device_id=device_id,
                            device_target="Ascend",
                            mode=context.GRAPH_MODE)
        init()
        args.device_num = get_group_size()
        context.set_auto_parallel_context(gradients_mean=True,
                                          device_num=args.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        device_id = args.device_id
        args.device_num = 1
        context.set_context(save_graphs=False,
                            mode=context.GRAPH_MODE,
                            device_target="Ascend")
        context.set_context(device_id=device_id)

    print(os.system('env'))
    return args

def obs_data2modelarts(args):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args.data_url, args.modelarts_data_dir))
    mox.file.copy_parallel(src_url=args.data_url, dst_url=args.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(args.modelarts_data_dir)
    print("===>>>Files:", files)
    files = os.listdir(args.modelarts_data_dir + "/MNIST_Data/train")
    print("===>>>Train files:", files)
    files = os.listdir(args.modelarts_data_dir + "/MNIST_Data/test")
    print("===>>>Test files:", files)

    if not mox.file.exists(args.obs_result_dir):
        mox.file.make_dirs(args.obs_result_dir)
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args.obs_result_dir, args.modelarts_result_dir))
    mox.file.copy_parallel(src_url=args.obs_result_dir, dst_url=args.modelarts_result_dir)
    files = os.listdir(args.modelarts_result_dir)
    print("===>>>Files:", files)

def modelarts_result2obs(args):
    """
    Copy result data from modelarts to obs.
    """
    obs_result_dir = args.obs_result_dir
    if not mox.file.exists(obs_result_dir):
        print(f"obs_result_dir[{obs_result_dir}] not exist!")
        mox.file.make_dirs(obs_result_dir)
    mox.file.copy_parallel(src_url=os.path.join(args.modelarts_result_dir, 'ckpt'),
                           dst_url=os.path.join(obs_result_dir, 'ckpt'))
    print("===>>>Copy Event or Checkpoint from modelarts to obs:{}".format(obs_result_dir))
    mox.file.copy(src_url='CGAN.air',
                  dst_url=os.path.join(obs_result_dir, 'CGAN.air'))
    files = os.listdir(obs_result_dir)
    print("===>>>current Files:", files)

def export_AIR(args):
    """
    start modelarts export
    """
    # training argument
    input_dim = 100
    n_col = 20
    n_row = 10
    n_image = n_row * n_col

    # create G Cell
    netG = Generator(input_dim)

    latent_code_eval = Tensor(np.random.randn(n_image, input_dim).astype(np.float32), dtype=mstype.float32)
    label_eval = Tensor((np.arange(n_image) % 10).astype(np.int32), mstype.int32)

    param_G = load_checkpoint("ckpt/CGAN.ckpt")
    load_param_into_net(netG, param_G)
    netG.set_train(False)
    export(netG, latent_code_eval, label_eval, file_name="CGAN", file_format="AIR")
    print("CGAN exported")

def main():
    # before training, we should set some arguments
    args = preLauch()

    ## copy dataset from obs to modelarts
    obs_data2modelarts(args)
    args.train_path = args.modelarts_data_dir + "/MNIST_Data/train"

    # training argument
    lr = 0.0002
    classes_num = 10
    img_channels = 1
    img_size = 32
    input_dim = 100
    embed_size = 100
    start_epochs = 0
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    output_path = args.modelarts_result_dir

    dataset = create_dataset(args.train_path,
                             img_size=img_size,
                             batch_size=batch_size,
                             real_valued_mnist=args.real_valued_mnist,
                             num_parallel_workers=args.device_num)
    dataset_size = dataset.get_dataset_size()
    dataset = dataset.create_dict_iterator()
    # create G Cell & D Cell
    netG = Generator(input_dim, img_channels, classes_num, embed_size=embed_size)
    netD = Discriminator(img_channels, classes_num, embed_size=img_size * img_size)
    init_weights(netG)
    init_weights(netD)
    # create WithLossCell
    netG_with_loss = GenWithLossCell(netG, netD)
    netD_with_loss = DisWithLossCell(netG, netD)
    # create optimizer cell
    optimizerG = nn.Adam(netG.trainable_params(), lr)
    optimizerD = nn.Adam(netD.trainable_params(), lr)

    net_G_train = TrainOneStepG(netG_with_loss, optimizerG)
    net_D_train = TrainOneStepD(netD_with_loss, optimizerD)
    netG.set_train()
    netD.set_train()

    reporter = Reporter(output_path=output_path, stage='train', start_epochs=start_epochs, dataset_size=dataset_size,
                        batch_size=batch_size)
    reporter.info('==========start training===============')
    fixed_noise = Tensor(np.random.normal(size=(batch_size, input_dim)),
                         dtype=mstype.float32)
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
    save_checkpoint(netG, './ckpt/CGAN.ckpt')
    ## start export air
    export_AIR(args)
    ## copy result from modelarts to obs
    modelarts_result2obs(args)

if __name__ == '__main__':
    main()
