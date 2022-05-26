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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import ast
import os

import numpy as np

from mindspore import Tensor, nn, ops, context, load_checkpoint, export
import mindspore.common.dtype as mstype

from src.cell import WithLossCellD, WithLossCellG
from src.dcgan import DCGAN
from src.discriminator import Discriminator
from src.generator import Generator
from src.config import dcgan_imagenet_cfg as cfg


def load_dcgan(ckpt_url):
    """
        load dcgan from checkpoint file
    """
    netD = Discriminator()
    netG = Generator()

    criterion = nn.BCELoss(reduction='mean')

    netD_with_criterion = WithLossCellD(netD, netG, criterion)
    netG_with_criterion = WithLossCellG(netD, netG, criterion)

    optimizerD = nn.Adam(netD.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)
    optimizerG = nn.Adam(netG.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)

    myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD)
    myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG)

    net = DCGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
    load_checkpoint(ckpt_url, net)
    net.set_train(False)
    return net


def load_discriminator(dcgan_net):
    """
        load discriminator layers from dcgan model
    """
    netD_trained = dcgan_net.myTrainOneStepCellForD.network.netD
    for m in netD_trained.discriminator.cells_and_names():
        if m[0] == '0':
            print(m[0], m[1])
            conv_1 = m[1]
        elif m[0] == '1':
            print(m[0], m[1])
            leakyReLU_1 = m[1]
        elif m[0] == '2':
            print(m[0], m[1])
            conv_2 = m[1]
        elif m[0] == '3':
            print(m[0], m[1])
            bm_1 = m[1]
        elif m[0] == '4':
            print(m[0], m[1])
            leakyReLU_2 = m[1]
        elif m[0] == '5':
            print(m[0], m[1])
            conv_3 = m[1]
    return conv_1, leakyReLU_1, conv_2, bm_1, leakyReLU_2, conv_3


class DiscriminatorConvert(nn.Cell):
    """
    Discriminator_convert
    """

    def __init__(self, conv1, leakyReLU1, conv2, bm1, leakyReLU2, conv3):
        super(DiscriminatorConvert, self).__init__()
        self.conv1 = conv1
        self.leakyReLU1 = leakyReLU1
        self.conv2 = conv2
        self.bm1 = bm1
        self.leakyReLU2 = leakyReLU2
        self.conv3 = conv3
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = ops.Concat(1)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        output1 = self.maxpool1(x)
        x = self.conv2(self.leakyReLU1(x))
        output2 = self.maxpool2(x)
        x = self.conv3(self.leakyReLU2(self.bm1(x)))
        output3 = x
        result = self.concat((output1, output2, output3))
        result = self.reshape(result, (1, -1))
        return result


parser = argparse.ArgumentParser(description='dcgan export')
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--ckpt_url", default=None, help="Checkpoint file url.")
parser.add_argument("--ckpt_file", default=None, help="Checkpoint file name.")
parser.add_argument('--data_url', default=None, help='Directory contains dataset.')
parser.add_argument('--train_url', default=None, help='Directory contains checkpoint file')
parser.add_argument("--file_name", type=str, default="dcgan", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", help="file format")
parser.add_argument("--load_netG", type=str, default=False, help="export  netG, default is false.")
parser.add_argument("--load_netD", type=str, default=True, help="export  netD for infer, default is True.")
parser.add_argument("--load_G_and_D", type=str, default=False, help="export  netG and netD, default is false.")
args = parser.parse_args()

if args.run_modelart:
    local_ckpt_url = '/cache/train_outputs'
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        save_graphs=False)
    context.set_context(device_id=device_id)
    import moxing as mox
    mox.file.copy_parallel(src_url=args.ckpt_url, dst_url=local_ckpt_url)
    local_ckpt_url = local_ckpt_url + args.ckpt_file
else:
    local_ckpt_url = args.ckpt_file
    device_target = args.device_target
    device_id = args.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, save_graphs=False, device_id=device_id)


if __name__ == '__main__':
    if args.load_netD:
        dcgan = load_dcgan(local_ckpt_url)
        d_conv1, d_leakyReLU1, d_conv2, d_bm1, d_leakyReLU2, d_conv3 = load_discriminator(dcgan)
        discriminator_convert = DiscriminatorConvert(conv1=d_conv1, leakyReLU1=d_leakyReLU1, conv2=d_conv2, bm1=d_bm1,
                                                     leakyReLU2=d_leakyReLU2, conv3=d_conv3)
        discriminator_convert.set_train(False)

        inputs = Tensor(np.random.rand(args.batch_size, 3, 32, 32), mstype.float32)
        export(discriminator_convert, inputs, file_name=args.file_name, file_format=args.file_format)
    elif args.load_netG:
        dcgan = load_dcgan(local_ckpt_url)
        netG_trained = dcgan.myTrainOneStepCellForG.network.netG
        netG_trained.set_train(False)
        latent_code = Tensor(np.random.rand(args.batch_size, 100, 1, 1), mstype.float32)
        export(netG_trained, latent_code, file_name=args.file_name, file_format=args.file_format)
    else:
        dcgan = load_dcgan(local_ckpt_url)
        # inputs = Tensor(np.random.rand(args.batch_size, 3, 448, 448), mstype.float32)
        real_data = Tensor(np.random.rand(args.batch_size, 3, 32, 32), mstype.float32)
        latent_code = Tensor(np.random.rand(args.batch_size, 100, 1, 1), mstype.float32)
        inputs = [real_data, latent_code]
        export(dcgan, *inputs, file_name=args.file_name, file_format=args.file_format)
    if args.run_modelart:
        file_name = args.file_name + "." + args.file_format.lower()
        mox.file.copy_parallel(src_url=file_name,
                               dst_url=os.path.join(args.ckpt_url, file_name))
