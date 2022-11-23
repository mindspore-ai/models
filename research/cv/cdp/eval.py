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
import os
import sys
import time
import glob
from collections import namedtuple
import logging
import argparse
import utils
import genotypes
from model import NetworkCIFAR as Network
from mindspore import dtype as mstype
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.dataset import transforms
import mindspore.nn as nn
from mindvision.classification.dataset import Cifar10
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='CDP_torch2ms.ckpt', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='CIFAR', help='experiment name')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--arch', type=str, default='CDP_cifar', help='which architecture to use')
parser.add_argument('--retrain', type=str, default=None)
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

class SoftmaxCrossEntropyLoss(nn.Cell):
    """
    Define the loss use auxiliary
    """
    def __init__(self, auxiliary, auxiliary_weight):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='mean')

    def construct(self, data_s, labels):
        if self.auxiliary and self.training:
            logits, logits_aux = data_s
            loss = self.criterion(logits, labels)
            loss_aux = self.criterion(logits_aux, labels)
            loss += self.auxiliary_weight * loss_aux
        else:
            logits = data_s
            loss = self.criterion(logits, labels)
        return loss



if __name__ == "__main__":
    CIFAR_CLASSES = 10
    logging.info(genotypes.Genotype)
    if args.arch == 'PDARTS':
        genotype = genotypes.PDARTS
    logging.info('---------Genotype---------')
    genotype = Genotype(
        normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3),
                ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5],
        reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
                ('skip_connect', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
    logging.info(genotype)
    logging.info('--------------------------')

    network = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    network.training = False
    network.drop_path_prob = args.drop_path_prob * 600 / args.epochs

    tc_dict = load_checkpoint(args.model_path)
    load_param_into_net(network, tc_dict, strict_load=True)

    _, transform = utils.data_transforms_cifar10(args)
    type_cast_op = transforms.TypeCast(mstype.int32)
    data = Cifar10(path=args.data, split='test', batch_size=100, resize=32, download=True)
    datasets = data.dataset
    datasets = datasets.map(transform, 'image')
    datasets = datasets.map(operations=type_cast_op, input_columns='label')
    datasets = datasets.batch(batch_size=100, drop_remainder=True)
    net_loss = SoftmaxCrossEntropyLoss(args.auxiliary, args.auxiliary_weight)
    model = Model(network, net_loss, metrics={'loss', 'top_1_accuracy', 'top_5_accuracy'}, amp_level='O0')
    val_result = model.eval(datasets, dataset_sink_mode=False)
    logging.info("=========================val metrics:=========================")
    logging.info('Top_1_accuracy: %s, Top_5_accuracy: %s',
                 val_result['top_1_accuracy'], val_result['top_5_accuracy'])
