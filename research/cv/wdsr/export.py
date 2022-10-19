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
"""export net together with checkpoint into air/mindir/onnx models"""
import os
import argparse
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export, dataset
import src.model as wdsr
from src.data.srdata import SRData
from src.data.div2k import DIV2K

parser = argparse.ArgumentParser(description='wdsr export')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_path", type=str, required=True, help="path of checkpoint file")
parser.add_argument("--file_name", type=str, default="wdsr", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
parser.add_argument("--device_target", type=str, default="Ascend", choices=['Ascend', 'GPU'], help="device target")
parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--dir_data', type=str, default='/cache/data/', help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K', help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K', help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810', help='train/test data range')
parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')
parser.add_argument('--model', default='WDSR', help='model name')
parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=48, help='output patch size')
parser.add_argument('--no_augment', action='store_true', help='do not use data augmentation')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

args1 = parser.parse_args()


def run_export(args):
    """run_export"""
    device_id = int(os.getenv("DEVICE_ID", '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)
    net = wdsr.WDSR(scale=args.scale[0], n_resblocks=args.n_resblocks, n_feats=args.n_feats)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    print("load mindspore net and checkpoint successfully.", flush=True)
    if args.data_test[0] == 'DIV2K':
        train_dataset = DIV2K(args, name=args.data_test, train=False, benchmark=False)
    else:
        train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    train_de_dataset = dataset.GeneratorDataset(train_dataset, ['LR', 'HR'], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)
    for _, imgs in enumerate(train_loader):
        lr = imgs['LR']
        inputs = Tensor(np.zeros([args.batch_size, 3, lr.shape[2], lr.shape[3]], np.float32))
        export(net, inputs, file_name=args.file_name + '_' + str(lr.shape[3]) + '_' + str(lr.shape[2]),
               file_format=args.file_format)
        print("export successfully!", flush=True)

if __name__ == "__main__":
    args1.scale = [int(x) for x in args1.scale.split("+")]
    args1.data_train = args1.data_train.split('+')
    args1.data_test = args1.data_test.split('+')
    run_export(args1)
