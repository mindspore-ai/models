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
"""export net together with checkpoint into air/mindir/onnx models"""
import os
import argparse
import numpy as np
from src.args import args as args_1
from src.data.srdata import SRData
from src.data.div2k import DIV2K
from src.rcan_model import RCAN
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor, context, load_checkpoint, export


parser = argparse.ArgumentParser(description='rcan export')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_path", type=str, required=True, help="path of checkpoint file")
parser.add_argument("--file_name", type=str, default="rcan", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
parser.add_argument('--scale', type=int, default='2', help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--data_range', type=str, default='1-800/801-810', help='train/test data range')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--model', default='RCAN', help='model name')
parser.add_argument('--dir_data', type=str, default='', help='dataset directory')
parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')

args = parser.parse_args()

MAX_HR_SIZE = 2040

def run_export():
    """ export """
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id)
    net = RCAN(args)
    max_lr_size = MAX_HR_SIZE // args.scale  #  max_lr_size = MAX_HR_SIZE / scale
    param_dict = load_checkpoint(args.ckpt_path)
    net.load_pre_trained_param_dict(param_dict, strict=False)
    net.set_train(False)
    print('load mindspore net and checkpoint successfully.')

    if args.file_format == 'ONNX':
        if args_1.data_test[0] == 'DIV2K':
            train_dataset = DIV2K(args_1, name=args_1.data_test, train=False, benchmark=False)
        else:
            train_dataset = SRData(args_1, name=args_1.data_test, train=False, benchmark=False)
        train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR'], shuffle=False)
        train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
        train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)

        for _, imgs in enumerate(train_loader):
            img_shape = imgs['LR'].shape
            export_path = str(img_shape[2]) + '_' + str(img_shape[3])
            inputs = Tensor(np.ones([args.batch_size, 3, img_shape[2], img_shape[3]]), ms.float32)
            export(net, inputs, file_name=export_path, file_format=args.file_format)
    else:
        inputs = Tensor(np.ones([args.batch_size, 3, 678, max_lr_size]), ms.float32)
        export(net, inputs, file_name=args.file_name, file_format=args.file_format)
    print('export successfully!')


if __name__ == "__main__":
    run_export()
