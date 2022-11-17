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
"""export checkpoint file into onnx models"""

import argparse as argp
import mindspore
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
import numpy as np
from src.mcnn import MCNN
from src.dataset import create_dataset
from src.data_loader_3channel import ImageDataLoader_3channel

parser = argp.ArgumentParser(description='MCNN ONNX Export ')
parser.add_argument("--device_id", type=int, default=4, help="Device id")
parser.add_argument("--ckpt_file", type=str, default="./best.ckpt", help="Checkpoint file path.")
parser.add_argument('--val_path',
                    default='../MCNN/data/original/shanghaitech/part_A_final/test_data/images',
                    help='Location of data.')
parser.add_argument('--val_gt_path',
                    default='../MCNN/data/original/shanghaitech/part_A_final/test_data/ground_truth_csv',
                    help='Location of data.')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    data_loader_val = ImageDataLoader_3channel(args.val_path, args.val_gt_path, shuffle=False, gt_downsample=True,
                                               pre_load=True)
    ds_val = create_dataset(data_loader_val, target="GPU", train=False)
    ds_val = ds_val.batch(1)

    network = MCNN()
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    for sample in ds_val.create_dict_iterator(output_numpy=True):
        im_data_shape = sample['data'].shape
        export_path = str(im_data_shape[2]) + '_' + str(im_data_shape[3])
        inputs = Tensor(np.ones(list(im_data_shape)), mindspore.float32)
        export(network, inputs, file_name=export_path, file_format="ONNX")
