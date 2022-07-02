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
"""
eval for exported onnx model
"""


import argparse
import onnxruntime as ort
from mindspore.common import set_seed
import numpy as np
from src.dataset import create_dataset
from src.data_loader_3channel import ImageDataLoader_3channel


parser = argparse.ArgumentParser(description='MindSpore MCNN Example')
parser.add_argument('--onnx_path', type=str, default="./", help='Location of exported onnx model.')
parser.add_argument('--val_path',
                    default='../MCNN/data/original/shanghaitech/part_A_final/test_data/images',
                    help='Location of data.')
parser.add_argument('--val_gt_path',
                    default='../MCNN/data/original/shanghaitech/part_A_final/test_data/ground_truth_csv',
                    help='Location of data.')
args = parser.parse_args()
set_seed(64678)


def create_session(onnx_checkpoint_path):
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(onnx_checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


if __name__ == "__main__":
    local_path = args.val_path
    local_gt_path = args.val_gt_path
    onnx_path = args.onnx_path

    data_loader_val = ImageDataLoader_3channel(local_path, local_gt_path, shuffle=False, gt_downsample=True,
                                               pre_load=True)
    ds_val = create_dataset(data_loader_val, target="GPU", train=False)
    ds_val = ds_val.batch(1)

    mae = 0.0
    mse = 0.0
    for sample in ds_val.create_dict_iterator(output_numpy=True):
        im_data = sample['data']
        gt_data = sample['gt_density']
        im_data_shape = im_data.shape
        onnx_file = onnx_path + str(im_data_shape[2]) + '_' + str(im_data_shape[3]) + '.onnx'
        sess, input_each = create_session(onnx_file)
        density_map = sess.run(None, {input_each: im_data})[0]
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
    mae = mae / ds_val.get_dataset_size()
    mse = np.sqrt(mse / ds_val.get_dataset_size())
    print('MAE:', mae, '  MSE:', mse)
