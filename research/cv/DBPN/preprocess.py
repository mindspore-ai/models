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
"""preprocess"""
import os
import argparse
import ast
import numpy as np
from src.dataset.dataset import DatasetVal, create_val_dataset

parser = argparse.ArgumentParser(description="Preporcess")
parser.add_argument("--val_LR_path", type=str, default=r"Set5/LR")
parser.add_argument("--val_GT_path", type=str, default=r"Set5/HR")
parser.add_argument("--upscale_factor", type=int, default="4", help="scale.")
parser.add_argument('--vgg', type=ast.literal_eval, default=True, help="use vgg")
parser.add_argument('--isgan', type=ast.literal_eval, default=False,
                    help="the param of is_gan decides the way of training ")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
args = parser.parse_args()

def padding(_img, target_shape):
    """padding image for 310."""
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = _img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {_img.shape}")
    if dh != 0 or dw != 0:
        _img = np.pad(_img, ((0, dh), (0, dw), (0, 0)), "constant")
    return _img


if __name__ == '__main__':
    result_path = "./preprocess_Result/"
    dataset = DatasetVal(args.val_GT_path, args.val_LR_path, args)
    ds = create_val_dataset(dataset, args)
    img_path = os.path.join(result_path, "00_data")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        file_name = "DBPN_data_bs" + str(args.testBatchSize) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        lr = data['input_image']
        lr = np.squeeze(lr, axis=0)
        lr = lr.transpose(1, 2, 0)
        org_img = padding(lr, [200, 200])
        org_img = org_img.transpose(2, 0, 1)
        img = org_img.copy()
        img.tofile(file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
