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

"""preprocess"""
import os
import argparse
import numpy as np
from mindspore import context
from src.dataset.testdataset import create_testdataset

parser = argparse.ArgumentParser(description="SRGAN eval")
parser.add_argument("--test_LR_path", type=str, default='./Set14/LR')
parser.add_argument("--test_GT_path", type=str, default='./Set14/HR')
parser.add_argument("--result_path", type=str, default='./preprocess_path')
parser.add_argument("--device_id", type=int, default=1, help="device id, default: 0.")
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)

def padding(_img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = _img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {_img.shape}")
    if dh != 0 or dw != 0:
        _img = np.pad(_img, ((0, dh), (0, dw), (0, 0)), "constant")
    return _img
if __name__ == '__main__':
    test_ds = create_testdataset(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator(output_numpy=True)
    i = 0
    img_path = args.result_path
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for data in test_data_loader:
        file_name = "SRGAN_data"  + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        lr = data['LR']
        lr = lr[0]
        lr = lr.transpose(1, 2, 0)
        org_img = padding(lr, [200, 200])
        org_img = org_img.transpose(2, 0, 1)
        img = org_img.copy()
        img.tofile(file_path)
        i = i + 1
