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

"""file for evaling"""
import argparse
import numpy as np
import onnxruntime as ort
import cv2
from mindspore.common import set_seed
from src.dataset.testdataset import create_testdataset
from src.util.util import calculate_psnr

parser = argparse.ArgumentParser(description="ESRGAN eval")
parser.add_argument("--test_LR_path", type=str, default='/home/root/ESRGAN/datasets/Set14/LRbicx4')
parser.add_argument("--test_GT_path", type=str, default='/home/root/ESRGAN/datasets/Set14/GTmod12')
parser.add_argument('--onnx_path', type=str, default="./", help='Location of exported onnx model.')
set_seed(114514)

def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    sess = ort.InferenceSession(checkpoint_path, providers=providers)
    name = sess.get_inputs()[0].name
    return sess, name

def img_convert(img_nps, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert image numpy arrays for 310."""
    result = []
    for img_np in img_nps:
        img_np = np.clip(img_np, *min_max)
        img_np = (img_np - min_max[0]) / (min_max[1] - min_max[0])
        img_np = img_np.transpose(1, 2, 0)
        if img_np.shape[2] == 1:  # gray image
            img_np = np.squeeze(img_np, axis=2)
        else:
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

if __name__ == '__main__':
    args = parser.parse_args()
    test_ds = create_testdataset(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator(output_numpy=True)
    psnr_list = []
    print("=======starting test=====")
    for test in test_data_loader:
        lr = test['LR']
        hr = test['HR']
        im_data_shape = lr.shape
        onnx_file = args.onnx_path + str(im_data_shape[2]) + '_' + str(im_data_shape[3]) + '.onnx'
        session, input_name = create_session(onnx_file, 'GPU')
        sr = session.run(None, {input_name: lr})[0]
        sr_img = img_convert(sr)
        gt_img = img_convert(hr)
        psnr = calculate_psnr(sr_img, gt_img)
        psnr_list.append(psnr)
    print("avg PSNR:", np.mean(psnr_list))
