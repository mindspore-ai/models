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
import argparse
import onnxruntime
import numpy as np
from src.dataset.testdataset import create_testdataset
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2ycbcr
def get_args():
    parser = argparse.ArgumentParser(description="SRGAN infer_onnx")
    parser.add_argument("--test_LR_path", type=str, default='./Set14/LR')
    parser.add_argument("--test_GT_path", type=str, default='./Set14/HR')
    parser.add_argument("--res_num", type=int, default=16)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    parser.add_argument('--device_target', type=str, default='GPU', choices=('Tensort', 'GPU', 'CPU'))
    parser.add_argument('--onnx_path', type=str, default='./SRGAN_model.onnx')
    args = parser.parse_args()
    return args


def create_session(onnx_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    name = session.get_inputs()[0].name
    return session, name
def to_image(x):
    x = np.clip(x, -1.0, 1.0)
    x = (x+ 1.0) / 2.0
    x = np.squeeze(x)
    x = x.transpose(1, 2, 0)
    return Image.fromarray((x * 255.0).astype(np.uint8))

def run_eval():
    args = get_args()
    test_ds = create_testdataset(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator()
    psnr_list = []
    session, name = create_session(args.onnx_path, args.device_target)
    print("=======starting infer_onnx=====")
    for data in test_data_loader:
        lr = data['LR']
        lr = lr.asnumpy().copy()
        lr = np.array(to_image(lr).resize((126, 126))).astype(np.float32).transpose(2, 0, 1)
        lr = np.expand_dims(lr, 0)
        lr = 2*(lr/255)-1
        gt = data['HR'].asnumpy().copy()
        gt = np.array(to_image(gt).resize((504, 504))).astype(np.float32)/255

        sr = session.run(None, {name: lr})[0]
        sr = np.clip(sr, -1.0, 1.0)
        sr = (sr + 1.0) / 2.0
        sr = np.squeeze(sr)
        sr = sr.transpose(1, 2, 0)
        sr_image = rgb2ycbcr(sr)[args.scale:-args.scale, args.scale:-args.scale, :1]
        gt_image = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
        psnr = peak_signal_noise_ratio(sr_image / 255.0, gt_image / 255.0, data_range=1.0)
        psnr_list.append(psnr)
    print("avg PSNR:", np.mean(psnr_list))
if __name__ == '__main__':
    run_eval()
