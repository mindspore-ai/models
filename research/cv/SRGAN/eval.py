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
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import context
import mindspore.ops as ops
import mindspore as ms
from src.model.generator import Generator
from src.dataset.testdataset import create_testdataset

set_seed(1)
parser = argparse.ArgumentParser(description="SRGAN eval")
parser.add_argument("--test_LR_path", type=str, default='./Set14/LR')
parser.add_argument("--test_GT_path", type=str, default='./Set14/HR')
parser.add_argument("--res_num", type=int, default=16)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--generator_path", type=str, default='./ckpt/srgan_ascend_v190_div2k_research_cv_PSNR27.07.ckpt')
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument('--platform', type=str, default='GPU', choices=('Ascend', 'GPU', 'CPU'))


def to_image(x):
    x = np.clip(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = np.squeeze(x)
    x = x.transpose(1, 2, 0)
    return Image.fromarray((x * 255.0).astype(np.uint8))
i = 0

if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)
    test_ds = create_testdataset(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator()
    generator = Generator(4)
    params = load_checkpoint(args.generator_path)
    load_param_into_net(generator, params)
    op = ops.ReduceSum(keep_dims=False)
    psnr_list = []
    i = 0
    print("=======starting test=====")
    for data in test_data_loader:
        lr = data['LR']
        gt = data['HR']
        lr = lr.asnumpy().copy()
        lr = np.array(to_image(lr).resize((126, 126))).astype(np.float32).transpose(2, 0, 1)
        lr = np.expand_dims(lr, 0)
        lr = ms.Tensor(2 * (lr / 255) - 1)
        gt = data['HR'].asnumpy().copy()
        gt = np.array(to_image(gt).resize((504, 504))).astype(np.float32).transpose(2, 0, 1)
        gt = np.expand_dims(gt, 0)
        gt = ms.Tensor(2 * (gt / 255) - 1)
        output = generator(lr)
        output = op(output, 0)
        output = output.asnumpy()
        output = np.clip(output, -1.0, 1.0)

        gt = op(gt, 0)

        output = (output + 1.0) / 2.0
        gt = (gt + 1.0) / 2.0

        output = output.transpose(1, 2, 0)
        gt = gt.asnumpy()
        gt = gt.transpose(1, 2, 0)

        result = Image.fromarray((output * 255.0).astype(np.uint8))
        y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
        y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

        psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
        psnr_list.append(psnr)
    print("avg PSNR:", np.mean(psnr_list))
