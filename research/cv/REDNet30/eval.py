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
"""eval rednet30."""
import argparse
import os
import time
import glob
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import REDNet30


def PSNR(img1, img2):
    """metrics"""
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr


def SSIM(img1, img2):
    """metrics"""
    ssim = structural_similarity(img1, img2, data_range=255, multichannel=True)
    return ssim


def get_metric(ori_path, res_path):
    """metrics"""
    files = glob.glob(os.path.join(ori_path, "*"))
    names = []
    for i in files:
        names.append(i.split("/")[-1])

    # PSNR
    print("PSNR...")
    res = 0
    for i in tqdm(names):
        ori = Image.open(os.path.join(ori_path, i))
        gen = Image.open(os.path.join(res_path, i))
        res += PSNR(np.array(ori), np.array(gen))
    psnr_res = res / len(names)

    # SSIM
    print("SSIM...")
    res = 0
    for i in tqdm(names):
        ori = Image.open(os.path.join(ori_path, i))
        gen = Image.open(os.path.join(res_path, i))
        res += SSIM(np.array(ori), np.array(gen))
    ssim_res = res / len(names)

    print("PSNR: ", psnr_res)
    print("SSIM: ", ssim_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/BSD200', help='evaling image path')
    parser.add_argument('--noise_path', type=str, default='./data/BSD200_jpeg_quality10', help='evaling image path')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt/RedNet30-1000_18.ckpt", help='ckpt path')
    parser.add_argument('--platform', type=str, default='GPU', choices=('Ascend', 'GPU'), help='run platform')
    opt = parser.parse_args()

    device_id = int(os.getenv('DEVICE_ID', '0'))

    if opt.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=device_id)

    save_path = "./output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = REDNet30()
    if opt.ckpt_path:
        param_dict = load_checkpoint(opt.ckpt_path)
        load_param_into_net(model, param_dict)
    model.set_train(False)

    # data
    img_files = glob.glob(opt.noise_path + "/*")

    time_start = time.time()
    for file in tqdm(img_files):
        name = file.split("/")[-1]
        img = np.array(Image.open(file))
        img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
        input_img = Tensor(img, dtype=mindspore.float32)
        result = model(input_img)
        out_img = result[0].asnumpy().transpose(1, 2, 0)

        out_img = np.clip(out_img, 0, 255)
        out_img = np.uint8(out_img)
        out_img = Image.fromarray(out_img)
        out_img.save(os.path.join(save_path, name), quality=95)
    print("finished!")

    time_end = time.time()
    print('--------------------')
    print('test time: %f' % (time_end - time_start))
    print('--------------------')

    get_metric(opt.dataset_path, save_path)
