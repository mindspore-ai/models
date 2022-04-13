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
"""eval PGAN"""
import argparse
import os
import random

import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.image_transform import Crop
from src.image_transform import Normalize, TransporeAndMul, Resize
from src.metric import msssim
from src.network_G import GNet4_4_Train, GNet4_4_last, GNetNext_Train, GNetNext_Last


def set_every(num):
    """set random seed"""
    random.seed(num)
    set_seed(num)
    np.random.seed(num)


set_every(1)


def pre_launch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore PGAN training')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='Target device (Ascend or GPU, default Ascend)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--checkpoint_g', type=str, default='',
                        help='checkpoint of g net (default )')
    parser.add_argument('--img_out_dir', type=str,
                        default='img_eval', help='the dir of output img')
    parser.add_argument('--measure_ms_ssim', type=bool,
                        default=False, help='measure ms-ssim metric flag')
    parser.add_argument('--original_img_dir', type=str,
                        default='', help='the dir of real img')

    args = parser.parse_args()

    context.set_context(device_id=args.device_id,
                        mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    # if not exists 'img_out', make it
    if not os.path.exists(args.img_out_dir):
        os.mkdir(args.img_out_dir)
    return args


def build_noise_data(n_samples):
    """build_noise_data

    Returns:
        output.
    """
    input_latent = np.random.randn(n_samples, 512)
    input_latent = Tensor(input_latent, mstype.float32)
    return input_latent


def image_compose(out_images, size=(8, 8)):
    """image_compose

    Returns:
        output.
    """
    to_image = Image.new('RGB', (size[0] * 128, size[1] * 128))
    for y in range(size[0]):
        for x in range(size[1]):
            from_image = Image.fromarray(out_images[y * size[0] + x])
            to_image.paste(from_image, (x * 128, y * 128))
    return to_image


def to_img_list(out_images):
    """to_img_list

    Returns:
        output.
    """
    img_list = []
    for img in out_images:
        img_list.append(Image.fromarray(img))
    return img_list


def resize_tensor(data, out_size_image):
    """resize_tensor

    Returns:
        output.
    """
    out_data_size = (data.shape[0], data.shape[1], out_size_image[0], out_size_image[1])
    outdata = []
    data = data.asnumpy()
    data = np.clip(data, a_min=-1, a_max=1)
    transform_list = [Normalize((-1., -1., -1.), (2, 2, 2)),
                      TransporeAndMul(), Resize(out_size_image)]
    for img in range(out_data_size[0]):
        processed = data[img]
        for transform in transform_list:
            processed = transform(processed)
        processed = np.array(processed)
        outdata.append(processed)
    return outdata


def construct_gnet():
    """construct_gnet"""
    scales = [4, 8, 16, 32, 64, 128]
    depth = [512, 512, 512, 512, 256, 128]
    for scale_index, scale in enumerate(scales):
        if scale == 4:
            avg_gnet = GNet4_4_Train(512, depth[scale_index], leakyReluLeak=0.2, dimOutput=3)
        elif scale == 8:
            last_avg_gnet = GNet4_4_last(avg_gnet)
            avg_gnet = GNetNext_Train(depth[scale_index], last_Gnet=last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
        else:
            last_avg_gnet = GNetNext_Last(avg_gnet)
            avg_gnet = GNetNext_Train(depth[scale_index], last_avg_gnet, leakyReluLeak=0.2, dimOutput=3)
    return avg_gnet


def load_original_images(original_img_dir, img_number):
    """load_original_images"""
    file_names = [f for f in os.listdir(original_img_dir)
                  if os.path.isfile(os.path.join(original_img_dir, f)) and '.jpg' in f]
    file_names = random.sample(file_names, img_number)
    crop = Crop()
    img_list = []
    for im_name in file_names:
        img = Image.open(os.path.join(original_img_dir, im_name))
        img = np.array(crop(img))
        img_list.append(img)
    return img_list


def main():
    """main"""
    print("Creating evaluation image...")
    args = pre_launch()
    avg_gnet = construct_gnet()
    param_dict_g = load_checkpoint(args.checkpoint_g)
    load_param_into_net(avg_gnet, param_dict_g)
    input_noise = build_noise_data(64)
    gen_imgs_eval = avg_gnet(input_noise, 0.0)
    out_images = resize_tensor(gen_imgs_eval, (128, 128))
    to_image = image_compose(out_images)
    to_image.save(os.path.join(args.img_out_dir, "result.jpg"))

    if args.measure_ms_ssim:
        print("Preparing images for metric calculation...")
        n_eval_batch = 200

        real_img_list = load_original_images(args.original_img_dir, n_eval_batch * 64 * 2)
        real_img_list = np.stack(real_img_list)

        fake_img_list = []
        for _ in range(n_eval_batch):
            input_noise = build_noise_data(64)
            gen_imgs_eval = avg_gnet(input_noise, 0.0)
            out_images = resize_tensor(gen_imgs_eval, (128, 128))
            fake_img_list += to_img_list(out_images)

        fake_img_list = np.stack(fake_img_list)

        print("Calculating metrics...")
        mssim_real = msssim(real_img_list[::2], real_img_list[1::2])
        mssim_fake = msssim(fake_img_list, real_img_list[1::2])
        print(f"Structure similarity for reals with reals: {mssim_real}")
        print(f"Structure similarity for reals with fakes: {mssim_fake}")


if __name__ == '__main__':
    main()
