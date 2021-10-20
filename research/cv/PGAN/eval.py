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
"""eval PGAN"""
import os
import argparse
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore
from src.network_G import GNet4_4_Train, GNet4_4_last, GNetNext_Train, GNetNext_Last
from src.image_transform import Normalize, TransporeAndMul, Resize
from PIL import Image


def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore PGAN training')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--checkpoint_g', type=str, default='',
                        help='checkpoint of g net (default )')
    parser.add_argument('--img_out', type=str,
                        default='img_eval', help='the dir of output img')

    args = parser.parse_args()

    context.set_context(device_id=args.device_id,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend")
    # if not exists 'img_out', make it
    if not os.path.exists(args.img_out):
        os.mkdir(args.img_out)
    return args


def buildNoiseData(n_samples):
    """buildNoiseData

    Returns:
        output.
    """
    inputLatent = np.random.randn(n_samples, 512)
    inputLatent = mindspore.Tensor(inputLatent, mindspore.float32)
    return inputLatent


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


def resizeTensor(data, out_size_image):
    """resizeTensor

    Returns:
        output.
    """
    out_data_size = (data.shape[0], data.shape[
        1], out_size_image[0], out_size_image[1])
    outdata = []
    data = data.asnumpy()
    data = np.clip(data, a_min=-1, a_max=1)
    transformList = [Normalize((-1., -1., -1.), (2, 2, 2)), TransporeAndMul(), Resize(out_size_image)]
    for img in range(out_data_size[0]):
        processed = data[img]
        for transform in transformList:
            processed = transform(processed)
        processed = np.array(processed)
        outdata.append(processed)
    return outdata


def main():
    """main"""
    args = preLauch()
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
    param_dict_g = load_checkpoint(args.checkpoint_g)
    load_param_into_net(avg_gnet, param_dict_g)
    inputNoise = buildNoiseData(64)
    gen_imgs_eval = avg_gnet(inputNoise, 0.0)
    out_images = resizeTensor(gen_imgs_eval, (128, 128))
    to_image = image_compose(out_images)
    to_image.save(os.path.join(args.img_out, "result.jpg"))


if __name__ == '__main__':
    main()
