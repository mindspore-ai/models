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
"eval"
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
import numpy as np
from PIL import Image

from src.model import create_G
from src.args import get_args

def save_image(img, img_path, IMAGE_SIZE):
    """save image"""
    mul = ops.Mul()
    add = ops.Add()
    if isinstance(img, Tensor):
        img = mul(img, 255 * 0.5)
        img = add(img, 255 * 0.5)

        img = img.asnumpy().astype(np.uint8).transpose((0, 2, 3, 1))

    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

    IMAGE_ROW = 8  # Row num
    IMAGE_COLUMN = 8  # Column num
    PADDING = 2  # Interval of small pictures
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE + PADDING * (IMAGE_COLUMN + 1),
                                 IMAGE_ROW * IMAGE_SIZE + PADDING * (IMAGE_ROW + 1)))  # create a new picture
    # cycle
    ii = 0
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.fromarray(img[ii])
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE + PADDING * x, (y - 1) * IMAGE_SIZE + PADDING * y))
            ii = ii + 1

    to_image.save(img_path)  # save

if __name__ == "__main__":

    args_opt = get_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)

    netG = create_G(args_opt.model_type, args_opt.imageSize, args_opt.nz, args_opt.nc, args_opt.ngf)

    # load weights
    load_param_into_net(netG, load_checkpoint(args_opt.ckpt_file))

    fixed_noise = Tensor(np.random.normal(0, 1, size=[args_opt.batchSize, args_opt.nz, 1, 1]), dtype=ms.float32)

    fake = netG(fixed_noise)
    save_image(fake, '{0}/generated_samples.png'.format(args_opt.output_dir), args_opt.imageSize)

    print("Generate images success!")
