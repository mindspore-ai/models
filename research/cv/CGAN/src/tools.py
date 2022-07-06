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
""" tools """
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype


def save_image(img, img_path, batch_size):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        img_path (str): the path of the image.

    Returns:
        img_path: output path
    """
    if isinstance(img, Tensor):
        img = img.asnumpy()
        img = decode_image(img, batch_size)
    elif isinstance(img, np.ndarray):
        img = decode_image(img, batch_size)
    else:
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))
    img_pil = Image.fromarray(img)
    img_pil.save(img_path)

    return img_path

def decode_image(img, batch_size):
    """
    Decode a [B, C, H, W] Tensor to image numpy array.

    Args:
        img(Tensor or Numpy): img
        args(Namespace): parameters

    Returns:
        img: concatenated img
    """
    size = 0
    if batch_size >= 200:
        size = 200
        l = 20
    elif batch_size > 50:
        size = 50
        l = 5
    img = img[:size]
    img = (img * 255).astype(np.uint8)
    img = np.concatenate(list(img), axis=1)#[C, H*B, W]
    img = np.split(img, l, axis=1)   # list of [C, H*B/5, W]
    img = np.concatenate(img, axis=2)  # [C, H*5, W*5]
    img = img.transpose((1, 2, 0))

    if img.shape[-1] == 1:
        img = np.concatenate((img, img, img), axis=2)
    return img


def get_lr(args):
    """
    Learning rate generator.
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    """

    if args.lr_policy == 'exponential':
        lr_epoch = args.learning_rate
        lrs = [lr_epoch] * args.dataset_size
        for _ in range(args.n_epochs_decay - 1):
            lr_epoch = lr_epoch / args.decay_factor
            lrs += [lr_epoch] * args.dataset_size
        if args.epoch_size > args.n_epochs_decay:
            lrs += [lr_epoch] * args.steps_per_epoch * (args.epoch_size - args.n_epochs_decay)
        lrs = lrs[args.start_epochs * args.dataset_size:]
        lrs = Tensor(np.array(lrs).astype(np.float32))
    else:
        lrs = Tensor(args.learning_rate).astype(mstype.float32)
    return lrs

def load_ckpt(ckpt_path, net):
    """Load parameter from checkpoint."""
    if ckpt_path is not None:
        param_G = load_checkpoint(ckpt_path)
        load_param_into_net(net, param_G)
        print("load module from ", ckpt_path)
    else:
        print("No G_ckpt was imported")
