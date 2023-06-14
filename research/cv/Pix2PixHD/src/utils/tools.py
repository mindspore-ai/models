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
# ===========================================================================

"""
    Tools for Pix2PixHD model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint, load_param_into_net, load_checkpoint
from src.utils.config import config


def save_losses(save_path, G_losses, D_losses, idx):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig(save_path + "/{}.png".format(idx))


def save_image(img, img_path, format_name=".jpg"):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
        format_name(str): save format
    """
    if isinstance(img, Tensor):
        img = decode_image(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

    img_pil = Image.fromarray(img)
    img_pil.save(img_path + format_name)


def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    mean = 0.5 * 255
    std = 0.5 * 255
    # （256，256，3）
    return (img.asnumpy()[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))


def save_network(net, save_path, epoch_num):
    save_checkpoint(net.netG, os.path.join(save_path, f"{epoch_num + 1}_net_G.ckpt"))
    save_checkpoint(net.netG, os.path.join(save_path, f"latest_net_G.ckpt"))
    save_checkpoint(net.netD, os.path.join(save_path, f"{epoch_num + 1}_net_D.ckpt"))
    save_checkpoint(net.netD, os.path.join(save_path, f"latest_net_D.ckpt"))
    if (config.instance_feat or config.label_feat) and not config.load_features:
        save_checkpoint(net.netE, os.path.join(save_path, f"{epoch_num + 1}_net_E.ckpt"))
        save_checkpoint(net.netE, os.path.join(save_path, f"latest_net_E.ckpt"))


def load_network(net, net_label, which_epoch, save_dir):
    save_filename = "{}_net_{}.ckpt".format(which_epoch, net_label)
    if not save_dir:
        save_dir = os.path.join(config.save_ckpt_dir, config.name)
    load_param_into_net(net, load_checkpoint(os.path.join(save_dir, save_filename)))


def get_lr(steps_per_epoch):
    """
    Linear learning-rate generator.
    Keep the same learning rate for the first <config.niter> niter
    and linearly decay the rate to zero over the next <config.niter_decay> niter_decay.
    """
    lrs = [config.lr] * steps_per_epoch * config.niter
    lr_epoch = config.lr
    lr_decay = config.lr / config.niter_decay
    epoch = 0
    while epoch < config.niter_decay:
        lr_epoch = lr_epoch - lr_decay
        lrs += [lr_epoch] * steps_per_epoch
        epoch += 1

    return Tensor(np.array(lrs).astype(np.float32))
