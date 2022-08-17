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

"""return train and eval dataset"""

import os
import os.path
from os.path import join
import random
import pyflow
import numpy as np
import mindspore.dataset.vision.py_transforms as P
from mindspore import dataset as de, context
from mindspore.context import ParallelMode
from PIL import Image, ImageOps



def is_image_file(filename):
    """Judge whether it is a picture."""
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, nFrames, scale, other_dataset):
    """"Load image"""
    seq = [i for i in range(1, nFrames)]
    # random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        x = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neighbor = []

        for i in seq:
            index = int(filepath[char_len - 7:char_len - 4]) - i
            file_name = filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png'

            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png').convert('RGB'),
                               scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neighbor.append(temp)
            else:
                print('neighbor frame is not exist')
                temp = x
                neighbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath, 'im' + str(nFrames) + '.png')).convert('RGB'), scale)
        x = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neighbor = [modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC) for j in reversed(seq)]

    return target, x, neighbor


def load_img_future(filepath, nFrames, scale, other_dataset):
    """"Load image using the past and future ways"""
    tt = int(nFrames / 2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        x = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neighbor = []
        if nFrames % 2 == 0:
            seq = [x for x in range(-tt, tt) if x != 0]  # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt, tt + 1) if x != 0]
        # random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len - 7:char_len - 4]) + i
            file_name1 = filepath[0:char_len - 7] + '{0:03d}'.format(index1) + '.png'

            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neighbor.append(temp)
            else:
                print('neighbor frame- is not exist')
                temp = x
                neighbor.append(temp)

    else:
        target = modcrop(Image.open(join(filepath, 'im4.png')).convert('RGB'), scale)
        x = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neighbor = []
        seq = [x for x in range(4 - tt, 5 + tt) if x != 4]
        # random.shuffle(seq) #if random sequence
        for j in seq:
            neighbor.append(modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
                (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC))
    return target, x, neighbor


def get_flow(im1, im2):
    """Get flow image"""
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, _ = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                                      nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # flow = rescale_flow(flow,0,1)
    return flow


def rescale_flow(x, max_range, min_range):
    """rescale flow"""
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range - min_range) / (max_val - min_val) * (x - max_val) + max_range


def modcrop(img, modulo):
    """Image crop"""
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    """Patch the image"""
    (ih, iw) = img_in.size
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch


def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    """data augment"""
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


class RBPNDataset():
    """
        This dataset class can load images from image folder.
        Args:
            image_dir (str): Images root directory.
            nFrames (int): neighbor frames.
            upscale_factior (int): Super resolution upscale factor.
            data_augmentation (bool) : Data augmentation.
            file_list (str): The images names.
            other_dataset (bool): Use other dataset.
            patch size (int) : The patch size
            future_frame (bool): Use the past and future ways to get image

        Returns:
            Image path list.
        """
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                 future_frame, transform=None):
        super(RBPNDataset, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame


    def __getitem__(self, index):
        if self.future_frame:
            target, x, neighbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                                  self.other_dataset)

        else:
            target, x, neighbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                           self.other_dataset)

        if self.patch_size != 0:
            x, target, neighbor = get_patch(x, target, neighbor, self.patch_size, self.upscale_factor,
                                            self.nFrames)

        if self.data_augmentation:
            x, target, neighbor = augment(x, target, neighbor)

        flow = [get_flow(x, j) for j in neighbor]
        flow = [j.transpose(2, 0, 1).astype(np.float32) for j in flow]

        neighbor_all = []

        for pic in neighbor:
            neighbor_temp = np.array(pic)
            neighbor_temp = neighbor_temp.transpose((2, 0, 1)).astype(np.float32)
            neighbor_temp = neighbor_temp/255.
            neighbor_all.append(neighbor_temp)

        bicubic = rescale_img(x, self.upscale_factor)
        return target, x, bicubic, neighbor_all, flow

    def __len__(self):
        return len(self.image_filenames)

def _get_rank_info():

    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def create_train_dataset(dataset, args):
    """return train dataset """
    device_num, rank_id = _get_rank_info()
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
        train_ds = de.GeneratorDataset(dataset, num_parallel_workers=args.threads,
                                       column_names=['target_image', 'input_image', 'bicubic_image', 'neighbor_image',
                                                     'flow_image'],
                                       shuffle=True, num_shards=device_num, shard_id=rank_id)
    else:
        train_ds = de.GeneratorDataset(dataset, num_parallel_workers=args.threads,
                                       column_names=['target_image', 'input_image', 'bicubic_image', 'neighbor_image',
                                                     'flow_image'],
                                       shuffle=True)

    trans = [
        P.ToTensor(),
    ]
    train_ds = train_ds.map(operations=trans, input_columns=['target_image'])
    train_ds = train_ds.map(operations=trans, input_columns=['input_image'])
    train_ds = train_ds.map(operations=trans, input_columns=['bicubic_image'])
    train_ds = train_ds.batch(args.batchSize, drop_remainder=True)
    return train_ds



class RBPNDatasetTest():
    """Define val dataset"""
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(RBPNDatasetTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, x, neighbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                                  self.other_dataset)
        else:
            target, x, neighbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                           self.other_dataset)

        flow = [get_flow(x, j) for j in neighbor]
        flow = [j.transpose(2, 0, 1).astype(np.float32) for j in flow]
        neighbor_all = []

        for pic in neighbor:
            neighbor_temp = np.array(pic)
            neighbor_temp = neighbor_temp.transpose((2, 0, 1)).astype(np.float32)
            neighbor_temp = neighbor_temp / 255.
            neighbor_all.append(neighbor_temp)

        bicubic = rescale_img(x, self.upscale_factor)

        return target, x, bicubic, neighbor_all, flow

    def __len__(self):
        return len(self.image_filenames)



def create_val_dataset(dataset, args):
    """Create val dataset."""
    val_ds = de.GeneratorDataset(dataset, column_names=['target_image', 'input_image', 'bicubic_image',
                                                        'neighbor_image', 'flow_image'],
                                 shuffle=False)
    trans = [P.ToTensor()]
    val_ds = val_ds.map(operations=trans, input_columns=["target_image"])
    val_ds = val_ds.map(operations=trans, input_columns=["input_image"])
    val_ds = val_ds.map(operations=trans, input_columns=["bicubic_image"])
    val_ds = val_ds.batch(args.testBatchSize, drop_remainder=True)
    return val_ds
