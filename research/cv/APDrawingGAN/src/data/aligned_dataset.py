# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""create aligned dataset"""

import os.path
import random
import csv
import cv2
import mindspore.dataset.vision as vision
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from PIL import Image
import numpy as np
from .base_dataset import BaseDataset
from .image_folder import make_dataset


def getfeats(featpath):
    """getfeats"""
    trans_points = np.empty([5, 2], dtype=np.int64)
    with open(featpath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for ind, row in enumerate(reader):
            trans_points[ind, :] = row
    return trans_points

def tocv2(A, B, which_direction):
    """tocv2"""
    if which_direction == 'AtoB':
        ts = B
    else:
        ts = A
    img = (ts / 2 + 0.5) * 255
    img = img.astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, ::-1]  #rgb->bgr
    return img

def dt(img):
    """dt"""
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #convert to BW
    _, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dt1 = cv2.distanceTransform(thresh1, cv2.DIST_L2, 5)
    dt2 = cv2.distanceTransform(thresh2, cv2.DIST_L2, 5)
    dt1 = dt1 / dt1.max()  #->[0,1]
    dt2 = dt2 / dt2.max()
    return dt1, dt2


def get_soft(size, xb, yb, boundwidth=5.0):
    """get_soft"""
    xarray = np.tile(np.arange(0, size[1]), (size[0], 1))
    yarray = np.tile(np.arange(0, size[0]), (size[1], 1)).transpose()
    cxdists = []
    cydists = []
    for i in range(len(xb)):
        xba = np.tile(xb[i], (size[1], 1)).transpose()
        yba = np.tile(yb[i], (size[0], 1))
        cxdists.append(np.abs(xarray - xba))
        cydists.append(np.abs(yarray - yba))
    xdist = np.minimum.reduce(cxdists)
    ydist = np.minimum.reduce(cydists)
    manhdist = np.minimum.reduce([xdist, ydist])
    im = (manhdist + 1) / (boundwidth + 1) * 1.0
    im[im >= 1.0] = 1.0
    return im


def tensor2Int(x):
    """tensor2Int"""
    s = x.astype(mstype.int64)
    return int('%s' % s)

def get_nc(opt):
    """get_nc"""
    if opt.which_direction == 'BtoA':
        return opt.output_nc, opt.input_nc
    return opt.input_nc, opt.output_nc

def soft_border_process(opt, mask, center, rws, rhs):
    """soft_border_process"""
    imgsize = opt.fineSize
    maskn = mask[0].numpy()
    masks = [
        np.ones([imgsize, imgsize]),
        np.ones([imgsize, imgsize]),
        np.ones([imgsize, imgsize]),
        np.ones([imgsize, imgsize])
    ]
    masks[0][1:] = maskn[:-1]
    masks[1][:-1] = maskn[1:]
    masks[2][:, 1:] = maskn[:, :-1]
    masks[3][:, :-1] = maskn[:, 1:]
    masks2 = [maskn - e for e in masks]
    bound = np.minimum.reduce(masks2)
    bound = -bound
    xb = []
    yb = []
    for i in range(4):
        xbi = [
            int(center[i, 0] - rws[i] / 2),
            int(center[i, 0] + rws[i] / 2 - 1)
        ]
        ybi = [
            int(center[i, 1] - rhs[i] / 2),
            int(center[i, 1] + rhs[i] / 2 - 1)
        ]
        for j in range(2):
            maskx = bound[:, xbi[j]]
            masky = bound[ybi[j], :]
            xb += [(1 - maskx) * 10000 + maskx * xbi[j]]
            yb += [(1 - masky) * 10000 + masky * ybi[j]]
    soft = 1 - get_soft([imgsize, imgsize], xb, yb)
    soft = self.unsqueeze(Tensor(soft), 0)
    return (np.ones(mask.shape) - mask) * soft + mask

def init_AB(opt, AB_path):
    """init_AB"""
    AB = Image.open(AB_path).convert('RGB')
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h)).resize(
        (opt.loadSize, opt.loadSize), Image.BICUBIC)
    B = AB.crop((w2, 0, w, h)).resize(
        (opt.loadSize, opt.loadSize), Image.BICUBIC)
    A = vision.ToTensor()(A)
    B = vision.ToTensor()(B)
    w_offset = random.randint(
        0, max(0, opt.loadSize - opt.fineSize - 1))
    h_offset = random.randint(
        0, max(0, opt.loadSize - opt.fineSize - 1))

    A = A[:, h_offset:h_offset + opt.fineSize,
          w_offset:w_offset + opt.fineSize]  #C,H,W
    B = B[:, h_offset:h_offset + opt.fineSize,
          w_offset:w_offset + opt.fineSize]

    A = vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)(A)
    B = vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)(B)
    return A, B

def regions_process(opt, regions, feats, item, A, B, input_nc, output_nc):
    """regions_process"""
    mouth_x = int((feats[3, 0] + feats[4, 0]) / 2.0)
    mouth_y = int((feats[3, 1] + feats[4, 1]) / 2.0)
    ratio = opt.fineSize / 256
    EYE_H = opt.EYE_H * ratio
    EYE_W = opt.EYE_W * ratio
    NOSE_H = opt.NOSE_H * ratio
    NOSE_W = opt.NOSE_W * ratio
    MOUTH_H = opt.MOUTH_H * ratio
    MOUTH_W = opt.MOUTH_W * ratio
    center = np.array(
        [[feats[0, 0], feats[0, 1] - 4 * ratio],
         [feats[1, 0], feats[1, 1] - 4 * ratio],
         [feats[2, 0], feats[2, 1] - NOSE_H / 2 + 16 * ratio],
         [mouth_x, mouth_y]],
        dtype=np.float32)
    item['center'] = center
    rhs = [EYE_H, EYE_H, NOSE_H, MOUTH_H]
    rws = [EYE_W, EYE_W, NOSE_W, MOUTH_W]
    if opt.soft_border:
        soft_border_mask4 = []
        for i in range(4):
            xb = [np.zeros(rhs[i]), np.ones(rhs[i]) * (rws[i] - 1)]
            yb = [np.zeros(rws[i]), np.ones(rws[i]) * (rhs[i] - 1)]
            soft_border_mask = get_soft([rhs[i], rws[i]], xb, yb)
            soft_border_mask4.append(
                self.unsqueeze(Tensor(soft_border_mask), 0))
            item['soft_' + regions[i] + '_mask'] = soft_border_mask4[i]
    for i in range(4):
        item[regions[i] +
             '_A'] = A[:,
                       int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                       int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
        item[regions[i] +
             '_B'] = B[:,
                       int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                       int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
        if opt.soft_border:
            item[regions[i] +
                 '_A'] = item[regions[i] + '_A'] * soft_border_mask4[i].repeat(input_nc / output_nc, 1, 1)
            item[regions[i] +
                 '_B'] = item[regions[i] + '_B'] * soft_border_mask4[i]

    mask = np.ones(B.shape)  # mask out eyes, nose, mouth
    for i in range(4):
        mask[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
             int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)] = 0
    if opt.soft_border:
        mask = soft_border_process(opt, mask, center, rws, rhs)
    return mask


class AlignedDataset(BaseDataset):
    """AlignedDataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super(AlignedDataset, self).__init__()
        self.opt = opt
        self.unsqueeze = ops.ExpandDims()
        self.index_select = ops.Gather()
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert opt.resize_or_crop == 'resize_and_crop'

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        A, B = init_AB(self.opt, AB_path)
        input_nc, output_nc = get_nc(self.opt)

        flipped = False
        if (not self.opt.no_flip) and random.random() < 0.5:
            flipped = True
            idx = [i for i in range(A.shape[2] - 1, -1, -1)]
            A = np.take(A, idx, axis=2)
            B = np.take(B, idx, axis=2)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = np.expand_dims(tmp, axis=0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = np.expand_dims(tmp, axis=0)

        item = {'A': A, 'B': B}
        item['A_paths'] = item.get('A_paths', [])
        item['A_paths'].append(AB_path)
        item['B_paths'] = item.get('B_paths', [])
        item['B_paths'].append(AB_path)

        if self.opt.use_local:
            regions = ['eyel', 'eyer', 'nose', 'mouth']
            basen = os.path.basename(AB_path)[:-4] + '.txt'
            featdir = self.opt.lm_dir
            featpath = os.path.join(featdir, basen)
            feats = getfeats(featpath)
            if flipped:
                for i in range(5):
                    feats[i, 0] = self.opt.fineSize - feats[i, 0] - 1
                tmp = [feats[0, 0], feats[0, 1]]
                feats[0, :] = [feats[1, 0], feats[1, 1]]
                feats[1, :] = tmp


            mask = regions_process(self.opt, regions, feats, item, A, B, input_nc, output_nc)

            bgdir = self.opt.bg_dir
            bgpath = os.path.join(bgdir, basen[:-4] + '.png')
            im_bg = Image.open(bgpath)
            mask2 = vision.ToTensor()(im_bg)  # mask out background

            if flipped:
                mask2 = np.take(mask2, idx, axis=2)
            mask2 = np.array([mask2 >= 0.5], dtype=np.float32)[0]

            hair_A = (A / 2 + 0.5) * mask.repeat(
                input_nc // output_nc, axis=0) * mask2.repeat(
                    input_nc // output_nc, axis=0) * 2 - 1
            hair_B = (B / 2 + 0.5) * mask * mask2 * 2 - 1
            bg_A = (A / 2 + 0.5) * (np.ones(mask2.shape) - mask2).repeat(
                input_nc // output_nc, 0) * 2 - 1
            bg_B = (B / 2 + 0.5) * (np.ones(mask2.shape) - mask2) * 2 - 1
            item['hair_A'] = hair_A.astype(np.float32)
            item['hair_B'] = hair_B.astype(np.float32)
            item['bg_A'] = bg_A.astype(np.float32)
            item['bg_B'] = bg_B.astype(np.float32)
            item['mask'] = mask.astype(np.float32)
            item['mask2'] = mask2

        if self.opt.isTrain:
            img = tocv2(A, B, self.opt.which_direction)
            dt1, dt2 = dt(img)
            dt1 = np.expand_dims(dt1, axis=0)
            dt2 = np.expand_dims(dt2, axis=0)
            item['dt1gt'] = dt1
            item['dt2gt'] = dt2

        return item['A'], item['B'], item['center'], item['eyel_A'], item['eyel_B'], item['eyer_A'], \
               item['eyer_B'], item['nose_A'], item['nose_B'], item['mouth_A'], item['mouth_B'], item['hair_A'], \
               item['hair_B'], item['bg_A'], item['bg_B'], item['mask'], item['mask2'], item['dt1gt'], item['dt2gt']

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
