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
"""single dataloader"""

import os.path
from PIL import Image
import numpy as np
from src.data.base_dataloader import getSoft, getfeats

def soft_border_process(opt, mask, center, rws, rhs):
    """soft_border_process"""
    imgsize = opt.fineSize
    maskn = mask[0].asnumpy()
    masks = [np.ones([imgsize, imgsize]), np.ones([imgsize, imgsize]), np.ones([imgsize, imgsize]),
             np.ones([imgsize, imgsize])]
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
        xbi = [center[i, 0] - rws[i] / 2, center[i, 0] + rws[i] / 2 - 1]
        ybi = [center[i, 1] - rhs[i] / 2, center[i, 1] + rhs[i] / 2 - 1]
        for j in range(2):
            maskx = bound[:, xbi[j]]
            masky = bound[ybi[j], :]
            xb += [(1 - maskx) * 10000 + maskx * xbi[j]]
            yb += [(1 - masky) * 10000 + masky * ybi[j]]
    soft = 1 - getSoft([imgsize, imgsize], xb, yb)
    soft = np.expand_dims(soft, axis=0)
    return (np.ones_like(mask) - mask) * soft + mask


def single_process(data_A, data_A_path, data_length, item, opt):
    """single_process"""
    if data_length == 1:
        A = data_A
        A_path = str(data_A_path)
    else:
        A = data_A[data_index, ...]
        A_path = str(data_A_path[data_index, ...])

    if opt.which_direction == 'BtoA':
        input_nc = opt.output_nc
        output_nc = opt.input_nc
    else:
        input_nc = opt.input_nc
        output_nc = opt.output_nc

    if input_nc == 1:  # RGB to gray
        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        A = np.expand_dims(tmp, axis=0)

    item['A'] = item.get('A', [])
    item['A'].append(A)
    item['A_path'] = item.get('A_path', [])
    item['A_path'].append(A_path)

    if opt.use_local:
        regions = ['eyel', 'eyer', 'nose', 'mouth']
        basen = os.path.basename(A_path)[:-5] + '.txt'
        featdir = opt.lm_dir
        featpath = os.path.join(featdir, basen)
        feats = getfeats(featpath)
        mouth_x = int((feats[3, 0] + feats[4, 0]) / 2.0)
        mouth_y = int((feats[3, 1] + feats[4, 1]) / 2.0)
        ratio = opt.fineSize / 256
        EYE_H = opt.EYE_H * ratio
        EYE_W = opt.EYE_W * ratio
        NOSE_H = opt.NOSE_H * ratio
        NOSE_W = opt.NOSE_W * ratio
        MOUTH_H = opt.MOUTH_H * ratio
        MOUTH_W = opt.MOUTH_W * ratio
        center = np.array([[feats[0, 0], feats[0, 1] - 4 * ratio], [feats[1, 0], feats[1, 1] - 4 * ratio],
                           [feats[2, 0], feats[2, 1] - NOSE_H / 2 + 16 * ratio], [mouth_x, mouth_y]])
        item['center'] = item.get('center', [])
        item['center'].append(center)
        rhs = [EYE_H, EYE_H, NOSE_H, MOUTH_H]
        rws = [EYE_W, EYE_W, NOSE_W, MOUTH_W]
        if opt.soft_border:
            soft_border_mask4 = []
            for i in range(4):
                xb = [np.zeros(rhs[i]), np.ones(rhs[i]) * (rws[i] - 1)]
                yb = [np.zeros(rws[i]), np.ones(rws[i]) * (rhs[i] - 1)]
                soft_border_mask = getSoft([rhs[i], rws[i]], xb, yb)
                soft_border_mask4.append(np.expand_dims(soft_border_mask, axis=0))
                item['soft_' + regions[i] + '_mask'] = item.get('soft_' + regions[i] + '_mask', [])
                item['soft_' + regions[i] + '_mask'].append(soft_border_mask4[i])
        regions_dict = {}
        for i in range(4):
            regions_dict[regions[i] + '_A'] = A[:,
                                                int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                                                int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
            if opt.soft_border:
                regions_dict[regions[i] + '_A'] = regions_dict[regions[i] + '_A'] * \
                                                np.tile(soft_border_mask4[i], (input_nc / output_nc, 1, 1))
            item[regions[i] + '_A'] = item.get(regions[i] + '_A', [])
            item[regions[i] + '_A'].append(regions_dict[regions[i] + '_A'])

        mask = np.ones((output_nc, A.shape[1], A.shape[2]), np.float32)  # mask out eyes, nose, mouth
        for i in range(4):
            mask[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                 int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)] = 0
        if opt.soft_border:
            mask = soft_border_process(opt, mask, center, rws, rhs)

        bgdir = opt.bg_dir
        bgpath = os.path.join(bgdir, basen[:-4] + '.png')
        im_bg = Image.open(bgpath).convert('L')
        mask2 = np.array(im_bg)  # mask out background
        mask2 = (mask2 >= 0.5).astype(np.float32)

        hair_A = (A / 2 + 0.5) * np.tile(mask, (3, 1, 1)) * np.tile(mask2, (3, 1, 1)) * 2 - 1
        bg_A = (A / 2 + 0.5) * np.tile((np.ones_like(mask2) - mask2), (3, 1, 1)) * 2 - 1

        item['hair_A'] = item.get('hair_A', [])
        item['bg_A'] = item.get('bg_A', [])
        item['mask'] = item.get('mask', [])
        item['mask2'] = item.get('mask2', [])
        item['hair_A'].append(hair_A)
        item['bg_A'].append(bg_A)
        item['mask'].append(mask)
        item['mask2'].append(mask2)

def single_dataloader(data, opt):
    """single_dataloader"""
    data_A = data['A']
    data_A_path = data['A_path']

    if len(data_A.shape) == 3:
        data_length = 1
    else:
        data_length = data_A.shape[0]
        assert data_A.shape[0] == data_A_path.shape[0] == opt.batch_size

    item = {}

    for _ in range(data_length):
        single_process(data_A, data_A_path, data_length, item, opt)
    return item
