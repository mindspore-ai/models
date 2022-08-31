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

import os
import pickle
import numpy as np
import scipy.misc
from tqdm import tqdm


def up_3d_extract(dataset_path, out_path, mode):
    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_ = [], []

    # training/test splits
    if mode == 'trainval':
        txt_file = os.path.join(dataset_path, 'trainval.txt')
    elif mode == 'lsp_test':
        txt_file = 'data/namesUPlsp.txt'
    elif mode == 'train':
        txt_file = os.path.join(dataset_path, 'train.txt')
    elif mode == 'test':
        txt_file = os.path.join(dataset_path, 'test.txt')
    elif mode == 'val':
        txt_file = os.path.join(dataset_path, 'val.txt')


    file = open(txt_file, 'r')
    txt_content = file.read()
    imgs = txt_content.split('\n')
    for img_i in tqdm(imgs):
        # skip empty row in txt
        if not img_i:
            continue

        # image name
        img_base = img_i[1:-10]
        img_name = '%s_image.png' % img_base

        keypoints_file = os.path.join(dataset_path, '%s_joints.npy'%img_base)
        keypoints = np.load(keypoints_file)
        vis = keypoints[2]
        keypoints = keypoints[:2].T

        part = np.zeros([24, 3])
        part[:14] = np.hstack([keypoints, np.vstack(vis)])

        render_name = os.path.join(dataset_path, '%s_render_light.png' % img_base)
        I = scipy.misc.imread(render_name)
        ys, xs = np.where(np.min(I, axis=2) < 255)
        bbox = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1])
        center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
        scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

        # pose and shape
        pkl_file = os.path.join(dataset_path, '%s_body.pkl' % img_base)
        pkl = pickle.load(open(pkl_file, 'rb'), encoding='iso-8859-1')
        pose = pkl['pose']
        shape = pkl['betas']
        rt = pkl['rt']
        if max(rt) > 0:
            print(rt)

        # store data
        imgnames_.append(img_name)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        poses_.append(pose)
        shapes_.append(shape)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'up_3d_%s.npz' % mode)
    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             pose=poses_,
             shape=shapes_)
