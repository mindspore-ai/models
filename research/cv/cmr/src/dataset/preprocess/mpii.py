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
from os.path import join
import h5py
import numpy as np


def mpii_extract(dataset_path, out_path):

    # convert joints to global order
    joints_idx = [0, 1, 2, 3, 4, 5, 14, 15, 12, 13, 6, 7, 8, 9, 10, 11]

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []

    # annotation files
    annot_file = os.path.join('data', 'train.h5')

    # read annotations
    f = h5py.File(annot_file, 'r')
    centers, imgnames, parts, scales = \
        f['center'], f['imgname'], f['part'], f['scale']

    # go over all annotated examples
    for center, imgname, part16, scale in zip(centers, imgnames, parts, scales):
        # check if all major body joints are annotated
        if (part16 > 0).sum() < 2*len(joints_idx):
            continue
        # keypoints
        part = np.zeros([24, 3])
        part[joints_idx] = np.hstack([part16, np.ones([16, 1])])

        # store data
        imgname = str(imgname, 'utf-8')
        imgnames_.append(join('images', imgname))
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)

    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'mpii_train.npz')
    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_)
