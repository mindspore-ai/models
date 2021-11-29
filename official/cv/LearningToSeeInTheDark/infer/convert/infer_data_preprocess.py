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
"""test"""
import os
import glob
import h5py
import numpy as np


def pack_raw(raw):
    """ pack sony raw data into 4 channels """

    im = np.maximum(raw - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],  # 取从0到H-1和W-1，步长为2，下同
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def get_test_data(input_dir1, gt_dir1, test_ids1, result_dir1):
    """ preprocess input data into .bin files """
    for test_id in test_ids1:
        in_files = glob.glob(input_dir1 + '%05d_00*.hdf5' % test_id)

        gt_files = glob.glob(gt_dir1 + '%05d_00*.hdf5' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        gt_exposure = float(gt_fn[9: -6])

        for in_path in in_files:  # 复写label

            in_fn = os.path.basename(in_path)
            in_exposure = float(in_fn[9: -6])
            ratio = min(gt_exposure / in_exposure, 300.0)
            ima = h5py.File(in_path, 'r')
            in_rawed = ima.get('in')[:]
            input_image = np.expand_dims(pack_raw(in_rawed), axis=0) * ratio
            input_image = np.minimum(input_image, 1.0)
            input_image = input_image.transpose([0, 3, 1, 2])
            input_image = np.float32(input_image)
            bin_name = os.path.join(result_dir1, in_fn[0: 9]) + '.bin'
            print(bin_name)
            input_image.tofile(bin_name)


if __name__ == '__main__':

    local_data_path = '../rawed_sony'
    input_dir = os.path.join(local_data_path, 'short/')
    gt_dir = os.path.join(local_data_path, 'long/')
    result_dir = './bin'
    test_fns = glob.glob(gt_dir + '1*.hdf5')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
    get_test_data(input_dir, gt_dir, test_ids, result_dir)
