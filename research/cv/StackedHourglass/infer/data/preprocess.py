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
""" preprocess for mindx inference"""
import os
import h5py
import cv2
import tqdm
import numpy as np

annot_dir = '../MPII/annot'
img_dir = '../MPII/images'
preprocess_path = './'


def get_transform(center_, scale_, res_):
    """
    generate trainsform matrix
    """
    h = 200 * scale_
    t = np.zeros((3, 3))
    t[0, 0] = float(res_[1]) / h
    t[1, 1] = float(res_[0]) / h
    t[0, 2] = res_[1] * (-float(center_[0]) / h + 0.5)
    t[1, 2] = res_[0] * (-float(center_[1]) / h + 0.5)
    t[2, 2] = 1
    return t


def transform(pt, center_t, scale_t, res_t, invert=0):
    """
    transform points
    """
    t = get_transform(center_t, scale_t, res_t)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def crop(img, center_c, scale_c, res_c):
    """
    crop images
    """
    # Left up
    ul = np.array(transform([0, 0], center_c, scale_c, res_c, invert=1))
    # Right down
    br = np.array(transform(res_c, center_c, scale_c, res_c, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return cv2.resize(new_img, res_c)


def get_img(num_eval=2958, num_train=300):
    """
    load validation and training images
    """
    input_res = 256
    val_f = h5py.File(os.path.join(annot_dir, "valid.h5"), "r")

    tr = tqdm.tqdm(range(0, num_train), total=num_train)
    # Train set
    train_f = h5py.File(os.path.join(annot_dir, "train.h5"), "r")
    for i in tr:
        path_t = "%s/%s" % (img_dir, train_f["imgname"][i].decode("UTF-8"))

        orig_img = cv2.imread(path_t)[:, :, ::-1]
        c = train_f["center"][i]
        s = train_f["scale"][i]
        im = crop(orig_img, c, s, (input_res, input_res))

        kp = train_f["part"][i]
        vis = train_f["visible"][i]
        kp2 = np.insert(kp, 2, vis, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2

        n = train_f["normalize"][i]

        yield kps, im, c, s, n

    tr2 = tqdm.tqdm(range(0, num_eval), total=num_eval)
    # Valid
    for i in tr2:
        path_t = "%s/%s" % (img_dir, val_f["imgname"][i].decode("UTF-8"))

        orig_img = cv2.imread(path_t)[:, :, ::-1]
        c = val_f["center"][i]
        s = val_f["scale"][i]
        im = crop(orig_img, c, s, (input_res, input_res))

        kp = val_f["part"][i]  # (16, 2)
        vis = val_f["visible"][i]
        kp2 = np.insert(kp, 2, vis, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2

        n = val_f["normalize"][i]

        yield kps, im, c, s, n


if __name__ == '__main__':
    img_path0 = os.path.join(preprocess_path, "00_data")
    img_path1 = os.path.join(preprocess_path, "11_data")
    img_path2 = os.path.join(preprocess_path, "22_data")
    img_path3 = os.path.join(preprocess_path, "33_data")
    img_path4 = os.path.join(preprocess_path, "44_data")
    img_path5 = os.path.join(preprocess_path, "55_data")
    img_path6 = os.path.join(preprocess_path, "66_data")
    os.makedirs(img_path0)
    os.makedirs(img_path1)
    os.makedirs(img_path2)
    os.makedirs(img_path3)
    os.makedirs(img_path4)
    os.makedirs(img_path5)
    os.makedirs(img_path6)
    j = 0
    for anns, o_img, cc, ss, nn in get_img():
        height, width = o_img.shape[0:2]
        center = (width / 2, height / 2)
        scale = max(height, width) / 200
        res = (256, 256)
        mat_ = get_transform(center, scale, res)[:2]
        mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
        inp = o_img / 255
        file_name = "data" + str(j) + ".bin"
        path0 = os.path.join(img_path0, file_name)
        np.array([inp], dtype=np.float32).tofile(path0)
        path1 = os.path.join(img_path1, file_name)
        np.array([inp[:, ::-1]], dtype=np.float32).tofile(path1)
        path2 = os.path.join(img_path2, file_name)
        np.array(anns, dtype=np.float32).tofile(path2)
        path3 = os.path.join(img_path3, file_name)
        np.array(cc, dtype=np.float32).tofile(path3)
        path4 = os.path.join(img_path4, file_name)
        np.array(ss, dtype=np.float32).tofile(path4)
        path5 = os.path.join(img_path5, file_name)
        np.array(nn, dtype=np.float32).tofile(path5)
        path6 = os.path.join(img_path6, file_name)
        np.array(mat, dtype=np.float32).tofile(path6)
        j = j + 1
