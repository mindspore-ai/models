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
# ===========================================================================
"""Prepare & transform Cityscapes dataset to mindrecord format."""
import os
import glob
import ast
import argparse
from io import BytesIO

import cv2
import numpy as np

from mindspore.mindrecord import FileWriter


def encode_segmap(lbl, ignore_label):
    """encode segmap"""
    mask = np.uint8(lbl)

    num_classes = 19
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    class_map = dict(zip(valid_classes, range(num_classes)))
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_label
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]

    return mask


def data_to_mindrecord_img(prefix='cityscapes.mindrecord',
                           file_num=1,
                           root="/cache/data",
                           mindrecord_dir="/cache/train",
                           split='train'):
    """read and transform dataset to mindrecord"""
    IGNORE_LABLE = 255
    MINDRECORD_FILE = os.path.join(mindrecord_dir, prefix)
    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    writter = FileWriter(MINDRECORD_FILE, file_num)

    images_base = os.path.join(root, 'leftImg8bit', split)
    annotations_base = os.path.join(root, 'gtFine', split)
    file_pattern = images_base + os.sep + '*' + os.sep + '*.png'
    files = glob.glob(file_pattern, recursive=True)

    cityscapes_json = {
        "image": {"type": "bytes"},
        "label": {"type": "bytes"}
    }

    writter.add_schema(cityscapes_json, "cityscapes_json")

    images_files_num = len(files)
    for index in range(images_files_num):
        img_path = files[index].rstrip()
        lbl_path = os.path.join(annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR), np.uint8)
        _lbl = np.array(cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE), np.uint8)
        _lbl = encode_segmap(_lbl, IGNORE_LABLE)

        img_encode = cv2.imencode(".png", _img)[1]
        lbl_encode = cv2.imencode(".png", _lbl)[1]
        image_bytes = BytesIO(img_encode).getvalue()
        label_bytes = BytesIO(lbl_encode).getvalue()

        row = {"image": image_bytes, "label": label_bytes}
        if (index + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(index + 1, images_files_num))
        writter.write_raw_data([row])
    writter.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mindrecord")
    parser.add_argument('--data_url', type=str, default='', help='Obs url of original dataset')
    parser.add_argument('--train_url', type=str, default='', help='Obs url to store output mindrecord')
    parser.add_argument('--data_path', type=str, default='', help='Data path to dataset')
    parser.add_argument('--train_path', type=str, default='', help='Output path to store mindrecord')
    parser.add_argument('--split', type=str, default='train', help='train or val')
    parser.add_argument('--modelArts', type=ast.literal_eval, default=True,
                        help='whether on modelarts or not, default: True')
    Args = parser.parse_args()

    if Args.modelArts:
        import moxing as mox
        local_data_path = "/cache/data"
        local_train_path = "/cache/train"

        mox.file.make_dirs(local_data_path)
        mox.file.make_dirs(local_train_path)

        # server path
        local_img_path = os.path.join(local_data_path, 'gtFine')
        local_lbl_path = os.path.join(local_data_path, 'leftImg8bit')
        mox.file.make_dirs(local_img_path)
        mox.file.make_dirs(local_lbl_path)

        # obs path
        obs_img_path = os.path.join(Args.data_url, 'gtFine')
        obs_lbl_path = os.path.join(Args.data_url, 'leftImg8bit')

        # upload dataset
        mox.file.copy_parallel(src_url=obs_img_path, dst_url=local_img_path)
        mox.file.copy_parallel(src_url=obs_lbl_path, dst_url=local_lbl_path)
        os.system("ls -l /cache/data/")
    else:
        local_data_path = Args.data_path
        local_train_path = Args.train_path

    data_to_mindrecord_img(prefix='cityscapes_'+Args.split+'.mindrecord',
                           root=local_data_path,
                           mindrecord_dir=local_train_path,
                           split=Args.split)

    if Args.modelArts:
        mox.file.copy_parallel(src_url=local_train_path, dst_url=Args.train_url)
