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
"""
python postprocess.py
"""
import os
import numpy as np

from src.config import config
from src.dataset import vox_generator_test


def one_hot(label, num_classes):
    """ one-hot encode """
    label_ = np.zeros([len(label), num_classes])
    label_[np.arange(len(label)), label] = 1
    return label_


def calculate_dice(true_label, pred_label, num_classes):
    """
        calculate dice

        Args:
            true_label: true sparse labels
            pred_label: predict sparse labels
            num_classes: number of classes

        Returns:
            dice evaluation index
    """
    true_label = true_label.astype(int)
    pred_label = pred_label.astype(int)
    true_label = true_label.flatten()
    true_label = one_hot(true_label, num_classes)
    pred_label = pred_label.flatten()
    pred_label = one_hot(pred_label, num_classes)
    intersection = np.sum(true_label * pred_label, axis=0)
    return (2. * intersection) / (np.sum(true_label, axis=0) + np.sum(pred_label, axis=0))


if __name__ == '__main__':
    # test dataset
    test_files = []
    with open(config.test_path) as f:
        for line in f:
            test_files.append(line[:-1])

    data_gen_test = vox_generator_test(config.data_path, test_files, config.correction)

    OFFSET_H = config.offset_height
    OFFSET_W = config.offset_width
    OFFSET_C = config.offset_channel
    HSIZE = config.height_size
    WSIZE = config.width_size
    CSIZE = config.channel_size
    PSIZE = config.pred_size

    OFFSET_PH = (config.height_size - config.pred_size) // 2
    OFFSET_PW = (config.width_size - config.pred_size) // 2
    OFFSET_PC = (config.channel_size - config.pred_size) // 2

    batches_w = int(np.ceil((config.image_width - config.width_size) / float(config.offset_width))) + 1
    batches_h = int(np.ceil((config.image_height - config.height_size) / float(config.offset_height))) + 1
    batches_c = int(np.ceil((config.image_channel - config.channel_size) / float(config.offset_channel))) + 1

    dice_whole, dice_core, dice_et = [], [], []
    data_count = 0
    post_result_path = os.path.join(config.post_result_path)
    for i in range(len(test_files)):
        print('predicting %s' % test_files[i])
        x, x_n, y = data_gen_test.__next__()
        pred = np.zeros([config.image_height, config.image_width, config.image_channel, config.num_classes])
        for hi in range(batches_h):
            offset_h = min(OFFSET_H * hi, config.image_height - HSIZE)
            offset_ph = offset_h + OFFSET_PH
            for wi in range(batches_w):
                offset_w = min(OFFSET_W * wi, config.image_width - WSIZE)
                offset_pw = offset_w + OFFSET_PW
                for ci in range(batches_c):
                    offset_c = min(OFFSET_C * ci, config.image_channel - CSIZE)
                    offset_pc = offset_c + OFFSET_PC
                    data = x[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE, :]
                    if not np.max(data) == 0 and np.min(data) == 0:
                        t1_t1ce_score_path = os.path.join(post_result_path, "flair_t2_node_{}_1.bin".format(data_count))
                        t1_t1ce_score = np.fromfile(t1_t1ce_score_path, dtype=np.float32).reshape(1, 5, 12, 12, 12)
                        t1_t1ce_score = np.transpose(t1_t1ce_score, axes=[0, 2, 3, 4, 1])
                        pred[offset_ph:offset_ph + PSIZE, offset_pw:offset_pw + PSIZE, offset_pc:offset_pc + PSIZE, :] \
                            += np.squeeze(t1_t1ce_score)
                        data_count = data_count + 1
        pred = np.argmax(pred, axis=-1)
        pred = pred.astype(int)
        print('calculating dice...')
        whole_pred = (pred > 0).astype(int)
        whole_gt = (y > 0).astype(int)
        core_pred = (pred == 1).astype(int) + (pred == 4).astype(int)
        core_gt = (y == 1).astype(int) + (y == 4).astype(int)
        et_pred = (pred == 4).astype(int)
        et_gt = (y == 4).astype(int)
        dice_whole_batch = calculate_dice(whole_gt, whole_pred, 2)
        dice_core_batch = calculate_dice(core_gt, core_pred, 2)
        dice_et_batch = calculate_dice(et_gt, et_pred, 2)
        dice_whole.append(dice_whole_batch)
        dice_core.append(dice_core_batch)
        dice_et.append(dice_et_batch)
        print(dice_whole_batch)
        print(dice_core_batch)
        print(dice_et_batch)

    dice_whole = np.array(dice_whole)
    dice_core = np.array(dice_core)
    dice_et = np.array(dice_et)

    print('mean dice whole:')
    print(np.mean(dice_whole, axis=0))
    print('mean dice core:')
    print(np.mean(dice_core, axis=0))
    print('mean dice enhance:')
    print(np.mean(dice_et, axis=0))
