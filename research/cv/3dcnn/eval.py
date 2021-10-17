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
python eval.py
"""
import numpy as np

from mindspore import dtype as mstype
from mindspore import Model, context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.models import Dense24
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
    target = config.device_target

    # init context
    device_id = config.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False, device_id=device_id)

    # test dataset
    test_files = []
    with open(config.test_path) as f:
        for line in f:
            test_files.append(line[:-1])

    data_gen_test = vox_generator_test(config.data_path, test_files, config.correction)

    # network
    network = Dense24(num_classes=5)
    network.set_train(False)

    # load model parameters
    param_dict = load_checkpoint(config.ckpt_path)
    load_param_into_net(network, param_dict)
    model = Model(network)

    OFFSET_H = config.offset_height
    OFFSET_W = config.offset_width
    OFFSET_C = config.offset_channel
    HSIZE = config.height_size
    WSIZE = config.width_size
    CSIZE = config.channel_size
    PSIZE = config.pred_size

    OFFSET_PH = (HSIZE - PSIZE) // 2
    OFFSET_PW = (WSIZE - PSIZE) // 2
    OFFSET_PC = (CSIZE - PSIZE) // 2

    batches_w = int(np.ceil((240 - WSIZE) / float(OFFSET_W))) + 1
    batches_h = int(np.ceil((240 - HSIZE) / float(OFFSET_H))) + 1
    batches_c = int(np.ceil((155 - CSIZE) / float(OFFSET_C))) + 1

    dice_whole, dice_core, dice_et = [], [], []
    for i in range(len(test_files)):
        print('predicting %s' % test_files[i])
        x, x_n, y = data_gen_test.__next__()
        pred = np.zeros([240, 240, 155, 5])
        for hi in range(batches_h):
            offset_h = min(OFFSET_H * hi, 240 - HSIZE)
            offset_ph = offset_h + OFFSET_PH
            for wi in range(batches_w):
                offset_w = min(OFFSET_W * wi, 240 - WSIZE)
                offset_pw = offset_w + OFFSET_PW
                for ci in range(batches_c):
                    offset_c = min(OFFSET_C * ci, 155 - CSIZE)
                    offset_pc = offset_c + OFFSET_PC
                    data = x[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE, :]
                    data_norm = x_n[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE, :]
                    data_norm = np.expand_dims(data_norm, 0)
                    if not np.max(data) == 0 and np.min(data) == 0:
                        flair_t2_node = data_norm[:, :, :, :, :2]
                        t1_t1ce_node = data_norm[:, :, :, :, 2:]
                        flair_t2_node = np.transpose(flair_t2_node, axes=[0, 4, 1, 2, 3])
                        t1_t1ce_node = np.transpose(t1_t1ce_node, axes=[0, 4, 1, 2, 3])
                        flair_t2_node = Tensor(flair_t2_node, mstype.float32)
                        t1_t1ce_node = Tensor(t1_t1ce_node, mstype.float32)
                        flair_t2_score, t1_t1ce_score = model.predict(flair_t2_node, t1_t1ce_node)
                        t1_t1ce_score = t1_t1ce_score.asnumpy()
                        t1_t1ce_score = np.transpose(t1_t1ce_score, axes=[0, 2, 3, 4, 1])
                        pred[offset_ph:offset_ph + PSIZE, offset_pw:offset_pw + PSIZE, offset_pc:offset_pc + PSIZE, :] \
                            += np.squeeze(t1_t1ce_score)

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
