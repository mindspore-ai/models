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
'''General tools'''
import operator
import datetime
import numpy as np
from sklearn.metrics import average_precision_score


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def compute_ap(class_score_matrix, labels):
    '''compute_ap'''
    num_classes = class_score_matrix.shape[1]
    one_hot_labels = dense_to_one_hot(labels, num_classes)
    average_precision = []
    for i in range(num_classes):
        ps = average_precision_score(one_hot_labels[:, i], class_score_matrix[:, i])
        average_precision.append(ps)
    return np.array(average_precision)


def calculate_IoU(i0, i1):
    '''calculate_IoU'''
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def nms_temporal(x1, x2, s, overlap):
    '''nms_temporal'''
    pick = []
    assert len(x1) == len(s)
    assert len(x2) == len(s)
    if not x1:
        return pick

    union = list(map(operator.sub, x2, x1)) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x: x[1])] # sort and get index

    while I:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i], x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i], x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <= overlap:
                I_new.append(I[j])
        I = I_new
    return pick


def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    '''
    compute recall at certain IoU
    '''
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k, :, 0]]
        ends = [e for e in sentence_image_reg_mat[k, :, 1]]
        picks = nms_temporal(starts, ends, sim_v, iou_thresh-0.05)
        if top_n < len(picks):
            picks = picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end), (pred_start, pred_end))
            if iou >= iou_thresh:
                correct_num += 1
                break
    return correct_num

def print_log(msg='', end='\n'):
    '''print_log'''
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' +\
        str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)
    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]
    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else:
            print('[' + t + '] ' + str(line))
