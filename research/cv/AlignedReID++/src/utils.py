"""some functions maybe used"""
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

from __future__ import absolute_import
import os
import sys
import errno
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import array, argmin

import mindspore
import mindspore.ops as ops


def mkdir_if_missing(directory):
    """make diratory"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter():
    """averagemeter"""
    def __init__(self):
        self.reset()

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

def save_checkpoint(state, is_best, fpath='checkpoint.ckpt'):
    mkdir_if_missing(os.path.dirname(fpath))
    mindspore.save_checkpoint(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'best_model.ckpt'))

class Logger():
    '''Logger'''
    def __init__(self, fpath=None):
        self.terminal = sys.stdout
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.log = open(fpath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def _traceback(D):
    """traceback"""
    i, j = array(D.shape)-1
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j-1], D[i-1, j]))
        if tb == 0:
            j -= 1
        else:
            i -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def dtw(dist_mat):
    """dtw"""
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    path = _traceback(dist)
    return dist[-1, -1]/sum(dist.shape), dist, path

def read_image(img_path):
    """read inage"""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will Redo. Don't worry. Just chill".format(img_path))
    return img

def img_to_tensor(img, transform):
    img = transform(img)
    expand_dims = ops.ExpandDims()
    img = expand_dims(img, 0)
    return img

def show_feature(x):
    for j in range(len(x)):
        for i in range(len(64)):
            ax = plt.subplot(4, 16, i+1)
            ax.set_title('No #{}'.format(i))
            ax.axis('off')
            plt.imshow(x[j].cpu().data.numpy()[0, i, :, :], cmap='jet')
        plt.show()

def feat_flatten(feat):
    shp = feat.shape
    feat = feat.reshape(shp[0] * shp[1], shp[2])
    return feat

def show_similar(local_img_path, img_path, similarity, bbox):
    """show_similar"""
    img1 = cv2.imread(local_img_path)
    img2 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (64, 128))
    img2 = cv2.resize(img2, (64, 128))
    cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)

    p = np.where(similarity == np.max(similarity))
    y, x = p[0][0], p[1][0]
    cv2.rectangle(img2, (x - bbox[2] / 2, y - bbox[3] / 2), (x + bbox[2] / 2, y + bbox[3] / 2), (0, 255, 0), 1)
    plt.subplot(1, 3, 1).set_title('patch')
    plt.imshow(img1)
    plt.subplot(1, 3, 2).set_title(('max similarity: ' + str(np.max(similarity))))
    plt.imshow(img2)
    plt.subplot(1, 3, 3).set_title('similarity')
    plt.imshow(similarity)

def show_alignedreid(local_img_path, img_path, dist):
    """show_alignedreid"""
    def drow_line(img, similarity):
        for i in range(1, len(similarity)):
            cv2.line(img, (0, i*16), (63, i*16), color=(0, 255, 0))
            cv2.line(img, (96, i*16), (160, i*16), color=(0, 255, 0))
    def drow_path(img, path):
        for i in range(len(path[0])):
            cv2.line(img, (64, 8+16*path[0][i]), (96, 8+16*path[1][i]), color=(255, 255, 0))
    img1 = cv2.imread(local_img_path)
    img2 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (64, 128))
    img2 = cv2.resize(img2, (64, 128))
    img = np.zeros((128, 160, 3)).astype(img1.dtype)
    img[:, :64, :] = img1
    img[:, -64:, :] = img2
    drow_line(img, dist)
    d, _, sp = dtw(dist)
    origin_dist = np.mean(np.diag(dist))
    drow_path(img, sp)
    plt.subplot(1, 2, 1).set_title('Aligned distance: %.4f \n Original distance: %.4f' %(d, origin_dist))
    plt.subplot(1, 2, 1).set_xlabel('Aligned Result')
    plt.imshow(img)
    plt.subplot(1, 2, 2).set_title('Distance Map')
    plt.subplot(1, 2, 2).set_xlabel('Right Image')
    plt.subplot(1, 2, 2).set_ylabel('Left Image')
    plt.imshow(dist)
    plt.subplots_adjust(bottom=0.1, left=0.075, right=0.85, top=0.9)
    cax = plt.axes([0.9, 0.25, 0.025, 0.5])
    plt.colorbar(cax=cax)
    plt.show()

def merge_feature(feature_list, shp, sample_rate=None):
    """merge_feature"""
    def pre_process(torch_feature_map):
        numpy_feature_map = torch_feature_map.cpu().data.numpy()[0]
        numpy_feature_map = numpy_feature_map.transpose(1, 2, 0)
        shp = numpy_feature_map.shape[:2]
        return numpy_feature_map, shp
    def resize_as(tfm, shp):
        nfm, shp2 = pre_process(tfm)
        scale = shp[0]/shp2[0]
        nfm1 = nfm.repeat(scale, axis=0).repeat(scale, axis=1)
        return nfm1
    final_nfm = resize_as(feature_list[0], shp)
    for i in range(1, len(feature_list)):
        temp_nfm = resize_as(feature_list[i], shp)
        final_nfm = np.concatenate((final_nfm, temp_nfm), axis=-1)
    if sample_rate > 0:
        final_nfm = final_nfm[0:-1:sample_rate, 0:-1, sample_rate, :]
    return final_nfm
