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
import argparse
import json
import pickle
import random
from random import sample
from pathlib import Path
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np

from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore.ops import operations as P


from src.operator import (embedding_concat, prep_dirs, view)

parser = argparse.ArgumentParser(description='postprocess')

parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--img_dir', type=str, default='')
parser.add_argument('--label_dir', type=str, default='')
parser.add_argument('--class_name', type=str, default='bottle')

args = parser.parse_args()

if __name__ == '__main__':
    test_label_path = Path(os.path.join(args.label_dir, "infer_label.json"))
    train_result_path = os.path.join(args.result_dir, 'pre')
    test_result_path = os.path.join(args.result_dir, 'infer')
    with test_label_path.open('r') as dst_file:
        test_label = json.load(dst_file)
    random.seed(1024)
    t_d = 1792
    d = 550
    idx = Tensor(sample(range(0, t_d), d))
    class_name = args.class_name
    # dataset
    embedding_dir_path, sample_path = prep_dirs('./', args.class_name)
    # train
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    train_feature_filepath = os.path.join(embedding_dir_path, 'wide_resnet50_2', 'train_%s.pkl' % class_name)
    for i in range(int(len(os.listdir(train_result_path)) / 3)):
        features_one_path = os.path.join(train_result_path, "data_img_{}_0.bin".format(i))
        features_two_path = os.path.join(train_result_path, "data_img_{}_1.bin".format(i))
        features_three_path = os.path.join(train_result_path, "data_img_{}_2.bin".format(i))
        features_one = Tensor(np.fromfile(features_one_path, dtype=np.float32).reshape(1, 256, 56, 56))
        features_two = Tensor(np.fromfile(features_two_path, dtype=np.float32).reshape(1, 512, 28, 28))
        features_three = Tensor(np.fromfile(features_three_path, dtype=np.float32).reshape(1, 1024, 14, 14))
        outputs = [features_one, features_two, features_three]
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v)
        outputs = []
    concat_op = ops. Concat(0)
    for k, v in train_outputs.items():
        train_outputs[k] = concat_op(v)
    # Embedding concat
    embedding_vectors = train_outputs['layer1'].asnumpy()
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name].asnumpy())
    embedding_vectors = Tensor(embedding_vectors)
    # randomly select d dimension
    gather = P.Gather()
    embedding_vectors = gather(embedding_vectors, idx, 1)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    if B > 300:
        embedding_vectors = view(embedding_vectors, B, C, H, W)
    else:
        embedding_vectors = embedding_vectors.view((B, C, H * W))
    op = ops.ReduceMean()
    mean = op(embedding_vectors, 0).asnumpy()
    cov = np.zeros((C, C, H * W), dtype=np.float32)
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].asnumpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    train_outputs = [mean, cov]
    with open(train_feature_filepath, 'wb') as f:
        pickle.dump(train_outputs, f)
    # extract train set features
    train_outputs = []
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    gt_list = []
    gt_mask_list = []
    embedding_vectors = embedding_vectors.asnumpy()
    dist_list = []

    for i in range(int(len(os.listdir(test_result_path)) / 3)):
        test_single_label = test_label['{}'.format(i)]
        gt_list = test_single_label['label']
        gt_mask_list = test_single_label['gt']
        features_one_path = os.path.join(test_result_path, "data_img_{}_0.bin".format(i))
        features_two_path = os.path.join(test_result_path, "data_img_{}_1.bin".format(i))
        features_three_path = os.path.join(test_result_path, "data_img_{}_2.bin".format(i))
        features_one = Tensor(np.fromfile(features_one_path, dtype=np.float32).reshape(1, 256, 56, 56))
        features_two = Tensor(np.fromfile(features_two_path, dtype=np.float32).reshape(1, 512, 28, 28))
        features_three = Tensor(np.fromfile(features_three_path, dtype=np.float32).reshape(1, 1024, 14, 14))

        outputs = [features_one, features_two, features_three]
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v)
        outputs = []
    concat_op = ops.Concat(0)
    for k, v in test_outputs.items():
        test_outputs[k] = concat_op(v)
    # Embedding concat
    embedding_vectors = test_outputs['layer1'].asnumpy()
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name].asnumpy())
    embedding_vectors = Tensor(embedding_vectors)
    # randomly select d dimension
    gather = P.Gather()
    embedding_vectors = gather(embedding_vectors, idx, 1)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    if B > 300:
        embedding_vectors = view(embedding_vectors, B, C, H, W)
    else:
        embedding_vectors = embedding_vectors.view((B, C, H * W))
    embedding_vectors = embedding_vectors.asnumpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)
    dist_list = Tensor(np.array(dist_list).transpose(1, 0).reshape((B, H, W)))
    # upsample
    expand_dims = ops.ExpandDims()
    resize_bilinear = nn.ResizeBilinear()
    score_map = resize_bilinear(expand_dims(dist_list, 1), size=(224, 224)).squeeze().asnumpy()
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=np.float32)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    print('class_name is {}'.format(class_name))
    print('img_auc: %.3f, pixel_auc: %.3f' % (img_roc_auc, per_pixel_rocauc))
    