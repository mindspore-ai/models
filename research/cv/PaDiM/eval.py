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
import pickle
import random
from random import sample
from math import ceil
from collections import OrderedDict
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import src.dataset as dataset
from src.model import wide_resnet50_2
from src.operator import embedding_concat
from src.operator import view


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--device_id', type=int, default=7, help='Device id')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--class_name', type=str, default='bottle')
parser.add_argument('--dataset_path', type=str, default='./mvtec_anomaly_detection/', help='Dataset path')
parser.add_argument('--save_path', type=str, default='/mass_store/dataset/zjq/mvtec_result/')
parser.add_argument('--pre_ckpt_path', type=str, default='./wide_resnet50_2.ckpt', help='Pretrain checkpoint path')

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)


if __name__ == '__main__':
    # load model
    model = wide_resnet50_2()
    param_dict = load_checkpoint(args.pre_ckpt_path)
    load_param_into_net(model, param_dict)
    for p in model.trainable_params():
        p.requires_grad = False
    random.seed(1024)
    t_d = 1792
    d = 550
    idx = Tensor(sample(range(0, t_d), d))
    class_name = args.class_name
    batch_size = 16
    _, _, test_dataset, test_dataset_len = dataset.createDataset(args.dataset_path, class_name, batch_size)
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract train set features
    train_feature_filepath = os.path.join(args.save_path, 'wide_resnet50_2', 'train_%s.pkl' % class_name)

    print('load train set feature from: %s' % train_feature_filepath)
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)
    gt_list = []
    gt_mask_list = []
    test_imgs = []
    test_data_iter = test_dataset.create_dict_iterator()
    for data in tqdm(test_data_iter, '| feature extraction | test | %s |' % class_name,
                     total=ceil(test_dataset_len/batch_size)):
        test_imgs.extend(data['img'].asnumpy())
        gt_list.extend(data['label'].asnumpy())
        gt_mask_list.extend(data['gt'].asnumpy())
        # model prediction
        outputs = model(data['img'])
        # get intermediate layer outputs
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
    # calculate distance matrix
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
    