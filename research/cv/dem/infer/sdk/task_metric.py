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
"""calculate infer result accuracy"""

import argparse
import numpy as np
import scipy.io as sio


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--res_path", type=str, default="./python_DEM/res", help="result numpy path")
    parser.add_argument("--data_dir", type=str, default="/home/dataset/DEM_data",
                        help="path where the dataset is saved")
    parser.add_argument("--dataset", type=str, default="AwA", choices=['AwA', 'CUB'],
                        help="dataset which is chosen to use")
    args_opt = parser.parse_args()
    return args_opt


def kNNClassify(newInput, dataSet, labels, k):
    """classify using kNN"""
    numSamples = dataSet.shape[0]
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex
    #return sortedDistIndices


def compute_accuracy_att(att_pred_0, pred_len, test_att_0, test_visual_0, test_id_0, test_label_0):
    """calculate accuracy using infer result"""
    outpred = [0] * pred_len
    test_label_0 = test_label_0.astype("float32")
    for i in range(pred_len):
        outputLabel = kNNClassify(test_visual_0[i, :], att_pred_0, test_id_0, 1)
        outpred[i] = outputLabel
    outpred = np.array(outpred)
    acc_0 = np.equal(outpred, test_label_0).mean()
    return acc_0


def dataset_CUB(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/CUB_data/train_attr.mat')
    train_att_0 = np.array(f['train_attr'])
    # print('train attr:', train_att.shape)

    f = sio.loadmat(data_path+'/CUB_data/train_cub_googlenet_bn.mat')
    train_x_0 = np.array(f['train_cub_googlenet_bn'])
    # print('train x:', train_x.shape)

    f = sio.loadmat(data_path+'/CUB_data/test_cub_googlenet_bn.mat')
    test_x_0 = np.array(f['test_cub_googlenet_bn'])
    # print('test x:', test_x.shape)

    f = sio.loadmat(data_path+'/CUB_data/test_proto.mat')
    test_att_0 = np.array(f['test_proto'])
    test_att_0 = test_att_0.astype("float16")
    # test_att_0 = Tensor(test_att_0, mindspore.float32)
    # print('test att:', test_att.shape)

    f = sio.loadmat(data_path+'/CUB_data/test_labels_cub.mat')
    test_label_0 = np.squeeze(np.array(f['test_labels_cub']))
    # print('test x2label:', test_x2label)

    f = sio.loadmat(data_path+'/CUB_data/testclasses_id.mat')
    test_id_0 = np.squeeze(np.array(f['testclasses_id']))
    # print('test att2label:', test_att2label)

    return train_att_0, train_x_0, test_x_0, test_att_0, test_label_0, test_id_0


def dataset_AwA(data_path):
    """input:*.mat, output:array"""
    f = sio.loadmat(data_path+'/AwA_data/train_googlenet_bn.mat')
    train_x_0 = np.array(f['train_googlenet_bn'])

    # useless data
    train_att_0 = np.empty(1)

    f = sio.loadmat(data_path+'/AwA_data/wordvector/train_word.mat')
    train_word_0 = np.array(f['train_word'])

    f = sio.loadmat(data_path+'/AwA_data/test_googlenet_bn.mat')
    test_x_0 = np.array(f['test_googlenet_bn'])

    f = sio.loadmat(data_path+'/AwA_data/attribute/pca_te_con_10x85.mat')
    test_att_0 = np.array(f['pca_te_con_10x85'])
    test_att_0 = test_att_0.astype("float16")

    f = sio.loadmat(data_path+'/AwA_data/wordvector/test_vectors.mat')
    test_word_0 = np.array(f['test_vectors'])
    test_word_0 = test_word_0.astype("float16")

    f = sio.loadmat(data_path+'/AwA_data/test_labels.mat')
    test_label_0 = np.squeeze(np.array(f['test_labels']))

    f = sio.loadmat(data_path+'/AwA_data/testclasses_id.mat')
    test_id_0 = np.squeeze(np.array(f['testclasses_id']))

    return train_x_0, train_att_0, train_word_0, test_x_0, \
        test_att_0, test_word_0, test_label_0, test_id_0


def read_res(res_path):
    """load result"""
    return np.loadtxt(res_path, dtype=np.float32)


def test_result(dir_name, res_path, dataset):
    """calculate"""
    if dataset == 'AwA':
        pred_len = 6180
        _, _, _, test_x, test_att, _, test_label, test_id = dataset_AwA(dir_name)
    elif dataset == 'CUB':
        pred_len = 2933
        _, _, test_x, test_att, test_label, test_id = dataset_CUB(dir_name)
    att_pred_res = read_res(res_path)

    acc = compute_accuracy_att(att_pred_res, pred_len, test_att, test_x, test_id, test_label)
    return acc


if __name__ == '__main__':
    args = parse_args()
    print("dataset:", args.dataset)
    final_acc = test_result(args.data_dir, args.res_path, args.dataset)
    print('accuracy :', final_acc)
