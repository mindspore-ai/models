# coding=utf-8
"""
Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import json
import os
import argparse
import pickle
import numpy as np
import cv2
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import interpolate
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput


class LFold:
    '''
    LFold
    '''

    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    '''
    calculate_roc
    '''
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    '''calculate_acc
    '''
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    '''
    calculate_val
    '''
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    '''
    calculate_val_far
    '''
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    '''evaluate
    '''
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def inference(dir_name, res_dir_name, PL_PATH):
    '''inference
    '''
    _, issame_list = pickle.load(open(data_path + '.bin', 'rb'), encoding='bytes')
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(PL_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # Construct the input of the stream
    data_input = MxDataInput()
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    cnt = 0
    leng = int(len(file_list) / 2)
    embeding = np.zeros(shape=[leng, 512], dtype=np.float32)
    embeding_f = np.zeros(shape=[leng, 512], dtype=np.float32)

    for file_name in file_list:
        stream_name = b'im_arcface'
        in_plugin_id = 0
        file_path = os.path.join(dir_name, file_name)
        img_decode = cv2.imread(file_path)
        img_decode = np.transpose(img_decode, axes=(2, 0, 1))
        img_decode = img_decode.reshape([112, 112, 3])

        _, encoded_image = cv2.imencode(".jpg", img_decode)
        img_bytes = encoded_image.tobytes()
        data_input.data = img_bytes
        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        start_time = datetime.datetime.now()
        infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
        end_time = datetime.datetime.now()
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()
        resultList = MxpiDataType.MxpiTensorPackageList()
        resultList.ParseFromString(infer_result[0].messageBuf)
        output = np.frombuffer(resultList.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        if file_name.startswith('f'):
            tmp_list = file_name.split('_')
            tmp_num = int(tmp_list[1])
            embeding_f[tmp_num] = output
            save_path = os.path.join(res_dir_name, f"f_{tmp_num}.json")
        else:
            tmp_list = file_name.split('_')
            tmp_num = int(tmp_list[0])
            embeding[tmp_num] = output
            save_path = os.path.join(res_dir_name, f"{tmp_num}.json")
        cnt += 1

        with open(save_path, "w") as fp:
            fp.write(json.dumps(output.tolist()))
        if cnt % 1000 == 0:
            print('sdk run time: {}'.format((end_time - start_time).microseconds))
            print(
                f"End-2end inference, file_name: {save_path}, {cnt + 1}/{len(file_list)}, elapsed_time: {end_time}.\n"
            )
    # destroy streams

    stream_manager_api.DestroyAllStreams()

    embeddings = embeding
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, acc, _, _, _ = evaluate(embeddings, issame_list)
    acc1 = np.mean(acc)
    std1 = np.std(acc)
    embeddings = embeding + embeding_f
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    _, _, accuracy, _, _, _ = evaluate(
        embeddings, issame_list)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    print('[%s]Accuracy: %1.5f+-%1.5f' % (dir_name.split('/')[-1], acc1, std1))
    print('[%s]Accuracy-Flip: %1.5f+-%1.5f' %
          (dir_name.split('/')[-1], acc2, std2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Datasets
    parser.add_argument('--eval_url', default='../data/input/data/', type=str,
                        help='data path')
    parser.add_argument('--PL_PATH', default='../data/config/arcface.pipeline', type=str,
                        help='output path')
    parser.add_argument('--result_url', default='../data/sdk_out/', type=str)
    parser.add_argument('--target',
                        default='lfw',
                        help='test targets.')
    # lfw,cfp_fp,agedb_30,calfw,cplfw
    args = parser.parse_args()
    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        data_path = os.path.join(args.eval_url, name)
        output_path = os.path.join(args.result_url, name)
        if os.path.exists(data_path):
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            inference(data_path, output_path, args.PL_PATH)
