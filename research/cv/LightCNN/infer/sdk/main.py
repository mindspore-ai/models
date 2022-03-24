#!/usr/bin/env python
# coding=utf-8
"""
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
"""
import os
import argparse
import json
import datetime
import cv2
import numpy as np
import scipy.io
import MxpiDataType_pb2 as MxpiDataType
from sklearn.metrics import roc_curve

from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

def load_image_list(img_dir, list_file):
    """load image list"""
    list_file_path = list_file
    j = open(list_file_path, 'r')
    image_list = []
    label = []
    for line in j:
        items = line.split()
        image_list.append(items[0].strip())
        label.append(items[1].strip())
    return label, image_list


def label_list_to_int(label):
    """convert type of labels to integer"""
    int_label = []
    for e in label:
        try:
            inte = int(e)
        except ValueError:
            print('Label are not int numbers. A mapping will be used.')
            break
        int_label.append(inte)
    if len(int_label) == len(label):
        return int_label
    return None


def string_list_to_cells(lst):
    """
    Uses numpy.ndarray with dtype=object. Convert list to np.ndarray().
    """
    cells = np.ndarray(len(lst), dtype='object')
    for idx, ele in enumerate(lst):
        cells[idx] = ele
    return cells


def extract_features_to_dict(image_dir, list_file):
    """extract features and save them with dictionary"""
    label, img_list = load_image_list(image_dir, list_file)
    ftr = feature
    integer_label = label_list_to_int(label)
    feature_dict = {'features': ftr,
                    'label': integer_label,
                    'label_original': string_list_to_cells(label),
                    'image_path': string_list_to_cells(img_list)}
    return feature_dict


def compute_cosine_score(feature1, feature2):
    """compute cosine score"""
    feature1_norm = np.linalg.norm(feature1)
    feature2_norm = np.linalg.norm(feature2)
    score = np.dot(feature1, feature2) / (feature1_norm * feature2_norm)
    return score


def lfw_eval(lightcnn_result, lfw_pairs_mat_path):
    """eval lfw"""
    features = lightcnn_result['features']
    lfw_pairs_mat = scipy.io.loadmat(lfw_pairs_mat_path)
    pos_pair = lfw_pairs_mat['pos_pair']
    neg_pair = lfw_pairs_mat['neg_pair']

    pos_scores = np.zeros(len(pos_pair[1]))

    for idx, _ in enumerate(pos_pair[1]):
        feat1 = features[pos_pair[0, idx] - 1, :]
        feat2 = features[pos_pair[1, idx] - 1, :]
        pos_scores[idx] = compute_cosine_score(feat1, feat2)
    pos_label = np.ones(len(pos_pair[1]))

    neg_scores = np.zeros(len(neg_pair[1]))

    for idx, _ in enumerate(neg_pair[1]):
        feat1 = features[neg_pair[0, idx] - 1, :]
        feat2 = features[neg_pair[1, idx] - 1, :]
        neg_scores[idx] = compute_cosine_score(feat1, feat2)
    neg_label = -1 * np.ones(len(neg_pair[1]))

    scores = np.concatenate((pos_scores, neg_scores), axis=0)
    labele = np.concatenate((pos_label, neg_label), axis=0)

    fpr, tpr, _ = roc_curve(labele, scores, pos_label=1)
    res = tpr - (1 - fpr)
    eer = tpr[np.squeeze(np.where(res >= 0))[0]] * 100
    far_10 = tpr[np.squeeze(np.where(fpr <= 0.01))[-1]] * 100
    far_01 = tpr[np.squeeze(np.where(fpr <= 0.001))[-1]] * 100
    far_00 = tpr[np.squeeze(np.where(fpr <= 0.0))[-1]] * 100

    print('100%eer:      ', round(eer, 2))
    print('tpr@far=1%:   ', round(far_10, 2))
    print('tpr@far=0.1%: ', round(far_01, 2))
    print('tpr@far=0%:   ', round(far_00, 2))

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    rets = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if rets < 0:
        print("Failed to send data to stream.")
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="lightcnn process")
    parser.add_argument("--image_path", type=str, default='data/image', help="root path of image")
    parser.add_argument("--img_list", type=str, default='data/image.txt', help="root path of image.txt")
    parser.add_argument("--result_path", type=str, default='output', help="root path of result")
    parser.add_argument("--lfw_pairs_mat", type=str, default='../../mat_files/lfw_pairs.mat')

    args = parser.parse_args()
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("lightcnn.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    streamName = b'lightcnn'
    labels, imag_list = load_image_list(args.image_path, args.img_list)
    feature_shape = (13233, 256)
    feature = np.empty(feature_shape, dtype='float32', order='C')
    num = 0
    for idxs, img_name in enumerate(imag_list):
        img = cv2.imread(os.path.join(args.image_path, img_name), cv2.IMREAD_GRAYSCALE)
        if img.shape != (128, 128):
            img = cv2.resize(img, (128, 128))
        img = np.reshape(img, (1, 1, 128, 128))
        inputs = img.astype(np.float32) / 255.0
        if not send_source_data(0, inputs, streamName, streamManagerApi):
            exit()
        keyVec = StringVector()
        keyVec.push_back(b"mxpi_tensorinfer0")
        start_time = datetime.datetime.now()
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.size() == 0:
            print("inferResult is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            exit()
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        data = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)
        data = data.reshape(1, 256)
        feature[num, :] = data
        num = num+1
        with open(os.path.join(args.result_path, 'res_' +
                               os.path.basename(img_name).replace('.bmp', '.txt')), 'w') as f_write:
            f_write.writelines(json.dumps(data.tolist()))
            f_write.write('\n')
    dic = extract_features_to_dict(image_dir=args.image_path, list_file=args.img_list)
    lfw_eval(dic, lfw_pairs_mat_path=args.lfw_pairs_mat)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
