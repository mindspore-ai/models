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
import datetime
import argparse
import cv2
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        ktest = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(ktest))
        folds.append([train, ktest])
    return folds

def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    print(tfm)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true == y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T)/(a_norm * b_norm)
    return similarity

def inference(dir_name, res_dir_name, PL_PATH):
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
    file_list = sorted(file_list)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    cnt = 1
    predict = []
    for file_name in file_list:
        stream_name = b'im_sphereface'
        in_plugin_id = 0
        file_path = os.path.join(dir_name, file_name)
        img_decode = cv2.imread(file_path)
        img_decode = np.transpose(img_decode, axes=(2, 0, 1))
        img_decode = img_decode.reshape([112, 96, 3])

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

        _, _, label = file_name.split('_')
        label, _ = label.split('.')
        file_path, _ = file_name.split('.')
        output_path = res_dir_name + file_path + ".txt"
        temp = output.tolist()
        with open(output_path, 'w') as file:
            for i in range(len(temp)):
                s = "{:.6}".format(temp[i])
                s = s.replace("'", '').replace(",", '') + ' '
                file.write(s)

        if cnt % 2 != 0:
            embeding = output
            name = file_name
        else:
            cosdistance = cosine_distance(embeding, output)
            predict.append('{}\t{}\t{}\t{}\n'.format(name, file_name, cosdistance, label))
        if cnt % 1000 == 0:
            print('sdk run time: {}'.format((end_time - start_time).microseconds))
            print(
                f"End-2end inference, file_name: sphereface, {cnt + 1}/{len(file_list)}, elapsed_time: {end_time}.\n"
            )
        cnt += 1
    return predict

def pred_threshold(predicts):
    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    ans = lambda line: line.strip('\n').split()
    predicts = np.array([ans(x) for x in predicts])
    for idx, (train, ktest) in enumerate(folds):
        print("now the idx is %d", idx)
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[ktest]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

def test():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Datasets
    parser.add_argument('--eval_url', default='../data/lfw_aligned/', type=str,
                        help='data path')
    parser.add_argument('--PL_PATH', default='../data/config/sphereface.pipeline', type=str,
                        help='output path')
    parser.add_argument('--result_url', default='../data/sdk_out/', type=str)
    parser.add_argument('--target',
                        default='lfw',
                        help='test targets.')
    args = parser.parse_args()
    data_path = args.eval_url
    output_path = args.result_url
    if os.path.exists(data_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        predicts = inference(data_path, output_path, args.PL_PATH)

    pred_threshold(predicts)

if __name__ == '__main__':
    test()
