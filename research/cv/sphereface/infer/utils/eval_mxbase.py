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
import cv2
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2

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

def inference(dir_name):
    file_list = os.listdir(dir_name)
    file_list = sorted(file_list)
    # file_list = file_list[2:]
    cnt = 1
    predict = []
    for file_name in file_list:
        direct = dir_name + file_name
        _, _, label = file_name.split('_')
        label, _ = label.split('.')
        f = open(direct)
        res = f.read()
        ans = res.split(' ')
        ans = ans[0:512]
        output = []
        for temp in ans:
            output.append(float(temp))
        output = np.array(output)
        if cnt % 2 != 0:
            embeding = output
            name = file_name
        else:
            cosdistance = cosine_distance(embeding, output)
            predict.append('{}\t{}\t{}\t{}\n'.format(name, file_name, cosdistance, label))
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
    parser.add_argument('--eval_url', default='../data/mxbase_out/', type=str,
                        help='data path')
    parser.add_argument('--target',
                        default='lfw',
                        help='test targets.')
    args = parser.parse_args()
    data_path = args.eval_url
    predicts = inference(data_path)

    pred_threshold(predicts)

if __name__ == '__main__':
    test()
