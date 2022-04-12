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
"""pre process for 310 inference"""
import os
import argparse
import cv2
import numpy as np
from PIL import Image
from matlab_cp2tform import get_similarity_transform_for_cv2

parser = argparse.ArgumentParser(description="sphereface preprocess data")
args = parser.parse_args()


def calcul_acc(labels, preds):
    return sum(1 for x, y in zip(labels, preds) if x == y) / len(labels)

def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def save_mnist_to_jpg(datatxt_path, pairtxt_path, datasrc, output_path):
    output_path = output_path
    landmark = {}
    with open(datatxt_path) as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.replace('\n', '').split('\t')
        ascend = []
        for i in range(5):
            ascend.append([int(l[2 * i + 1]), int(l[2 * i + 2])])
        landmark[datasrc + l[0]] = ascend
    with open(pairtxt_path) as f:
        pairs_lines = f.readlines()[1:]
    for j in range(6000):
        p = pairs_lines[j].replace('\n', '').split('\t')
        sameflag = 0
        if len(p) == 3:
            sameflag = 1
            name1 = datasrc + p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = datasrc + p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))

        if len(p) == 4:
            sameflag = 0
            name1 = datasrc + p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = datasrc + p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = np.array(Image.open(name1).convert('RGB'), np.float32)
        img1 = img1[:, :, ::-1]
        img1 = alignment(img1, landmark[name1])
        img1path = output_path + '/' + '{}_{}_{}.jpg'.format(j, 1, sameflag)
        img2 = np.array(Image.open(name2).convert('RGB'), np.float32)
        img2 = img2[:, :, ::-1]
        img2 = alignment(img2, landmark[name2])
        img2path = output_path + '/' + '{}_{}_{}.jpg'.format(j, 2, sameflag)
        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
        cv2.imwrite(img1path, imglist[0])
        cv2.imwrite(img2path, imglist[2])
    print("=" * 20, "preprocess data finished", "=" * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mindspore SphereFace')
    # Datasets
    parser.add_argument('--datatxt', default='../lfw/lfw_landmark.txt', type=str,
                        help='data path')
    parser.add_argument('--pairtxt', default='../lfw/pairs.txt', type=str,
                        help='output path')
    parser.add_argument('--data', default='../lfw/', type=str)
    parser.add_argument('--output_path',
                        default='../lfw_aligned/',
                        type=str)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    save_mnist_to_jpg(args.datatxt, args.pairtxt, args.data, args.output_path)
