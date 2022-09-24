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
import cv2
import numpy as np
from tqdm import tqdm
from mindspore import load_checkpoint, load_param_into_net

from src.utils import convert_to_square, delete_old_img, nms, combine_data_list, data_to_mindrecord, crop_landmark_image
from src.utils import read_annotation, save_hard_example, get_landmark_from_lfw_neg, pad, calibrate_box
from src.models.mtcnn import PNet, RNet
from src.prepare_data.generate_RNet_data import detect_pnet
from src.models.predict_nets import predict_rnet
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ONet data")
    parser.add_argument('--pnet_ckpt', type=str, required=True, help="Path of PNet checkpoint to detect")
    parser.add_argument('--rnet_ckpt', type=str, required=True, help="Path of RNet checkpoint to detect")

    return parser.parse_args()

def detect_rnet(im, dets, thresh, net):
    """Filter box and landmark by RNet"""
    h, w, _ = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])

    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    if int(num_boxes) == 0:
        print('Detection reasult of PNet is null!')
        return None, None

    for i in range(int(num_boxes)):
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24))
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except ValueError:
            continue

    cls_scores, reg = predict_rnet(cropped_ims, net)
    if cls_scores.ndim < 2:
        cls_scores = cls_scores[None, :]
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]

    if keep_inds.size != 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None, None

    keep = nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]

    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes, boxes_c

def crop_48_size_images(min_face_size, scale_factor, p_thresh, r_thresh, pnet, rnet):
    """Collect positive, negative, part images and resize to 48*48 as the input of ONet"""
    dataset_path = cfg.DATASET_DIR
    train_data_dir = cfg.TRAIN_DATA_DIR
    anno_file = os.path.join(dataset_path, 'wider_face_train.txt')

    # save positive, part, negative images
    pos_save_dir = os.path.join(train_data_dir, '48/positive')
    part_save_dir = os.path.join(train_data_dir, '48/part')
    neg_save_dir = os.path.join(train_data_dir, '48/negative')

     # save PNet train data
    save_dir = os.path.join(train_data_dir, '48')

    save_dir_list = [save_dir, pos_save_dir, part_save_dir, neg_save_dir]
    for dir_ in save_dir_list:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    # Read annotation data
    data = read_annotation(dataset_path, anno_file)
    all_boxes = []
    landmarks = []
    empty_array = np.array([])

    for image_path in tqdm(data['images']):
        assert os.path.exists(image_path), 'image not exists'
        im = cv2.imread(image_path)
        boxes_c = detect_pnet(im, min_face_size, scale_factor, p_thresh, pnet, 0.5)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

        _, boxes_c = detect_rnet(im, boxes_c, r_thresh, rnet)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

        all_boxes.append(boxes_c)

    # Save result to pickle file
    save_file = os.path.join(save_dir, 'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)

    save_hard_example(dataset_path, 48)

if __name__ == '__main__':
    args = parse_args()
    pnet_params = load_checkpoint(args.pnet_ckpt)
    rnet_params = load_checkpoint(args.rnet_ckpt)
    pnet_ = PNet()
    rnet_ = RNet()
    load_param_into_net(pnet_, pnet_params)
    load_param_into_net(rnet_, rnet_params)
    pnet_.set_train(False)
    rnet_.set_train(False)

    min_face_size_ = cfg.MIN_FACE_SIZE
    scale_factor_ = cfg.SCALE_FACTOR
    p_thresh_ = cfg.P_THRESH
    r_thresh_ = cfg.R_THRESH

    print("Start generating Box images")
    if not os.path.exists(cfg.TRAIN_DATA_DIR):
        os.mkdir(cfg.TRAIN_DATA_DIR)
    crop_48_size_images(min_face_size_, scale_factor_, p_thresh_, r_thresh_, pnet_, rnet_)

    print("Start generating landmark image")
    data_list = get_landmark_from_lfw_neg(cfg.DATASET_DIR)

    crop_landmark_image(cfg.TRAIN_DATA_DIR, data_list, 48, argument=True)

    print("Start combine data list")
    combine_data_list(os.path.join(cfg.TRAIN_DATA_DIR, '48'))

    data_to_mindrecord(os.path.join(cfg.TRAIN_DATA_DIR, '48'), cfg.MINDRECORD_DIR, 'ONet_train.mindrecord')
    delete_old_img(cfg.TRAIN_DATA_DIR, 48)
