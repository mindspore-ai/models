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

from src.utils import delete_old_img, nms, combine_data_list, data_to_mindrecord
from src.utils import crop_landmark_image, process_image, generate_box
from src.utils import read_annotation, save_hard_example, get_landmark_from_lfw_neg
from src.models.mtcnn import PNet
from src.models.predict_nets import predict_pnet
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RNet data")
    parser.add_argument('--pnet_ckpt', type=str, required=True, help="Path of PNet checkpoint to detect")


    return parser.parse_args()

def detect_pnet(im, min_face_size, scale_factor, thresh, net, nms_thresh=0.7):
    """Filter box and landmark by PNet"""
    net_size = 12
    # Ratio of face and input image
    current_scale = float(net_size) / min_face_size
    im_resized = process_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()

    # Image pymaid
    while min(current_height, current_width) > net_size:
        cls, reg = predict_pnet(im_resized, net)
        boxes = generate_box(cls[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor
        im_resized = process_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue

        keep = nms(boxes[:, :5], nms_thresh, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)

    if not all_boxes:
        return None

    all_boxes = np.vstack(all_boxes)

    keep = nms(all_boxes[:, 0:5], 0.7)
    all_boxes = all_boxes[keep]

    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])

    return boxes_c.T

def crop_24_size_images(min_face_size, scale_factor, thresh, net):
    """Collect positive, negative, part images and resize to 24*24 as the input of RNet"""
    dataset_path = cfg.DATASET_DIR
    train_data_dir = cfg.TRAIN_DATA_DIR
    anno_file = os.path.join(dataset_path, 'wider_face_train.txt')

    # save positive, part, negative images
    pos_save_dir = os.path.join(train_data_dir, '24/positive')
    part_save_dir = os.path.join(train_data_dir, '24/part')
    neg_save_dir = os.path.join(train_data_dir, '24/negative')

    # save PNet train data
    save_dir = os.path.join(train_data_dir, '24')

    save_dir_list = [save_dir, pos_save_dir, part_save_dir, neg_save_dir]
    for dir_ in save_dir_list:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    # Read annotation data
    data = read_annotation(dataset_path, anno_file)
    all_boxes, landmarks = [], []
    empty_array = np.array([])

    # Rec image with PNet
    for image_path in tqdm(data['images']):
        assert os.path.exists(image_path), 'image not exists'
        im = cv2.imread(image_path)
        boxes_c = detect_pnet(im, min_face_size, scale_factor, thresh, net)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue
        all_boxes.append(boxes_c)


    # Save result to pickle file
    save_file = os.path.join(save_dir, 'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)

    save_hard_example(dataset_path, 24)

if __name__ == '__main__':
    args = parse_args()
    params = load_checkpoint(args.pnet_ckpt)
    pnet_ = PNet()
    load_param_into_net(pnet_, params)
    pnet_.set_train(False)

    min_face_size_ = cfg.MIN_FACE_SIZE
    scale_factor_ = cfg.SCALE_FACTOR
    p_thresh_ = cfg.P_THRESH

    print("Start generating Box images")
    if not os.path.exists(cfg.TRAIN_DATA_DIR):
        os.mkdir(cfg.TRAIN_DATA_DIR)
    crop_24_size_images(min_face_size_, scale_factor_, p_thresh_, pnet_)

    print("Start generating landmark image")
    data_list = get_landmark_from_lfw_neg(cfg.DATASET_DIR)

    crop_landmark_image(cfg.TRAIN_DATA_DIR, data_list, 24, argument=True)

    print("Start combine data list")
    combine_data_list(os.path.join(cfg.TRAIN_DATA_DIR, '24'))

    data_to_mindrecord(os.path.join(cfg.TRAIN_DATA_DIR, '24'), cfg.MINDRECORD_DIR, 'RNet_train.mindrecord')
    delete_old_img(cfg.TRAIN_DATA_DIR, 24)
