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
from tqdm import tqdm
import cv2
import numpy as np
from numpy import random as npr

from src.utils import IoU, crop_landmark_image, combine_data_list, data_to_mindrecord, get_landmark_from_lfw_neg
from src.utils import delete_old_img, check_dir
import config as cfg

def write_neg_data(save_dir, anno_file, idx, resized_image):
    """Save negative data"""
    save_file = os.path.join(save_dir, "%s.jpg" % idx)
    anno_file.write(save_dir + '/%s.jpg' % idx + ' 0\n')
    cv2.imwrite(save_file, resized_image)
    idx += 1
    return idx

def write_pos_data(save_dir, anno_file, idx, resized_image, x1, y1, x2, y2):
    """Save positive data"""
    save_file = os.path.join(save_dir, '%s.jpg' % idx)
    anno_file.write(save_dir + '/%s.jpg' % idx + ' 1 %.2f %.2f %.2f %.2f\n' % (x1, y1, x2, y2))
    cv2.imwrite(save_file, resized_image)
    idx += 1
    return idx

def write_part_data(save_dir, anno_file, idx, resized_image, x1, y1, x2, y2):
    """Save part data"""
    save_file = os.path.join(save_dir, '%s.jpg' % idx)
    anno_file.write(save_dir + '/%s.jpg' % idx + ' -1 %.2f %.2f %.2f %.2f\n' % (x1, y1, x2, y2))
    cv2.imwrite(save_file, resized_image)
    idx += 1
    return idx

def crop_12_size_images():
    """Collect positive, negative, part images and resize to 12*12 as the input of PNet"""
    dataset_path = cfg.DATASET_DIR
    train_data_dir = cfg.TRAIN_DATA_DIR
    # annotataion file of WIDER dataset
    anno_file = os.path.join(dataset_path, 'wider_face_train.txt')
    # path of WIDER images
    img_dir = os.path.join(dataset_path, 'WIDER_train/images')

    # save positive, part, negative images
    pos_save_dir = os.path.join(train_data_dir, '12/positive')
    part_save_dir = os.path.join(train_data_dir, '12/part')
    neg_save_dir = os.path.join(train_data_dir, '12/negative')

    # save PNet train data
    save_dir = os.path.join(train_data_dir, '12')
    save_dir_list = [save_dir, pos_save_dir, part_save_dir, neg_save_dir]
    check_dir(save_dir_list)

    # Generate annotation files of positive, part, negative images
    pos_anno = open(os.path.join(save_dir, 'positive.txt'), 'w')
    part_anno = open(os.path.join(save_dir, 'part.txt'), 'w')
    neg_anno = open(os.path.join(save_dir, 'negative.txt'), 'w')

    # original dataset
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    total_num = len(annotations)
    print(f"Total images number: {total_num}")
    # record number of positive, negative and part images
    p_idx, n_idx, d_idx = 0, 0, 0
    # record number of processed images
    idx = 0
    for annotation in tqdm(annotations):
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        # box data
        box = list(map(float, annotation[1:]))
        # split box data
        boxes = np.array(box, dtype=np.float32).reshape(-1, 4)
        # load image
        img = cv2.imread(os.path.join(img_dir, im_path + '.jpg'))
        idx += 1
        height, width, _ = img.shape
        neg_num = 0
        while neg_num < 50:
            # random select image size to crop
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            # crop box
            crop_box = np.array([nx, ny, nx + size, ny + size])
            iou = IoU(crop_box, boxes)
            cropped_im = img[ny:ny + size, nx:nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            # negative image if IoU < 0.3
            if np.max(iou) < 0.3:
                n_idx = write_neg_data(neg_save_dir, neg_anno, n_idx, resized_im)
                neg_num += 1

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            for _ in range(5):
                size = npr.randint(12, min(width, height) / 2)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # exclude image which is too large
                if nx1 + size > width or ny1 + size > height:
                    continue
                # get crop box
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                # calculate iou
                iou = IoU(crop_box, boxes)
                cropped_im = img[ny1:ny1 + size, nx1:nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                # negative image if iou < 0.3
                if np.max(iou) < 0.3:
                    n_idx = write_neg_data(neg_save_dir, neg_anno, n_idx, resized_im)

            for _ in range(20):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                if w < 5:
                    continue
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                cropped_im = img[ny1:ny2, nx1:nx2, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                box_ = box.reshape(1, -1)
                iou = IoU(crop_box, box_)
                # positive image if iou > 0.65
                if iou > 0.65:
                    p_idx = write_pos_data(pos_save_dir, pos_anno, p_idx, resized_im, offset_x1, offset_y1,
                                           offset_x2, offset_y2)
                # part image if iou > 0.4 and iou < 0.65
                elif iou >= 0.4:
                    d_idx = write_part_data(part_save_dir, part_anno, d_idx, resized_im, offset_x1, offset_y1,
                                            offset_x2, offset_y2)

    print(f"{idx} images processed, pos: {p_idx} part: {d_idx} neg: {n_idx}")
    pos_anno.close()
    part_anno.close()
    neg_anno.close()

if __name__ == '__main__':
    print("Start generating Box images")
    if not os.path.exists(cfg.TRAIN_DATA_DIR):
        os.mkdir(cfg.TRAIN_DATA_DIR)
    crop_12_size_images()
    print("Start generating landmark image")
    data_list = get_landmark_from_lfw_neg(cfg.DATASET_DIR)
    crop_landmark_image(cfg.TRAIN_DATA_DIR, data_list, 12, argument=True)
    print("Start combine data list")
    combine_data_list(os.path.join(cfg.TRAIN_DATA_DIR, '12'))
    data_to_mindrecord(os.path.join(cfg.TRAIN_DATA_DIR, '12'), cfg.MINDRECORD_DIR, 'PNet_train.mindrecord')
    delete_old_img(cfg.TRAIN_DATA_DIR, 12)
