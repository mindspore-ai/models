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

import shutil
import pickle
import os
import numpy as np
from numpy import random
import cv2
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import ops, Tensor
from mindspore import dtype as mstype

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    return inter / (box_area + area - inter + 1e-10)

def convert_to_square(bbox):
    """Convert bbox to square
    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox
    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


def nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def pad(bboxes, w, h):
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list

def calibrate_box(bbox, reg):
    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c

def process_image(img, scale):
    """Preprocess image"""
    height, width, _ = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)

    image = np.array(img_resized).astype(np.float32)
    # HWC2CHW
    image = image.transpose((2, 0, 1))
    # Normalize
    image = (image - 127.5) / 128
    return image

def generate_box(cls_map, reg, scale, threshold):
    """get box"""
    stride = 2
    cellsize = 12

    t_index = np.where(cls_map > threshold)

    # Zero face
    if t_index[0].size == 0:
        return np.array([])

    # Offset
    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]
    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # Box, score, offset
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape = [n, 9]
    return boundingbox.T

def read_annotation(data_path, label_path):
    """Load image path and box from original dataset"""
    data = dict()
    images = []
    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        labels = line.strip().split(' ')
        # image path
        imagepath = labels[0]
        # if has empty line, then break
        if not imagepath:
            break
        # absolute image path
        imagepath = os.path.join(data_path, 'WIDER_train/images', imagepath + '.jpg')
        images.append(imagepath)

        one_image_boxes = []
        for i in range(0, len(labels) - 1, 4):
            xmin = float(labels[1 + i])
            ymin = float(labels[2 + i])
            xmax = float(labels[3 + i])
            ymax = float(labels[4 + i])

            one_image_boxes.append([xmin, ymin, xmax, ymax])

        boxes.append(one_image_boxes)

    data['images'] = images
    data['boxes'] = boxes
    return data

def save_hard_example(data_path, save_size):
    """Save data according to the predicted result"""
    filename = os.path.join(data_path, 'wider_face_train.txt')
    data = read_annotation(data_path, filename)

    im_idx_list = data['images']
    gt_boxes_list = data['boxes']

    pos_save_dir = os.path.join(data_path, 'train_data/%d/positive' % save_size)
    part_save_dir = os.path.join(data_path, 'train_data/%d/part' % save_size)
    neg_save_dir = os.path.join(data_path, 'train_data/%d/negative' % save_size)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    neg_file = open(os.path.join(data_path, 'train_data/%d/negative.txt' % save_size), 'w')
    pos_file = open(os.path.join(data_path, 'train_data/%d/positive.txt' % save_size), 'w')
    part_file = open(os.path.join(data_path, 'train_data/%d/part.txt' % save_size), 'w')

    det_boxes = pickle.load(open(os.path.join(data_path, 'train_data/%d/detections.pkl' % save_size), 'rb'))

    assert len(det_boxes) == len(im_idx_list), "Predicted result are not consistent with local data"

    n_idx, p_idx, d_idx = 0, 0, 0

    pbar = tqdm(total=len(im_idx_list))
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        pbar.update(1)
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)

        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # delete small object
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # cal iou
            iou = IoU(box, gts)

            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size), interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)

                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # Offset
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos and part
                if np.max(iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    # label=1
                    pos_file.write(
                        save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    # label=-1
                    part_file.write(
                        save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

    pbar.close()
    neg_file.close()
    part_file.close()
    pos_file.close()

class BBox:
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def project(self, point):
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

# flip image
def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_flipped_by_x, landmark_

# rotate image
def rotate(img, box, landmark, alpha):
    center = ((box.left + box.right) / 2, (box.top + box.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[box.top:box.bottom + 1, box.left:box.right + 1]
    return face, landmark_

def check_dir(dirs):
    """Check directory"""
    if isinstance(dirs, list):
        for d in dirs:
            if not os.path.exists(d):
                os.mkdir(d)
    else:
        if not os.path.exists(dirs):
            os.mkdir(dirs)

def do_argument(image, resized_image, landmark_, box_, size_, F_imgs_, F_landmarks_):
    """Flip, rotate image"""
    if random.choice([0, 1]) > 0:
        face_flipped, landmark_flipped = flip(resized_image, landmark_)
        face_flipped = cv2.resize(face_flipped, (size_, size_))
        F_imgs_.append(face_flipped)
        F_landmarks_.append(landmark_flipped.reshape(10))

    if random.choice([0, 1]) > 0:
        face_rotated_by_alpha, landmark_rorated = rotate(image, box_, box_.reprojectLandmark(landmark_), 5)
        landmark_rorated = box_.projectLandmark(landmark_rorated)
        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size_, size_))
        F_imgs_.append(face_rotated_by_alpha)
        F_landmarks_.append(landmark_rorated.reshape(10))
        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
        face_flipped = cv2.resize(face_flipped, (size_, size_))
        F_imgs_.append(face_flipped)
        F_landmarks_.append(landmark_flipped.reshape(10))

    if random.choice([0, 1]) > 0:
        face_rotated_by_alpha, landmark_rorated = rotate(image, box_, box_.reprojectLandmark(landmark_), -5)
        landmark_rorated = box_.projectLandmark(landmark_rorated)
        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size_, size_))
        F_imgs_.append(face_rotated_by_alpha)
        F_landmarks_.append(landmark_rorated.reshape(10))
        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
        face_flipped = cv2.resize(face_flipped, (size_, size_))
        F_imgs_.append(face_flipped)
        F_landmarks_.append(landmark_flipped.reshape(10))

def crop_landmark_image(data_dir, data_list, size, argument=True):
    """crop and save landmark image"""
    npr = np.random
    image_id = 0
    output = os.path.join(data_dir, str(size))

    check_dir(output)
    dstdir = os.path.join(output, 'landmark')

    check_dir(dstdir)
    f = open(os.path.join(output, 'landmark.txt'), 'w')

    idx = 0
    for (imgPath, box, landmarkGt) in tqdm(data_list):
        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)
        img_h, img_w, _ = img.shape
        gt_box = np.array([box.left, box.top, box.right, box.bottom])
        f_face = img[box.top:box.bottom + 1, box.left:box.right + 1]
        try:
            f_face = cv2.resize(f_face, (size, size))
        except ValueError as e:
            print(e)
            continue
        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        if argument:
            landmark = np.zeros((5, 2))
            idx = idx + 1
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(10):
                box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                try:
                    delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                    delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                except ValueError as e:
                    print(e)
                    continue
                nx1 = int(max(x1 + gt_w / 2 - box_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - box_size / 2 + delta_y, 0))
                nx2 = nx1 + box_size
                ny2 = ny1 + box_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                iou = IoU(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / box_size, (one[1] - ny1) / box_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    box = BBox([nx1, ny1, nx2, ny2])
                    do_argument(img, resized_im, landmark_, box, size, F_imgs, F_landmarks)
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        for i in range(len(F_imgs)):
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            cv2.imwrite(os.path.join(dstdir, '%d.jpg' % (image_id)), F_imgs[i])
            landmarks = list(map(str, list(F_landmarks[i])))
            f.write(os.path.join(dstdir, '%d.jpg' % (image_id)) + ' -2 ' + ' '.join(landmarks) + '\n')
            image_id += 1
    f.close()


def combine_data_list(data_dir):
    """Combine all data list"""
    with open(os.path.join(data_dir, 'positive.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, 'negative.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, 'part.txt'), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir, 'landmark.txt'), 'r') as f:
        landmark = f.readlines()

    with open(os.path.join(data_dir, 'all_data_list.txt'), 'w') as f:
        base_num = len(pos) // 1000 * 1000

        print(f"Original: neg {len(neg)} pos {len(pos)} part {len(part)} landmark {len(landmark)} base {base_num}")

        neg_keep = random.choice(len(neg), size=base_num * 3, replace=base_num * 3 > len(neg))
        part_keep = random.choice(len(part), size=base_num, replace=base_num > len(part))
        pos_keep = random.choice(len(pos), size=base_num, replace=base_num > len(pos))
        landmark_keep = random.choice(len(landmark), size=base_num * 2, replace=base_num * 2 > len(landmark))

        print(f"After sampling: neg {len(neg_keep)} pos {len(pos_keep)} part {len(part_keep)} \
              landmark {len(landmark_keep)}")

        for i in pos_keep:
            f.write(pos[i].replace('\\', '/'))
        for i in neg_keep:
            f.write(neg[i].replace('\\', '/'))
        for i in part_keep:
            f.write(part[i].replace('\\', '/'))
        for i in landmark_keep:
            f.write(landmark[i].replace('\\', '/'))


def data_to_mindrecord(data_folder, mindrecord_prefix, mindrecord_name):
    # Load all data list
    data_list_path = os.path.join(data_folder, 'all_data_list.txt')
    with open(data_list_path, 'r') as f:
        train_list = f.readlines()

    if not os.path.exists(mindrecord_prefix):
        os.mkdir(mindrecord_prefix)
    mindrecord_path = os.path.join(mindrecord_prefix, mindrecord_name)
    writer = FileWriter(mindrecord_path, 1, overwrite=True)

    mtcnn_json = {
        "image": {"type": "bytes"},
        "label": {"type": "int32"},
        "box_target": {"type": "float32", "shape": [4]},
        "landmark_target": {"type": "float32", "shape": [10]}
    }

    writer.add_schema(mtcnn_json, "mtcnn_json")

    count = 0
    for item in tqdm(train_list):
        sample = item.split(' ')
        image = sample[0]
        label = int(sample[1])
        box = [0] * 4
        landmark = [0] * 10

        # Only has box
        if len(sample) == 6:
            box = sample[2:]

        # Only has landmark
        if len(sample) == 12:
            landmark = sample[2:]
        box = np.array(box).astype(np.float32)
        landmark = np.array(landmark).astype(np.float32)
        img = cv2.imread(image)
        _, encoded_img = cv2.imencode('.jpg', img)

        row = {
            "image": encoded_img.tobytes(),
            "label": label,
            "box_target": box,
            "landmark_target": landmark
        }
        writer.write_raw_data([row])
        count += 1
    writer.commit()
    print("Total train data: ", count)
    print("Create mindrecord done!")


def get_landmark_from_lfw_neg(dataset_path, with_landmark=True):
    """Get landmark data"""

    anno_file = os.path.join(dataset_path, 'trainImageList.txt')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    result = []
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        img_path = os.path.join(dataset_path, annotation[0]).replace('\\', '/')

        # box
        box = (annotation[1], annotation[3], annotation[2], annotation[4])
        box = [float(_) for _ in box]
        box = list(map(int, box))

        if not with_landmark:
            result.append((img_path, BBox(box)))
            continue

        # 5 landmark points
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(annotation[5 + 2 * index]), float(annotation[5 + 2 * index + 1]))
            landmark[index] = rv
        result.append((img_path, BBox(box), landmark))

    return result

def delete_old_img(old_image_folder, image_size):
    """Delete original data"""
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'positive'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'negative'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'part'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'landmark'), ignore_errors=True)

    os.remove(os.path.join(old_image_folder, str(image_size), 'positive.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'negative.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'part.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'landmark.txt'))

class MultiEpochsDecayLR(LearningRateSchedule):
    """
    Calculate learning rate base on multi epochs decay function.

    Args:
        learning_rate(float): Initial learning rate.
        multi_steps(list int): The steps corresponding to decay learning rate.
        steps_per_epoch(int): How many steps for each epoch.
        factor(int): Learning rate decay factor. Default: 10.

    Returns:
        Tensor, learning rate.
    """
    def __init__(self, learning_rate, multi_epochs, steps_per_epoch, factor=10):
        super(MultiEpochsDecayLR, self).__init__()
        if not isinstance(multi_epochs, (list, tuple)):
            raise TypeError("multi_epochs must be list or tuple.")
        self.multi_epochs = Tensor(np.array(multi_epochs, dtype=np.float32) * steps_per_epoch)
        self.num = len(multi_epochs)
        self.start_learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.factor = factor
        self.pow = ops.Pow()
        self.cast = ops.Cast()
        self.less_equal = ops.LessEqual()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, global_step):
        cur_step = self.cast(global_step, mstype.float32)
        multi_epochs = self.cast(self.multi_epochs, mstype.float32)
        epochs = self.cast(self.less_equal(multi_epochs, cur_step), mstype.float32)
        lr = self.start_learning_rate / self.pow(self.factor, self.reduce_sum(epochs, ()))
        return lr
