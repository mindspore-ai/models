# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Some functions used in training.
"""
import os

import cv2
import numpy as np
import xmltodict

all_cls = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor", "background"]

label_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]


def check_dir(data_dir):
    """

    :param data_dir: dir to check
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_xml(xml_path):
    """

    :param xml_path: xml path
    :return:
    """
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        all_bndboxs = list()
        obj_names = list()
        objects = xml_dict['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj['name']
                difficult = int(obj['difficult'])
                if difficult != 1:
                    bndbox = obj['bndbox']
                    all_bndboxs.append(
                        (int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
                    obj_names.append(obj_name)
        elif isinstance(objects, dict):
            obj_name = objects['name']
            difficult = int(objects['difficult'])
            if difficult != 1:
                bndbox = objects['bndbox']
                all_bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
                obj_names.append(obj_name)
        else:
            pass
        return np.array(all_bndboxs), obj_names


def iou(pred_box, all_bndboxs):
    """

    :param pred_box: predict boxes
    :param all_bndboxs: all boundary boxes
    :return: scores
    """
    if len(all_bndboxs.shape) == 1:
        all_bndboxs = all_bndboxs[np.newaxis, :]

    xA = np.maximum(pred_box[0], all_bndboxs[:, 0])
    yA = np.maximum(pred_box[1], all_bndboxs[:, 1])
    xB = np.minimum(pred_box[2], all_bndboxs[:, 2])
    yB = np.minimum(pred_box[3], all_bndboxs[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)

    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (all_bndboxs[:, 2] - all_bndboxs[:, 0]) * (all_bndboxs[:, 3] - all_bndboxs[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)

    return scores


def compute_ious(rects, all_bndboxs, obj_names):
    """

    :param rects: rects
    :param all_bndboxs: all boundary boxes
    :param obj_names: obj_names
    :return: max_score_iou_list, max_label_list, max_bnd_list
    """
    max_score_iou_list = list()
    max_label_list = list()
    max_bnd_list = list()

    for rect in rects:
        scores = iou(rect, all_bndboxs)
        max_score = max(scores)
        max_score_iou_list.append(max_score)
        scores = scores.tolist()
        index = scores.index(max_score)
        max_label_list.append(obj_names[index])
        max_bnd_list.append(all_bndboxs[index])

    return max_score_iou_list, max_label_list, max_bnd_list


def compute_trans(rect, bndbox):
    """

    :param rect: rect
    :param bndbox: bndbox
    :return: trans
    """
    p_x, p_y, p_w, p_h = rect

    xmin, ymin, xmax, ymax = bndbox
    g_w = xmax - xmin
    g_h = ymax - ymin
    g_x = xmin + g_w / 2
    g_y = ymin + g_h / 2

    t_x = (g_x - p_x) / p_w
    t_y = (g_y - p_y) / p_h
    t_w = np.log(g_w / p_w)
    t_h = np.log(g_h / p_h)

    return np.array((t_x, t_y, t_w, t_h))


def compute_transto(rect, trans, img_height, img_width):
    """

    :param rect: rect
    :param trans: trans
    :param img_height: img_height
    :param img_width: img_width
    :return: trans to
    """
    xmin, ymin, xmax, ymax = rect
    p_w = xmax - xmin
    p_h = ymax - ymin
    p_x = xmin + p_w / 2
    p_y = ymin + p_h / 2

    t_x, t_y, t_w, t_h = trans

    g_x = t_x * p_w + p_x
    g_y = t_y * p_h + p_y
    g_w = np.exp(t_w) * p_w
    g_h = p_h * np.exp(t_h)
    minx = (g_x - g_w / 2).clip(0, img_width)
    miny = (g_y - g_h / 2).clip(0, img_height)
    maxx = (g_x + g_w / 2).clip(0, img_width)
    maxy = (g_y + g_h / 2).clip(0, img_height)
    if minx == maxx or miny == maxy:
        flag = False
    else:
        flag = True

    return np.array((minx, miny, maxx, maxy)), flag


def norm(img, mean, std):
    """

    :param img: image
    :param mean: mean
    :param std: std
    :return: normalized image
    """
    B, G, R = cv2.split(img)
    B = (B - mean[2]) / std[2]
    G = (G - mean[1]) / std[1]
    R = (R - mean[0]) / std[0]
    new_img = cv2.merge([R, B, G])

    return new_img


def pre_process(img, mean, std):
    """

    :param img: image
    :param mean: mean
    :param std: std
    :return: processed image
    """
    img = cv2.resize(img, (256, 256))
    img = img[16:240, 16:240]
    img = norm(img, mean, std)
    img = img.transpose((2, 0, 1))
    return img
