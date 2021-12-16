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
"""Face detection eval_util."""
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


class Box:
    """ This is a generic bounding box representation.
    This class provides some base functionality to both annotations and detections.

    Attributes:
        class_label (string): class string label; Default **''**
        object_id (int): Object identifier for reid purposes; Default **0**
        x_top_left (Number): X pixel coordinate of the top left corner of the bounding box; Default **0.0**
        y_top_left (Number): Y pixel coordinate of the top left corner of the bounding box; Default **0.0**
        width (Number): Width of the bounding box in pixels; Default **0.0**
        height (Number): Height of the bounding box in pixels; Default **0.0**
    """

    def __init__(self):
        self.class_label = ''  # class string label
        self.object_id = 0  # object identifier
        self.x_top_left = 0.0  # x pixel coordinate top left of the box
        self.y_top_left = 0.0  # y pixel coordinate top left of the box
        self.width = 0.0  # width of the box in pixels
        self.height = 0.0  # height of the box in pixels

    @classmethod
    def create(cls, obj=None):
        """ Create a bounding box from a string or other detection object.

        Args:
            obj (Box or string, optional): Bounding box object to copy attributes from or string to deserialize
        """
        instance = cls()

        if obj is None:
            return instance

        if isinstance(obj, str):
            instance.deserialize(obj)
        elif isinstance(obj, Box):
            instance.class_label = obj.class_label
            instance.object_id = obj.object_id
            instance.x_top_left = obj.x_top_left
            instance.y_top_left = obj.y_top_left
            instance.width = obj.width
            instance.height = obj.height
        else:
            raise TypeError(f'Object is not of type Box or not a string [obj.__class__.__name__]')

        return instance

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def serialize(self):
        """ abstract serializer, implement in derived classes. """
        raise NotImplementedError

    def deserialize(self, string):
        """ abstract parser, implement in derived classes. """
        raise NotImplementedError


class Detection(Box):
    """ This is a generic detection class that provides some base functionality all detections need.
    It builds upon :class:`~brambox.boxes.box.Box`.

    Attributes:
        confidence (Number): confidence score between 0-1 for that detection; Default **0.0**
    """

    def __init__(self):
        """ x_top_left,y_top_left,width,height are in pixel coordinates """
        super(Detection, self).__init__()
        self.confidence = 0.0  # Confidence score between 0-1

    @classmethod
    def create(cls, obj=None):
        """ Create a detection from a string or other box object.

        Args:
            obj (Box or string, optional): Bounding box object to copy attributes from or string to deserialize

        Note:
            The obj can be both an :class:`~brambox.boxes.annotations.Annotation` or
            a :class:`~brambox.boxes.detections.Detection`.
            For Detections the confidence score is copied over, for Annotations it is set to 1.
        """
        instance = super(Detection, cls).create(obj)

        if obj is None:
            return instance

        if isinstance(obj, Detection):
            instance.confidence = obj.confidence

        return instance

    def __repr__(self):
        """ Unambiguous representation """
        string = f'{self.__class__.__name__} ' + '{'
        string += f'class_label = {self.class_label}, '
        string += f'object_id = {self.object_id}, '
        string += f'x = {self.x_top_left}, '
        string += f'y = {self.y_top_left}, '
        string += f'w = {self.width}, '
        string += f'h = {self.height}, '
        string += f'confidence = {self.confidence}'
        return string + '}'

    def __str__(self):
        """ Pretty print """
        string = 'Detection {'
        string += f'\'{self.class_label}\' {self.object_id}, '
        string += f'[{int(self.x_top_left)}, {int(self.y_top_left)}, {int(self.width)}, {int(self.height)}]'
        string += f', {round(self.confidence * 100, 2)} %'
        return string + '}'

    def serialize(self):
        """ abstract serializer, implement in derived classes. """
        raise NotImplementedError

    def deserialize(self, string):
        """ abstract parser, implement in derived classes. """
        raise NotImplementedError


def get_bounding_boxes(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2, conf_thresh,
                       input_shape, num_classes):
    """get_bounding_boxes"""
    batch = cls_scores_0.shape[0]
    w_0 = int(input_shape[0] / 64)
    h_0 = int(input_shape[1] / 64)
    w_1 = int(input_shape[0] / 32)
    h_1 = int(input_shape[1] / 32)
    w_2 = int(input_shape[0] / 16)
    h_2 = int(input_shape[1] / 16)
    num_anchors_0 = cls_scores_0.shape[1]
    num_anchors_1 = cls_scores_1.shape[1]
    num_anchors_2 = cls_scores_2.shape[1]

    score_thresh_0 = cls_scores_0 > conf_thresh
    score_thresh_1 = cls_scores_1 > conf_thresh
    score_thresh_2 = cls_scores_2 > conf_thresh

    score_thresh_flat_0 = score_thresh_0.reshape(-1)
    score_thresh_flat_1 = score_thresh_1.reshape(-1)
    score_thresh_flat_2 = score_thresh_2.reshape(-1)

    score_thresh_expand_0 = np.expand_dims(score_thresh_0, axis=3)
    score_thresh_expand_1 = np.expand_dims(score_thresh_1, axis=3)
    score_thresh_expand_2 = np.expand_dims(score_thresh_2, axis=3)

    score_thresh_cat_0 = np.concatenate((score_thresh_expand_0, score_thresh_expand_0), axis=3)
    score_thresh_cat_0 = np.concatenate((score_thresh_cat_0, score_thresh_cat_0), axis=3)
    score_thresh_cat_1 = np.concatenate((score_thresh_expand_1, score_thresh_expand_1), axis=3)
    score_thresh_cat_1 = np.concatenate((score_thresh_cat_1, score_thresh_cat_1), axis=3)
    score_thresh_cat_2 = np.concatenate((score_thresh_expand_2, score_thresh_expand_2), axis=3)
    score_thresh_cat_2 = np.concatenate((score_thresh_cat_2, score_thresh_cat_2), axis=3)

    coords_0 = coords_0[score_thresh_cat_0].reshape(-1, 4)
    coords_1 = coords_1[score_thresh_cat_1].reshape(-1, 4)
    coords_2 = coords_2[score_thresh_cat_2].reshape(-1, 4)

    scores_0 = cls_scores_0[score_thresh_0].reshape(-1, 1)
    scores_1 = cls_scores_1[score_thresh_1].reshape(-1, 1)
    scores_2 = cls_scores_2[score_thresh_2].reshape(-1, 1)

    idx_0 = np.tile((np.arange(num_classes)), (batch, num_anchors_0, w_0 * h_0))
    idx_0 = idx_0[score_thresh_0].reshape(-1, 1)
    idx_1 = np.tile((np.arange(num_classes)), (batch, num_anchors_1, w_1 * h_1))
    idx_1 = idx_1[score_thresh_1].reshape(-1, 1)
    idx_2 = np.tile((np.arange(num_classes)), (batch, num_anchors_2, w_2 * h_2))
    idx_2 = idx_2[score_thresh_2].reshape(-1, 1)

    detections_0 = np.concatenate([coords_0, scores_0, idx_0.astype(np.float32)], axis=1)
    detections_1 = np.concatenate([coords_1, scores_1, idx_1.astype(np.float32)], axis=1)
    detections_2 = np.concatenate([coords_2, scores_2, idx_2.astype(np.float32)], axis=1)

    max_det_per_batch_0 = num_anchors_0 * h_0 * w_0 * num_classes
    slices_0 = [slice(max_det_per_batch_0 * i, max_det_per_batch_0 * (i + 1)) for i in range(batch)]
    det_per_batch_0 = np.array([score_thresh_flat_0[s].astype(np.int32).sum() for s in slices_0], dtype=np.int32)
    max_det_per_batch_1 = num_anchors_1 * h_1 * w_1 * num_classes
    slices_1 = [slice(max_det_per_batch_1 * i, max_det_per_batch_1 * (i + 1)) for i in range(batch)]
    det_per_batch_1 = np.array([score_thresh_flat_1[s].astype(np.int32).sum() for s in slices_1], dtype=np.int32)
    max_det_per_batch_2 = num_anchors_2 * h_2 * w_2 * num_classes
    slices_2 = [slice(max_det_per_batch_2 * i, max_det_per_batch_2 * (i + 1)) for i in range(batch)]
    det_per_batch_2 = np.array([score_thresh_flat_2[s].astype(np.int32).sum() for s in slices_2], dtype=np.int32)

    split_idx_0 = np.cumsum(det_per_batch_0, axis=0)
    split_idx_1 = np.cumsum(det_per_batch_1, axis=0)
    split_idx_2 = np.cumsum(det_per_batch_2, axis=0)

    boxes_0 = []
    boxes_1 = []
    boxes_2 = []
    start = 0
    for end in split_idx_0:
        boxes_0.append(detections_0[start: end])
        start = end
    start = 0
    for end in split_idx_1:
        boxes_1.append(detections_1[start: end])
        start = end
    start = 0
    for end in split_idx_2:
        boxes_2.append(detections_2[start: end])
        start = end

    return boxes_0, boxes_1, boxes_2


def tensor_to_brambox(boxes_0, boxes_1, boxes_2, input_shape, labels):
    """tensor_to_brambox"""
    converted_boxes_0 = []
    converted_boxes_1 = []
    converted_boxes_2 = []

    for box in boxes_0:
        if box.size == 0:
            converted_boxes_0.append([])
        else:
            converted_boxes_0.append(convert_tensor_to_brambox(box, input_shape[0], input_shape[1], labels))

    for box in boxes_1:
        if box.size == 0:
            converted_boxes_1.append([])
        else:
            converted_boxes_1.append(convert_tensor_to_brambox(box, input_shape[0], input_shape[1], labels))

    for box in boxes_2:
        if box.size == 0:
            converted_boxes_2.append([])
        else:
            converted_boxes_2.append(convert_tensor_to_brambox(box, input_shape[0], input_shape[1], labels))

    return converted_boxes_0, converted_boxes_1, converted_boxes_2


def convert_tensor_to_brambox(boxes, width, height, class_label_map):
    """convert_tensor_to_brambox"""
    boxes[:, 0:3:2] = boxes[:, 0:3:2] * width
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1:4:2] = boxes[:, 1:4:2] * height
    boxes[:, 1] -= boxes[:, 3] / 2

    brambox = []
    for box in boxes:
        det = Detection()

        det.x_top_left = box[0]
        det.y_top_left = box[1]
        det.width = box[2]
        det.height = box[3]
        det.confidence = box[4]
        if class_label_map is not None:
            det.class_label = class_label_map[int(box[5])]
        else:
            det.class_label = str(int(box[5]))

        brambox.append(det)

    return brambox


def parse_gt_from_anno(img_anno, classes):
    """parse_gt_from_anno"""
    print('parse ground truth files...')
    ground_truth = {}

    for img_name, annos in img_anno.items():
        objs = []
        for anno in annos:
            if anno[1] == 0. and anno[2] == 0. and anno[3] == 0. and anno[4] == 0.:
                continue
            if int(anno[0]) == -1:
                continue
            xmin = anno[1]
            ymin = anno[2]
            xmax = xmin + anno[3] - 1
            ymax = ymin + anno[4] - 1
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            cls = classes[int(anno[0])]
            gt_box = {'class': cls, 'box': [xmin, ymin, xmax, ymax]}
            objs.append(gt_box)
        ground_truth[img_name] = objs

    return ground_truth


def parse_rets(ret_files_set):
    """parse_rets"""
    print('parse ret files...')
    ret_list = {}
    for cls in ret_files_set:
        ret_file = open(ret_files_set[cls])
        ret_list[cls] = []
        for line in ret_file.readlines():
            info = line.strip().split()
            img_name = info[0]
            scole = float(info[1])
            xmin = float(info[2])
            ymin = float(info[3])
            xmax = float(info[4])
            ymax = float(info[5])
            ret_list[cls].append({'img_name': img_name, 'scole': scole, 'ret': [xmin, ymin, xmax, ymax]})
    return ret_list


def calc_recall_precision_ap(ground_truth, ret_list, iou_thr=0.5):
    """
        calc_recall_precision_ap
    """
    print('calculate [recall | persicion | ap]...')
    evaluate = {}
    for cls in ret_list:
        ret = ret_list[cls]
        n_gt_obj = calc_gt_count(ground_truth, cls)
        print('class [%s] ground truth:%d' % (cls, n_gt_obj))

        ret = sorted(ret, key=lambda ret: ret['scole'], reverse=True)
        tp = np.zeros(len(ret))
        fp = np.zeros(len(ret))

        for ret_idx, info in enumerate(ret):
            img_name = info['img_name']
            if img_name not in ground_truth:
                print('%s not in ground truth' % img_name)
                continue
            else:
                img_gts = ground_truth[img_name]
                max_iou = 0
                max_idx = -1

                for idx, gt in enumerate(img_gts):
                    if (not gt['class'] == cls) or 'used' in gt:
                        continue
                    iou = calc_iou(info['ret'], gt['box'])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx
                if max_iou > iou_thr:
                    tp[ret_idx] = 1
                    img_gts[max_idx]['used'] = 1
                else:
                    fp[ret_idx] = 1

        tp = tp.cumsum()
        fp = fp.cumsum()

        recall = tp / n_gt_obj
        precision = tp / (tp + fp)
        ap = cal_ap_voc2012(recall, precision)
        evaluate[cls] = {'recall': recall, 'precision': precision, 'ap': ap}

    return evaluate


def calc_gt_count(gt_set, cls):
    """
        Calculate the number of gt_set equals cls
    """
    count = 0
    for img in gt_set:
        for obj in gt_set[img]:
            if obj['class'] == cls:
                count += 1
    return count


def cal_ap_voc2012(recall, precision):
    """
        cal_ap_voc2012
    """
    ap_val = 0.0
    eps = 1e-6
    assert len(recall) == len(precision)
    length = len(recall)
    cur_prec = precision[length - 1]
    cur_rec = recall[length - 1]

    for i in range(0, length - 1)[::-1]:
        cur_prec = max(precision[i], cur_prec)
        if abs(recall[i] - cur_rec) > eps:
            ap_val += cur_prec * abs(recall[i] - cur_rec)

        cur_rec = recall[i]

    return ap_val


def calc_iou(rect1, rect2):
    """
        Calculate iou
    """
    bd_i = (max(rect1[0], rect2[0]), max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]), min(rect1[3], rect2[3]))
    iw = bd_i[2] - bd_i[0] + 0.001
    ih = bd_i[3] - bd_i[1] + 0.001
    iou = 0
    if iw > 0 and ih > 0:
        ua = calc_rect_area(rect1) + calc_rect_area(rect2) - iw * ih
        iou = iw * ih / ua
    return iou


def calc_rect_area(rect):
    """
        calc_rect_area
    """
    return (rect[2] - rect[0] + 0.001) * (rect[3] - rect[1] + 0.001)


def prepare_file_paths(dataset_path):
    """
        prepare_file_paths
    """
    image_files = []
    anno_files = []
    image_names = []
    if not os.path.isdir(dataset_path):
        print("dataset_train root is invalid!")
        exit()
    anno_dir = os.path.join(dataset_path, "Annotations")
    image_dir = os.path.join(dataset_path, "JPEGImages")
    valid_txt = os.path.join(dataset_path, "ImageSets/Main/val.txt")

    ret_image_files, ret_anno_files, ret_image_names = filter_valid_files_by_txt(image_dir, anno_dir, valid_txt)
    image_files.extend(ret_image_files)
    anno_files.extend(ret_anno_files)
    image_names.extend(ret_image_names)
    return image_files, anno_files, image_names


def filter_valid_files_by_txt(image_dir, anno_dir, valid_txt):
    """
        filter_valid_files_by_txt
    """
    with open(valid_txt, "r") as txt:
        valid_names = txt.readlines()
    image_files = []
    anno_files = []
    image_names = []
    for name in valid_names:
        strip_name = name.strip("\n")
        anno_joint_path = os.path.join(anno_dir, strip_name + ".xml")
        if os.path.isfile(anno_joint_path):
            image_joint_path = os.path.join(image_dir, strip_name + ".jpg")
            image_name = image_joint_path.split('/')[-1].replace('.jpg', '')
            if os.path.isfile(image_joint_path):
                image_files.append(image_joint_path)
                anno_files.append(anno_joint_path)
                image_names.append(image_name)
                continue
            image_joint_path = os.path.join(image_dir, strip_name + ".png")
            image_name = image_joint_path.split('/')[-1].replace('.png', '')
            if os.path.isfile(image_joint_path):
                image_files.append(image_joint_path)
                anno_files.append(anno_joint_path)
                image_names.append(image_name)
    return image_files, anno_files, image_names


def get_data(image_file, anno_file, image_name):
    """
        get_data
    """
    count = 0
    annotation = []
    tree = ET.parse(anno_file)
    root = tree.getroot()
    class_indexing_1 = {'face': 0}

    with Image.open(image_file) as fd:
        orig_width, orig_height = fd.size

    with open(image_file, 'rb') as f:
        img = f.read()

    for member in root.findall('object'):
        anno = deserialize(member, class_indexing_1)
        if anno is not None:
            annotation.extend(anno)
            count += 1

    if count == 0:
        annotation_res = np.array([[-1, -1, -1, -1, -1, -1]], dtype='float64')
        count = 1
    else:
        annotation_res = np.zeros((200, 6), dtype='float64')
        index = 0
        for j in range(0, annotation_res.shape[0]):
            for k in range(0, annotation_res.shape[1]):
                if index < len(annotation):
                    annotation_res[j][k] = annotation[index]
                    index += 1
                else:
                    break
    data = {
        "image": img,
        "annotation": annotation_res,
        "image_name": image_name,
        "image_size": np.array([[orig_width, orig_height]], dtype='int32')
    }
    return data


def deserialize(member, class_indexing):
    """
        deserialize
    """
    class_name = member[0].text
    if class_name in class_indexing:
        class_num = class_indexing[class_name]
    else:
        return None
    bnx = member.find('bndbox')
    box_x_min = float(bnx.find('xmin').text)
    box_y_min = float(bnx.find('ymin').text)
    box_x_max = float(bnx.find('xmax').text)
    box_y_max = float(bnx.find('ymax').text)
    width = float(box_x_max - box_x_min + 1)
    height = float(box_y_max - box_y_min + 1)

    # try:
    #     ignore = float(member.find('ignore').text)
    # except ValueError:
    ignore = 0.0
    return [class_num, box_x_min, box_y_min, width, height, ignore]
