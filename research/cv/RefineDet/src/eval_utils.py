# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""eval metrics utils"""

import json
import xml.etree.ElementTree as et
import os
import numpy as np

def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


def coco_metrics(pred_data, anno_json, config):
    """Calculate mAP of predicted bboxes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    num_classes = config.num_classes

    #Classes need to train or test.
    val_cls = config.classes
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls
    coco_gt = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["name"]] = cat["id"]

    predictions = []
    img_ids = []

    for sample in pred_data:
        pred_boxes = sample['boxes']
        box_scores = sample['box_scores']
        img_id = sample['img_id']
        h, w = sample['image_shape']

        final_boxes = []
        final_label = []
        final_score = []
        img_ids.append(img_id)

        for c in range(1, num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > config.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, config.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
            res['score'] = score
            res['category_id'] = label
            predictions.append(res)
    if not os.path.exists('./eval_out'):
        os.makedirs('./eval_out')
    with open('./eval_out/predictions.json', 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes('./eval_out/predictions.json')
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats[0]


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = et.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def voc_metrics(pred_data, annojson, config, use_07=True):
    """calc voc ap"""
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    aps = voc_eval(pred_data, config, ovthresh=0.5, use_07_metric=use_07_metric)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    return np.mean(aps)


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_pred_process(pred_data, val_cls, recs):
    """process pred data for voc"""
    num_classes = config.num_classes
    cls_img_ids = {}
    cls_bboxes = {}
    cls_scores = {}
    classes = {}
    cls_npos = {}
    for cls in val_cls:
        if cls == 'background':
            continue
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == cls]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
        cls_npos[cls] = npos
        classes[cls] = class_recs
        cls_img_ids[cls] = []
        cls_bboxes[cls] = []
        cls_scores[cls] = []

    for sample in pred_data:
        pred_boxes = sample['boxes']
        box_scores = sample['box_scores']
        img_id = sample['img_id']
        h, w = sample['image_shape']

        final_boxes = []
        final_label = []
        final_score = []

        for c in range(1, num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > config.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, config.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [c] * len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            cls_img_ids[val_cls[label]].append(img_id)
            cls_bboxes[val_cls[label]].append([loc[1], loc[0], loc[3], loc[2]])
            cls_scores[val_cls[label]].append(score)
    return classes, cls_img_ids, cls_bboxes, cls_scores, cls_npos

def voc_eval(pred_data, config, ovthresh=0.5, use_07_metric=False):
    """VOC metric utils"""
    # first load gt
    # load annots
    print("Create VOC label")
    val_cls = config.classes
    voc_root = config.voc_root
    sub_dir = 'eval'
    voc_dir = os.path.join(voc_root, sub_dir)
    if not os.path.isdir(voc_dir):
        raise ValueError(f'Cannot find {sub_dir} dataset path.')

    image_dir = anno_dir = voc_dir
    if os.path.isdir(os.path.join(voc_dir, 'Images')):
        image_dir = os.path.join(voc_dir, 'Images')
    if os.path.isdir(os.path.join(voc_dir, 'Annotations')):
        anno_dir = os.path.join(voc_dir, 'Annotations')
    print("finding dir ", image_dir, anno_dir)
    imagenames = []
    image_paths = []
    for anno_file in os.listdir(anno_dir):
        if not anno_file.endswith('xml'):
            continue
        tree = et.parse(os.path.join(anno_dir, anno_file))
        root_node = tree.getroot()
        file_name = root_node.find('filename').text
        imagenames.append(int(file_name[:-4]))
        image_paths.append(os.path.join(anno_dir, anno_file))

    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(image_paths[i])

    # extract gt objects for this class
    classes = {}
    cls_img_ids = {}
    cls_bboxes = {}
    cls_scores = {}
    cls_npos = {}
    #pred data
    classes, cls_img_ids, cls_bboxes, cls_scores, cls_npos = voc_pred_process(pred_data, val_cls, recs)
    aps = []
    for cls in val_cls:
        if cls == 'background':
            continue
        npos = cls_npos[cls]
        class_recs = classes[cls]
        image_ids = cls_img_ids[cls]
        confidence = np.array(cls_scores[cls])
        BB = np.array(cls_bboxes[cls])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        aps.append(ap)
    return np.array(aps)
