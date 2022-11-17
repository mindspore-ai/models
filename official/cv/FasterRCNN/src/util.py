# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""coco eval for fasterrcnn"""

import json
import os
import csv
import shutil
import numpy as np
import mmcv
from pycocotools.coco import COCO
from src.detecteval import DetectEval

_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}


def write_list_to_csv(file_path, data_to_write, append=False):
    # print('Saving data into file [{}]...'.format(file_path))
    if append:
        open_mode = 'a'
    else:
        open_mode = 'w'
    with open(file_path, open_mode) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_to_write)


def coco_eval(config, result_files, result_types, coco, max_dets=(100, 300, 1000), single_result=False,
              plot_detect_result=False):
    """coco eval for fasterrcnn"""
    anns = json.load(open(result_files['bbox']))
    if not anns:
        return summary_init

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = DetectEval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result:
            res_dict = dict()
            for id_i in tgt_ids:
                cocoEval = DetectEval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.params.maxDets = list(max_dets)

                cocoEval.params.imgIds = [id_i]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                res_dict.update({coco.imgs[id_i]['file_name']: cocoEval.stats[1]})

        cocoEval = DetectEval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.params.imgIds = tgt_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        summary_metrics = {
            'Precision/mAP': cocoEval.stats[0],
            'Precision/mAP@.50IOU': cocoEval.stats[1],
            'Precision/mAP@.75IOU': cocoEval.stats[2],
            'Precision/mAP (small)': cocoEval.stats[3],
            'Precision/mAP (medium)': cocoEval.stats[4],
            'Precision/mAP (large)': cocoEval.stats[5],
            'Recall/AR@1': cocoEval.stats[6],
            'Recall/AR@10': cocoEval.stats[7],
            'Recall/AR@100': cocoEval.stats[8],
            'Recall/AR@100 (small)': cocoEval.stats[9],
            'Recall/AR@100 (medium)': cocoEval.stats[10],
            'Recall/AR@100 (large)': cocoEval.stats[11],
        }

        print("summary_metrics: ")
        print(summary_metrics)

        if plot_detect_result:
            res = calcuate_pr_rc_f1(config, coco, coco_dets, tgt_ids, iou_type)

    if plot_detect_result:
        return res

    return summary_metrics


def calcuate_pr_rc_f1(config, coco, coco_dets, tgt_ids, iou_type):
    cocoEval = DetectEval(coco, coco_dets, iou_type)
    cocoEval.params.imgIds = tgt_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats_all = cocoEval.stats

    eval_result_path = os.path.abspath("./eval_result")
    if os.path.exists(eval_result_path):
        shutil.rmtree(eval_result_path)
    os.mkdir(eval_result_path)

    result_csv = os.path.join(eval_result_path, "statistics.csv")
    eval_item = ["ap@0.5:0.95", "ap@0.5", "ap@0.75", "ar@0.5:0.95", "ar@0.5", "ar@0.75"]
    write_list_to_csv(result_csv, eval_item, append=False)
    eval_result = [round(stats_all[0], 3), round(stats_all[1], 3), round(stats_all[2], 3), round(stats_all[6], 3),
                   round(stats_all[7], 3), round(stats_all[8], 3)]
    write_list_to_csv(result_csv, eval_result, append=True)
    write_list_to_csv(result_csv, [], append=True)
    # 1.2 plot_pr_curve
    cocoEval.plot_pr_curve(eval_result_path)

    # 2
    E = DetectEval(coco, coco_dets, iou_type)
    E.params.iouThrs = [0.5]
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.evaluate()
    # 2.1 plot hist_curve of every class's tp's confidence and fp's confidence
    confidence_dict = E.compute_tp_fp_confidence()
    E.plot_hist_curve(confidence_dict, eval_result_path)

    # 2.2 write best_threshold and p r to csv and plot
    cat_pr_dict, cat_pr_dict_origin = E.compute_precison_recall_f1()
    # E.write_best_confidence_threshold(cat_pr_dict, cat_pr_dict_origin, eval_result_path)
    best_confidence_thres = E.write_best_confidence_threshold(cat_pr_dict, cat_pr_dict_origin, eval_result_path)
    print("best_confidence_thres: ", best_confidence_thres)
    E.plot_mc_curve(cat_pr_dict, eval_result_path)

    # 3
    # 3.1 compute every class's p r and save every class's p and r at iou = 0.5
    E = DetectEval(coco, coco_dets, iouType='bbox')
    E.params.iouThrs = [0.5]
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.evaluate()
    E.accumulate()
    result = E.evaluate_every_class()
    print_info = ["class_name", "tp_num", "gt_num", "dt_num", "precision", "recall"]
    write_list_to_csv(result_csv, print_info, append=True)
    print("class_name", "tp_num", "gt_num", "dt_num", "precision", "recall")
    for class_result in result:
        print(class_result)
        write_list_to_csv(result_csv, class_result, append=True)

    # 3.2 save ng / ok images
    E.save_images(config, eval_result_path, 0.5)

    return stats_all[0]


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
    return result


def proposal2json(dataset, results):
    """convert proposal to json mode"""
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = dataset.get_dataset_size() * 2
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    """convert det to json mode"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = len(img_ids)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results): break
        result = results[idx]
        for label, result_label in enumerate(result):
            bboxes = result_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    """convert segm to json mode"""
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label, det_label in enumerate(det):
            # bbox results
            bboxes = det_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(dataset, results, out_file):
    """convert result convert to json mode"""
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files
