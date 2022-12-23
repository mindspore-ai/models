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
# This file refers to the project https://github.com/MhLiao/DB.git
"""DBNet metric tools"""
import numpy as np
import cv2
from shapely.geometry import Polygon


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DetectionIoUEvaluator:
    def __init__(self, is_output_polygon=False, iou_constraint=0.5, area_precision_constraint=0.5):
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        # num of 'pred' & 'gt' matching
        det_match = 0
        pairs = []
        iou_mat = np.empty([1, 1])

        # num of cared polys
        num_gt_care = 0
        num_det_care = 0

        # list of polys
        gt_polys = []
        det_polys = []

        # idx of dontcare polys
        gt_polys_dontcare = []
        det_polys_dontcare = []

        # log string
        evaluation_log = ""

        ## gt
        for i in range(len(gt)):
            poly = gt[i]['polys']
            dontcare = gt[i]['dontcare']
            if not Polygon(poly).is_valid or not Polygon(poly).is_simple:
                continue
            gt_polys.append(poly)
            if dontcare:
                gt_polys_dontcare.append(len(gt_polys) - 1)

        evaluation_log += f"GT polygons: {str(len(gt_polys))}" + \
        (f" ({len(gt_polys_dontcare)} don't care)\n" if gt_polys_dontcare else "\n")

        for i in range(len(pred)):
            poly = pred[i]
            if not Polygon(poly).is_valid or not Polygon(poly).is_simple:
                continue
            det_polys.append(poly)

            # For dontcare gt
            if gt_polys_dontcare:
                for idx in gt_polys_dontcare:
                    dontcare_poly = gt_polys[idx]
                    intersected_area = get_intersection(dontcare_poly, poly)
                    poly_area = Polygon(poly).area
                    precision = 0 if poly_area == 0 else intersected_area / poly_area
                    # If precision enough, append as dontcare det.
                    if precision > self.area_precision_constraint:
                        det_polys_dontcare.append(len(det_polys) - 1)
                        break

        evaluation_log += f"DET polygons: {len(det_polys)}" + \
        (f" ({len(det_polys_dontcare)} don't care)\n" if det_polys_dontcare else "\n")

        if gt_polys and det_polys:
            # visit arrays
            iou_mat = np.empty([len(gt_polys), len(det_polys)])
            gt_rect_mat = np.zeros(len(gt_polys), np.int8)
            det_rect_mat = np.zeros(len(det_polys), np.int8)

            # IoU
            if self.is_output_polygon:
                # polygon
                for gt_idx in range(len(gt_polys)):
                    for det_idx in range(len(det_polys)):
                        pG = gt_polys[gt_idx]
                        pD = det_polys[det_idx]
                        iou_mat[gt_idx, det_idx] = get_intersection_over_union(pD, pG)
            else:
                # rectangle
                for gt_idx in range(len(gt_polys)):
                    for det_idx in range(len(det_polys)):
                        pG = np.float32(gt_polys[gt_idx])
                        pD = np.float32(det_polys[det_idx])
                        iou_mat[gt_idx, det_idx] = self.iou_rotate(pD, pG)

            for gt_idx in range(len(gt_polys)):
                for det_idx in range(len(det_polys)):
                    # If IoU == 0 and care
                    if gt_rect_mat[gt_idx] == 0 and det_rect_mat[det_idx] == 0 \
                    and (gt_idx not in gt_polys_dontcare) and (det_idx not in det_polys_dontcare):
                        # If IoU enough
                        if iou_mat[gt_idx, det_idx] > self.iou_constraint:
                            # Mark the visit arrays
                            gt_rect_mat[gt_idx] = 1
                            det_rect_mat[det_idx] = 1
                            det_match += 1
                            pairs.append({'gt': gt_idx, 'det': det_idx})
                            evaluation_log += f"Match GT #{gt_idx} with Det #{det_idx}\n"

        ## summary
        num_gt_care += (len(gt_polys) - len(gt_polys_dontcare))
        num_det_care += (len(det_polys) - len(det_polys_dontcare))

        if num_gt_care == 0:
            recall = 1.0
            precision = 0.0 if num_det_care > 0 else 1.0
        else:
            recall = float(det_match) / num_gt_care
            precision = 0 if num_det_care == 0 else float(
                det_match) / num_det_care
        hmean = 0 if (precision + recall) == 0 else \
                2.0 * precision * recall / (precision + recall)

        metric = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iou_mat': [] if len(det_polys) > 100 else iou_mat.tolist(),
            'gt_polys': gt_polys,
            'det_polys': det_polys,
            'gt_care_num': num_gt_care,
            'det_care_num': num_det_care,
            'gt_dont_care': gt_polys_dontcare,
            'det_dont_care': det_polys_dontcare,
            'det_matched': det_match,
            'evaluation_log': evaluation_log
        }
        return metric

    def combine_results(self, results):
        num_global_care_gt = 0
        num_global_care_det = 0
        matched_sum = 0
        for result in results:
            num_global_care_gt += result['gt_care_num']
            num_global_care_det += result['det_care_num']
            matched_sum += result['det_matched']

        method_recall = 0 if num_global_care_gt == 0 else float(
            matched_sum) / num_global_care_gt
        method_precision = 0 if num_global_care_det == 0 else float(
            matched_sum) / num_global_care_det
        methodHmean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / \
                                                                      (method_recall + method_precision)
        method_metrics = {'precision': method_precision,
                          'recall': method_recall, 'hmean': methodHmean}
        return method_metrics

    @staticmethod
    def iou_rotate(box_a, box_b, method='union'):
        rect_a = cv2.minAreaRect(box_a)
        rect_b = cv2.minAreaRect(box_b)
        r1 = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if r1[0] == 0:
            return 0
        inter_area = cv2.contourArea(r1[1])
        area_a = cv2.contourArea(box_a)
        area_b = cv2.contourArea(box_b)
        union_area = area_a + area_b - inter_area
        if union_area == 0 or inter_area == 0:
            return 0
        if method == 'union':
            iou = inter_area / union_area
        elif method == 'intersection':
            iou = inter_area / min(area_a, area_b)
        else:
            raise NotImplementedError
        return iou


class QuadMetric:
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon)

    def measure(self, batch, output, box_thresh=0.7):
        '''
        batch: (image, polygons, ignore_tags)
            image: numpy array of shape (N, C, H, W).
            polys: numpy array of shape (N, K, 4, 2), the polygons of objective regions.
            dontcare: numpy array of shape (N, K), indicates whether a region is ignorable or not.
        output: (polygons, ...)
        '''
        gt_polys = batch['polys'].astype(np.float32)
        gt_dontcare = batch['dontcare']
        pred_polys = np.array(output[0])
        pred_scores = np.array(output[1])

        # Loop i for every batch
        for i in range(len(gt_polys)):
            gt = [{'polys': gt_polys[i][j], 'dontcare': gt_dontcare[i][j]}
                  for j in range(len(gt_polys[i]))]
            if self.is_output_polygon:
                pred = [pred_polys[i][j] for j in range(len(pred_polys[i]))]
            else:
                pred = [pred_polys[i][j, :, :].astype(np.int32)
                        for j in range(pred_polys[i].shape[0]) if pred_scores[i][j] >= box_thresh]
        return self.evaluator.evaluate_image(gt, pred)


    def validate_measure(self, batch, output):
        return self.measure(batch, output, box_thresh=0.55)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics for image_metrics in raw_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }
