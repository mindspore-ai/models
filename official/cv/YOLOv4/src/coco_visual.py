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
import sys
import json
import shutil
import csv
import warnings
from typing import List
import cv2
import mmcv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

plt.switch_backend('agg')
warnings.filterwarnings("ignore")


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class CocoDraw:
    def __init__(self):
        self.COLOR_MAP = [
            (0, 255, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 128, 128),
            (0, 128, 0),
            (128, 0, 0),
            (0, 0, 128),
            (128, 128, 0),
            (128, 0, 128),
        ]

    def write_list_to_csv(self, file_path, data_to_write, append=False):
        print('Saving data into file [{}]...'.format(file_path))
        if append:
            open_mode = 'a'
        else:
            open_mode = 'w'
        with open(file_path, open_mode) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_to_write)

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return False, None
        return True, image

    def save_image(self, image_path, image):
        return cv2.imwrite(image_path, image)

    def draw_rectangle(self, image, pt1, pt2, label=None):
        if label is not None:
            map_index = label % len(self.COLOR_MAP)
            color = self.COLOR_MAP[map_index]
        else:
            color = self.COLOR_MAP[0]
        thickness = 5
        cv2.rectangle(image, pt1, pt2, color, thickness)

    def draw_text(self, image, text, org, label=None):
        if label is not None:
            map_index = label % len(self.COLOR_MAP)
            color = self.COLOR_MAP[map_index]
        else:
            color = self.COLOR_MAP[0]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        cv2.putText(image, text, org, font_face, font_scale, color, thickness)

    def draw_one_box(self, image, label, box, cat_id, line_thickness=None):
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        if cat_id is not None:
            map_index = cat_id % len(self.COLOR_MAP)
            color = self.COLOR_MAP[map_index]
        else:
            color = self.COLOR_MAP[0]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf // 2)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 6, [255, 255, 255], thickness=tf // 2,
                    lineType=cv2.LINE_AA)


class DetectEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType="bbox"):
        assert iouType == "bbox", "iouType only supported bbox"
        self.draw_tool = CocoDraw()
        super().__init__(cocoGt, cocoDt, iouType)
        if self.cocoGt is not None:
            cat_infos = cocoGt.loadCats(cocoGt.getCatIds())
            self.params.labels = {}
            # self.params.labels = ["" for i in range(len(self.params.catIds))]
            for cat in cat_infos:
                self.params.labels[cat["id"]] = cat["name"]

    # add new
    def catId_summarize(self, catId, iouThr=None, areaRng="all", maxDets=100):
        p = self.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        s = self.eval["recall"]
        if iouThr is not None:
            iou = np.where(iouThr == p.iouThrs)[0]
            s = s[iou]

        if isinstance(catId, int):
            s = s[:, catId, aind, mind]
        else:
            s = s[:, :, aind, mind]

        not_empty = len(s[s > -1]) == 0
        if not_empty:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def compute_gt_dt_num(self):
        p = self.params
        catIds_gt_num = {}
        catIds_dt_num = {}

        for ids in p.catIds:
            gts_cat_id = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(catIds=[ids]))
            dts_cat_id = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(catIds=[ids]))
            catIds_gt_num[ids] = len(gts_cat_id)
            catIds_dt_num[ids] = len(dts_cat_id)

        return catIds_gt_num, catIds_dt_num

    def evaluate_ok_ng(self, img_id, catIds, iou_threshold=0.5):
        """
        evaluate every if this image is ok、precision_ng、recall_ng
        img_id: int
        cat_ids:list
        iou_threshold:int
        """
        p = self.params
        img_id = int(img_id)

        # Save the results of precision_ng and recall_ng for each category on a picture
        cat_id_result = {}
        for cat_id in catIds:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
            ious = self.computeIoU(img_id, cat_id)

            # Sort dt in descending order, and only take the first 100
            inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in inds]

            # p.maxDets must be set in ascending order
            if len(dt) > p.maxDets[-1]:
                dt = dt[0:p.maxDets[-1]]

            # The first case: gt, dt are both 0: skip
            if not gt and not dt:
                cat_id_result[cat_id] = (False, False)
                continue
            # The second case: gt = 0, dt !=0: precision_ng
            if not gt and dt:
                cat_id_result[cat_id] = (True, False)
                continue
            # The third case: gt != 0, dt = 0: recall_ng
            if gt and not dt:
                cat_id_result[cat_id] = (False, True)
                continue
            # The fourth case: gt and dt are matched in pairs
            gtm = [0] * len(gt)
            dtm = [0] * len(dt)

            for dind in range(len(dt)):
                # dt:[a] gt [b] ious = [a*b]
                iou = min([iou_threshold, 1 - 1e-10])
                # m records the position of the gt with the best match
                m = -1
                for gind in range(len(gt)):
                    # If gt[gind] already matches, skip it.
                    if gtm[gind] > 0:
                        continue
                    # If the iou(dind, gind) is less than the threshold, traverse
                    if ious[dind, gind] < iou:
                        continue
                    iou = ious[dind, gind]
                    m = gind
                if m == -1:
                    continue
                dtm[dind] = 1
                gtm[m] = 1

            # If gt is all matched, gtm is all 1
            precision_ng = sum(dtm) < len(dtm)
            recall_ng = sum(gtm) < len(gtm)
            cat_id_result[cat_id] = (precision_ng, recall_ng)

        # As long as the precision_ng in a class is True, the picture is precision_ng, and recall_ng is the same
        # Subsequent development of NG pictures for each category can be saved
        precision_result = False
        recall_result = False
        for ng in cat_id_result.values():
            precision_ng = ng[0]
            recall_ng = ng[1]
            if precision_ng:
                precision_result = precision_ng
            if recall_ng:
                recall_result = recall_ng
        return precision_result, recall_result

    def evaluate_every_class(self):
        """
        compute every class's:
        [label, tp_num, gt_num, dt_num, precision, recall]
        """
        print("Evaluate every class's predision and recall")
        p = self.params
        cat_ids = p.catIds
        labels = p.labels
        result = []
        catIds_gt_num, catIds_dt_num = self.compute_gt_dt_num()
        sum_gt_num = 0
        sum_dt_num = 0
        for value in catIds_gt_num.values():
            sum_gt_num += value
        for value in catIds_dt_num.values():
            sum_dt_num += value
        sum_tp_num = 0

        for i, cat_id in enumerate(cat_ids):
            # Here is hard-coded
            stats = self.catId_summarize(catId=i)
            recall = stats
            gt_num = catIds_gt_num[cat_id]
            tp_num = recall * gt_num
            sum_tp_num += tp_num
            dt_num = catIds_dt_num[cat_id]
            if dt_num <= 0:
                if gt_num == 0:
                    precision = -1
                else:
                    precision = 0
            else:
                precision = tp_num / dt_num
            label = labels[cat_id]
            class_result = [label, int(round(tp_num)), gt_num, int(round(dt_num)), round(precision, 3),
                            round(recall, 3)]
            result.append(class_result)
        all_precision = sum_tp_num / sum_dt_num
        all_recall = sum_tp_num / sum_gt_num
        all_result = ["all", int(round(sum_tp_num)), sum_gt_num, int(round(sum_dt_num)), round(all_precision, 3),
                      round(all_recall, 3)]
        result.append(all_result)
        return result

    def plot_pr_curve(self, eval_result_path):

        """
        precisions[T, R, K, A, M]
        T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
        R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
        K: category, idx from 0 to ...
        A: area range, (all, small, medium, large), idx from 0 to 3
        M: max dets, (1, 10, 100), idx from 0 to 2
        """
        print("Plot pr curve about every class")
        precisions = self.eval["precision"]
        p = self.params
        cat_ids = p.catIds
        labels = p.labels

        pr_dir = os.path.join(eval_result_path, "./pr_curve_image")
        if not os.path.exists(pr_dir):
            os.mkdir(pr_dir)

        for i, cat_id in enumerate(cat_ids):
            pr_array1 = precisions[0, :, i, 0, 2]  # iou = 0.5
            x = np.arange(0.0, 1.01, 0.01)
            # plot PR curve
            plt.plot(x, pr_array1, label="iou=0.5," + labels[cat_id])
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.legend(loc="lower left")
            plt_path = os.path.join(pr_dir, "pr_curve_" + labels[cat_id] + ".png")
            plt.savefig(plt_path)
            plt.close(1)

    def save_anns_images(self, config, im_path_dir, score_threshold=None):
        """
        save annotations images
        """
        print("saving eval images")
        p = self.params
        img_ids = p.imgIds
        labels = p.labels

        dt = self.cocoDt.getAnnIds()
        dts = self.cocoDt.loadAnns(dt)

        eval_result_path = os.path.abspath("./eval_image")
        if os.path.exists(eval_result_path):
            shutil.rmtree(eval_result_path)
        os.mkdir(eval_result_path)

        for img_id in img_ids:
            img_id = int(img_id)
            img_info = self.cocoGt.loadImgs(img_id)
            assert config.dataset in ("coco", "voc")
            im_path = os.path.join(im_path_dir, img_info[0]['file_name'])
            # output image path
            im_path_out = os.path.join(eval_result_path, img_info[0]['file_name'])
            success, image = self.draw_tool.read_image(im_path)
            assert success

            for obj in dts:
                _id = obj["image_id"]
                if _id == img_id:
                    bbox = obj["bbox"]
                    score = obj["score"]
                    if score_threshold is not None and score < float(score_threshold):
                        continue
                    category_id = obj["category_id"]
                    label = labels[category_id]

                    xmin = int(bbox[0])
                    ymin = int(bbox[1])
                    width = int(bbox[2])
                    height = int(bbox[3])
                    xmax = xmin + width
                    ymax = ymin + height

                    label = label + " " + str(round(score, 3))
                    self.draw_tool.draw_one_box(image, label, (xmin, ymin, xmax, ymax), category_id)

            self.draw_tool.save_image(im_path_out, image)
        print("save images completed")

    def save_category_images(self, config, eval_result_path, im_path_dir, iou_threshold=0.5):
        """
        save ok_images, precision_ng_images, recall_ng_images
        Arguments:
            config: dict, config about parameters
            eval_result_path: str, path to save images
            iou_threshold: int, iou_threshold
        """
        print("Saving images of ok ng")
        p = self.params
        img_ids = p.imgIds
        cat_ids = p.catIds if p.useCats else [-1]  # list: [0,1,2,3....]
        labels = p.labels

        dt = self.cocoDt.getAnnIds()
        dts = self.cocoDt.loadAnns(dt)

        for img_id in img_ids:
            img_id = int(img_id)
            img_info = self.cocoGt.loadImgs(img_id)
            assert config.dataset in ("coco", "voc")

            # Return whether the image is precision_ng or recall_ng
            precision_ng, recall_ng = self.evaluate_ok_ng(img_id, cat_ids, iou_threshold)
            # Save to ok_images
            if not precision_ng and not recall_ng:
                # origin image path
                im_path = os.path.join(im_path_dir, img_info[0]['file_name'])
                # output image path
                im_path_out_dir = os.path.join(eval_result_path, 'ok_images')
                if not os.path.exists(im_path_out_dir):
                    os.makedirs(im_path_out_dir)
                im_path_out = os.path.join(im_path_out_dir, img_info[0]['file_name'])

                success, image = self.draw_tool.read_image(im_path)
                assert success

                for obj in dts:
                    _id = obj["image_id"]
                    if _id == img_id:
                        bbox = obj["bbox"]
                        score = obj["score"]
                        category_id = obj["category_id"]
                        label = labels[category_id]

                        xmin = int(bbox[0])
                        ymin = int(bbox[1])
                        width = int(bbox[2])
                        height = int(bbox[3])
                        xmax = xmin + width
                        ymax = ymin + height

                        label = label + " " + str(round(score, 3))
                        self.draw_tool.draw_one_box(image, label, (xmin, ymin, xmax, ymax), category_id)
                self.draw_tool.save_image(im_path_out, image)
            else:
                # Save to precision_ng_images
                if precision_ng:
                    # origin image path
                    im_path = os.path.join(im_path_dir, img_info[0]['file_name'])
                    assert os.path.exists(im_path), "{} not exist, please check image directory".format(im_path)
                    # output image path
                    im_path_out_dir = os.path.join(eval_result_path, 'precision_ng_images')
                    if not os.path.exists(im_path_out_dir):
                        os.makedirs(im_path_out_dir)
                    im_path_out = os.path.join(im_path_out_dir, img_info[0]['file_name'])

                    success, image = self.draw_tool.read_image(im_path)
                    assert success

                    for obj in dts:
                        _id = obj["image_id"]
                        if _id == img_id:
                            bbox = obj["bbox"]
                            score = obj["score"]
                            category_id = obj["category_id"]
                            label = labels[category_id]

                            xmin = int(bbox[0])
                            ymin = int(bbox[1])
                            width = int(bbox[2])
                            height = int(bbox[3])
                            xmax = xmin + width
                            ymax = ymin + height

                            label = label + " " + str(round(score, 3))
                            self.draw_tool.draw_one_box(image, label, (xmin, ymin, xmax, ymax), category_id)
                    self.draw_tool.save_image(im_path_out, image)

                # Save to recall_ng_images
                if recall_ng:
                    # origin image path
                    im_path = os.path.join(im_path_dir, img_info[0]['file_name'])
                    # output image path
                    im_path_out_dir = os.path.join(eval_result_path, 'recall_ng_images')
                    if not os.path.exists(im_path_out_dir):
                        os.makedirs(im_path_out_dir)

                    im_path_out = os.path.join(im_path_out_dir, img_info[0]['file_name'])
                    success, image = self.draw_tool.read_image(im_path)
                    if not success:
                        raise Exception('Failed reading image from [{}]'.format(im_path))
                    for obj in dts:
                        _id = obj["image_id"]
                        if _id == img_id:
                            bbox = obj["bbox"]
                            score = obj["score"]
                            category_id = obj["category_id"]
                            label = labels[category_id]

                            xmin = int(bbox[0])
                            ymin = int(bbox[1])
                            width = int(bbox[2])
                            height = int(bbox[3])
                            xmax = xmin + width
                            ymax = ymin + height

                            label = label + " " + str(round(score, 3))
                            self.draw_tool.draw_one_box(image, label, (xmin, ymin, xmax, ymax), category_id)
                    self.draw_tool.save_image(im_path_out, image)

    def compute_precison_recall_f1(self, min_score=0.1):
        print('Compute precision, recall, f1...')
        if not self.evalImgs:
            print('Please run evaluate() first')
        p = self.params
        catIds = p.catIds if p.useCats == 1 else [-1]
        labels = p.labels

        assert len(p.maxDets) == 1
        assert len(p.iouThrs) == 1
        assert len(p.areaRng) == 1

        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)]
        m_list = [m for n, m in enumerate(p.maxDets)]
        a_list: List[int] = [n for n, a in enumerate(p.areaRng)]
        i_list = [n for n, i in enumerate(p.imgIds)]
        I0 = len(p.imgIds)
        A0 = len(p.areaRng)

        # cat_pr_dict:
        # {label1:[precision_li, recall_li, f1_li, score_li], label2:[precision_li, recall_li, f1_li, score_li]}
        cat_pr_dict = {}
        cat_pr_dict_origin = {}

        for k0 in k_list:
            Nk = k0 * A0 * I0
            # areagRng
            for a0 in a_list:
                Na = a0 * I0
                # maxDet
                for maxDet in m_list:
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if not E:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    # Ensure that iou has only one value
                    assert (tps.shape[0]) == 1
                    assert (fps.shape[0]) == 1

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    ids = catIds[k0]
                    label = labels[ids]

                    self.calculate_pr_dict(tp_sum, fp_sum, label, npig, dtScoresSorted, cat_pr_dict, cat_pr_dict_origin,
                                           min_score=min_score)
        return cat_pr_dict, cat_pr_dict_origin

    def calculate_pr_dict(self, tp_sum, fp_sum, label, npig, dtScoresSorted, cat_pr_dict, cat_pr_dict_origin,
                          min_score=0.1):
        # iou
        for (tp, fp) in zip(tp_sum, fp_sum):
            tp = np.array(tp)
            fp = np.array(fp)
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))

            f1 = np.divide(2 * (rc * pr), pr + rc, out=np.zeros_like(2 * (rc * pr)), where=pr + rc != 0)

            conf_thres = [int(i) * 0.01 for i in range(10, 100, 10)]
            dtscores_ascend = dtScoresSorted[::-1]
            inds = np.searchsorted(dtscores_ascend, conf_thres, side='left')
            pr_new = [0.0] * len(conf_thres)
            rc_new = [0.0] * len(conf_thres)
            f1_new = [0.0] * len(conf_thres)
            pr_ascend = pr[::-1]
            rc_ascend = rc[::-1]
            f1_ascend = f1[::-1]
            try:
                for i, ind in enumerate(inds):
                    if conf_thres[i] >= min_score:
                        pr_new[i] = pr_ascend[ind]
                        rc_new[i] = rc_ascend[ind]
                        f1_new[i] = f1_ascend[ind]
                    else:
                        pr_new[i] = 0.0
                        rc_new[i] = 0.0
                        f1_new[i] = 0.0
            except IndexError:
                pass
            # Ensure that the second, third, and fourth for loops only enter once
            if label not in cat_pr_dict.keys():
                cat_pr_dict_origin[label] = [pr[::-1], rc[::-1], f1[::-1], dtScoresSorted[::-1]]
                cat_pr_dict[label] = [pr_new, rc_new, f1_new, conf_thres]
            else:
                break

    def compute_tp_fp_confidence(self):
        print('Compute tp and fp confidences')
        if not self.evalImgs:
            print('Please run evaluate() first')
        p = self.params
        catIds = p.catIds if p.useCats == 1 else [-1]
        labels = p.labels

        assert len(p.maxDets) == 1
        assert len(p.iouThrs) == 1
        assert len(p.areaRng) == 1

        # get inds to evaluate
        m_list = [m for n, m in enumerate(p.maxDets)]
        k_list = list(range(len(p.catIds)))
        a_list = list(range(len(p.areaRng)))
        i_list = list(range(len(p.imgIds)))

        I0 = len(p.imgIds)
        A0 = len(p.areaRng)
        # cat_dict
        correct_conf_dict = {}
        incorrect_conf_dict = {}

        for k0 in k_list:
            Nk = k0 * A0 * I0
            # areagRng
            for a0 in a_list:
                Na = a0 * I0
                # maxDet
                for maxDet in m_list:
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if not E:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    # Ensure that iou has only one value
                    assert (tps.shape[0]) == 1
                    assert (fps.shape[0]) == 1

                    tp_inds = np.where(tps)
                    fp_inds = np.where(fps)

                    tp_confidence = dtScoresSorted[tp_inds[1]]
                    fp_confidence = dtScoresSorted[fp_inds[1]]
                    tp_confidence_li = tp_confidence.tolist()
                    fp_confidence_li = fp_confidence.tolist()
                    ids = catIds[k0]
                    label = labels[ids]

                    # Ensure that the second and third for loops only enter once
                    if label not in correct_conf_dict.keys():
                        correct_conf_dict[label] = tp_confidence_li
                    else:
                        print("maxDet:", maxDet, " ", "areagRng:", p.areagRng)
                        break

                    if label not in incorrect_conf_dict.keys():
                        incorrect_conf_dict[label] = fp_confidence_li
                    else:
                        print("maxDet:", maxDet, " ", "areagRng:", p.areagRng)
                        break
        return correct_conf_dict, incorrect_conf_dict

    def write_best_confidence_threshold(self, cat_pr_dict, cat_pr_dict_origin, eval_result_path):
        """
        write best confidence threshold
        """
        print("Write best confidence threshold to csv")
        result_csv = os.path.join(eval_result_path, "best_threshold.csv")
        result = ["cat_name", "best_f1", "best_precision", "best_recall", "best_score"]
        self.draw_tool.write_list_to_csv(result_csv, result, append=False)
        return_result = []
        for cat_name, cat_info in cat_pr_dict.items():
            f1_li = cat_info[2]
            score_li = cat_info[3]
            max_f1 = [f1 for f1 in f1_li if abs(f1 - max(f1_li)) <= 0.001]
            thre_ = [0.003] + [int(i) * 0.001 for i in range(10, 100, 10)] + [0.099]
            # Find the best confidence threshold for 10 levels of confidence thresholds
            if len(max_f1) == 1:
                # max_f1 is on the far right
                if f1_li.index(max_f1) == len(f1_li) - 1:
                    index = f1_li.index(max_f1) - 1
                # max_f1 is in the middle
                elif f1_li.index(max_f1) != len(f1_li) - 1 and f1_li.index(max_f1) != 0:
                    index_a = f1_li.index(max_f1) - 1
                    index_b = f1_li.index(max_f1) + 1
                    if f1_li[index_a] >= f1_li[index_b]:
                        index = index_a
                    else:
                        index = f1_li.index(max_f1)
                # max_f1 is on the far left
                elif f1_li.index(max_f1) == 0:
                    index = f1_li.index(max_f1)

                best_thre = score_li[index]
                # thre_ = [0.003] + [int(i) * 0.001 for i in range(10, 100, 10)] + [0.099]
                second_thre = [best_thre + i for i in thre_]

            elif len(max_f1) > 1:
                thre_pre = [index for (index, value) in enumerate(f1_li) if abs(value - max(f1_li)) <= 0.001]
                best_thre = score_li[thre_pre[int((len(thre_pre) - 1) / 2)]]
                # thre_ = [0.003] + [int(i) * 0.001 for i in range(10, 100, 10)] + [0.099]
                second_thre = [best_thre + i for i in thre_]

            # Reduce the step unit to find the second confidence threshold
            cat_info_origin = cat_pr_dict_origin[cat_name]
            dtscores_ascend = cat_info_origin[3]
            inds = np.searchsorted(dtscores_ascend, second_thre, side='left')

            pr_second = [0] * len(second_thre)
            rc_second = [0] * len(second_thre)
            f1_second = [0] * len(second_thre)

            try:
                for i, ind in enumerate(inds):
                    if ind >= len(cat_info_origin[0]):
                        ind = len(cat_info_origin[0]) - 1
                    pr_second[i] = cat_info_origin[0][ind]
                    rc_second[i] = cat_info_origin[1][ind]
                    f1_second[i] = cat_info_origin[2][ind]
            except IndexError:
                pass

            best_f1 = max(f1_second)
            best_index = f1_second.index(best_f1)
            best_precision = pr_second[best_index]
            best_recall = rc_second[best_index]
            best_score = second_thre[best_index]
            result = [cat_name, best_f1, best_precision, best_recall, best_score]
            return_result.append(result)
            self.draw_tool.write_list_to_csv(result_csv, result, append=True)
        return return_result

    def plot_mc_curve(self, cat_pr_dict, eval_result_path):
        """
        plot matrix-confidence curve
        cat_pr_dict:{"label_name":[precision, recall, f1, score]}
        """
        print('Plot mc curve')
        savefig_path = os.path.join(eval_result_path, 'pr_cofidence_fig')
        if not os.path.exists(savefig_path):
            os.mkdir(savefig_path)

        xlabel = "Confidence"
        ylabel = "Metric"
        for cat_name, cat_info in cat_pr_dict.items():
            precision = [round(p, 3) for p in cat_info[0]]
            recall = [round(r, 3) for r in cat_info[1]]
            f1 = [round(f, 3) for f in cat_info[2]]
            score = [round(s, 3) for s in cat_info[3]]
            plt.figure(figsize=(9, 9))
            gs = gridspec.GridSpec(4, 1)

            plt.subplot(gs[:3, 0])
            # 1.precision-confidence
            plt.plot(score, precision, linewidth=2, color="deepskyblue", label="precision")

            # 2.recall-confidence
            plt.plot(score, recall, linewidth=2, color="limegreen", label="recall")

            # 3.f1-confidence
            plt.plot(score, f1, linewidth=2, color="tomato", label="f1_score")

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(cat_name, fontsize=15)

            plt.xlim(0, 1)
            plt.xticks((np.arange(0, 1, 0.1)))
            plt.ylim(0, 1.10)
            plt.legend(loc="lower left")

            row_name = ["conf_threshold", "precision", "recall", "f1"]
            plt.grid(True)
            plt.subplot(gs[3, 0])
            plt.axis('off')

            colors = ["white", "deepskyblue", "limegreen", "tomato"]
            plt.table(cellText=[score, precision, recall, f1], rowLabels=row_name, loc='center', cellLoc='center',
                      rowLoc='center', rowColours=colors)

            plt.subplots_adjust(left=0.2, bottom=0.2)
            plt.savefig(os.path.join(savefig_path, cat_name) + '.png', dpi=250)

    def plot_hist_curve(self, input_data, eval_result_path):
        correct_conf_dict, incorrect_conf_dict = input_data[0], input_data[1]
        savefig_path = os.path.join(eval_result_path, 'hist_curve_fig')
        if not os.path.exists(savefig_path):
            os.mkdir(savefig_path)
        for l in correct_conf_dict.keys():
            plt.figure(figsize=(7, 7))
            if l in incorrect_conf_dict.keys() and \
                    len(correct_conf_dict[l]) > 1 and \
                    len(incorrect_conf_dict[l]) > 1:
                gs = gridspec.GridSpec(4, 1)
                plt.subplot(gs[:3, 0])
                correct_conf_dict[l].sort()
                correct_conf_dict[l].reverse()
                col_name_correct = ['number', 'mean', 'max', 'min', 'min99%', 'min99.9%']
                col_val_correct = [len(correct_conf_dict[l]),
                                   ('%.2f' % np.mean(correct_conf_dict[l])),
                                   ('%.2f' % max(correct_conf_dict[l])), ('%.2f' % min(correct_conf_dict[l])),
                                   ('%.2f' % correct_conf_dict[l][int(len(correct_conf_dict[l]) * 0.99) - 1]),
                                   ('%.2f' % correct_conf_dict[l][int(len(correct_conf_dict[l]) * 0.999) - 1])]
                sns.set_palette('hls')
                sns.distplot(correct_conf_dict[l], bins=50, kde_kws={'color': 'b', 'lw': 3},
                             hist_kws={'color': 'b', 'alpha': 0.3})
                plt.xlim((0, 1))
                plt.xlabel(l)
                plt.ylabel("numbers")
                ax1 = plt.twinx()
                incorrect_conf_dict[l].sort()
                incorrect_conf_dict[l].reverse()
                col_val_incorrect = [len(incorrect_conf_dict[l]),
                                     ('%.2f' % np.mean(incorrect_conf_dict[l])),
                                     ('%.2f' % max(incorrect_conf_dict[l])), ('%.2f' % min(incorrect_conf_dict[l])),
                                     ('%.2f' % incorrect_conf_dict[l][int(len(incorrect_conf_dict[l]) * 0.99) - 1]),
                                     ('%.2f' % incorrect_conf_dict[l][int(len(incorrect_conf_dict[l]) * 0.999) - 1])]
                sns.distplot(incorrect_conf_dict[l], bins=50, kde_kws={'color': 'r', 'lw': 3},
                             hist_kws={'color': 'r', 'alpha': 0.3}, ax=ax1)
                plt.grid(True)
                plt.subplot(gs[3, 0])
                plt.axis('off')
                row_name = ['', 'correct', 'incorrect']
                table = plt.table(cellText=[col_name_correct, col_val_correct, col_val_incorrect], rowLabels=row_name,
                                  loc='center', cellLoc='center', rowLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                plt.savefig(os.path.join(savefig_path, l) + '.jpg')
            elif len(correct_conf_dict[l]) > 1:
                gs = gridspec.GridSpec(4, 1)
                plt.subplot(gs[:3, 0])
                correct_conf_dict[l].sort()
                correct_conf_dict[l].reverse()
                col_name_correct = ['number', 'mean', 'max', 'min', 'min99%', 'min99.9%']
                col_val_correct = [len(correct_conf_dict[l]),
                                   ('%.4f' % np.mean(correct_conf_dict[l])),
                                   ('%.4f' % max(correct_conf_dict[l])), ('%.2f' % min(correct_conf_dict[l])),
                                   ('%.2f' % correct_conf_dict[l][int(len(correct_conf_dict[l]) * 0.99) - 1]),
                                   ('%.2f' % correct_conf_dict[l][int(len(correct_conf_dict[l]) * 0.999) - 1])]
                sns.set_palette('hls')
                sns.distplot(correct_conf_dict[l], bins=50, kde_kws={'color': 'b', 'lw': 3},
                             hist_kws={'color': 'b', 'alpha': 0.3})
                plt.xlim((0, 1))
                plt.xlabel(l)
                plt.ylabel("numbers")
                plt.grid(True)
                plt.subplot(gs[3, 0])
                plt.axis('off')
                row_name = ['', 'correct']
                table = plt.table(cellText=[col_name_correct, col_val_correct], rowLabels=row_name,
                                  loc='center', cellLoc='center', rowLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                plt.savefig(os.path.join(savefig_path, l) + '.jpg')
            elif l in incorrect_conf_dict.keys() and len(incorrect_conf_dict[l]) > 1:
                gs = gridspec.GridSpec(4, 1)
                plt.subplot(gs[:3, 0])
                incorrect_conf_dict[l].sort()
                incorrect_conf_dict[l].reverse()
                col_name_correct = ['number', 'mean', 'max', 'min', 'min99%', 'min99.9%']
                col_val_correct = [len(incorrect_conf_dict[l]),
                                   ('%.4f' % np.mean(incorrect_conf_dict[l])),
                                   ('%.4f' % max(incorrect_conf_dict[l])), ('%.2f' % min(incorrect_conf_dict[l])),
                                   ('%.2f' % incorrect_conf_dict[l][int(len(incorrect_conf_dict[l]) * 0.99) - 1]),
                                   ('%.2f' % incorrect_conf_dict[l][int(len(incorrect_conf_dict[l]) * 0.999) - 1])]
                sns.set_palette('hls')
                sns.distplot(incorrect_conf_dict[l], bins=50, kde_kws={'color': 'b', 'lw': 3},
                             hist_kws={'color': 'b', 'alpha': 0.3})
                plt.xlim((0, 1))
                plt.xlabel(l)
                plt.grid(True)
                plt.subplot(gs[3, 0])
                plt.axis('off')
                row_name = ['', 'incorrect']
                table = plt.table(cellText=[col_name_correct, col_val_correct], rowLabels=row_name,
                                  loc='center', cellLoc='center', rowLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                plt.savefig(os.path.join(savefig_path, l) + '.jpg')


class CocoVisualUtil(CocoDraw):
    def __init__(self):
        super(CocoVisualUtil, self).__init__()
        _init_value = np.array(0.0)
        self.summary_init = {
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

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0] + 1,
            _bbox[3] - _bbox[1] + 1,
            ]

    def bbox2result_1image(self, bboxes, labels, num_classes):
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

    def proposal2json(self, dataset, results):
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
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def det2json(self, dataset, results):
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
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = cat_ids[label]
                    json_results.append(data)
        return json_results

    def segm2json(self, dataset, results):
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
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
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

    def results2json(self, dataset, results, out_file):
        """convert result convert to json mode"""
        result_files = dict()
        if isinstance(results, list):
            result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
            mmcv.dump(results, result_files['bbox'])
        elif isinstance(results[0], list):
            json_results = self.det2json(dataset, results)
            result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self.segm2json(dataset, results)
            result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
            result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self.proposal2json(dataset, results)
            result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def calcuate_pr_rc_f1(self, config, coco, coco_dets, tgt_ids, im_path_dir, iou_type):
        cocoEval = DetectEval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = tgt_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        cocoEval.summarize()
        sys.stdout = stdout
        stats_all = cocoEval.stats

        eval_result_path = os.path.abspath("./recommend_result")
        if os.path.exists(eval_result_path):
            shutil.rmtree(eval_result_path)
        os.mkdir(eval_result_path)

        # record statistics result
        result_csv = os.path.join(eval_result_path, "statistics.csv")
        eval_item = ["ap@0.5:0.95", "ap@0.5", "ap@0.75", "ar@0.5:0.95", "ar@0.5", "ar@0.75"]
        self.write_list_to_csv(result_csv, eval_item, append=False)
        eval_result = [round(stats_all[0], 3), round(stats_all[1], 3), round(stats_all[2], 3), round(stats_all[6], 3),
                       round(stats_all[7], 3), round(stats_all[8], 3)]
        self.write_list_to_csv(result_csv, eval_result, append=True)
        self.write_list_to_csv(result_csv, [], append=True)

        # 1.2 plot_pr_curve
        cocoEval.plot_pr_curve(eval_result_path)

        # 2 coco.evaluate
        E = DetectEval(coco, coco_dets, iou_type)
        E.params.iouThrs = [0.5]
        E.params.maxDets = [100]
        E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
        E.evaluate()

        # 2.1 plot hist_curve of every class's tp's confidence and fp's confidence
        confidence_dict = E.compute_tp_fp_confidence()
        E.plot_hist_curve(confidence_dict, eval_result_path)

        # 2.2 write best_threshold and pr to csv and plot
        cat_pr_dict, cat_pr_dict_origin = E.compute_precison_recall_f1()
        # E.write_best_confidence_threshold(cat_pr_dict, cat_pr_dict_origin, eval_result_path)
        best_confidence_thres = E.write_best_confidence_threshold(cat_pr_dict, cat_pr_dict_origin, eval_result_path)
        print("best_confidence_thres: ", best_confidence_thres)
        E.plot_mc_curve(cat_pr_dict, eval_result_path)

        # 3
        # 3.1 compute every class's pr and save every class's p and r at iou = 0.5
        E = DetectEval(coco, coco_dets, iouType='bbox')
        E.params.iouThrs = [0.5]
        E.params.maxDets = [100]
        E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
        E.evaluate()
        E.accumulate()
        result = E.evaluate_every_class()
        print_info = ["class_name", "tp_num", "gt_num", "dt_num", "precision", "recall"]
        self.write_list_to_csv(result_csv, print_info, append=True)
        print("class_name", "tp_num", "gt_num", "dt_num", "precision", "recall")
        for class_result in result:
            print(class_result)
            self.write_list_to_csv(result_csv, class_result, append=True)

        # 3.2 save ng / ok images
        E.save_category_images(config, eval_result_path, im_path_dir, 0.5)
        return stats_all[0]

    def coco_eval(self, config, result_files, result_types, coco, im_path_dir,
                  score_threshold=None, recommend_threshold=False):
        """coco eval"""

        def check_success():
            flags = [0] * 3
            eval_result_path = os.path.abspath("./recommend_result")
            if os.path.exists(eval_result_path):
                result_files = os.listdir(eval_result_path)
                for file in result_files:
                    if file == "statistics.csv":
                        with open(os.path.join(eval_result_path, "statistics.csv"), "r") as f:
                            res = f.readlines()
                        if len(res) > 1:
                            if "class_name" in res[3] and "tp_num" in res[3] and len(res[4].strip().split(",")) > 1:
                                flags[0] = 1
                    elif file in ("precision_ng_images", "recall_ng_images", "ok_images"):
                        imgs = os.listdir(os.path.join(eval_result_path, file))
                        if imgs:
                            flags[1] = 1

                    elif file == "pr_curve_image":
                        imgs = os.listdir(os.path.join(eval_result_path, "pr_curve_image"))
                        if imgs:
                            flags[2] = 1
                    else:
                        pass

            if sum(flags) == 3:
                print("Successfully created 'eval_results' visualizations")
            else:
                print("Failed to create 'eval_results' visualizations")

        anns = json.load(open(result_files['bbox']))
        if not anns:
            return self.summary_init

        if mmcv.is_str(coco):
            coco = COCO(coco)
        assert isinstance(coco, COCO)

        for res_type in result_types:
            result_file = result_files[res_type]
            assert result_file.endswith('.json')

            coco_dets = coco.loadRes(result_file)
            gt_img_ids = coco.getImgIds()
            iou_type = 'bbox' if res_type == 'proposal' else res_type

            # draw eval image
            E = DetectEval(coco, coco_dets, iou_type)
            E.save_anns_images(config, im_path_dir, score_threshold=score_threshold)

            # set recommend threshold
            if recommend_threshold:
                self.calcuate_pr_rc_f1(config, coco, coco_dets, gt_img_ids, im_path_dir, iou_type)
                check_success()
        return True
