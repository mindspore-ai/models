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
"""Util class or function."""
import sys
from collections import defaultdict
import datetime
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms

from .yolo import YoloLossBlock


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def default_wd_filter(x):
    """default weight decay filter."""
    parameter_name = x.name
    if parameter_name.endswith('.bias'):
        # all bias not using weight decay
        return False
    if parameter_name.endswith('.gamma'):
        # bn weight bias not using weight decay, be carefully for now x not
        # include BN
        return False
    if parameter_name.endswith('.beta'):
        # bn weight bias not using weight decay, be carefully for now x not
        # include BN
        return False

    return True


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not
            # include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not
            # include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params}]


class ShapeRecord:
    """Log image shape."""

    def __init__(self):
        self.shape_record = {
            416: 0,
            448: 0,
            480: 0,
            512: 0,
            544: 0,
            576: 0,
            608: 0,
            640: 0,
            672: 0,
            704: 0,
            736: 0,
            'total': 0
        }

    def set(self, shape):
        if len(shape) > 1:
            shape = shape[0]
        shape = int(shape)
        self.shape_record[shape] += 1
        self.shape_record['total'] += 1

    def show(self, logger):
        for key in self.shape_record:
            rate = self.shape_record[key] / float(self.shape_record['total'])
            logger.info('shape {}: {:.2f}%'.format(key, rate * 100))


def keep_loss_fp32(network):
    """Keep loss of network with float32"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, (YoloLossBlock,)):
            cell.to_float(ms.float32)


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


def cpu_affinity(rank_id, device_num):
    """Bind CPU cores according to rank_id and device_num."""
    import psutil
    cores = psutil.cpu_count()
    if cores < device_num:
        return
    process = psutil.Process()
    used_cpu_num = cores // device_num
    rank_id = rank_id % device_num
    used_cpu_list = [i for i in range(rank_id * used_cpu_num, (rank_id + 1) * used_cpu_num)]
    process.cpu_affinity(used_cpu_list)
    print(f"==== {rank_id}/{device_num} ==== bind cpu: {used_cpu_list}")


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args_detection, threshold):
        self.ignore_threshold = threshold
        self.labels = args_detection.labels
        self.num_classes = len(self.labels)
        self.results = {}
        self.file_path = ''
        self.save_prefix = args_detection.output_dir
        self.ann_file = args_detection.ann_file
        self._coco = COCO(self.ann_file)
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.det_boxes = []
        self.nms_thresh = args_detection.eval_nms_thresh
        self.multi_label = args_detection.multi_label
        self.multi_label_thresh = args_detection.multi_label_thresh
        self.coco_catids = self._coco.getCatIds()
        self.coco_catIds = args_detection.coco_ids

    def do_nms_for_results(self):
        """Get result boxes."""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._diou_nms(dets, thresh=self.nms_thresh)

                keep_box = [{'image_id': int(img_id), 'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)} for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / \
                (areas[i] + areas[order[1:]] - intersect_area)

            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _diou_nms(self, dets, thresh=0.5):
        """
        convert xywh -> xmin ymin xmax ymax
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
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
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self):
        """Save result to file."""
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def get_eval_result(self):
        """Get eval result."""
        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        sys.stdout = stdout
        return rdct.content

    def detect(self, outputs, batch, image_shape, image_id):
        """Detect boxes."""
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_item in outputs:
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])
                if img_id not in self.results:
                    self.results[img_id] = defaultdict(list)
                x = ori_w * out_item_single[..., 0].reshape(-1)
                y = ori_h * out_item_single[..., 1].reshape(-1)
                w = ori_w * out_item_single[..., 2].reshape(-1)
                h = ori_h * out_item_single[..., 3].reshape(-1)
                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]
                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                if self.multi_label:
                    confidence = conf.reshape(-1, 1) * cls_emb
                    # create all False
                    flag = cls_emb > self.multi_label_thresh
                    flag = flag.nonzero()
                    for i, j in zip(*flag):
                        confi = confidence[i][j]
                        if confi < self.ignore_threshold:
                            continue
                        x_lefti, y_lefti = max(0, x_top_left[i]), max(0, y_top_left[i])
                        wi, hi = min(w[i], ori_w), min(h[i], ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_catIds[j]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
                else:
                    cls_argmax = cls_argmax.reshape(-1)
                    # create all False
                    flag = np.random.random(cls_emb.shape) > sys.maxsize
                    for i in range(flag.shape[0]):
                        c = cls_argmax[i]
                        flag[i, c] = True
                    confidence = conf.reshape(-1) * cls_emb[flag]
                    for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left,
                                                                     w, h, confidence, cls_argmax):
                        if confi < self.ignore_threshold:
                            continue
                        x_lefti, y_lefti = max(0, x_lefti), max(0, y_lefti)
                        wi, hi = min(wi, ori_w), min(hi, ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_catids[clsi]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
