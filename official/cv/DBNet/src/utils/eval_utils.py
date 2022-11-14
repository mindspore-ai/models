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
"""DBNet evaluation tools"""

import os
import time
import numpy as np
import cv2
from tqdm.auto import tqdm

import mindspore as ms

from .metric import QuadMetric
from .post_process import SegDetectorRepresenter

class WithEval:
    def __init__(self, model, config):
        super(WithEval, self).__init__()
        self.model = model
        self.config = config
        self.metric = QuadMetric(config.eval.polygon)
        self.post_process = SegDetectorRepresenter(config.eval.thresh, config.eval.box_thresh,
                                                   config.eval.max_candidates,
                                                   config.eval.unclip_ratio,
                                                   config.eval.polygon,
                                                   config.eval.dest)
    def once_eval(self, batch):
        start = time.time()
        img = ms.Tensor(batch['img'])
        preds = self.model(img)
        boxes, scores = self.post_process(preds)
        cur_time = time.time() - start
        raw_metric = self.metric.validate_measure(batch, (boxes, scores))

        cur_frame = img.shape[0]

        return raw_metric, (cur_frame, cur_time)

    def eval(self, dataset, show_imgs=True):
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []
        count = 0

        for batch in tqdm(dataset):
            raw_metric, (cur_frame, cur_time) = self.once_eval(batch)
            raw_metrics.append(raw_metric)
            if count:
                total_frame += cur_frame
                total_time += cur_time

            count += 1
            if show_imgs:
                img = batch['original'].squeeze().astype('uint8')
                # gt
                for idx, poly in enumerate(raw_metric['gt_polys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['gt_dont_care']:
                        cv2.polylines(img, [poly], True, (255, 160, 160), 4)
                    else:
                        cv2.polylines(img, [poly], True, (255, 0, 0), 4)
                # pred
                for idx, poly in enumerate(raw_metric['det_polys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['det_dont_care']:
                        cv2.polylines(img, [poly], True, (200, 255, 200), 4)
                    else:
                        cv2.polylines(img, [poly], True, (0, 255, 0), 4)
                if not os.path.exists(self.config.eval.image_dir):
                    os.makedirs(self.config.eval.image_dir)
                cv2.imwrite(self.config.eval.image_dir + f'eval_{count}.jpg', img)

        metrics = self.metric.gather_measure(raw_metrics)
        fps = total_frame / total_time
        return metrics, fps


class Evaluate310:
    def __init__(self, config):
        super(Evaluate310, self).__init__()
        self.config = config
        self.metric = QuadMetric(config.eval.polygon)
        self.post_process = SegDetectorRepresenter(config.eval.thresh, config.eval.box_thresh,
                                                   config.eval.max_candidates,
                                                   config.eval.unclip_ratio,
                                                   config.eval.polygon,
                                                   config.eval.dest)
        self.gt_path = config.output_dir
        self.pred_path = os.path.join(config.output_dir, "eval_result_bin")

    def once_eval(self, batch):
        start = time.time()
        preds = batch['pred']
        boxes, scores = self.post_process(preds)
        cur_time = time.time() - start
        raw_metric = self.metric.validate_measure(batch, (boxes, scores))
        cur_frame = batch['img'].shape[0]
        return raw_metric, (cur_frame, cur_time)

    def eval(self, show_imgs=False):
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []
        count = 0

        for batch in tqdm(self.get_batch()):
            raw_metric, (cur_frame, cur_time) = self.once_eval(batch)
            raw_metrics.append(raw_metric)
            if count:
                total_frame += cur_frame
                total_time += cur_time
            count += 1
            if show_imgs:
                img = batch['original'].squeeze().astype('uint8')
                # gt
                for idx, poly in enumerate(raw_metric['gt_polys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['gt_dont_care']:
                        cv2.polylines(img, [poly], True, (255, 160, 160), 4)
                    else:
                        cv2.polylines(img, [poly], True, (255, 0, 0), 4)
                # pred
                for idx, poly in enumerate(raw_metric['det_polys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['det_dont_care']:
                        cv2.polylines(img, [poly], True, (200, 255, 200), 4)
                    else:
                        cv2.polylines(img, [poly], True, (0, 255, 0), 4)
                if not os.path.exists(self.config.eval.image_dir):
                    os.makedirs(self.config.eval.image_dir)
                cv2.imwrite(self.config.eval.image_dir + f'eval_{count}.jpg', img)

        metrics = self.metric.gather_measure(raw_metrics)
        fps = total_frame / total_time
        return metrics, fps

    def get_shape(self, x):
        x = x.strip()[1:-1].split(", ")
        shape = []
        for i in x:
            shape.append(int(i))
        return tuple(shape)


    def get_batch(self):
        polys_dir = os.path.join(self.gt_path, "eval_polys_bin")
        dontcare_dir = os.path.join(self.gt_path, "eval_dontcare_bin")
        img_dir = os.path.join(self.gt_path, "eval_input_bin")
        ori_dir = os.path.join(self.gt_path, "eval_ori")

        eval_shapes = open(os.path.join(self.gt_path, "eval_shapes"), "r")

        for i in range(len(os.listdir(img_dir))):
            batch = {}
            polys = np.fromfile(os.path.join(polys_dir, f"eval_polys_{i + 1}.bin"), dtype=np.float64)
            polys.shape = self.get_shape(eval_shapes.readline())
            batch['polys'] = polys
            dontcare = np.fromfile(os.path.join(dontcare_dir, f"eval_dontcare_{i + 1}.bin"), dtype=np.bool)
            dontcare.shape = self.get_shape(eval_shapes.readline())
            batch['dontcare'] = dontcare

            img = np.fromfile(os.path.join(img_dir, f"eval_input_{i + 1}.bin"), dtype=np.float32)
            img.shape = self.get_shape(eval_shapes.readline())
            batch['img'] = img

            pred = np.fromfile(os.path.join(self.pred_path, f"eval_input_{i + 1}_0.bin"), dtype=np.float32)
            pred.shape = (img.shape[0], 1, img.shape[2], img.shape[3])
            batch["pred"] = (pred,)
            batch["original"] = np.load(os.path.join(ori_dir, f"eval_org_{i + 1}.npy"))
            yield batch

        eval_shapes.close()
