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
"""Evaluation script."""
import json
import time

import numpy as np
from mindspore import Model
from mindspore import context
from mindspore import dataset as ds
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.dataset.vision import transforms as vision
from mindspore.train.serialization import load_checkpoint

from cfg.config import config as default_config
from src.darknet import DarkNet, ResidualBlock
from src.dataset import JointDatasetDetection
from src.model import JDEeval
from src.model import YOLOv3
from src.utils import ap_per_class
from src.utils import bbox_iou
from src.utils import non_max_suppression
from src.utils import xywh2xyxy

set_seed(1)


def _get_rank_info(device_target):
    """
    Get rank size and rank id.
    """
    if device_target == 'GPU':
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        raise ValueError("Unsupported platform.")

    return rank_size, rank_id


def main(
        opt,
        iou_thres,
        conf_thres,
        nms_thres,
        nc,
):
    img_size = opt.img_size

    with open(opt.data_cfg_url) as f:
        data_config = json.load(f)
        test_paths = data_config['test']

    dataset = JointDatasetDetection(
        opt.dataset_root,
        test_paths,
        augment=False,
        transforms=vision.ToTensor(),
        config=opt,
    )

    dataloader = ds.GeneratorDataset(
        dataset,
        column_names=opt.col_names_val,
        shuffle=False,
        num_parallel_workers=1,
        max_rowsize=12,
    )

    dataloader = dataloader.batch(opt.batch_size, True)

    darknet53 = DarkNet(
        ResidualBlock,
        opt.backbone_layers,
        opt.backbone_input_shape,
        opt.backbone_shape,
        detect=True,
    )

    model = YOLOv3(
        backbone=darknet53,
        backbone_shape=opt.backbone_shape,
        out_channel=opt.out_channel,
    )

    model = JDEeval(model, opt)

    load_checkpoint(opt.ckpt_url, model)
    print(f'Evaluation for {opt.ckpt_url}')
    model = Model(model)

    mean_map, mean_r, mean_p, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    maps, mr, mp = [], [], []
    ap_accum, ap_accum_count = np.zeros(nc), np.zeros(nc)

    for batch_i, inputs in enumerate(dataloader):
        imgs, targets, targets_len = inputs
        targets = targets.asnumpy()
        targets_len = targets_len.asnumpy()

        t = time.time()

        raw_output, _ = model.predict(imgs)
        output = non_max_suppression(raw_output.asnumpy(), conf_thres=conf_thres, nms_thres=nms_thres)

        for i, o in enumerate(output):
            if o is not None:
                output[i] = o[:, :6]

        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for labels, detections in zip(targets, output):
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero ap
                if labels.shape[0] != 0:
                    maps.append(0)
                    mr.append(0)
                    mp.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections[np.argsort(-detections[:, 4])]

            # If no labels add number of detections as incorrect
            correct = []
            if labels.shape[0] == 0:
                maps.append(0)
                mr.append(0)
                mp.append(0)
                continue

            target_cls = labels[:, 0]

            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = xywh2xyxy(labels[:, 2:6])
            target_boxes[:, 0] *= img_size[0]
            target_boxes[:, 2] *= img_size[0]
            target_boxes[:, 1] *= img_size[1]
            target_boxes[:, 3] *= img_size[1]

            detected = []
            for *pred_bbox, _, _  in detections:
                obj_pred = 0
                pred_bbox = np.array(pred_bbox, dtype=np.float32).reshape(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

            # Compute Average Precision (ap) per class
            ap, ap_class, r, p = ap_per_class(
                tp=correct,
                conf=detections[:, 4],
                pred_cls=np.zeros_like(detections[:, 5]),  # detections[:, 6]
                target_cls=target_cls,
            )

            # Accumulate AP per class
            ap_accum_count += np.bincount(ap_class, minlength=nc)
            ap_accum += np.bincount(ap_class, minlength=nc, weights=ap)

            # Compute mean AP across all classes in this image, and append to image list
            maps.append(ap.mean())
            mr.append(r.mean())
            mp.append(p.mean())

            # Means of all images
            mean_map = np.sum(maps) / (ap_accum_count + 1E-16)
            mean_r = np.sum(mr) / (ap_accum_count + 1E-16)
            mean_p = np.sum(mp) / (ap_accum_count + 1E-16)

        if (batch_i + 1) % 1000 == 0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, dataset.nf, mean_p, mean_r, mean_map, time.time() - t))

    # Print results
    print(f'mean_mAP: {mean_map[0]:.4f}, mean_R: {mean_r[0]:.4f}, mean_P: {mean_p[0]:.4f}')


if __name__ == "__main__":
    config = default_config

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(device_id=config.device_id)

    main(
        opt=config,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        nc=config.num_classes,
    )
