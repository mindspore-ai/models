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
"""Coco metrics utils"""

import os
import json
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms
from mindspore.common import Parameter
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset, parse_json_annos_from_txt
from src.util import bbox2result_1image, results2json


def create_eval_mindrecord(config):
    """ eval_fasterrcnn """
    print("CHECKING MINDRECORD FILES ...")
    if not os.path.exists(config.mindrecord_file):
        if not os.path.isdir(config.mindrecord_dir):
            os.makedirs(config.mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", False, config.prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(config.mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", False, config.prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(config.mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")


def apply_eval(net, config, dataset_path, ckpt_path, anno_path):
    """FasterRcnn evaluation."""
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))
    ds = create_fasterrcnn_dataset(config, dataset_path, batch_size=config.test_batch_size, is_training=False)

    param_dict = ms.load_checkpoint(ckpt_path)
    if config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    ms.load_param_into_net(net, param_dict)

    net.set_train(False)
    device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(ms.float16)

    eval_iter = 0
    total = ds.get_dataset_size()
    outputs = []

    if config.dataset != "coco":
        dataset_coco = COCO()
        dataset_coco.dataset, dataset_coco.anns, dataset_coco.cats, dataset_coco.imgs = dict(), dict(), dict(), dict()
        dataset_coco.imgToAnns, dataset_coco.catToImgs = defaultdict(list), defaultdict(list)
        dataset_coco.dataset = parse_json_annos_from_txt(anno_path, config)
        dataset_coco.createIndex()
    else:
        dataset_coco = COCO(anno_path)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = config.num_gts
    for data in ds.create_dict_iterator(num_epochs=1):
        eval_iter = eval_iter + 1

        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        # run net
        output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num)

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)

            outputs.append(outputs_tmp)

    eval_types = ["bbox"]
    reslut_path = "./{}epoch_results.pkl".format(config.current_epoch)
    result_files = results2json(dataset_coco, outputs, reslut_path)

    return metrics_map(result_files, eval_types, dataset_coco, single_result=False)


def metrics_map(result_files, result_types, coco, max_dets=(100, 300, 1000), single_result=False):
    """coco eval for fasterrcnn"""

    anns = json.load(open(result_files['bbox']))
    if not anns:
        return 0

    if isinstance(coco, str):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        det_img_ids = coco_dets.getImgIds()
        gt_img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.params.imgIds = tgt_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    return cocoEval.stats[0]
