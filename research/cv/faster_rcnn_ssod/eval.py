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
"""Eval"""
import os
import time

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context, set_seed, Parameter
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pycocotools.coco import COCO

from src.eval_utils import coco_eval, bbox2result_1image, results2json
from src.utils import ValueInfo, update_config
from src.config import FasterRcnnConfig
from src.dataset import create_semisup_dataset
from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50 as Faster_Rcnn_Resnet


def eval_combine():
    print("========================================")
    print("[INFO] Config contents:")
    for cfg_k, cfg_v in FasterRcnnConfig.__dict__.items():
        if not cfg_k.startswith("_"):
            print("{}: {}".format(cfg_k, cfg_v))
    print("========================================")

    cfg = FasterRcnnConfig()
    cfg.test_batch_size = 1
    print("[INFO] set config test_batch_size = 1")

    set_seed(cfg.global_seed)
    device_id = os.getenv('DEVICE_ID', str(cfg.eval_device_id))
    device_id = int(device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=device_id)
    context.set_context(max_call_depth=6000)
    print("[INFO] current device_id: {}".format(device_id))

    ####################################################################################
    # semisup dataset
    semisup_loader = create_semisup_dataset(cfg, is_training=False)

    ####################################################################################
    # net init
    net = Faster_Rcnn_Resnet(config=cfg)

    ckpt_file_path = str(os.getenv('CKPT_PATH', cfg.checkpoint_path))
    param_dict = load_checkpoint(ckpt_file_name=ckpt_file_path)
    if cfg.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    param_not_load, _ = load_param_into_net(net, param_dict, strict_load=True)

    if cfg.device_target == "Ascend":
        net.to_float(mstype.float16)
    net = net.set_train(mode=False)
    print("[DEBUG] load {} to net, params not load: {}".format(ckpt_file_path, param_not_load))
    print("[INFO] net create ok.")

    ####################################################################################
    print("========================================")
    steps_per_epoch = semisup_loader.get_dataset_size()
    print("[INFO] total_steps: {}".format(steps_per_epoch))
    print("[INFO] Processing, please wait a moment.")

    time_infos = [ValueInfo("iter_time"), ValueInfo("data_time")]
    outputs = []
    step_iter = cfg.start_iter
    start_time = time.perf_counter()
    for data in semisup_loader.create_dict_iterator(num_epochs=-1):

        label_data_k, label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums = \
            data["label_img_weak"], data["label_img_metas"], data["label_gt_bboxes"], \
            data["label_gt_labels"], data["label_gt_nums"]
        data_time = time.perf_counter() - start_time

        output = net(label_data_k, label_img_metas, label_gt_bboxes, label_gt_labels, label_gt_nums)
        post_process(cfg, output, outputs)

        iter_time = time.perf_counter() - start_time

        # update eval infos
        time_infos[0].update(iter_time)
        time_infos[1].update(data_time)

        if step_iter % cfg.print_interval_iter == 0:
            print("[INFO] device_id: {}, step: {}/{}, iter_time: {:.3f}, data_time: {:.3f}"
                  .format(device_id, step_iter, steps_per_epoch, time_infos[0].avg(), time_infos[1].avg()))

        step_iter += 1
        start_time = time.perf_counter()

    dataset_coco = COCO(cfg.eval_ann_file)
    result_files = results2json(dataset_coco, outputs, os.path.join(cfg.eval_output_dir, "eval_results.pkl"))
    eval_types = ["bbox"]
    result = coco_eval(result_files, eval_types, dataset_coco, single_result=False)

    print("[INFO] ckpt: {}, coco eval result:\n".format(ckpt_file_path))
    for key, value in result.items():
        print("{}: {}".format(key, value))

    print("[INFO] end.")


def post_process(cfg, output, outputs):
    max_num = 128
    all_bbox = output[0]
    all_label = output[1]
    all_mask = output[2]
    for j in range(cfg.test_batch_size):
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

        outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, cfg.num_classes)
        outputs.append(outputs_tmp)


if __name__ == '__main__':
    update_config()
    eval_combine()
