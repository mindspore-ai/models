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

"""Evaluation of ONNX model"""
import os
import sys
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from mindspore.common import set_seed
from pycocotools.coco import COCO

from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset, parse_json_annos_from_txt
from src.model_utils.config import get_config
from src.util import coco_eval, bbox2result_1image, results2json


def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def eval_fasterrcnn(config, dataset_path, ckpt_path, anno_path, target_device):
    """FasterRcnn evaluation."""
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"CheckPoint file {ckpt_path} is not valid.")
    ds = create_fasterrcnn_dataset(config, dataset_path, batch_size=config.test_batch_size, is_training=False)
    session, input_names = create_session(ckpt_path, target_device)

    eval_iter = 0
    total = ds.get_dataset_size()
    outputs = []

    if config.dataset != "coco":
        dataset_coco = COCO()
        dataset_coco.dataset, dataset_coco.anns, dataset_coco.cats, dataset_coco.imgs = {}, {}, {}, {}
        dataset_coco.imgToAnns, dataset_coco.catToImgs = defaultdict(list), defaultdict(list)
        dataset_coco.dataset = parse_json_annos_from_txt(anno_path, config)
        dataset_coco.createIndex()
    else:
        dataset_coco = COCO(anno_path)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128
    for data in ds.create_dict_iterator(num_epochs=1):
        eval_iter = eval_iter + 1

        start = time.time()
        input_data = [data[i].asnumpy() for i in ('image', 'image_shape', 'box', 'label', 'valid_num')]
        output = session.run(None, dict(zip(input_names, input_data)))
        end = time.time()

        print(f"Iter {eval_iter} cost time {end - start}")

        # output
        all_bbox, all_label, all_mask = output

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox[j, :, :])
            all_label_squee = np.squeeze(all_label[j, :, :])
            all_mask_squee = np.squeeze(all_mask[j, :, :])

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
    result_files = results2json(dataset_coco, outputs, "./results.pkl")

    coco_eval(config, result_files, eval_types, dataset_coco, single_result=True, plot_detect_result=True)


def main():
    """Main function"""
    set_seed(1)

    config = get_config()

    prefix = "FasterRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...")

    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", False, prefix, file_num=1)
                print(f"Create Mindrecord Done, at {mindrecord_dir}")
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", False, prefix, file_num=1)
                print(f"Create Mindrecord Done, at {mindrecord_dir}")
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    print("CHECKING MINDRECORD FILES DONE!")
    print("Start Eval!")
    eval_fasterrcnn(config, mindrecord_file, config.file_name, config.anno_path, config.device_target)

    flags = [0] * 3
    config.eval_result_path = os.path.abspath("./eval_result")
    if os.path.exists(config.eval_result_path):
        result_files = os.listdir(config.eval_result_path)
        for file in result_files:
            if file == "statistics.csv":
                with open(os.path.join(config.eval_result_path, "statistics.csv"), "r", encoding="utf-8") as f:
                    res = f.readlines()
                if len(res) > 1:
                    if "class_name" in res[3] and "tp_num" in res[3] and len(res[4].strip().split(",")) > 1:
                        flags[0] = 1
            elif file in ("precision_ng_images", "recall_ng_images", "ok_images"):
                imgs = os.listdir(os.path.join(config.eval_result_path, file))
                if imgs:
                    flags[1] = 1
            elif file == "pr_curve_image":
                imgs = os.listdir(os.path.join(config.eval_result_path, "pr_curve_image"))
                if imgs:
                    flags[2] = 1

    if sum(flags) == 3:
        print("eval success.")
    else:
        print("eval failed.")
        sys.exit(-1)


if __name__ == '__main__':
    main()
