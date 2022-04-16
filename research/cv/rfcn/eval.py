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

"""Evaluation for RFCN"""
import os
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed, Parameter
from src.dataset import data_to_mindrecord_byte_image, create_rfcn_dataset, parse_json_annos_from_txt
from src.util import coco_eval, bbox2result_1image, results2json
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.rfcn.rfcn_resnet import Rfcn_Resnet

set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

def rfcn_eval(dataset_path, ckpt_path, anno_path):
    """Rfcn evaluation."""
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))
    ds = create_rfcn_dataset(config, dataset_path, batch_size=config.test_batch_size, is_training=False,
                             num_parallel_workers=config.num_parallel_workers)
    net = Rfcn_Resnet(config)
    param_dict = load_checkpoint(ckpt_path)
    if config.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    else:
        raise RuntimeError("now, RFCN only support GPU.")
    load_param_into_net(net, param_dict)

    net.set_train(False)

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
    max_num = 128
    for data in ds.create_dict_iterator(num_epochs=1):
        eval_iter = eval_iter + 1
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        start = time.time()
        # run net
        output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))

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
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    coco_eval(config, result_files, eval_types, dataset_coco, single_result=False, plot_detect_result=False)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def eval_rfcn():
    """ eval_rfcn """
    if config.dataset == "coco":
        prefix = "Rfcn_coco_eval.mindrecord"
    else:
        prefix = "Rfcn_other_eval.mindrecord"

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
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    print("CHECKING MINDRECORD FILES DONE!")
    print("Start Eval!")
    rfcn_eval(mindrecord_file, config.checkpoint_path, config.anno_path)
    print("eval success.")



if __name__ == '__main__':
    eval_rfcn()
