# Copyright 2023 Huawei Technologies Co., Ltd
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
import time
import collections
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from src.model_utils.config import get_eval_config
from src.segformer import SegFormer
from src.dataset import CitySpacesDataset, get_eval_dataset, prepare_cityscape_dataset


IntersectAndUnion = collections.namedtuple('IntersectAndUnion', ['area_intersect', 'area_union', 'area_pred_label',
                                                                 'area_label'])


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return IntersectAndUnion(area_intersect, area_union, area_pred_label, area_label)


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return IntersectAndUnion(total_area_intersect, total_area_union, total_area_pred_label, total_area_label)


def do_eval(config, net=None, dataset_iterator=None, dataset_size=None):
    class_num = config.class_num
    if dataset_iterator is None:
        prepare_cityscape_dataset(config.data_path)
        dataset = get_eval_dataset(config)
        dataset_iterator = dataset.create_dict_iterator()
        dataset_size = dataset.get_dataset_size()

    eval_dataset_size = dataset_size
    print(f"eval dataset size:{eval_dataset_size}")

    if net is None:
        net = SegFormer(config.backbone, class_num, False).to_float(ms.float16)
        param_dict = load_checkpoint(ckpt_file_name=config.eval_ckpt_path)
        load_param_into_net(net, param_dict)
        print(f"load {config.eval_ckpt_path} success.")
        net.set_train(False)

    eval_begin_time = int(time.time())
    pred_labels = []
    gt_labels = []
    for step_idx, item in enumerate(dataset_iterator):
        begin_time = int(time.time() * 1000)
        image = item['image']
        label = item['label']
        pred = net(image)
        end_time = int(time.time() * 1000)
        pred = pred[-1]
        predn = pred.asnumpy()
        predn = np.asarray(np.argmax(predn, axis=0), dtype=np.uint8)
        pred_labels.append(predn)
        gt_labels.append(label.asnumpy()[-1])
        if (step_idx + 1) % config.eval_log_interval == 0:
            print(f"eval image {step_idx + 1}/{eval_dataset_size} done, step cost: {end_time - begin_time}ms")
    total_area_intersect, total_area_union, _, total_area_label = \
        total_intersect_and_union(pred_labels, gt_labels, class_num, config.dataset_ignore_label)
    mean_acc = total_area_intersect.sum() / total_area_label.sum()
    acc_array = total_area_intersect / total_area_label
    iou_array = total_area_intersect / total_area_union
    mean_iou = iou_array.mean()
    classes = CitySpacesDataset.CLASSES
    print(f"======================= Evaluation Result =======================")
    for idx, item in enumerate(iou_array):
        print(f"===> class: {classes[idx]:^14}   IoU: {item:.4f}   Acc: {acc_array[idx]:.4f}")
    print(f"=================================================================")
    print(f"===> mIoU: {mean_iou:.4f}, mAcc: {mean_acc:.4f}, ckpt: {os.path.basename(config.eval_ckpt_path)}")
    print(f"=================================================================")
    eval_end_time = int(time.time())
    print(f"all eval process done, cost: {eval_end_time - eval_begin_time}s")
    return mean_iou


if __name__ == '__main__':
    eval_config = get_eval_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=eval_config.device_target)
    print(f"eval config:{eval_config}")
    do_eval(eval_config, None, None, None)
