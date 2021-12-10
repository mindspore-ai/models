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
"""Evaluation for RefineDet"""

import os
import argparse
import time
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.eval_utils import coco_metrics
from src.eval_utils import voc_metrics
from src.box_utils import box_init
from src.config import get_config
from src.dataset import create_refinedet_dataset, create_mindrecord
from src.refinedet import refinedet_vgg16, refinedet_resnet101, RefineDetInferWithDecoder

def refinedet_eval(net_config, dataset_path, ckpt_path, anno_json, net_metrics):
    """RefineDet evaluation."""
    batch_size = 1
    ds = create_refinedet_dataset(net_config, dataset_path, batch_size=batch_size, repeat_num=1,
                                  is_training=False, use_multiprocessing=False)
    if net_config.model == "refinedet_vgg16":
        net = refinedet_vgg16(net_config, is_training=False)
    elif net_config.model == "refinedet_resnet101":
        net = refinedet_resnet101(net_config, is_training=False)
    else:
        raise ValueError(f'config.model: {net_config.model} is not supported')
    default_boxes = box_init(net_config)
    net = RefineDetInferWithDecoder(net, Tensor(default_boxes), net_config)

    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    i = batch_size
    total = ds.get_dataset_size() * batch_size
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(Tensor(img_np))
        for batch_idx in range(img_np.shape[0]):
            pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                              "box_scores": output[1].asnumpy()[batch_idx],
                              "img_id": int(np.squeeze(img_id[batch_idx])),
                              "image_shape": image_shape[batch_idx]})
        percent = round(i / total * 100., 2)

        print(f'    {str(percent)} [{i}/{total}]', end='\r')
        i += batch_size
    cost_time = int((time.time() - start) * 1000)
    print(f'    100% [{total}/{total}] cost {cost_time} ms')
    mAP = net_metrics(pred_data, anno_json, net_config)
    print("\n========================================\n")
    print(f"mAP: {mAP}")

def get_eval_args():
    """Get args for eval"""
    parser = argparse.ArgumentParser(description='RefineDet evaluation')
    parser.add_argument("--using_mode", type=str, default="refinedet_vgg16_320",
                        choices=("refinedet_vgg16_320", "refinedet_vgg16_512",
                                 "refinedet_resnet101_320", "refinedet_resnet101_512"),
                        help="using mode, same as training.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend ,GPU and CPU.")
    parser.add_argument('--debug', type=str, default="0", choices=["0", "1"],
                        help="Active the debug mode. Under debug mode, the network would be run as PyNative mode.")
    return parser.parse_args()

if __name__ == '__main__':
    args_opt = get_eval_args()
    config = get_config(args_opt.using_mode, args_opt.dataset)
    box_init(config)
    if args_opt.dataset == "coco":
        json_path = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    elif args_opt.dataset[:3] == "voc":
        json_path = os.path.join(config.voc_root, config.voc_json)
    else:
        json_path = config.instances_set

    if args_opt.debug == "1":
        network_mode = context.PYNATIVE_MODE
    else:
        network_mode = context.GRAPH_MODE

    context.set_context(mode=network_mode, device_target=args_opt.run_platform, device_id=args_opt.device_id)

    mindrecord_file = create_mindrecord(config, args_opt.dataset,
                                        "refinedet_eval.mindrecord", False,
                                        file_num=1)

    print("Start Eval!")
    metrics = coco_metrics if args_opt.dataset == 'coco' else voc_metrics
    refinedet_eval(config, mindrecord_file, args_opt.checkpoint_path, json_path, metrics)
