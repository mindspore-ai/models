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

"""Evaluation for SSD"""

import argparse
import os
import time

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.box_utils import default_boxes
from src.config import config
from src.dataset import create_mindrecord
from src.dataset import create_ssd_dataset
from src.eval_utils import metrics
from src.ssd_resnet34 import SSDInferWithDecoder


def ssd_eval(
        dataset_path: str,
        batch_size: int,
        ckpt_path: str,
        anno_json: str,
) -> None:
    """Eval SSD-Resnet34 model

    Args:
        dataset_path (str): A path to the COCO2017 or VOC dataset.
        batch_size (int): A batch size for the evaluation.
        ckpt_path (str): A path to the model checkpoint for the evaluation.
        anno_json (str): A path to the file with annotations.

    Returns:
        None
    """
    ds = create_ssd_dataset(
        dataset_path,
        batch_size=batch_size,
        repeat_num=1,
        is_training=False,
        use_multiprocessing=False,
    )
    if config.model != "ssd-resnet34":
        raise ValueError(f'config.model: {config.model} is not supported')

    net = SSDInferWithDecoder(Tensor(default_boxes), config)

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
    mAP = metrics(pred_data, anno_json)
    print("\n========================================\n")
    print(f"mAP: {mAP}")


def get_eval_args():
    """Get eval args"""
    parser = argparse.ArgumentParser(description='SSD-Resnet34 evaluation')
    parser.add_argument("--data_url", type=str, help="Path to the dataset.")
    parser.add_argument("--mindrecord", type=str, help="Path to the folder with mindrecords.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--run_platform", type=str, default="GPU", choices=("GPU", "CPU"),
                        help="run platform, support Ascend, GPU and CPU.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="A batch size for the evaluation, default is 1.")
    return parser.parse_args()


if __name__ == '__main__':
    args_opt = get_eval_args()

    config.checkpoint_path = args_opt.checkpoint_path
    config.coco_root = args_opt.data_url
    config.mindrecord_dir = args_opt.mindrecord
    config.batch_size = args_opt.batch_size

    if args_opt.dataset == "coco":
        json_path = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    elif args_opt.dataset == "voc":
        json_path = os.path.join(config.voc_root, config.voc_json)
    else:
        raise ValueError('SSD eval only support dataset mode is coco and voc!')

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args_opt.run_platform,
        device_id=args_opt.device_id,
    )
    mindrecord_file = create_mindrecord(
        args_opt.dataset,
        "ssd_eval.mindrecord",
        False,
    )

    print("Start Evaluation!")
    ssd_eval(mindrecord_file, config.batch_size, config.checkpoint_path, json_path)
