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
"""HRNet inference."""
import os
import ast
import timeit
import argparse
import numpy as np

import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.seg_hrnet import get_seg_model
from src.dataset.dataset_generator import create_seg_dataset
from src.config import hrnetw48_config as model_config


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="HRNet training.")

    parser.add_argument("--data_url", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cityscapes")
    parser.add_argument("--train_url", type=str, default=None)
    parser.add_argument("--checkpoint_url", type=str, default=None)
    parser.add_argument("--task", type=str, default="seg", choices=["seg", "cls"])
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False)

    return parser.parse_args()


def get_confusion_matrix(label, pred, shape, num_class, ignore=-1):
    """Calcute the confusion matrix by given label and pred."""
    output = pred.asnumpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label.asnumpy()[:, :shape[-2], :shape[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def inference_cityscapes(net, helper, num_classes):
    """Inference with Cityscapes."""
    confusion_matrix = np.zeros((num_classes, num_classes))
    count = 0
    for batch in helper:
        print("=====> Image: ", count, flush=True)
        image, label = batch
        shape = label.shape
        pred = net(image)
        pred = ops.ResizeBilinear((shape[-2], shape[-1]))(pred)
        pred = ops.Exp()(pred)

        confusion_matrix += get_confusion_matrix(label, pred, shape, num_classes, 255)
        count += 1
    return confusion_matrix, count


def inference_lip(net, helper, num_classes):
    """Inference with LIP."""
    confusion_matrix = np.zeros((num_classes, num_classes))
    count = 0
    for batch in helper:
        print("=====> Image: ", count, flush=True)
        image, label = batch
        shape = image.shape
        pred = net(image)
        pred = ops.ResizeBilinear((shape[-2], shape[-1]))(pred)
        # flip
        flip_img = image.asnumpy()[:, :, :, ::-1]
        flip_output = net(Tensor(flip_img))
        flip_output = ops.ResizeBilinear((shape[-2], shape[-1]))(flip_output).asnumpy()
        flip_pred = flip_output
        flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
        flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
        flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
        flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
        flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
        flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
        flip_pred = Tensor(flip_pred[:, :, :, ::-1])
        pred += flip_pred
        pred = pred * 0.5
        pred = ops.Exp()(pred)

        confusion_matrix += get_confusion_matrix(label, pred, shape, num_classes, 255)
        count += 1
    return confusion_matrix, count


def main():
    """Inference process."""
    # Set context
    args = parse_args()
    if args.modelarts:
        import moxing as mox
        local_data_url = "/cache/data"
        mox.file.copy_parallel(args.data_url, local_data_url)
        ckpt_name = args.checkpoint_url.strip().split("/")[-1]
        local_checkpoint_url = os.path.join("/cache/ckpt", ckpt_name)
        mox.file.copy_parallel(args.checkpoint_url, local_checkpoint_url)
    else:
        local_data_url = args.data_url
        local_checkpoint_url = args.checkpoint_url

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # Prepare dataset
    helper, _, num_classes, _ = create_seg_dataset(
        args.dataset, local_data_url, batchsize=1, run_distribute=False, is_train=False)
    # Initialize network
    net = get_seg_model(model_config, num_classes)
    param_dict = load_checkpoint(ckpt_file_name=local_checkpoint_url)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # Calculate results
    start = timeit.default_timer()
    if args.dataset == "cityscapes":
        confusion_matrix, count = inference_cityscapes(net, helper, num_classes)
    elif args.dataset == "lip":
        confusion_matrix, count = inference_lip(net, helper, num_classes)
    else:
        raise ValueError("Unsupported dataset {}.".format(args.dataset))
    end = timeit.default_timer()
    total_time = end - start
    avg_time = total_time / count
    print("Number of samples: {:4d}, total time: {:4.2f}s, average time: {:4.2f}s".format(
        count, total_time, avg_time), flush=True)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    iou_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_iou = iou_array.mean()

    # Show results
    print("============= 910 Inference =============", flush=True)
    print("miou:", mean_iou, flush=True)
    print("iou array: \n", iou_array, flush=True)
    print("=========================================", flush=True)


if __name__ == '__main__':
    main()
