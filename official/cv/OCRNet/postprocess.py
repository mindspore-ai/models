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
"""Post-process for 310 inference: calculate miou."""
import os
import argparse
import cv2
import numpy as np


def parse_args():
    """Post process parameters from command line."""
    parser = argparse.ArgumentParser(description="OCRNet Semantic Segmentation 310 Inference.")
    parser.add_argument("--result_path", type=str, help="Storage path of pred bin.")
    parser.add_argument("--label_path", type=str, help="Storage path of label bin.")
    args, _ = parser.parse_known_args()
    return args


def get_confusion_matrix(label, output, shape, num_class, ignore_label=255):
    """Calcute the confusion matrix by given label and pred."""
    # output = output.transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label[:, :shape[-2], :shape[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore_label
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype(np.int32)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def main(args):
    """Main function for miou calculation."""
    result_list = os.listdir(args.label_path)
    num_classes = 19
    confusion_matrix = np.zeros((num_classes, num_classes))
    ignore_label = 255
    count = 0
    for result in result_list:
        prefix = result.rstrip(".bin")
        pred = np.fromfile(os.path.join(args.result_path, prefix + "_1.bin"),
                           dtype=np.float32).reshape(19, 256, 512)
        label = np.fromfile(os.path.join(args.label_path, prefix + ".bin"),
                            dtype=np.int32).reshape(1, 1024, 2048)
        shape = label.shape
        output = pred.transpose(1, 2, 0)
        output = cv2.resize(output, (shape[-1], shape[-2]), interpolation=cv2.INTER_LINEAR)
        output = np.exp(output)
        output = np.expand_dims(output, axis=0)
        confusion_matrix += get_confusion_matrix(label, output, shape, num_classes, ignore_label)
        count += 1
    print("Total number of images: ", count, flush=True)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    iou_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_iou = iou_array.mean()

    # Show results
    print("=========== 310 Inference Result ===========", flush=True)
    print("miou:", mean_iou, flush=True)
    print("iou array: \n", iou_array, flush=True)
    print("============================================", flush=True)


if __name__ == "__main__":
    args_opt = parse_args()
    main(args=args_opt)
