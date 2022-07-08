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

"""OCRNet inference."""
import argparse
import ast

import numpy as np
from mindspore import context, DatasetHelper
from mindspore import ops as P
from mindspore.dataset import engine as de
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.cityscapes import Cityscapes
from src.config import config_hrnetv2_w48 as config
from src.config import organize_configuration
from src.model_utils.moxing_adapter import moxing_wrapper
from src.seg_hrnet_ocr import get_seg_model


def parse_args():
    """
    Get arguments from command-line.
    """
    parser = argparse.ArgumentParser(description='OCRNet Semantic Segmentation Inference.')
    parser.add_argument("--data_url", type=str, default=None,
                        help="Storage path of dataset.")
    parser.add_argument("--train_url", type=str, default=None,
                        help="Storage path of evaluation results in OBS. It's useless here.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Storage path of dataset in OBS.")
    parser.add_argument("--device_target", type=str, default=None, help="Target device [Ascend, GPU]")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Storage path of evaluation results on machine. It's useless here.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run online or offline.")
    parser.add_argument("--flip_eval", type=ast.literal_eval, default=False,
                        help="Add result for flipped image.")
    parser.add_argument("--checkpoint_url", type=str,
                        help="Storage path of checkpoint file in OBS.")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Storage path of checkpoint file on machine.")
    return parser.parse_args()


def get_confusion_matrix(label, pred, shape, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred.
    """
    output = pred.asnumpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
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


def testval(dataset, helper, model, num_classes=19, ignore_label=255, scales=None, flip=False):
    """
    Inference function.
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    count = 0
    for batch in helper:
        print("=====> Image: ", count)
        image, label = batch  # NCHW, NHW
        shape = label.shape
        pred = dataset.multi_scale_inference(model, image,
                                             scales=scales,
                                             flip=flip)

        if pred.shape[-2] != shape[-2] or pred.shape[-1] != shape[-1]:
            pred = P.ResizeBilinear((shape[-2], shape[-1]))(pred)  # Tensor

        confusion_matrix += get_confusion_matrix(label, pred, shape, num_classes, ignore_label)
        count += 1
    print("Total number of images: ", count)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    iou_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_iou = iou_array.mean()

    return mean_iou, iou_array


@moxing_wrapper(config)
def main():
    """Inference process."""
    # Set context
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    # Initialize network
    net = get_seg_model(config)
    param_dict = load_checkpoint(ckpt_file_name=config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # Prepare dataset
    ori_dataset = Cityscapes(config.data_path,
                             num_samples=None,
                             num_classes=config.dataset.num_classes,
                             multi_scale=False,
                             flip=False,
                             ignore_label=config.dataset.ignore_label,
                             base_size=config.eval.base_size,
                             crop_size=config.eval.image_size,
                             downsample_rate=1,
                             scale_factor=16,
                             mean=config.dataset.mean,
                             std=config.dataset.std,
                             is_train=False)
    dataset = de.GeneratorDataset(ori_dataset, column_names=["image", "label"],
                                  shuffle=False,
                                  num_parallel_workers=config.workers)
    dataset = dataset.batch(1, drop_remainder=True)
    helper = DatasetHelper(dataset, dataset_sink_mode=False)

    # Calculate results
    mean_iou, iou_array = testval(ori_dataset, helper, net,
                                  num_classes=config.dataset.num_classes,
                                  ignore_label=config.dataset.ignore_label,
                                  scales=config.eval.scale_list, flip=config.eval.flip)
    # Show results
    print("=========== Validation Result ===========")
    print("===> mIoU:", mean_iou)
    print("===> IoU array: \n", iou_array)
    print("=========================================")


if __name__ == '__main__':
    args = parse_args()
    organize_configuration(cfg=config, args=args)
    config.eval.flip = args.flip_eval
    main()
