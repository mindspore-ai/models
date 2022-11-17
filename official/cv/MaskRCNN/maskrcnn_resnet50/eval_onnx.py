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
"""Run evaluation for a model exported to ONNX"""

import os
import time
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO

from src.dataset import create_maskrcnn_dataset
from src.model_utils.config import config
from src.util import coco_eval, bbox2result_1image, results2json, get_seg_masks


def create_session(checkpoint_path, target_device):
    """Load ONNX model and create ORT session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def run_eval(onnx_checkpoint_path, mindrecord_path, batch_size, num_classes, target_device, input_type,
             max_predictions_count=128):
    """Run ONNX model evaluation

    Args:
        onnx_checkpoint_path (str): path to ONNX checkpoint
        mindrecord_path (str): path to MindRecord, including name prefix
        batch_size (int): batch size
        num_classes (int): number of classes
        target_device (str): 'GPU' / 'CPU'
        input_type (str): 'float16' / 'float32'
        max_predictions_count (int, optional): maximum number of predictions. Defaults to 128.

    Returns:
        List[Tuple]: predictions - (boxes, masks) tuples for each image
    """
    session, (images_input_name, shapes_input_name) = create_session(onnx_checkpoint_path, target_device)

    dataset = create_maskrcnn_dataset(mindrecord_path, batch_size, is_training=False)
    it = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

    outputs = []
    for eval_iter, batch in enumerate(it):
        start_time = time.time()

        inputs = {
            images_input_name: batch['image'].astype(input_type),
            shapes_input_name: batch['image_shape'].astype(input_type)
        }
        pred_boxes, pred_labels, valid_predictions_masks, pred_masks = session.run(None, inputs)
        pred_labels = np.squeeze(pred_labels, -1)
        valid_predictions_masks = np.squeeze(valid_predictions_masks, -1)

        it = zip(valid_predictions_masks, pred_boxes, pred_labels, pred_masks, batch['image_shape'])
        for valid_mask, boxes_batch, labels_batch, masks_batch, metas_batch in it:
            boxes = boxes_batch[valid_mask]
            labels = labels_batch[valid_mask]
            masks = masks_batch[valid_mask]

            if len(boxes) > max_predictions_count:
                indices = boxes[:, -1].argsort()[::-1]
                indices = indices[:max_predictions_count]
                boxes = boxes[indices]
                labels = labels[indices]
                masks = masks[indices]

            result_boxes = bbox2result_1image(boxes, labels, num_classes)
            result_masks = get_seg_masks(masks, boxes, labels, metas_batch,
                                         rescale=True, num_classes=num_classes)
            outputs.append((result_boxes, result_masks))

        end_time = time.time()
        # flush must be true to avoid losing output when this script is launched with "&> log_file.txt"
        print(f"Iter {eval_iter} / {dataset.get_dataset_size()} took {end_time - start_time:.2f} s", flush=True)

    return outputs


def report_metrics(outputs, coco_annotation_path, eval_types=('bbox', 'segm')):
    """Print COCO metrics"""
    coco_dataset = COCO(coco_annotation_path)
    with TemporaryDirectory() as tempdir:
        prefix = os.path.join(tempdir, 'results.pkl')
        result_files = results2json(coco_dataset, outputs, prefix)
        coco_eval(result_files, eval_types, coco_dataset, single_result=False)


def main():
    """Run ONNX eval from command line"""
    mindrecord_path = os.path.join(config.coco_root, config.mindrecord_dir, 'MaskRcnn_eval.mindrecord')
    coco_annotation_path = os.path.join(config.coco_root, config.ann_file)

    outputs = run_eval(config.file_name, mindrecord_path, config.test_batch_size,
                       config.num_classes, config.device_target, config.export_input_type)
    report_metrics(outputs, coco_annotation_path)


if __name__ == '__main__':
    main()
