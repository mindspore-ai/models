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

"""ONNX Evaluation script"""
import os
import time

import numpy as np
import onnxruntime as ort
from mindspore.common import set_seed

from model_utils.config import config
from src.dataset import data_to_mindrecord_byte_image, create_deeptext_dataset
from src.utils import metrics


def create_session(checkpoint_path, target_device):
    """Create onnxruntime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def eval_deeptext(dataset_path, file_name):
    """Deeptext evaluation."""
    ds = create_deeptext_dataset(dataset_path, batch_size=config.test_batch_size,
                                 repeat_num=1, is_training=False)
    total = ds.get_dataset_size()
    session, input_names = create_session(file_name, config.device_target)
    eval_iter = 0

    print("\n========================================\n", flush=True)
    print("Processing, please wait a moment.", flush=True)

    max_num = 32

    pred_data = []
    for data in ds.create_dict_iterator():
        eval_iter = eval_iter + 1

        gt_bboxes = data['box'].asnumpy()
        gt_labels = data['label'].asnumpy()
        gt_num = data['valid_num'].asnumpy()

        start = time.time()

        inputs = [data[name].asnumpy() for name in ('image', 'image_shape', 'box', 'label', 'valid_num')]
        inputs = dict(zip(input_names, inputs))

        all_bbox, all_label, all_mask = session.run(None, inputs)
        all_label += 1

        gt_bboxes = gt_bboxes[gt_num.astype(bool), :]
        print(gt_bboxes, flush=True)
        gt_labels = gt_labels[gt_num.astype(bool)]
        print(gt_labels, flush=True)
        end = time.time()
        print(f"Iter {eval_iter} cost time {end - start}", flush=True)

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

            pred_data.append({"boxes": all_bboxes_tmp_mask,
                              "labels": all_labels_tmp_mask,
                              "gt_bboxes": gt_bboxes,
                              "gt_labels": gt_labels})

            percent = round(eval_iter / total * 100, 2)

            print(f"    {percent}% [{eval_iter}/{total}]", end="\r", flush=True)

    precisions, recalls = metrics(pred_data)
    print("\n========================================\n", flush=True)
    for i in range(config.num_classes - 1):
        j = i + 1
        f1 = (2 * precisions[j] * recalls[j]) / (precisions[j] + recalls[j] + 1e-6)
        print(f"class {j} precision is {precisions[j] * 100:.2f}%,",
              f"recall is {recalls[j] * 100:.2f}%,",
              f"F1 is {f1 * 100:.2f}%", flush=True)
        if config.use_ambigous_sample:
            break


def main():
    """Main function"""
    set_seed(1)
    prefix = config.eval_mindrecord_prefix
    config.test_images = config.imgs_path
    config.test_txts = config.annos_path
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...", flush=True)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create Mindrecord. It may take some time.", flush=True)
        data_to_mindrecord_byte_image(False, prefix, file_num=1)
        print(f"Create Mindrecord Done, at {mindrecord_dir}", flush=True)

    print("CHECKING MINDRECORD FILES DONE!", flush=True)
    print("Start Eval!", flush=True)
    eval_deeptext(mindrecord_file, config.file_name)


if __name__ == '__main__':
    main()
