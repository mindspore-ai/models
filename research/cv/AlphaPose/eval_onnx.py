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
'''
This file evaluates the model used.
'''
from __future__ import division

import os
import time
import numpy as np
import onnxruntime as ort

from mindspore import context
from mindspore.common import set_seed

from src.config import config
from src.dataset import flip_pairs
from src.dataset import CreateDatasetCoco
from src.utils.coco import evaluate
from src.utils.transforms import flip_back
from src.utils.inference import get_final_preds

set_seed(config.EVAL_SEED)
device_id = int(os.getenv('DEVICE_ID', '0'))

def create_session(checkpoint_path, target_device):
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
    input_name = session.get_inputs()[0].name
    return session, input_name


def validate(cfg, val_dataset, ckpt_path, output_dir, ann_path):
    '''
    validate
    '''
    num_samples = val_dataset.get_dataset_size() * cfg.ONNX_TEST_BATCH_SIZE
    all_preds = np.zeros((num_samples, cfg.MODEL_NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    session, input_name = create_session(ckpt_path, config.ONNX_device_target)

    start = time.time()
    for item in val_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        inputs = item['image']
        output = np.array(session.run(None, {input_name: inputs})[0])

        if cfg.TEST_FLIP_TEST:
            inputs_flipped = inputs[:, :, :, ::-1]
            output_flipped = np.array(session.run(None, {input_name: inputs_flipped})[0])
            output_flipped = flip_back(output_flipped, flip_pairs)

            if cfg.TEST_SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        c = item['center']
        s = item['scale']
        score = item['score']
        file_id = list(item['id'])

        preds, maxvals = get_final_preds(output.copy(), c, s)
        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(cfg, all_preds[:idx], output_dir,
                                 all_boxes[:idx], image_id, ann_path)
    print("AP:", perf_indicator)
    return perf_indicator


def main():

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.ONNX_device_target,
                        device_id=config.ONNX_device_id)

    ckpt_path = config.ONNX_CKPT_PATH
    print("load checkpoint from [{}].".format(ckpt_path))
    valid_dataset = CreateDatasetCoco(
        train_mode=False,
        onnx_eval_mode=True,
        num_parallel_workers=config.TEST_NUM_PARALLEL_WORKERS,
    )

    ckpt_name = ckpt_path.split('/')
    ckpt_name = ckpt_name[len(ckpt_name) - 1]
    ckpt_name = ckpt_name.split('.')[0]

    output_dir = config.TEST_OUTPUT_DIR + ckpt_name
    ann_path = os.path.join(config.DATASET_ROOT, config.DATASET_TEST_JSON)
    validate(config, valid_dataset, ckpt_path, output_dir, ann_path)

if __name__ == '__main__':
    main()
