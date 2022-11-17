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
"""Run evaluation for a model exported to ONNX"""
import os.path

import numpy as np
import onnxruntime as ort

from src.dataset import create_dataset
from src.model_utils.config import get_config
from src.utils import create_sliding_window, CalculateDice


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


def run_eval(data_path, ckpt_path, target_device, batch_size,
             num_classes, roi_size, overlap):
    session, input_name = create_session(ckpt_path, target_device)

    data_dir = os.path.join(data_path, 'image')
    seg_dir = os.path.join(data_path, 'seg')
    eval_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, is_training=False)
    eval_data_size = eval_dataset.get_dataset_size()
    print(f'train dataset length is: {eval_data_size}')

    total_dice = 0
    for index, batch in enumerate(eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image = batch['image']
        seg = batch['seg']
        print(f'current image shape is {image.shape}')
        sliding_window_list, slice_list = create_sliding_window(image, roi_size, overlap)
        image_size = (batch_size, num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(roi_size, np.float32)
        for window, slice_ in zip(sliding_window_list, slice_list):
            pred_probs = session.run(None, {input_name: window})[0]
            output_image[slice_] += pred_probs
            count_map[slice_] += importance_map
        output_image = output_image / count_map
        dice, _ = CalculateDice(output_image, seg)
        print(f'The {index} batch dice is {dice}')
        total_dice += dice
    avg_dice = total_dice / eval_data_size

    return {
        'average dice': avg_dice,
    }


if __name__ == '__main__':
    config = get_config()

    results = run_eval(config.data_path, config.file_name,
                       config.device_target, config.batch_size,
                       config.num_classes, config.roi_size,
                       config.overlap)

    for name, value in results.items():
        print(f'{name}: {value:.4f}')
