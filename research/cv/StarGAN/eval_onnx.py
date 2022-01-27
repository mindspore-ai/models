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
"""ONNX evaluation for StarGAN"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.utils import create_labels, denorm
from src.config import get_config
from src.dataset import dataloader


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


def main():
    """ONNX evaluation"""
    config = get_config()
    generator, (image_input_name, domain_input_name) = create_session(config.export_file_name, config.device_target)

    if not os.path.exists(config.result_dir):
        os.mkdir(config.result_dir)
    # Define Dataset

    data_path = config.celeba_image_dir
    attr_path = config.attr_path

    dataset, length = dataloader(img_path=data_path,
                                 attr_path=attr_path,
                                 batch_size=config.batch_size,
                                 selected_attr=config.selected_attrs,
                                 device_num=config.num_workers,
                                 dataset=config.dataset,
                                 mode=config.mode,
                                 shuffle=False)

    ds = dataset.create_dict_iterator(output_numpy=True)
    print(length)
    print('Start Evaluating!')
    for i, data in enumerate(ds):
        x_real = data['image'].astype(np.float32)
        result_list = [x_real]
        c_trg_list = create_labels(data['attr'], selected_attrs=config.selected_attrs)
        c_trg_list = c_trg_list.asnumpy().astype(np.float32)

        for c_trg in c_trg_list:
            inputs = {
                image_input_name: x_real,
                domain_input_name: c_trg,
            }
            [x_fake] = generator.run(None, inputs)

            result_list.append(x_fake)

        x_fake_list = np.concatenate(result_list, axis=3)

        result = denorm(x_fake_list)
        result = np.reshape(result, (-1, 768, 3))

        im = Image.fromarray(np.uint8(result))
        im.save(config.result_dir + f'/test_{i}.jpg')
        print('Successful save image in ' + config.result_dir + f'/test_{i}.jpg')


if __name__ == '__main__':
    main()
