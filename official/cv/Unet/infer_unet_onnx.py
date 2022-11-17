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

import logging
import onnxruntime

from src.data_loader import create_dataset, create_multi_class_dataset
from src.utils import dice_coeff
from src.model_utils.config import config


def test_net():

    print(config.device_target)
    if config.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif config.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {config.device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    onnx_path = config.file_name if config.file_name.rfind('.onnx') else config.file_name+'.onnx'
    session = onnxruntime.InferenceSession(onnx_path, provider_options=providers)
    if hasattr(config, "dataset") and config.dataset != "ISBI":
        split = config.split if hasattr(config, "split") else 0.8
        valid_dataset = create_multi_class_dataset(config.data_path, config.image_size, repeat=1, batch_size=1,
                                                   num_classes=config.num_classes, is_train=False,
                                                   eval_resize=config.eval_resize, split=split, shuffle=False)
    else:
        _, valid_dataset = create_dataset(config.data_path, repeat=1, train_batch_size=1, augment=False,
                                          cross_val_ind=1, run_distribute=False, do_crop=config.crop,
                                          img_size=config.image_size)
    dice_metric = dice_coeff()

    print("============== Starting Evaluating ============")
    # valid_dataset
    for data in valid_dataset.create_tuple_iterator():
        img, label = data

        inputs = {session.get_inputs()[0].name: img.asnumpy()}
        model_predict = session.run(None, inputs)
        dice_metric.update(model_predict[0], label)

    eval_score = dice_metric.eval()
    print("============== Cross valid dice coeff is:", eval_score[0])
    print("============== Cross valid IOU is:", eval_score[1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_net()
