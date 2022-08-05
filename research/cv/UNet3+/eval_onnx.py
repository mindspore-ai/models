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
import os
import datetime
import onnxruntime as ort
import numpy as np
from mindspore.common import set_seed

from src.logger import get_logger
from src.dataset import create_Dataset
from src.config import config
from eval import AverageMeter, dice_coef


def sigmoid(prediction):
    """sigmoid function"""
    negative = prediction < 0
    positive = prediction > 0
    sig_prediction = np.zeros_like(prediction)
    sig_prediction[negative] = np.exp(prediction[negative]) / (1 + np.exp(prediction[negative]))
    sig_prediction[positive] = 1 / (1 + np.exp(-prediction[positive]))
    return sig_prediction


def create_session(checkpoint_path, target_device):
    """create_session for onnx"""
    if target_device == "GPU":
        providers = ['CUDAExecutionProvider']
    elif target_device in ['CPU', 'Ascend']:
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'.\
         Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_onnx_eval():
    """run_onnx_eval"""
    set_seed(1)
    config.save_dir = os.path.join(config.output_path,
                                   datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.logger = get_logger(config.save_dir, "UNet3Plus", 0)
    config.logger.save_args(config)
    dataset, _ = create_Dataset(data_path=config.val_data_path, aug=0,
                                batch_size=config.batch_size,
                                device_num=1, rank=0, shuffle=False)
    data_loader = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    dices = AverageMeter()
    session, input_name = create_session(config.file_name, config.device_target)
    for episode, data in enumerate(data_loader):
        y_pred = sigmoid(session.run(None, {input_name: data['image']})[0])
        dice = dice_coef(y_pred, data['mask'])
        config.logger.info("episode %d : dice = %f ", episode, dice)
        dices.update(dice, config.batch_size)
    config.logger.info("Final dice: %s", str(dices.avg))


if __name__ == "__main__":
    run_onnx_eval()
