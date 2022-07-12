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
"""Evaluation script."""
from pathlib import Path
from time import time

import cv2
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore.common import set_seed

from src.cfg.config import config as default_config
from src.dataset import ImageMattingDatasetVal
from src.model import MobileNetV2UNetDecoderIndexLearning
from src.utils import compute_connectivity_loss
from src.utils import compute_gradient_loss
from src.utils import compute_mse_loss
from src.utils import compute_sad_loss
from src.utils import image_alignment


def evaluation(config):
    """
    Init model, dataset, run evaluation.

    Args:
        config: Config parameters.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    val_loader = ImageMattingDatasetVal(
        data_dir=config.data_dir,
        config=config,
        sub_folder='validation',
        data_file='data.txt',
    )

    net = MobileNetV2UNetDecoderIndexLearning(
        encoder_rate=config.rate,
        encoder_current_stride=config.current_stride,
        encoder_settings=config.inverted_residual_setting,
        output_stride=config.output_stride,
        width_mult=config.width_mult,
        conv_operator=config.conv_operator,
        decoder_kernel_size=config.decoder_kernel_size,
        apply_aspp=config.apply_aspp,
        use_nonlinear=config.use_nonlinear,
        use_context=config.use_context,
    )

    load_checkpoint(config.ckpt_url, net)
    net.set_train(False)

    with Path(config.data_dir, 'validation/data.txt').open() as file:
        image_list = [name.split('|') for name in file.read().splitlines()]

    eval_logs_dir = Path(config.logs_dir)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    sad = []
    mse = []
    grad = []
    conn = []
    avg_frame_rate = 0.0
    stride = config.output_stride
    start = time()
    for i, (image, gt_alpha, trimap, transposed, pad_mask, size) in enumerate(val_loader):
        h, w = image.shape[1:]
        image = image.transpose(1, 2, 0)
        image = image_alignment(image, stride, odd=False)
        inputs = Tensor(np.expand_dims(image.transpose(2, 0, 1), axis=0), mstype.float32)

        # Inference
        outputs = net(inputs).asnumpy().squeeze()

        alpha = cv2.resize(outputs, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

        alpha = alpha[pad_mask].reshape(size)
        alpha = np.clip(alpha, 0, 1) * 255.

        # Trimap edge region
        mask = np.equal(trimap, 128).astype(np.float32)

        alpha = (1 - mask) * trimap + mask * alpha
        gt_alpha = gt_alpha * 255.

        save_path = eval_logs_dir / Path(image_list[i][0]).name
        if transposed:
            Image.fromarray(alpha.transpose(1, 0).astype(np.uint8)).save(save_path)
        else:
            Image.fromarray(alpha.astype(np.uint8)).save(save_path)

        # compute loss
        sad.append(compute_sad_loss(alpha, gt_alpha, mask))
        mse.append(compute_mse_loss(alpha, gt_alpha, mask))
        grad.append(compute_gradient_loss(alpha, gt_alpha, mask))
        conn.append(compute_connectivity_loss(alpha, gt_alpha, mask))

        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        avg_frame_rate = (avg_frame_rate * i + running_frame_rate) / (i + 1)
        print(
            f'test: {i + 1}/{len(val_loader)}, sad: {sad[-1]:.2f},'
            f' mse: {mse[-1]:.4f}, grad: {grad[-1]:.2f}, conn: {conn[-1]:.2f},'
            f' frame: {running_frame_rate:.2f}FPS',
        )
        start = time()

    print(60 * '=')
    print(
        f'SAD: {np.mean(sad):.2f}, MSE: {np.mean(mse):.4f},'
        f' Grad: {np.mean(grad):.2f}, Conn: {np.mean(conn):.2f},'
        f' frame: {avg_frame_rate:.2f}FPS',
    )
    print('Evaluation success')


if __name__ == '__main__':
    set_seed(1)
    evaluation(config=default_config)
