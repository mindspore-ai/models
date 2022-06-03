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
"""CTSDG evaluation"""

from pathlib import Path
from time import time
from typing import Union

from PIL import Image
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore import numpy as mnp
from mindspore.nn import PSNR
from mindspore.nn import SSIM
from mindspore.ops import Concat

from model_utils.config import get_config
from src.dataset import create_ctsdg_dataset
from src.generator.generator import Generator
from src.utils import check_args

psnr_dict = {
    '0-20%': [],
    '20-40%': [],
    '40-60%': [],
}
ssim_dict = {
    '0-20%': [],
    '20-40%': [],
    '40-60%': [],
}


def postprocess(x: Tensor) -> Tensor:
    """
    Map tensor values from [-1, 1] to [0, 1]

    Args:
        x: Input tensor

    Returns:
        Normalized tensor
    """
    x = (x + 1.) / 2.
    x = mnp.clip(x, 0, 1)
    return x


def save_img(img: Tensor, path: Union[str, Path]):
    """
    Normalize and save output image

    Args:
        img: Tensor image with values in [0, 1] range
        path: Image save path

    Returns:
        None
    """
    if img.ndim == 4:
        img = img[0]
    img_np = mnp.clip((img * 255) + 0.5, 0, 255).transpose((1, 2, 0)).astype(mstype.uint8).asnumpy()
    Image.fromarray(img_np).save(path)


def eval_ctsgd(cfg):
    """
    Evaluate CTSDG model and get PSNR and SSIM values.
     If cfg.result_root is a valid directory, function also save output images.

    Args:
        cfg: Model configuration

    Returns:
        None
    """
    generator = Generator(
        image_in_channels=3,
        edge_in_channels=2,
        out_channels=3,
    )
    print('Starting evaluation process! Please wait for a few minutes...', flush=True)

    if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
        load_checkpoint(cfg.checkpoint_path, generator)

    generator.set_train(False)

    dataset = create_ctsdg_dataset(cfg, is_training=False)

    calc_psnr = PSNR()
    calc_ssim = SSIM()

    if cfg.output_path:
        save_dir = Path(cfg.output_path)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
    else:
        save_dir = None

    total = dataset.get_dataset_size() * cfg.test_batch_size
    print(f'Total number of images in dataset: {total}', flush=True)

    start_time = time()
    pic_index = 0

    for ground_truth, mask, edge, gray_image in dataset.create_tuple_iterator():
        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask

        output, _, _ = generator(input_image, Concat(axis=1)((input_edge, input_gray_image)), mask)

        output_comp = ground_truth * mask + output * (1 - mask)
        output_comp = postprocess(output_comp)

        ground_truth_post = postprocess(ground_truth)

        psnr_value = calc_psnr(output_comp, ground_truth_post)
        ssim_value = calc_ssim(output_comp, ground_truth_post)

        part = 1 - mask.sum() / mask.size

        if part <= 0.2:
            psnr_dict['0-20%'].append(psnr_value)
            ssim_dict['0-20%'].append(ssim_value)
        elif 0.2 < part <= 0.4:
            psnr_dict['20-40%'].append(psnr_value)
            ssim_dict['20-40%'].append(ssim_value)
        elif 0.4 < part <= 0.6:
            psnr_dict['40-60%'].append(psnr_value)
            ssim_dict['40-60%'].append(ssim_value)

        if save_dir:
            for i in range(cfg.test_batch_size):
                pic_index += 1
                save_img(output_comp[i, ...], save_dir / f'{pic_index:05d}.png')
        else:
            pic_index += cfg.test_batch_size

        if pic_index % cfg.verbose_step == 0:
            end = time()
            pic_cost = (end - start_time) / cfg.verbose_step
            time_left = (total - pic_index) * pic_cost
            print(f"Processed images: {pic_index} of {total}, "
                  f"Fps: {1 / pic_cost:.2f}, "
                  f"Image time: {pic_cost:.2f} sec, "
                  f"Time left: ~{time_left:.2f} sec.", flush=True)
            start_time = time()

    print('PSNR:', flush=True)
    for k, v in psnr_dict.items():
        if v:
            metric_value = (sum(v) / len(v)).asnumpy()[0]
            print(f'{k}: {metric_value:.2f}', flush=True)

    print('SSIM:', flush=True)
    for k, v in ssim_dict.items():
        if v:
            metric_value = (sum(v) / len(v)).asnumpy()[0]
            print(f'{k}: {metric_value:.3f}', flush=True)


if __name__ == '__main__':
    config = get_config()
    check_args(config)
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        device_id=config.device_id,
    )
    eval_ctsgd(config)
