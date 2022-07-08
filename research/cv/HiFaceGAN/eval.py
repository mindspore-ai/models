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
"""Eval HiFaceGAN model"""
import os
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.dataset.dataset import create_eval_dataset
from src.model.generator import HiFaceGANGenerator
from src.model.reporter import Reporter
from src.model_utils.config import get_config
from src.util import enable_batch_statistics
from src.util import make_joined_image
from src.util import save_image


class Metrics(nn.Metric):
    """Metrics class"""

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        self.sum = ops.ReduceSum()
        self._total_pred = 0
        self._total_num = 0

    def clear(self):
        """Clear metrics data"""
        self._total_pred = 0
        self._total_num = 0

    def eval(self):
        """Evaluate metrics"""
        return self._total_pred / self._total_num

    def update(self, hq, generated):
        """Update metrics"""
        batch_pred = self.metrics(hq, generated)
        self._total_pred += self.sum(batch_pred).asnumpy().item()
        self._total_num += hq.shape[0]


def run_multiple_eval(config):
    """HiFaceGAN multiple checkpoint evaluation"""
    dataset, dataset_size = create_eval_dataset(config.data_path, config.degradation_type, batch_size=1,
                                                img_size=config.img_size)
    config.dataset_size = dataset_size

    ckpt_list = [str(x) for x in Path(config.ckpt_path).glob('*.ckpt')]
    # sort by the time of creation
    ckpt_list.sort(key=os.path.getmtime, reverse=True)
    ckpt_list = ckpt_list[:config.num_to_eval]

    dataloader = dataset.create_dict_iterator(num_epochs=config.num_to_eval + 1)

    imgs_out = os.path.join(config.outputs_dir, 'predict')
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)

    generator = HiFaceGANGenerator(
        ngf=config.ngf,
        input_nc=config.input_nc
    )
    enable_batch_statistics(generator)

    reporter = Reporter(config)
    reporter.dataset_size = config.dataset_size
    reporter.start_predict()

    psnr = Metrics(nn.PSNR())
    ssim = Metrics(nn.SSIM())

    results_psnr = {}
    results_ssim = {}

    for checkpoint_path in ckpt_list:
        ms.load_checkpoint(checkpoint_path, net=generator)
        psnr.clear()
        ssim.clear()
        reporter.info('Start validation for checkpoint %s', checkpoint_path)

        for i, data_i in enumerate(dataloader):
            if i % 100 == 0:
                reporter.info('Processing... [%d / %d]', i, reporter.dataset_size)
            lq = data_i['low_quality']
            hq = data_i['high_quality']
            generated = generator(lq)
            psnr.update(hq, generated)
            ssim.update(hq, generated)

        psnr_ = psnr.eval()
        ssim_ = ssim.eval()
        results_psnr[checkpoint_path] = psnr_
        results_ssim[checkpoint_path] = ssim_

        reporter.info('Metrics: PSNR = %.4f, SSIM = %.4f for checkpoint %s', psnr_, ssim_, checkpoint_path)

    best_checkpoint_path = max(results_psnr, key=results_psnr.get)
    best_psnr = results_psnr[best_checkpoint_path]
    best_ssim = results_ssim[best_checkpoint_path]
    reporter.info('Best checkpoint path: %s', best_checkpoint_path)
    reporter.info('Metrics for the best checkpoint: PSNR = %.4f, SSIM = %.4f', best_psnr, best_ssim)

    ms.load_checkpoint(best_checkpoint_path, net=generator)

    reporter.info('Start generating images for the best checkpoint...')
    for i, data_i in enumerate(dataloader):
        if i > 100:
            break
        lq = data_i['low_quality']
        hq = data_i['high_quality']
        generated = generator(lq)
        # as batch_size=1, we take only 0th index from the tensors
        joined_image = make_joined_image(lq[0], generated[0], hq[0])
        save_image(joined_image[:, :, ::-1], os.path.join(imgs_out, '%d.png' % i))

    reporter.info('Save generated images at %s', imgs_out)
    reporter.end_predict()


if __name__ == '__main__':
    cfg = get_config()
    ms.context.set_context(device_target=cfg.device_target, mode=ms.context.GRAPH_MODE)
    run_multiple_eval(cfg)
