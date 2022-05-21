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
"""Inference script."""
import logging
import os
import os.path as osp

from mindspore import Model
from mindspore import context
from mindspore.train.serialization import load_checkpoint

from cfg.config import config as default_config
from eval import eval_seq
from src.darknet import DarkNet, ResidualBlock
from src.dataset import LoadVideo
from src.log import logger
from src.model import JDEeval
from src.model import YOLOv3
from src.utils import mkdir_if_missing

logger.setLevel(logging.INFO)

def track(opt):
    """
    Inference of the input video.

    Save the results into output-root (video, annotations and frames.).
    """

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    anchors = opt.anchor_scales

    dataloader = LoadVideo(
        opt.input_video,
        anchor_scales=anchors,
        img_size=opt.img_size,
    )

    darknet53 = DarkNet(
        ResidualBlock,
        opt.backbone_layers,
        opt.backbone_input_shape,
        opt.backbone_shape,
        detect=True,
    )
    model = YOLOv3(
        backbone=darknet53,
        backbone_shape=opt.backbone_shape,
        out_channel=opt.out_channel,
    )

    model = JDEeval(model, opt)
    load_checkpoint(opt.ckpt_url, model)
    model = Model(model)
    logger.info('Starting tracking...')

    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    try:
        eval_seq(
            opt,
            dataloader,
            'mot',
            result_filename,
            net=model,
            save_dir=frame_dir,
            frame_rate=frame_rate,
        )
    except TypeError as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = f"ffmpeg -f image2 -i {osp.join(result_root, 'frame')}/%05d.jpg -c:v copy {output_video_path}"
        os.system(cmd_str)


if __name__ == '__main__':
    config = default_config

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    context.set_context(device_id=config.device_id)

    track(config)
