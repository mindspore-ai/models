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
"""run export"""
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore.train.serialization import export

from cfg.config import config as default_config
from src.darknet import DarkNet, ResidualBlock
from src.model import JDEeval
from src.model import YOLOv3


def run_export(config):
    """
    Export model to MINDIR.
    """
    darknet53 = DarkNet(
        ResidualBlock,
        config.backbone_layers,
        config.backbone_input_shape,
        config.backbone_shape,
        detect=True,
    )

    yolov3 = YOLOv3(
        backbone=darknet53,
        backbone_shape=config.backbone_shape,
        out_channel=config.out_channel,
    )

    net = JDEeval(yolov3, default_config)
    load_checkpoint(config.ckpt_url, net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, 1088, 608]), dtype=mstype.float32)
    name = Path(config.ckpt_url).stem

    export(net, input_data, file_name=name, file_format='MINDIR')
    print('Model exported successfully!')


if __name__ == "__main__":
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=default_config.device_target,
        device_id=default_config.device_id,
    )

    run_export(default_config)
