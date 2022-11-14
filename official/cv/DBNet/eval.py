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
"""DBNet Evaluation."""
import os
import sys

import mindspore as ms

from src.datasets.load import create_dataset
from src.utils.eval_utils import WithEval
from src.utils.env import init_env
from src.modules.model import get_dbnet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@moxing_wrapper()
def evaluate(cfg, path):
    cfg.device_num = 1
    init_env(cfg)
    cfg.backbone.pretrained = False
    eval_net = get_dbnet(cfg.net, cfg, isTrain=False)
    eval_net = WithEval(eval_net, cfg)
    val_dataset, _ = create_dataset(cfg, False)
    val_dataset = val_dataset.create_dict_iterator(output_numpy=True)
    ms.load_checkpoint(path, eval_net.model)
    eval_net.model.set_train(False)
    metrics, fps = eval_net.eval(val_dataset, show_imgs=cfg.eval.show_images)
    params = sum([param.size for param in eval_net.model.get_parameters()]) / (1024 ** 2)
    print(f"Param: {params} M")
    print(f"FPS: {fps}\n"
          f"Recall: {metrics['recall'].avg}\n"
          f"Precision: {metrics['precision'].avg}\n"
          f"Fmeasure: {metrics['fmeasure'].avg}\n")
    return metrics


if __name__ == '__main__':
    evaluate(config, config.ckpt_path)
