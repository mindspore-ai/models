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
from src.utils.logger import get_logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_default():
    config.output_dir = os.path.join(config.output_dir, config.net, config.backbone.initializer)
    config.save_ckpt_dir = os.path.join(config.output_dir, 'ckpt')
    config.log_dir = os.path.join(config.output_dir, 'log')
    os.makedirs(config.save_ckpt_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)


@moxing_wrapper()
def evaluate(cfg, path):
    cfg.device_num = 1
    set_default()
    init_env(cfg)
    cfg.logger = get_logger(cfg.log_dir, cfg.rank_id)
    cfg.backbone.pretrained = False

    val_dataset, _ = create_dataset(cfg, False)
    val_dataset = val_dataset.create_dict_iterator(output_numpy=True)
    paths = [path]
    if os.path.isdir(path):
        paths = []
        files = os.listdir(path)
        for file in files:
            if file.endswith(".ckpt"):
                paths.append(os.path.join(path, file))
    for p in paths:
        eval_net = get_dbnet(cfg.net, cfg, isTrain=False)
        eval_net = WithEval(eval_net, cfg)
        eval_net.model.set_train(False)
        cfg.logger.info(f"infer {p}")
        ms.load_checkpoint(p, eval_net.model)
        metrics, fps = eval_net.eval(val_dataset, show_imgs=cfg.eval.show_images)
        params = sum([param.size for param in eval_net.model.get_parameters()]) / (1024 ** 2)
        cfg.logger.info(f"Param: {params} M")
        cfg.logger.info(f"FPS: {fps}\n"
                        f"Recall: {metrics['recall'].avg}\n"
                        f"Precision: {metrics['precision'].avg}\n"
                        f"Fmeasure: {metrics['fmeasure'].avg}\n")
    return metrics


if __name__ == '__main__':
    evaluate(config, config.ckpt_path)
