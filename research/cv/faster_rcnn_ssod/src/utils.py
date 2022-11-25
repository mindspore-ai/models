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
"""utils functions"""

import logging
import argparse

from src.config import FasterRcnnConfig


class ValueInfo:
    def __init__(self, name: str):
        self.name = name
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1

    def avg(self):
        if self.count == 0:
            raise ZeroDivisionError("[ERROR] ValueInfo count == 0")
        mean = self.sum / self.count
        self.reset()
        return mean

    def reset(self):
        self.sum = 0.0
        self.count = 0


def update_config():
    parser = argparse.ArgumentParser(description="args to train fasterrcnn ssod.")
    parser.add_argument("--device_target", default="Ascend", type=str, choices=["Ascend", "GPU"],
                        help="set training device")
    parser.add_argument("--device_id", default=0, type=int,
                        help="set training device id")
    parser.add_argument("--run_distribute", action="store_true",
                        help="run distribute to train model")
    parser.add_argument("--save_checkpoint_path", default="./outputs", type=str,
                        help="path to save train model checkpoint")
    parser.add_argument("--pre_trained", type=str,
                        help="pre trained checkpoint")
    parser.add_argument("--train_img_dir", type=str,
                        help="train dataset images dir path")
    parser.add_argument("--train_ann_file", type=str,
                        help="train dataset annotations json path")
    parser.add_argument("--eval_img_dir", type=str,
                        help="eval dataset images dir path")
    parser.add_argument("--eval_ann_file", type=str,
                        help="eval dataset annotations json path")
    parser.add_argument("--eval_output_dir", type=str,
                        help="eval output dir")
    parser.add_argument("--output_dir", type=str,
                        help="train output dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="eval or infer checkpoint path")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value:
            setattr(FasterRcnnConfig, key, value)


def init_log():
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(message)s"
    logging.basicConfig(level="INFO", format=fmt)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level="INFO")
