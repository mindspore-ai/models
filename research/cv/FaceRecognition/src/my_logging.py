# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Custom logger."""
import sys
import os
import logging
from datetime import datetime

logger_name_1 = "FaceRecognition"


class HFLogger(logging.Logger):
    """HFLogger"""

    def __init__(self, logger_name, local_rank=0):
        super(HFLogger, self).__init__(logger_name)
        if local_rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, local_rank=0):
        """setup_logging_file"""
        self.local_rank = local_rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S") + "_rank_{}.log".format(local_rank)
        log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        fh.setFormatter(formatter)
        self.addHandler(fh)
        fh.close()

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)


def get_logger(path, rank):
    logger = HFLogger(logger_name_1, rank)
    logger.setup_logging_file(path, rank)
    return logger


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        return "{}".format(self.avg)
