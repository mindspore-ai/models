# Copyright 2021 Huawei Technologies Co., Ltd
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

import logging
import sys
import os

def create_logger():
    """
    Create a new logger
    """
    path = "./logs"
    if not os.path.exists(path):
        os.makedirs(path)
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
