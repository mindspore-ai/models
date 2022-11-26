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

import logging.config
import logging.handlers

import yaml

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
sh = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[sh])
log = logging.getLogger('ArtTrack')


def setup_log(file=None):
    with open(file=file or './config/log.yaml', mode='r', encoding="utf-8") as f:
        logging_yaml = yaml.load(stream=f, Loader=yaml.FullLoader)
        logging.config.dictConfig(config=logging_yaml)
