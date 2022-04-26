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
"""utils"""


def check_args(cfg):
    if cfg.device_target != 'GPU':
        raise ValueError(f'Only GPU device is supported now, got {cfg.device_target}')

    if cfg.file_format and cfg.file_format != 'MINDIR':
        raise ValueError(f'Only MINDIR format is supported for export now, got {cfg.file_format}')
