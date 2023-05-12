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

"""Local adapter"""

import os

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices is None:
        return int(device_id)
    if not isinstance(cuda_visible_devices, int):
        return 0
    if int(device_id) != 0 and int(device_id) != int(cuda_visible_devices):
        raise ValueError(f"CUDA_VISIBLE_DEVICES:{cuda_visible_devices} is different from "
                         f"DEVICE_ID:{device_id}, please unset CUDA_VISIBLE_DEVICES")
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    return "Local Job"
