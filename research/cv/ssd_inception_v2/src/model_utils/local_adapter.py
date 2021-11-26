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
    """
    Returns device ID

    Returns:
        Device ID
    """
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    """
    Get number of devices

    Returns:
        Number of devices

    """
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    """
        Get id of current device
    Returns:
        Current device ID
    """
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    """
    Returns string "Local Job"

    Returns:
        "Local Job"
    """
    return "Local Job"
