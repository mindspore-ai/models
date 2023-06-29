# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import time
import mindspore as ms


GLOBAL_SYNC_COUNT = 0


def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def version_gt(v1, v2="2.0.0"):
    """
    :param v1: version, format like 1.8.1
    :param v2: version, format like 2.0.0
    :return: v1 >/=/< v2, return True/True/False
    """

    l1 = str(v1).split(".")
    l2 = str(v2).split(".")
    for i in range(min(len(l1), len(l2))):
        if int(l1[i]) == int(l2[i]):
            continue
        elif int(l1[i]) < int(l2[i]):
            return False
        else:
            return True
    return len(l1) >= len(l2)


VERSION_GT_2_0_0 = version_gt(ms.__version__)


def rank_sync(func):
    def wrapper(*args, **kwargs):
        global GLOBAL_SYNC_COUNT
        GLOBAL_SYNC_COUNT += 1
        sync_lock = '/tmp/segformer_sync.lock' + str(GLOBAL_SYNC_COUNT)
        device_id = get_device_id()
        func_name = func.__name__
        if device_id % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            func(*args, **kwargs)
            try:
                os.mknod(sync_lock)
            except (FileExistsError, NotImplementedError) as e:
                print(f"Device {device_id} rank_sync generate lock error, func [{func_name}], exception:{e}")
        wait_count = 0
        while True:
            if os.path.exists(sync_lock):
                if wait_count > 0:
                    print(f"Device {device_id} finish wait, func [{func_name}], lock:{sync_lock}")
                break
            if wait_count % 1800 == 0:
                print(f"Device {device_id} begin to wait, func [{func_name}], lock:{sync_lock}, "
                      f"wait_count:{wait_count}")
            wait_count += 1
            time.sleep(1)

    return wrapper
