# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Moxing adapter for ModelArts"""

import os
import functools
from mindspore import context
from mindspore.profiler import Profiler
from .config import config

_global_sync_count = 0


def get_device_id():
    """
      get device id
       """
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    """
      get device num
       """
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    """
      get rank id
       """
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    """
      get job id
       """
    job_id = os.getenv('JOB_ID')
    job_id = job_id if job_id != "" else "default"
    return job_id


def sync_data(from_path, to_path):
    """
       sync data
       """
    import moxing as mox
    import time
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print("===save flag===")
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)
    print("Finish sync data from {} to {}.".format(from_path, to_path))


def moxing_wrapper(pre_process=None, post_process=None):
    """
       moxing wrapper
       """

    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            if config.enable_modelarts:
                if config.data_url:
                    sync_data(config.data_url, config.local_data_url)
                    print("Dataset downloaded: ", os.listdir(config.local_data_url))
                if config.train_url:
                    sync_data(config.train_url, config.local_train_url)
                    print("Preload downloaded: ", os.listdir(config.local_train_url))
                if config.checkpoint_path:
                    sync_data(config.checkpoint_path, config.local_local_checkpoint_path)
                    print("Preload downloaded: ", config.local_checkpoint_path)
                context.set_context(save_graphs_path=os.path.join(config.local_train_url, str(get_rank_id())))
                config.device_num = get_device_num()
                config.device_id = get_device_id()
                if not os.path.exists(config.local_train_url):
                    os.makedirs(config.local_train_url)
                if pre_process:
                    pre_process()
            if config.enable_profiling:
                profiler = Profiler()
            run_func(*args, **kwargs)
            if config.enable_profiling:
                profiler.analyse()
            if config.enable_modelarts:
                if post_process:
                    post_process()
                if config.train_url:
                    print("Start to copy output directory")
                    sync_data(config.local_train_url, config.train_url)

        return wrapped_func

    return wrapper
