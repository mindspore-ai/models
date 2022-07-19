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

"""Local adapter"""

import os
import functools

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def moxing_wrapper(pre_process=None):
    """
    Moxing wrapper to download dataset and upload outputs.
    """
    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            config = args[0]
            if config.is_modelarts:
                import moxing as mox
                obs_data_url = config.data_url
                config.data_url = '/home/work/user-job-dir/inputs/data/'
                config.dataset_path = config.data_url
                obs_train_url = config.train_url
                config.train_url = '/home/work/user-job-dir/outputs/model/'
                config.save_checkpoint_path = config.train_url

                if config.run_distribute:
                    if get_rank_id() == 0:
                        mox.file.copy_parallel(obs_data_url, config.data_url)
                        print(f"Successfully Download {obs_data_url} to {config.data_url}")
                else:
                    mox.file.copy_parallel(obs_data_url, config.data_url)
                    print(f"Successfully Download {obs_data_url} to {config.data_url}")

            if pre_process:
                pre_process()

            if config.enable_profiling:
                from mindspore.profiler import Profiler
                profiler = Profiler(output_path='./Profile')

            run_func(*args, **kwargs)

            if config.enable_profiling:
                profiler.analyse()

            # Upload data to train_url
            if config.is_modelarts:
                if config.run_distribute:
                    if get_rank_id() == 0:
                        mox.file.copy_parallel(config.train_url, obs_train_url)
                        print(f"Successfully Upload {config.train_url} to {obs_train_url}")
                else:
                    mox.file.copy_parallel(config.train_url, obs_train_url)
                    print(f"Successfully Upload {config.train_url} to {obs_train_url}")
        return wrapped_func
    return wrapper
