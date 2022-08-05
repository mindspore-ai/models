# Copyright 2022 Huawei Technologies Co., Ltd
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
from .param import param


_global_syn_count = 0


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    job_id = os.getenv('JOB_ID')
    job_id = job_id if job_id != "" else "default"
    return job_id


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local
    Uploca data from local directory to remote obs in contrast
    """
    import moxing as mox
    import time
    global _global_syn_count
    sync_lock = '/tmp/copy_sync.lock' + str(_global_syn_count)
    _global_syn_count += 1

    # Each server contains 8 devices as most
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print('from path: ', from_path)
        print('to path: ', to_path)
        mox.file.copy_parallel(from_path, to_path)
        print('===finished data synchronization===')
        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print('===save flag===')

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)
    print('Finish sync data from {} to {}'.format(from_path, to_path))


def moxing_wrapper(pre_process=None, post_process=None):
    """
    Moxing wrapper to download dataset and upload outputs
    """
    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            # Download data from data_url
            if param.enable_modelarts:
                if param.data_url:
                    sync_data(param.data_url, param.data_path)
                    print('Dataset downloaded: ', os.listdir(param.data_path))
                if param.checkpoint_url or param.ckpt_url:
                    if not os.path.exists(param.load_path):
                        os.makedirs(param.load_path)
                        print('=' * 20 + 'makedirs')
                        if os.path.isdir(param.load_path):
                            print('=' * 20 + 'makedirs success')
                        else:
                            print('=' * 20 + 'makedirs fail')
                    if param.checkpoint_url:
                        sync_data(param.checkpoint_url, param.load_path)
                    else:
                        sync_data(os.path.dirname(param.ckpt_url), param.load_path)
                    print('Preload downloaded: ', os.listdir(param.load_path))
                if param.train_url:
                    sync_data(param.train_url, param.output_path)
                    print('Workspace downloaded: ', os.listdir(param.output_path))

                context.set_context(save_graphs_path=os.path.join(param.output_path, str(get_rank_id())))
                param.device_num = get_device_num()
                param.device_id = get_device_id()
                if not os.path.exists(param.output_path):
                    os.makedirs(param.output_path)

                if pre_process:
                    pre_process()

            run_func(*args, **kwargs)

            # Upload data to train_url
            if param.enable_modelarts:
                if post_process:
                    post_process()

                if param.train_url:
                    print('Start to copy output directory')
                    sync_data(param.output_path, param.train_url)

                if param.result_url:
                    print('Start to copy output directory')
                    sync_data(param.output_path, param.result_url)

        return wrapped_func
    return wrapper
