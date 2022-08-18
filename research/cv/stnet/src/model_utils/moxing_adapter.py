# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""Moxing adapter for ModelArts"""

import os
import functools
from mindspore import context
from src.config import config as cfg

_global_sync_count = 0

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

def unzip(zip_file, save_dir):
    """
    unzip function
    """
    import zipfile
    import time
    s_time = time.time()

    zip_isexist = zipfile.is_zipfile(zip_file)
    if zip_isexist:
        fz = zipfile.ZipFile(zip_file, 'r')
        data_num = len(fz.namelist())
        print("Extract Start...")
        print("unzip file num: {}".format(data_num))
        data_print = int(data_num / 100) if data_num > 100 else 1
        i = 0
        for file in fz.namelist():
            if i % data_print == 0:
                print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
            i += 1
            fz.extract(file, save_dir)
        print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60), int(int(time.time() - s_time) % 60)))
        print("Extract Done.")
    else:
        print("This is not zip.")


def sync_data(from_path, to_path, mode=0):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    import moxing as mox
    import time
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        if mode == 1:
            for idx in range(16):
                i = idx + 1
                pack_name = str(i) + ".zip"
                print("now load train %s" % i)
                print("load train pre", flush=True)
                mox.file.copy_parallel(os.path.join(from_path, "train", pack_name), os.path.join(to_path, "train",
                                                                                                 pack_name))
                print("load train after", flush=True)

                path_train_2 = os.path.join(to_path, "train", pack_name)
                path_train_3 = os.path.join(to_path, "train")

                if os.path.exists(path_train_2):
                    os.system("file %s" % path_train_2)
                    unzip(path_train_2, path_train_3)
                    os.system('rm -f %s' % path_train_2)
                print('%s success' % i)

            for idx in range(2):
                i = idx + 1
                pack_name = str(i) + ".zip"
                print("now load val %s" % i)
                print("load val pre", flush=True)
                mox.file.copy_parallel(os.path.join(from_path, "val", pack_name), os.path.join(to_path, "val",
                                                                                               pack_name))
                print("load val after", flush=True)
                path_val_2 = os.path.join(to_path, "val", pack_name)
                path_val_3 = os.path.join(to_path, "val")

                if os.path.exists(path_val_2):
                    os.system("file %s" % path_val_2)
                    unzip(path_val_2, path_val_3)
                print('%s success' % i)
        else:
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
    Moxing wrapper to download dataset and upload outputs.
    """
    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            # Download data from data_url
            config = cfg
            if config.run_online:
                if config.data_url:
                    sync_data(config.data_url, config.local_data_url, mode=1)
                    print("Dataset downloaded: ", os.listdir(config.local_data_url))
                if config.pre_url:
                    sync_data(config.pre_url, config.load_path)
                    print("Preload downloaded: ", os.listdir(config.load_path))
                if config.train_url:
                    sync_data(config.train_url, config.output_path)
                    print("Workspace downloaded: ", os.listdir(config.output_path))

                context.set_context(save_graphs_path=os.path.join(config.output_path, str(get_rank_id())))

                if not os.path.exists(config.output_path):
                    os.makedirs(config.output_path)

                if pre_process:
                    pre_process()

            run_func(*args, **kwargs)

            # Upload data to train_url
            if config.run_online:
                if post_process:
                    post_process()

                if config.train_url:
                    print("Start to copy output directory")
                    sync_data(config.output_path, config.train_url)
        return wrapped_func
    return wrapper
