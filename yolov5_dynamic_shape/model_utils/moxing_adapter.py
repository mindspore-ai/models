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

"""Moxing adapter for ModelArts"""

import os
import functools
import mindspore as ms
from .config import config

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

def sync_data(from_path, to_path):
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

def modelarts_pre_process(args):
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, args.modelarts_dataset_unzip_name)):
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
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if args.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(args.data_path, args.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(args.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    args.output_dir = os.path.join(args.output_path, args.output_dir)
    args.ckpt_path = os.path.join(args.output_path, args.ckpt_path)

def modelarts_post_process():
    sync_data(from_path='/cache/output', to_path='obs://hit-cyf/yolov5_npu/outputs/')

def modelarts_export_preprocess(args):
    args.file_name = os.path.join(args.output_path, args.file_name)

def moxing_wrapper(pre_process=None, post_process=None, **kwargs):
    """
    Moxing wrapper to download dataset and upload outputs.
    """
    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            # Download data from data_url
            if config.enable_modelarts:
                if config.data_url:
                    sync_data(config.data_url, config.data_path)
                    print("Dataset downloaded: ", os.listdir(config.data_path))
                if config.checkpoint_url:
                    sync_data(config.checkpoint_url, config.load_path)
                    print("Preload downloaded: ", os.listdir(config.load_path))
                if config.train_url:
                    sync_data(config.train_url, config.output_path)
                    print("Workspace downloaded: ", os.listdir(config.output_path))

                ms.set_context(save_graphs_path=os.path.join(config.output_path, str(get_rank_id())))
                config.device_num = get_device_num()
                config.device_id = get_device_id()
                if not os.path.exists(config.output_path):
                    os.makedirs(config.output_path)

                if pre_process:
                    if "pre_args" in kwargs.keys():
                        pre_process(*kwargs["pre_args"])
                    else:
                        pre_process()

            # Run the main function
            run_func(*args, **kwargs)

            # Upload data to train_url
            if config.enable_modelarts:
                if post_process:
                    if "post_args" in kwargs.keys():
                        post_process(*kwargs["post_args"])
                    else:
                        post_process()

                if config.train_url:
                    print("Start to copy output directory")
                    sync_data(config.output_path, config.train_url)
        return wrapped_func
    return wrapper
