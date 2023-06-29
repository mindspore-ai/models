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
import zipfile
from src.model_utils.common import get_device_id, rank_sync


@rank_sync
def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    import moxing as mox
    device_id = get_device_id()

    print(f"Device_id {device_id} begin to sync data, from_path:{from_path}, to_path:{to_path}")
    mox.file.copy_parallel(from_path, to_path)
    print(f"Device_id {device_id} finish sync data, from_path:{from_path}, to_path:{to_path}")

    if "cache/data/" in to_path and os.path.exists(os.path.join(to_path, "cityscapes_segformer_dataset.zip")):
        print(f"Device_id {device_id} begin to unzip cityscapes_segformer_dataset.zip")
        f = zipfile.ZipFile(os.path.join(to_path, "cityscapes_segformer_dataset.zip"), 'r')
        for file in f.namelist():
            f.extract(file, to_path)
        f.close()
        print(f"Device_id {device_id} finish unzip cityscapes_segformer_dataset.zip")


@rank_sync
def check_update_package():
    print("Start to check package version")
    try:
        print("Start to install mmcv.")
        os.system("pip install mmcv==0.2.14")
        print("Install mmcv success.")
    except TypeError as e:
        print("check_update_package error", e)

    print("Finish to check package version")
