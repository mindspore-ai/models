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
""" Create Mindrecord"""
import os
import argparse
import moxing as mox
from src.dataset import data_to_mindrecord_byte_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Yolact++ training")
    parser.add_argument("--train_url", type=str, default="obs://xxx", help="ckpt output dir in obs")
    parser.add_argument("--data_url", type=str, default="obs://xxx", help="mindrecord file path.")
    args_opt = parser.parse_args()

    local_data_url = "/cache/data"
    local_mr_url = "/cache/mr"
    local_pretrained_url = "/cache/weights"
    local_train_url = "/cache/ckpt"

    mox.file.make_dirs(local_data_url)
    mox.file.make_dirs(local_mr_url)
    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(local_pretrained_url)


    filename = "yolact.mindrecord"
    mox.file.copy_parallel(args_opt.data_url, local_data_url)

    local_mr_path = os.path.join(local_mr_url, filename)

    mindrecord_dir = local_mr_url
    mindrecord_file = os.path.join(mindrecord_dir, filename + "0")

    data_to_mindrecord_byte_image("coco", True, filename, mind_path=local_mr_url, coco_path=local_data_url)

    mox.file.copy_parallel(local_mr_url, args_opt.train_url)
