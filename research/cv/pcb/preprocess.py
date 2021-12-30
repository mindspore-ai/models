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


"""pre process for 310 inference"""
import os
from src.dataset import create_dataset
from src.model_utils.config import config

def preprocess():
    """Preprocess"""
    print("Loading query dataset...")
    query_dataset, _ = create_dataset(dataset_name=config.dataset_name, dataset_path=config.dataset_path, \
                                              subset_name="query", batch_size=config.batch_size, \
                                              num_parallel_workers=config.num_parallel_workers)
    print("Loading gallery dataset...")
    gallery_dataset, _ = create_dataset(dataset_name=config.dataset_name, dataset_path=config.dataset_path, \
                                                  subset_name="gallery", batch_size=config.batch_size, \
                                                  num_parallel_workers=config.num_parallel_workers)
    query_img_path = os.path.join(config.preprocess_result_path, "query", "image")
    query_fid_path = os.path.join(config.preprocess_result_path, "query", "fid")
    query_pid_path = os.path.join(config.preprocess_result_path, "query", "pid")
    query_camid_path = os.path.join(config.preprocess_result_path, "query", "camid")
    gallery_img_path = os.path.join(config.preprocess_result_path, "gallery", "image")
    gallery_fid_path = os.path.join(config.preprocess_result_path, "gallery", "fid")
    gallery_pid_path = os.path.join(config.preprocess_result_path, "gallery", "pid")
    gallery_camid_path = os.path.join(config.preprocess_result_path, "gallery", "camid")
    paths = [query_img_path, query_fid_path, query_pid_path, query_camid_path, \
             gallery_img_path, gallery_fid_path, gallery_pid_path, gallery_camid_path]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    print("Processing query dataset...")
    for idx, data in enumerate(query_dataset.create_dict_iterator(output_numpy=True)):
        img = data["image"]
        fid = data["fid"]
        pid = data["pid"]
        camid = data["camid"]
        file_name = config.dataset_name + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(query_img_path, file_name)
        fid_file_path = os.path.join(query_fid_path, file_name)
        pid_file_path = os.path.join(query_pid_path, file_name)
        camid_file_path = os.path.join(query_camid_path, file_name)
        img.tofile(img_file_path)
        fid.tofile(fid_file_path)
        pid.tofile(pid_file_path)
        camid.tofile(camid_file_path)
    print("Processing gallery dataset...")
    for idx, data in enumerate(gallery_dataset.create_dict_iterator(output_numpy=True)):
        img = data["image"]
        fid = data["fid"]
        pid = data["pid"]
        camid = data["camid"]
        file_name = config.dataset_name + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(gallery_img_path, file_name)
        fid_file_path = os.path.join(gallery_fid_path, file_name)
        pid_file_path = os.path.join(gallery_pid_path, file_name)
        camid_file_path = os.path.join(gallery_camid_path, file_name)
        img.tofile(img_file_path)
        fid.tofile(fid_file_path)
        pid.tofile(pid_file_path)
        camid.tofile(camid_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == "__main__":
    preprocess()
