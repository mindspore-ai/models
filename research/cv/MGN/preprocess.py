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
import numpy as np
from src.dataset import create_dataset
from model_utils.config import get_config
from model_utils.device_adapter import get_device_id
from mindspore import context

config_path = "../configs/market1501_config.yml"
config = get_config()


def text_save(filename, data):
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("save successfully")


def fliphor(tensor):
    """ Flip tensor """
    return tensor[..., ::-1].copy()


def save_data_to_bin(data_path, output_path):
    config.image_size = list(map(int, config.image_size.split(',')))
    config.image_mean = list(map(float, config.image_mean.split(',')))
    config.image_std = list(map(float, config.image_std.split(',')))

    _enable_graph_kernel = False
    context.set_context(
        mode=context.GRAPH_MODE,
        enable_graph_kernel=_enable_graph_kernel,
        device_target=config.device_target,
    )

    config.rank = 0
    config.device_id = get_device_id()
    config.group_size = 1

    t_dataset, t_cams, t_ids = create_dataset(
        data_path,
        ims_per_id=4,
        ids_per_batch=12,
        mean=config.image_mean,
        std=config.image_std,
        resize_h_w=config.image_size,
        batch_size=config.per_batch_size,
        rank=config.rank,
        group_size=config.group_size,
        data_part='test'
    )

    q_dataset, q_cams, q_ids = create_dataset(
        data_path,
        ims_per_id=4,
        ids_per_batch=12,
        mean=config.image_mean,
        std=config.image_std,
        resize_h_w=config.image_size,
        batch_size=config.per_batch_size,
        rank=config.rank,
        group_size=config.group_size,
        data_part='query'
    )
    output_test_path = os.path.join(output_path, "test")
    output_query_path = os.path.join(output_path, "query")

    test_img_path = os.path.join(output_test_path, "dataset")
    os.makedirs(test_img_path)
    label_list = []
    idx = 0
    for idx, data in enumerate(t_dataset.create_dict_iterator(output_numpy=True)):
        if data["image"].shape[0] == config.per_batch_size:
            file_name = "market1501_test_bs" + str(config.per_batch_size) + "_" + str(idx) + "_0" + ".bin"
            file_path = os.path.join(test_img_path, file_name)
            data["image"].tofile(file_path)

            images_ = data["image"]
            images_ = fliphor(images_)
            file_name = "market1501_test_bs" + str(config.per_batch_size) + "_" + str(idx) + "_1" + ".bin"
            file_path = os.path.join(test_img_path, file_name)
            images_.tofile(file_path)

            label_list.append(data["label"])

    np.save(os.path.join(output_test_path, "market1501_label_ids.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)

    file_path = os.path.join(output_test_path, 't_cams.txt')
    text_save(file_path, t_cams[:idx*config.per_batch_size])

    file_path = os.path.join(output_test_path, 't_ids.txt')
    text_save(file_path, t_ids[:idx*config.per_batch_size])


    query_img_path = os.path.join(output_query_path, "dataset")
    os.makedirs(query_img_path)
    label_list = []
    for idx, data in enumerate(q_dataset.create_dict_iterator(output_numpy=True)):
        if data["image"].shape[0] == config.per_batch_size:
            file_name = "market1501_query_bs" + str(config.per_batch_size) + "_" + str(idx) + "_0" + ".bin"
            file_path = os.path.join(query_img_path, file_name)
            data["image"].tofile(file_path)

            images_ = data["image"]
            images_ = fliphor(images_)
            file_name = "market1501_query_bs" + str(config.per_batch_size) + "_" + str(idx) + "_1" + ".bin"
            file_path = os.path.join(query_img_path, file_name)
            images_.tofile(file_path)

            label_list.append(data["label"])
    np.save(os.path.join(output_query_path, "market1501_label_ids.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)

    file_path = os.path.join(output_query_path, 'q_cams.txt')
    text_save(file_path, q_cams[:idx*config.per_batch_size])

    file_path = os.path.join(output_query_path, 'q_ids.txt')
    text_save(file_path, q_ids[:idx*config.per_batch_size])


if __name__ == '__main__':
    save_data_to_bin(config.data_dir, config.output_path)
