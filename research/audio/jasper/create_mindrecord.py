# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from multiprocessing import Pool
import mindspore.dataset.engine as de
from mindspore.mindrecord import FileWriter
from src.dataset import ASRDataset
from src.config import train_config, symbols


def _exec_task(task_id):
    """
    Execute task with specified task id
    """
    print("exec task {}...".format(task_id))
    # get number of files
    writer = FileWriter(mindrecord_file.format(task_id), 1)
    writer.set_page_size(1 << 25)
    jasper_json = {
        "batch_spect": {"type": "float32", "shape": [1, 64, -1]},
        "batch_script": {"type": "int32", "shape": [-1,]}
    }
    writer.add_schema(jasper_json, "jasper_json")
    output_columns = ["batch_spect", "batch_script"]
    dataset = ASRDataset(data_dir=train_config.DataConfig.Data_dir,
                         manifest_fpaths=train_config.DataConfig.train_manifest,
                         labels=symbols,
                         batch_size=1,
                         train_mode=True)
    ds = de.GeneratorDataset(dataset, output_columns,
                             num_shards=num_tasks, shard_id=task_id)
    dataset_size = ds.get_dataset_size()
    for c, item in enumerate(ds.create_dict_iterator(output_numpy=True)):
        row = {"batch_spect": item["batch_spect"],
               "batch_script": item["batch_script"]}
        writer.write_raw_data([row])
        print(f"{c}/{dataset_size}", flush=True)
    writer.commit()


if __name__ == "__main__":
    mindrecord_file = train_config.DataConfig.mindrecord_format
    mindrecord_dir = os.path.dirname(mindrecord_file)
    if not os.path.isdir(mindrecord_dir):
        os.makedirs(mindrecord_dir)
    num_tasks = 8

    print("Write mindrecord ...")

    task_list = list(range(num_tasks))

    if os.name == 'nt':
        for window_task_id in task_list:
            _exec_task(window_task_id)
    elif num_tasks > 1:
        with Pool(num_tasks) as p:
            p.map(_exec_task, task_list)
    else:
        _exec_task(0)
