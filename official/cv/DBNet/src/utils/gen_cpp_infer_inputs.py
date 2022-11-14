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
"""Generate bin file as cpp infer inputs."""
import os
import sys

import numpy as np
from tqdm import tqdm

from src.datasets.load import create_dataset
from src.model_utils.config import config

sys.path.insert(0, os.path.join("../..", os.path.dirname(os.path.abspath(__file__))))


def main():
    val_dataset, _ = create_dataset(config, False)
    val_dataset = val_dataset.create_dict_iterator(output_numpy=True)

    data_path = config.output_dir
    os.makedirs(data_path, exist_ok=True)

    input_dir = os.path.join(data_path, "eval_input_bin")
    os.makedirs(input_dir)

    ori_dir = os.path.join(data_path, "eval_ori")
    os.makedirs(ori_dir)

    polys_dir = os.path.join(data_path, "eval_polys_bin")
    os.makedirs(polys_dir)

    dontcare_dir = os.path.join(data_path, "eval_dontcare_bin")
    os.makedirs(dontcare_dir)

    # Record the shape, because when numpy read binary file(np.fromfile()), the shape should be given.
    # Otherwise, the data would be wrong
    shape_recorder = open(os.path.join(data_path, "eval_shapes"), 'w')

    for i, data in tqdm(enumerate(val_dataset)):
        input_name = "eval_input_" + str(i + 1) + ".bin"
        polys_name = "eval_polys_" + str(i + 1) + ".bin"
        dontcare_name = "eval_dontcare_" + str(i + 1) + ".bin"

        input_path = os.path.join(input_dir, input_name)
        polys_path = os.path.join(polys_dir, polys_name)
        dontcare_path = os.path.join(dontcare_dir, dontcare_name)

        np.save(os.path.join(ori_dir, f"eval_org_{i+1}"), data['original'])
        data['img'].tofile(input_path)
        data['polys'].tofile(polys_path)
        data['dontcare'].tofile(dontcare_path)
        shape_recorder.write(str(data['polys'].shape) + "\n")
        shape_recorder.write(str(data['dontcare'].shape) + "\n")
        shape_recorder.write(str(data['img'].shape) + "\n")

    shape_recorder.close()

    print("finished")


if __name__ == '__main__':
    main()
