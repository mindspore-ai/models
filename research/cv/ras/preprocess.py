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
import argparse
import numpy as np
from src.dataset_test import TrainDataLoader

def parse(arg=None):
    """Define configuration of preprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str)
    return parser.parse_args(arg)

def preprocess_data():
    """ preprocess data """
    testdataloader = TrainDataLoader(args.dataroot)
    Names = []
    for data in os.listdir(args.dataroot):
        name = data.split(".")[0]
        Names.append(name)
    Names = sorted(Names)
    i = 0
    for data in testdataloader.dataset.create_dict_iterator():
        data, data_org = data["data"], data["data_org"]
        file_name = Names[i]

        data_name = os.path.join("./preprocess_Data/data/", file_name + ".bin")
        data_shape_name = os.path.join("./preprocess_Data/data_shape/", file_name + ".bin")
        data.asnumpy().tofile(data_name)
        data_shape = np.array([data_org.shape[1], data_org.shape[2]]).astype(np.int64)
        data_shape.tofile(data_shape_name)
        i += 1
if __name__ == "__main__":
    args = parse()
    preprocess_data()
