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
"""post process for 310 inference"""
import os
import argparse
import numpy as np
from src.config import vnet_cfg as cfg
from src.dataset import InferImagelist
from src.utils import evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument("--result_path", type=str, required=True, help="result files path.")
    parser.add_argument("--data_path", type=str, default="./promise", help="Path of dataset, default is ./promise")
    parser.add_argument("--split_file_path", type=str, default="./split/eval.csv",
                        help="Path of dataset, default is ./split/eval.csv")
    args = parser.parse_args()

    dataInferlist = InferImagelist(cfg, args.data_path, args.split_file_path)
    dataManagerInfer = dataInferlist.dataManagerInfer
    for i in range(dataInferlist.__len__()):
        _, img_id = dataInferlist.__getitem__(i)
        result_file = os.path.join(args.result_path, img_id + "_0.bin")
        output = np.fromfile(result_file, dtype=np.float32)
        output = output.reshape(cfg.VolSize[0], cfg.VolSize[1], cfg.VolSize[2])
        print("save predicted label for test '{}'".format(img_id))
        dataManagerInfer.writeResultsFromNumpyLabel(output, img_id, '_test', '.mhd')
    evaluation(os.path.join(args.data_path, 'gt'), cfg['dirPredictionImage'])
