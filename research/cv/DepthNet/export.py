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

import os
import argparse
import numpy as np

from mindspore import Tensor
from mindspore import export, load_checkpoint, load_param_into_net

from src.net import CoarseNet
from src.net import FineNet


def check_folder(input_dir):
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        print("create ", input_dir, " success")
    else:
        print(input_dir, " already exists, no need to create")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindSpore Depth Estimation Demo")
    parser.add_argument(
        "--coarse_or_fine",
        type=str,
        default="coarse",
        choices=["coarse", "fine"],
        help="input coarse or fine, export coarse or fine model",
    )
    args = parser.parse_args()

    model_dir = "./Model"
    check_folder(model_dir)
    ckpt_dir = "./Model/Ckpt"
    check_folder(ckpt_dir)
    mindir_dir = "./Model/MindIR"
    check_folder(mindir_dir)
    air_dir = "./Model/AIR"
    check_folder(air_dir)

    if args.coarse_or_fine == "coarse":
        coarse_net = CoarseNet()
        coarse_net_file_name = os.path.join(ckpt_dir, "FinalCoarseNet.ckpt")
        coarse_param_dict = load_checkpoint(coarse_net_file_name)
        load_param_into_net(coarse_net, coarse_param_dict)
        input_rgb_coarsenet = np.random.uniform(0.0, 1.0, size=[1, 3, 228, 304]).astype(np.float32)
        input_coarse_depth = np.random.uniform(0.0, 1.0, size=[1, 1, 55, 74]).astype(np.float32)
        export(
            coarse_net,
            Tensor(input_rgb_coarsenet),
            file_name=os.path.join(mindir_dir, "FinalCoarseNet"),
            file_format="MINDIR",
        )
        export(
            coarse_net,
            Tensor(input_rgb_coarsenet),
            file_name=os.path.join(air_dir, "FinalCoarseNet"),
            file_format="AIR",
        )
    else:
        fine_net = FineNet()
        fine_net_file_name = os.path.join(ckpt_dir, "FinalFineNet.ckpt")
        fine_param_dict = load_checkpoint(fine_net_file_name)
        load_param_into_net(fine_net, fine_param_dict)
        input_rgb_finenet = np.random.uniform(0.0, 1.0, size=[1, 3, 228, 304]).astype(np.float32)
        input_coarse_depth = np.random.uniform(0.0, 1.0, size=[1, 1, 55, 74]).astype(np.float32)
        export(
            fine_net,
            Tensor(input_rgb_finenet),
            Tensor(input_coarse_depth),
            file_name=os.path.join(mindir_dir, "FinalFineNet"),
            file_format="MINDIR",
        )
        export(
            fine_net,
            Tensor(input_rgb_finenet),
            Tensor(input_coarse_depth),
            file_name=os.path.join(air_dir, "FinalFineNet"),
            file_format="AIR",
        )
