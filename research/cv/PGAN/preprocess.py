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
""" preprocess """
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)

    parser.add_argument("--pre_result_path", type=str, default="",
                        help="out file path")
    parser.add_argument("--nimages", type=int, default="",
                        help="number of images")
    args_opt, _ = parser.parse_known_args()
    # initialize noise
    fixed_noise = np.random.randn(args_opt.nimages, 512).astype(np.float32)
    file_name = "wgan_bs" + str(args_opt.nimages) + ".bin"
    file_path = os.path.join(args_opt.pre_result_path, file_name)
    alpha = np.array(0.0).astype(np.float32)
    fixed_noise.tofile(file_path)
    file_path = os.path.join(args_opt.pre_result_path, "alpha.bin")
    alpha.tofile(file_path)
    print("*" * 20, "export bin files finished", "*" * 20)
