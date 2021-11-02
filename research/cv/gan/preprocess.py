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
''' preprocess '''
import os
import numpy as np
from src.param_parse import parameter_parser

os.makedirs(os.path.join("ascend310_infer", "data_dir"), exist_ok=True)
data_path = os.path.join("ascend310_infer", "data_dir")

if __name__ == "__main__":
    opt = parameter_parser()
    fixed_noise = np.random.normal(size=(10000, opt.latent_dim)).astype(np.float32)
    file_name = "gan_bs.bin"
    file_path = os.path.join(data_path, file_name)
    fixed_noise.tofile(file_path)
    print("*" * 20, "export bin files finished", "*" * 20)
