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
"""postprocess."""
import os
import numpy as np
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def cal_acc():
    """
    Calculate the accuracy of inference results.
    """
    result_path0 = os.path.join(config.post_result_path, "h2o_0.bin")
    result_path1 = os.path.join(config.post_result_path, "h2o_1.bin")
    energy = np.fromfile(result_path0, np.float32).reshape(1,)
    atom_ener = np.fromfile(result_path1, np.float32).reshape(192,)
    print('energy:', energy)
    print('atom_energy:', atom_ener)

    baseline = np.load(config.baseline_path)
    ae = baseline['e']

    if not np.mean((ae - atom_ener.reshape(-1,)) ** 2) < 3e-6:
        raise ValueError("Failed to varify atom_ener")

    print('successful')


if __name__ == '__main__':
    cal_acc()
