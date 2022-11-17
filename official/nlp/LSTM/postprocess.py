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
"""
#################lstm postprocess########################
"""
import os
import numpy as np
from mindspore.nn import Accuracy
from src.model_utils.config import config


if __name__ == '__main__':
    metrics = Accuracy()
    rst_path = config.result_dir
    labels = np.load(config.label_dir)

    for i in range(len(os.listdir(rst_path))):
        file_name = os.path.join(rst_path, "LSTM_data_bs" + str(config.batch_size) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float32).reshape(config.batch_size, config.num_classes)
        metrics.update(output, labels[i])

    print("result of Accuracy is: ", metrics.eval())
