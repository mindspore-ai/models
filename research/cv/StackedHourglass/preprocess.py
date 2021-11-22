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
"""preprocess  for 310 inference"""
import os
import numpy as np
from src.utils.inference import get_img
from src.config import parse_args

args = parse_args()
if __name__ == "__main__":
    num_eval = args.num_eval
    num_train = args.train_num_eval
    img_path0 = os.path.join(args.result_path, "00_data")
    img_path1 = os.path.join(args.result_path, "11_data")
    os.makedirs(img_path0)
    os.makedirs(img_path1)
    i = 0
    for anns, img, c, s, n in get_img(num_eval, num_train):
        inp = img / 255
        file_name0 = "StackedHourglass" + str(i) + ".bin"
        file_path0 = os.path.join(img_path0, file_name0)
        np.array([inp], dtype=np.float32).tofile(file_path0)
        file_name1 = "StackedHourglass" + str(i) + ".bin"
        file_path1 = os.path.join(img_path1, file_name1)
        np.array([inp[:, ::-1]], dtype=np.float32).tofile(file_path1)
        i = i + 1
