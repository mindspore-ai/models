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
""" load .txt to generate jpg picture. """
import os
import datetime
import numpy as np
from PIL import Image

if __name__ == "__main__":

    imageSize = 64
    nc = 3

    f_name = os.path.join("../data/mxbase_result/result.txt")

    fake = np.loadtxt(f_name, np.float32).reshape(1, nc, imageSize, imageSize)

    img_pil = fake[0, ...].reshape(1, nc, imageSize, imageSize)
    img_pil = img_pil[0].astype(np.uint8).transpose((1, 2, 0))
    img_pil = Image.fromarray(img_pil)
    now_time = datetime.datetime.now()
    img_name = str(now_time.hour) + str(now_time.minute) + str(now_time.second)
    img_pil.save(os.path.join("../data/mxbase_result/", "generated_%s.png") % img_name)

    print("Generate images success!")
