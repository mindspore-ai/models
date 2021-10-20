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
""" postprocess """
import os
import argparse
import numpy as np
from PIL import Image
from src.image_transform import Normalize, TransporeAndMul, Resize

def resizeTensor(data, out_size_image=(128, 128)):
    """resizeTensor

    Returns:
        output.
    """
    out_data_size = (data.shape[0], data.shape[
        1], out_size_image[0], out_size_image[1])
    outdata = []
    data = np.clip(data, a_min=-1, a_max=1)
    transformList = [Normalize((-1., -1., -1.), (2, 2, 2)), TransporeAndMul(), Resize(out_size_image)]
    for img in range(out_data_size[0]):
        processed = data[img]
        for transform in transformList:
            processed = transform(processed)
        processed = np.array(processed)
        outdata.append(processed)
    return outdata
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--post_result_path", type=str, default="", help="result path")
    parser.add_argument("--output_dir", type=str, default="", help="output file path")
    parser.add_argument("--nimages", type=int, default="", help="number of images")
    args_opt, _ = parser.parse_known_args()
    if not os.path.exists(args_opt.output_dir):
        os.mkdir(args_opt.output_dir)
    f_name = os.path.join(args_opt.post_result_path, "wgan_bs" + str(64) + "_0.bin")
    fake = np.fromfile(f_name, np.float32).reshape(args_opt.nimages, 3, 128, 128)
    fake = resizeTensor(fake)
    for i in range(args_opt.nimages):
        img_pil = fake[i]
        img_pil = Image.fromarray(img_pil)
        img_pil.save(os.path.join(args_opt.output_dir, "generated_%02d.png" % i))

    print("Generate images success!")
