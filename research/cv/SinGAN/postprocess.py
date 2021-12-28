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
import numpy as np
import matplotlib.pyplot as plt
import src.functions as functions
from src.imresize import imresize
from src.config import get_arguments
def preLauch():
    """parse the console argument"""
    parser = get_arguments()
    # Directories.
    parser.add_argument('--output_path', type=str, default='./postprocess_Result', help='eval data dir')
    parser.add_argument('--input_path', type=str, default='./result_Files', help='eval data dir')
    parser.add_argument('--input_dir', type=str, default='data')
    parser.add_argument('--input_name', help='input image name', default='thunder.jpg')
    parser.add_argument('--scale_num', type=int, default=0, help='scale_num')

    return parser.parse_args()

if __name__ == "__main__":
    opt = preLauch()
    functions.post_config(opt)
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    reals = []
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    f_name = os.path.join(opt.input_path, "z_curr_0.bin")

    scale_num = opt.scale_num
    fake = np.fromfile(f_name, dtype=np.float32).reshape(reals[scale_num].shape)
    dir2save = '%s/RandomSamples/%s' % (opt.output_path, opt.input_name[:-4])
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    np.save('%s/I_curr_%d.npy' % (dir2save, scale_num), fake)
    plt.imsave('%s/%d.png' % (dir2save, scale_num), functions.convert_image_np(fake), vmin=0, vmax=1)
    print("scale %d: post process finished!" % (scale_num))
