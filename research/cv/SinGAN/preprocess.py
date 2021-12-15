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
"""preprocess"""
import os
import shutil
import random
import numpy as np
import src.functions as functions
from src.imresize import imresize
from src.config import get_arguments
def preLauch():
    """parse the console argument"""
    random.seed(2)
    parser = get_arguments()
    # Directories.
    parser.add_argument('--output_path', type=str, default='./preprocess_Result', help='eval data dir')
    parser.add_argument('--input_path', type=str, default='./postprocess_Result', help='eval data dir')
    parser.add_argument('--input_dir', type=str, default='data')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--scale_num', type=int, default=0, help='scale_num')
    parser.add_argument('--noise_amp_path', type=str, \
        default='./TrainedModels/thunder/scale_factor=0.750000,alpha=10', help='noise_amp file dir')
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

    I_prev_path = os.path.join(opt.output_path, "I_prev")
    z_curr_path = os.path.join(opt.output_path, "z_curr")
    if os.path.exists(opt.output_path):
        shutil.rmtree(opt.output_path)
        os.makedirs(I_prev_path)
        os.makedirs(z_curr_path)
    else:
        os.makedirs(I_prev_path)
        os.makedirs(z_curr_path)

    scale_num = opt.scale_num
    nzx = reals[scale_num].shape[2]
    nzy = reals[scale_num].shape[3]

    if scale_num == 0:
        z_curr = functions.generate_noise([1, nzx, nzy])
        z_curr = np.broadcast_to(z_curr, (1, 3, z_curr.shape[2], z_curr.shape[3]))
    else:
        z_curr = functions.generate_noise([opt.nc_z, nzx, nzy])

    if scale_num == 0:
        I_prev = np.zeros_like((reals[scale_num]), dtype=np.float32)
    else:
        dir2save = '%s/RandomSamples/%s' % (opt.input_path, opt.input_name[:-4])
        I_prev = np.load('%s/I_curr_%d.npy' % (dir2save, scale_num-1))
        I_prev = imresize(I_prev, 1/opt.scale_factor, opt)
        I_prev = I_prev[:, :, 0:round(reals[scale_num].shape[2]), 0:round(reals[scale_num].shape[3])]
        I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]

    noise_amp = np.load('%s/%d/noise_amp.npy' % (opt.noise_amp_path, scale_num))
    z_in = noise_amp * (z_curr) + I_prev
    I_prev_file_path = os.path.join(I_prev_path, 'I_prev.bin')
    I_prev.tofile(I_prev_file_path)

    z_curr_file_path = os.path.join(z_curr_path, 'z_curr.bin')
    z_in.tofile(z_curr_file_path)

    print("scale %d: export bin files finished!" % (scale_num))
