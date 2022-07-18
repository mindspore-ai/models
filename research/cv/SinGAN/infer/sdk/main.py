# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import argparse
from api.infer import SdkApi
import matplotlib.pyplot as plt
import numpy as np
import src.functions as functions
from src.imresize import imresize
from config import config as cfg

def preLauch():
    """ Args Setting """
    parser = argparse.ArgumentParser(description="SinGAN inference")

    parser.add_argument("--pipeline_path", type=str, required=False, default="../data/config/SinGAN.pipeline",
                        help=" The default is 'config/SinGAN.pipeline'. ")

    parser.add_argument('--input_dir', type=str, help='input image dir')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--nc_im', type=int, default=3, help='image # channels')
    parser.add_argument('--min_size', type=int, default=25, help='image minimal size at the coarser scale')
    parser.add_argument('--scale_factor_init', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--max_size', type=int, default=250, help='image maximal size at the finer scale')

    parser.add_argument("--infer_result_dir", type=str, required=False,
                        help="cache dir of inference result. The default is '../data/sdk_result'.")
    opt = parser.parse_args()
    return opt

def image_inference(pipeline_path, stream_name, result_dir):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    reals = []

    img_data_plugin_id = 0
    img_label_plugin_id = 1
    opt = preLauch()

    noise_amp = np.load('config/noise_amp7.npy')
    z_curr = np.load('config/z_curr7.npy', allow_pickle=True)
    nzx = (z_curr.shape[2])
    nzy = (z_curr.shape[3])
    z_curr = functions.generate_noise([3, nzx, nzy])
    z_curr = z_curr.reshape(1, 3, 169, 250)

    I_curr = np.load('config/I_curr6.npy', allow_pickle=True)

    images_cur = []
    images_cur.append(I_curr)
    real = functions.read_image(opt)


    functions.adjust_scales2image(real, opt)
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    I_prev = imresize(I_curr.reshape(1, 3, 129, 191), 1/opt.scale_factor, opt)
    I_prev = I_prev[:, :, 0:round(reals[7].shape[2]), 0:round(reals[7].shape[3])]

    I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]

    print("I_prev", I_prev.shape)

    z_in = noise_amp* (z_curr) + I_prev

    sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0", z_in.tobytes(),
                              z_in.shape, cfg.TENSOR_DTYPE_FLOAT32)
    sdk_api.send_tensor_input(stream_name, img_label_plugin_id, "appsrc1", I_prev.tobytes(),
                              I_prev.shape, cfg.TENSOR_DTYPE_FLOAT32)

    result = sdk_api.get_result(stream_name)
    data = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

    data = data.reshape(1, 3, 169, 250)
    plt.imsave('%s/result.png' % (result_dir), functions.convert_image_np(data), vmin=0, vmax=1)

if __name__ == "__main__":
    args = preLauch()
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, args.stream_name, args.infer_result_dir)
