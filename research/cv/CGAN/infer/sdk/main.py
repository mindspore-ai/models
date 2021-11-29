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
""" Model Main """
import os
import time
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from api.infer import SdkApi
from config import config as cfg

def parser_args():
    """ Args Setting """
    parser = argparse.ArgumentParser(description="cgan inference")

    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/cgan.pipeline",
        help="image file path. The default is 'config/cgan.pipeline'. ")

    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/sdk_result",
        help=
        "cache dir of inference result. The default is '../data/sdk_result'.")

    args_ = parser.parse_args()
    return args_

def image_inference(pipeline_path, stream_name, result_dir):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    img_label_plugin_id = 1

    input_dim = 100
    n_image = 200
    n_col = 20
    n_cow = 10
    latent_code_eval = np.random.randn(n_image, input_dim).astype(np.float32)

    label_eval = np.zeros((n_image, 10)).astype(np.float32)
    for i in range(n_image):
        j = i // n_col
        label_eval[i][j] = 1
    label_eval = label_eval.astype(np.float32)

    fake = []
    for idx in range(n_image):
        img_np = latent_code_eval[idx].reshape(1, 100)
        label_np = label_eval[idx].reshape(1, 10)

        start_time = time.time()

        # set img data
        sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0",
                                  img_np.tobytes(), img_np.shape, cfg.TENSOR_DTYPE_FLOAT32)

        # set label data
        sdk_api.send_tensor_input(stream_name, img_label_plugin_id, "appsrc1",
                                  label_np.tobytes(), label_np.shape, cfg.TENSOR_DTYPE_FLOAT32)

        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time
        print(f"The image({idx}) inference time is {end_time}")
        data = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        data = data.reshape(1, 784)
        fake.append(data)

    fig, ax = plt.subplots(n_cow, n_col, figsize=(10, 5))
    for digit, num in itertools.product(range(n_cow), range(n_col)):
        ax[digit, num].get_xaxis().set_visible(False)
        ax[digit, num].get_yaxis().set_visible(False)

    for i in range(n_image):
        if (i + 1) % n_col == 0:
            print("process ========= {}/200".format(i+1))
        digit = i // n_col
        num = i % n_col
        img = fake[i].reshape((28, 28))
        ax[digit, num].cla()
        ax[digit, num].imshow(img * 127.5 + 127.5, cmap="gray")

    label = 'infer result'
    fig.text(0.5, 0.01, label, ha='center')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    print("===========saving image===========")
    post_result_file = os.path.join(result_dir, 'result.png')
    plt.savefig(post_result_file)

if __name__ == "__main__":
    args = parser_args()
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, args.stream_name, args.infer_result_dir)
