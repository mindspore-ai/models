# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import itertools
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('proprocess')
parser.add_argument('--input_path', type=str, default='./result_Files/', help='eval data dir')
parser.add_argument('--output_path', type=str, default='./postprocess_Result/', help='eval data dir')
args = parser.parse_args()

if __name__ == "__main__":

    f_name = os.path.join(args.input_path, "cgan_bs" + str(200) + "_0.bin")
    fake = np.fromfile(f_name, dtype=np.float32).reshape((-1, 1024))

    fig, ax = plt.subplots(10, 20, figsize=(10, 5))
    for digit, num in itertools.product(range(10), range(20)):
        ax[digit, num].get_xaxis().set_visible(False)
        ax[digit, num].get_yaxis().set_visible(False)

    for i in range(200):
        if (i + 1) % 20 == 0:
            print("process ========= {}/200".format(i+1))
        digit = i // 20
        num = i % 20
        img = fake[i].reshape((32, 32))
        ax[digit, num].cla()
        ax[digit, num].imshow(img * 127.5 + 127.5, cmap="gray")

    label = 'infer result'
    fig.text(0.5, 0.01, label, ha='center')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    print("===========saving image===========")
    post_result_file = os.path.join(args.output_path, 'result.png')
    plt.savefig(post_result_file)
