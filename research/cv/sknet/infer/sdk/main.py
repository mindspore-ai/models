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
# ============================================================================
""" Model Main """
import argparse
import os
import time
import numpy as np
from infer import SdkApi
import cv2


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="sknet process")
    parser.add_argument("--pipeline_path", type=str, default="../data/config/sknet.pipeline", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="../data/image/test_batch/",
                        help="Dataset contain batch_spare batch_label batch_dense")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="cache dir of inference result. The default is './results'.")
    args_opt = parser.parse_args()
    return args_opt


def image_proc(img_path, bgr2rgb=False, dsize=(224, 224), rescale=1.0 / 255.0, mean=(0.4914, 0.4822, 0.4465),
               std=(0.2023, 0.1994, 0.2010)):
    ''' get input data '''
    data_numpy = cv2.imread(
        img_path, cv2.IMREAD_UNCHANGED)

    if bgr2rgb:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    if data_numpy is None:
        raise ValueError('Fail to read {}'.format(img_path))

    data_numpy = cv2.resize(data_numpy, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    data_numpy = data_numpy * rescale
    data_numpy = (data_numpy - mean) / std
    data_numpy = data_numpy.transpose(2, 0, 1)
    return data_numpy


def inference(pipeline_path, stream_name, batch_size):
    args_ = parse_args()
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    dir_name = args_.data_dir
    file_list = os.listdir(dir_name)
    if not os.path.exists(args_.result_dir):
        os.makedirs(args_.result_dir)
    cnt = 0
    data_input = list()
    file_name_list = list()
    if os.path.exists("result.txt"):
        os.remove("result.txt")
    for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg") or
                file_name.lower().endswith(".png")):
            continue
        file_name_list.append(file_name.split(".")[0])
        start_time = time.time()
        data_input.append(image_proc(file_path))
        cnt += 1
        if cnt % batch_size != 0:
            continue
        data_input = np.array(data_input, dtype=np.float32)
        sdk_api.send_tensor_input(stream_name, 0, "appsrc0", data_input, 0, batch_size)
        data_input = list()
        result = sdk_api.get_result(stream_name)
        pred = np.array(
            [np.frombuffer(result.tensorPackageVec[i].tensorVec[0].dataStr, dtype=np.float32) for i in
             range(batch_size)])
        with open('result.txt', 'a+') as f:
            for i in range(batch_size):
                f.write(file_name_list[i] + ":")
                for j in range(10):
                    f.write(str(pred[i][j]) + " ")
                f.write("\n")
        file_name_list.clear()
        end_time = time.time() - start_time
        print(f"image {cnt - batch_size}:{cnt} run time {end_time}")


def save_result(result_dir, top_n):
    with open("result.txt", "r") as f:
        for line in f.readlines():
            file_name = line.split(":")[0]
            data = line.split(":")[1].strip().split(" ")
            confidences = {}
            for i in range(len(data)):
                confidences[i] = np.float32(data[i])
            confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
            top = list()
            for i in range(top_n):
                top.append(str(confidences[i][0]))
            with open(result_dir + "/" + file_name + ".txt", "w") as f_write:
                f_write.writelines(" ".join(top))


if __name__ == "__main__":
    args = parse_args()
    args.stream_name = b'im_sknet'
    inference(args.pipeline_path, args.stream_name, args.batch_size)
    save_result(args.result_dir, 5)
