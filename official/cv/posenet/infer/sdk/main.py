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
"""main"""

import argparse
import os
import time
import cv2
from api.infer import SdkApi
from config import config as cfg


def parser_args():
    """parser_args"""
    parser = argparse.ArgumentParser(description="fairmot inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="/home/data/xd_mindx/rsd/infer/sdk/config/posenet.pipeline",
        help="image file path. The default is '/home/data/xd_mindx/czp/fairmot/infer/sdk/config/fairmot.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=False,
        default="infer",
        help=
        "infer:only infer, eval: accuracy evaluation. The default is 'infer'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="./infer_result",
        help=
        "cache dir of inference result. The default is '../data/infer_result'."
    )
    arg = parser.parse_args()
    return arg


def process_img(img_file):
    #img2 = Image.open(img_file)
    #img1 = img2.resize((224, 455),Image.ANTIALIAS)
    img = cv2.imread(img_file)
    #pad_img = cv2.resize(img, (224, 455), interpolation=cv2.INTER_AREA)
    #img,__, _, _ = letterbox(img0, height=cfg.MODEL_HEIGHT, width=cfg.MODEL_WIDTH)
    return img

def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    """resize a rectangular image to a padded rectangular"""
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def image_inference(pipeline_path, stream_name, img_dir, result_dir):
    """image_inference"""
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    print(stream_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_data_plugin_id = 0
    i = 0
    print(f"\nBegin to inference for {img_dir}.\n")
    file_list = os.listdir(img_dir)
    txt_file = "../dataset_test.txt"
    with open(txt_file, 'r') as f:
        next(f)
        next(f)
        next(f)
        for line in f:
            fname, _, _, _, _, _, _, _ = line.split()
            file_name1 = fname.split('/')[0] # seq
            file_name2 = fname.split('/')[1] # frame0009
            for filedirname in os.listdir(img_dir): # bianli file name
                if filedirname == file_name1:
                    files = img_dir + "/" + file_name1
                    file_list = os.listdir(files)
                    for img_id, file_name in enumerate(file_list):
                        total_len = len(file_list)
                        if file_name == file_name2:
                            file_name1 = file_name1 + "/" + file_name
                            file_path = os.path.join(img_dir, file_name1)
                            file_binname0 = "posenet" + "_" + str(i) + "_0.bin"
                            file_binname1 = "posenet" + "_" + str(i) + "_1.bin"
                            file_binname2 = "posenet" + "_" + str(i) + "_2.bin"
                            file_binname3 = "posenet" + "_" + str(i) + "_3.bin"
                            file_binname4 = "posenet" + "_" + str(i) + "_4.bin"
                            file_binname5 = "posenet" + "_" + str(i) + "_5.bin"
                            i = i + 1
                            save_path = [os.path.join(result_dir, file_binname0),
                                         os.path.join(result_dir, file_binname1),
                                         os.path.join(result_dir, file_binname2),
                                         os.path.join(result_dir, file_binname3),
                                         os.path.join(result_dir, file_binname4),
                                         os.path.join(result_dir, file_binname5)]
                            img_np = process_img(file_path)
                            img_shape = img_np.shape
                            sdk_api.send_img_input(stream_name,
                                                   img_data_plugin_id, "appsrc0",
                                                   img_np.tobytes(), img_shape)
                            start_time = time.time()
                            result = sdk_api.get_result(stream_name)
                            end_time = time.time() - start_time
                            with open(save_path[0], "wb") as fp:
                                fp.write(result[0])
                                print(
                                    f"End-2end inference, file_name: {file_path}, "
                                    f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
                                )
                            with open(save_path[1], "wb") as fp:
                                fp.write(result[1])
                                print(
                                    f"End-2end inference, file_name: {file_path}, "
                                    f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
                                )
                            with open(save_path[2], "wb") as fp:
                                fp.write(result[2])
                                print(
                                    f"End-2end inference, file_name: {file_path}, "
                                    f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
                                )
                            with open(save_path[3], "wb") as fp:
                                fp.write(result[3])
                                print(
                                    f"End-2end inference, file_name: {file_path}, "
                                    f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
                                )
                            with open(save_path[4], "wb") as fp:
                                fp.write(result[4])
                                print(
                                    f"End-2end inference, file_name: {file_path}, "
                                    f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
                                )
                            with open(save_path[5], "wb") as fp:
                                fp.write(result[5])
                                print(
                                    f"End-2end inference, file_name: {file_path}, "
                                    f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
                                )




if __name__ == "__main__":
    args = parser_args()
    seqs_str = '''MOT20-01
      MOT20-02
     MOT20-03
     MOT20-05
     '''
    image_inference(args.pipeline_path, cfg.STREAM_NAME.encode("utf-8"), args.img_path,
                    args.infer_result_dir)
