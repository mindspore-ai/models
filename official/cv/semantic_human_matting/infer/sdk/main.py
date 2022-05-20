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
import os
import time
import argparse
import glob
from config import config as cfg
import numpy as np
from api.infer import SdkApi


def parser_args():
    """ Args Setting """
    parser = argparse.ArgumentParser(description="semantic_human_matting inference")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="convert/semantic_human_matting.pipeline",
        help="image file path. The default is 'convert/shm.pipeline'. ")
    parser.add_argument(
        "--pre_path",
        type=str,
        required=False,
        default="../data/preprocess_Result",
        help="image file path. The default is '../data/preprocess_Result'. ")
    parser.add_argument(
        "--result_path",
        type=str,
        required=False,
        default="/results/result_Files",
        help=
        "cache dir of inference result. The default is 'results/result_Files'.")

    args_ = parser.parse_args()
    return args_


def image_inference(dataset_path, pipeline_path, stream_name, result_dir):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    img_data_plugin_id = 0
    file_list = glob.glob(dataset_path+'/*')
    t0 = time.time()
    for file_name in file_list:
        data = np.fromfile(file_name, dtype=np.float32).reshape(1, 3, 320, 320)
        # set img data 1 3 320 320
        sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0",
                                  data.tobytes(), data.shape, cfg.TENSOR_DTYPE_FLOAT32)
        result = sdk_api.get_result(stream_name)
        trimap = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
        alpha = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype='<f4')
        file_name_split = file_name.split('/')[-1].split('.')
        with open(os.path.join(result_dir, file_name_split[0]+'_0.'+file_name_split[1]), 'wb') as f_write:
            f_write.writelines(trimap)
        with open(os.path.join(result_dir, file_name_split[0]+'_1.'+file_name_split[1]), 'wb') as f_write:
            f_write.writelines(alpha)

    print("total time:", time.time()-t0)


if __name__ == "__main__":
    args = parser_args()
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    preprecess_img = os.path.join(args.pre_path, 'img_data')

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    image_inference(preprecess_img, args.pipeline_path, args.stream_name, args.result_path)
