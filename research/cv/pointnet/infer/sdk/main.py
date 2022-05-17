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
    parser = argparse.ArgumentParser(description="pointnet inference")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="../data/datapath_BS1/",
        help=
        "cache dir of datapath. The default is '../data/sdk_result'.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="../data/config/pointnet.pipeline",
        help="image file path. The default is '../data/config/pointnet.pipeline'. ")

    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/sdk_result",
        help=
        "cache dir of inference result. The default is '../data/sdk_result'.")

    args_ = parser.parse_args()
    return args_

def image_inference(dataset_path, pipeline_path, stream_name, result_dir):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_data_plugin_id = 0

    num_classes = 4
    shape_ious = []
    img_path = os.path.join(dataset_path, '00_data')
    label_path = os.path.join(dataset_path, 'labels_ids.npy')
    file_list = []
    file_list1 = glob.glob(img_path+'/*')
    t0 = time.time()
    for i in range(len(file_list1)):
        file_list.append(img_path+'/shapenet_data_bs1_%03d'%i+'.bin')

    for i, file_name in enumerate(file_list):
        time_0 = time.time()
        data = np.fromfile(file_name, dtype=np.float32).reshape(1, 3, -1)
        label = np.load(label_path)
        label = label[i]
        # set img data
        sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0",
                                  data.tobytes(), data.shape, cfg.TENSOR_DTYPE_FLOAT32)
        result = sdk_api.get_result(stream_name)

        data = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        #

        print(file_name)
        pred = data.reshape(1, 2500, -1)
        pred_np = np.argmax(pred, axis=2)
        target_np = label - 1


        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
            print('='*50)
            print("part_ious :{} , time each pic: {}".format(np.mean(part_ious), time.time()-time_0))
            print('='*50)

    print("final Miou {}".format(np.mean(shape_ious)))
    print("total time:", time.time()-t0)

if __name__ == "__main__":
    args = parser_args()
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.dataset_dir, args.pipeline_path, args.stream_name, args.infer_result_dir)
