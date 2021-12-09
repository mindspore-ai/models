"""
推理文件
"""

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

import argparse
import json
import os

from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi

SUPPORT_IMG_SUFFIX = (".jpg", ".JPG", ".jpeg", ".JPEG")

current_path = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(
    description="SSD Ghost infer " "example.",
    fromfile_prefix_chars="@",
)

parser.add_argument(
    "--pipeline_path",
    type=str,
    help="mxManufacture pipeline file path",
    default=os.path.join(current_path, "../conf/ssd_ghost.pipeline"),
)
parser.add_argument(
    "--stream_name",
    type=str,
    help="Infer stream name in the pipeline config file",
    default="detection",
)
parser.add_argument(
    "--img_path",
    type=str,
    help="Image pathname, can be a image file or image directory",
    default=os.path.join(current_path, "../coco/val2017"),
)
parser.add_argument(
    "--res_path",
    type=str,
    help="Directory to store the inferred result",
    required=False,
)

args = parser.parse_args()

def infer():
    """Infer images by DVPP + OM.    """
    pipeline_path = args.pipeline_path
    stream_name = args.stream_name.encode()
    img_path = os.path.abspath(args.img_path)
    res_dir_name = args.res_path

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, "rb") as f:
        pipeline_str = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    in_plugin_id = 0
    # Construct the input of the stream
    data_input = MxDataInput()

    if os.path.isfile(img_path) and img_path.endswith(SUPPORT_IMG_SUFFIX):
        file_list = [os.path.abspath(img_path)]
    else:
        file_list = os.listdir(img_path)
        file_list = [
            os.path.join(img_path, img)
            for img in file_list
            if img.endswith(SUPPORT_IMG_SUFFIX)
        ]

    if not res_dir_name:
        res_dir_name = os.path.join(".", "infer_res")

    print(f"res_dir_name={res_dir_name}")
    os.makedirs(res_dir_name, exist_ok=True)
    pic_infer_dict_list = []
    for file_name in file_list:
        with open(file_name, "rb") as f:
            img_data = f.read()
            if not img_data:
                print(f"read empty data from img:{file_name}")
                continue

            data_input.data = img_data
            unique_id = stream_manager_api.SendDataWithUniqueId(
                stream_name, in_plugin_id, data_input
            )
            if unique_id < 0:
                print("Failed to send data to stream.")
                exit()
            infer_result = stream_manager_api.GetResultWithUniqueId(
                stream_name, unique_id, 3000
            )
            if infer_result.errorCode != 0:
                print(
                    "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
                    % (infer_result.errorCode, infer_result.data.decode())
                )
                exit()

            pic_infer_dict_list.extend(
                parse_img_infer_result(file_name, infer_result)
            )

        print(f"Inferred image:{file_name} success!")

    with open(os.path.join(res_dir_name, "det_result.json"), "w") as fw:
        fw.write(json.dumps(pic_infer_dict_list))

    stream_manager_api.DestroyAllStreams()

def parse_img_infer_result(file_name, infer_result):
    """
    Inference result
    """
    obj_list_obj = json.loads(infer_result.data.decode())
    if obj_list_obj == {}:
        return []
    obj_list = obj_list_obj['MxpiObject']

    det_obj_list = []
    for o in obj_list:
        x0, y0, x1, y1 = (
            o.get("x0"),
            o.get("y0"),
            o.get("x1"),
            o.get("y1"),
        )
        bbox_for_map = [x0, y0, x1 - x0, y1 - y0]
        score = o.get("classVec")[0].get("confidence")
        category_id = o.get("classVec")[0].get("classId")
        img_fname_without_suffix = os.path.basename(file_name).split(".")[0]
        image_id = img_fname_without_suffix
        det_obj_list.append(
            dict(
                image_id=image_id,
                bbox=bbox_for_map,
                category_id=category_id,
                score=score,
            )
        )
    return det_obj_list
if __name__ == "__main__":
    infer()
