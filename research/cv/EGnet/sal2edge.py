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

"""Extract edge"""

import os
import argparse
from concurrent import futures
import cv2
import numpy as np


def sal2edge_one(image_file, output_file):
    """
    process one image
    """
    if not os.path.exists(image_file):
        print("file not exist:", image_file)
        return
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    b_image = image > 128
    b_image = b_image.astype(np.float64)
    dx, dy = np.gradient(b_image)
    temp_edge = dx * dx + dy * dy
    temp_edge[temp_edge != 0] = 255
    bound = temp_edge.astype(np.uint8)
    cv2.imwrite(output_file, bound)


def sal2edge(data_root, output_path, image_list_file):
    """
    extract edge from salience image (use thread pool)
    """
    if not os.path.exists(data_root):
        print("data root not exist", data_root)
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(image_list_file):
        print("image list file not exist", image_list_file)
        return
    image_list = np.loadtxt(image_list_file, str)
    file_list = []
    ext = ".png"
    for image in image_list:
        file_list.append(image[:-4])
    pair_file = open(data_root+"/../train_pair_edge.lst", "w")
    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tp:
        all_task = []
        for file in file_list:
            img_path = os.path.join(data_root, file + ext)
            result_path = os.path.join(output_path, file + "_edge" + ext)
            all_task.append(tp.submit(sal2edge_one, img_path, result_path))
            pair_file.write(f"DUTS-TR-Image/{file}.jpg DUTS-TR-Mask/{file}.png DUTS-TR-Mask/{file}_edge.png\n")
        futures.wait(all_task)
    pair_file.close()
    print("all done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Covert Salient Image to Edge Image")
    parser.add_argument("--data_root", type=str, help="root of salient images", required=True)
    parser.add_argument("--output_path", type=str, help="output path of edge images", required=True)
    parser.add_argument("--image_list_file", type=str, help="image list of salient images", required=True)
    args = parser.parse_known_args()[0]
    sal2edge(args.data_root, args.output_path, args.image_list_file)
