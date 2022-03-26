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

"""Dataset crop"""

import os
import argparse
from concurrent import futures
import cv2
import pandas as pd


def crop_one(input_img_path, output_img_path):
    """
    center crop one image
    """

    image = cv2.imread(input_img_path)
    img_shape = image.shape
    img_height = img_shape[0]
    img_width = img_shape[1]

    if (img_width < 200) or (img_height < 200):
        os.remove(input_img_path)
    else:
        cropped = image[(img_height - 200) // 2:(img_height + 200) // 2, (img_width - 200) // 2:(img_width + 200) // 2]
        cv2.imwrite(output_img_path, cropped)


def crop(data_root, output_path):
    """
    crop all images with thread pool
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError("data root not exist: " + data_root)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_filenames = [(os.path.join(data_root, x), os.path.join(output_path, x))
                       for x in os.listdir(data_root)]
    file_list = []
    for file in image_filenames:
        file_list.append(file)
    print(len(file_list))
    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as tp:
        all_task = [tp.submit(crop_one, file[0], file[1]) for file in file_list]
        futures.wait(all_task)
    print("all done!")


def save(data_root, output_path):
    file_list = []
    for path in os.listdir(data_root):
        _, filename = os.path.split(path)
        file_list.append(filename)
    df = pd.DataFrame(file_list, columns=["one"])
    df.to_csv(os.path.join(output_path, "test.lst"), columns=["one"], index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Image to 200*200")
    parser.add_argument("--data_name", type=str, help="dataset name", required=True,
                        choices=["ECSSD", "SOD", "DUT-OMRON", "PASCAL-S", "HKU-IS", "DUTS-TE", "DUTS-TR"])
    parser.add_argument("--data_root", type=str, help="root of images", required=True,
                        default="/home/data")
    parser.add_argument("--output_path", type=str, help="output path of cropped images", required=True,
                        default="/home/data")
    args = parser.parse_known_args()[0]
    if args.data_name == "DUTS-TE":
        Mask = "DUTS-TE-Mask"
        Image = "DUTS-TE-Image"
    elif args.data_name == "DUTS-TR":
        Mask = "DUTS-TR-Mask"
        Image = "DUTS-TR-Image"
    else:
        Mask = "ground_truth_mask"
        Image = "images"
    crop(os.path.join(args.data_root, args.data_name, Mask),
         os.path.join(args.output_path, args.data_name, Mask))
    crop(os.path.join(args.data_root, args.data_name, Image),
         os.path.join(args.output_path, args.data_name, Image))
    save(os.path.join(args.output_path, args.data_name, Image),
         os.path.join(args.output_path, args.data_name))
