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
"""pre process for 310 inference"""
import argparse
import os
import time

from PIL import Image
import numpy as np

import delf_config

parser = argparse.ArgumentParser(description='MindSpore delf Example')

parser.add_argument('--use_list_txt', type=str, default="False", choices=['True', 'False'])
parser.add_argument('--list_images_path', type=str, default="list_images.txt")

parser.add_argument('--output_path', type=str, default="")
parser.add_argument('--size_path', type=str, default="")
parser.add_argument('--images_path', type=str, default="")
parser.add_argument('--keep_max_num', type=int, default=10)

args = parser.parse_known_args()[0]

def ReadImageList(list_path):
    f = open(list_path, "r")
    image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

def main():
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.size_path):
        os.makedirs(args.size_path)
    # Read list of images.
    print('Preprocess: Reading list of images...')
    if args.use_list_txt == "True":
        image_paths = ReadImageList(args.list_images_path)
    else:
        names = os.listdir(args.images_path)
        image_paths = []
        for name in names:
            image_name = name.replace('.jpg', '')
            image_paths.append(image_name)

    num_images = len(image_paths)
    print(f'done! Found {num_images} images')

    # extract config
    config = delf_config.config()
    image_scales_tensor = np.array(config.image_scales, np.float32)

    for i in range(num_images):
        # Report progress once in a while.
        if i == 0:
            print('Starting to extract DELF features from images...')
        temp_files = os.listdir(args.output_path)
        while len(temp_files) >= 10:
            time.sleep(5)
            temp_files = os.listdir(args.output_path)

        img = Image.open(os.path.join(args.images_path, image_paths[i]) + '.jpg')

        im = np.array(img, np.float32)
        original_image_shape = np.array([im.shape[0], im.shape[1]])
        original_image_shape_float = original_image_shape.astype(np.float32)
        new_image_size = np.array([2048, 2048])

        images_batch = np.zeros((image_scales_tensor.shape[0], 3, 2048, 2048), np.float32)
        size_list = np.zeros((image_scales_tensor.shape[0], 2), int)
        # generate images pyramids
        for j in range(image_scales_tensor.shape[0]):
            scale_size = np.round(original_image_shape_float * image_scales_tensor[j]).astype(int)
            size_list[j] = scale_size
            img_pil = img.resize((scale_size[1], scale_size[0]))
            scale_image = np.array(img_pil, np.float32)

            H_pad = new_image_size[0] - scale_size[0]
            W_pad = new_image_size[1] - scale_size[1]
            new_image = np.pad(scale_image, ((0, H_pad), (0, W_pad), (0, 0)))

            new_image = (new_image-128.0) / 128.0

            perm = (2, 0, 1)
            new_image = np.transpose(new_image, perm)
            new_image = np.expand_dims(new_image, 0)
            images_batch[j] = new_image

        print('preprocess image: ', image_paths[i])
        images_batch.tofile(os.path.join(args.output_path, os.path.basename(image_paths[i]))+'.bin')
        np.save(os.path.join(args.size_path, os.path.basename(image_paths[i])),
                size_list)
        images_batch = None
        im = None
        img = None

if __name__ == "__main__":
    main()
