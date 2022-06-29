'''
The scripts to execute sdk infer
'''
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

import argparse
import os
import numpy as np
import PIL.Image as Image


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="ENET process")
    parser.add_argument("--image_path", type=str, default=None, help="root path of image")
    parser.add_argument('--image_width', default=1024, type=int, help='image width')
    parser.add_argument('--image_height', default=512, type=int, help='image height')
    parser.add_argument('--output_path', default='./bin', type=str, help='bin file path')
    args_opt = parser.parse_args()
    return args_opt


def _get_city_pairs(folder, split='train'):
    """_get_city_pairs"""
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + split)  # os.sep:/
        mask_folder = os.path.join(folder, 'gtFine' + os.sep + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    assert split == 'trainval'
    print('trainval set')
    train_img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + 'train')
    train_mask_folder = os.path.join(folder, 'gtFine' + os.sep + 'train')
    val_img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + 'val')
    val_mask_folder = os.path.join(folder, 'gtFine' + os.sep + 'val')
    train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
    val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
    img_paths = train_img_paths + val_img_paths
    mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


def _val_sync_transform(outsize, img):
    """_val_sync_transform"""
    short_size = min(outsize)
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)

    img = img.resize((ow, oh), Image.BILINEAR)
    w, h = img.size
    x1 = int(round((w - outsize[1]) / 2.))
    y1 = int(round((h - outsize[0]) / 2.))
    img = img.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
    img = np.array(img)
    return img


def main():
    args = parse_args()

    images, mask_paths = _get_city_pairs(args.image_path, 'val')
    assert len(images) == len(mask_paths)
    if not images:
        raise RuntimeError("Found 0 images in subfolders of:" + args.image_path + "\n")

    for index in range(len(images)):
        image_name = images[index].split(os.sep)[-1].split(".")[0]  # get the name of image file
        print("Processing ---> ", image_name)
        img = Image.open(images[index]).convert('RGB')

        img = _val_sync_transform((args.image_height, args.image_width), img)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))  # HWC->CHW(H:height W:width C:channel)
        for channel, _ in enumerate(img):
            img[channel] /= 255
        img = np.expand_dims(img, 0)  # NCHW

        # save bin file
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        data = img
        dataname = image_name + ".bin"
        data.tofile(args.output_path + '/' + dataname)


if __name__ == '__main__':
    main()
