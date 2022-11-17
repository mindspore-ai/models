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
"""
export mnist dataset to bin.
"""
import os
import glob
import argparse
import PIL
import numpy as np
import cv2
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms as C


def ResziePadding(img, fixed_side=256):
    h, w = img.shape[0], img.shape[1]
    scale = max(w, h) / float(fixed_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))

    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (
            fixed_side - new_w) // 2 + 1, (fixed_side - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 != 0:
        top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
            fixed_side - new_w) // 2, (fixed_side - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (
            fixed_side - new_w) // 2
    else:
        top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
            fixed_side - new_w) // 2 + 1, (fixed_side - new_w) // 2

    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

    return pad_img

class DnCNN_eval_Dataset():
    def __init__(self, dataset_path, task_type, noise_level):
        self.im_list = []
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*.png")))
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*.bmp")))
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*.jpg")))
        self.task_type = task_type
        self.noise_level = noise_level

    def __getitem__(self, i):
        img = cv2.imread(self.im_list[i], 0)

        if self.task_type == "denoise":
            noisy = self.add_noise(img, self.noise_level)
        elif self.task_type == "super-resolution":
            h, w = img.shape
            noisy = cv2.resize(img, (int(w/self.noise_level), int(h/self.noise_level)))
            noisy = cv2.resize(noisy, (w, h))
        elif self.task_type == "jpeg-deblock":
            noisy = self.jpeg_compression(img, self.noise_level)

        #add channel dimension
        noisy = noisy[np.newaxis, :, :]
        noisy = noisy / 255.0

        noisy = ResziePadding(noisy[0])
        img = ResziePadding(img)
        return noisy, img

    def __len__(self):
        return len(self.im_list)

    def add_noise(self, im, sigma):
        gauss = np.random.normal(0, sigma, im.shape)
        noisy = im + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype('float32')
        return noisy

    def jpeg_compression(self, img, severity):
        im_pil = PIL.Image.fromarray(img)
        output = io.BytesIO()
        im_pil.save(output, 'JPEG', quality=severity)
        im_pil = PIL.Image.open(output)
        img_np = np.asarray(im_pil)
        return img_np


def create_eval_dataset(data_path, task_type, noise_level, batch_size=1):
    # define dataset
    dataset = DnCNN_eval_Dataset(data_path, task_type, noise_level)
    dataloader = ds.GeneratorDataset(dataset, ["noisy", "clear"])
    # apply map operations on images
    dataloader = dataloader.map(input_columns="noisy", operations=C.TypeCast(mindspore.float32))
    dataloader = dataloader.map(input_columns="clear", operations=C.TypeCast(mindspore.uint8))
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    return dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST to bin')
    parser.add_argument('--device_target', type=str, default="Ascend",
                        choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--dataset_dir', type=str, default='', help='dataset path')
    parser.add_argument('--save_dir', type=str, default='', help='path to save bin file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for bin')
    parser.add_argument('--model_type', type=str, default='DnCNN-S', \
                        choices=['DnCNN-S', 'DnCNN-B', 'DnCNN-3'], help='type of DnCNN')
    parser.add_argument('--noise_type', type=str, default="denoise", \
                        choices=["denoise", "super-resolution", "jpeg-deblock"], help='trained ckpt')
    parser.add_argument('--noise_level', type=int, default=25, help='trained ckpt')
    args_, _ = parser.parse_known_args()
    return args_

if __name__ == "__main__":
    args = parse_args()
    os.environ["RANK_SIZE"] = '1'
    os.environ["RANK_ID"] = '0'
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    mnist_path = os.path.join(args.dataset_dir, 'test')
    batchsize = args.batch_size
    save_dir = os.path.join(args.save_dir, 'dncnn_infer_data')
    folder_noisy = os.path.join(save_dir, 'dncnn_bs_' + str(batchsize) + '_noisy_bin')
    folder_clear = os.path.join(save_dir, 'dncnn_bs_' + str(batchsize) + '_clear_bin')
    if not os.path.exists(folder_clear):
        os.makedirs(folder_clear)
    if not os.path.exists(folder_noisy):
        os.makedirs(folder_noisy)
    ds = create_eval_dataset(args.dataset_dir, args.noise_type, args.noise_level, batch_size=args.batch_size)
    iter_num = 0
    label_file = os.path.join(save_dir, './dncnn_bs_' + str(batchsize) + '_label.txt')

    with open(label_file, 'w') as f:
        for data in ds.create_dict_iterator():
            noisy_img = data['noisy']
            clear_img = data['clear']
            noisy_file_name = "dncnn_noisy_" + str(iter_num) + ".bin"
            noisy_file_path = folder_noisy + "/" + noisy_file_name
            noisy_img.asnumpy().tofile(noisy_file_path)
            clear_file_name = "dncnn_clear_" + str(iter_num) + ".bin"
            clear_file_path = folder_clear + "/" + clear_file_name
            clear_img.asnumpy().tofile(clear_file_path)
            f.write(noisy_file_name + ',' + clear_file_name + '\n')
            iter_num += 1
    print("=====iter_num:{}=====".format(iter_num))
