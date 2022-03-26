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
"""operation to generate semantically segmented pictures"""
import argparse
import os
import time

import cv2
import imageio
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import src.blocks as blocks

parser = argparse.ArgumentParser()
parser.add_argument("--content_path", type=str, help='content_path, default: None')
parser.add_argument('--pre_trained', type=str, help='model_path, local pretrained model to load')
parser.add_argument("--output_dir", type=str, default='output_dir', help='output_path, path to store output')

# additional params for online generating
parser.add_argument("--run_online", type=int, default=0, help='whether train online, default: false')
parser.add_argument("--data_url", type=str, help='path to data on obs, default: None')
parser.add_argument("--train_url", type=str, help='output path on obs, default: None')

args = parser.parse_args()

if __name__ == '__main__':

    if args.run_online:
        import moxing as mox

        mox.file.copy_parallel(args.data_url, "/cache/dataset")
        local_dataset_dir = "/cache/dataset/content_dir"
        pre_ckpt_dir = "/cache/dataset/pre_ckpt"
        pre_ckpt_path = pre_ckpt_dir + "/" + os.listdir(pre_ckpt_dir)[0]
        output_dir = "/cache/dataset/pred_dir"
    else:
        local_dataset_dir = args.content_path
        pre_ckpt_path = args.pre_trained
        output_dir = args.output_dir

    context.set_context(mode=context.GRAPH_MODE)
    param_dict = load_checkpoint(pre_ckpt_path)
    net = blocks.U2NET()
    net.set_train(False)
    load_param_into_net(net, param_dict)


    def normPRED(d):
        """rescale the value of tensor to between 0 and 1"""
        ma = d.max()
        mi = d.min()
        dn = (d - mi) / (ma - mi)
        return dn


    def normalize(img):
        """normalize tensor"""
        if len(img.shape) == 3:
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        else:
            img = (img - 0.485) / 0.229
        return img


    def resize_im(img_path, size=320):
        """crop and resize tensors"""
        img = np.array(Image.open(img_path), dtype='float32')
        img = img / 255
        img = normalize(img)
        h, w = img.shape[:2]
        img = cv2.resize(img, dsize=(0, 0), fx=size / w, fy=size / h)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2).repeat(1, axis=2)
        im = img
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 0, 1)
        im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
        return im


    content_list = os.listdir(local_dataset_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_time = time.time()
    for j in range(0, len(content_list)):
        pic_path = os.path.join(local_dataset_dir, content_list[j])
        content_pic = resize_im(pic_path, size=320)
        image = net(Tensor(content_pic))
        content_name = content_list[j].replace(".jpg", "")
        content_name = content_name.replace(".png", "")
        file_path = os.path.join(local_dataset_dir, content_list[j])
        original = np.array(Image.open(file_path), dtype='float32')
        shape = original.shape
        image = normPRED(image[0][0].asnumpy())
        image = cv2.resize(image, dsize=(0, 0), fx=shape[1] / image.shape[1], fy=shape[0] / image.shape[0])
        file_path = os.path.join(output_dir, content_name) + ".png"
        imageio.imsave(file_path, image)
        print("%d / %d , %s \n" % (j, len(content_list), content_name))
    end_time = time.time()
    dtime = end_time - start_time
    print("finish generating in %.8s s" % (dtime))
    if args.run_online:
        mox.file.copy_parallel(output_dir, args.train_url)
