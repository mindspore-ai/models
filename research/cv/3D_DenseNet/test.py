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
"""
This module is used for test iseg-2017-testing subject
"""
import argparse
import os
import SimpleITK as sitk
import numpy as np
from mindspore import load_checkpoint
from mindspore.dataset import context
from mindspore.common import Tensor
from src.common import crop_size, num_classes
from src.model import DenseNet
from src.model_utils.config import config

def read_med_image(file_path, dtype):
    """
    Read Nii/hdr file using stk
    """
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk


def convert_label_submit(label_img):
    """
    covert label
    """
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for index in range(label_img.shape[2]):
        label_slice = label_img[:, :, index]
        label_slice[label_slice == 1] = 10
        label_slice[label_slice == 2] = 150
        label_slice[label_slice == 3] = 250
        label_processed[:, :, index] = label_slice
    return label_processed

parser = argparse.ArgumentParser(description='model_3d_denseseg_v1')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
net = DenseNet(num_init_features=config.num_init_features, growth_rate=config.growth_rate, \
                block_config=config.block_config, \
                num_classes=config.num_classes, drop_rate=config.drop_rate)

if __name__ == '__main__':
    #Testing
    #Load the checkpoint (weights)
    checkpoint = config.checkpoint_file_path
    print('Checkpoint: ', checkpoint)
    load_checkpoint(checkpoint, net=net)
    #Load testing data
    test_path = config.test_dir
    index_file = 0
    xstep = 8
    ystep = 8  # 16
    zstep = 8  # 16
    for subject_id in range(11, 24):
        subject_name = 'subject-%d-' % subject_id
        f_T1 = os.path.join(test_path, subject_name + 'T1.hdr')
        f_T2 = os.path.join(test_path, subject_name + 'T2.hdr')
        inputs_T1, img_T1_itk = read_med_image(f_T1, dtype=np.float32)
        inputs_T2, img_T2_itk = read_med_image(f_T2, dtype=np.float32)
        mask = inputs_T1 > 0
        mask = mask.astype(np.bool)
        #normalize to 0 mean and 1 variance
        #Normalization
        inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
        inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()
        inputs_T1_norm = inputs_T1_norm[:, :, :, None]
        inputs_T2_norm = inputs_T2_norm[:, :, :, None]
        inputs = np.concatenate((inputs_T1_norm, inputs_T2_norm), axis=3)
        inputs = inputs[None, :, :, :, :]
        image = inputs.transpose(0, 4, 1, 3, 2)
        _, _, C, H, W = image.shape
        deep_slices = np.arange(0, C - crop_size[0] + xstep, xstep)
        height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
        width_slices = np.arange(0, W - crop_size[2] + zstep, zstep)
        whole_pred = np.zeros((1,) + (num_classes,) + image.shape[2:])
        count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5
        #no update parameter gradients during testing
        for i in range(len(deep_slices)):
            for j in range(len(height_slices)):
                for k in range(len(width_slices)):
                    deep = deep_slices[i]
                    height = height_slices[j]
                    width = width_slices[k]
                    image_crop = image[:, :, deep: deep + crop_size[0], \
                                 height: height + crop_size[1], \
                                 width: width + crop_size[2]]
                    image_crop = Tensor(image_crop)
                    outputs = net(image_crop)
                    #Average
                    whole_pred[slice(None), slice(None), deep: deep + crop_size[0],\
                    height: height + crop_size[1],\
                    width: width + crop_size[2]] += outputs.asnumpy()
                    count_used[deep: deep + crop_size[0],\
                    height: height + crop_size[1],\
                    width: width + crop_size[2]] += 1

        whole_pred = whole_pred / count_used
        whole_pred = whole_pred[0, :, :, :, :]
        whole_pred = np.argmax(whole_pred, axis=0)
        #Write to file
        f_pred = os.path.join(test_path, subject_name + 'label.hdr')
        whole_pred = whole_pred.transpose(0, 2, 1)
        whole_pred = convert_label_submit(whole_pred)
        whole_pred_itk = sitk.GetImageFromArray(whole_pred.astype(np.uint8))
        whole_pred_itk.SetSpacing(img_T1_itk.GetSpacing())
        whole_pred_itk.SetDirection(img_T1_itk.GetDirection())
        sitk.WriteImage(whole_pred_itk, f_pred)
