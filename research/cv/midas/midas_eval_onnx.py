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
"""eval midas."""
import glob
import os
import json
import numpy as np
from mindspore import Tensor
from mindspore import context
import mindspore.ops as ops
from src.util import depth_read_kitti, depth_read_sintel, BadPixelMetric
from src.config import config
from src.utils import transforms
import cv2
from PIL import Image
import onnxruntime as ort

def create_session(checkpoint_path, target_device):
    '''
    create onnx session
    '''
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU"')
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name

    return session, input_name

def eval_onnx_Kitti(data_path, session2, input_name2):
    """
    eval Kitti.
    Return the value, loss.
    """
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="lower_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    metric = BadPixelMetric(1.25, 80, 'KITTI')
    loss_sum = 0
    sample = {}
    image_path = glob.glob(os.path.join(data_path, '*', 'image', '*.png'))
    num = 0
    for file_name in image_path:
        num += 1
        print(f"processing: {num} / {len(image_path)}")
        image = np.array(Image.open(file_name)).astype(float)  # (436,1024,3)
        image = image / 255
        print(file_name)
        all_path = file_name.split('/')
        depth_path_name = all_path[-1].split('.')[0]

        depth = depth_read_kitti(os.path.join(data_path, all_path[-3], 'depth', depth_path_name + '.png'))  # (436,1024)
        mask = (depth > 0) & (depth < 80)
        sample['image'] = image
        sample["depth"] = depth
        sample["mask"] = mask
        sample = img_input_1(sample)
        sample = img_input_2(sample)
        sample = img_input_3(sample)
        sample['image'] = np.expand_dims(sample['image'], axis=0)
        sample['depth'] = np.expand_dims(sample['depth'], axis=0)
        sample['mask'] = np.expand_dims(sample['mask'], axis=0)

        print(sample['image'].shape, sample['depth'].shape)

        prediction = session2.run(None, {input_name2: sample['image']})[0]

        mask = sample['mask']
        depth = sample['depth']

        expand_dims = ops.ExpandDims()
        prediction = expand_dims(Tensor(prediction), 0)
        resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
        prediction = resize_bilinear(prediction)
        prediction = np.squeeze(prediction.asnumpy())
        loss = metric(prediction, depth, mask)

        print('loss is ', loss)
        loss_sum += loss

    print(f"Kitti bad pixel: {loss_sum / num:.3f}")
    return loss_sum / num

def eval_onnx_TUM(datapath, session3, input_name3):
    """
    eval TUM.
    Return the value, loss.
    """
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    # get data
    metric = BadPixelMetric(1.25, 10, 'TUM')
    loss_sum = 0
    sample = {}
    file_path = glob.glob(os.path.join(datapath, '*_person', 'associate.txt'))

    num = 0
    for ind in file_path:
        all_path = ind.split('/')

        for line in open(ind):
            num += 1
            print(f"processing: {num}")
            data = line.split('\n')[0].split(' ')
            image_path = os.path.join(datapath, all_path[-2], data[0])  # (480,640,3)
            depth_path = os.path.join(datapath, all_path[-2], data[1])  # (480,640,3)
            image = cv2.imread(image_path) / 255
            depth = cv2.imread(depth_path)[:, :, 0] / 5000
            mask = (depth > 0) & (depth < 10)
            print('mask is ', np.unique(mask))
            sample['image'] = image
            sample["depth"] = depth
            sample["mask"] = mask

            sample = img_input_1(sample)
            sample = img_input_2(sample)
            sample = img_input_3(sample)

            sample['image'] = np.expand_dims(sample['image'], axis=0)
            sample['depth'] = np.expand_dims(sample['depth'], axis=0)
            sample['mask'] = np.expand_dims(sample['mask'], axis=0)

            print(sample['image'].shape, sample['depth'].shape)

            prediction = session3.run(None, {input_name3: sample['image']})[0]

            mask = sample['mask']
            depth = sample['depth']

            expand_dims = ops.ExpandDims()
            prediction = expand_dims(Tensor(prediction), 0)
            print(prediction.shape, mask.shape)
            resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
            prediction = resize_bilinear(prediction)
            prediction = np.squeeze(prediction.asnumpy())

            loss = metric(prediction, depth, mask)

            print('loss is ', loss)
            loss_sum += loss

    print(f"TUM bad pixel: {loss_sum / num:.2f}")

    return loss_sum / num

def eval_onnx_Sintel(datapath, session1, input_name1):
    """
    eval Sintel.
    Return the value, loss.
    """
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    # get data
    metric = BadPixelMetric(1.25, 72, 'sintel')
    loss_sum = 0
    sample = {}
    image_path = glob.glob(os.path.join(datapath, 'final_left', '*', '*.png'))

    num = 0
    for file_name in image_path:
        num += 1
        print(f"processing: {num} / {len(image_path)}")
        image = np.array(Image.open(file_name)).astype(float)  # (436,1024,3)
        image = image / 255
        print(file_name)
        all_path = file_name.split('/')
        depth_path_name = all_path[-1].split('.')[0]

        depth = depth_read_sintel(os.path.join(datapath, 'depth', all_path[-2], depth_path_name + '.dpt'))  # (436,1024)

        mask1 = np.array(Image.open(os.path.join(datapath, 'occlusions', all_path[-2], all_path[-1]))).astype(int)
        mask1 = mask1 / 255

        mask = (mask1 == 1) & (depth > 0) & (depth < 72)
        sample['image'] = image
        sample["depth"] = depth
        sample["mask"] = mask
        sample = img_input_1(sample)
        sample = img_input_2(sample)
        sample = img_input_3(sample)

        sample['image'] = np.expand_dims(sample['image'], axis=0)
        sample['depth'] = np.expand_dims(sample['depth'], axis=0)
        sample['mask'] = np.expand_dims(sample['mask'], axis=0)

        print(sample['image'].shape, sample['depth'].shape)

        prediction = session1.run(None, {input_name1: sample['image']})[0]

        mask = sample['mask']
        depth = sample['depth']

        expand_dims = ops.ExpandDims()
        prediction = expand_dims(Tensor(prediction), 0)
        resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
        prediction = resize_bilinear(prediction)
        prediction = np.squeeze(prediction.asnumpy())
        loss = metric(prediction, depth, mask)

        print('loss is ', loss)
        loss_sum += loss

    print(f"sintel bad pixel: {loss_sum / len(image_path):.3f}")
    return loss_sum / len(image_path)

def run_eval():
    """run."""
    datapath_TUM = config.train_data_dir + config.datapath_TUM
    datapath_Sintel = config.train_data_dir + config.datapath_Sintel
    datapath_Kitti = config.train_data_dir + config.datapath_Kitti

    session1, input_name1 = create_session(config.Sintel_onnx, config.device_target)
    session2, input_name2 = create_session(config.Kitti_onnx, config.device_target)
    session3, input_name3 = create_session(config.TUM_onnx, config.device_target)

    results = {}
    if config.data_name == 'Sintel' or config.data_name == "all":
        result_sintel = eval_onnx_Sintel(datapath_Sintel, session1, input_name1)
        results['Sintel'] = result_sintel
    if config.data_name == 'Kitti' or config.data_name == "all":
        result_kitti = eval_onnx_Kitti(datapath_Kitti, session2, input_name2)
        results['Kitti'] = result_kitti
    if config.data_name == 'TUM' or config.data_name == "all":
        result_tum = eval_onnx_TUM(datapath_TUM, session3, input_name3)
        results['TUM'] = result_tum

    print(results)
    json.dump(results, open(config.onnx_file, 'w'))

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)
    run_eval()
