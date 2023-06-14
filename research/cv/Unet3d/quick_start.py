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

import os
import numpy as np
import mindspore

from mindspore import Tensor, Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.unet3d_model import UNet3d
from src.utils import create_sliding_window, CalculateDice
from src.model_utils.config import config
from src.convert_nifti import convert_nifti

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)


def quick_start(data_path, output_path, ckpt_path):
    """
    data_path: Path of origin data. (mhd file)
    output_path: Path of output data. (nii.gz file)
    ckpt_path: Path of checkpoint file.
    """
    origin_image_path = os.path.join(data_path, "image")
    origin_seg_path = os.path.join(data_path, "seg")
    output_image_path = os.path.join(output_path, "image")
    output_seg_path = os.path.join(output_path, "seg")
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)
    if not os.path.exists(output_seg_path):
        os.mkdir(output_seg_path)
    convert_nifti(origin_image_path, output_image_path, config.roi_size, "*.mhd")
    convert_nifti(origin_seg_path, output_seg_path, config.roi_size, "*.mhd")

    dataset = create_dataset(data_path=output_image_path, seg_path=output_seg_path, is_training=False)

    network = UNet3d()
    network.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    for data in dataset.take(1):  # test one
        image = data[0].asnumpy()
        seg = data[1].asnumpy()
        print("current image shape is {}".format(image.shape), flush=True)
        sliding_window_list, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        image_size = (config.batch_size, config.num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(config.roi_size, np.float32)
        for window, slice_ in zip(sliding_window_list, slice_list):
            window_image = Tensor(window, mindspore.float32)
            pred_probs = model.predict(window_image)
            output_image[slice_] += pred_probs.asnumpy()
            count_map[slice_] += importance_map
        output_image = output_image / count_map
        dice, _ = CalculateDice(output_image, seg)
        print("The dice is", dice)


if __name__ == "__main__":
    quick_start(data_path=config.data_path, output_path=config.output_path, ckpt_path=config.checkpoint_file_path)
