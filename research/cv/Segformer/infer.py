# Copyright 2023 Huawei Technologies Co., Ltd
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
import time
import shutil

import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint
from PIL import Image

from src.model_utils.config import get_infer_config
from src.segformer import SegFormer
from src.dataset import CitySpacesDataset
from src.base_dataset import BaseDataset


def infer(config):
    ori_dataset = BaseDataset(ignore_label=config.dataset_ignore_label,
                              base_size=config.base_size,
                              crop_size=config.crop_size,
                              mean=config.img_norm_mean,
                              std=config.img_norm_std)
    infer_model = SegFormer(config.backbone, config.class_num, sync_bn=config.run_distribute).to_float(
        ms.float16)
    param_dict = load_checkpoint(ckpt_file_name=config.infer_ckpt_path)
    ms.load_param_into_net(infer_model, param_dict)
    print(f"load {config.infer_ckpt_path} success.")
    infer_model.set_train(False)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_file_list = []
    image_path = config.data_path
    if os.path.isdir(image_path):
        for dirpath, _, filenames in os.walk(image_path):
            if filenames:
                for filename in filenames:
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        image_file_list.append((os.path.join(current_directory, dirpath, filename), filename))
    else:
        file_name = os.path.basename(image_path)
        if os.path.isabs(image_path):
            image_file_list.append((image_path, file_name))
        else:
            image_file_list.append((os.path.join(current_directory, image_path), file_name))

    output_path = config.infer_output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(current_directory, output_path)
    os.makedirs(output_path, exist_ok=True)
    print(f"get image size:{len(image_file_list)}, infer result will save to {output_path}")

    infer_begin_time = int(time.time())
    for idx, item in enumerate(image_file_list):
        step_begin_time = int(time.time() * 1000)
        image_name = item[1]
        image_name = image_name[:-4]
        image = ori_dataset.get_one_image(item[0])
        input_img = Tensor(image)
        c, h, w = input_img.shape

        input_img = input_img.reshape([1, c, h, w])
        pred = infer_model(input_img)
        pred = pred[-1]
        output = pred.asnumpy()
        seg_pred = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        if config.infer_copy_original_img:
            path = os.path.join(output_path, item[1])
            if not os.path.exists(path):
                shutil.copyfile(item[0], path)

        if config.infer_save_gray_img:
            img_gray = Image.fromarray(np.uint8(seg_pred), mode="P")
            img_gray.save(os.path.join(output_path, image_name + "_gray.png"))

        img_color = None
        if config.infer_save_color_img:
            palette = CitySpacesDataset.PALETTE
            seg_pred_r = seg_pred.copy()
            seg_pred_g = seg_pred.copy()
            seg_pred_b = seg_pred.copy()
            for i in range(config.class_num):
                seg_pred_r[seg_pred == i] = palette[i][0]
                seg_pred_g[seg_pred == i] = palette[i][1]
                seg_pred_b[seg_pred == i] = palette[i][2]

            seg_pred_r = np.expand_dims(seg_pred_r, axis=2)
            seg_pred_g = np.expand_dims(seg_pred_g, axis=2)
            seg_pred_b = np.expand_dims(seg_pred_b, axis=2)

            img_color = np.concatenate((seg_pred_r, seg_pred_g, seg_pred_b), axis=-1)
            img_color = Image.fromarray(np.uint8(img_color), mode="RGB")
            img_color.save(os.path.join(output_path, image_name + "_color.png"))

        if config.infer_save_overlap_img and img_color is not None:
            ori_img = Image.open(item[0])
            img_color = img_color.resize(ori_img.size)
            overlap_img = Image.blend(ori_img, img_color, alpha=0.5)
            overlap_img.save(os.path.join(output_path, image_name + "_overlap.png"))

        step_end_time = int(time.time() * 1000)
        if (idx + 1) % config.infer_log_interval == 0:
            print(f"infer {idx + 1}/{len(image_file_list)} done, step cost:{step_end_time - step_begin_time}ms")

    infer_end_time = int(time.time())
    print(f"all infer process done, cost:{infer_end_time - infer_begin_time}s")


if __name__ == '__main__':
    infer_config = get_infer_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=infer_config.device_target)
    print(f"infer config:{infer_config}")
    infer(infer_config)
