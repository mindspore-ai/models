# !/usr/bin/env python

# Copyright (c) 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
Sdk internece
"""
import argparse
import os
import time
import sys
import ast
import numpy as np

from PIL import Image
from pycocotools.coco import COCO

from api.infer import SdkApi
from api.eval import DetectionEngine
from config import config as cfg


def parser_args():
    """
    configuration parameter, input from outside
    """
    parser = argparse.ArgumentParser(description="yolov3_tiny inference")
    parser.add_argument("--pipeline_path", type=str, required=False, default="./config/yv3.pipeline",
                        help="pipeline file path. The default is './config/yv3.pipeline'. ")

    parser.add_argument('--log_path', type=str, default='../../data/eval_result', help='inference result save location')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--ann_file', type=str, default='../../data/annotations/instances_val2017.json',
                        help='path to annotation')
    parser.add_argument('--eval_ignore_threshold', type=float, default=0.001,
                        help='threshold to throw low quality boxes')
    parser.add_argument('--dataset_path', type=str, default='../../data/val2017', help='path of image dataset')
    parser.add_argument('--result_files', type=str, default='../data/preds/sdk/val2017',
                        help='path to 310 infer result path')
    parser.add_argument('--multi_label', type=ast.literal_eval, default=True, help='whether to use multi label')
    parser.add_argument('--multi_label_thresh', type=float, default=0.1, help='threshold to throw low quality boxes')

    arg = parser.parse_args()
    return arg

def get_interp_method(interp, sizes=()):
    """
    Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic or Bilinear.

    Args:
        interp (int): Interpolation method for all resizing operations.

            - 0: Nearest Neighbors Interpolation.
            - 1: Bilinear interpolation.
            - 2: Bicubic interpolation over 4x4 pixel neighborhood.
            - 3: Nearest Neighbors. Originally it should be Area-based, as we cannot find Area-based,
              so we use NN instead. Area-based (resampling using pixel area relation).
              It may be a preferred method for image decimation, as it gives moire-free results.
              But when the image is zoomed, it is similar to the Nearest Neighbors method. (used by default).
            - 4: Lanczos interpolation over 8x8 pixel neighborhood.
            - 9: Cubic for enlarge, area for shrink, bilinear for others.
            - 10: Random select from interpolation method mentioned above.

        sizes (tuple): Format should like (old_height, old_width, new_height, new_width),
            if None provided, auto(9) will return Area(2) anyway. Default: ()

    Returns:
        int, interp method from 0 to 4.
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp

def pil_image_reshape(interp):
    """Reshape pil image."""#from PIL import Image
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]

def statistic_normalize_img(img, statistic_norm):
    """Statistic normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    if statistic_norm:
        img = (img - mean) / std
    return img

def process_img(img_file):
    """
    Preprocessing the images
    """
    orimg = Image.open(img_file).convert("RGB")
    if not isinstance(orimg, Image.Image):
        orimg = Image.fromarray(orimg)
    ori_w, ori_h = orimg.size
    h, w = [640, 640]
    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, h, w))
    img = orimg.resize((w, h), pil_image_reshape(interp))  # ---------------------------resize
    img = statistic_normalize_img(img, statistic_norm=True)  # -----------------------normlize

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.concatenate([img, img, img], axis=-1)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    return img

def image_inference(pipeline_path, stream_name, img_dir, detection, result_files):
    """
    image inference: get inference for images
    """
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        return

    img_data_plugin_id = 0
    print(f"\nBegin to inference for {img_dir}.\n")

    file_list = os.listdir(img_dir)
    coco = COCO(args.ann_file)
    start_time = time.time()
    max_len = 5001

    for _, file_name in enumerate(file_list):
        if _ >= max_len:
            break
        if (_ + 1) % 100 == 0:
            print(f"{_ + 1}/{max_len}\n")
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue

        img_ids_name = file_name.split('.')[0]
        img_id_ = int(np.squeeze(img_ids_name))
        imgIds = coco.getImgIds(imgIds=[img_id_])
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        image_shape = ((img['width'], img['height']),)
        img_id_ = (np.squeeze(img_ids_name),)

        imgs = process_img(os.path.join(img_dir, file_name))

        sdk_api.send_tensor_input(stream_name,
                                  img_data_plugin_id, "appsrc0",
                                  imgs.tobytes(), imgs.shape, cfg.TENSOR_DTYPE_FLOAT32)

        result_0, result_1 = sdk_api.get_result(stream_name)

        output_small = np.frombuffer(result_0,
                                     dtype='float32').reshape((1, 20, 20, 3, 85))
        output_big = np.frombuffer(result_1,
                                   dtype='float32').reshape((1, 40, 40, 3, 85))
        detection.detect([output_small, output_big], 1, image_shape, img_id_)

        save_bin_path_0 = os.path.join(result_files, img_ids_name) + "_0.bin"
        save_bin_path_1 = os.path.join(result_files, img_ids_name) + "_1.bin"
        with open(save_bin_path_0, "wb") as fp:
            fp.write(result_0)
        with open(save_bin_path_1, "wb") as fp_1:
            fp_1.write(result_1)

    cost_time = time.time() - start_time
    print('testing cost time {:.2f}s'.format(cost_time), 'img/sec=', 5000/cost_time)

    print('do_nms_for_results...')
    detection.do_nms_for_results()
    detection.write_result()
    eval_result = detection.get_eval_result()
    print('\n=============coco eval result=========\n' + eval_result)

if __name__ == "__main__":
    lib_path = os.path.abspath(os.path.join('../..'))
    sys.path.append(lib_path)
    args = parser_args()
    args.outputs_dir = os.path.join(args.log_path,
                                    "06-27-1")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    if not os.path.exists(args.result_files):
        os.makedirs(args.result_files)
    detections = DetectionEngine(args)
    stream_name0 = cfg.STREAM_NAME.encode("utf-8")
    print("stream_name0:")
    print(stream_name0)
    image_inference(args.pipeline_path, stream_name0, args.dataset_path, detections, args.result_files)
