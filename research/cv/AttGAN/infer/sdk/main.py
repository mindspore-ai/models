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
""" Model Main """
import argparse
import os
import numpy as np
import cv2
from api import SdkApi
from PIL import Image
from config import config as cfg

def parser_args():
    """ Args Setting """
    parser = argparse.ArgumentParser(description="attgan inference")
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        default="/home/data/sdx_mindx/AttGAN/dataset",
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/attgan.pipeline",
        help="image file path. The default is 'config/attgan.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/sdk_result",
        help=
        "cache dir of inference result. The default is '../data/sdk_result'."
    )

    parser.add_argument("--ann_file",
                        type=str,
                        required=False,
                        help="eval ann_file.")

    args_ = parser.parse_args()
    return args_

def get_labels(img_dir):
    """ Get Labels Setting """
    # labels preprocess
    index = 0
    selected_attrs = cfg.SELECTED_ATTRS
    lines = [
        line.rstrip() for line in open(
            os.path.join(img_dir, 'list_attr_celeba.txt'), 'r')
    ]

    all_attr_names = lines[1].split()
    attr_path = os.path.join(img_dir, 'list_attr_celeba.txt')
    all_atts = [all_attr_names.index(att) + 1 for att in selected_attrs]
    images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
    labels = np.loadtxt(attr_path, skiprows=2, usecols=all_atts, dtype=np.int)
    final_labels = np.array([labels]) if images.size == 1 else labels[0:]

    att = np.asarray((final_labels[index] + 1) // 2)
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name
    lines = lines[2:]
    items = {}
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append((int(values[idx]) + 1))
        items[filename] = np.array(label).astype(np.float32)
    return att

def check_attribute_conflict(att_batch, att_name, att_names):
    """Check Attributes"""
    def _set(att, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = 0.0

    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            _set(att, 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            _set(att, 'Bald')
            _set(att, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name:
                    _set(att, n)
    return att_batch

def process_labels(attr_a):
    """ Preprocess Attr """
    attr = []
    selected_attrs = cfg.SELECTED_ATTRS
    n_attrs = len(selected_attrs)
    for index in range(n_attrs):
        attr.append(attr_a[index])
        list_attr = [attr]
    for i in range(n_attrs):
        tmp = np.expand_dims(attr, axis=0)
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, selected_attrs[i], selected_attrs)
        list_attr.append(tmp)
    return list_attr

def process_img(img_file):
    """ Preprocess Image """
    img = cv2.imread(img_file)
    model_img = cv2.resize(img, (cfg.MODEL_WIDTH, cfg.MODEL_HEIGHT))
    img_ = model_img[:, :, ::-1].transpose((2, 0, 1))
    img_ = np.expand_dims(img_, axis=0)
    img_ = np.array((img_-127.5)/127.5).astype(np.float32)
    return img_

def decode_image(img):
    """ Decode Image """
    mean = 0.5 * 255
    std = 0.5 * 255
    return (img * std + mean).astype(np.uint8).transpose(
        (1, 2, 0))

def image_inference(pipeline_path, stream_name, img_dir, result_dir,
                    replace_last, model_type):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    img_label_plugin_id = 1
    label_items = get_labels(img_dir)

    file_list = os.listdir(os.path.join(img_dir, 'image'))
    for _, file_name in enumerate(file_list):
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        file_path = os.path.join(img_dir, 'image', file_name)
        save_path = os.path.join(result_dir, f"{os.path.splitext(file_name)[0]}.jpg")
        if not replace_last and os.path.exists(save_path):
            print(f"The infer result image({save_path}) has existed, will be skip.")
            continue

        img_np = process_img(file_path)
        sample = []
        final_attr = process_labels(label_items)
        for i, att_b in enumerate(final_attr):
            att_b_ = np.array(att_b, dtype='float32')
            att_b_ = (att_b_ * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[:, i - 1] = att_b_[:, i - 1] * args.test_int / args.thres_int
            if i > 0:
                att_b_ = att_b_.reshape(1, 13)
            sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0", img_np.tobytes(), img_np.shape,
                                      cfg.TENSOR_DTYPE_FLOAT32)
            sdk_api.send_tensor_input(stream_name, img_label_plugin_id, "appsrc1", att_b_.tobytes(), [1, 13],
                                      cfg.TENSOR_DTYPE_FLOAT32)
            result = sdk_api.get_result(stream_name)
            data = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
            data = data.reshape(3, 128, 128).transpose(0, 2, 1)
            img = decode_image(data)
            sample.append(img)
        last_image = np.array(sample)
        last_sample = np.reshape(last_image, (1792, 128, -1)).transpose(1, 0, 2)
        im = Image.fromarray(np.uint8(last_sample))
        im.save(save_path)

if __name__ == "__main__":
    args = parser_args()
    test_int = args.test_int
    thres_int = args.thres_int
    args.replace_last = True
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, args.stream_name, args.img_path,
                    args.infer_result_dir, args.replace_last, args.model_type)
