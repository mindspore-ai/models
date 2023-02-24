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
"""Infer"""
import os
import time
import json

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context, set_seed, Parameter, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.utils import ValueInfo, update_config
from src.config import FasterRcnnConfig
from src.dataset import build_weak_augmentation
from src.FasterRcnnInfer.faster_rcnn_r50 import FasterRcnn_Infer_Online


def get_unlabel_images(cfg):
    train_ann_file = cfg.train_ann_file
    train_img_dir = cfg.train_img_dir
    with open(train_ann_file, 'r') as file:
        train_ann = json.load(file)
    images = train_ann["images"]
    annotations = train_ann["annotations"]
    unlabel_ids = [annotation["image_id"] for annotation in annotations if annotation["category_id"] == -1]
    unlabel_images = [image["file_name"] for image in images if image["id"] in unlabel_ids]
    unlabel_image_files = [os.path.join(train_img_dir, image_file) for image_file in unlabel_images]
    return unlabel_image_files


def rescale_with_tuple(img, scale):
    width, height = img.size
    scale_factor = min(max(scale) / max(height, width), min(scale) / min(height, width))
    new_size = int(width * float(scale_factor) + 0.5), int(height * float(scale_factor) + 0.5)

    image_resize = img.resize(new_size, Image.BILINEAR)
    return image_resize, scale_factor


def process_pic(image, cfg):
    ori_w, ori_h = image.size
    if cfg.keep_ratio:
        # rescale
        image, scale_factor = rescale_with_tuple(image, (cfg.img_width, cfg.img_height))
        if image.size[1] > cfg.img_height:
            image, scale_factor_2 = rescale_with_tuple(image,
                                                       (cfg.img_height, cfg.img_height))
            scale_factor = scale_factor * scale_factor_2

        pad_h = cfg.img_height - image.size[1]
        pad_w = cfg.img_width - image.size[0]
        if not ((pad_h >= 0) and (pad_w >= 0)):
            raise ValueError("[ERROR] rescale w, h = {}, {} failed.".format(image.size[0], image.size[1]))

        pad_img = Image.new("RGB", (cfg.img_width, cfg.img_height), (0, 0, 0))
        pad_img.paste(image, (0, 0))
        pad_img = pad_img.convert("RGB")
        image_data = np.array(pad_img)

        scale_h = scale_factor
        scale_w = scale_factor
    else:
        image_resize = image.resize(cfg.img_width, cfg.img_height, Image.BILINEAR).convert("RGB")
        scale_h = cfg.img_height / ori_h
        scale_w = cfg.img_width / ori_w
        image_data = np.array(image_resize)

    img_metas = (ori_h, ori_w, scale_h, scale_w)
    img_metas = Tensor(np.asarray(img_metas, dtype=np.float32))

    weak_augmentation = build_weak_augmentation()
    img_weak = Tensor(weak_augmentation(image_data)[0][np.newaxis, :])
    return img_weak, img_metas


def infer_combine(images_path=None):
    print("========================================")
    print("[INFO] Config contents:")
    for cfg_k, cfg_v in FasterRcnnConfig.__dict__.items():
        if not cfg_k.startswith("_"):
            print("{}: {}".format(cfg_k, cfg_v))
    print("========================================")

    cfg = FasterRcnnConfig()
    cfg.test_batch_size = cfg.batch_size = 1

    set_seed(cfg.global_seed)
    device_id = os.getenv('DEVICE_ID', str(cfg.eval_device_id))
    device_id = int(device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=device_id)
    context.set_context(max_call_depth=6000)
    print("[INFO] current device_id: {}".format(device_id))

    ####################################################################################
    # net init
    net = FasterRcnn_Infer_Online(config=cfg)

    ckpt_file_path = str(os.getenv('CKPT_PATH', cfg.checkpoint_path))
    param_dict = load_checkpoint(ckpt_file_name=ckpt_file_path)
    if cfg.device_target == "GPU":
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    param_not_load, _ = load_param_into_net(net, param_dict, strict_load=True)

    if cfg.device_target == "Ascend":
        net.to_float(mstype.float16)
    net = net.set_train(mode=False)
    print("[DEBUG] load {} to net, params not load: {}".format(ckpt_file_path, param_not_load))
    print("[INFO] net create ok.")

    ####################################################################################
    print("========================================")
    if images_path is None:
        images_path = get_unlabel_images(cfg)
    steps_per_epoch = len(images_path)
    print("[INFO] total_steps: {}".format(steps_per_epoch))
    print("[INFO] Processing, please wait a moment.")

    time_infos = [ValueInfo("iter_time"), ValueInfo("data_time")]
    results_dict = {}
    step_iter = 0
    start_time = time.perf_counter()
    for index in range(steps_per_epoch):
        data_time = time.perf_counter() - start_time
        img_path = images_path[index]
        image = Image.open(img_path)
        image_data, image_metas = process_pic(image, cfg)
        inference_output = net(image_data, image_metas)
        result = post_process(cfg, inference_output)
        results_dict[os.path.basename(img_path)] = result

        if cfg.draw_pics:
            draw_image(result, img_path, cfg.eval_output_dir)

        iter_time = time.perf_counter() - start_time

        # update eval infos
        time_infos[0].update(iter_time)
        time_infos[1].update(data_time)

        if step_iter % cfg.print_interval_iter == 0:
            print("[INFO] device_id: {}, step: {}/{}, iter_time: {:.3f}, data_time: {:.3f}"
                  .format(device_id, step_iter, steps_per_epoch, time_infos[0].avg(), time_infos[1].avg()))

        step_iter += 1
        start_time = time.perf_counter()

    output_results_file = os.path.join(cfg.eval_output_dir, "infer_results.json")
    with open(output_results_file, "w") as file_handle:
        json.dump(results_dict, file_handle)
    print("[INFO] save infer results to {}".format(output_results_file))

    print("[INFO] end.")


def post_process(cfg, inference_output):
    outputs = inference_post_process(cfg, inference_output)
    results = process_result(cfg, outputs)
    return results


def inference_post_process(cfg, inference_output):
    outputs = []
    max_num = 128
    all_bbox = inference_output[0]
    all_label = inference_output[1]
    all_mask = inference_output[2]
    all_score = inference_output[3]
    for j in range(cfg.test_batch_size):
        all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
        all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
        all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])
        all_score_squee = np.squeeze(all_score.asnumpy()[j, :, :])

        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]
        all_scores_tmp_mask = all_score_squee[all_mask_squee, :]

        if all_bboxes_tmp_mask.shape[0] > max_num:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
            inds = inds[:max_num]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]
            all_scores_tmp_mask = all_scores_tmp_mask[inds]

        outputs_tmp = [all_bboxes_tmp_mask, all_labels_tmp_mask, all_scores_tmp_mask]
        outputs.append(outputs_tmp)

    return outputs


def process_result(cfg, outputs):
    json_results = []
    for batch_idx in range(cfg.test_batch_size):
        output = outputs[batch_idx]
        for idx in range(output[0].shape[0]):
            bbox = output[0][idx]
            label = output[1][idx]
            score = output[2][idx]
            data = dict()
            data["pred box"] = bbox[:4].tolist()
            data["pred box"][2] = data["pred box"][2] - data["pred box"][0] + 1
            data["pred box"][3] = data["pred box"][3] - data["pred box"][1] + 1
            data["confidence score"] = float(bbox[4])
            data["pred score"] = score.tolist()     # [background, cls1, cls2, ...]
            data["pred class"] = int(label)
            data["width"] = int(4608)       # ori image width
            data["height"] = int(3288)      # ori image height
            json_results.append(data)
    return json_results


def draw_image(result, infer_image_path, infer_output_dir):
    image = Image.open(infer_image_path).convert("RGB")
    save_file_name = "{}{}".format(os.path.basename(infer_image_path).rsplit(".", 1)[0], "_infer.jpg")
    save_file_path = os.path.join(infer_output_dir, save_file_name)

    bbox_list = []
    category_list = []
    score_list = []
    for data in result:
        # filter
        if (int(data["pred class"]) == 0 and data["confidence score"] < 0.0) \
                or (int(data["pred class"]) == 1 and data["confidence score"] < 0.0) \
                or (int(data["pred class"]) == 2 and data["confidence score"] < 0.0):
            continue
        bbox_list.append(data["pred box"])
        category_list.append(int(data["pred class"]))
        score_list.append(data["confidence score"])

    draw_bbox_cls_score(image, bbox_list, category_list, score_list, save_file_path)


def draw_bbox_cls_score(image, bbox_list, category_list, score_list, save_file: str):
    name_mapping = {0: "screw_1", 1: "screw_2", 2: "screw_3"}
    color_mapping = {0: "springgreen", 1: "red", 2: "mediumblue"}
    category_id_count = [0, 0, 0]

    label_data_draw = ImageDraw.Draw(image)
    # need enter command "fc-list" to choose one ttf file
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf", 10, encoding='utf-8')
    for bbox, category_id, score in zip(bbox_list, category_list, score_list):
        label_data_draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  fill=None, outline=color_mapping[category_id], width=1)
        label_data_draw.text((bbox[0], bbox[1] - 10), "{}: {:.2f}"
                             .format(name_mapping[category_id], score), color_mapping[category_id], font=font)
        category_id_count[category_id] += 1

    label_x = 10
    label_y = 10
    for index in range(len(category_id_count)):
        label_data_draw.text((label_x, label_y),
                             "{}: {}".format(name_mapping[index], category_id_count[index]),
                             color_mapping[index], font=font)
        label_y += 10

    image.save(save_file)


if __name__ == '__main__':
    localtime_start = time.asctime(time.localtime(time.time()))
    print("[INFO] start time: {}".format(localtime_start))
    update_config()
    infer_combine()
    localtime_end = time.asctime(time.localtime(time.time()))
    print("[INFO] end time: {}".format(localtime_end))
