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

"""Evaluation for TextFuseNet"""
import os
import time
import numpy as np
import cv2
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.textfusenet.text_fuse_net_r101 import Text_Fuse_Net_Resnet101
from src.dataset import data_to_mindrecord_byte_image, create_textfusenet_dataset

from pycocotools.coco import COCO
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed


set_seed(1)


def compute_area(point):
    """compute the poly area"""
    s = 0.0
    point_num = len(point)
    if point_num < 3:
        return 0.0
    for i in range(len(point)):
        s += point[i][1] * (point[i-1][0]-point[(i+1)%point_num][0])
    return abs(s/2.0)


def save_result(masks, boxes, labels, img_metas_, txt):
    """save the detection result"""
    f = open(txt, 'w')
    for k in range(len(masks)):
        box = boxes[k].tolist()
        label = labels[k].tolist()
        if label == 0 and box[-1] > 0.9:
            scale = [img_metas_[0][3], img_metas_[0][2], img_metas_[0][3], img_metas_[0][2]]
            [x1, y1, x2, y2] = [int(box[l] / scale[l]) for l in range(len(box) - 1)]
            w, h = x2 - x1, y2 - y1
            image_height, image_width = int(img_metas_[0][0]), int(img_metas_[0][1])
            if x2 > image_width or y2 > image_height or w <= 0 or h <= 0:
                continue

            mask = masks[k].tolist()
            mask = np.array(mask)
            mask = mask > 0.5
            mask = mask.astype(np.uint8)
            mask = mask * 255
            mask = cv2.resize(mask, (w, h))
            canvas = np.zeros((image_height, image_width))
            canvas[y1:y2, x1:x2] = mask
            temp = cv2.findContours(canvas.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            temp = temp[-2]
            temp = [x.flatten() for x in temp]
            temp = [x for x in temp if len(x) > 6]
            poly = temp
            poly = poly[0]
            point = []
            for i in range(0, len(poly) - 1, 2):
                point.append([poly[i], poly[i + 1]])
            area = compute_area(point)
            if area < 120:
                continue
            for p in range(0, len(poly) - 1):
                f.write(str(poly[p]) + ',')
            f.write(str(poly[len(poly) - 1]) + '\n')
    f.close()


def textfusenet_eval(dataset_path, ckpt_path, ann_file):
    """TextFusenNet evaluation."""
    ds = create_textfusenet_dataset(dataset_path, batch_size=config.test_batch_size, is_training=False)

    net = Text_Fuse_Net_Resnet101(config)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    eval_iter = 0
    total = ds.get_dataset_size()
    dataset_coco = COCO(ann_file)

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128
    image_ids = dataset_coco.getImgIds()
    image_number = 0
    imgs = dataset_coco.imgs
    if not os.path.exists('temp'):
        os.mkdir('temp')
    dataset = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    for data in dataset:
        eval_iter = eval_iter + 1
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']
        gt_mask = data["mask"]

        start = time.time()
        # run net
        output = net(Tensor(img_data), Tensor(img_metas), Tensor(gt_bboxes), Tensor(gt_labels), Tensor(gt_num),
                     Tensor(gt_mask))
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]
        all_mask_fb = output[3]
        img_name = imgs[image_ids[image_number]]['file_name']
        image_number += 1
        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])
            all_mask_fb_squee = np.squeeze(all_mask_fb.asnumpy()[j, :, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]
            _all_mask_fb_tmp_mask = all_mask_fb_squee[all_mask_squee, :, :, :]
            all_mask_fb_tmp_mask = np.zeros((all_bboxes_tmp_mask.shape[0], 28, 28)).astype(np.float16)
            for i in range(all_bboxes_tmp_mask.shape[0]):
                all_mask_fb_tmp_mask[i] = _all_mask_fb_tmp_mask[i, all_labels_tmp_mask[i]+1, :, :]
            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]
                all_mask_fb_tmp_mask = all_mask_fb_tmp_mask[inds]
            save_result(all_mask_fb_tmp_mask, all_bboxes_tmp_mask, all_labels_tmp_mask,
                        img_metas, 'temp/'+img_name.replace('.jpg', '.txt'))


def modelarts_process():
    """ modelarts process """
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print("Extract Done")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))
        print("#" * 200, os.listdir(save_dir_1))
        print("#" * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

        config.coco_root = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.checkpoint_path = os.path.join(config.output_path, config.ckpt_path)
    config.ann_file = os.path.join(config.coco_root, config.ann_file)


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id())


@moxing_wrapper(pre_process=modelarts_process)
def eval_():
    """evaluation"""
    config.mindrecord_dir = os.path.join(config.coco_root, config.mindrecord_dir)
    print('\neval.py config:\n', config)
    prefix = "TextFuseNet_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    print("Start Eval!")
    textfusenet_eval(mindrecord_file, config.checkpoint_path, config.ann_file)
    print("ckpt_path=", config.checkpoint_path)


if __name__ == '__main__':
    eval_()
    os.system('cd temp && rm -rf temp.zip && zip temp.zip *.txt && '
              'cd .. && python eval_code/curved_tiou/script.py -g=total-text-gt.zip -s=temp/temp.zip')
