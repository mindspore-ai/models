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

"""visualize for YOLOV4"""

import random
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

random.seed(11)
pred_blue = (0, 0, 255)  # 'with_mask': 1
true_blue = (0, 191, 255)
pred_red = (255, 0, 0)  # 'without_mask': 2,
true_red = (255, 140, 0)
pred_green = (0, 255, 0)  # 'mask_weared_incorrect': 3
true_green = (0, 128, 0)


def visualize_model():
    # load best ckpt to generate instances_val.json and predictions.json

    dataset_dir = r'../../dataset/face_mask_detection/val/images/'
    ann_file = '../../dataset/face_mask_detection/annotations/val.json'
    coco = COCO(ann_file)
    catids = coco.getCatIds()
    imgids = coco.getImgIds()
    img_list = random.sample(imgids, 8)
    coco_res = coco.loadRes('outputs/2022-10-12_time_23_07_44/predict_2022_10_12_23_09_48.json')
    catids_res = coco_res.getCatIds()
    for i in img_list:
        img = coco.loadImgs(i)[0]
        image = cv2.imread(dataset_dir + img['file_name'])
        image_res = image
        annids = coco.getAnnIds(imgIds=img['id'], catIds=catids, iscrowd=None)
        annos = coco.loadAnns(annids)
        annids_res = coco_res.getAnnIds(imgIds=img['id'], catIds=catids_res, iscrowd=None)
        annos_res = coco_res.loadAnns(annids_res)
        plt.figure(figsize=(6, 6))
        for anno in annos:
            bbox = anno['bbox']
            x, y, w, h = bbox
            if anno['category_id'] == 1:  # with_mask
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), true_blue, 2)
            elif anno['category_id'] == 2:  # without_mask
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), true_red, 2)
            else:  # mask_weared_incorrect
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), true_green, 2)
            plt.subplot(1, 2, 1)
            plt.plot([-2, 3], [1, 5])
            plt.title('true')
            plt.imshow(anno_image)
        for anno_res in annos_res:
            bbox_res = anno_res['bbox']
            x, y, w, h = bbox_res
            if anno_res['category_id'] == 1:
                res_image = cv2.rectangle(image_res, (int(x), int(y)), (int(x + w), int(y + h)), pred_blue, 2)
            elif anno_res['category_id'] == 2:
                res_image = cv2.rectangle(image_res, (int(x), int(y)), (int(x + w), int(y + h)), pred_red, 2)
            else:
                res_image = cv2.rectangle(image_res, (int(x), int(y)), (int(x + w), int(y + h)), pred_green, 2)
            plt.subplot(1, 2, 2)
            plt.title('pred')
            plt.imshow(res_image)
    plt.show()


if __name__ == '__main__':
    visualize_model()
