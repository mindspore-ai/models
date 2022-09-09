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

"""visualize for retinanet"""

import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from src.model_utils.config import config


def visualize_model():
    # load best ckpt to generate instances_val.json and predictions.json

    dataset_dir = r'./dataset/val/images/'
    coco_root = config.voc_root
    data_type = config.val_data_type
    annotation_file = os.path.join(coco_root, config.instances_set.format(data_type))
    coco = COCO(annotation_file)
    catids = coco.getCatIds()
    imgids = coco.getImgIds()
    coco_res = coco.loadRes('./predictions.json')
    catids_res = coco_res.getCatIds()
    for i in range(10):
        img = coco.loadImgs(imgids[i])[0]
        image = cv2.imread(dataset_dir + img['file_name'])
        image_res = image
        annids = coco.getAnnIds(imgIds=img['id'], catIds=catids, iscrowd=None)
        annos = coco.loadAnns(annids)
        annids_res = coco_res.getAnnIds(imgIds=img['id'], catIds=catids_res, iscrowd=None)
        annos_res = coco_res.loadAnns(annids_res)
        plt.figure(figsize=(7, 7))
        for anno in annos:
            bbox = anno['bbox']
            x, y, w, h = bbox
            if anno['category_id'] == 1:
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (153, 153, 255), 2)
            elif anno['category_id'] == 2:
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (153, 255, 153), 2)
            else:
                anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 153, 153), 2)
            plt.subplot(1, 2, 1)
            plt.plot([-2, 3], [1, 5])
            plt.title('true-label')
            plt.imshow(anno_image)
        for anno_res in annos_res:
            bbox_res = anno_res['bbox']
            x, y, w, h = bbox_res
            if anno_res['category_id'] == 1:
                res_image = cv2.rectangle(image_res, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            elif anno_res['category_id'] == 2:
                res_image = cv2.rectangle(image_res, (int(x), int(y)), (int(x + w), int(y + h)), (0, 153, 0), 2)
            else:
                res_image = cv2.rectangle(image_res, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            plt.subplot(1, 2, 2)
            plt.title('pred-label')
            plt.imshow(res_image)
    plt.show()


if __name__ == '__main__':
    visualize_model()
