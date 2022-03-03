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

import json
import os

import imageio as io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from src.dataset.pose import Batch, DataItem, PoseDataset


def get_gt_visibilities(in_file, visibilities):
    """
    get ground truth visibilities
    """
    with open(in_file) as data_file:
        data = json.load(data_file)

    for person_id in range(len(data)):
        keypoints = data[person_id]["keypoints"]
        keypoints = [visibilities[person_id][i // 3] if i % 3 == 2 else int(keypoints[i]) for i in
                     range(len(keypoints))]
        data[person_id]["keypoints"] = keypoints

    with open(in_file, 'w') as data_file:
        json.dump(data, data_file)


class MSCOCO(PoseDataset):

    def load_dataset(self):
        dataset = self.cfg.dataset.path
        dataset_phase = self.cfg.dataset.phase
        dataset_ann = self.cfg.dataset.ann

        # initialize COCO api
        ann_file = '%s/annotations/%s_%s.json' % (dataset, dataset_ann, dataset_phase)
        self.coco = COCO(ann_file)

        img_ids = self.coco.getImgIds()

        data = []

        # loop through each image
        for imgId in img_ids:
            item = DataItem()

            img = self.coco.loadImgs(imgId)[0]
            item.im_path = "%s/images/%s/%s" % (dataset, dataset_phase, img["file_name"])
            item.im_size = [3, img["height"], img["width"]]
            item.coco_id = imgId
            item.scale = self.get_scale()
            if not self.is_valid_size(item.im_size, item.scale):
                continue
            ann_ids = self.coco.getAnnIds(imgIds=img['id'], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            all_person_key_points = []
            masked_persons_rle = []
            visible_persons_rle = []
            all_visibilities = []

            # Consider only images with people
            has_people = len(anns)
            if not has_people and self.cfg.coco_only_images_with_people:
                continue

            for ann in anns:  # loop through each person
                person_key_points = []
                visibilities = []
                if ann["num_keypoints"] != 0:
                    for i in range(self.cfg.num_joints):
                        x_coord = ann["keypoints"][3 * i]
                        y_coord = ann["keypoints"][3 * i + 1]
                        visibility = ann["keypoints"][3 * i + 2]
                        visibilities.append(visibility)
                        if visibility != 0:  # i.e. if labeled
                            person_key_points.append([i, x_coord, y_coord])
                    all_person_key_points.append(np.array(person_key_points))
                    visible_persons_rle.append(mask_utils.decode(self.coco.annToRLE(ann)))
                    all_visibilities.append(visibilities)
                if ann["num_keypoints"] == 0:
                    masked_persons_rle.append(self.coco.annToRLE(ann))

            item.joints = np.array(all_person_key_points)
            item.im_neg_mask = mask_utils.merge(masked_persons_rle)
            if self.cfg.use_gt_segm:
                item.gt_segm = np.moveaxis(np.array(visible_persons_rle), 0, -1)
                item.visibilities = all_visibilities
            data.append(item)

        self.has_gt = self.cfg.dataset != "image_info"
        return data

    def compute_score_map_weights(self, scmap_shape, joint_id, data_item):
        size = scmap_shape[0:2]
        scmask = np.ones(size)
        m = mask_utils.decode(data_item.im_neg_mask)
        if m.size:
            img = Image.fromarray(m)
            img = img.resize((size[1], size[0]))
            scmask = 1.0 - np.array(img)
        scmask = np.stack([scmask] * self.cfg.num_joints, axis=-1)
        return scmask

    def get_pose_segments(self):
        return [[0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [6, 8], [7, 9], [8, 10], [11, 13], [12, 14], [13, 15], [14, 16]]

    def visualize_coco(self, coco_img_results, visibilities):
        """
        visualize coco
        """
        in_file = "tmp.json"
        with open(in_file, 'w') as outfile:
            json.dump(coco_img_results, outfile)
        get_gt_visibilities(in_file, visibilities)

        # initialize coco_pred api
        coco_pred = self.coco.loadRes(in_file)
        os.remove(in_file)

        img_ids = [coco_img_results[0]["image_id"]]

        for imgId in img_ids:
            img = coco_pred.loadImgs(imgId)[0]
            im_path = "%s/images/%s/%s" % (self.cfg.dataset.path, self.cfg.dataset.phase, img["file_name"])
            I = io.imread(im_path)

            fig = plt.figure()
            a = fig.add_subplot(2, 2, 1)
            plt.imshow(I)
            a.set_title('Initial Image')

            a = fig.add_subplot(2, 2, 2)
            plt.imshow(I)
            a.set_title('Predicted Keypoints')
            ann_ids = coco_pred.getAnnIds(imgIds=img['id'])
            anns = coco_pred.loadAnns(ann_ids)
            coco_pred.showAnns(anns)

            a = fig.add_subplot(2, 2, 3)
            plt.imshow(I)
            a.set_title('GT Keypoints')
            ann_ids = self.coco.getAnnIds(imgIds=img['id'])
            anns = self.coco.loadAnns(ann_ids)
            self.coco.showAnns(anns)

            plt.show()

    def __getitem__(self, item):
        batch = self.get_item(item)
        res = (
            batch.get(Batch.inputs, None),
            batch.get(Batch.part_score_targets, None),
            batch.get(Batch.part_score_weights, None),
            batch.get(Batch.locref_targets, None),
            batch.get(Batch.locref_mask, None),
            batch.get(Batch.pairwise_targets, None),
            batch.get(Batch.pairwise_mask, None),
        )
        if not self.cfg.train:
            res = (batch[Batch.data_item].im_path,) + res
        return res
