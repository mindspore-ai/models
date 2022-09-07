# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" dataset """
from __future__ import division
import os
import numpy as np
import cv2
import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter
from pycocotools.coco import COCO
from src.config import config

def _rand(a=0., b=1.):
    """Generate random."""
    return np.random.rand() * (b - a) + a


class Augmenter:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            _, cols, _ = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample

class Normalizer:
    """ normalizer """
    def __init__(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class Resizer:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        annots = annots.astype(np.float32)
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': new_image.astype(np.float32), 'annot': annots.astype(np.float32), 'scale': scale}


def preprocess_fn(image, box):
    """ preprocess img and anno """
    def _data_aug(image, box, image_size=(512, 512)):

        box = box.astype(np.float32)

        image = image.astype(np.float32) / 255.

        sample = {'img': image.astype(np.float32), 'annot': box.astype(np.float32)}

        sample = Normalizer()(sample)
        sample = Augmenter()(sample)
        sample = Resizer()(sample)

        box_new = sample['annot']

        bottom = 128 - box_new.shape[0]

        padded_box = np.pad(box_new, ((0, bottom), (0, 0)))

        return sample['img'], padded_box

    return _data_aug(image, box, image_size=config.img_shape)


def create_EfficientDet_datasets(mindrecord_file, batch_size, repeat_num, device_num=1, rank=0,
                                 is_training=True, num_parallel_workers=24):
    """Create EfficientDet dataset with MindDataset."""

    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation"], num_shards=device_num,
                        shard_id=rank, num_parallel_workers=num_parallel_workers, shuffle=True)

    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])

    output_columns = ["image", "anno"]

    ds = ds.map(operations=preprocess_fn, input_columns=["image", "annotation"],
                output_columns=output_columns,
                python_multiprocessing=is_training,
                num_parallel_workers=num_parallel_workers)

    change_swap_op = C.HWC2CHW()

    ds = ds.map(operations=change_swap_op, input_columns=["image"], python_multiprocessing=is_training,
                num_parallel_workers=num_parallel_workers)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_num)

    return ds

def create_coco_label(is_training=True, coco_path=None):
    """Get image path and annotation from COCO."""
    coco_root = config.coco_root
    if not coco_path is None:
        coco_root = coco_path
    data_type = config.train_data_type  # "train2017"

    anno_json = os.path.join(coco_root, config.instances_set.format(data_type))

    coco = COCO(anno_json)  # COCO api class that loads COCO annotation file and prepare data structures.

    image_ids = coco.getImgIds()

    img_ids = []
    image_path_dict = {}
    image_anno_dict = {}


    for index in range(0, len(image_ids) - 1):

        img_id = image_ids[index]

        image_info = coco.loadImgs(img_id)[0]
        file_name = image_info["file_name"]
        image_path = os.path.join(coco_root, data_type, file_name)

        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        coco_annotations = coco.loadAnns(anno_ids)
        annotations = np.zeros((0, 5))
        for _, a in enumerate(coco_annotations):

            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        if len(anno_ids) >= 1:
            img_ids.append(index)
            image_path_dict[index] = image_path
            image_anno_dict[index] = np.array(annotations)
        else:
            img_ids.append(index)
            image_path_dict[index] = image_path
            annotations = np.zeros((1, 5))
            image_anno_dict[index] = np.array(annotations)

    return img_ids, image_path_dict, image_anno_dict

def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="EfficientDet.mindrecord", file_num=8):
    """ Create MindRecord file. """
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)

    img_ids, image_path_dict, image_anno_dict = create_coco_label(is_training)

    EfficientDet_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "float32", "shape": [-1, 5]},

    }

    writer.add_schema(EfficientDet_json, "EfficientDet_json")

    for img_id in img_ids:
        image_path = image_path_dict[img_id]
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[img_id], dtype=np.float32)
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()

def create_mindrecord(dataset="coco", prefix="EfficientDet.mindrecord", is_training=True):
    """ Create mindrecord"""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", is_training, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
    return mindrecord_file
