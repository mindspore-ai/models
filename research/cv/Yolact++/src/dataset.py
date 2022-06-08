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

""" Operate on the dataset """
import os
import cv2
import mmcv
import numpy as np
from numpy import random
import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter
from src.config import MEANS
from src.config import yolact_plus_resnet50_config as cfg


def pad_to_max(img, gt_bboxes, gt_label, crowd_boxes, gt_mask, instance_count, crowd_count):
    pad_max_number = cfg['max_instance_count']
    gt_box_new = np.pad(gt_bboxes, ((0, pad_max_number - instance_count), (0, 0)), mode="constant", constant_values=0)
    crowd_box_new = np.pad(crowd_boxes, ((0, 10 - crowd_count), (0, 0)), mode="constant", constant_values=0)
    gt_label_new = np.pad(gt_label, ((0, pad_max_number - instance_count)), mode="constant", constant_values=-1)

    return img, gt_box_new, gt_label_new, crowd_box_new, gt_mask

def transpose_column(img, gt_bboxes, gt_label, crowd_boxes, gt_mask):
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)
    crowd_box_new = crowd_boxes.astype(np.float32)
    gt_label = gt_label.astype(np.int32)
    gt_mask_data = gt_mask.astype(np.bool)
    return (img_data, gt_bboxes, gt_label, crowd_box_new, gt_mask_data)

def toPercentCoords(img, gt_bboxes, gt_label, num_crowds, gt_mask):
    height, width, _ = img.shape
    gt_bboxes[:, 0] /= width
    gt_bboxes[:, 2] /= width
    gt_bboxes[:, 1] /= height
    gt_bboxes[:, 3] /= height

    return (img, gt_bboxes, gt_label, num_crowds, gt_mask)

def imnormalize_column(img, gt_bboxes, gt_label, gt_num, gt_mask):

    """imnormalize operation for image"""
    img_data = mmcv.imnormalize(img, [123.68, 116.78, 103.94], [58.40, 57.12, 57.38], True)
    img_data = img_data.astype(np.float32)
    return (img_data, gt_bboxes, gt_label, gt_num, gt_mask)

def backboneTransform(img, gt_bboxes, gt_label, num_crowds, gt_mask):

    c = BackboneTransform()
    img_data = c(img)
    return (img_data, gt_bboxes, gt_label, num_crowds, gt_mask)

class BackboneTransform():
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    def __init__(self):
        self.mean = np.array((103.94, 116.78, 123.68), dtype=np.float32)
        self.std = np.array((57.38, 57.12, 58.40), dtype=np.float32)

        self.channel_map = {c: idx for idx, c in enumerate('BGR')}
        self.channel_permutation = [self.channel_map[c] for c in 'RGB']

    def __call__(self, img):

        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32)

def resize_column(img, gt_bboxes, gt_label, num_crowds, gt_mask, resize_gt=True):
    """resize operation for image"""
    img_data = img
    img_data, w_scale, h_scale = mmcv.imresize(
        img_data, (cfg['img_width'], cfg['img_height']), return_scale=True)

    if resize_gt:
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = (cfg['img_height'], cfg['img_width'], 1.0)
        img_shape = np.asarray(img_shape, dtype=np.float32)
        gt_bboxes = gt_bboxes * scale_factor
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)  # x1, x2   [0, W-1]
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)  # y1, y2   [0, H-1]
        gt_mask_data = np.array([
            mmcv.imresize(mask, (cfg['img_width'], cfg['img_height']), interpolation='nearest')
            for mask in gt_mask])

    w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    h = gt_bboxes[:, 3] - gt_bboxes[:, 1]

    keep = (w > cfg['discard_box_width']) * (h > cfg['discard_box_height'])
    gt_mask_data = gt_mask_data[keep]
    gt_bboxes = gt_bboxes[keep]
    gt_label = gt_label[keep]
    num_crowds[0] = (gt_label < 0).sum()

    return (img_data, gt_bboxes, gt_label, num_crowds, gt_mask_data)

def randomMirror(image, boxes, gt_label, num_crowds, masks):

    _, width, _ = image.shape
    if random.randint(2):
        image = image[:, ::-1]
        masks = masks[:, :, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes, gt_label, num_crowds, masks


def randomSampleCrop(image, boxes, gt_label, num_crowds, masks):
    """Random Crop the image and boxes"""
    height, width, _ = image.shape
    while True:
        min_iou = np.random.choice([None, 0.1, 0.3, 0.7, 0.9])
        if min_iou is None:
            return image, boxes, gt_label, num_crowds, masks
        # max trails (50)
        for _ in range(50):
            image_t = image

            w = _rand(0.3, 1.0) * width
            h = _rand(0.3, 1.0) * height

            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            left = _rand() * (width - w)
            top = _rand() * (height - h)

            rect = np.array([int(left), int(top), int(left + w), int(top + h)])
            overlap = jaccard_numpy(boxes, rect)


            if overlap.min() < min_iou and overlap.max() > (min_iou + 0.2):
                continue

            # cut the crop from the image
            image_t = image_t[rect[1]:rect[3], rect[0]:rect[2], :]
            centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

            m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
            m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

            # mask in that both m1 and m2 are true
            mask = m1 * m2

            crowd_mask = np.zeros(mask.shape, dtype=np.int32)
            if num_crowds[0] > 0:
                crowd_mask[-num_crowds[0]:] = 1

            # have any valid boxes? try again if not
            # Also make sure you have at least one regular gt
            if not mask.any() or np.sum(1 - crowd_mask[mask]) == 0:
                continue
            masks_t = masks[mask, :, :].copy()

            # # take only matching gt labels
            # take only matching gt boxes
            boxes_t = boxes[mask, :].copy()

            labels_t = gt_label[mask]
            if num_crowds[0] > 0:
                num_crowds[0] = np.sum(crowd_mask[mask])

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
            boxes_t[:, :2] -= rect[:2]
            boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
            boxes_t[:, 2:4] -= rect[:2]
            masks_t = masks_t[:, rect[1]:rect[3], rect[0]:rect[2]]

            return (image_t, boxes_t, labels_t, num_crowds, masks_t)

def expand_column(img, gt_bboxes, gt_label, num_crowds, gt_mask):
    """expand operation for image"""
    expand = Expand(MEANS)
    img, gt_bboxes, gt_label, gt_mask = expand(img, gt_bboxes, gt_label, gt_mask)

    return (img, gt_bboxes, gt_label, num_crowds, gt_mask)

class Expand():
    """expand image"""
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img, boxes, labels, mask):
        if random.randint(2):
            return img, boxes, labels, mask

        h, w, c = img.shape
        ratio = random.uniform(1, 4)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        # Deal with bounding box
        boxes += np.tile((left, top), 2)

        mask_count, mask_h, mask_w = mask.shape
        expand_mask = np.zeros((mask_count, int(mask_h * ratio), int(mask_w * ratio))).astype(mask.dtype)
        expand_mask[:, top:top + h, left:left + w] = mask
        mask = expand_mask

        return img, boxes, labels, mask

def photoMetricDistortion(img, gt_bboxes, gt_label, num_crowds, gt_mask):

    c = PhotoMetricDistortion()
    img, gt_bboxes, gt_label = c(img, gt_bboxes, gt_label)

    return (img, gt_bboxes, gt_label, num_crowds, gt_mask)

class PhotoMetricDistortion:
    """Photo Metric Distortion"""
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        img = img.astype('float32')
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha
        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)
        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)
        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha
        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]
        return img, boxes, labels

def preprocess_fn(image, box, mask, num_crowdses, is_training):
    """Preprocess function for dataset."""
    def _infer_data(image_bgr, gt_box, gt_label, num_crowdses,
                    gt_mask):
        image_new = image_bgr.astype('float32')
        input_data = image_new, gt_box, gt_label, num_crowdses, gt_mask
        input_data = resize_column(*input_data, resize_gt=False)
        output_data = backboneTransform(*input_data)
        return output_data

    def _data_aug(image, box, mask, num_crowdses, is_training):
        """Data augmentation function."""

        gt_box = box[:, :4]
        gt_label = box[:, 4]
        gt_mask = mask.copy()
        h = image.shape[0]
        w = image.shape[1]
        gt_mask = gt_mask.reshape(-1, h, w)

        if not is_training:
            return _infer_data(image, gt_box, gt_label, num_crowdses, gt_mask)

        input_data = image, gt_box, gt_label, num_crowdses, gt_mask
        input_data = photoMetricDistortion(*input_data)
        input_data = expand_column(*input_data)
        input_data = resize_column(*input_data)
        input_data = backboneTransform(*input_data)
        input_data = toPercentCoords(*input_data)
        input_data = split_crowd(*input_data)
        _, gt_box, _, crowd_boxes, _ = input_data
        instance_count_after = gt_box.shape[0]
        crowd_count = crowd_boxes.shape[0]
        input_data = pad_to_max(*input_data, instance_count_after, crowd_count)
        output_data = transpose_column(*input_data)

        return output_data

    return _data_aug(image, box, mask, num_crowdses, is_training)

def split_crowd(img, gt_bboxes, gt_label, num_crowds, gt_mask):
    """Split the crowd"""
    cur_crowds = num_crowds[0]
    if cur_crowds > 0:
        split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
        crowd_boxes, truths = split(gt_bboxes)

        # We don't use the crowd labels or masks
        _, split_label = split(gt_label)

        _, split_mask = split(gt_mask)

    else:
        crowd_boxes = np.array([[0, 0, 0, 0]])
        truths = gt_bboxes
        split_label = gt_label
        split_mask = gt_mask

    return (img, truths, split_label, crowd_boxes, split_mask)


def create_coco_label(is_training, coco_path=None):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = cfg['coco_root']

    if coco_path:
        coco_root = coco_path

    data_type = cfg['val_data_type']
    if is_training:
        data_type = cfg['train_data_type']

    anno_json = os.path.join(coco_root, cfg['instance_set'].format(data_type))

    coco = COCO(anno_json)

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}
    masks = {}
    masks_shape = {}
    num_crowdses = {}
    images_num = len(image_ids)
    for ind, img_id in enumerate(image_ids):
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(coco_root, data_type, file_name)
        if not os.path.isfile(image_path):
            continue
        crowd = coco.getAnnIds(imgIds=img_id, iscrowd=True)
        target = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anno_crowd = coco.loadAnns(crowd)
        anno_target = coco.loadAnns(target)
        num_crowds = len(anno_crowd)
        for x in anno_crowd:
            x['category_id'] = -1
        anno = anno_target + anno_crowd
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        instance_masks = []
        image_height = coco.imgs[img_id]["height"]
        image_width = coco.imgs[img_id]["width"]
        if (ind + 1) % 1000 == 0:
            print("{}/{}: parsing annotation for image={}".format(ind + 1, images_num, file_name))
        if not is_training:
            image_files.append(image_path)
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])
            masks[image_path] = np.zeros([1, 1, 1], dtype=np.bool).tobytes()
            masks_shape[image_path] = np.array([1, 1, 1], dtype=np.int32)
        else:
            for label in anno:
                # get coco mask
                m = annToMask(label, image_height, image_width)
                instance_masks.append(m)
                bbox = label["bbox"]
                label_idx = label['category_id']
                _map = get_label_map()
                if label_idx >= 0:
                    label_idx = _map[label_idx] - 1

                # get coco bbox
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append([x1, y1, x2, y2] + [label_idx])

            if annos:
                image_anno_dict[image_path] = np.array(annos)
                instance_masks = np.stack(instance_masks, axis=0).astype(np.bool)
                masks[image_path] = np.array(instance_masks).tobytes()
                num_crowdses[image_path] = np.array(num_crowds)
            else:
                print("no annotations for image ", file_name)
                continue

            image_files.append(image_path)

    return image_files, image_anno_dict, masks, num_crowdses

def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="yolact.mindrecord",
                                  file_num=8, mind_path=None, coco_path=None):
    """Create MindRecord file."""
    mindrecord_dir = cfg['mindrecord_dir']
    if mind_path:
        mindrecord_dir = mind_path
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        image_files, image_anno_dict, masks, num_crowdses = create_coco_label(is_training, coco_path)
    else:
        print("Error unsupported other dataset")
        return

    yolact_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
        "mask": {"type": "bytes"},
        "num_crowdses": {"type": "int32", "shape": [-1]}
    }
    writer.add_schema(yolact_json, "yolact_json")

    image_files_num = len(image_files)
    for ind, image_name in enumerate(image_files):
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        mask = masks[image_name]
        num_crowds = num_crowdses[image_name]
        row = {"image": img, "annotation": annos, "mask": mask, "num_crowdses": num_crowds}
        if (ind + 1) % 1000 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])
    writer.commit()

def create_yolact_dataset(mindrecord_file, batch_size=2, device_num=1, rank_id=0,
                          is_training=True, num_parallel_workers=8):
    """Create MaskRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation", "mask", "num_crowdses"],
                        num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4, shuffle=is_training)

    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    compose_map_func = (lambda image, annotation, mask, num_crowds:
                        preprocess_fn(image, annotation, mask, num_crowds, is_training))


    if is_training:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation", "mask", "num_crowdses"],
                    output_columns=["image", "box", "label", "crowd_box", "mask"],
                    column_order=["image", "box", "label", "crowd_box", "mask"],
                    python_multiprocessing=False,
                    num_parallel_workers=8)

        ds = ds.batch(batch_size, drop_remainder=True, pad_info={"mask": ([cfg['max_instance_count'], None, None], 0)})

    else:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation", "mask", "mask_shape"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds

def save_mask(img, name1, name2):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """
    img_path = './afterpicture'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    filename = str(name1)+ str(name2)+"mm"+".jpg"
    img.save(os.path.join(img_path, filename))

def save_image(img, name):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """

    img_path = './afterpicture'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    filename = str(name)+"yuan"+".jpg"
    img.save(os.path.join(img_path, filename))


def annToMask(ann, height, width):
    """Convert annotation to RLE and then to binary mask."""
    from pycocotools import mask as maskHelper
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = maskHelper.frPyObjects(segm, height, width)
        rle = maskHelper.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskHelper.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    m = maskHelper.decode(rle)
    return m

def get_label_map():
    return cfg['dataset']['label_map_eval']

def _rand(a=0., b=1.):
    """Generate random."""
    return np.random.rand() * (b - a) + a

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))
    union = area_a + area_b - inter
    return inter / union
