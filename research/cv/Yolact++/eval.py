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

""" eval my model """

import json
import time
import os
import argparse
import math
import random
import numpy as np
import cv2
from tqdm import tqdm
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.yolact.layers.modules.loss import mask_iou_net
from src.yolact.yolactpp import Yolact
from.src.config import COCO_CLASSES, COCO_LABEL_MAP_EVAL

MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--pre_trained',
                    default='/data/weights/yolact_plus_resnet50_800000.ckpt', type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                    help='The output file for coco bbox results if --coco_results is set.')
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--seed', default=None, type=int,
                    help='The seed to pass into random.seed. Note: this is"\
                    "only really for the shuffle and does not (I think) affect cuda stuff.')
parser.add_argument('--score_threshold', default=0, type=float,
                    help='Detections with a score under this threshold"\
                    "will not be considered. This currently only works in display mode.')
parser.add_argument('--valid_img_path', default="/data/coco2017/val2017", type=str)
parser.add_argument('--gt_ann_file', default="/data/coco2017/annotations/instances_val2017.json", type=str)
parser.add_argument('--eval_type', default='both', choices=['bbox', 'mask', 'both'], type=str)
parser.add_argument("--device_id", type=int, default=1, help="Device id, default is 0.")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                    device_id=args.device_id, enable_reduce_precision=True)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = {}

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id

def get_label_map():
    return COCO_LABEL_MAP_EVAL

class COCOAnnotationTransform():
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]
            else:
                print("No bbox found for object ", obj)

        return res


class CocoDataset():
    """ Operate on the dataset """
    def __init__(self, image_path, info_file, transform=None,
                 dataset_name='MS COCO', has_gt=True):

        self.root = image_path
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())

        if self.ids.isspace() or not has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = COCOAnnotationTransform()

        self.name = dataset_name
        self.has_gt = has_gt

    def __len__(self):
        return len(self.ids)

    def pull_anno(self, img_id):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def pull_image(self, img_id):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:

        Return:
             img info . ndarray
        '''

        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_item(self, img_id):
        """
        Returns:
                tuple: Tuple (image, target, masks, height, width, crowd).
                       target is the object returned by ``coco.loadAnns``.
                Note that if no crowd annotations exist, crowd will be None

        """

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = os.path.join(self.root, file_name)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if target.isspace() is not True:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and target.isspace() is not True:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if target.isspace() is not True:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                           {'num_crowds': num_crowds, 'labels': target[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float),
                                              np.array([[0, 0, 1, 1]]),
                                              {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        if target[0].isspace() is True:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids) - 1))

        return img.transpose([2, 0, 1]), target, masks, height, width, num_crowds

def sanitize_coordinates(_x1, _x2, img_size: int, padding: int = 0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=None)
    x2 = np.clip(x2 + padding, a_min=None, a_max=img_size)

    return x1, x2

def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    np.arange(w, dtype=x1.dtype)
    rows = np.arange(w, dtype=x1.dtype).reshape((1, -1, 1)).repeat(repeats=h, axis=0).repeat(repeats=n, axis=2)
    cols = np.arange(h, dtype=x1.dtype).reshape((-1, 1, 1)).repeat(repeats=w, axis=1).repeat(repeats=n, axis=2)

    masks_left = rows >= x1.reshape((1, 1, -1))
    masks_right = rows < x2.reshape((1, 1, -1))
    masks_up = cols >= y1.reshape((1, 1, -1))
    masks_down = cols < y2.reshape((1, 1, -1))

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.astype(np.float32)


def postprocess(det_output, w, h, batch_idx=0, crop_masks=True, score_threshold=0.15):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    dets = det_output[batch_idx]
    dets = dets['detection'] # dict, contains 5 tensor

    if dets is None:
        return [np.array([]), np.array([]), np.array([]), np.array([])]  # Warning, this is 4 copies of the same thing

    for k in dets:
        dets[k] = dets[k].asnumpy()

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        # modified
        if dets['score'].shape[0] == 0:
            return [np.array([]), np.array([]), np.array([]), np.array([])]

    # Actually extract everything from dets now
    classes = dets['class']   #[100,]  need to be expand to [100,1]
    boxes = dets['box']
    scores = dets['score']
    masks = dets['mask']

    # At this points masks is only the coefficients
    proto_data = dets['proto']

    masks = np.dot(proto_data, masks.T)  # [138, 138, 32] * [32, 100] -> [138, 138, 100]
    masks = 1 / (1 + np.exp(-masks))  # sigmoid

    # Crop masks before upsampling because you know why
    if crop_masks:  # True
        masks = crop(masks, boxes)

    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.transpose((2, 0, 1))

    # =========== maskiou_net  ==================
    masks = np.transpose(cv2.resize(np.transpose(masks, (2, 1, 0)), (h, w), interpolation=cv2.INTER_LINEAR), (2, 1, 0)) # it is bilinear interpolate

    # Binarize the masks.
    # this op will create a array like [[1,0,0,1...,1],[...]], all elements below 0.5 will be replaced by 0, otherwise 1.
    masks = np.greater(masks, 0.5).astype(np.float32)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)

    return classes, scores, boxes, masks


def gather_numpy(self, dim, index):

    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections):
    """ postprocess """
    # classes [100,], scores [100, ], boxes [100,4], masks [100, H, W]
    classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=True, score_threshold=args.threshold)
    if classes.shape.isspace() is True:
        return

    classes = list(classes.astype(int))

    if isinstance(scores, list):
        box_scores = list(scores[0].astype(float))
        mask_scores = list(scores[1].astype(float))
    else:
        scores = list(scores.astype(float))
        box_scores = scores
        mask_scores = scores
    masks = masks.reshape((-1, h * w))

    # ============== output json module ====================
    # add bboxes abd masks into detection obj.
    masks = masks.reshape((-1, h, w))
    for i in range(masks.shape[0]):
        # Make sure that the bounding box actually makes sense and a mask was produced
        if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
            detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
            detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
    return

class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

class Detections:
    """ Detect the model """
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ x1 y1 x2 y2 -> x1 y1 w h """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

def eval_network(net, dataset):
    """ eval """
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False

    dataset_size = len(dataset)
    print("dataset_size : {}".format(dataset_size))

    # For each class and iou, stores tuples (score, isPositive)
    # Index ap_data[type][iouIdx][classIdx]
    ap_data = {
        'box': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds]
    }
    detections = Detections()

    for image_idx in tqdm(dataset.ids):
    # ============== load data =================
    # img :  tensor([3,550,550]) in source code, now ndarray
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
        img = np.expand_dims(img, axis=0)  # [1,3,550,550]

        cur_time = time.time()
        preds = net(ms.Tensor(img).astype(ms.float32))
        print("forward consume {} s. ".format(time.time()-cur_time))

        prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, image_idx, detections)

    print('Dumping detections...')
    detections.dump()

    print("Dump json success. ")

def _eval():
    """ _eval """
    eval_bbox = (args.eval_type in ('bbox', 'both'))
    eval_mask = (args.eval_type in ('mask', 'both'))

    print('Loading annotations...')
    gt_annotations = COCO(args.gt_ann_file)
    if eval_bbox:
        bbox_dets = gt_annotations.loadRes(args.bbox_det_file)
    if eval_mask:
        mask_dets = gt_annotations.loadRes(args.mask_det_file)

    if eval_bbox:
        print('\nEvaluating BBoxes:')
        bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

    if eval_mask:
        print('\nEvaluating Masks:')
        bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()




class ConvertFromInts():
    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image.astype(np.float32), masks, boxes, labels



class Resize():
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I matched this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = math.sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = 550
        self.preserve_aspect_ratio = False

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape

        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        image = cv2.resize(image, (width, height))

        if self.resize_gt:
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))

            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > 4/550) * (h > 4/550)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, labels


class Compose():
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        # >>> augmentations.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels



class BackboneTransform():
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    def __init__(self, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in 'RGB']

    def __call__(self, img, masks=None, boxes=None, labels=None):

        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32), masks, boxes, labels



class BaseTransform():
    """BaseTransform"""
    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            Resize(resize_gt=False),
            BackboneTransform(mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


class ToAbsoluteCoords():
    def __call__(self, image, masks=None, boxes=None, labels=None):
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, masks, boxes, labels


if __name__ == '__main__':
    parse_args()

    if not os.path.exists(args.bbox_det_file) or not os.path.exists(args.mask_det_file):

        datasets = CocoDataset(args.valid_img_path, args.gt_ann_file, transform=BaseTransform(), has_gt=True)

        prep_coco_cats()

        network = Yolact()
        mask_iou_net = mask_iou_net()

        ckpt_path = args.pre_trained

        if ckpt_path:
            param_dict = load_checkpoint(ckpt_path)
            load_param_into_net(network, param_dict)

        eval_network(net, datasets)

    # eval bbox and masks json
    _eval()
