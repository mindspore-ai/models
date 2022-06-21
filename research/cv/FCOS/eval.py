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
"""FCOS EVAL"""
import json
import os
import argparse
import cv2
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import Tensor
from mindspore import context
from mindspore.ops import stop_gradient

from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.fcos import FCOSDetector
from src.eval_utils import post_process
from src.eval_utils import ClipBoxes

class COCOGenerator:
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, dataset_dir, annotation_file, resize_size):
        self.coco = COCO(annotation_file)
        self.root = dataset_dir
        ids = list(sorted(self.coco.imgs.keys()))
        print("INFO====>check annos, filtering invalid data......")
        new_ids = []
        for i in ids:
            ann_id = self.coco.getAnnIds(imgIds=i, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                new_ids.append(i)
        self.ids = new_ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.resize_size = resize_size

        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]



    def getImg(self, index):
        img_id = self.ids[index]
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        return img, target

    def __getitem__(self, index):

        img, ann = self.getImg(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        img = np.array(img)
        img, boxes, scale = self.preprocess_img_boxes(img, boxes, self.resize_size)
        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]
        to_tensor = py_vision.ToTensor()
        img = to_tensor(img)
        max_h = 1344
        max_w = 1344
        max_num = 90
        img = np.pad(img, ((0, 0), (0, max(int(max_h - img.shape[1]), 0)), \
        (0, max(int(max_w - img.shape[2]), 0))))
        normalize_op = c_vision.Normalize(mean=[0.40789654, 0.44719302, 0.47026115], \
        std=[0.28863828, 0.27408164, 0.27809835])
        img = img.transpose(1, 2, 0)    # chw to hwc
        img = normalize_op(img)
        img = img.transpose(2, 0, 1)     #hwc to chw
        boxes = np.pad(boxes, ((0, max(max_num-boxes.shape[0], 0)), (0, 0)), 'constant', constant_values=-1)
        classes = np.pad(classes, (0, max(max_num - len(classes), 0)), 'constant', constant_values=-1).astype('int32')
        box_info = {"boxes": boxes, "classes": classes, "scale": scale}
        return img, box_info
    def __len__(self):
        return len(self.ids)

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32
        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized
        if boxes is not None:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
        return image_paded, boxes, scale

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if annot is None:
            return False
        if self._has_only_empty_bbox(annot):
            return False

        return True

def evaluate_coco(_generator, _model, threshold=0.05):
    results = []
    image_ids = []
    for index in tqdm(range(len(_generator))):
        img, box_info = _generator[index]
        scale = box_info["scale"]
        img = Tensor(img, mindspore.float32)
        expand_dims = ops.ExpandDims()
        img = expand_dims(img, 0)
        batch_imgs = img
        scores, labels, boxes = _model(img)
        scores, labels, boxes = post_process([scores, labels, boxes], 0.05, 0.6)
        boxes = ClipBoxes(batch_imgs, boxes)
        scores = stop_gradient(scores)
        labels = stop_gradient(labels)
        boxes = stop_gradient(boxes)
        boxes /= scale
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]
        boxes = boxes.asnumpy()
        labels = labels.asnumpy()
        scores = scores.asnumpy()
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < threshold:
                break
            image_result = {
                'image_id': _generator.ids[index],
                'category_id': _generator.id2category[label],
                'score': float(score),
                'bbox': box.tolist(),
            }
            results.append(image_result)
        image_ids.append(_generator.ids[index])
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    coco_true = _generator.coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=int, default=0, help="DEVICE_ID to run ")
parser.add_argument("--eval_path", type=str, default="/data2/dataset/coco2017/val2017")
parser.add_argument("--anno_path", type=str, default="/data2/dataset/coco2017/annotations/instances_val2017.json")
parser.add_argument("--ckpt_path", type=str, default="/data1/FCOS/checkpoint/backbone/s1.ckpt")
opt = parser.parse_args()
if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=opt.device_id)
    generator = COCOGenerator(opt.eval_path, opt.anno_path, [800, 1333])
    model = FCOSDetector(mode="inference")
    model.set_train(False)
    mindspore.load_param_into_net(model, mindspore.load_checkpoint(opt.ckpt_path))
    evaluate_coco(generator, model)
