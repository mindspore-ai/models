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
"""eval"""
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.backbone import EfficientDetBackbone
from src.utils import preprocess, invert_affine, postprocess, boolean_string
from src.config import config
from mindspore import load_checkpoint, load_param_into_net
import mindspore
from mindspore import context

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
ap.add_argument("--checkpoint_path", type=str, default="/data/efficientdet_ch/efdet.ckpt", help="") # ckpt path.
ap.add_argument("--data_url", type=str, default="", help="dataset path on modelarts.")
ap.add_argument("--train_url", type=str, default="", help="necessary for modelarts")
ap.add_argument("--is_modelarts", type=str, default="False", help="")
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
override_prev_results = args.override
project_name = args.project

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", enable_reduce_precision=True)
device_id = int(os.getenv("DEVICE_ID"))
context.set_context(device_id=device_id)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

obj_list = config.coco_classes

def evaluate_coco(img_path, set_name, img_ids, coco, net, threshold=0.05):
    """ eval on coco dataset """
    results = []

    for image_id in tqdm(img_ids):

        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + '/' + image_info['file_name']

        _, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                  mean=config.mean, std=config.std)
        x = framed_imgs[0].astype(np.float32)

        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, axes=(0, 3, 1, 2))
        x = mindspore.Tensor(x)

        _, regression, classification, anchors = net(x)

        preds = postprocess(x=x, anchors=anchors, regression=regression, classification=classification,
                            threshold=threshold, iou_threshold=nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]
            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    print("results len :{}".format(len(results)))

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    print("save json success.")


def _eval(gt, img_ids, pred_json_path):
    """ call coco api to eval output json"""

    # load results in COCO evaluation tool
    coco_pred = gt.loadRes(pred_json_path)
    # run COCO evaluation
    coco_eval = COCOeval(gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':

    if args.is_modelarts == "True":
        import moxing as mox
        local_data_url = "/cache/data/"
        mox.file.make_dirs(local_data_url)
        mox.file.copy_parallel(args.data_url, local_data_url)

        checkpoint_path = "/cache/ckpt"
        mox.file.make_dirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, "efdet.ckpt")
        mox.file.copy(args.checkpoint_path, checkpoint_path)

    else:
        local_data_url = config.coco_root
        checkpoint_path = args.checkpoint_path

    coco_root = local_data_url
    data_type = config.val_data_type
    VAL_GT = os.path.join(coco_root, config.instances_set.format(data_type))

    SET_NAME = config.val_data_type
    VAL_IMGS = os.path.join(coco_root, SET_NAME)
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):

        model = EfficientDetBackbone(config.num_classes, 0, False, False)
        print("Load Checkpoint!")

        param_dict = load_checkpoint(checkpoint_path)
        model.init_parameters_data()
        load_param_into_net(model, param_dict)
        model.set_train(False)
        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    print("go into eval json")
    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json') # call coco api to eval output json
