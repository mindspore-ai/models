"""
Precision generation
"""

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

import os
from datetime import datetime
import json

from absl import flags
from absl import app
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PRINT_LINES_TEMPLATE = """
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f
"""

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="annotations_json",
    default=None,
    help="annotations_json file path name",
)

flags.DEFINE_string(
    name="det_result_json", default=None, help="det_result json file"
)

flags.DEFINE_enum(
    name="anno_type",
    default="bbox",
    enum_values=["segm", "bbox", "keypoints"],
    help="Annotation type",
)

flags.DEFINE_string(
    name="output_path_name",
    default=None,
    help="Where to out put the result files.",
)

flags.mark_flag_as_required("annotations_json")
flags.mark_flag_as_required("det_result_json")
flags.mark_flag_as_required("output_path_name")

def get_category_id(k):
    """
    :param: class id which corresponding coco.names
    :return: category id is used in instances_val2014.json
    """
    kk = k
    if 12 <= k <= 24:
        kk = k + 1
    elif 25 <= k <= 26:
        kk = k + 2
    elif 27 <= k <= 40:
        kk = k + 4
    elif 41 <= k <= 60:
        kk = k + 5
    elif k == 61:
        kk = k + 6
    elif k == 62:
        kk = k + 8
    elif 63 <= k <= 73:
        kk = k + 9
    elif 74 <= k <= 80:
        kk = k + 10
    return kk


def get_dict_from_file(file_path):
    """
    :param: file_path contain all infer result
    :return: dict_list contain infer result of every images
    """
    ls = []
    # Opening JSON file
    f = open(file_path)

    # returns JSON object as
    # a dictionary
    ban_list = json.load(f)

    for item in ban_list:
        item_copy = item.copy()
        item_copy['category_id'] = get_category_id(int(item['category_id']))
        item_copy['image_id'] = int(item['image_id'])
        ls.append(item_copy)

    return ls


def main(unused_arg):
    del unused_arg
    out_put_dir = os.path.dirname(FLAGS.output_path_name)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    fw = open(FLAGS.output_path_name, "a+")
    now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head_info = f"{'-'*50}mAP Test starts @ {now_time_str}{'-'*50}\n"
    fw.write(head_info)
    fw.flush()

    cocoGt = COCO(FLAGS.annotations_json)

    image_ids = cocoGt.getImgIds()

    need_img_ids = []
    for img_id in image_ids:
        iscrowd = False
        anno_ids = cocoGt.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = cocoGt.loadAnns(anno_ids)
        for label in anno:
            iscrowd = iscrowd or label["iscrowd"]

        if iscrowd:
            continue
        need_img_ids.append(img_id)

    result_dict = get_dict_from_file(FLAGS.det_result_json)
    json_file_name = './result.json'
    with open(json_file_name, 'w') as f:
        json.dump(result_dict, f)

    cocoDt = cocoGt.loadRes(json_file_name)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=FLAGS.anno_type)
    cocoEval.params.imgIds = sorted(need_img_ids)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    format_lines = [
        line for line in PRINT_LINES_TEMPLATE.splitlines() if line.strip()
    ]
    for i, line in enumerate(format_lines):
        fw.write(line % cocoEval.stats[i] + "\n")

    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tail_info = f"{'-'*50}mAP Test ends @ {end_time_str}{'-'*50}\n"
    fw.write(tail_info)
    fw.close()


if __name__ == "__main__":
    app.run(main)
