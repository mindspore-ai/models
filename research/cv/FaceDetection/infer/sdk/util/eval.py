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
"""Face detection eval."""

import os
import matplotlib.pyplot as plt

from util import voc_wrapper
from util.eval_util import get_bounding_boxes, tensor_to_brambox, parse_gt_from_anno, parse_rets, \
    calc_recall_precision_ap

plt.switch_backend('agg')


def eval_according_output(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2, det,
                          img_anno, img_size, batch_labels, batch_image_name, batch_image_size):
    """
    eval_according_output
    """
    labels = ['face']
    num_classes = 1

    dets = []
    tdets = []
    input_shape = [768, 448]
    conf_thresh = 0.1
    boxes_0, boxes_1, boxes_2 = get_bounding_boxes(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2,
                                                   cls_scores_2, conf_thresh, input_shape,
                                                   num_classes)

    converted_boxes_0, converted_boxes_1, converted_boxes_2 = tensor_to_brambox(boxes_0, boxes_1, boxes_2,
                                                                                input_shape, labels)

    tdets.append(converted_boxes_0)
    tdets.append(converted_boxes_1)
    tdets.append(converted_boxes_2)

    batch = len(tdets[0])
    for b in range(batch):
        single_dets = []
        for op in range(3):
            single_dets.extend(tdets[op][b])
        dets.append(single_dets)
    det.update({batch_image_name[k]: v for k, v in enumerate(dets)})
    img_size.update({batch_image_name[k]: v for k, v in enumerate(batch_image_size)})
    img_anno.update({batch_image_name[k]: v for k, v in enumerate(batch_labels)})


def gen_eval_result(eval_times, det, img_size, img_anno):
    """
    gen_eval_result
    :param eval_times: the eval times
    :param det: the infer result
    :param img_size: the image size
    :param img_anno:the image label
    """
    result_path = os.path.join('./result')
    if os.path.exists(result_path):
        pass
    if not os.path.isdir(result_path):
        os.makedirs(result_path, exist_ok=True)

    # result file
    ret_files_set = {'face': os.path.join(result_path, 'comp4_det_test_face_rm5050.txt')}

    print('eval times:', eval_times)
    input_shape = [768, 448]
    netw, neth = input_shape
    reorg_dets = voc_wrapper.reorg_detection(det, netw, neth, img_size)
    nms_thresh = 0.45
    voc_wrapper.gen_results(reorg_dets, result_path, img_size, nms_thresh)

    # compute mAP
    classes = {0: 'face'}
    ground_truth = parse_gt_from_anno(img_anno, classes)

    ret_list = parse_rets(ret_files_set)
    iou_thr = 0.5
    evaluate = calc_recall_precision_ap(ground_truth, ret_list, iou_thr)

    aps_str = ''
    for cls in evaluate:
        print('precision:')
        print(evaluate[cls]['precision'])
        per_line, = plt.plot(evaluate[cls]['recall'], evaluate[cls]['precision'], 'b-')
        per_line.set_label('%s:AP=%.3f' % (cls, evaluate[cls]['ap']))
        aps_str += '_%s_AP_%.3f' % (cls, evaluate[cls]['ap'])
        plt.plot([i / 1000.0 for i in range(1, 1001)], [i / 1000.0 for i in range(1, 1001)], 'y--')
        plt.axis([0, 1.2, 0, 1.2])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid()

        plt.legend()
        plt.title('PR')

    # save mAP
    ap_save_path = os.path.join(result_path, result_path.replace('/', '_') + aps_str + '.png')
    print('Saving {}'.format(ap_save_path))
    plt.savefig(ap_save_path)

    print('=============yolov3 evaluating finished==================')
